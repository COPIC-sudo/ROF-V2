#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rtbev.config import load_config
from rtbev.tube.rt_library import PrimitiveLibrary, TubeLibrary

FEATURE_FIELDS = [
    "sample_id",
    "commonroad_scenario_id",
    "status",
    "error",
    "agent_count",
    "lanelet_count",
    "current_min_distance_m",
    "current_ttc_s",
    "rcr",
    "rfr_drv",
    "c_time",
    "msr",
    "asr_cum_final",
    "asr_slice_final",
    "ttad_s",
    "collapse_rate_max_per_s",
    "redi_actionability",
    "runtime_s",
]

_CFG: dict[str, Any] | None = None
_SAMPLES_DIR: Path | None = None
_DEVICE = "cpu"
_LIB: TubeLibrary | None = None
_PRIM_LIB: PrimitiveLibrary | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-dir", default=None)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def _resolve(path_value: str | None, cfg: dict[str, Any], required: str) -> Path:
    if not path_value:
        raise ValueError(f"missing {required}")
    value = os.path.expandvars(os.path.expanduser(str(path_value)))
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(cfg["project"]["work_dir"]) / p


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FEATURE_FIELDS).writeheader()


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FEATURE_FIELDS})


def _read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("sample_id", ""))
            if sid:
                done.add(sid)
    return done


def _sample_path(row: dict[str, Any], samples_dir: Path) -> Path:
    explicit = str(row.get("json_gz_path", "") or "")
    if explicit:
        p = Path(os.path.expandvars(os.path.expanduser(explicit)))
        if p.exists():
            return p
    return samples_dir / f"{row.get('sample_id', '')}.json.gz"


def _short_error(exc: BaseException, limit: int = 800) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()[:limit]


def _empty_row(row: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "commonroad_scenario_id": row.get("commonroad_scenario_id", row.get("scenario_id", "")),
        "status": status,
        "error": error,
        "agent_count": row.get("agent_count", ""),
        "lanelet_count": row.get("lanelet_count", ""),
        "current_min_distance_m": row.get("current_min_distance_m", ""),
        "current_ttc_s": row.get("current_ttc_s", ""),
    }


def _read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _init_worker(config_path: str, samples_dir: str, device: str) -> None:
    global _CFG, _SAMPLES_DIR, _DEVICE, _LIB, _PRIM_LIB
    from rtbev.pipeline import sample_to_bev_tensor_and_features  # noqa: F401

    _CFG = load_config(config_path)
    _CFG.setdefault("runtime", {})["device"] = device
    work_dir = Path(_CFG["project"]["work_dir"])
    _SAMPLES_DIR = Path(samples_dir)
    _DEVICE = device
    _LIB = TubeLibrary.from_workdir(work_dir / "tube_library")
    _PRIM_LIB = PrimitiveLibrary.from_workdir(work_dir / "tube_library")


def _process_one(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if _CFG is None or _SAMPLES_DIR is None or _LIB is None or _PRIM_LIB is None:
        raise RuntimeError("worker not initialized")
    from rtbev.pipeline import sample_to_bev_tensor_and_features

    order = int(row.get("_order", 0))
    out_row = _empty_row(row, "failed")
    t0 = time.perf_counter()
    try:
        sample = _read_json_gz(_sample_path(row, _SAMPLES_DIR))
        _, feats = sample_to_bev_tensor_and_features(
            sample,
            _LIB,
            _CFG,
            device=_DEVICE,
            primitive_lib=_PRIM_LIB,
            return_tensors=False,
        )
        out_row.update(
            {
                "status": "ok",
                "error": "",
                "agent_count": feats.get("agent_count", row.get("agent_count", "")),
                "lanelet_count": row.get("lanelet_count", ""),
                "runtime_s": f"{time.perf_counter() - t0:.6f}",
            }
        )
        for field in FEATURE_FIELDS:
            if field not in out_row and field in feats:
                out_row[field] = feats.get(field)
    except Exception as exc:
        out_row.update(
            {
                "status": "failed",
                "error": _short_error(exc),
                "runtime_s": f"{time.perf_counter() - t0:.6f}",
            }
        )
    return order, out_row


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["section", "metric", "value", "count", "mean", "min", "p50", "max"]
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    out: list[dict[str, Any]] = [
        {"section": "counts", "metric": "total", "value": len(rows)},
        {"section": "counts", "metric": "ok_count", "value": len(ok)},
        {"section": "counts", "metric": "failed_count", "value": len(failed)},
    ]
    for metric in ["runtime_s", "agent_count", "lanelet_count", "current_min_distance_m", "current_ttc_s", "rcr", "asr_cum_final", "redi_actionability"]:
        vals = sorted(float(r[metric]) for r in ok if str(r.get(metric, "")) not in ("", "nan", "None"))
        if vals:
            out.append(
                {
                    "section": "describe",
                    "metric": metric,
                    "count": len(vals),
                    "mean": f"{sum(vals) / len(vals):.6f}",
                    "min": f"{vals[0]:.6f}",
                    "p50": f"{vals[len(vals) // 2]:.6f}",
                    "max": f"{vals[-1]:.6f}",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inputs = cfg.get("inputs", {})
    samples_dir = _resolve(args.samples_dir or inputs.get("samples_dir"), cfg, "samples_dir")
    manifest_csv = _resolve(args.manifest_csv or inputs.get("sample_candidates_csv"), cfg, "manifest_csv")
    out_csv = _resolve(args.out_csv or inputs.get("features_csv"), cfg, "features_csv")
    workers = int(args.workers or (cfg.get("runtime") or {}).get("num_workers") or 1)
    workers = max(1, workers)

    rows = [r for r in _read_csv(manifest_csv) if str(r.get("export_status", "ok")).lower() == "ok"]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    for i, row in enumerate(rows):
        row["_order"] = i
    done_ids = _read_done_ids(out_csv) if args.resume else set()
    done_orders = {int(r["_order"]) for r in rows if str(r.get("sample_id", "")) in done_ids}
    next_write = 0
    while next_write in done_orders:
        next_write += 1
    if done_ids:
        rows = [r for r in rows if str(r.get("sample_id", "")) not in done_ids]
        print(f"[resume] done={len(done_ids)} remaining={len(rows)} next_write={next_write}")
    elif out_csv.exists():
        raise FileExistsError(f"{out_csv} exists; pass --resume or choose a new --out-csv")
    if not out_csv.exists():
        _write_header(out_csv)

    t0 = time.perf_counter()
    completed = 0
    buffer: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(args.config, str(samples_dir), args.device)) as pool:
        future_to_order = {pool.submit(_process_one, row): int(row["_order"]) for row in rows}
        for future in as_completed(future_to_order):
            order, result = future.result()
            buffer[order] = result
            completed += 1
            ready: list[dict[str, Any]] = []
            while next_write in buffer:
                ready.append(buffer.pop(next_write))
                next_write += 1
            _append_rows(out_csv, ready)
            if completed == 1 or completed % max(1, int(args.progress_every)) == 0 or completed == len(rows):
                elapsed = time.perf_counter() - t0
                print(f"[progress] completed={completed}/{len(rows)} written_order={next_write} elapsed_s={elapsed:.1f}")

    final_rows = _read_csv(out_csv)
    summary_path = out_csv.with_name(f"{out_csv.stem}_summary.csv")
    failures_path = out_csv.with_name(f"{out_csv.stem}_failures.csv")
    _write_summary(summary_path, final_rows)
    failures = [r for r in final_rows if r.get("status") != "ok"]
    with failures_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "commonroad_scenario_id", "error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in failures:
            writer.writerow({field: row.get(field, "") for field in fields})
    ok_count = sum(1 for r in final_rows if r.get("status") == "ok")
    print(f"[done] features={out_csv}")
    print(f"[done] summary={summary_path}")
    print(f"[done] failures={failures_path}")
    print(f"[done] total={len(final_rows)} ok={ok_count} failed={len(final_rows)-ok_count}")


if __name__ == "__main__":
    main()
