#!/usr/bin/env python
"""Run a small ROF-v2 feature smoke test on exported CommonRoad JSON samples."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.pipeline import sample_to_bev_tensor_and_features
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sample-ids-file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-name", default="commonroad_rtbev_smoke_features.csv")
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _read_sample_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _short_error(exc: BaseException, limit: int = 800) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()[:limit]


def _sample_path(row: Dict[str, Any], samples_dir: Path) -> Path:
    explicit = row.get("json_gz_path", "")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    return samples_dir / f"{row.get('sample_id', '')}.json.gz"


def _empty_row(row: Dict[str, Any], status: str, error: str = "") -> Dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "commonroad_scenario_id": row.get("commonroad_scenario_id", ""),
        "status": status,
        "error": error,
        "agent_count": row.get("agent_count", ""),
        "lanelet_count": row.get("lanelet_count", ""),
    }


def _write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["section", "metric", "value", "count", "mean", "min", "p50", "max"]
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    out: List[Dict[str, Any]] = [
        {"section": "counts", "metric": "total", "value": len(rows)},
        {"section": "counts", "metric": "ok_count", "value": len(ok)},
        {"section": "counts", "metric": "failed_count", "value": len(failed)},
    ]
    for metric in ["runtime_s", "agent_count", "lanelet_count", "current_min_distance_m", "current_ttc_s", "rcr", "asr_cum_final", "redi_actionability"]:
        vals = sorted(float(r[metric]) for r in ok if str(r.get(metric, "")) not in ("", "nan", "None"))
        if not vals:
            continue
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
    _write_csv(path, out, fields)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("runtime", {})["device"] = args.device
    work_dir = Path(cfg["project"]["work_dir"])
    external_eval = False
    planner_eval = False
    if args.sample_ids_file and "commonroad_external_eval" in str(args.sample_ids_file):
        external_eval = True
    if args.sample_ids_file and "commonroad_planner_feasibility" in str(args.sample_ids_file):
        planner_eval = True
    if "external_eval" in str(args.out_name):
        external_eval = True
    if "planner_pilot1000" in str(args.out_name) or "planner" in str(args.out_name):
        planner_eval = True
    if external_eval:
        out_dir = work_dir / "results" / "commonroad_external_eval" / "features"
    elif planner_eval:
        out_dir = work_dir / "results" / "commonroad_planner_feasibility" / "pilot1000" / "features"
    else:
        out_dir = work_dir / "results" / "commonroad_samples" / "rtbev_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = Path(args.samples_dir)
    manifest_rows = [r for r in _read_csv(Path(args.manifest_csv)) if r.get("export_status", "ok") == "ok"]
    if args.sample_ids_file:
        sample_ids = _read_sample_ids(Path(args.sample_ids_file))
        manifest_by_id = {str(r.get("sample_id", "")): r for r in manifest_rows}
        selected = [manifest_by_id[sid] for sid in sample_ids if sid in manifest_by_id]
        missing = [sid for sid in sample_ids if sid not in manifest_by_id]
        if missing:
            print(f"[warn] sample_ids_missing_from_manifest={len(missing)}")
    else:
        selected = manifest_rows[: max(0, int(args.limit))]

    lib = TubeLibrary.from_workdir(work_dir / "tube_library")
    prim_lib = PrimitiveLibrary.from_workdir(work_dir / "tube_library")
    if not prim_lib.available:
        print("[warn] primitive library unavailable; primitive-dependent ROF fields may be NaN.")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for index, manifest in enumerate(selected, start=1):
        if index == 1 or index % 50 == 0 or index == len(selected):
            print(f"[progress] sample {index}/{len(selected)}")
        out_row = _empty_row(manifest, "failed")
        t0 = time.perf_counter()
        try:
            sample = _read_json_gz(_sample_path(manifest, samples_dir))
            _, feats = sample_to_bev_tensor_and_features(
                sample,
                lib,
                cfg,
                device=args.device,
                primitive_lib=prim_lib,
                return_tensors=False,
            )
            runtime_s = time.perf_counter() - t0
            out_row.update(
                {
                    "status": "ok",
                    "error": "",
                    "agent_count": feats.get("agent_count", manifest.get("agent_count", "")),
                    "lanelet_count": manifest.get("lanelet_count", ""),
                    "runtime_s": f"{runtime_s:.6f}",
                }
            )
            for field in FEATURE_FIELDS:
                if field in out_row:
                    continue
                if field in feats:
                    out_row[field] = feats.get(field)
        except Exception as exc:
            runtime_s = time.perf_counter() - t0
            out_row.update({"status": "failed", "error": _short_error(exc), "runtime_s": f"{runtime_s:.6f}"})
            failures.append(
                {
                    "sample_id": manifest.get("sample_id", ""),
                    "commonroad_scenario_id": manifest.get("commonroad_scenario_id", ""),
                    "error": _short_error(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(out_row)

    features_path = out_dir / args.out_name
    if external_eval or planner_eval:
        stem = Path(args.out_name).stem
        summary_path = out_dir / f"{stem}_summary.csv"
        failures_path = out_dir / f"{stem}_failures.csv"
    else:
        summary_path = out_dir / "commonroad_rtbev_smoke_summary.csv"
        failures_path = out_dir / "commonroad_rtbev_smoke_failures.csv"
    _write_csv(features_path, rows, FEATURE_FIELDS)
    _write_summary(summary_path, rows)
    _write_csv(failures_path, failures, ["sample_id", "commonroad_scenario_id", "error", "traceback"])
    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[done] features={features_path}")
    print(f"[done] summary={summary_path}")
    print(f"[done] failures={failures_path}")
    print(f"[done] total={len(rows)} ok={ok_count} failed={len(rows) - ok_count}")


if __name__ == "__main__":
    main()
