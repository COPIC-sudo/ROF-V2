#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _utils import append_blockers, load_yaml, output_dir, resolve_path, write_csv


TYPE_NAMES = {
    0: "unknown",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}


def load_sample(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v095/nc_v095_p0_extension.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    samples_dir = resolve_path(cfg["inputs"]["waymo_samples_dir"])
    label_path = resolve_path(cfg["inputs"]["waymo_actionability_map_labels_csv"])
    nomap_path = resolve_path(cfg["inputs"]["waymo_actionability_nomap_labels_csv"])

    labels = pd.read_csv(label_path, usecols=["sample_id", "scenario_id", "actionability_label_id", "actionability_label_name", "original_label_id"])
    labels["sample_id"] = labels["sample_id"].astype(str)
    if nomap_path.exists():
        nomap = pd.read_csv(nomap_path, usecols=["sample_id", "actionability_label_id"]).rename(columns={"actionability_label_id": "nomap_actionability_label_id"})
        nomap["sample_id"] = nomap["sample_id"].astype(str)
        labels = labels.merge(nomap, on="sample_id", how="left")
    if args.limit:
        labels = labels.head(int(args.limit)).copy()

    sample_rows: list[dict[str, Any]] = []
    time_counts: dict[tuple[int, float], Counter[str]] = defaultdict(Counter)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    label_counts: dict[int, Counter[str]] = defaultdict(Counter)
    missing = 0

    for i, row in enumerate(labels.itertuples(index=False), start=1):
        sid = str(row.sample_id)
        p = samples_dir / f"{sid}.pkl.gz"
        if not p.exists():
            missing += 1
            sample_rows.append(
                {
                    "sample_id": sid,
                    "scenario_id": getattr(row, "scenario_id", ""),
                    "status": "MISSING_SAMPLE_PKL",
                    "map_actionability_label_id": getattr(row, "actionability_label_id", ""),
                    "nomap_actionability_label_id": getattr(row, "nomap_actionability_label_id", ""),
                }
            )
            continue
        try:
            sample = load_sample(p)
            valid = np.asarray(sample.get("future_valid"), dtype=bool)
            future_xy = np.asarray(sample.get("future_xy"))
            future_heading = np.asarray(sample.get("future_heading"))
            agent_types = np.asarray(sample.get("agent_types", np.zeros(valid.shape[0], dtype=int)))
            ego_index = int(sample.get("ego_index", -1))
            times = np.asarray(sample.get("times_s", np.arange(valid.shape[1]) * float(cfg["actionability_labels"]["dt_s"])), dtype=float)
            if valid.ndim != 2:
                raise ValueError(f"future_valid must be 2D, got {valid.shape}")
            actor_mask = np.ones(valid.shape[0], dtype=bool)
            if 0 <= ego_index < len(actor_mask):
                actor_mask[ego_index] = False
            horizon = float(cfg["actionability_labels"]["horizon_s"])
            time_mask = times <= horizon + 1e-6
            if time_mask.shape[0] != valid.shape[1]:
                time_mask = np.ones(valid.shape[1], dtype=bool)
            sub_valid = valid[np.ix_(actor_mask, time_mask)]
            total_slots = int(sub_valid.size)
            valid_slots = int(sub_valid.sum())
            actor_valid_any = int(np.any(sub_valid, axis=1).sum()) if sub_valid.size else 0
            actor_valid_all = int(np.all(sub_valid, axis=1).sum()) if sub_valid.size else 0
            xy_finite = np.isfinite(future_xy[np.ix_(actor_mask, time_mask, [0, 1])]).all(axis=2) if future_xy.ndim == 3 else np.zeros_like(sub_valid, dtype=bool)
            heading_finite = np.isfinite(future_heading[np.ix_(actor_mask, time_mask)]) if future_heading.ndim == 2 else np.zeros_like(sub_valid, dtype=bool)
            invalid_xy_finite = int((~sub_valid & xy_finite).sum()) if sub_valid.size else 0
            valid_xy_nonfinite = int((sub_valid & ~xy_finite).sum()) if sub_valid.size else 0
            valid_heading_nonfinite = int((sub_valid & ~heading_finite).sum()) if sub_valid.size else 0
            agent_type_sub = agent_types[actor_mask] if len(agent_types) == len(actor_mask) else np.zeros(sub_valid.shape[0], dtype=int)
            time_indices = np.where(time_mask)[0]
            for t_pos, orig_t_idx in enumerate(time_indices):
                vals = sub_valid[:, t_pos] if sub_valid.size else np.array([], dtype=bool)
                key = int(orig_t_idx)
                time_counts[key]["total_slots"] += int(vals.size)
                time_counts[key]["valid_slots"] += int(vals.sum())
                time_counts[key]["time_s_sum"] += float(times[orig_t_idx]) if orig_t_idx < len(times) else float("nan")
                time_counts[key]["sample_count"] += 1
            for type_id, vals in zip(agent_type_sub, sub_valid):
                tname = TYPE_NAMES.get(int(type_id), f"type_{int(type_id)}")
                type_counts[tname]["actors"] += 1
                type_counts[tname]["total_slots"] += int(vals.size)
                type_counts[tname]["valid_slots"] += int(vals.sum())
                type_counts[tname]["fully_valid_actors"] += int(vals.all()) if vals.size else 0
                type_counts[tname]["any_valid_actors"] += int(vals.any()) if vals.size else 0
            map_label = int(getattr(row, "actionability_label_id"))
            label_counts[map_label]["samples"] += 1
            label_counts[map_label]["total_slots"] += total_slots
            label_counts[map_label]["valid_slots"] += valid_slots
            sample_rows.append(
                {
                    "sample_id": sid,
                    "scenario_id": getattr(row, "scenario_id", ""),
                    "status": "OK",
                    "map_actionability_label_id": map_label,
                    "map_actionability_label_name": getattr(row, "actionability_label_name", ""),
                    "nomap_actionability_label_id": getattr(row, "nomap_actionability_label_id", ""),
                    "original_label_id": getattr(row, "original_label_id", ""),
                    "non_ego_actor_count": int(actor_mask.sum()),
                    "future_step_count": int(time_mask.sum()),
                    "total_future_slots": total_slots,
                    "valid_future_slots": valid_slots,
                    "future_valid_rate": safe_rate(valid_slots, total_slots),
                    "any_valid_actor_count": actor_valid_any,
                    "fully_valid_actor_count": actor_valid_all,
                    "invalid_slots_with_finite_xy": invalid_xy_finite,
                    "valid_slots_with_nonfinite_xy": valid_xy_nonfinite,
                    "valid_slots_with_nonfinite_heading": valid_heading_nonfinite,
                    "low_future_validity_flag": bool(safe_rate(valid_slots, total_slots) < 0.75) if total_slots else True,
                    "missing_future_validity_flag": bool(total_slots == 0 or valid_slots == 0),
                }
            )
        except Exception as exc:
            sample_rows.append(
                {
                    "sample_id": sid,
                    "scenario_id": getattr(row, "scenario_id", ""),
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "map_actionability_label_id": getattr(row, "actionability_label_id", ""),
                    "nomap_actionability_label_id": getattr(row, "nomap_actionability_label_id", ""),
                }
            )
        if args.progress_every and i % int(args.progress_every) == 0:
            print(f"[v095-future] processed {i}/{len(labels)}")

    write_csv(out_dir / "future_validity_audit_v095.csv", sample_rows)
    if "envswitch" in out_dir.name.lower():
        write_csv(out_dir / "future_validity_audit_envswitch.csv", sample_rows)
        low_rows = [r for r in sample_rows if r.get("status") == "OK" and bool(r.get("low_future_validity_flag"))]
        write_csv(out_dir / "future_validity_low_validity_samples.csv", low_rows)
    by_time = []
    for idx, c in sorted(time_counts.items()):
        sample_count = int(c["sample_count"])
        by_time.append(
            {
                "time_index": idx,
                "time_s": float(c["time_s_sum"] / sample_count) if sample_count else float("nan"),
                "sample_count": sample_count,
                "total_slots": int(c["total_slots"]),
                "valid_slots": int(c["valid_slots"]),
                "future_valid_rate": safe_rate(int(c["valid_slots"]), int(c["total_slots"])),
            }
        )
    write_csv(out_dir / "future_validity_by_time_v095.csv", by_time)
    by_type = []
    for tname, c in sorted(type_counts.items()):
        by_type.append(
            {
                "actor_type": tname,
                "actors": int(c["actors"]),
                "total_slots": int(c["total_slots"]),
                "valid_slots": int(c["valid_slots"]),
                "future_valid_rate": safe_rate(int(c["valid_slots"]), int(c["total_slots"])),
                "any_valid_actors": int(c["any_valid_actors"]),
                "fully_valid_actors": int(c["fully_valid_actors"]),
            }
        )
    write_csv(out_dir / "future_validity_by_actor_type_v095.csv", by_type)
    if "envswitch" in out_dir.name.lower():
        combined = []
        for row in by_time:
            x = dict(row)
            x["dimension"] = "time"
            combined.append(x)
        for row in by_type:
            x = dict(row)
            x["dimension"] = "actor_type"
            combined.append(x)
        write_csv(out_dir / "future_validity_by_time_actor.csv", combined)

    robustness_rows = []
    names = {0: "high_actionability", 1: "reduced_actionability", 2: "critical_actionability", 3: "candidate_set_infeasible"}
    for label_id, c in sorted(label_counts.items()):
        robustness_rows.append(
            {
                "comparison": "current_skip_invalid_oracle_future",
                "status": "AVAILABLE_AUDIT_ONLY",
                "map_actionability_label_id": int(label_id),
                "map_actionability_label_name": names.get(int(label_id), str(label_id)),
                "samples": int(c["samples"]),
                "total_slots": int(c["total_slots"]),
                "valid_slots": int(c["valid_slots"]),
                "future_valid_rate": safe_rate(int(c["valid_slots"]), int(c["total_slots"])),
                "notes": "This summarizes current observed/oracle future validity; labels were not regenerated.",
            }
        )
    robustness_rows.append(
        {
            "comparison": "cv_fallback_relabeling",
            "status": "BLOCKED_NOT_RUN",
            "notes": "Requested CV-fallback label sensitivity requires full actionability relabeling and model reevaluation. This v0.9.5 run only audits existing oracle-future validity.",
        }
    )
    write_csv(out_dir / "future_handling_label_robustness_v095.csv", robustness_rows)
    if "envswitch" in out_dir.name.lower():
        write_csv(out_dir / "future_validity_label_shift_skip_vs_cv.csv", robustness_rows)
    write_csv(
        out_dir / "future_handling_model_metrics_v095.csv",
        [
            {
                "status": "BLOCKED_NOT_RUN",
                "reason": "CV-fallback labels were not generated; no model metrics can be computed without changing the endpoint labels.",
            }
        ],
    )
    if "envswitch" in out_dir.name.lower():
        write_csv(
            out_dir / "future_validity_model_sensitivity.csv",
            [
                {
                    "status": "BLOCKED_NOT_RUN",
                    "reason": "CV-fallback labels were not generated; primary model effect under CV-fallback labels cannot be evaluated without relabeling and OOF model rerun.",
                }
            ],
        )
    if missing:
        append_blockers(
            out_dir,
            [
                {
                    "category": "future_validity",
                    "item": "missing_sample_pkls",
                    "status": "PARTIAL",
                    "details": f"{missing} sample pkl.gz files were missing.",
                    "resume_command": "Regenerate missing sample cache before rerunning scripts/nc_v095/03_future_validity_audit.py.",
                }
            ],
        )
    append_blockers(
        out_dir,
        [
            {
                "category": "future_validity",
                "item": "cv_fallback_relabeling",
                "status": "BLOCKED_NOT_RUN",
                "details": "Current run audited observed/oracle future validity but did not regenerate CV-fallback labels or rerun models.",
                "resume_command": "Implement --future-handling cv_fallback in a v0.9.5 label generator, run pilot, then full relabeling and model evaluation.",
            }
        ],
    )
    ok_rows = [r for r in sample_rows if r.get("status") == "OK"]
    overall_valid = safe_rate(sum(int(r["valid_future_slots"]) for r in ok_rows), sum(int(r["total_future_slots"]) for r in ok_rows))
    gate = [
        "# Future-Validity Claim Gate",
        "",
        f"Audited samples: {len(ok_rows)}",
        f"Missing/error samples: {len(sample_rows) - len(ok_rows)}",
        f"Overall non-ego oracle-future valid-slot rate: {overall_valid:.6f}",
        "",
        "Status: `PASS_WITH_NARROW_WORDING` for validity audit; `BLOCKED` for CV-fallback endpoint sensitivity.",
        "",
        "The audit quantifies the current oracle-future validity surface. It does not prove invariance to CV fallback because that requires a separate relabeling/model run.",
    ]
    (out_dir / "future_validity_claim_gate.md").write_text("\n".join(gate) + "\n", encoding="utf-8")
    print(f"[v095-future] wrote {len(sample_rows)} sample rows elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
