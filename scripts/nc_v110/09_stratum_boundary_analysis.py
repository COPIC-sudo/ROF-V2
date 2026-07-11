#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.common import (
    add_config_hash,
    artifact_manifest_rows,
    config_hash,
    experiment_out_dir,
    load_yaml_config,
    require_work_dir,
    run_manifest,
    write_csv,
    write_json,
)
from rtbev.external.metrics import evaluate_external_scores_strict, merge_scores_labels, scenario_bootstrap_deltas_strict


SCORE_COLS = ["temporal_composite", "ROF_v2_no_asr_composite", "distance_inverse", "TTC_inverse"]
COMPARISONS = [
    ("temporal_composite", "distance_inverse"),
    ("temporal_composite", "TTC_inverse"),
    ("ROF_v2_no_asr_composite", "distance_inverse"),
    ("ROF_v2_no_asr_composite", "TTC_inverse"),
]
STRATUM_COLS = ["neutral_stratum", "speed_stratum", "distance_stratum", "ttc_stratum"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-positives", type=int, default=10)
    return parser.parse_args()


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def _metric_value(metrics: pd.DataFrame, score: str, key: str) -> float:
    row = metrics[metrics["score"].astype(str) == score]
    if row.empty:
        return float("nan")
    return float(pd.to_numeric(pd.Series([row.iloc[0].get(key)]), errors="coerce").iloc[0])


def _point_rows(frame: pd.DataFrame, stratum_column: str, stratum: str) -> list[dict[str, Any]]:
    rows = []
    score_cols = [c for c in SCORE_COLS if c in frame.columns]
    for row in evaluate_external_scores_strict(frame, score_cols, endpoint="known_failure"):
        row.update({"row_type": "point_metric", "stratum_column": stratum_column, "stratum": stratum})
        rows.append(row)
    return rows


def _speed_stratum_metrics_bootstrap(merged: pd.DataFrame, bootstrap_n: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "speed_stratum" not in merged.columns:
        return rows
    for idx, (stratum, group) in enumerate(merged.groupby("speed_stratum", dropna=False)):
        label = str(stratum)
        rows.extend(_point_rows(group, "speed_stratum", label))
        deltas = scenario_bootstrap_deltas_strict(
            group,
            [(a, b) for a, b in COMPARISONS if a in group.columns and b in group.columns],
            metrics=("auprc", "recall_at_5pct_fpr_strict"),
            n_bootstrap=bootstrap_n,
            seed=seed + idx,
        )
        for row in deltas:
            row.update({"row_type": "bootstrap_delta", "stratum_column": "speed_stratum", "stratum": label})
            rows.append(row)
    return rows


def _flag_mean(frame: pd.DataFrame, col: str, positive_only: bool = False) -> float:
    work = frame
    if positive_only and "failure_taxonomy" in work.columns:
        work = work[work["failure_taxonomy"].astype(str) == "known_failure"]
    if work.empty or col not in work.columns:
        return float("nan")
    vals = pd.to_numeric(work[col], errors="coerce").fillna(0)
    return float((vals > 0).mean()) if len(vals) else float("nan")


def _negative_delta_audit(merged: pd.DataFrame, min_positives: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum_col in [c for c in STRATUM_COLS if c in merged.columns]:
        for stratum, group in merged.groupby(stratum_col, dropna=False):
            metrics = pd.DataFrame(evaluate_external_scores_strict(group, SCORE_COLS, endpoint="known_failure"))
            if metrics.empty:
                continue
            positive_count = int(pd.to_numeric(metrics["positive_count"], errors="coerce").max())
            if positive_count < min_positives:
                continue
            y_pos = group[group["failure_taxonomy"].astype(str) == "known_failure"]
            for enhanced, baseline in COMPARISONS:
                if enhanced not in group.columns or baseline not in group.columns:
                    continue
                enhanced_auprc = _metric_value(metrics, enhanced, "AUPRC")
                baseline_auprc = _metric_value(metrics, baseline, "AUPRC")
                delta = enhanced_auprc - baseline_auprc if np.isfinite(enhanced_auprc) and np.isfinite(baseline_auprc) else float("nan")
                rows.append(
                    {
                        "stratum_column": stratum_col,
                        "stratum": str(stratum),
                        "enhanced_score": enhanced,
                        "baseline_score": baseline,
                        "n": int(pd.to_numeric(metrics["n"], errors="coerce").max()),
                        "positive_count": positive_count,
                        "positive_scenario_count": int(y_pos["scenario_id"].astype(str).nunique()) if "scenario_id" in y_pos.columns else 0,
                        "enhanced_AUPRC": enhanced_auprc,
                        "baseline_AUPRC": baseline_auprc,
                        "AUPRC_delta": delta,
                        "negative_delta": bool(np.isfinite(delta) and delta < 0.0),
                        "clearly_negative_delta": bool(np.isfinite(delta) and delta < -0.02),
                        "low_speed_fraction": _flag_mean(
                            group.assign(
                                _low_speed=group.get("speed_stratum", pd.Series("", index=group.index)).astype(str) == "lt5mps"
                            ),
                            "_low_speed",
                        ),
                        "near_overlap_fraction": _flag_mean(
                            group.assign(
                                _near_overlap=pd.to_numeric(
                                    group.get("current_min_distance_m", pd.Series(np.nan, index=group.index)),
                                    errors="coerce",
                                )
                                < 1.0
                            ),
                            "_near_overlap",
                        ),
                        "initial_overlap_fraction": _flag_mean(group, "initial_overlap_count"),
                        "collision_positive_fraction": _flag_mean(group, "collision_flag", positive_only=True),
                        "road_boundary_positive_fraction": _flag_mean(group, "road_boundary_flag", positive_only=True),
                        "kinematic_positive_fraction": _flag_mean(group, "kinematic_flag", positive_only=True),
                        "top_failure_subtypes": ";".join(
                            f"{k}:{v}" for k, v in Counter(y_pos.get("failure_subtype", pd.Series(dtype=str)).fillna("").astype(str)).most_common(5)
                        ),
                    }
                )
    return rows


def _low_speed_failure_subtype_summary(merged: pd.DataFrame) -> list[dict[str, Any]]:
    if "speed_stratum" in merged.columns:
        low = merged[merged["speed_stratum"].astype(str) == "lt5mps"].copy()
    else:
        speed = pd.to_numeric(merged.get("ego_speed_mps", pd.Series(np.nan, index=merged.index)), errors="coerce")
        low = merged[speed < 5.0].copy()
    known = low[low["failure_taxonomy"].astype(str) == "known_failure"].copy()
    rows: list[dict[str, Any]] = []
    if known.empty:
        return [{"speed_stratum": "lt5mps", "failure_subtype": "", "planner_failure_reason": "", "count": 0}]
    group_cols = [c for c in ["failure_subtype", "planner_failure_reason"] if c in known.columns]
    for keys, group in known.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: str(value) for col, value in zip(group_cols, keys)}
        row.update(
            {
                "speed_stratum": "lt5mps",
                "count": int(len(group)),
                "scenario_count": int(group["scenario_id"].astype(str).nunique()) if "scenario_id" in group.columns else 0,
                "collision_flag_count": int(pd.to_numeric(group.get("collision_flag", pd.Series(0, index=group.index)), errors="coerce").fillna(0).gt(0).sum()),
                "road_boundary_flag_count": int(pd.to_numeric(group.get("road_boundary_flag", pd.Series(0, index=group.index)), errors="coerce").fillna(0).gt(0).sum()),
                "kinematic_flag_count": int(pd.to_numeric(group.get("kinematic_flag", pd.Series(0, index=group.index)), errors="coerce").fillna(0).gt(0).sum()),
                "initial_overlap_count": int(pd.to_numeric(group.get("initial_overlap_count", pd.Series(0, index=group.index)), errors="coerce").fillna(0).gt(0).sum()),
            }
        )
        rows.append(row)
    return sorted(rows, key=lambda r: (-int(r["count"]), r.get("failure_subtype", "")))


def _initial_overlap_by_stratum(merged: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overlap = pd.to_numeric(merged.get("initial_overlap_count", pd.Series(0, index=merged.index)), errors="coerce").fillna(0) > 0
    for stratum_col in [c for c in STRATUM_COLS if c in merged.columns]:
        for stratum, group in merged.groupby(stratum_col, dropna=False):
            group_overlap = pd.to_numeric(group.get("initial_overlap_count", pd.Series(0, index=group.index)), errors="coerce").fillna(0) > 0
            all_metrics = pd.DataFrame(evaluate_external_scores_strict(group, SCORE_COLS, endpoint="known_failure"))
            filtered = group[~group.index.isin(group.index[group_overlap])].copy()
            filtered_metrics = pd.DataFrame(evaluate_external_scores_strict(filtered, SCORE_COLS, endpoint="known_failure"))
            for enhanced, baseline in COMPARISONS:
                all_delta = _metric_value(all_metrics, enhanced, "AUPRC") - _metric_value(all_metrics, baseline, "AUPRC")
                filt_delta = _metric_value(filtered_metrics, enhanced, "AUPRC") - _metric_value(filtered_metrics, baseline, "AUPRC")
                rows.append(
                    {
                        "stratum_column": stratum_col,
                        "stratum": str(stratum),
                        "enhanced_score": enhanced,
                        "baseline_score": baseline,
                        "n": int(len(group)),
                        "positive_count": int((group["failure_taxonomy"].astype(str) == "known_failure").sum()),
                        "initial_overlap_count_gt0": int(group_overlap.sum()),
                        "initial_overlap_fraction": float(group_overlap.mean()) if len(group_overlap) else float("nan"),
                        "AUPRC_delta_all": all_delta,
                        "AUPRC_delta_exclude_initial_overlap": filt_delta,
                        "AUPRC_delta_improvement_after_exclusion": filt_delta - all_delta
                        if np.isfinite(filt_delta) and np.isfinite(all_delta)
                        else float("nan"),
                        "excluded_initial_overlap_global_count": int(overlap.sum()),
                    }
                )
    return rows


def _report(
    speed_rows: pd.DataFrame,
    negative: pd.DataFrame,
    low_subtypes: pd.DataFrame,
    overlap: pd.DataFrame,
    cfg_hash: str,
    bootstrap_n: int,
) -> str:
    neg = negative[negative["negative_delta"].astype(str).str.lower().isin(["true", "1"])].copy() if not negative.empty else pd.DataFrame()
    clear = negative[negative["clearly_negative_delta"].astype(str).str.lower().isin(["true", "1"])].copy() if not negative.empty else pd.DataFrame()
    low_improve = overlap[
        (overlap["stratum_column"].astype(str) == "speed_stratum")
        & (overlap["stratum"].astype(str) == "lt5mps")
        & (overlap["enhanced_score"].astype(str) == "temporal_composite")
    ].copy() if not overlap.empty else pd.DataFrame()
    lines = [
        "# v110 Full 10k Stratum Boundary Analysis",
        "",
        f"- config_hash: {cfg_hash}",
        f"- bootstrap_n_for_speed_strata: {bootstrap_n}",
        "- analysis type: boundary analysis, not gate shopping; claim gates were not changed.",
        "- inputs: existing v110 full_10k_fixed_taxonomy_lattice_base CSV outputs only.",
        "",
        "## Adequate-Positive Negative AUPRC Deltas",
        "",
        f"- negative_delta_rows: {len(neg)}",
        f"- clearly_negative_delta_rows_lt_minus_0.02: {len(clear)}",
    ]
    if not neg.empty:
        for _, row in neg.sort_values("AUPRC_delta").head(20).iterrows():
            lines.append(
                f"- {row['stratum_column']}={row['stratum']} {row['enhanced_score']} vs {row['baseline_score']}: "
                f"delta={float(row['AUPRC_delta']):.6g}, positives={int(row['positive_count'])}, "
                f"low_speed_fraction={float(row['low_speed_fraction']):.3g}, "
                f"near_overlap_fraction={float(row['near_overlap_fraction']):.3g}, "
                f"collision_positive_fraction={float(row['collision_positive_fraction']):.3g}"
            )
    if not neg.empty:
        low_conc = float((pd.to_numeric(neg["low_speed_fraction"], errors="coerce") >= 0.5).mean())
        overlap_conc = float((pd.to_numeric(neg["near_overlap_fraction"], errors="coerce") >= 0.2).mean())
        collision_conc = float((pd.to_numeric(neg["collision_positive_fraction"], errors="coerce") >= 0.5).mean())
    else:
        low_conc = overlap_conc = collision_conc = 0.0
    lines.extend(
        [
            "",
            "## Concentration Diagnostics",
            "",
            f"- fraction_negative_rows_majority_lt5mps: {low_conc:.6g}",
            f"- fraction_negative_rows_near_overlap_fraction_ge_0.2: {overlap_conc:.6g}",
            f"- fraction_negative_rows_collision_positive_fraction_ge_0.5: {collision_conc:.6g}",
            "",
            "## Low-Speed Initial-Overlap Sensitivity",
            "",
        ]
    )
    if low_improve.empty:
        lines.append("- lt5mps temporal deltas missing or not evaluable.")
    else:
        for _, row in low_improve.iterrows():
            improved = bool(float(row["AUPRC_delta_improvement_after_exclusion"]) > 0) if pd.notna(row["AUPRC_delta_improvement_after_exclusion"]) else False
            lines.append(
                f"- temporal_composite vs {row['baseline_score']}: all_delta={float(row['AUPRC_delta_all']):.6g}, "
                f"exclude_initial_overlap_delta={float(row['AUPRC_delta_exclude_initial_overlap']):.6g}, improved={improved}"
            )
    lines.extend(["", "## Low-Speed Failure Subtypes", ""])
    if low_subtypes.empty:
        lines.append("- no low-speed known failures.")
    else:
        for _, row in low_subtypes.head(10).iterrows():
            lines.append(
                f"- {row.get('failure_subtype', '')} / {row.get('planner_failure_reason', '')}: "
                f"count={int(row.get('count', 0))}, scenarios={int(row.get('scenario_count', 0))}, "
                f"collision={int(row.get('collision_flag_count', 0))}, road={int(row.get('road_boundary_flag_count', 0))}, "
                f"kinematic={int(row.get('kinematic_flag_count', 0))}, initial_overlap={int(row.get('initial_overlap_count', 0))}"
            )
    lines.extend(
        [
            "",
            "## Claim Boundary Wording",
            "",
            "- Main claim should be framed as overall CommonRoad lattice_base external validation on a neutral, outcome-blind full cohort.",
            "- Do not claim uniform superiority in every stratum; adequate-positive low-speed / near-overlap / collision-heavy strata should be reported as boundary or Supplementary analysis.",
            "- Initial-overlap-excluded analyses should be cited when discussing low-speed boundary behavior.",
            "- This analysis did not alter claim gates, cohort membership, labels, planner outputs, or feature outputs.",
            "",
            "## Outputs",
            "",
            "- speed_stratum_metrics_bootstrap.csv",
            "- neutral_stratum_negative_delta_audit.csv",
            "- low_speed_failure_subtype_summary.csv",
            "- initial_overlap_by_stratum.csv",
        ]
    )
    _ = speed_rows
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg_hash = config_hash(args.config)
    base_dir = experiment_out_dir(cfg, "nc_v110_commonroad_scaleup")
    work_dir = require_work_dir(cfg)
    out_dir = work_dir / "results" / "nc_v110_commonroad_scaleup" / "full_10k_stratum_boundary_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _read_csv(base_dir / "planner_labels.csv", "planner labels")
    scores = _read_csv(base_dir / "feature_score_table.csv", "feature score table")
    sample = _read_csv(base_dir / "sample_manifest.csv", "sample manifest")
    merged = merge_scores_labels(scores, labels, sample)

    speed_rows = pd.DataFrame(_speed_stratum_metrics_bootstrap(merged, int(args.bootstrap_n), int(args.seed)))
    negative_rows = pd.DataFrame(_negative_delta_audit(merged, int(args.min_positives)))
    low_rows = pd.DataFrame(_low_speed_failure_subtype_summary(merged))
    overlap_rows = pd.DataFrame(_initial_overlap_by_stratum(merged))

    speed_path = out_dir / "speed_stratum_metrics_bootstrap.csv"
    negative_path = out_dir / "neutral_stratum_negative_delta_audit.csv"
    low_path = out_dir / "low_speed_failure_subtype_summary.csv"
    overlap_path = out_dir / "initial_overlap_by_stratum.csv"
    report_path = out_dir / "stratum_boundary_report.md"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"

    write_csv(speed_path, add_config_hash(speed_rows.to_dict("records"), cfg_hash))
    write_csv(negative_path, add_config_hash(negative_rows.to_dict("records"), cfg_hash))
    write_csv(low_path, add_config_hash(low_rows.to_dict("records"), cfg_hash))
    write_csv(overlap_path, add_config_hash(overlap_rows.to_dict("records"), cfg_hash))
    report_path.write_text(_report(speed_rows, negative_rows, low_rows, overlap_rows, cfg_hash, int(args.bootstrap_n)), encoding="utf-8")
    outputs = [speed_path, negative_path, low_path, overlap_path, report_path]
    write_csv(manifest_path, artifact_manifest_rows(args.config, outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*outputs, manifest_path]))
    print(f"[stratum-boundary] out_dir={out_dir} negative_rows={len(negative_rows)} low_subtypes={len(low_rows)}")


if __name__ == "__main__":
    main()
