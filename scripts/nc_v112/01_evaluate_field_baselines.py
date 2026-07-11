#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.common import (
    add_config_hash,
    artifact_manifest_rows,
    config_hash,
    experiment_out_dir,
    load_yaml_config,
    resolve_input_path,
    run_manifest,
    write_csv,
    write_json,
)
from rtbev.external.metrics import evaluate_external_scores, merge_scores_labels, parse_comparisons, scenario_bootstrap_deltas


FAMILY_SCORE_COLS = {
    "commonroad_crime_style": [
        "commonroad_crime_risk_score",
        "HW_inverse",
        "THW_inverse",
        "TTC_inverse_crime",
        "ALongReq_mps2",
        "ALatReq_mps2",
    ],
    "rss_style": [
        "rss_danger_score",
        "rss_longitudinal_margin_inverse",
        "rss_lateral_margin_inverse",
    ],
    "drivability": [
        "drivability_risk_score",
        "emergency_brake_infeasible_score",
        "keep_lane_cv_infeasible_score",
        "min_collision_time_keep_lane_inverse",
        "min_road_margin_keep_lane_inverse",
    ],
    "forecast_risk": [
        "forecast_risk_score",
        "cv_forecast_collision_risk",
        "ca_forecast_collision_risk",
        "occupancy_overlap_integral_3s",
        "minimum_predicted_separation_3s_inverse",
    ],
}

REFERENCE_SCORE_COLS = [
    "temporal_composite",
    "ROF_v2_no_asr_composite",
    "ROF_v2_composite",
    "REDI_actionability",
    "distance_inverse",
    "TTC_inverse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v112/nc_v112_field_baselines.yaml")
    parser.add_argument("--planner-labels-csv", default=None)
    parser.add_argument("--rof-scores-csv", default=None)
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--comparisons", default=None)
    return parser.parse_args()


def _load_score_files(out_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[Path]]:
    paths = {
        "commonroad_crime_scores": out_dir / "commonroad_crime_scores.csv",
        "rss_scores": out_dir / "rss_scores.csv",
        "drivability_baseline_scores": out_dir / "drivability_baseline_scores.csv",
        "forecast_risk_scores": out_dir / "forecast_risk_scores.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame["sample_id"] = frame["sample_id"].astype(str)
            frames[name] = frame
    if not frames:
        raise FileNotFoundError(f"no v112 baseline score files found in {out_dir}")
    ordered = list(frames.values())
    merged = ordered[0]
    for frame in ordered[1:]:
        keep = [c for c in frame.columns if c not in merged.columns or c == "sample_id"]
        merged = merged.merge(frame[keep], on="sample_id", how="inner")
    return merged, frames, [p for p in paths.values() if p.exists()]


def _score_family(score: str) -> str:
    for family, cols in FAMILY_SCORE_COLS.items():
        if score in cols:
            return family
    if score in REFERENCE_SCORE_COLS:
        return "reference"
    return "other"


def _available_scores(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for cols in FAMILY_SCORE_COLS.values():
        candidates.extend(cols)
    candidates.extend(REFERENCE_SCORE_COLS)
    out: list[str] = []
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            out.append(col)
    return out


def _default_comparisons(score_cols: list[str]) -> list[tuple[str, str]]:
    baselines = [c for family, cols in FAMILY_SCORE_COLS.items() for c in cols if c in score_cols and c != "ca_forecast_collision_risk"]
    refs = [c for c in ["temporal_composite", "ROF_v2_no_asr_composite", "ROF_v2_composite", "REDI_actionability"] if c in score_cols]
    pairs: list[tuple[str, str]] = []
    for ref in refs:
        for baseline in ["distance_inverse", "TTC_inverse", *baselines]:
            if baseline in score_cols and ref != baseline:
                pairs.append((ref, baseline))
    return pairs


def _availability_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reason_metrics: set[tuple[str, str]] = set()
    for file_name, frame in frames.items():
        for col in frame.columns:
            if not col.endswith("_unavailable_reason"):
                continue
            metric = col.removesuffix("_unavailable_reason")
            reasons = frame[col].fillna("").astype(str)
            nonempty = reasons[reasons != ""]
            if len(nonempty):
                rows.append(
                    {
                        "score_file": file_name,
                        "metric": metric,
                        "available": False,
                        "reason": nonempty.value_counts().idxmax(),
                        "affected_rows": int(len(nonempty)),
                    }
                )
                reason_metrics.add((file_name, metric))
        for col in frame.columns:
            if not col.endswith("_available"):
                continue
            metric = col.removesuffix("_available")
            if (file_name, metric) in reason_metrics:
                continue
            vals = frame[col].astype(str).str.lower()
            available_count = int(vals.isin(["true", "1", "yes"]).sum())
            rows.append(
                {
                    "score_file": file_name,
                    "metric": metric,
                    "available": bool(available_count > 0),
                    "reason": "" if available_count else "no rows marked available",
                    "affected_rows": int(len(frame) - available_count),
                }
            )
    if not rows:
        rows.append({"score_file": "", "metric": "all_primary_baselines", "available": True, "reason": "", "affected_rows": 0})
    return rows


def _best_by_family(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, cols in FAMILY_SCORE_COLS.items():
        sub = metrics[metrics["score"].isin(cols)].copy()
        if sub.empty:
            rows.append({"family": family, "best_score": "", "selection_metric": "AUPRC", "available": False, "reason": "no evaluable score columns"})
            continue
        vals = pd.to_numeric(sub["AUPRC"], errors="coerce")
        if not vals.notna().any():
            rows.append({"family": family, "best_score": "", "selection_metric": "AUPRC", "available": False, "reason": "all AUPRC values are NaN"})
            continue
        row = sub.loc[vals.idxmax()]
        rows.append(
            {
                "family": family,
                "best_score": row["score"],
                "selection_metric": "AUPRC",
                "available": True,
                "AUPRC": float(row["AUPRC"]),
                "Recall@5%FPR": float(row["Recall@5%FPR"]),
                "AUROC": float(row["AUROC"]),
                "Recall@1%FPR": float(row["Recall@1%FPR"]),
                "n": int(row["n"]),
                "positive_count": int(row["positive_count"]),
                "reason": "",
            }
        )
    return pd.DataFrame(rows)


def _delta_row(deltas: pd.DataFrame, enhanced: str, baseline: str, metric: str) -> dict[str, Any] | None:
    if deltas.empty:
        return None
    sub = deltas[
        (deltas["enhanced_score"].astype(str) == enhanced)
        & (deltas["baseline_score"].astype(str) == baseline)
        & (deltas["metric"].astype(str) == metric)
    ]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _fmt_delta(deltas: pd.DataFrame, enhanced: str, baseline: str, metric: str) -> str:
    row = _delta_row(deltas, enhanced, baseline, metric)
    if row is None:
        return "not evaluated"
    return f"delta={float(row['delta']):.6g}, CI=({float(row['ci_low']):.6g}, {float(row['ci_high']):.6g})"


def _wins(deltas: pd.DataFrame, enhanced: str, baseline: str, metric: str) -> bool | None:
    row = _delta_row(deltas, enhanced, baseline, metric)
    if row is None:
        return None
    return bool(float(row["ci_low"]) > 0.0)


def _report(metrics: pd.DataFrame, deltas: pd.DataFrame, best: pd.DataFrame, availability: pd.DataFrame, cfg_hash: str, n_merged: int) -> str:
    lines = [
        "# v112 Field Baseline Report",
        "",
        f"- config_hash: {cfg_hash}",
        f"- fixed_v110_cohort_rows: {n_merged}",
        "- cohort policy: fixed v110 full_10k_fixed_taxonomy_lattice_base; no resampling, no replacement, no v112 result-driven cohort changes.",
        "- endpoint: positive=known_failure, negative=no_failure, unknown_failure excluded.",
        "- bootstrap_unit: scenario_id",
        "- primary metrics: AUPRC, Recall@5%FPR",
        "- primary baseline inputs: current state and CV/CA forecast quantities only; no recorded future, no candidate-action survival, no endpoint intermediate fields, no planner-label tuning.",
        "- RSS note: RSS-style margins only; this is not a complete RSS stack.",
        "",
        "## Best Baselines",
        "",
    ]
    for _, row in best.iterrows():
        if not bool(row.get("available", False)):
            lines.append(f"- {row['family']}: unavailable ({row.get('reason', '')})")
        else:
            lines.append(
                f"- {row['family']}: {row['best_score']} AUPRC={row['AUPRC']:.6g}, "
                f"Recall@5%FPR={row['Recall@5%FPR']:.6g}, AUROC={row['AUROC']:.6g}"
            )
    lines.extend(["", "## Reference Metrics", ""])
    for score in ["temporal_composite", "ROF_v2_no_asr_composite", "ROF_v2_composite", "REDI_actionability", "distance_inverse", "TTC_inverse"]:
        sub = metrics[metrics["score"].astype(str) == score]
        if sub.empty:
            continue
        row = sub.iloc[0]
        lines.append(f"- {score}: AUPRC={row['AUPRC']:.6g}, Recall@5%FPR={row['Recall@5%FPR']:.6g}, AUROC={row['AUROC']:.6g}")
    lines.extend(["", "## Required Comparisons", ""])
    compare_baselines = ["distance_inverse", "TTC_inverse"]
    for family in ["commonroad_crime_style", "rss_style", "forecast_risk"]:
        sub = best[(best["family"] == family) & (best["available"].astype(bool))]
        if not sub.empty:
            compare_baselines.append(str(sub.iloc[0]["best_score"]))
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        lines.append(f"### {enhanced}")
        for baseline in compare_baselines:
            auprc_win = _wins(deltas, enhanced, baseline, "auprc")
            rec_win = _wins(deltas, enhanced, baseline, "recall_at_5pct_fpr")
            lines.append(
                f"- vs {baseline}: AUPRC_win={auprc_win}; {_fmt_delta(deltas, enhanced, baseline, 'auprc')}; "
                f"Recall@5%FPR_win={rec_win}; {_fmt_delta(deltas, enhanced, baseline, 'recall_at_5pct_fpr')}"
            )
        lines.append("")
    lines.extend(["## Unavailable Or Limited Metrics", ""])
    for _, row in availability.iterrows():
        if bool(row.get("available", False)) and not row.get("reason", ""):
            continue
        lines.append(f"- {row.get('metric', '')}: available={row.get('available', '')}; reason={row.get('reason', '')}; affected_rows={row.get('affected_rows', '')}")
    lines.extend(["", "## Top Metrics", ""])
    for _, row in metrics.sort_values(["AUPRC", "Recall@5%FPR"], ascending=False).head(20).iterrows():
        lines.append(
            f"- {row['score']} ({_score_family(str(row['score']))}): "
            f"AUPRC={row.get('AUPRC'):.6g}, Recall@5%FPR={row.get('Recall@5%FPR'):.6g}, AUROC={row.get('AUROC'):.6g}"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v112_field_baselines")
    inputs = cfg.get("inputs", {})
    eval_cfg = cfg.get("evaluation", {})
    labels_path = resolve_input_path(args.planner_labels_csv or inputs.get("planner_labels_csv"), cfg)
    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError("planner-labels CSV is required for v112 baseline evaluation")
    scores, score_frames, score_paths = _load_score_files(out_dir)
    rof_path = resolve_input_path(args.rof_scores_csv or inputs.get("rof_scores_csv"), cfg)
    if rof_path and rof_path.exists():
        rof = pd.read_csv(rof_path)
        rof["sample_id"] = rof["sample_id"].astype(str)
        scores["sample_id"] = scores["sample_id"].astype(str)
        rof_keep = [c for c in ["sample_id", *REFERENCE_SCORE_COLS] if c in rof.columns]
        scores = scores.merge(rof[rof_keep], on="sample_id", how="inner")
    labels = pd.read_csv(labels_path)
    merged = merge_scores_labels(scores, labels)
    score_cols = _available_scores(merged)
    bootstrap_n = int(args.bootstrap_n if args.bootstrap_n is not None else eval_cfg.get("bootstrap_replicates", 1000))
    comparisons_arg = args.comparisons or eval_cfg.get("comparisons", "")
    comparisons = parse_comparisons(comparisons_arg) if comparisons_arg else _default_comparisons(score_cols)
    cfg_hash = config_hash(args.config)

    metric_rows = evaluate_external_scores(merged, score_cols, endpoint="known_failure")
    delta_rows = scenario_bootstrap_deltas(merged, comparisons, n_bootstrap=bootstrap_n, seed=int(eval_cfg.get("bootstrap_seed", 42))) if comparisons else []
    for row in metric_rows:
        row["score_family"] = _score_family(str(row.get("score", "")))
    for row in delta_rows:
        row["enhanced_family"] = _score_family(str(row.get("enhanced_score", "")))
        row["baseline_family"] = _score_family(str(row.get("baseline_score", "")))
    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    best = _best_by_family(metrics)
    availability = pd.DataFrame(_availability_rows(score_frames))

    metrics_path = out_dir / "field_baseline_metrics.csv"
    deltas_path = out_dir / "field_baseline_bootstrap_deltas.csv"
    best_path = out_dir / "best_baseline_summary.csv"
    availability_path = out_dir / "baseline_metric_availability.csv"
    report_path = out_dir / "v112_field_baseline_report.md"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"
    write_csv(metrics_path, add_config_hash(metric_rows, cfg_hash))
    write_csv(deltas_path, add_config_hash(delta_rows, cfg_hash))
    write_csv(best_path, add_config_hash(best.to_dict("records"), cfg_hash))
    write_csv(availability_path, add_config_hash(availability.to_dict("records"), cfg_hash))
    report_path.write_text(_report(metrics, deltas, best, availability, cfg_hash, len(merged)), encoding="utf-8")
    outputs = [*score_paths, metrics_path, deltas_path, best_path, availability_path, report_path]
    write_csv(manifest_path, artifact_manifest_rows(args.config, outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*outputs, manifest_path]))
    print(f"[v112-eval] merged={len(merged)} scores={len(score_cols)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
