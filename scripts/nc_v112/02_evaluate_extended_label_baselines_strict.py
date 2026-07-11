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
from rtbev.external.metrics import (
    evaluate_external_scores_strict,
    merge_scores_labels,
    scenario_bootstrap_deltas_strict,
)


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

BINARY_BASELINES = [
    "emergency_brake_infeasible_score",
    "keep_lane_cv_infeasible_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v112/nc_v112b_field_baselines_extended_label.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def _input_path(cfg: dict[str, Any], key: str) -> Path:
    path = resolve_input_path((cfg.get("inputs") or {}).get(key), cfg)
    if path is None:
        raise FileNotFoundError(f"inputs.{key} is required")
    return path


def _load_baseline_scores(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[Path]]:
    keys = [
        "commonroad_crime_scores_csv",
        "rss_scores_csv",
        "drivability_baseline_scores_csv",
        "forecast_risk_scores_csv",
    ]
    paths = [_input_path(cfg, key) for key in keys]
    frames: list[pd.DataFrame] = []
    for path, key in zip(paths, keys):
        frame = _read_csv(path, key)
        frame["sample_id"] = frame["sample_id"].astype(str)
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        keep = [c for c in frame.columns if c not in merged.columns or c == "sample_id"]
        merged = merged.merge(frame[keep], on="sample_id", how="inner")
    return merged, paths


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


def _comparisons(score_cols: list[str]) -> list[tuple[str, str]]:
    baselines = [c for cols in FAMILY_SCORE_COLS.values() for c in cols if c in score_cols]
    for col in ["distance_inverse", "TTC_inverse"]:
        if col in score_cols:
            baselines.insert(0, col)
    out: list[tuple[str, str]] = []
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        if enhanced not in score_cols:
            continue
        for baseline in dict.fromkeys(baselines):
            if baseline != enhanced:
                out.append((enhanced, baseline))
    return out


def _best_by_family(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, cols in FAMILY_SCORE_COLS.items():
        sub = metrics[metrics["score"].isin(cols)].copy()
        if sub.empty:
            rows.append({"family": family, "available": False, "best_score": "", "reason": "no evaluable columns"})
            continue
        vals = pd.to_numeric(sub["AUPRC"], errors="coerce")
        if not vals.notna().any():
            rows.append({"family": family, "available": False, "best_score": "", "reason": "all AUPRC values are NaN"})
            continue
        row = sub.loc[vals.idxmax()]
        rows.append(
            {
                "family": family,
                "available": True,
                "best_score": row["score"],
                "selection_metric": "AUPRC",
                "AUPRC": float(row["AUPRC"]),
                "AUROC": float(row["AUROC"]),
                "Recall@5%FPR_strict": float(row["Recall@5%FPR_strict"]),
                "strict_actual_fpr_at_5%FPR": float(row["strict_actual_fpr_at_5%FPR"]),
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
    return (
        f"delta={float(row['delta']):.6g}, "
        f"CI=({float(row['ci_low']):.6g}, {float(row['ci_high']):.6g}), "
        f"pairwise_n={int(row.get('pairwise_n', 0))}, pairwise_positives={int(row.get('pairwise_positive_count', 0))}"
    )


def _win(deltas: pd.DataFrame, enhanced: str, baseline: str, metric: str) -> bool | None:
    row = _delta_row(deltas, enhanced, baseline, metric)
    if row is None:
        return None
    return bool(float(row.get("ci_low", float("nan"))) > 0.0)


def _binary_baseline_notes(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score in BINARY_BASELINES:
        sub = metrics[metrics["score"].astype(str) == score]
        if sub.empty:
            rows.append({"score": score, "available": False, "note": "score not available"})
            continue
        row = sub.iloc[0]
        recall = float(row.get("Recall@5%FPR_strict", float("nan")))
        actual = float(row.get("strict_actual_fpr_at_5%FPR", float("nan")))
        threshold = float(row.get("strict_threshold_at_5%FPR", float("nan")))
        rows.append(
            {
                "score": score,
                "available": True,
                "Recall@5%FPR_strict": recall,
                "strict_actual_fpr_at_5%FPR": actual,
                "strict_threshold_at_5%FPR": threshold,
                "direct_comparison_caveat": bool(recall == 0.0 or actual == 0.0),
                "note": "binary/tied score; strict Recall@5%FPR can collapse to zero if the positive threshold jump exceeds the FPR budget",
            }
        )
    return rows


def _report(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    best: pd.DataFrame,
    binary_notes: pd.DataFrame,
    cfg_hash: str,
    n_merged: int,
) -> str:
    label_counts = {}
    if not metrics.empty:
        label_counts["n"] = int(metrics.iloc[0]["n"])
        label_counts["positive_count"] = int(metrics.iloc[0]["positive_count"])
    compare_baselines = ["distance_inverse", "TTC_inverse"]
    for family in ["commonroad_crime_style", "rss_style", "forecast_risk"]:
        sub = best[(best["family"].astype(str) == family) & (best["available"].astype(str).str.lower().isin(["true", "1"]))]
        if not sub.empty:
            compare_baselines.append(str(sub.iloc[0]["best_score"]))
    lines = [
        "# v112b Field Baseline Extended-Label Strict-FPR Report",
        "",
        f"- config_hash: {cfg_hash}",
        f"- merged_rows: {n_merged}",
        f"- endpoint_n_after_unknown_exclusion: {label_counts.get('n', 'NA')}",
        f"- extended_known_failure_positive_count: {label_counts.get('positive_count', 'NA')}",
        "- endpoint: positive=extended known_failure; negative=extended no_failure; extended unknown_failure excluded.",
        "- metric_definition: strict tied-threshold Recall@FPR; actual FPR must be <= target FPR.",
        "- baseline scores: reused from v112 full_10k_fixed_taxonomy_lattice_base; no baseline extraction rerun.",
        "- cohort: fixed full 10k cohort; no resampling and no label-result filtering.",
        "- claim boundary: action-library-extended endpoint sensitivity only; no native-planner robustness claim.",
        "",
        "## Best Baselines By Family",
        "",
    ]
    for _, row in best.iterrows():
        if not bool(row.get("available", False)):
            lines.append(f"- {row['family']}: unavailable ({row.get('reason', '')})")
        else:
            lines.append(
                f"- {row['family']}: {row['best_score']} AUPRC={float(row['AUPRC']):.6g}, "
                f"strict Recall@5%FPR={float(row['Recall@5%FPR_strict']):.6g}, "
                f"strict_actual_fpr={float(row['strict_actual_fpr_at_5%FPR']):.6g}"
            )
    lines.extend(["", "## Required Comparisons", ""])
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        lines.append(f"### {enhanced}")
        for baseline in compare_baselines:
            lines.append(
                f"- vs {baseline}: AUPRC_win={_win(deltas, enhanced, baseline, 'auprc')}; "
                f"{_fmt_delta(deltas, enhanced, baseline, 'auprc')}; "
                f"strict Recall@5%FPR_win={_win(deltas, enhanced, baseline, 'recall_at_5pct_fpr_strict')}; "
                f"{_fmt_delta(deltas, enhanced, baseline, 'recall_at_5pct_fpr_strict')}"
            )
        lines.append("")
    lines.extend(["## Drivability Component AUPRC Checks", ""])
    drivability_components = [
        "emergency_brake_infeasible_score",
        "keep_lane_cv_infeasible_score",
        "min_collision_time_keep_lane_inverse",
        "min_road_margin_keep_lane_inverse",
    ]
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        lines.append(f"### {enhanced}")
        for baseline in drivability_components:
            if baseline in set(metrics["score"].astype(str)):
                lines.append(f"- vs {baseline}: AUPRC_win={_win(deltas, enhanced, baseline, 'auprc')}; {_fmt_delta(deltas, enhanced, baseline, 'auprc')}")
        lines.append("")
    lines.extend(["## Binary / Tied Baseline Strict-FPR Notes", ""])
    if binary_notes.empty:
        lines.append("- no binary baseline notes.")
    for _, row in binary_notes.iterrows():
        lines.append(
            f"- {row['score']}: available={row.get('available')}, "
            f"strict Recall@5%FPR={row.get('Recall@5%FPR_strict', '')}, "
            f"strict_actual_fpr={row.get('strict_actual_fpr_at_5%FPR', '')}, "
            f"caveat={row.get('direct_comparison_caveat', '')}; {row.get('note', '')}"
        )
    lines.extend(["", "## Strict Metrics", ""])
    for _, row in metrics.sort_values(["AUPRC", "Recall@5%FPR_strict"], ascending=False).iterrows():
        lines.append(
            f"- {row['score']} ({row['score_family']}): AUPRC={float(row['AUPRC']):.6g}, "
            f"strict Recall@5%FPR={float(row['Recall@5%FPR_strict']):.6g}, AUROC={float(row['AUROC']):.6g}"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg_hash = config_hash(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v112_field_baselines")
    eval_cfg = cfg.get("evaluation", {})
    labels = _read_csv(_input_path(cfg, "planner_labels_csv"), "extended planner labels")
    rof_scores = _read_csv(_input_path(cfg, "rof_scores_csv"), "extended feature score table")
    baseline_scores, baseline_paths = _load_baseline_scores(cfg)

    rof_scores["sample_id"] = rof_scores["sample_id"].astype(str)
    baseline_scores["sample_id"] = baseline_scores["sample_id"].astype(str)
    rof_keep = [c for c in ["sample_id", *REFERENCE_SCORE_COLS] if c in rof_scores.columns]
    scores = baseline_scores.merge(rof_scores[rof_keep], on="sample_id", how="inner")
    merged = merge_scores_labels(scores, labels)
    score_cols = _available_scores(merged)
    comparisons = _comparisons(score_cols)
    bootstrap_n = int(args.bootstrap_n if args.bootstrap_n is not None else eval_cfg.get("bootstrap_replicates", 2000))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("bootstrap_seed", 42))

    metrics = pd.DataFrame(evaluate_external_scores_strict(merged, score_cols, endpoint="known_failure"))
    metrics["score_family"] = metrics["score"].astype(str).map(_score_family)
    deltas = pd.DataFrame(
        scenario_bootstrap_deltas_strict(
            merged,
            comparisons,
            metrics=("auprc", "recall_at_5pct_fpr_strict"),
            n_bootstrap=bootstrap_n,
            seed=seed,
        )
    )
    if not deltas.empty:
        deltas["enhanced_family"] = deltas["enhanced_score"].astype(str).map(_score_family)
        deltas["baseline_family"] = deltas["baseline_score"].astype(str).map(_score_family)
    best = _best_by_family(metrics)
    binary_notes = pd.DataFrame(_binary_baseline_notes(metrics))

    metrics_path = out_dir / "field_baseline_metrics_strict_fpr.csv"
    deltas_path = out_dir / "field_baseline_bootstrap_deltas_strict_fpr.csv"
    best_path = out_dir / "best_baseline_summary_strict_fpr.csv"
    notes_path = out_dir / "binary_baseline_strict_fpr_notes.csv"
    report_path = out_dir / "v112b_field_baseline_extended_label_report.md"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"
    write_csv(metrics_path, add_config_hash(metrics.to_dict("records"), cfg_hash))
    write_csv(deltas_path, add_config_hash(deltas.to_dict("records"), cfg_hash))
    write_csv(best_path, add_config_hash(best.to_dict("records"), cfg_hash))
    write_csv(notes_path, add_config_hash(binary_notes.to_dict("records"), cfg_hash))
    report_path.write_text(_report(metrics, deltas, best, binary_notes, cfg_hash, len(merged)), encoding="utf-8")
    outputs = [*baseline_paths, metrics_path, deltas_path, best_path, notes_path, report_path]
    write_csv(manifest_path, artifact_manifest_rows(args.config, outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*outputs, manifest_path]))
    print(f"[v112b-extended-label] out_dir={out_dir} metrics={len(metrics)} deltas={len(deltas)} merged={len(merged)}")


if __name__ == "__main__":
    main()
