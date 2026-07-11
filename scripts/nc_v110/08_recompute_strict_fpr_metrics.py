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
    parse_comparisons,
    scenario_bootstrap_deltas_strict,
)


V110_RUNS = [
    {
        "name": "v110_lattice_base",
        "config": "configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml",
        "report": "v110_commonroad_scaleup_report_strict_fpr.md",
    },
    {
        "name": "v110b_lattice_extended",
        "config": "configs/nc_v110/nc_v110b_lattice_extended_full_10k.yaml",
        "report": "v110b_lattice_extended_report_strict_fpr.md",
    },
]

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
    parser.add_argument("--v110-bootstrap-n", type=int, default=None)
    parser.add_argument("--v112-bootstrap-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-v110", action="store_true")
    parser.add_argument("--skip-v112", action="store_true")
    parser.add_argument("--v112-config", default="configs/nc_v112/nc_v112_field_baselines_full_10k.yaml")
    return parser.parse_args()


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def _old_metric_warning_rows(old_metrics: pd.DataFrame, source_file: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in old_metrics.iterrows():
        for pct, target in [(1, 0.01), (5, 0.05)]:
            actual = pd.to_numeric(pd.Series([row.get(f"actual_fpr_at_{pct}%FPR")]), errors="coerce").iloc[0]
            if pd.notna(actual) and float(actual) > target + 1e-12:
                rows.append(
                    {
                        "source_file": source_file,
                        "endpoint": row.get("endpoint", ""),
                        "score": row.get("score", ""),
                        "legacy_metric": f"Recall@{pct}%FPR",
                        "target_fpr": target,
                        "legacy_recall": row.get(f"Recall@{pct}%FPR", ""),
                        "legacy_actual_fpr": float(actual),
                        "legacy_threshold": row.get(f"threshold_at_{pct}%FPR", ""),
                        "warning": "legacy quantile threshold exceeds target FPR; do not cite as strict Recall@FPR",
                    }
                )
    return rows


def _score_family(score: str) -> str:
    for family, cols in FAMILY_SCORE_COLS.items():
        if score in cols:
            return family
    if score in REFERENCE_SCORE_COLS:
        return "reference"
    return "other"


def _metric_lookup(deltas: pd.DataFrame, enhanced: str, baseline: str, metric: str) -> dict[str, Any] | None:
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
    row = _metric_lookup(deltas, enhanced, baseline, metric)
    if row is None:
        return "not evaluated"
    return f"delta={float(row['delta']):.6g}, CI=({float(row['ci_low']):.6g}, {float(row['ci_high']):.6g}), pairwise_n={int(row.get('pairwise_n', 0))}"


def _write_strict_manifest(config_path: str, cfg: dict[str, Any], out_dir: Path, outputs: list[Path]) -> None:
    manifest = out_dir / "strict_fpr_artifact_manifest.csv"
    run = out_dir / "strict_fpr_run_manifest.json"
    write_csv(manifest, artifact_manifest_rows(config_path, outputs))
    write_json(run, run_manifest(config_path, cfg, [*outputs, manifest]))


def _report_v110(run_name: str, cfg_hash: str, metrics: pd.DataFrame, deltas: pd.DataFrame, warnings: pd.DataFrame) -> str:
    lines = [
        f"# {run_name} Strict-FPR Report",
        "",
        f"- config_hash: {cfg_hash}",
        "- metric_definition: strict tied-threshold Recall@FPR; actual FPR must be <= target FPR.",
        "- old external_metrics.csv is retained but legacy Recall@FPR rows with actual_fpr > target_fpr are not citable as strict Recall@FPR.",
        "- planner/cohort/features were not rerun.",
        "",
        "## Warning Table",
        "",
        f"- rows_flagged: {len(warnings)}",
    ]
    if not warnings.empty:
        for _, row in warnings.head(20).iterrows():
            lines.append(
                f"- {row['score']} {row['legacy_metric']}: actual_fpr={float(row['legacy_actual_fpr']):.6g} > target={float(row['target_fpr']):.6g}"
            )
    lines.extend(["", "## Strict Metrics", ""])
    for _, row in metrics.sort_values(["AUPRC", "Recall@5%FPR_strict"], ascending=False).iterrows():
        lines.append(
            f"- {row['score']}: AUPRC={row['AUPRC']:.6g}, "
            f"strict Recall@5%FPR={row['Recall@5%FPR_strict']:.6g}, "
            f"strict_actual_fpr={row['strict_actual_fpr_at_5%FPR']:.6g}, AUROC={row['AUROC']:.6g}"
        )
    lines.extend(["", "## Primary Deltas", ""])
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        for baseline in ["distance_inverse", "TTC_inverse"]:
            lines.append(f"- {enhanced} vs {baseline} AUPRC: {_fmt_delta(deltas, enhanced, baseline, 'auprc')}")
            lines.append(
                f"- {enhanced} vs {baseline} strict Recall@5%FPR: {_fmt_delta(deltas, enhanced, baseline, 'recall_at_5pct_fpr_strict')}"
            )
    return "\n".join(lines).rstrip() + "\n"


def _run_v110(config_path: str, report_name: str, bootstrap_n_override: int | None, seed_override: int | None) -> None:
    cfg = load_yaml_config(config_path)
    cfg_hash = config_hash(config_path)
    out_dir = experiment_out_dir(cfg, "nc_v110_commonroad_scaleup")
    labels = _read_csv(out_dir / "planner_labels.csv", "planner labels")
    scores = _read_csv(out_dir / "feature_score_table.csv", "feature score table")
    sample = _read_csv(out_dir / "sample_manifest.csv", "sample manifest")
    old_metrics = _read_csv(out_dir / "external_metrics.csv", "legacy external metrics")
    old_deltas = _read_csv(out_dir / "external_bootstrap_deltas.csv", "legacy external deltas")
    merged = merge_scores_labels(scores, labels, sample)
    score_cols = [str(s) for s in old_metrics["score"].dropna().unique() if str(s) in merged.columns]
    comparisons = sorted({(str(r["enhanced_score"]), str(r["baseline_score"])) for _, r in old_deltas.iterrows()})
    eval_cfg = cfg.get("evaluation", {})
    bootstrap_n = int(bootstrap_n_override if bootstrap_n_override is not None else eval_cfg.get("bootstrap_replicates", 2000))
    seed = int(seed_override if seed_override is not None else eval_cfg.get("bootstrap_seed", 42))

    metrics = pd.DataFrame(evaluate_external_scores_strict(merged, score_cols, endpoint="known_failure"))
    deltas = pd.DataFrame(
        scenario_bootstrap_deltas_strict(
            merged,
            comparisons,
            metrics=("auprc", "recall_at_5pct_fpr_strict"),
            n_bootstrap=bootstrap_n,
            seed=seed,
        )
    )
    warnings = pd.DataFrame(_old_metric_warning_rows(old_metrics, "external_metrics.csv"))
    metrics_path = out_dir / "external_metrics_strict_fpr.csv"
    deltas_path = out_dir / "external_bootstrap_deltas_strict_fpr.csv"
    warning_path = out_dir / "metric_warning_table.csv"
    report_path = out_dir / report_name
    write_csv(metrics_path, add_config_hash(metrics.to_dict("records"), cfg_hash))
    write_csv(deltas_path, add_config_hash(deltas.to_dict("records"), cfg_hash))
    write_csv(warning_path, add_config_hash(warnings.to_dict("records"), cfg_hash))
    report_path.write_text(_report_v110(report_name.removesuffix(".md"), cfg_hash, metrics, deltas, warnings), encoding="utf-8")
    _write_strict_manifest(config_path, cfg, out_dir, [metrics_path, deltas_path, warning_path, report_path])
    print(f"[strict-fpr:v110] out_dir={out_dir} metrics={len(metrics)} deltas={len(deltas)} warnings={len(warnings)}")


def _load_v112_score_files(out_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = [
        out_dir / "commonroad_crime_scores.csv",
        out_dir / "rss_scores.csv",
        out_dir / "drivability_baseline_scores.csv",
        out_dir / "forecast_risk_scores.csv",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path)
            frame["sample_id"] = frame["sample_id"].astype(str)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no baseline score CSVs found in {out_dir}")
    merged = frames[0]
    for frame in frames[1:]:
        keep = [c for c in frame.columns if c not in merged.columns or c == "sample_id"]
        merged = merged.merge(frame[keep], on="sample_id", how="inner")
    return merged, [p for p in paths if p.exists()]


def _available_v112_scores(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for cols in FAMILY_SCORE_COLS.values():
        candidates.extend(cols)
    candidates.extend(REFERENCE_SCORE_COLS)
    out: list[str] = []
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            out.append(col)
    return out


def _v112_comparisons(score_cols: list[str]) -> list[tuple[str, str]]:
    baselines = [c for cols in FAMILY_SCORE_COLS.values() for c in cols if c in score_cols]
    for col in ["distance_inverse", "TTC_inverse"]:
        if col in score_cols:
            baselines.insert(0, col)
    out: list[tuple[str, str]] = []
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        if enhanced not in score_cols:
            continue
        for baseline in dict.fromkeys(baselines):
            if baseline != enhanced and baseline in score_cols:
                out.append((enhanced, baseline))
    return out


def _best_by_family(metrics: pd.DataFrame, family: str) -> str | None:
    cols = FAMILY_SCORE_COLS[family]
    sub = metrics[metrics["score"].isin(cols)].copy()
    if sub.empty:
        return None
    vals = pd.to_numeric(sub["AUPRC"], errors="coerce")
    if not vals.notna().any():
        return None
    return str(sub.loc[vals.idxmax(), "score"])


def _report_v112(cfg_hash: str, metrics: pd.DataFrame, deltas: pd.DataFrame, warnings: pd.DataFrame) -> str:
    compare = ["distance_inverse", "TTC_inverse"]
    for family in ["commonroad_crime_style", "rss_style", "forecast_risk"]:
        best = _best_by_family(metrics, family)
        if best:
            compare.append(best)
    lines = [
        "# v112 Field Baseline Strict-FPR Report",
        "",
        f"- config_hash: {cfg_hash}",
        "- metric_definition: strict tied-threshold Recall@FPR; actual FPR must be <= target FPR.",
        "- cohort policy: fixed v110 full cohort; no resampling or sample manifest replacement.",
        "- primary baseline inputs unchanged; no planner/features rerun.",
        "- RSS note: RSS-style margins only; not a complete RSS stack.",
        "",
        "## Warning Table",
        "",
        f"- rows_flagged: {len(warnings)}",
    ]
    if not warnings.empty:
        for _, row in warnings.iterrows():
            lines.append(
                f"- {row['score']} {row['legacy_metric']}: actual_fpr={float(row['legacy_actual_fpr']):.6g} > target={float(row['target_fpr']):.6g}"
            )
    lines.extend(["", "## Required Comparisons", ""])
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        lines.append(f"### {enhanced}")
        for baseline in compare:
            lines.append(f"- vs {baseline} AUPRC: {_fmt_delta(deltas, enhanced, baseline, 'auprc')}")
            lines.append(f"- vs {baseline} strict Recall@5%FPR: {_fmt_delta(deltas, enhanced, baseline, 'recall_at_5pct_fpr_strict')}")
        lines.append("")
    lines.extend(["## Strict Top Metrics", ""])
    for _, row in metrics.sort_values(["AUPRC", "Recall@5%FPR_strict"], ascending=False).head(25).iterrows():
        lines.append(
            f"- {row['score']} ({row.get('score_family', '')}): AUPRC={row['AUPRC']:.6g}, "
            f"strict Recall@5%FPR={row['Recall@5%FPR_strict']:.6g}, strict_actual_fpr={row['strict_actual_fpr_at_5%FPR']:.6g}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _run_v112(config_path: str, bootstrap_n_override: int | None, seed_override: int | None) -> None:
    cfg = load_yaml_config(config_path)
    cfg_hash = config_hash(config_path)
    out_dir = experiment_out_dir(cfg, "nc_v112_field_baselines")
    inputs = cfg.get("inputs", {})
    labels_path = resolve_input_path(inputs.get("planner_labels_csv"), cfg)
    rof_path = resolve_input_path(inputs.get("rof_scores_csv") or inputs.get("features_csv"), cfg)
    if labels_path is None:
        raise FileNotFoundError("planner_labels_csv missing from v112 config")
    labels = _read_csv(labels_path, "v110 planner labels")
    scores, score_paths = _load_v112_score_files(out_dir)
    if rof_path and rof_path.exists():
        rof = pd.read_csv(rof_path)
        rof["sample_id"] = rof["sample_id"].astype(str)
        keep = [c for c in ["sample_id", *REFERENCE_SCORE_COLS] if c in rof.columns]
        scores = scores.merge(rof[keep], on="sample_id", how="inner")
    merged = merge_scores_labels(scores, labels)
    score_cols = _available_v112_scores(merged)
    comparisons = _v112_comparisons(score_cols)
    eval_cfg = cfg.get("evaluation", {})
    bootstrap_n = int(bootstrap_n_override if bootstrap_n_override is not None else eval_cfg.get("bootstrap_replicates", 2000))
    seed = int(seed_override if seed_override is not None else eval_cfg.get("bootstrap_seed", 42))
    metrics = pd.DataFrame(evaluate_external_scores_strict(merged, score_cols, endpoint="known_failure"))
    if not metrics.empty:
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
    old_metrics = _read_csv(out_dir / "field_baseline_metrics.csv", "legacy field baseline metrics")
    warnings = pd.DataFrame(_old_metric_warning_rows(old_metrics, "field_baseline_metrics.csv"))
    metrics_path = out_dir / "field_baseline_metrics_strict_fpr.csv"
    deltas_path = out_dir / "field_baseline_bootstrap_deltas_strict_fpr.csv"
    warning_path = out_dir / "metric_warning_table.csv"
    report_path = out_dir / "v112_field_baseline_report_strict_fpr.md"
    write_csv(metrics_path, add_config_hash(metrics.to_dict("records"), cfg_hash))
    write_csv(deltas_path, add_config_hash(deltas.to_dict("records"), cfg_hash))
    write_csv(warning_path, add_config_hash(warnings.to_dict("records"), cfg_hash))
    report_path.write_text(_report_v112(cfg_hash, metrics, deltas, warnings), encoding="utf-8")
    _write_strict_manifest(config_path, cfg, out_dir, [*score_paths, metrics_path, deltas_path, warning_path, report_path])
    print(f"[strict-fpr:v112] out_dir={out_dir} metrics={len(metrics)} deltas={len(deltas)} warnings={len(warnings)}")


def main() -> None:
    args = parse_args()
    if not args.skip_v110:
        for run in V110_RUNS:
            _run_v110(run["config"], run["report"], args.v110_bootstrap_n, args.seed)
    if not args.skip_v112:
        _run_v112(args.v112_config, args.v112_bootstrap_n, args.seed)


if __name__ == "__main__":
    main()
