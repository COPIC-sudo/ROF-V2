#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    default_score_columns,
    evaluate_external_scores,
    failure_taxonomy_rows,
    merge_scores_labels,
    parse_comparisons,
    scenario_bootstrap_deltas,
    stratum_metrics,
    unknown_failure_sensitivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v110/nc_v110_commonroad_scaleup.yaml")
    parser.add_argument("--scores-csv", default=None, help="ROF/CommonRoad scalar score CSV.")
    parser.add_argument("--features-csv", default=None, help="Alias for --scores-csv.")
    parser.add_argument("--planner-labels-csv", default=None)
    parser.add_argument("--sample-manifest-csv", default=None)
    parser.add_argument("--score-columns", default=None)
    parser.add_argument("--comparisons", default=None)
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _report(metrics: pd.DataFrame, deltas: pd.DataFrame, taxonomy: pd.DataFrame, merged_rows: int) -> str:
    lines = [
        "# v110 CommonRoad Scale-up Report",
        "",
        "Primary endpoint is known planner failure only; unknown failures are excluded from the primary metric frame.",
        "",
        f"- merged samples: {merged_rows}",
        f"- taxonomy rows: {len(taxonomy)}",
        "",
        "## Primary Metrics",
        "",
    ]
    primary = metrics[metrics["endpoint"] == "known_failure"] if not metrics.empty else metrics
    for _, row in primary.sort_values(["AUPRC", "Recall@5%FPR"], ascending=False).head(10).iterrows():
        lines.append(f"- {row['score']}: AUPRC={row.get('AUPRC'):.6g}, Recall@5%FPR={row.get('Recall@5%FPR'):.6g}, AUROC={row.get('AUROC'):.6g}")
    lines.extend(["", "## Bootstrap Deltas", ""])
    if deltas.empty:
        lines.append("- no bootstrap deltas generated")
    else:
        for _, row in deltas.sort_values(["metric", "delta"], ascending=[True, False]).head(10).iterrows():
            lines.append(
                f"- {row['enhanced_score']} vs {row['baseline_score']} {row['metric']}: "
                f"delta={row.get('delta'):.6g}, CI=({row.get('ci_low'):.6g}, {row.get('ci_high'):.6g})"
            )
    lines.extend(["", "## Failure Taxonomy", ""])
    for _, row in taxonomy.iterrows():
        lines.append(f"- {row['failure_taxonomy']}: {int(row['count'])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v110_commonroad_scaleup")
    inputs = cfg.get("inputs", {})
    eval_cfg = cfg.get("evaluation", {})
    scores_path = resolve_input_path(args.scores_csv or args.features_csv or inputs.get("features_csv") or inputs.get("scores_csv"), cfg)
    labels_path = resolve_input_path(args.planner_labels_csv or inputs.get("planner_labels_csv"), cfg)
    sample_manifest_path = resolve_input_path(args.sample_manifest_csv or inputs.get("sample_manifest_csv") or (out_dir / "sample_manifest.csv"), cfg)
    if scores_path is None or labels_path is None:
        raise FileNotFoundError("scores/features CSV and planner-labels CSV are required")
    scores = pd.read_csv(scores_path)
    labels = pd.read_csv(labels_path)
    sample_manifest = pd.read_csv(sample_manifest_path) if sample_manifest_path and sample_manifest_path.exists() else None
    merged = merge_scores_labels(scores, labels, sample_manifest)
    score_cols = _split_csv(args.score_columns) or eval_cfg.get("score_columns") or default_score_columns(merged)
    if not score_cols:
        raise ValueError("no score columns available for external metric evaluation")
    comparisons_arg = args.comparisons or eval_cfg.get("comparisons", "")
    comparisons = parse_comparisons(comparisons_arg) if comparisons_arg else []
    bootstrap_n = int(args.bootstrap_n if args.bootstrap_n is not None else eval_cfg.get("bootstrap_replicates", 1000))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("bootstrap_seed", 42))
    cfg_hash = config_hash(args.config)

    metrics_rows = evaluate_external_scores(merged, score_cols, endpoint="known_failure")
    delta_rows = scenario_bootstrap_deltas(merged, comparisons, n_bootstrap=bootstrap_n, seed=seed) if comparisons else []
    taxonomy_rows = failure_taxonomy_rows(merged)
    stratum_rows = stratum_metrics(merged, score_cols)
    unknown_rows = unknown_failure_sensitivity(merged, score_cols)

    metrics_path = out_dir / "external_metrics.csv"
    deltas_path = out_dir / "external_bootstrap_deltas.csv"
    taxonomy_path = out_dir / "failure_taxonomy.csv"
    stratum_path = out_dir / "stratum_metrics.csv"
    unknown_path = out_dir / "unknown_failure_sensitivity.csv"
    report_path = out_dir / "v110_commonroad_scaleup_report.md"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"

    write_csv(metrics_path, add_config_hash(metrics_rows, cfg_hash))
    write_csv(deltas_path, add_config_hash(delta_rows, cfg_hash))
    write_csv(taxonomy_path, add_config_hash(taxonomy_rows, cfg_hash))
    write_csv(stratum_path, add_config_hash(stratum_rows, cfg_hash))
    write_csv(unknown_path, add_config_hash(unknown_rows, cfg_hash))
    metrics_df = pd.DataFrame(metrics_rows)
    deltas_df = pd.DataFrame(delta_rows)
    taxonomy_df = pd.DataFrame(taxonomy_rows)
    report_path.write_text(_report(metrics_df, deltas_df, taxonomy_df, len(merged)), encoding="utf-8")
    outputs = [metrics_path, deltas_path, taxonomy_path, stratum_path, unknown_path, report_path]
    write_csv(manifest_path, artifact_manifest_rows(args.config, outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*outputs, manifest_path]))
    print(f"[v110-metrics] merged={len(merged)} scores={len(score_cols)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
