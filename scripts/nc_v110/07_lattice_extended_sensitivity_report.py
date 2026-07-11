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
    sha256_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-planner-labels-csv", default=None)
    parser.add_argument("--extended-planner-labels-csv", default=None)
    parser.add_argument("--features-csv", default=None)
    return parser.parse_args()


def _read(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    out = pd.read_csv(path)
    out["sample_id"] = out["sample_id"].astype(str)
    return out


def _counts(labels: pd.DataFrame) -> dict[str, int]:
    values = labels["failure_taxonomy"].fillna("").astype(str).value_counts()
    return {
        "known_failure": int(values.get("known_failure", 0)),
        "unknown_failure": int(values.get("unknown_failure", 0)),
        "no_failure": int(values.get("no_failure", 0)),
    }


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


def _wins(deltas: pd.DataFrame, enhanced: str, baseline: str, metric: str) -> bool:
    row = _delta_row(deltas, enhanced, baseline, metric)
    return bool(row is not None and float(row["ci_low"]) > 0.0)


def _agreement_rows(joined: pd.DataFrame, cfg_hash: str) -> list[dict[str, Any]]:
    total = len(joined)
    base_known = joined["base_failure_taxonomy"].eq("known_failure")
    ext_known = joined["extended_failure_taxonomy"].eq("known_failure")
    rows = [
        {"metric": "matched_samples", "value": total},
        {"metric": "taxonomy_agreement_count", "value": int((joined["base_failure_taxonomy"] == joined["extended_failure_taxonomy"]).sum())},
        {"metric": "taxonomy_agreement_rate", "value": float((joined["base_failure_taxonomy"] == joined["extended_failure_taxonomy"]).mean()) if total else float("nan")},
        {"metric": "known_failure_agreement_count", "value": int((base_known == ext_known).sum())},
        {"metric": "known_failure_agreement_rate", "value": float((base_known == ext_known).mean()) if total else float("nan")},
        {"metric": "base_known_failure_count", "value": int(base_known.sum())},
        {"metric": "extended_known_failure_count", "value": int(ext_known.sum())},
        {"metric": "base_only_positives", "value": int((base_known & ~ext_known).sum())},
        {"metric": "extended_only_positives", "value": int((~base_known & ext_known).sum())},
        {"metric": "both_positive", "value": int((base_known & ext_known).sum())},
        {"metric": "both_not_positive", "value": int((~base_known & ~ext_known).sum())},
    ]
    if "candidate_count" in joined.columns:
        rows.append({"metric": "extended_candidate_count_mean", "value": float(pd.to_numeric(joined["candidate_count"], errors="coerce").mean())})
        rows.append({"metric": "extended_candidate_count_min", "value": float(pd.to_numeric(joined["candidate_count"], errors="coerce").min())})
        rows.append({"metric": "extended_candidate_count_max", "value": float(pd.to_numeric(joined["candidate_count"], errors="coerce").max())})
    return add_config_hash(rows, cfg_hash)


def _disagreement_rows(joined: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = joined.copy()
    work["transition"] = work["base_failure_taxonomy"].astype(str) + " -> " + work["extended_failure_taxonomy"].astype(str)
    for transition, group in work.groupby("transition", dropna=False):
        rows.append(
            {
                "section": "taxonomy_transition",
                "transition": str(transition),
                "count": int(len(group)),
                "base_failure_subtype_top": group["base_failure_subtype"].fillna("").astype(str).value_counts().idxmax() if len(group) else "",
                "extended_failure_subtype_top": group["extended_failure_subtype"].fillna("").astype(str).value_counts().idxmax() if len(group) else "",
            }
        )
    base_known = work["base_failure_taxonomy"].eq("known_failure")
    ext_known = work["extended_failure_taxonomy"].eq("known_failure")
    case_masks = {
        "base_only_positive": base_known & ~ext_known,
        "extended_only_positive": ~base_known & ext_known,
        "positive_agreement": base_known & ext_known,
        "negative_or_unknown_agreement": ~base_known & ~ext_known,
    }
    for name, mask in case_masks.items():
        group = work[mask]
        rows.append(
            {
                "section": "positive_transition",
                "transition": name,
                "count": int(len(group)),
                "base_failure_subtype_top": group["base_failure_subtype"].fillna("").astype(str).value_counts().idxmax() if len(group) else "",
                "extended_failure_subtype_top": group["extended_failure_subtype"].fillna("").astype(str).value_counts().idxmax() if len(group) else "",
            }
        )
    scenario_col = "scenario_id" if "scenario_id" in work.columns else "commonroad_scenario_id"
    disagreements = work[work["base_failure_taxonomy"] != work["extended_failure_taxonomy"]]
    if scenario_col in disagreements.columns and not disagreements.empty:
        per_scenario = disagreements.groupby(scenario_col).size().sort_values(ascending=False).head(20)
        for scenario_id, count in per_scenario.items():
            rows.append({"section": "top_disagreement_scenario", "transition": str(scenario_id), "count": int(count)})
    return rows


def _report(
    cfg_hash: str,
    ext_counts: dict[str, int],
    agreement: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    feature_hash: str,
    output_paths: list[Path],
) -> str:
    lookup = {str(row["metric"]): row["value"] for _, row in agreement.iterrows()}
    stable_checks: list[bool] = []
    lines = [
        "# v110b Lattice Extended Sensitivity Report",
        "",
        f"- config_hash: {cfg_hash}",
        "- cohort: reused v110 full_10k_fixed_taxonomy_lattice_base sample_manifest.csv",
        "- no resampling: true",
        "- taxonomy_version: v110_fixed_taxonomy_001",
        "- planner_family: lattice_extended",
        "- action-library extension: stronger braking, multiple acceleration levels, multiple yaw-rate/curvature via lateral offset/duration primitives, lane/reference offset primitives.",
        f"- feature_score_table_source_sha256: {feature_hash}",
        "",
        "## Extended Failure Taxonomy",
        "",
        f"- known_failure: {ext_counts.get('known_failure', 0)}",
        f"- unknown_failure: {ext_counts.get('unknown_failure', 0)}",
        f"- no_failure: {ext_counts.get('no_failure', 0)}",
        "",
        "## Base vs Extended Agreement",
        "",
        f"- matched_samples: {int(lookup.get('matched_samples', 0))}",
        f"- taxonomy_agreement_rate: {float(lookup.get('taxonomy_agreement_rate', float('nan'))):.6g}",
        f"- known_failure_agreement_rate: {float(lookup.get('known_failure_agreement_rate', float('nan'))):.6g}",
        f"- base-only positives: {int(lookup.get('base_only_positives', 0))}",
        f"- extended-only positives: {int(lookup.get('extended_only_positives', 0))}",
        f"- extended candidate_count mean/min/max: {float(lookup.get('extended_candidate_count_mean', float('nan'))):.6g}/"
        f"{float(lookup.get('extended_candidate_count_min', float('nan'))):.6g}/"
        f"{float(lookup.get('extended_candidate_count_max', float('nan'))):.6g}",
        "",
        "## Extended-Label Metrics",
        "",
    ]
    for score in ["temporal_composite", "ROF_v2_no_asr_composite", "ROF_v2_composite", "REDI_actionability", "distance_inverse", "TTC_inverse"]:
        sub = metrics[metrics["score"].astype(str) == score]
        if sub.empty:
            continue
        row = sub.iloc[0]
        lines.append(f"- {score}: AUPRC={row['AUPRC']:.6g}, Recall@5%FPR={row['Recall@5%FPR']:.6g}, AUROC={row['AUROC']:.6g}")
    lines.extend(["", "## Robustness Checks", ""])
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        for baseline in ["distance_inverse", "TTC_inverse"]:
            auprc_win = _wins(deltas, enhanced, baseline, "auprc")
            recall_win = _wins(deltas, enhanced, baseline, "recall_at_5pct_fpr")
            stable_checks.extend([auprc_win, recall_win])
            lines.append(
                f"- {enhanced} vs {baseline}: AUPRC_win={auprc_win}; {_fmt_delta(deltas, enhanced, baseline, 'auprc')}; "
                f"Recall@5%FPR_win={recall_win}; {_fmt_delta(deltas, enhanced, baseline, 'recall_at_5pct_fpr')}"
            )
    stable = bool(stable_checks and all(stable_checks))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- temporal/ROF_v2_no_asr remain positive against distance/TTC under lattice_extended labels: {stable}",
            f"- full conclusion stable to action-library extension: {stable}",
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- {path}" for path in output_paths)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    inputs = cfg.get("inputs", {})
    cfg_hash = config_hash(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v110_commonroad_scaleup")
    base_path = resolve_input_path(args.base_planner_labels_csv or inputs.get("base_planner_labels_csv"), cfg)
    ext_path = resolve_input_path(args.extended_planner_labels_csv, cfg) or out_dir / "planner_labels.csv"
    features_path = resolve_input_path(args.features_csv or inputs.get("features_csv"), cfg)
    if base_path is None:
        raise FileNotFoundError("base planner labels path is required")
    if features_path is None:
        raise FileNotFoundError("features path is required")
    base = _read(base_path, "base planner labels")
    ext = _read(ext_path, "extended planner labels")
    keep = [
        "sample_id",
        "scenario_id",
        "commonroad_scenario_id",
        "failure_taxonomy",
        "failure_subtype",
        "taxonomy_rule_id",
        "planner_failure_reason",
    ]
    if "candidate_count" in ext.columns:
        keep.append("candidate_count")
    base_keep = [c for c in keep if c in base.columns and c != "candidate_count"]
    ext_keep = [c for c in keep if c in ext.columns]
    joined = base[base_keep].merge(ext[ext_keep], on="sample_id", suffixes=("_base", "_extended"), how="inner")
    rename = {
        "failure_taxonomy_base": "base_failure_taxonomy",
        "failure_taxonomy_extended": "extended_failure_taxonomy",
        "failure_subtype_base": "base_failure_subtype",
        "failure_subtype_extended": "extended_failure_subtype",
        "taxonomy_rule_id_base": "base_taxonomy_rule_id",
        "taxonomy_rule_id_extended": "extended_taxonomy_rule_id",
        "planner_failure_reason_base": "base_planner_failure_reason",
        "planner_failure_reason_extended": "extended_planner_failure_reason",
    }
    joined = joined.rename(columns=rename)
    if "scenario_id_base" in joined.columns and "scenario_id" not in joined.columns:
        joined["scenario_id"] = joined["scenario_id_base"]
    agreement_rows = _agreement_rows(joined, cfg_hash)
    disagreement_rows = add_config_hash(_disagreement_rows(joined), cfg_hash)
    metrics = pd.read_csv(out_dir / "external_metrics.csv")
    deltas = pd.read_csv(out_dir / "external_bootstrap_deltas.csv")
    ext_counts = _counts(ext)
    agreement_path = out_dir / "label_agreement_with_lattice_base.csv"
    disagreement_path = out_dir / "disagreement_case_summary.csv"
    report_path = out_dir / "v110b_lattice_extended_report.md"
    write_csv(agreement_path, agreement_rows)
    write_csv(disagreement_path, disagreement_rows)
    output_paths = [
        out_dir / "planner_labels.csv",
        out_dir / "failure_taxonomy.csv",
        out_dir / "feature_score_table.csv",
        out_dir / "external_metrics.csv",
        out_dir / "external_bootstrap_deltas.csv",
        agreement_path,
        disagreement_path,
        report_path,
    ]
    report_path.write_text(
        _report(cfg_hash, ext_counts, pd.DataFrame(agreement_rows), metrics, deltas, sha256_file(features_path), output_paths),
        encoding="utf-8",
    )
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"
    write_csv(manifest_path, artifact_manifest_rows(args.config, output_paths))
    write_json(run_path, run_manifest(args.config, cfg, [*output_paths, manifest_path]))
    print(f"[v110b-report] out_dir={out_dir} matched={len(joined)} agreement={agreement_path}")


if __name__ == "__main__":
    main()
