#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.cohort import select_neutral_stratified_cohort
from rtbev.external.common import (
    add_config_hash,
    artifact_manifest_rows,
    config_hash,
    experiment_out_dir,
    load_yaml_config,
    read_csv_required,
    resolve_input_path,
    run_manifest,
    write_csv,
    write_json,
)
from rtbev.external.metrics import (
    add_common_scores,
    classify_failure_taxonomy,
    default_score_columns,
    evaluate_external_scores,
    failure_taxonomy_rows,
    merge_scores_labels,
    parse_comparisons,
    scenario_bootstrap_deltas,
    stratum_metrics,
    unknown_failure_sensitivity,
)
from rtbev.external.taxonomy import DEFAULT_TAXONOMY_VERSION, annotate_planner_failure_taxonomy


REASON_BURST_TOKENS = ("parser_error", "parse_error", "no_route", "initial_invalid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v110/nc_v110_commonroad_dryrun.yaml")
    parser.add_argument("--sample-candidates-csv", default=None)
    parser.add_argument("--planner-labels-csv", default=None)
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _path_arg(value: str | None, cfg: dict[str, Any], fallback: str | None, name: str) -> Path:
    path = resolve_input_path(value or fallback, cfg)
    if path is None:
        raise FileNotFoundError(f"{name} is required")
    return path


def _filter_sample_ids(df: pd.DataFrame, sample_ids: set[str]) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise ValueError("input table missing sample_id")
    out = df.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    return out[out["sample_id"].isin(sample_ids)].copy()


def _add_scenario_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "scenario_id" not in out.columns:
        if "commonroad_scenario_id" in out.columns:
            out["scenario_id"] = out["commonroad_scenario_id"].astype(str)
        else:
            out["scenario_id"] = out["sample_id"].astype(str)
    if "commonroad_scenario_id" not in out.columns:
        out["commonroad_scenario_id"] = out["scenario_id"].astype(str)
    return out


def _prepare_feature_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = add_common_scores(features)
    if "REDI_actionability" not in out.columns and "redi_actionability" in out.columns:
        out["REDI_actionability"] = pd.to_numeric(out["redi_actionability"], errors="coerce")
    if "REDI_actionability" in out.columns and "redi_actionability" in out.columns:
        out = out.drop(columns=["redi_actionability"])
    return _add_scenario_id(out)


def _prepare_planner_labels(
    labels: pd.DataFrame,
    taxonomy_version: str = DEFAULT_TAXONOMY_VERSION,
    planner_family: str = "lattice_base",
) -> pd.DataFrame:
    out = _add_scenario_id(labels)
    if "planner_family" not in out.columns:
        out["planner_family"] = planner_family
    else:
        out["planner_family"] = out["planner_family"].fillna("").astype(str).replace("", planner_family)
    out = annotate_planner_failure_taxonomy(out, taxonomy_version=taxonomy_version)
    return out


def _finalize_planner_label_schema(labels: pd.DataFrame, sample_manifest: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = labels.copy()
    manifest_cols = [c for c in ["sample_id", "ego_obstacle_id", "current_time_step"] if c in sample_manifest.columns]
    if manifest_cols:
        extra = sample_manifest[manifest_cols].copy()
        extra["sample_id"] = extra["sample_id"].astype(str)
        out["sample_id"] = out["sample_id"].astype(str)
        out = out.merge(extra, on="sample_id", how="left", suffixes=("", "_manifest"))
    if "time_step" not in out.columns:
        if "current_time_step" in out.columns:
            out["time_step"] = out["current_time_step"]
        elif "current_time_step_manifest" in out.columns:
            out["time_step"] = out["current_time_step_manifest"]
        else:
            out["time_step"] = ""
    if "ego_obstacle_id" not in out.columns and "ego_obstacle_id_manifest" in out.columns:
        out["ego_obstacle_id"] = out["ego_obstacle_id_manifest"]
    out["known_failure"] = (out["failure_taxonomy"].astype(str) == "known_failure").astype(int)
    out["unknown_failure"] = (out["failure_taxonomy"].astype(str) == "unknown_failure").astype(int)
    out["no_failure"] = (out["failure_taxonomy"].astype(str) == "no_failure").astype(int)
    planner_cfg = cfg.get("planner") or {}
    out["horizon_s"] = out["horizon_s"] if "horizon_s" in out.columns else planner_cfg.get("horizon_s", 3.0)
    out["lane_buffer_m"] = out["lane_buffer_m"] if "lane_buffer_m" in out.columns else planner_cfg.get("lane_buffer_m", 4.0)
    for col in [
        "collision_flag",
        "road_boundary_flag",
        "lane_buffer_flag",
        "kinematic_flag",
        "no_candidate_flag",
        "no_route_flag",
        "initial_invalid_flag",
        "parser_error_flag",
        "candidate_any_feasible",
        "candidate_all_invalid",
    ]:
        if col not in out.columns:
            out[col] = 0
    if "candidate_failure_reasons" not in out.columns:
        out["candidate_failure_reasons"] = ""
    preferred = [
        "sample_id",
        "scenario_id",
        "commonroad_scenario_id",
        "ego_obstacle_id",
        "time_step",
        "known_failure",
        "unknown_failure",
        "no_failure",
        "failure_taxonomy",
        "failure_subtype",
        "taxonomy_rule_id",
        "taxonomy_version",
        "planner_family",
        "horizon_s",
        "lane_buffer_m",
        "collision_flag",
        "road_boundary_flag",
        "lane_buffer_flag",
        "kinematic_flag",
        "no_candidate_flag",
        "no_route_flag",
        "initial_invalid_flag",
        "parser_error_flag",
        "candidate_any_feasible",
        "candidate_all_invalid",
        "candidate_count",
        "feasible_candidate_count",
        "candidate_failure_reasons",
    ]
    remaining = [c for c in out.columns if c not in preferred and not c.endswith("_manifest")]
    return out[[c for c in preferred if c in out.columns] + remaining]


def _legacy_failure_taxonomy(labels: pd.DataFrame) -> pd.Series:
    failure = pd.to_numeric(labels.get("planner_failure", pd.Series(0, index=labels.index)), errors="coerce").fillna(0).astype(int)
    reason_col = "raw_planner_failure_reason" if "raw_planner_failure_reason" in labels.columns else "planner_failure_reason"
    reason = labels.get(reason_col, pd.Series("", index=labels.index)).fillna("").astype(str).str.lower()
    out = pd.Series("no_failure", index=labels.index, dtype=object)
    unknown = reason.isin(["", "unknown", "nan", "missing"]) | reason.str.contains(
        "unknown|parser|parse_error|runtime|software|numerical|missing|validator|timeout_without", regex=True
    )
    out.loc[(failure == 1) & unknown] = "unknown_failure"
    out.loc[(failure == 1) & ~unknown] = "known_failure"
    return out


def _reason_counts(labels: pd.DataFrame) -> Counter[str]:
    col = next((c for c in ["planner_failure_reason", "failure_reason", "dominant_failure_reason"] if c in labels.columns), None)
    if col is None:
        return Counter()
    values = labels[col].fillna("").astype(str).str.lower()
    return Counter(values)


def _burst_summary(labels: pd.DataFrame) -> list[dict[str, Any]]:
    reason_col = next((c for c in ["planner_failure_reason", "failure_reason", "dominant_failure_reason"] if c in labels.columns), None)
    if reason_col is None:
        return []
    work = _add_scenario_id(labels)
    work["_reason_norm"] = work[reason_col].fillna("").astype(str).str.lower()
    rows: list[dict[str, Any]] = []
    for token in REASON_BURST_TOKENS:
        mask = work["_reason_norm"].str.contains(token, regex=False)
        token_df = work[mask]
        if token_df.empty:
            rows.append(
                {
                    "reason_token": token,
                    "count": 0,
                    "scenario_count": 0,
                    "max_count_in_one_scenario": 0,
                    "max_scenario_fraction": 0.0,
                    "concentrated": False,
                }
            )
            continue
        per_scenario = token_df.groupby("scenario_id").size()
        rows.append(
            {
                "reason_token": token,
                "count": int(len(token_df)),
                "scenario_count": int(per_scenario.shape[0]),
                "max_count_in_one_scenario": int(per_scenario.max()),
                "max_scenario_fraction": float(per_scenario.max() / max(len(token_df), 1)),
                "concentrated": bool(per_scenario.max() / max(len(token_df), 1) >= 0.5 and len(token_df) >= 5),
            }
        )
    return rows


def _write_dataframe(path: Path, df: pd.DataFrame, cfg_hash: str) -> None:
    rows = df.to_dict("records")
    write_csv(path, add_config_hash(rows, cfg_hash))


def _initial_overlap_sensitivity(merged: pd.DataFrame, score_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overlap = pd.to_numeric(merged.get("initial_overlap_count", pd.Series(0, index=merged.index)), errors="coerce").fillna(0)
    for mode, frame in [
        ("all_primary", merged),
        ("exclude_initial_overlap_count_gt0", merged[overlap <= 0].copy()),
    ]:
        for row in evaluate_external_scores(frame, score_cols, endpoint="known_failure"):
            row["sensitivity_mode"] = mode
            row["excluded_initial_overlap_samples"] = int((overlap > 0).sum()) if mode != "all_primary" else 0
            rows.append(row)
    return rows


def _taxonomy_recode_audit(old_labels: pd.DataFrame, fixed_labels: pd.DataFrame, cfg_hash: str) -> list[dict[str, Any]]:
    if old_labels.empty or fixed_labels.empty:
        return []
    old = _add_scenario_id(old_labels.copy())
    fixed = _add_scenario_id(fixed_labels.copy())
    old["sample_id"] = old["sample_id"].astype(str)
    fixed["sample_id"] = fixed["sample_id"].astype(str)
    old_tax = _legacy_failure_taxonomy(old)
    old = old.assign(old_failure_taxonomy=old_tax)
    keep_old = [
        "sample_id",
        "scenario_id",
        "commonroad_scenario_id",
        "old_failure_taxonomy",
        "planner_failure_reason",
        "failure_subtype",
        "taxonomy_rule_id",
    ]
    keep_fixed = [
        "sample_id",
        "failure_taxonomy",
        "planner_failure_reason",
        "failure_subtype",
        "taxonomy_rule_id",
        "collision_flag",
        "road_boundary_flag",
        "lane_buffer_flag",
        "kinematic_flag",
        "candidate_any_feasible",
        "candidate_all_invalid",
        "candidate_failure_reasons",
    ]
    old_keep = old[[c for c in keep_old if c in old.columns]].rename(
        columns={
            "planner_failure_reason": "old_planner_failure_reason",
            "failure_subtype": "old_failure_subtype",
            "taxonomy_rule_id": "old_taxonomy_rule_id",
        }
    )
    fixed_keep = fixed[[c for c in keep_fixed if c in fixed.columns]].rename(
        columns={
            "failure_taxonomy": "fixed_failure_taxonomy",
            "planner_failure_reason": "fixed_planner_failure_reason",
            "failure_subtype": "fixed_failure_subtype",
            "taxonomy_rule_id": "fixed_taxonomy_rule_id",
        }
    )
    merged = old_keep.merge(fixed_keep, on="sample_id", how="inner")
    merged["recode"] = merged["old_failure_taxonomy"].astype(str) + "_to_" + merged["fixed_failure_taxonomy"].astype(str)
    merged["recode_changed"] = merged["old_failure_taxonomy"].astype(str) != merged["fixed_failure_taxonomy"].astype(str)
    merged["config_hash"] = cfg_hash
    return merged.to_dict("records")


def _delta_gate(deltas: pd.DataFrame, enhanced: str, baseline: str, metric: str) -> tuple[bool, float, float, float]:
    if deltas.empty:
        return False, float("nan"), float("nan"), float("nan")
    rows = deltas[
        (deltas["enhanced_score"].astype(str) == enhanced)
        & (deltas["baseline_score"].astype(str) == baseline)
        & (deltas["metric"].astype(str) == metric)
    ]
    if rows.empty:
        return False, float("nan"), float("nan"), float("nan")
    row = rows.iloc[0]
    delta = pd.to_numeric(pd.Series([row.get("delta")]), errors="coerce").iloc[0]
    ci_low = pd.to_numeric(pd.Series([row.get("ci_low")]), errors="coerce").iloc[0]
    ci_high = pd.to_numeric(pd.Series([row.get("ci_high")]), errors="coerce").iloc[0]
    return bool(pd.notna(delta) and pd.notna(ci_low) and delta > 0 and ci_low > 0), float(delta), float(ci_low), float(ci_high)


def _initial_overlap_gate(initial_overlap: pd.DataFrame | None) -> tuple[bool, list[str]]:
    if initial_overlap is None or initial_overlap.empty:
        return False, ["initial_overlap_sensitivity missing"]
    frame = initial_overlap[initial_overlap["sensitivity_mode"].astype(str) == "exclude_initial_overlap_count_gt0"].copy()
    lines: list[str] = []
    ok = True
    for baseline in ["distance_inverse", "TTC_inverse"]:
        for score in ["temporal_composite", "ROF_v2_no_asr_composite"]:
            a = frame[frame["score"].astype(str) == score]
            b = frame[frame["score"].astype(str) == baseline]
            if a.empty or b.empty:
                ok = False
                lines.append(f"{score} vs {baseline}: missing")
                continue
            auprc_delta = float(a.iloc[0].get("AUPRC", float("nan"))) - float(b.iloc[0].get("AUPRC", float("nan")))
            recall_delta = float(a.iloc[0].get("Recall@5%FPR", float("nan"))) - float(b.iloc[0].get("Recall@5%FPR", float("nan")))
            good = auprc_delta > 0 and recall_delta > 0
            ok = ok and good
            lines.append(f"{score} vs {baseline}: AUPRC delta={auprc_delta:.6g}, Recall@5%FPR delta={recall_delta:.6g}, positive={good}")
    return ok, lines


def _stratum_negative_gate(strata: pd.DataFrame | None, min_positives: int = 10) -> tuple[bool, list[str]]:
    if strata is None or strata.empty:
        return False, ["stratum_metrics missing"]
    rows: list[str] = []
    bad: list[str] = []
    for keys, group in strata.groupby(["stratum_column", "stratum"], dropna=False):
        pos = pd.to_numeric(group.get("positive_count", pd.Series(dtype=float)), errors="coerce").max()
        if pd.isna(pos) or int(pos) < min_positives:
            continue
        for baseline in ["distance_inverse", "TTC_inverse"]:
            temporal = group[group["score"].astype(str) == "temporal_composite"]
            base = group[group["score"].astype(str) == baseline]
            if temporal.empty or base.empty:
                continue
            delta = float(temporal.iloc[0].get("AUPRC", float("nan"))) - float(base.iloc[0].get("AUPRC", float("nan")))
            if delta < -0.02:
                bad.append(f"{keys[0]}={keys[1]} temporal_vs_{baseline}_AUPRC_delta={delta:.6g} positives={int(pos)}")
    rows.extend(bad[:20])
    return len(bad) == 0, rows or ["no adequate-positive stratum with clearly negative temporal AUPRC delta"]


def _report(
    cfg: dict[str, Any],
    cfg_hash: str,
    sample_manifest: pd.DataFrame,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    taxonomy: pd.DataFrame,
    burst_rows: list[dict[str, Any]],
    initial_overlap: pd.DataFrame | None,
    recode_audit: pd.DataFrame | None,
    stratum_df: pd.DataFrame | None,
    outputs: list[Path],
) -> str:
    counts = {str(row["failure_taxonomy"]): int(row["count"]) for _, row in taxonomy.iterrows()} if not taxonomy.empty else {}
    run_name = str((cfg.get("pilot") or {}).get("name") or (cfg.get("dryrun") or {}).get("name") or "v110_commonroad_scaleup")
    suite = cfg.get("planner_label_suite") or {}
    variants = ", ".join(str(v) for v in suite.get("variants", [])) or "lattice_base"
    optional = suite.get("optional_variants") or {}
    is_full = bool((cfg.get("full_gates") or {}).get("enabled")) or "full_10k" in run_name or "full" in run_name
    lines = [
        "# v110 CommonRoad Scale-up Report",
        "",
        f"- run_name: {run_name}",
        f"- config_hash: {cfg_hash}",
        "- cohort_selection: outcome_blind_neutral_stratified",
        f"- planner_family: {variants}",
        "- primary_endpoint: known planner failure only; unknown failures excluded",
        "- bootstrap_unit: scenario_id",
        "",
        "## Planner Suite Status",
        "",
        f"- executed_variants: {variants}",
    ]
    if optional:
        for name, meta in optional.items():
            reason = meta.get("reason", "") if isinstance(meta, dict) else str(meta)
            enabled = meta.get("enabled", False) if isinstance(meta, dict) else False
            lines.append(f"- {name}: enabled={enabled}; reason={reason}")
    if is_full:
        lines.extend(
            [
                "- planner variants not yet implemented; current result is lattice_base-only expanded external validation.",
                "- current claim boundary: lattice_base-only expanded external validation.",
                "- not claimed: multi-planner robustness.",
                "- next required step: v112 stronger baselines and v110b planner-variant sensitivity.",
            ]
        )
    lines.extend(
        [
        "",
        "## Cohort",
        "",
        f"- samples: {len(sample_manifest)}",
        f"- scenarios: {sample_manifest['scenario_id'].nunique() if 'scenario_id' in sample_manifest.columns else 'NA'}",
        f"- max_samples_per_scenario: {sample_manifest.groupby('scenario_id').size().max() if 'scenario_id' in sample_manifest.columns and len(sample_manifest) else 'NA'}",
        "",
        "## Failure Taxonomy",
        "",
        f"- known_failure: {counts.get('known_failure', 0)}",
        f"- unknown_failure: {counts.get('unknown_failure', 0)}",
        f"- no_failure: {counts.get('no_failure', 0)}",
        "",
        ]
    )
    if recode_audit is not None and not recode_audit.empty:
        old_counts = Counter(recode_audit["old_failure_taxonomy"].astype(str))
        fixed_counts = Counter(recode_audit["fixed_failure_taxonomy"].astype(str))
        rule_counts = Counter(recode_audit["fixed_taxonomy_rule_id"].astype(str))
        positive_scenarios = labels.loc[labels["failure_taxonomy"] == "known_failure", "scenario_id"].nunique() if "scenario_id" in labels.columns else 0
        unknown_gate = counts.get("unknown_failure", 0) <= counts.get("known_failure", 0)
        lines.extend(
            [
                "## Fixed Taxonomy Recode Audit",
                "",
                f"- old known_failure: {old_counts.get('known_failure', 0)}",
                f"- old unknown_failure: {old_counts.get('unknown_failure', 0)}",
                f"- old no_failure: {old_counts.get('no_failure', 0)}",
                f"- fixed known_failure: {fixed_counts.get('known_failure', 0)}",
                f"- fixed unknown_failure: {fixed_counts.get('unknown_failure', 0)}",
                f"- fixed no_failure: {fixed_counts.get('no_failure', 0)}",
                f"- known_failure positive scenario count: {positive_scenarios}",
                f"- unknown_failure / known_failure gate passes: {unknown_gate}",
                "",
                "### Recoded By Rule",
                "",
            ]
        )
        for rule, count in sorted(rule_counts.items()):
            lines.append(f"- {rule}: {count}")
        lines.append("")
    lines.extend(["## Primary Metrics", ""])
    if metrics.empty:
        lines.append("- no metrics generated")
    else:
        for _, row in metrics.sort_values(["AUPRC", "Recall@5%FPR"], ascending=False, na_position="last").iterrows():
            lines.append(
                f"- {row['score']}: AUPRC={row.get('AUPRC'):.6g}, "
                f"Recall@5%FPR={row.get('Recall@5%FPR'):.6g}, "
                f"AUROC={row.get('AUROC'):.6g}, Recall@1%FPR={row.get('Recall@1%FPR'):.6g}"
            )
    lines.extend(["", "## Bootstrap Deltas", ""])
    if deltas.empty:
        lines.append("- no bootstrap deltas generated")
    else:
        for _, row in deltas.sort_values(["metric", "delta"], ascending=[True, False], na_position="last").iterrows():
            lines.append(
                f"- {row['enhanced_score']} vs {row['baseline_score']} {row['metric']}: "
                f"delta={row.get('delta'):.6g}, CI=({row.get('ci_low'):.6g}, {row.get('ci_high'):.6g}), "
                f"valid_bootstrap={int(row.get('n_bootstrap_valid', 0))}"
            )
    lines.extend(["", "## Parser/Route/Initial Validity Burst Check", ""])
    for row in burst_rows:
        lines.append(
            f"- {row['reason_token']}: count={row['count']}, scenarios={row['scenario_count']}, "
            f"max_one_scenario={row['max_count_in_one_scenario']}, concentrated={row['concentrated']}"
        )
    if initial_overlap is not None and not initial_overlap.empty:
        lines.extend(["", "## Initial Overlap Sensitivity", ""])
        focus = initial_overlap[
            initial_overlap["score"].isin(
                ["temporal_composite", "ROF_v2_no_asr_composite", "REDI_actionability", "distance_inverse", "TTC_inverse"]
            )
        ]
        for _, row in focus.iterrows():
            lines.append(
                f"- {row['sensitivity_mode']} {row['score']}: "
                f"AUPRC={row.get('AUPRC'):.6g}, Recall@5%FPR={row.get('Recall@5%FPR'):.6g}, "
                f"n={int(row.get('n', 0))}, positives={int(row.get('positive_count', 0))}"
            )
    positive_scenarios_all = labels.loc[labels["failure_taxonomy"] == "known_failure", "scenario_id"].nunique() if "scenario_id" in labels.columns else 0
    if is_full:
        burst_ok = all(int(row.get("count", 0)) == 0 or not bool(row.get("concentrated", False)) for row in burst_rows)
        gate_rows: list[tuple[str, bool, str]] = []
        gate_rows.append(("known positives >= 150", counts.get("known_failure", 0) >= 150, str(counts.get("known_failure", 0))))
        gate_rows.append(("positive scenario IDs >= 75", positive_scenarios_all >= 75, str(positive_scenarios_all)))
        gate_rows.append(("unknown failures <= known failures", counts.get("unknown_failure", 0) <= counts.get("known_failure", 0), f"{counts.get('unknown_failure', 0)} <= {counts.get('known_failure', 0)}"))
        gate_rows.append(("parser_error/no_route/initial_invalid no concentrated burst", burst_ok, "see failure_reason_burst_check.csv"))
        for metric_name in ["auprc", "recall_at_5pct_fpr"]:
            label = "AUPRC" if metric_name == "auprc" else "Recall@5%FPR"
            for baseline in ["distance_inverse", "TTC_inverse"]:
                ok, delta, lo, hi = _delta_gate(deltas, "temporal_composite", baseline, metric_name)
                gate_rows.append((f"temporal_composite beats {baseline} in {label} with CI_low > 0", ok, f"delta={delta:.6g}, CI=({lo:.6g}, {hi:.6g})"))
        for baseline in ["distance_inverse", "TTC_inverse"]:
            ok, delta, lo, hi = _delta_gate(deltas, "ROF_v2_no_asr_composite", baseline, "auprc")
            gate_rows.append((f"ROF_v2_no_asr_composite beats {baseline} in AUPRC with CI_low > 0", ok, f"delta={delta:.6g}, CI=({lo:.6g}, {hi:.6g})"))
        init_ok, init_lines = _initial_overlap_gate(initial_overlap)
        gate_rows.append(("initial-overlap-excluded sensitivity remains positive", init_ok, "; ".join(init_lines[:4])))
        stratum_ok, stratum_lines = _stratum_negative_gate(stratum_df)
        gate_rows.append(("no major adequate-positive stratum shows clearly negative delta", stratum_ok, "; ".join(stratum_lines[:5])))
        all_full_gates = all(item[1] for item in gate_rows)
        lines.extend(["", "## Full Gate Checklist", ""])
        for name, ok, detail in gate_rows:
            lines.append(f"- {name}: {ok} ({detail})")
        lines.append(f"- full_gate_pass: {all_full_gates}")
        lines.extend(
            [
                "",
                "## Claim Boundary",
                "",
                "- lattice_extended: not implemented / not executed",
                "- planner_native_route_lattice: not implemented / not executed",
                "- current claim boundary: lattice_base-only expanded external validation",
                "- next required step: v112 stronger baselines and v110b planner-variant sensitivity",
            ]
        )
    eligible = counts.get("known_failure", 0) >= 15 and counts.get("unknown_failure", 0) <= counts.get("known_failure", 0)
    lines.extend(
        [
            "",
            "## Full 10k Eligibility",
            "",
            f"- taxonomy_gate_known_failure_ge_15: {counts.get('known_failure', 0) >= 15}",
            f"- taxonomy_gate_unknown_le_known: {counts.get('unknown_failure', 0) <= counts.get('known_failure', 0)}",
            f"- fixed pilot eligible for full 10k: {eligible}",
        ]
    )
    lines.extend(["", "## Outputs", ""])
    for path in outputs:
        lines.append(f"- {path}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg_hash = config_hash(args.config)
    inputs = cfg.get("inputs", {})
    cohort_cfg = cfg.get("cohort", {})
    eval_cfg = cfg.get("evaluation", {})
    taxonomy_version = str((cfg.get("taxonomy") or {}).get("version") or DEFAULT_TAXONOMY_VERSION)
    out_dir = experiment_out_dir(cfg, "nc_v110_commonroad_scaleup")

    planner_labels_path = _path_arg(args.planner_labels_csv, cfg, inputs.get("planner_labels_csv"), "planner labels CSV")
    features_path = _path_arg(args.features_csv, cfg, inputs.get("features_csv"), "features CSV")

    labels_all = read_csv_required(planner_labels_path, "planner labels")
    features_all = read_csv_required(features_path, "features")
    reuse_sample_path = resolve_input_path(inputs.get("reuse_sample_manifest_csv"), cfg)
    reuse_cohort_path = resolve_input_path(inputs.get("reuse_cohort_manifest_csv"), cfg)
    if reuse_sample_path and reuse_sample_path.exists():
        sample_manifest = read_csv_required(reuse_sample_path, "reuse sample manifest")
        sample_manifest = _add_scenario_id(sample_manifest)
        if reuse_cohort_path and reuse_cohort_path.exists():
            cohort_manifest = read_csv_required(reuse_cohort_path, "reuse cohort manifest")
        else:
            cohort_manifest = pd.DataFrame(
                [
                    {
                        "scenario_id": sid,
                        "commonroad_scenario_id": group["commonroad_scenario_id"].astype(str).iloc[0],
                        "selected_sample_count": len(group),
                        "cohort_role": "primary_neutral",
                        "selection_protocol": "reused_from_prior_manifest",
                        "selection_seed": cohort_cfg.get("seed", 42),
                    }
                    for sid, group in sample_manifest.groupby("scenario_id", sort=True)
                ]
            )
        diagnostics = [
            {"metric": "reused_sample_manifest", "value": int(len(sample_manifest)), "path": str(reuse_sample_path)},
            {"metric": "reused_cohort_manifest", "value": int(len(cohort_manifest)), "path": str(reuse_cohort_path or "")},
        ]
    else:
        sample_candidates_path = _path_arg(args.sample_candidates_csv, cfg, inputs.get("sample_candidates_csv"), "sample candidates CSV")
        candidates = read_csv_required(sample_candidates_path, "sample candidates")
        coverage_diagnostics: list[dict[str, Any]] = []
        if bool(cohort_cfg.get("require_label_feature_coverage", True)):
            label_ids = set(labels_all["sample_id"].astype(str))
            feature_ids = set(features_all["sample_id"].astype(str))
            available_ids = label_ids.intersection(feature_ids)
            before = int(len(candidates))
            candidates = candidates[candidates["sample_id"].astype(str).isin(available_ids)].copy()
            coverage_diagnostics.extend(
                [
                    {
                        "metric": "candidate_samples_before_label_feature_coverage_filter",
                        "value": before,
                        "note": "coverage only; planner outcomes and actionability scores not used for selection",
                    },
                    {
                        "metric": "candidate_samples_after_label_feature_coverage_filter",
                        "value": int(len(candidates)),
                        "note": "coverage only; planner outcomes and actionability scores not used for selection",
                    },
                    {
                        "metric": "label_feature_sample_id_intersection",
                        "value": int(len(available_ids)),
                        "note": "coverage only; planner outcomes and actionability scores not used for selection",
                    },
                ]
            )
        cohort_manifest, sample_manifest, diagnostics = select_neutral_stratified_cohort(
            candidates,
            target_samples_min=int(cohort_cfg.get("target_samples_min", 100)),
            target_samples_max=int(cohort_cfg.get("target_samples_max", 100)),
            min_unique_scenarios=int(cohort_cfg.get("min_unique_scenarios", 20)),
            max_samples_per_scenario=int(cohort_cfg.get("max_samples_per_scenario", 5)),
            seed=int(args.seed if args.seed is not None else cohort_cfg.get("seed", 42)),
        )
        diagnostics = [*coverage_diagnostics, *diagnostics]
    sample_manifest = _add_scenario_id(sample_manifest)
    sample_ids = set(sample_manifest["sample_id"].astype(str))

    raw_selected_labels = _filter_sample_ids(labels_all, sample_ids)
    planner_cfg = cfg.get("planner") or {}
    planner_family = str(planner_cfg.get("family") or planner_cfg.get("planner_family") or (cfg.get("full") or {}).get("planner_family") or "lattice_base")
    planner_labels = _prepare_planner_labels(raw_selected_labels, taxonomy_version=taxonomy_version, planner_family=planner_family)
    planner_labels = _finalize_planner_label_schema(planner_labels, sample_manifest, cfg)
    feature_scores = _prepare_feature_scores(_filter_sample_ids(features_all, sample_ids))

    present_ids = set(planner_labels["sample_id"].astype(str)).intersection(set(feature_scores["sample_id"].astype(str)))
    if present_ids != sample_ids:
        sample_manifest = sample_manifest[sample_manifest["sample_id"].astype(str).isin(present_ids)].copy()
        planner_labels = planner_labels[planner_labels["sample_id"].astype(str).isin(present_ids)].copy()
        feature_scores = feature_scores[feature_scores["sample_id"].astype(str).isin(present_ids)].copy()

    score_cols = _split_csv(",".join(eval_cfg.get("score_columns", []))) or default_score_columns(feature_scores)
    score_cols = [c for c in score_cols if c in feature_scores.columns]
    if not score_cols:
        raise ValueError("no available score columns after deriving feature scores")
    merged = merge_scores_labels(feature_scores, planner_labels, sample_manifest)
    metrics_rows = evaluate_external_scores(merged, score_cols, endpoint="known_failure")
    comparisons_arg = eval_cfg.get("comparisons", "")
    comparisons = parse_comparisons(comparisons_arg) if comparisons_arg else []
    bootstrap_n = int(args.bootstrap_n if args.bootstrap_n is not None else eval_cfg.get("bootstrap_replicates", 200))
    bootstrap_seed = int(args.seed if args.seed is not None else eval_cfg.get("bootstrap_seed", 42))
    delta_rows = scenario_bootstrap_deltas(merged, comparisons, n_bootstrap=bootstrap_n, seed=bootstrap_seed) if comparisons else []
    taxonomy_rows = failure_taxonomy_rows(planner_labels)
    stratum_rows = stratum_metrics(merged, score_cols)
    unknown_rows = unknown_failure_sensitivity(merged, score_cols)
    initial_overlap_rows = _initial_overlap_sensitivity(merged, score_cols)
    previous_labels_path = resolve_input_path(inputs.get("previous_planner_labels_csv"), cfg)
    previous_labels = read_csv_required(previous_labels_path, "previous planner labels") if previous_labels_path and previous_labels_path.exists() else pd.DataFrame()
    recode_source = previous_labels if not previous_labels.empty else raw_selected_labels
    recode_rows = _taxonomy_recode_audit(recode_source, planner_labels, cfg_hash) if not recode_source.empty else []
    burst_rows = _burst_summary(planner_labels)
    reason_rows = [
        {"planner_failure_reason": reason, "count": int(count)}
        for reason, count in sorted(_reason_counts(planner_labels).items(), key=lambda item: (-item[1], item[0]))
    ]

    cohort_path = out_dir / "cohort_manifest.csv"
    sample_path = out_dir / "sample_manifest.csv"
    labels_path = out_dir / "planner_labels.csv"
    feature_path = out_dir / "feature_score_table.csv"
    taxonomy_path = out_dir / "failure_taxonomy.csv"
    metrics_path = out_dir / "external_metrics.csv"
    deltas_path = out_dir / "external_bootstrap_deltas.csv"
    stratum_path = out_dir / "stratum_metrics.csv"
    unknown_path = out_dir / "unknown_failure_sensitivity.csv"
    initial_overlap_path = out_dir / "initial_overlap_sensitivity.csv"
    recode_audit_path = out_dir / "taxonomy_recode_audit.csv"
    diagnostics_path = out_dir / "cohort_selection_diagnostics.csv"
    reasons_path = out_dir / "failure_reason_summary.csv"
    burst_path = out_dir / "failure_reason_burst_check.csv"
    report_path = out_dir / "v110_commonroad_scaleup_report.md"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"

    _write_dataframe(cohort_path, cohort_manifest, cfg_hash)
    _write_dataframe(sample_path, sample_manifest, cfg_hash)
    _write_dataframe(labels_path, planner_labels, cfg_hash)
    _write_dataframe(feature_path, feature_scores, cfg_hash)
    write_csv(taxonomy_path, add_config_hash(taxonomy_rows, cfg_hash))
    write_csv(metrics_path, add_config_hash(metrics_rows, cfg_hash))
    write_csv(deltas_path, add_config_hash(delta_rows, cfg_hash))
    write_csv(stratum_path, add_config_hash(stratum_rows, cfg_hash))
    write_csv(unknown_path, add_config_hash(unknown_rows, cfg_hash))
    write_csv(initial_overlap_path, add_config_hash(initial_overlap_rows, cfg_hash))
    write_csv(recode_audit_path, add_config_hash(recode_rows, cfg_hash))
    write_csv(diagnostics_path, add_config_hash(diagnostics, cfg_hash))
    write_csv(reasons_path, add_config_hash(reason_rows, cfg_hash))
    write_csv(burst_path, add_config_hash(burst_rows, cfg_hash))

    metrics_df = pd.DataFrame(metrics_rows)
    deltas_df = pd.DataFrame(delta_rows)
    taxonomy_df = pd.DataFrame(taxonomy_rows)
    required_outputs = [
        cohort_path,
        sample_path,
        labels_path,
        taxonomy_path,
        feature_path,
        metrics_path,
        deltas_path,
        stratum_path,
        unknown_path,
        initial_overlap_path,
        recode_audit_path,
        report_path,
    ]
    extra_outputs = [diagnostics_path, reasons_path, burst_path]
    report_path.write_text(
        _report(
            cfg,
            cfg_hash,
            sample_manifest,
            planner_labels,
            metrics_df,
            deltas_df,
            taxonomy_df,
            burst_rows,
            pd.DataFrame(initial_overlap_rows),
            pd.DataFrame(recode_rows),
            pd.DataFrame(stratum_rows),
            required_outputs + extra_outputs,
        ),
        encoding="utf-8",
    )
    all_outputs = required_outputs + extra_outputs
    write_csv(manifest_path, artifact_manifest_rows(args.config, all_outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*all_outputs, manifest_path]))

    counts = {row["failure_taxonomy"]: row["count"] for row in taxonomy_rows}
    print(
        "[v110-scaleup] "
        f"samples={len(sample_manifest)} "
        f"scenarios={sample_manifest['scenario_id'].nunique()} "
        f"known_failure={counts.get('known_failure', 0)} "
        f"unknown_failure={counts.get('unknown_failure', 0)} "
        f"no_failure={counts.get('no_failure', 0)} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
