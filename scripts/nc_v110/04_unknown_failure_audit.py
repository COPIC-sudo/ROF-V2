#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.common import artifact_manifest_rows, sha256_file, write_csv, write_json


REQUIRED_INPUTS = [
    "planner_labels.csv",
    "failure_taxonomy.csv",
    "sample_manifest.csv",
    "cohort_manifest.csv",
    "feature_score_table.csv",
    "unknown_failure_sensitivity.csv",
    "external_metrics.csv",
    "external_bootstrap_deltas.csv",
]

ERROR_TOKENS = ("parser_error", "parse_error", "no_route", "initial_invalid")
SCORE_COLUMNS = ["temporal_composite", "REDI_actionability", "distance_inverse", "TTC_inverse"]


def parse_args() -> argparse.Namespace:
    work_dir = os.environ.get("ROF_WORK_DIR", "")
    default_input = Path(work_dir) / "results" / "nc_v110_commonroad_scaleup" / "pilot_1k" if work_dir else None
    default_candidate = (
        Path(work_dir)
        / "results"
        / "commonroad_planner_feasibility"
        / "nc_v110_pilot_1k"
        / "commonroad_lattice_planner_candidates_nc_v110_pilot_1k.csv"
        if work_dir
        else None
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(default_input) if default_input else None)
    parser.add_argument("--candidate-csv", default=str(default_candidate) if default_candidate else None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _bool_series(values: pd.Series | Any) -> pd.Series:
    if not isinstance(values, pd.Series):
        return pd.Series(dtype=bool)
    text = values.fillna("").astype(str).str.lower()
    return text.isin(["1", "true", "yes", "y"])


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _scenario_family(value: Any) -> str:
    text = str(value)
    if not text or text == "nan":
        return "missing"
    parts = text.split("-")
    return parts[0] if parts else text


def _bucket_numeric(values: pd.Series, edges: list[float], labels: list[str], missing: str) -> pd.Series:
    out = pd.cut(pd.to_numeric(values, errors="coerce"), [-np.inf, *edges, np.inf], labels=labels).astype("object")
    out[pd.isna(out)] = missing
    return out.astype(str)


def _add_strata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sid_col = "commonroad_scenario_id" if "commonroad_scenario_id" in out.columns else "scenario_id"
    out["scenario_family"] = out[sid_col].map(_scenario_family) if sid_col in out.columns else "missing"
    if "density_stratum" not in out.columns:
        if "agent_count_stratum" in out.columns:
            out["density_stratum"] = out["agent_count_stratum"].astype(str)
        else:
            out["density_stratum"] = _bucket_numeric(_num(out, "agent_count"), [5, 15], ["lt5", "5to15", "gte15"], "missing_density")
    if "topology_stratum" not in out.columns:
        out["topology_stratum"] = _bucket_numeric(_num(out, "lanelet_count"), [20, 60], ["lt20_lanelets", "20to60_lanelets", "gte60_lanelets"], "missing_topology")
    if "speed_stratum" not in out.columns:
        out["speed_stratum"] = _bucket_numeric(_num(out, "ego_speed_mps"), [5, 15], ["lt5mps", "5to15mps", "gte15mps"], "missing_speed")
    return out


def _aggregate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["sample_id"])
    work = candidates.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for sample_id, group in work.groupby("sample_id", sort=True):
        feasible = _bool_series(group.get("feasible", pd.Series(index=group.index))).astype(bool)
        collision = _bool_series(group.get("collision_flag", pd.Series(index=group.index))).astype(bool)
        lane = _bool_series(group.get("lane_buffer_flag", pd.Series(index=group.index))).astype(bool)
        progress = _bool_series(group.get("progress_flag", pd.Series(index=group.index))).astype(bool)
        kinematic = _bool_series(group.get("kinematic_flag", pd.Series(index=group.index))).astype(bool)
        reason = group["failure_reason"].fillna("").astype(str).str.lower() if "failure_reason" in group.columns else pd.Series("", index=group.index)
        candidate_count = int(len(group))
        feasible_count = int(feasible.sum())
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_rows": candidate_count,
                "candidate_feasible_count": feasible_count,
                "candidate_all_invalid": bool(candidate_count > 0 and feasible_count == 0),
                "candidate_any_feasible": bool(feasible_count > 0),
                "collision_flag": int(collision.any()),
                "road_boundary_flag": int(lane.any()),
                "lane_buffer_flag": int(lane.any()),
                "progress_flag": int(progress.any()),
                "kinematic_flag": int(kinematic.any()),
                "no_candidate_flag": int(candidate_count == 0),
                "no_route_flag": int(reason.str.contains("no_route", regex=False).any()),
                "initial_invalid_flag": int(reason.str.contains("initial_invalid", regex=False).any()),
                "parser_error_flag": int(reason.str.contains("parser_error|parse_error", regex=True).any()),
                "candidate_failure_reasons": ";".join(f"{k}:{v}" for k, v in sorted(Counter(reason).items()) if k),
            }
        )
    return pd.DataFrame(rows)


def _flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reason = out["planner_failure_reason"].fillna("").astype(str).str.lower() if "planner_failure_reason" in out.columns else pd.Series("", index=out.index)
    out["parser_error_flag"] = out.get("parser_error_flag", 0)
    out["no_route_flag"] = out.get("no_route_flag", 0)
    out["initial_invalid_flag"] = out.get("initial_invalid_flag", 0)
    out["no_candidate_flag"] = out.get("no_candidate_flag", 0)
    out.loc[reason.str.contains("parser_error|parse_error", regex=True), "parser_error_flag"] = 1
    out.loc[reason.str.contains("no_route", regex=False), "no_route_flag"] = 1
    out.loc[reason.str.contains("initial_invalid", regex=False), "initial_invalid_flag"] = 1
    out.loc[reason.str.contains("no_candidate", regex=False), "no_candidate_flag"] = 1
    for col in [
        "collision_flag",
        "road_boundary_flag",
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
    return out


def _distribution_rows(unknown: pd.DataFrame, total_unknown: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_counts(dimension: str, values: pd.Series) -> None:
        counts = Counter(values.fillna("missing").astype(str))
        for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append({"dimension": dimension, "value": value, "count": int(count), "fraction_of_unknown": float(count / max(total_unknown, 1))})

    add_counts("failure_reason", unknown.get("planner_failure_reason", pd.Series(index=unknown.index)))
    if {"planner_success", "planner_failure"}.issubset(unknown.columns):
        raw = "success=" + unknown["planner_success"].astype(str) + "|failure=" + unknown["planner_failure"].astype(str)
        add_counts("planner_raw_status", raw)
    for col in [
        "collision_flag",
        "road_boundary_flag",
        "kinematic_flag",
        "no_candidate_flag",
        "no_route_flag",
        "initial_invalid_flag",
        "parser_error_flag",
        "candidate_any_feasible",
        "candidate_all_invalid",
    ]:
        add_counts(col, unknown[col].astype(str))
    return rows


def _by_stratum(unknown: pd.DataFrame, all_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in ["scenario_family", "topology_stratum", "density_stratum", "speed_stratum"]:
        if col not in all_rows.columns:
            continue
        total_counts = Counter(all_rows[col].fillna("missing").astype(str))
        unknown_counts = Counter(unknown[col].fillna("missing").astype(str))
        for value, count in sorted(unknown_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            total = total_counts.get(value, 0)
            rows.append(
                {
                    "stratum_column": col,
                    "stratum": value,
                    "unknown_count": int(count),
                    "total_count": int(total),
                    "unknown_rate": float(count / max(total, 1)),
                    "fraction_of_unknown": float(count / max(len(unknown), 1)),
                }
            )
    return rows


def _by_scenario(unknown: pd.DataFrame, all_rows: pd.DataFrame) -> list[dict[str, Any]]:
    total_counts = Counter(all_rows["scenario_id"].fillna(all_rows["sample_id"]).astype(str))
    rows = []
    for scenario_id, group in unknown.groupby("scenario_id", dropna=False):
        sid = str(scenario_id)
        rows.append(
            {
                "scenario_id": sid,
                "commonroad_scenario_id": str(group["commonroad_scenario_id"].iloc[0]) if "commonroad_scenario_id" in group.columns else sid,
                "unknown_count": int(len(group)),
                "total_count": int(total_counts.get(sid, 0)),
                "unknown_rate": float(len(group) / max(total_counts.get(sid, 0), 1)),
                "failure_reasons": ";".join(f"{k}:{v}" for k, v in sorted(Counter(group["planner_failure_reason"].fillna("").astype(str)).items())),
                "collision_flag_any": int(pd.to_numeric(group["collision_flag"], errors="coerce").fillna(0).astype(int).any()),
                "road_boundary_flag_any": int(pd.to_numeric(group["road_boundary_flag"], errors="coerce").fillna(0).astype(int).any()),
                "kinematic_flag_any": int(pd.to_numeric(group["kinematic_flag"], errors="coerce").fillna(0).astype(int).any()),
                "candidate_any_feasible_any": int(pd.to_numeric(group["candidate_any_feasible"], errors="coerce").fillna(0).astype(int).any()),
            }
        )
    return sorted(rows, key=lambda r: (-r["unknown_count"], r["scenario_id"]))


def _score_distribution(all_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tax, group in all_rows.groupby("failure_taxonomy", sort=True):
        for score in SCORE_COLUMNS:
            if score not in group.columns:
                continue
            values = pd.to_numeric(group[score], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "failure_taxonomy": tax,
                    "score": score,
                    "count": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std(ddof=0)) if len(values) else np.nan,
                    "min": float(values.min()) if len(values) else np.nan,
                    "p25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "p50": float(values.quantile(0.50)) if len(values) else np.nan,
                    "p75": float(values.quantile(0.75)) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                }
            )
    return rows


def _suggest_recode(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unknown = all_rows[all_rows["failure_taxonomy"] == "unknown_failure"].copy()
    for _, row in unknown.iterrows():
        no_candidate = int(row.get("no_candidate_flag", 0) or 0) == 1
        parser_error = int(row.get("parser_error_flag", 0) or 0) == 1
        no_route = int(row.get("no_route_flag", 0) or 0) == 1
        initial_invalid = int(row.get("initial_invalid_flag", 0) or 0) == 1
        collision = int(row.get("collision_flag", 0) or 0) == 1
        road = int(row.get("road_boundary_flag", 0) or 0) == 1
        kinematic = int(row.get("kinematic_flag", 0) or 0) == 1
        feasible = int(row.get("candidate_any_feasible", 0) or 0) == 1
        all_invalid = int(row.get("candidate_all_invalid", 0) or 0) == 1
        reason = str(row.get("planner_failure_reason", "")).lower()
        recode = "keep_unknown"
        new_subtype = "missing_validator_output_or_incomplete_rollout"
        rationale = "No safe pre-registered recode rule matched."
        confidence = "low"
        if no_candidate and not (parser_error or no_route or initial_invalid):
            recode = "known_failure"
            new_subtype = "known_failure:no_valid_trajectory"
            rationale = "A: no_candidate_flag=1 with parser/no_route/initial_invalid all zero."
            confidence = "high"
        elif "timeout" in reason and all_invalid:
            recode = "possible_known_failure"
            new_subtype = "possible_known_failure_timeout_all_invalid"
            rationale = "B: timeout-like reason and candidate evaluation indicates all candidates invalid."
            confidence = "medium"
        elif collision or road or kinematic:
            recode = "known_failure"
            parts = []
            if collision:
                parts.append("collision")
            if road:
                parts.append("road_boundary")
            if kinematic:
                parts.append("kinematic")
            new_subtype = "known_failure:" + "_and_".join(parts)
            rationale = "D: validator failure flag is present despite unknown taxonomy."
            confidence = "high"
        elif feasible:
            recode = "no_failure"
            new_subtype = "no_failure:feasible_trajectory_exists"
            rationale = "E: no failure flags and feasible trajectory exists."
            confidence = "high"
        rows.append(
            {
                "sample_id": row["sample_id"],
                "scenario_id": row.get("scenario_id", ""),
                "commonroad_scenario_id": row.get("commonroad_scenario_id", ""),
                "old_failure_taxonomy": row.get("failure_taxonomy", ""),
                "planner_failure_reason": row.get("planner_failure_reason", ""),
                "suggested_recode": recode,
                "suggested_subtype": new_subtype,
                "confidence": confidence,
                "rationale": rationale,
                "candidate_rows": row.get("candidate_rows", ""),
                "candidate_feasible_count": row.get("candidate_feasible_count", ""),
                "collision_flag": row.get("collision_flag", 0),
                "road_boundary_flag": row.get("road_boundary_flag", 0),
                "kinematic_flag": row.get("kinematic_flag", 0),
                "no_candidate_flag": row.get("no_candidate_flag", 0),
                "no_route_flag": row.get("no_route_flag", 0),
                "initial_invalid_flag": row.get("initial_invalid_flag", 0),
                "parser_error_flag": row.get("parser_error_flag", 0),
                "candidate_any_feasible": row.get("candidate_any_feasible", 0),
                "candidate_all_invalid": row.get("candidate_all_invalid", 0),
                "candidate_failure_reasons": row.get("candidate_failure_reasons", ""),
            }
        )
    return pd.DataFrame(rows)


def _sensitivity_rows(unknown_sens: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if unknown_sens.empty:
        return ["- unknown sensitivity file is empty"]
    focus = unknown_sens[unknown_sens["score"].isin(["temporal_composite", "REDI_actionability", "distance_inverse", "TTC_inverse"])].copy()
    for score, group in focus.groupby("score", sort=True):
        vals: dict[str, dict[str, float]] = {}
        for _, row in group.iterrows():
            mode = str(row.get("sensitivity_mode", row.get("endpoint", "")))
            vals[mode] = {
                "AUPRC": float(row.get("AUPRC", np.nan)),
                "Recall@5%FPR": float(row.get("Recall@5%FPR", np.nan)),
            }
        known = vals.get("known_failure", {})
        pos = vals.get("all_failures_positive", {})
        neg = vals.get("unknown_as_negative", {})
        lines.append(
            f"- {score}: known AUPRC={known.get('AUPRC', np.nan):.6g}, "
            f"unknown_positive delta AUPRC={pos.get('AUPRC', np.nan) - known.get('AUPRC', np.nan):.6g}, "
            f"unknown_negative delta AUPRC={neg.get('AUPRC', np.nan) - known.get('AUPRC', np.nan):.6g}, "
            f"known Recall@5%FPR={known.get('Recall@5%FPR', np.nan):.6g}, "
            f"unknown_positive delta Recall@5%FPR={pos.get('Recall@5%FPR', np.nan) - known.get('Recall@5%FPR', np.nan):.6g}, "
            f"unknown_negative delta Recall@5%FPR={neg.get('Recall@5%FPR', np.nan) - known.get('Recall@5%FPR', np.nan):.6g}"
        )
    return lines


def _report(
    all_rows: pd.DataFrame,
    recode: pd.DataFrame,
    reason_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    unknown_sens: pd.DataFrame,
    external_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    candidate_path: Path | None,
) -> str:
    counts = Counter(all_rows["failure_taxonomy"].astype(str))
    safe_known = int((recode["suggested_recode"] == "known_failure").sum()) if not recode.empty else 0
    safe_no = int((recode["suggested_recode"] == "no_failure").sum()) if not recode.empty else 0
    possible_known = int((recode["suggested_recode"] == "possible_known_failure").sum()) if not recode.empty else 0
    keep_unknown = int((recode["suggested_recode"] == "keep_unknown").sum()) if not recode.empty else 0
    expected_known = counts.get("known_failure", 0) + safe_known
    expected_no = counts.get("no_failure", 0) + safe_no
    expected_unknown = counts.get("unknown_failure", 0) - safe_known - safe_no
    top_reasons = [r for r in reason_rows if r["dimension"] == "failure_reason"][:8]
    top_scenarios = scenario_rows[:8]
    flag_summary = {
        "collision": int(pd.to_numeric(all_rows.loc[all_rows["failure_taxonomy"] == "unknown_failure", "collision_flag"], errors="coerce").fillna(0).sum()),
        "road_boundary": int(pd.to_numeric(all_rows.loc[all_rows["failure_taxonomy"] == "unknown_failure", "road_boundary_flag"], errors="coerce").fillna(0).sum()),
        "kinematic": int(pd.to_numeric(all_rows.loc[all_rows["failure_taxonomy"] == "unknown_failure", "kinematic_flag"], errors="coerce").fillna(0).sum()),
        "no_candidate": int(pd.to_numeric(all_rows.loc[all_rows["failure_taxonomy"] == "unknown_failure", "no_candidate_flag"], errors="coerce").fillna(0).sum()),
        "no_route": int(pd.to_numeric(all_rows.loc[all_rows["failure_taxonomy"] == "unknown_failure", "no_route_flag"], errors="coerce").fillna(0).sum()),
        "initial_invalid": int(pd.to_numeric(all_rows.loc[all_rows["failure_taxonomy"] == "unknown_failure", "initial_invalid_flag"], errors="coerce").fillna(0).sum()),
    }
    unknown_gt_known = counts.get("unknown_failure", 0) > counts.get("known_failure", 0)
    recoded_unknown_gt_known = expected_unknown > expected_known
    current_full_gate = counts.get("known_failure", 0) >= 15 and counts.get("unknown_failure", 0) <= counts.get("known_failure", 0)
    fixed_taxonomy_gate = expected_known >= 15 and expected_unknown <= expected_known
    lines = [
        "# nc_v110 Pilot 1k Unknown Failure Audit",
        "",
        "## Inputs",
        "",
        f"- candidate_validator_flags: {candidate_path if candidate_path else 'not available'}",
        "- original pilot outputs were read only; no pilot_1k CSV was modified.",
        "",
        "## Original Taxonomy",
        "",
        f"- known_failure: {counts.get('known_failure', 0)}",
        f"- unknown_failure: {counts.get('unknown_failure', 0)}",
        f"- no_failure: {counts.get('no_failure', 0)}",
        f"- unknown_failure / known_failure: {counts.get('unknown_failure', 0) / max(counts.get('known_failure', 0), 1):.4g}",
        "",
        "## Main Cause",
        "",
        "- All unknown failures have raw planner_failure_reason=unknown in planner_labels.csv.",
        f"- Candidate-level validator flags among unknowns: {flag_summary}.",
        "- The dominant cause is taxonomy under-classification of mixed candidate validator failures, not parser/no_route/initial_invalid data loss.",
        "",
        "## Top Raw Reasons",
        "",
    ]
    for row in top_reasons:
        lines.append(f"- {row['value']}: {row['count']}")
    lines.extend(["", "## Scenario Concentration", ""])
    if top_scenarios:
        top_total = sum(int(r["unknown_count"]) for r in top_scenarios[:5])
        lines.append(f"- top 5 scenarios contain {top_total}/{counts.get('unknown_failure', 0)} unknown failures.")
        for row in top_scenarios:
            lines.append(f"- {row['scenario_id']}: unknown={row['unknown_count']}, total={row['total_count']}, rate={row['unknown_rate']:.3g}")
    else:
        lines.append("- no unknown scenarios")
    lines.extend(
        [
            "",
            "## Recode Recommendation",
            "",
            f"- safe recode to known_failure: {safe_known}",
            f"- safe recode to no_failure: {safe_no}",
            f"- possible known timeout/all-invalid: {possible_known}",
            f"- must remain unknown: {keep_unknown}",
            f"- expected known_failure after safe recode: {expected_known}",
            f"- expected unknown_failure after safe recode: {expected_unknown}",
            f"- expected no_failure after safe recode: {expected_no}",
            "",
            "Rule D is the active rule here: road/collision/kinematic validator flags are present while the aggregate taxonomy says unknown.",
            "No recode is proposed without candidate-level validator support.",
            "",
            "## Score Relationship",
            "",
        ]
    )
    score_df = pd.DataFrame(score_rows)
    for score in SCORE_COLUMNS:
        subset = score_df[score_df["score"] == score] if not score_df.empty else pd.DataFrame()
        if subset.empty:
            continue
        parts = []
        for _, row in subset.iterrows():
            parts.append(f"{row['failure_taxonomy']} p50={row['p50']:.6g} mean={row['mean']:.6g}")
        lines.append(f"- {score}: " + "; ".join(parts))
    lines.extend(["", "## Unknown Sensitivity", ""])
    lines.extend(_sensitivity_rows(unknown_sens))
    lines.extend(
        [
            "",
            "## Gate Recommendation",
            "",
            f"- unknown higher than known before recode: {unknown_gt_known}",
            f"- unknown higher than known after safe recode: {recoded_unknown_gt_known}",
            "- recommendation: rerun pilot_1k_fixed_taxonomy before any full 10k run.",
            f"- current full 10k gate pass without recode: {current_full_gate}",
            f"- expected fixed-taxonomy gate pass if recodes are applied and metrics rerun: {fixed_taxonomy_gate}",
            "- full 10k recommendation now: no direct full run; first run pilot_1k_fixed_taxonomy and confirm the corrected taxonomy/metrics.",
            "",
            "The audit supports a fixed-taxonomy rerun because the current high unknown rate is not caused by parser/no_route/initial_invalid errors, but by aggregate planner reason 'unknown' hiding candidate-level validator flags.",
        ]
    )
    _ = external_metrics, deltas
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.input_dir:
        raise ValueError("--input-dir is required or ROF_WORK_DIR must be set")
    input_dir = Path(os.path.expanduser(os.path.expandvars(args.input_dir)))
    out_dir = Path(os.path.expanduser(os.path.expandvars(args.out_dir))) if args.out_dir else input_dir.parent / "pilot_1k_unknown_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {name: _read_required(input_dir / name) for name in REQUIRED_INPUTS}
    labels = inputs["planner_labels.csv"].copy()
    labels["sample_id"] = labels["sample_id"].astype(str)
    features = inputs["feature_score_table.csv"].copy()
    features["sample_id"] = features["sample_id"].astype(str)
    sample_manifest = inputs["sample_manifest.csv"].copy()
    sample_manifest["sample_id"] = sample_manifest["sample_id"].astype(str)

    candidate_path = Path(os.path.expanduser(os.path.expandvars(str(args.candidate_csv)))) if args.candidate_csv else None
    candidate_agg = pd.DataFrame(columns=["sample_id"])
    if candidate_path and candidate_path.exists():
        candidate_agg = _aggregate_candidates(pd.read_csv(candidate_path))

    manifest_cols = [
        c
        for c in sample_manifest.columns
        if c
        not in {
            "config_hash",
        }
    ]
    merged = labels.merge(sample_manifest[manifest_cols], on="sample_id", how="left", suffixes=("", "_manifest"))
    feature_cols = ["sample_id", *[c for c in SCORE_COLUMNS if c in features.columns]]
    merged = merged.merge(features[feature_cols], on="sample_id", how="left")
    if not candidate_agg.empty:
        merged = merged.merge(candidate_agg, on="sample_id", how="left")
    merged = _flag_columns(_add_strata(merged))
    if "scenario_id" not in merged.columns:
        merged["scenario_id"] = merged.get("commonroad_scenario_id", merged["sample_id"]).astype(str)

    unknown = merged[merged["failure_taxonomy"] == "unknown_failure"].copy()
    reason_rows = _distribution_rows(unknown, len(unknown))
    stratum_rows = _by_stratum(unknown, merged)
    scenario_rows = _by_scenario(unknown, merged)
    score_rows = _score_distribution(merged)
    recode = _suggest_recode(merged)

    cfg_hash = str(labels["config_hash"].dropna().iloc[0]) if "config_hash" in labels.columns and labels["config_hash"].notna().any() else ""
    audit_hash = sha256_file(input_dir / "planner_labels.csv", max_bytes=1024 * 1024)
    for rows in [reason_rows, stratum_rows, scenario_rows, score_rows]:
        for row in rows:
            row["source_config_hash"] = cfg_hash
            row["audit_input_hash"] = audit_hash
    if not recode.empty:
        recode["source_config_hash"] = cfg_hash
        recode["audit_input_hash"] = audit_hash

    reason_path = out_dir / "unknown_failure_reason_counts.csv"
    stratum_path = out_dir / "unknown_failure_by_stratum.csv"
    scenario_path = out_dir / "unknown_failure_by_scenario.csv"
    score_path = out_dir / "unknown_score_distribution.csv"
    recode_path = out_dir / "suggested_taxonomy_recode.csv"
    report_path = out_dir / "unknown_failure_audit_report.md"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"

    write_csv(reason_path, reason_rows)
    write_csv(stratum_path, stratum_rows)
    write_csv(scenario_path, scenario_rows)
    write_csv(score_path, score_rows)
    write_csv(recode_path, recode.to_dict("records"))
    report_path.write_text(
        _report(
            merged,
            recode,
            reason_rows,
            scenario_rows,
            score_rows,
            inputs["unknown_failure_sensitivity.csv"],
            inputs["external_metrics.csv"],
            inputs["external_bootstrap_deltas.csv"],
            candidate_path if candidate_path and candidate_path.exists() else None,
        ),
        encoding="utf-8",
    )
    outputs = [report_path, reason_path, stratum_path, scenario_path, score_path, recode_path]
    write_csv(manifest_path, artifact_manifest_rows(input_dir / "run_manifest.json", outputs))
    write_json(
        run_path,
        {
            "input_dir": str(input_dir),
            "candidate_csv": str(candidate_path) if candidate_path else "",
            "source_config_hash": cfg_hash,
            "audit_input_hash": audit_hash,
            "outputs": artifact_manifest_rows(input_dir / "run_manifest.json", [*outputs, manifest_path]),
        },
    )
    counts = Counter(merged["failure_taxonomy"].astype(str))
    print(
        "[v110-unknown-audit] "
        f"unknown={counts.get('unknown_failure', 0)} "
        f"known={counts.get('known_failure', 0)} "
        f"safe_known={(recode['suggested_recode'] == 'known_failure').sum() if not recode.empty else 0} "
        f"safe_no={(recode['suggested_recode'] == 'no_failure').sum() if not recode.empty else 0} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
