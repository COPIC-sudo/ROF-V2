from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TAXONOMY_VERSION = "v110_fixed_taxonomy_001"

DIAGNOSTIC_FIELDS = [
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
    "candidate_failure_reasons",
    "taxonomy_rule_id",
    "taxonomy_version",
    "failure_subtype",
]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _boolish(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[col]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    text = values.fillna("").astype(str).str.lower()
    return text.isin(["1", "true", "yes", "y"])


def _reason(df: pd.DataFrame) -> pd.Series:
    for col in ["planner_failure_reason", "failure_reason", "dominant_failure_reason", "raw_planner_failure_reason"]:
        if col in df.columns:
            return df[col].fillna("").astype(str).str.lower()
    return pd.Series("", index=df.index, dtype=object)


def _fallback_known_from_reason(reason: str) -> str | None:
    text = str(reason).lower()
    if text in ("", "unknown", "nan", "missing"):
        return None
    if "collision" in text and ("lane" in text or "road" in text):
        return "known_failure:collision_and_road_boundary"
    if "collision" in text:
        return "known_failure:collision_only"
    if "lane" in text or "road" in text:
        return "known_failure:road_boundary_only"
    if "kinematic" in text:
        return "known_failure:kinematic_only"
    if "progress" in text:
        return "known_failure:progress_only"
    return None


def _subtype_from_flags(collision: bool, road: bool, kinematic: bool) -> str:
    if collision and road and kinematic:
        return "known_failure:collision_road_boundary_and_kinematic"
    if collision and kinematic:
        return "known_failure:collision_and_kinematic"
    if road and kinematic:
        return "known_failure:road_boundary_and_kinematic"
    if collision and road:
        return "known_failure:collision_and_road_boundary"
    if collision:
        return "known_failure:collision_only"
    if road:
        return "known_failure:road_boundary_only"
    if kinematic:
        return "known_failure:kinematic_only"
    return "unknown_failure:all_invalid_unattributed"


def annotate_planner_failure_taxonomy(df: pd.DataFrame, taxonomy_version: str = DEFAULT_TAXONOMY_VERSION) -> pd.DataFrame:
    """Apply nc_v110 fixed planner-failure taxonomy using validator diagnostics.

    Rules:
    A. parser/no_route/initial/validator-missing/diagnostic-free timeout -> unknown.
    B. any feasible candidate -> no_failure.
    C. no candidates with valid parser/route/initial state -> known no-valid-trajectory.
    D. all candidates invalid and concrete validator flag present -> known.
    E. kinematic-only is known but flagged as secondary sensitivity.
    F. all invalid without concrete flags -> unknown all-invalid-unattributed.
    """
    out = df.copy()
    if out.empty:
        out["failure_taxonomy"] = []
        out["failure_subtype"] = []
        out["taxonomy_rule_id"] = []
        out["taxonomy_version"] = []
        return out

    reason = _reason(out)
    candidate_count = _num(out, "candidate_count", np.nan)
    feasible_count = _num(out, "feasible_candidate_count", np.nan)
    if "candidate_any_feasible" in out.columns:
        any_feasible = _boolish(out, "candidate_any_feasible")
    else:
        any_feasible = feasible_count.fillna(0) > 0
    if "candidate_all_invalid" in out.columns:
        all_invalid = _boolish(out, "candidate_all_invalid")
    else:
        all_invalid = (candidate_count.fillna(0) > 0) & (feasible_count.fillna(0) == 0)

    collision = _boolish(out, "collision_flag")
    lane = _boolish(out, "lane_buffer_flag")
    road = _boolish(out, "road_boundary_flag") | lane
    kinematic = _boolish(out, "kinematic_flag")
    no_candidate = _boolish(out, "no_candidate_flag") | reason.str.contains("no_candidate", regex=False)
    no_route = _boolish(out, "no_route_flag") | reason.str.contains("no_route", regex=False)
    initial_invalid = _boolish(out, "initial_invalid_flag") | reason.str.contains("initial_invalid", regex=False)
    parser_error = _boolish(out, "parser_error_flag") | reason.str.contains("parser_error|parse_error", regex=True)
    validator_missing = _boolish(out, "validator_missing_flag") | reason.str.contains("validator_missing|validator_output_missing", regex=True)
    candidate_table_missing = _boolish(out, "candidate_table_missing_flag") | reason.str.contains("candidate_table_missing", regex=False)
    timeout = reason.str.contains("timeout", regex=False)
    concrete = collision | road | kinematic

    failure = _num(out, "planner_failure", 0).fillna(0).astype(int)
    success = _num(out, "planner_success", 0).fillna(0).astype(int)

    failure_taxonomy: list[str] = []
    failure_subtype: list[str] = []
    rule_id: list[str] = []
    secondary: list[int] = []
    for idx in out.index:
        if parser_error.loc[idx] or no_route.loc[idx] or initial_invalid.loc[idx] or validator_missing.loc[idx] or candidate_table_missing.loc[idx]:
            taxonomy = "unknown_failure"
            if parser_error.loc[idx]:
                subtype = "unknown_failure:parser_error"
                rule = "A_parser_error"
            elif no_route.loc[idx]:
                subtype = "unknown_failure:no_route"
                rule = "A_no_route"
            elif initial_invalid.loc[idx]:
                subtype = "unknown_failure:initial_invalid"
                rule = "A_initial_invalid"
            elif validator_missing.loc[idx]:
                subtype = "unknown_failure:validator_missing"
                rule = "A_validator_missing"
            else:
                subtype = "unknown_failure:candidate_table_missing"
                rule = "A_candidate_table_missing"
        elif any_feasible.loc[idx] or feasible_count.loc[idx] > 0 or success.loc[idx] == 1:
            taxonomy = "no_failure"
            subtype = "no_failure:feasible_trajectory_exists"
            rule = "B_feasible_candidate"
        elif no_candidate.loc[idx] or candidate_count.loc[idx] == 0:
            taxonomy = "known_failure"
            subtype = "known_failure:no_valid_trajectory_no_candidate"
            rule = "C_no_candidate_valid_context"
        elif timeout.loc[idx] and not concrete.loc[idx]:
            taxonomy = "unknown_failure"
            subtype = "unknown_failure:timeout_without_diagnostics"
            rule = "A_timeout_without_diagnostics"
        elif candidate_count.loc[idx] > 0 and feasible_count.loc[idx] == 0 and all_invalid.loc[idx] and concrete.loc[idx]:
            taxonomy = "known_failure"
            subtype = _subtype_from_flags(bool(collision.loc[idx]), bool(road.loc[idx]), bool(kinematic.loc[idx]))
            rule = "D_all_invalid_concrete_validator_flag"
        elif candidate_count.loc[idx] > 0 and feasible_count.loc[idx] == 0 and all_invalid.loc[idx]:
            taxonomy = "unknown_failure"
            subtype = "unknown_failure:all_invalid_unattributed"
            rule = "F_all_invalid_unattributed"
        elif failure.loc[idx] == 0:
            taxonomy = "no_failure"
            subtype = "no_failure:planner_success_or_no_failure"
            rule = "B_no_planner_failure"
        else:
            fallback = _fallback_known_from_reason(reason.loc[idx])
            if fallback:
                taxonomy = "known_failure"
                subtype = fallback
                rule = "legacy_reason_known_failure"
            else:
                taxonomy = "unknown_failure"
                subtype = "unknown_failure:missing_validator_output_or_incomplete_rollout"
                rule = "A_missing_diagnostics"
        failure_taxonomy.append(taxonomy)
        failure_subtype.append(subtype)
        rule_id.append(rule)
        secondary.append(int(subtype == "known_failure:kinematic_only"))

    out["failure_taxonomy"] = failure_taxonomy
    out["failure_subtype"] = failure_subtype
    out["taxonomy_rule_id"] = rule_id
    out["taxonomy_version"] = taxonomy_version
    out["taxonomy_secondary_sensitivity"] = secondary
    for col in ["collision_flag", "road_boundary_flag", "lane_buffer_flag", "kinematic_flag", "no_candidate_flag", "no_route_flag", "initial_invalid_flag", "parser_error_flag"]:
        if col not in out.columns:
            out[col] = 0
    if "candidate_any_feasible" not in out.columns:
        out["candidate_any_feasible"] = any_feasible.astype(int)
    if "candidate_all_invalid" not in out.columns:
        out["candidate_all_invalid"] = all_invalid.astype(int)
    if "candidate_failure_reasons" not in out.columns:
        out["candidate_failure_reasons"] = ""
    return out


def diagnostic_fields_from_candidate_rows(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_count = len(results)
    feasible_count = sum(1 for row in results if bool(row.get("feasible")))
    reasons = Counter(str(row.get("failure_reason", "")).lower() for row in results)
    collision = any(bool(row.get("collision_flag")) for row in results)
    lane = any(bool(row.get("lane_buffer_flag")) for row in results)
    kinematic = any(bool(row.get("kinematic_flag")) for row in results)
    reason_text = " ".join(reasons)
    return {
        "collision_flag": int(collision),
        "road_boundary_flag": int(lane),
        "lane_buffer_flag": int(lane),
        "kinematic_flag": int(kinematic),
        "no_candidate_flag": int(candidate_count == 0),
        "no_route_flag": int("no_route" in reason_text),
        "initial_invalid_flag": int("initial_invalid" in reason_text),
        "parser_error_flag": int("parser_error" in reason_text or "parse_error" in reason_text),
        "candidate_any_feasible": int(feasible_count > 0),
        "candidate_all_invalid": int(candidate_count > 0 and feasible_count == 0),
        "candidate_failure_reasons": ";".join(f"{k}:{v}" for k, v in sorted(reasons.items()) if k),
    }
