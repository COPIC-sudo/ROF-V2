from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


STRICT_NON_ACTION_CURRENT_CV = [
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_mps",
    "ego_speed_kph",
    "agent_count",
    "nearest_agent_rel_speed_mps",
    "nearest_agent_closing_speed_mps",
    "ttc_closing_speed_mps",
    "nearby_agent_count_10m",
    "nearby_agent_count_20m",
    "cv_rcr",
    "cv_rfr_drv",
    "cv_c_time",
    "cv_gtoa_norm_union",
    "cv_oce_norm",
    "cv_c_density",
    "cv_max_overlap_count",
    "current_collision",
    "max_overlap_count",
    "mean_overlap_count_nonzero",
    "overlap_count_entropy_norm",
]

ACTION_DERIVED_TOKENS = [
    "asr",
    "action",
    "survival",
    "ttad",
    "collapse",
    "comfort",
    "emergency",
    "candidate",
    "min_safe_action_cost",
    "redi_actionability",
]

RECORDED_FUTURE_TOKENS = [
    "future_",
    "oracle_future",
    "recorded_future",
]

LABEL_TOKENS = [
    "label",
    "horizon_h",
    "buffer_",
    "lane_buffer",
]


def lineage_flags(feature: str) -> dict[str, bool]:
    name = feature.lower()
    uses_action_library = any(tok in name for tok in ["action", "asr", "survival", "comfort", "emergency", "candidate", "ttad", "collapse"])
    uses_candidate_survival = any(tok in name for tok in ["asr", "survival", "ttad", "collapse", "min_safe_action_cost"])
    uses_label_horizon = any(tok in name for tok in ["horizon_h", "_h2", "_h3", "_h4", "label_horizon"])
    uses_label_lane_buffer = any(tok in name for tok in ["buffer_", "_b2", "_b3", "_b4", "lane_buffer"])
    reads_recorded_future = any(tok in name for tok in RECORDED_FUTURE_TOKENS)
    reads_label = name in {"label_id", "label_name", "actionability_label_id", "actionability_label_name", "y", "y_true"} or name.endswith("_label")
    uses_endpoint_intermediate = any(
        tok in name
        for tok in [
            "comfort_feasible",
            "emergency_feasible",
            "feasible_count",
            "feasible_ratio",
            "min_required_action_cost",
            "time_to_no_comfort",
            "time_to_no_emergency",
            "endpoint",
            "planner_failure",
            "known_failure",
            "unknown_failure",
            "no_failure",
        ]
    )
    return {
        "uses_action_library": uses_action_library,
        "uses_candidate_survival": uses_candidate_survival,
        "uses_label_horizon": uses_label_horizon,
        "uses_label_lane_buffer": uses_label_lane_buffer,
        "reads_recorded_future": reads_recorded_future,
        "reads_label": reads_label,
        "uses_endpoint_intermediate": uses_endpoint_intermediate,
    }


def feature_lineage_rows(features: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in sorted(dict.fromkeys(str(f) for f in features)):
        flags = lineage_flags(feature)
        is_strict = feature in STRICT_NON_ACTION_CURRENT_CV and not any(flags.values())
        rows.append(
            {
                "feature_name": feature,
                "feature": feature,
                "feature_set": "strict_non_action_current_cv" if is_strict else "excluded_or_diagnostic",
                "allowed_in_strict_non_action": bool(is_strict),
                "allowed_in_strict_non_action_current_cv": bool(is_strict),
                **flags,
            }
        )
    return rows


def strict_non_action_current_cv_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in STRICT_NON_ACTION_CURRENT_CV:
        if col in df.columns:
            flags = lineage_flags(col)
            if not any(flags.values()):
                cols.append(col)
    return cols
