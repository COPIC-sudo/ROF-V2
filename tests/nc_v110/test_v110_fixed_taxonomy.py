from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.taxonomy import annotate_planner_failure_taxonomy


def _row(**kwargs):
    base = {
        "sample_id": "s0",
        "planner_success": 0,
        "planner_failure": 1,
        "candidate_count": 35,
        "feasible_candidate_count": 0,
        "candidate_all_invalid": 1,
        "candidate_any_feasible": 0,
        "planner_failure_reason": "unknown",
        "collision_flag": 0,
        "road_boundary_flag": 0,
        "lane_buffer_flag": 0,
        "kinematic_flag": 0,
        "parser_error_flag": 0,
        "no_route_flag": 0,
        "initial_invalid_flag": 0,
        "no_candidate_flag": 0,
    }
    base.update(kwargs)
    return base


def _annotated(**kwargs):
    return annotate_planner_failure_taxonomy(pd.DataFrame([_row(**kwargs)])).iloc[0]


def test_lane_or_road_boundary_plus_kinematic_is_known_failure() -> None:
    row = _annotated(road_boundary_flag=1, lane_buffer_flag=1, kinematic_flag=1)
    assert row["failure_taxonomy"] == "known_failure"
    assert row["failure_subtype"] == "known_failure:road_boundary_and_kinematic"
    assert row["taxonomy_rule_id"] == "D_all_invalid_concrete_validator_flag"


def test_collision_plus_kinematic_is_known_failure() -> None:
    row = _annotated(collision_flag=1, kinematic_flag=1)
    assert row["failure_taxonomy"] == "known_failure"
    assert row["failure_subtype"] == "known_failure:collision_and_kinematic"


def test_feasible_candidate_wins_over_invalid_candidates() -> None:
    row = _annotated(
        planner_success=1,
        planner_failure=0,
        feasible_candidate_count=2,
        candidate_any_feasible=1,
        collision_flag=1,
        kinematic_flag=1,
    )
    assert row["failure_taxonomy"] == "no_failure"
    assert row["failure_subtype"] == "no_failure:feasible_trajectory_exists"


def test_parser_error_and_no_route_remain_unknown() -> None:
    parser_row = _annotated(parser_error_flag=1, planner_failure_reason="parser_error")
    route_row = _annotated(no_route_flag=1, planner_failure_reason="no_route")
    assert parser_row["failure_taxonomy"] == "unknown_failure"
    assert parser_row["failure_subtype"] == "unknown_failure:parser_error"
    assert route_row["failure_taxonomy"] == "unknown_failure"
    assert route_row["failure_subtype"] == "unknown_failure:no_route"


def test_all_invalid_without_validator_flags_remains_unknown() -> None:
    row = _annotated()
    assert row["failure_taxonomy"] == "unknown_failure"
    assert row["failure_subtype"] == "unknown_failure:all_invalid_unattributed"
    assert row["taxonomy_rule_id"] == "F_all_invalid_unattributed"


def test_kinematic_only_is_known_secondary_sensitivity() -> None:
    row = _annotated(kinematic_flag=1)
    assert row["failure_taxonomy"] == "known_failure"
    assert row["failure_subtype"] == "known_failure:kinematic_only"
    assert int(row["taxonomy_secondary_sensitivity"]) == 1
