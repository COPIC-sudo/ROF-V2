#!/usr/bin/env python
"""Independent CommonRoad lattice-planner feasibility pilot.

The script reads exported CommonRoad JSON samples only. It does not read ROF
features, does not train a model, and does not import CommonRoad IO.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.external.taxonomy import (
    DEFAULT_TAXONOMY_VERSION,
    annotate_planner_failure_taxonomy,
    diagnostic_fields_from_candidate_rows,
)


MAX_YAW_RATE_RAD_S = 1.20
MAX_LATERAL_ACCEL_MPS2 = 4.50

LABEL_FIELDS = [
    "sample_id",
    "commonroad_scenario_id",
    "planner_family",
    "horizon_s",
    "lane_buffer_m",
    "planner_success",
    "planner_failure",
    "feasible_candidate_count",
    "candidate_count",
    "feasible_candidate_ratio",
    "best_candidate_cost",
    "best_action_family",
    "planner_failure_reason",
    "raw_planner_failure_reason",
    "failure_taxonomy",
    "failure_subtype",
    "taxonomy_rule_id",
    "taxonomy_version",
    "taxonomy_secondary_sensitivity",
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
    "first_failure_time_s_min",
    "initial_overlap_count",
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_mps",
    "agent_count",
    "old_feasibility_label_id",
    "old_feasibility_label_name",
]

CANDIDATE_FIELDS = [
    "sample_id",
    "commonroad_scenario_id",
    "candidate_id",
    "longitudinal_profile",
    "lateral_profile",
    "action_family",
    "action_cost",
    "feasible",
    "first_failure_time_s",
    "failure_reason",
    "collision_flag",
    "lane_buffer_flag",
    "progress_flag",
    "kinematic_flag",
    "blocking_agent_id",
    "initial_overlap_count",
    "progress_m",
    "max_abs_accel_mps2",
    "max_abs_yaw_rate_rad_s",
    "max_lateral_accel_mps2",
    "trajectory_json",
]

FAILURE_FIELDS = ["sample_id", "commonroad_scenario_id", "json_gz_path", "error", "traceback"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--feasibility-labels-csv")
    parser.add_argument("--out-name", default="commonroad_lattice_planner_feasibility_pilot1000")
    parser.add_argument("--planner-family", choices=["lattice_base", "lattice_extended"], default=None)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--dt-s", type=float)
    parser.add_argument("--safety-buffer-m", type=float, default=0.2)
    parser.add_argument("--ignore-initial-s", type=float, default=0.2)
    parser.add_argument("--use-lane-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lane-buffer-m", type=float, default=4.0)
    parser.add_argument("--min-progress-m", type=float, default=2.0)
    parser.add_argument("--max-comfort-accel", type=float, default=2.0)
    parser.add_argument("--max-comfort-decel", type=float, default=-3.0)
    parser.add_argument("--max-emergency-decel", type=float, default=-6.0)
    parser.add_argument("--write-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-trajectory-json", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _sample_path(row: Dict[str, Any], samples_dir: Path) -> Path:
    explicit = row.get("json_gz_path", "")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    return samples_dir / f"{row.get('sample_id', '')}.json.gz"


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return "" if value is None else str(value)
    if not np.isfinite(f):
        return ""
    return f"{f:.6f}"


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _axis_radius(axis: np.ndarray, length: float, width: float, heading: float) -> float:
    c, s = math.cos(float(heading)), math.sin(float(heading))
    ux = np.asarray([c, s], dtype=np.float64)
    uy = np.asarray([-s, c], dtype=np.float64)
    return 0.5 * float(length) * abs(float(np.dot(axis, ux))) + 0.5 * float(width) * abs(float(np.dot(axis, uy)))


def _obb_intersects(
    cx1: float,
    cy1: float,
    length1: float,
    width1: float,
    heading1: float,
    cx2: float,
    cy2: float,
    length2: float,
    width2: float,
    heading2: float,
) -> bool:
    c1, s1 = math.cos(float(heading1)), math.sin(float(heading1))
    c2, s2 = math.cos(float(heading2)), math.sin(float(heading2))
    axes = (
        np.asarray([c1, s1], dtype=np.float64),
        np.asarray([-s1, c1], dtype=np.float64),
        np.asarray([c2, s2], dtype=np.float64),
        np.asarray([-s2, c2], dtype=np.float64),
    )
    delta = np.asarray([float(cx2) - float(cx1), float(cy2) - float(cy1)], dtype=np.float64)
    for axis in axes:
        center_dist = abs(float(np.dot(delta, axis)))
        radius = _axis_radius(axis, length1, width1, heading1) + _axis_radius(axis, length2, width2, heading2)
        if center_dist > radius + 1e-9:
            return False
    return True


def _smoothstep(u: float) -> float:
    z = min(max(float(u), 0.0), 1.0)
    return z * z * (3.0 - 2.0 * z)


def _base_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    longitudinal = [
        {"name": "keep_speed", "accel": 0.0, "cost": 0.00, "emergency": False},
        {"name": "coast", "accel": -0.5, "cost": 0.08, "emergency": False},
        {"name": "mild_accel", "accel": 1.0, "cost": 0.16, "emergency": False},
        {"name": "mild_brake", "accel": -2.0, "cost": 0.20, "emergency": False},
        {"name": "hard_brake", "accel": -6.0, "cost": 0.75, "emergency": True},
    ]
    lateral = [
        {"name": "keep_lane", "offset": 0.0, "duration": 1.0, "cost": 0.00, "emergency": False},
        {"name": "left_offset_1m", "offset": 1.0, "duration": 1.5, "cost": 0.18, "emergency": False},
        {"name": "left_offset_2m", "offset": 2.0, "duration": 2.0, "cost": 0.34, "emergency": False},
        {"name": "right_offset_1m", "offset": -1.0, "duration": 1.5, "cost": 0.18, "emergency": False},
        {"name": "right_offset_2m", "offset": -2.0, "duration": 2.0, "cost": 0.34, "emergency": False},
        {"name": "brake_left_1m", "offset": 1.0, "duration": 1.0, "cost": 0.42, "emergency": True},
        {"name": "brake_right_1m", "offset": -1.0, "duration": 1.0, "cost": 0.42, "emergency": True},
    ]
    return longitudinal, lateral


def _extended_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    longitudinal = [
        {"name": "strong_accel", "accel": 2.0, "cost": 0.30, "emergency": False},
        {"name": "mild_accel", "accel": 1.0, "cost": 0.16, "emergency": False},
        {"name": "gentle_accel", "accel": 0.4, "cost": 0.08, "emergency": False},
        {"name": "keep_speed", "accel": 0.0, "cost": 0.00, "emergency": False},
        {"name": "coast", "accel": -0.5, "cost": 0.08, "emergency": False},
        {"name": "gentle_brake", "accel": -1.0, "cost": 0.13, "emergency": False},
        {"name": "mild_brake", "accel": -2.0, "cost": 0.20, "emergency": False},
        {"name": "firm_brake", "accel": -4.0, "cost": 0.48, "emergency": True},
        {"name": "hard_brake", "accel": -6.0, "cost": 0.75, "emergency": True},
        {"name": "max_brake", "accel": -8.0, "cost": 0.92, "emergency": True},
    ]
    lateral = [
        {"name": "keep_lane", "offset": 0.0, "duration": 1.0, "cost": 0.00, "emergency": False},
        {"name": "left_offset_0p5m", "offset": 0.5, "duration": 1.2, "cost": 0.10, "emergency": False},
        {"name": "right_offset_0p5m", "offset": -0.5, "duration": 1.2, "cost": 0.10, "emergency": False},
        {"name": "left_offset_1m", "offset": 1.0, "duration": 1.5, "cost": 0.18, "emergency": False},
        {"name": "right_offset_1m", "offset": -1.0, "duration": 1.5, "cost": 0.18, "emergency": False},
        {"name": "left_offset_1p5m", "offset": 1.5, "duration": 1.8, "cost": 0.26, "emergency": False},
        {"name": "right_offset_1p5m", "offset": -1.5, "duration": 1.8, "cost": 0.26, "emergency": False},
        {"name": "left_offset_2m", "offset": 2.0, "duration": 2.0, "cost": 0.34, "emergency": False},
        {"name": "right_offset_2m", "offset": -2.0, "duration": 2.0, "cost": 0.34, "emergency": False},
        {"name": "left_offset_2p5m", "offset": 2.5, "duration": 2.4, "cost": 0.48, "emergency": True},
        {"name": "right_offset_2p5m", "offset": -2.5, "duration": 2.4, "cost": 0.48, "emergency": True},
        {"name": "fast_left_1m", "offset": 1.0, "duration": 0.8, "cost": 0.44, "emergency": True},
        {"name": "fast_right_1m", "offset": -1.0, "duration": 0.8, "cost": 0.44, "emergency": True},
    ]
    return longitudinal, lateral


def _make_candidates(planner_family: str = "lattice_base") -> List[Dict[str, Any]]:
    if planner_family == "lattice_extended":
        longitudinal, lateral = _extended_candidates()
    else:
        longitudinal, lateral = _base_candidates()
    candidates: List[Dict[str, Any]] = []
    for lon in longitudinal:
        for lat in lateral:
            emergency = bool(lon["emergency"] or lat["emergency"])
            candidates.append(
                {
                    "planner_family": planner_family,
                    "longitudinal_profile": lon["name"],
                    "lateral_profile": lat["name"],
                    "accel": lon["accel"],
                    "lat_offset": lat["offset"],
                    "lat_duration": lat["duration"],
                    "action_family": "emergency" if emergency else "comfort",
                    "action_cost": float(lon["cost"]) + float(lat["cost"]),
                }
            )
    return candidates


def _sample_dt(sample: Dict[str, Any], cfg: dict, args: argparse.Namespace) -> float:
    if args.dt_s is not None:
        return float(args.dt_s)
    if sample.get("dt") is not None:
        return float(sample.get("dt"))
    return float((cfg.get("labels") or {}).get("dt_s", 0.1))


def _build_lane_buffer(sample: Dict[str, Any], lane_buffer_m: float):
    geoms = []
    for line in sample.get("map_lane_centerlines", []) or []:
        arr = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] < 2:
            continue
        try:
            geoms.append(LineString(arr).buffer(float(lane_buffer_m), cap_style=2, join_style=2))
        except Exception:
            continue
    if not geoms:
        return None
    try:
        return unary_union(geoms)
    except Exception:
        return geoms[0]


def _precompute_agents(sample: Dict[str, Any], times: np.ndarray) -> Dict[str, Any]:
    agent_count = int(sample["agent_count"])
    current_xy = np.asarray(sample["current_xy"], dtype=np.float64)
    current_vel = np.asarray(sample["current_vel_xy"], dtype=np.float64)
    current_heading = np.asarray(sample["current_heading"], dtype=np.float64)
    current_size = np.asarray(sample["current_size_lw"], dtype=np.float64)
    future_xy = np.asarray(sample.get("future_xy", []), dtype=np.float64)
    future_heading = np.asarray(sample.get("future_heading", []), dtype=np.float64)
    future_valid = np.asarray(sample.get("future_valid", []), dtype=bool)
    xy = np.zeros((len(times), agent_count, 2), dtype=np.float64)
    heading = np.zeros((len(times), agent_count), dtype=np.float64)
    valid = np.ones((len(times), agent_count), dtype=bool)
    for k, t in enumerate(times):
        if future_valid.ndim == 2 and k < future_valid.shape[1]:
            valid[k] = future_valid[:, k]
            if future_xy.ndim == 3:
                xy[k] = future_xy[:, k, :]
            if future_heading.ndim == 2:
                heading[k] = future_heading[:, k]
            missing = ~valid[k]
            if bool(missing.any()):
                xy[k, missing] = current_xy[missing] + current_vel[missing] * float(t)
                heading[k, missing] = current_heading[missing]
                valid[k, missing] = True
        else:
            xy[k] = current_xy + current_vel * float(t)
            heading[k] = current_heading
            valid[k] = True
    return {
        "ego_idx": int(sample.get("ego_index", 0)),
        "agent_count": agent_count,
        "agent_ids": list(sample.get("agent_ids", [])),
        "current_size": current_size,
        "xy": xy,
        "heading": heading,
        "valid": valid,
    }


def _rollout_candidate(sample: Dict[str, Any], candidate: Dict[str, Any], times: np.ndarray) -> Dict[str, np.ndarray]:
    ego_idx = int(sample.get("ego_index", 0))
    current_xy = np.asarray(sample["current_xy"], dtype=np.float64)
    current_vel = np.asarray(sample["current_vel_xy"], dtype=np.float64)
    current_heading = np.asarray(sample["current_heading"], dtype=np.float64)
    start = current_xy[ego_idx]
    base_heading = float(current_heading[ego_idx])
    heading_vec = np.asarray([math.cos(base_heading), math.sin(base_heading)], dtype=np.float64)
    normal_vec = np.asarray([-math.sin(base_heading), math.cos(base_heading)], dtype=np.float64)
    v0 = max(0.0, float(np.linalg.norm(current_vel[ego_idx])))
    accel_cmd = float(candidate["accel"])
    lat_target = float(candidate["lat_offset"])
    lat_duration = max(float(candidate["lat_duration"]), 1e-3)

    s = np.zeros(len(times), dtype=np.float64)
    lateral = np.zeros(len(times), dtype=np.float64)
    speed_long = np.zeros(len(times), dtype=np.float64)
    speed_long[0] = v0
    for k in range(1, len(times)):
        dt = float(times[k] - times[k - 1])
        speed_long[k] = max(0.0, speed_long[k - 1] + accel_cmd * dt)
        s[k] = s[k - 1] + 0.5 * (speed_long[k - 1] + speed_long[k]) * dt
    for k, t in enumerate(times):
        lateral[k] = lat_target * _smoothstep(float(t) / lat_duration)
    xy = start[None, :] + s[:, None] * heading_vec[None, :] + lateral[:, None] * normal_vec[None, :]
    heading = np.full(len(times), base_heading, dtype=np.float64)
    for k in range(1, len(times)):
        delta = xy[k] - xy[k - 1]
        if float(np.linalg.norm(delta)) > 1e-8:
            heading[k] = math.atan2(float(delta[1]), float(delta[0]))
        else:
            heading[k] = heading[k - 1]
    if len(times) > 1:
        heading[0] = heading[1]
    speed = np.zeros(len(times), dtype=np.float64)
    for k in range(1, len(times)):
        dt = max(float(times[k] - times[k - 1]), 1e-6)
        speed[k] = float(np.linalg.norm(xy[k] - xy[k - 1]) / dt)
    if len(times) > 1:
        speed[0] = speed[1]
    accel = np.zeros(len(times), dtype=np.float64)
    yaw_rate = np.zeros(len(times), dtype=np.float64)
    lateral_accel = np.zeros(len(times), dtype=np.float64)
    for k in range(1, len(times)):
        dt = max(float(times[k] - times[k - 1]), 1e-6)
        accel[k] = (speed[k] - speed[k - 1]) / dt
        yaw_rate[k] = _wrap_to_pi(heading[k] - heading[k - 1]) / dt
        lateral_accel[k] = speed[k] * yaw_rate[k]
    progress = float(np.dot(xy[-1] - xy[0], heading_vec))
    return {
        "xy": xy,
        "heading": heading,
        "speed": speed,
        "accel": accel,
        "yaw_rate": yaw_rate,
        "lateral_accel": lateral_accel,
        "progress": np.asarray([progress], dtype=np.float64),
    }


def _kinematic_violation(candidate: Dict[str, Any], rollout: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[bool, float | None]:
    accel_cmd = float(candidate["accel"])
    if candidate["action_family"] == "comfort":
        if accel_cmd > float(args.max_comfort_accel) + 1e-9 or accel_cmd < float(args.max_comfort_decel) - 1e-9:
            return True, 0.0
    else:
        if accel_cmd < float(args.max_emergency_decel) - 1e-9 or accel_cmd > float(args.max_comfort_accel) + 1e-9:
            return True, 0.0
    yaw = np.abs(rollout["yaw_rate"])
    lat_accel = np.abs(rollout["lateral_accel"])
    bad = np.where((yaw > MAX_YAW_RATE_RAD_S) | (lat_accel > MAX_LATERAL_ACCEL_MPS2))[0]
    if len(bad):
        return True, None
    return False, None


def _evaluate_candidate(
    sample: Dict[str, Any],
    candidate: Dict[str, Any],
    candidate_id: int,
    times: np.ndarray,
    lane_buffer_geom: Any,
    agents: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rollout = _rollout_candidate(sample, candidate, times)
    xy = rollout["xy"]
    heading = rollout["heading"]
    ego_idx = int(agents["ego_idx"])
    current_size = agents["current_size"]
    ego_l = float(current_size[ego_idx, 0]) + 2.0 * float(args.safety_buffer_m)
    ego_w = float(current_size[ego_idx, 1]) + 2.0 * float(args.safety_buffer_m)
    ego_radius = 0.5 * math.hypot(ego_l, ego_w)
    all_xy = agents["xy"]
    all_heading = agents["heading"]
    all_valid = agents["valid"]
    agent_ids = agents["agent_ids"]
    agent_count = int(agents["agent_count"])

    collision_flag = False
    lane_flag = False
    progress_flag = False
    blocking_agent_id = ""
    first_collision_t: float | None = None
    first_lane_t: float | None = None
    initial_overlap_agents: set[str] = set()
    for k, t in enumerate(times):
        for j in range(agent_count):
            if j == ego_idx or not bool(all_valid[k, j]):
                continue
            other_xy = all_xy[k, j]
            other_l = float(current_size[j, 0])
            other_w = float(current_size[j, 1])
            other_radius = 0.5 * math.hypot(other_l, other_w)
            if float(np.linalg.norm(other_xy - xy[k])) > ego_radius + other_radius + 1e-6:
                continue
            if _obb_intersects(
                xy[k, 0],
                xy[k, 1],
                ego_l,
                ego_w,
                heading[k],
                other_xy[0],
                other_xy[1],
                other_l,
                other_w,
                float(all_heading[k, j]),
            ):
                agent_id = str(agent_ids[j] if j < len(agent_ids) else j)
                if float(t) < float(args.ignore_initial_s):
                    initial_overlap_agents.add(agent_id)
                elif first_collision_t is None:
                    collision_flag = True
                    first_collision_t = float(t)
                    blocking_agent_id = agent_id
                break
        if args.use_lane_buffer and lane_buffer_geom is not None:
            if not lane_buffer_geom.covers(Point(float(xy[k, 0]), float(xy[k, 1]))):
                if first_lane_t is None:
                    lane_flag = True
                    first_lane_t = float(t)

    progress_m = float(rollout["progress"][0])
    if progress_m < float(args.min_progress_m) and candidate["longitudinal_profile"] != "hard_brake":
        progress_flag = True
    kin_flag, kin_t = _kinematic_violation(candidate, rollout, args)

    first_times = [t for t in [first_collision_t, first_lane_t, kin_t, float(times[-1]) if progress_flag else None] if t is not None]
    first_failure_t = min(first_times) if first_times else None
    feasible = not (collision_flag or lane_flag or progress_flag or kin_flag)
    if feasible:
        reason = "no_failure"
    elif collision_flag:
        reason = "collision"
    elif lane_flag:
        reason = "lane_buffer"
    elif progress_flag:
        reason = "progress"
    elif kin_flag:
        reason = "kinematic"
    else:
        reason = "unknown"

    traj = []
    if bool(args.store_trajectory_json):
        for k, t in enumerate(times):
            traj.append(
                {
                    "time_s": round(float(t), 3),
                    "x": round(float(xy[k, 0]), 4),
                    "y": round(float(xy[k, 1]), 4),
                    "heading": round(float(heading[k]), 5),
                    "speed": round(float(rollout["speed"][k]), 4),
                    "acceleration": round(float(rollout["accel"][k]), 4),
                    "action_family": candidate["action_family"],
                    "action_cost": round(float(candidate["action_cost"]), 4),
                }
            )
    return {
        "candidate_id": candidate_id,
        "longitudinal_profile": candidate["longitudinal_profile"],
        "lateral_profile": candidate["lateral_profile"],
        "action_family": candidate["action_family"],
        "action_cost": float(candidate["action_cost"]),
        "feasible": feasible,
        "first_failure_time_s": first_failure_t,
        "failure_reason": reason,
        "collision_flag": collision_flag,
        "lane_buffer_flag": lane_flag,
        "progress_flag": progress_flag,
        "kinematic_flag": kin_flag,
        "blocking_agent_id": blocking_agent_id,
        "initial_overlap_count": len(initial_overlap_agents),
        "progress_m": progress_m,
        "max_abs_accel_mps2": float(np.nanmax(np.abs(rollout["accel"]))),
        "max_abs_yaw_rate_rad_s": float(np.nanmax(np.abs(rollout["yaw_rate"]))),
        "max_lateral_accel_mps2": float(np.nanmax(np.abs(rollout["lateral_accel"]))),
        "trajectory_json": json.dumps(traj, separators=(",", ":")),
    }


def _planner_failure_reason(results: List[Dict[str, Any]]) -> str:
    if any(bool(r["feasible"]) for r in results):
        return "no_failure"
    if not results:
        return "no_candidate_generated"
    flags = {
        "collision": any(bool(r["collision_flag"]) for r in results),
        "lane": any(bool(r["lane_buffer_flag"]) for r in results),
        "progress": any(bool(r["progress_flag"]) for r in results),
        "kinematic": any(bool(r["kinematic_flag"]) for r in results),
    }
    if flags["collision"] and flags["lane"]:
        return "collision_and_lane"
    if flags["collision"] and not any(v for k, v in flags.items() if k != "collision"):
        return "collision_only"
    if flags["lane"] and not any(v for k, v in flags.items() if k != "lane"):
        return "lane_buffer_only"
    if flags["progress"] and not any(v for k, v in flags.items() if k != "progress"):
        return "progress_only"
    if flags["kinematic"] and not any(v for k, v in flags.items() if k != "kinematic"):
        return "kinematic_only"
    return "unknown"


def _sample_old_label(old_by_id: Dict[str, Dict[str, Any]], sample_id: str, field: str) -> str:
    return str(old_by_id.get(sample_id, {}).get(field, ""))


def _taxonomy_label_fields(raw_reason: str, base_row: Dict[str, Any], results: List[Dict[str, Any]], cfg: dict) -> Dict[str, Any]:
    version = str((cfg.get("taxonomy") or {}).get("version") or DEFAULT_TAXONOMY_VERSION)
    diagnostics = diagnostic_fields_from_candidate_rows(results)
    row = {**base_row, **diagnostics, "raw_planner_failure_reason": raw_reason, "planner_failure_reason": raw_reason}
    annotated = annotate_planner_failure_taxonomy(pd.DataFrame([row]), taxonomy_version=version).iloc[0].to_dict()
    subtype = str(annotated.get("failure_subtype", ""))
    if subtype.startswith("known_failure:"):
        planner_reason = subtype.split(":", 1)[1]
    elif subtype.startswith("no_failure:"):
        planner_reason = "no_failure"
    elif subtype.startswith("unknown_failure:"):
        planner_reason = subtype.split(":", 1)[1]
    else:
        planner_reason = raw_reason
    out = {
        **diagnostics,
        "raw_planner_failure_reason": raw_reason,
        "planner_failure_reason": planner_reason,
        "failure_taxonomy": annotated.get("failure_taxonomy", ""),
        "failure_subtype": annotated.get("failure_subtype", ""),
        "taxonomy_rule_id": annotated.get("taxonomy_rule_id", ""),
        "taxonomy_version": annotated.get("taxonomy_version", version),
        "taxonomy_secondary_sensitivity": annotated.get("taxonomy_secondary_sensitivity", 0),
    }
    return out


def _evaluate_sample(
    sample: Dict[str, Any],
    manifest: Dict[str, Any],
    old_by_id: Dict[str, Dict[str, Any]],
    cfg: dict,
    args: argparse.Namespace,
    candidates: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dt = _sample_dt(sample, cfg, args)
    n_steps = max(1, int(round(float(args.horizon_s) / max(dt, 1e-6))))
    times = np.arange(n_steps + 1, dtype=np.float64) * dt
    lane_buffer_geom = _build_lane_buffer(sample, args.lane_buffer_m) if args.use_lane_buffer else None
    agents = _precompute_agents(sample, times)
    results = [
        _evaluate_candidate(sample, candidate, i, times, lane_buffer_geom, agents, args)
        for i, candidate in enumerate(candidates)
    ]
    feasible = [r for r in results if bool(r["feasible"])]
    planner_success = int(len(feasible) > 0)
    first_failures = [float(r["first_failure_time_s"]) for r in results if r.get("first_failure_time_s") not in ("", None)]
    best = min(feasible, key=lambda r: float(r["action_cost"])) if feasible else None
    sid = str(sample.get("sample_id", manifest.get("sample_id", "")))
    raw_reason = _planner_failure_reason(results)
    label_row = {
        "sample_id": sid,
        "commonroad_scenario_id": sample.get("commonroad_scenario_id", manifest.get("commonroad_scenario_id", "")),
        "planner_family": getattr(args, "planner_family", "") or (cfg.get("planner") or {}).get("family", "lattice_base"),
        "horizon_s": float(args.horizon_s),
        "lane_buffer_m": float(args.lane_buffer_m),
        "planner_success": planner_success,
        "planner_failure": int(not planner_success),
        "feasible_candidate_count": len(feasible),
        "candidate_count": len(results),
        "feasible_candidate_ratio": len(feasible) / max(len(results), 1),
        "best_candidate_cost": float(best["action_cost"]) if best else "",
        "best_action_family": f"{best['longitudinal_profile']}+{best['lateral_profile']}" if best else "",
        "planner_failure_reason": raw_reason,
        "first_failure_time_s_min": min(first_failures) if first_failures else "",
        "initial_overlap_count": sum(int(r.get("initial_overlap_count", 0) or 0) for r in results),
        "current_min_distance_m": manifest.get("current_min_distance_m", ""),
        "current_ttc_s": manifest.get("current_ttc_s", ""),
        "ego_speed_mps": manifest.get("ego_speed_mps", ""),
        "agent_count": sample.get("agent_count", manifest.get("agent_count", "")),
        "old_feasibility_label_id": _sample_old_label(old_by_id, sid, "feasibility_label_id"),
        "old_feasibility_label_name": _sample_old_label(old_by_id, sid, "feasibility_label_name"),
    }
    label_row.update(_taxonomy_label_fields(raw_reason, label_row, results, cfg))
    candidate_rows: List[Dict[str, Any]] = []
    if bool(args.write_candidates):
        for result in results:
            row = {
                "sample_id": label_row["sample_id"],
                "commonroad_scenario_id": label_row["commonroad_scenario_id"],
                **result,
            }
            for key in [
                "action_cost",
                "first_failure_time_s",
                "progress_m",
                "max_abs_accel_mps2",
                "max_abs_yaw_rate_rad_s",
                "max_lateral_accel_mps2",
            ]:
                row[key] = _fmt(row.get(key))
            row["feasible"] = str(bool(row["feasible"]))
            row["collision_flag"] = str(bool(row["collision_flag"]))
            row["lane_buffer_flag"] = str(bool(row["lane_buffer_flag"]))
            row["progress_flag"] = str(bool(row["progress_flag"]))
            row["kinematic_flag"] = str(bool(row["kinematic_flag"]))
            candidate_rows.append(row)
    return label_row, candidate_rows


def _select_samples(
    manifest_rows: List[Dict[str, Any]],
    old_rows: List[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(manifest_rows):
        return manifest_rows
    rng = random.Random(seed)
    manifest_by_id = {str(r.get("sample_id", "")): r for r in manifest_rows}
    if not old_rows:
        rows = list(manifest_rows)
        rng.shuffle(rows)
        return rows[:sample_size]

    old_available = [r for r in old_rows if str(r.get("sample_id", "")) in manifest_by_id]
    severe = [r for r in old_available if _safe_int(r.get("feasibility_label_id")) >= 2]
    reduced = [r for r in old_available if _safe_int(r.get("feasibility_label_id")) == 1]
    high = [r for r in old_available if _safe_int(r.get("feasibility_label_id")) == 0]
    rng.shuffle(severe)
    rng.shuffle(reduced)
    rng.shuffle(high)
    selected_ids: List[str] = []
    selected_ids.extend(str(r["sample_id"]) for r in severe[: min(200, len(severe), sample_size)])
    remaining = sample_size - len(selected_ids)
    selected_ids.extend(str(r["sample_id"]) for r in reduced[: max(0, min(300, remaining, len(reduced)))])
    remaining = sample_size - len(selected_ids)
    selected_ids.extend(str(r["sample_id"]) for r in high[: max(0, remaining)])

    if len(selected_ids) < sample_size:
        seen = set(selected_ids)
        leftovers = [r for r in manifest_rows if str(r.get("sample_id", "")) not in seen]
        rng.shuffle(leftovers)
        selected_ids.extend(str(r.get("sample_id", "")) for r in leftovers[: sample_size - len(selected_ids)])
    return [manifest_by_id[sid] for sid in selected_ids if sid in manifest_by_id]


def _describe(values: Iterable[Any]) -> Dict[str, str]:
    vals = sorted(float(v) for v in values if np.isfinite(_safe_float(v)))
    if not vals:
        return {k: "" for k in ["count", "mean", "std", "min", "p25", "p50", "p75", "max"]}

    def q(frac: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = frac * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        w = pos - lo
        return vals[lo] * (1.0 - w) + vals[hi] * w

    return {
        "count": str(len(vals)),
        "mean": f"{statistics.fmean(vals):.6f}",
        "std": f"{statistics.pstdev(vals):.6f}" if len(vals) > 1 else "0.000000",
        "min": f"{vals[0]:.6f}",
        "p25": f"{q(0.25):.6f}",
        "p50": f"{q(0.50):.6f}",
        "p75": f"{q(0.75):.6f}",
        "max": f"{vals[-1]:.6f}",
    }


def _distance_bucket(value: Any) -> str:
    v = _safe_float(value)
    if not np.isfinite(v):
        return "missing"
    if v < 3.0:
        return "<3m"
    if v < 10.0:
        return "3-10m"
    return ">=10m"


def _ttc_bucket(value: Any) -> str:
    v = _safe_float(value)
    if not np.isfinite(v):
        return "missing"
    if v < 0.0:
        return "ttc<0"
    if v < 1.0:
        return "0-1s"
    if v < 3.0:
        return "1-3s"
    return ">=3s"


def _spearman(a: Sequence[Any], b: Sequence[Any]) -> str:
    s1 = pd.to_numeric(pd.Series(list(a)), errors="coerce")
    s2 = pd.to_numeric(pd.Series(list(b)), errors="coerce")
    valid = s1.notna() & s2.notna()
    if int(valid.sum()) < 3:
        return ""
    corr = s1[valid].corr(s2[valid], method="spearman")
    return _fmt(corr)


def _summary_rows(labels: List[Dict[str, Any]], failures: List[Dict[str, Any]], runtime_s: float) -> List[Dict[str, Any]]:
    fields = ["section", "metric", "value", "count", "mean", "std", "min", "p25", "p50", "p75", "max"]
    _ = fields
    total = len(labels) + len(failures)
    planner_failures = sum(_safe_int(r.get("planner_failure")) for r in labels)
    rows: List[Dict[str, Any]] = [
        {"section": "counts", "metric": "total_samples", "value": total},
        {"section": "counts", "metric": "sample_error_count", "value": len(failures)},
        {"section": "counts", "metric": "planner_success_count", "value": sum(_safe_int(r.get("planner_success")) for r in labels)},
        {"section": "counts", "metric": "planner_failure_count", "value": planner_failures},
        {"section": "rates", "metric": "planner_failure_rate", "value": _fmt(planner_failures / max(len(labels), 1))},
        {"section": "runtime", "metric": "runtime_total_s", "value": _fmt(runtime_s)},
        {"section": "runtime", "metric": "runtime_per_sample_s", "value": _fmt(runtime_s / max(total, 1))},
    ]
    for metric in ["feasible_candidate_ratio", "feasible_candidate_count"]:
        desc = {"section": "describe", "metric": metric, "value": ""}
        desc.update(_describe(r.get(metric) for r in labels))
        rows.append(desc)
    for reason, count in Counter(str(r.get("planner_failure_reason", "")) for r in labels).most_common():
        rows.append({"section": "failure_reason_distribution", "metric": reason, "value": count})

    old_ids = sorted({str(r.get("old_feasibility_label_id", "")) for r in labels if str(r.get("old_feasibility_label_id", "")) != ""})
    for old_id in old_ids:
        subset = [r for r in labels if str(r.get("old_feasibility_label_id", "")) == old_id]
        for failure_value in [0, 1]:
            count = sum(_safe_int(r.get("planner_failure")) == failure_value for r in subset)
            rows.append({"section": "old_label_vs_planner_failure", "metric": f"old_label_{old_id}_planner_failure_{failure_value}", "value": count})

    rows.append(
        {
            "section": "spearman",
            "metric": "old_feasibility_label_id_vs_planner_failure",
            "value": _spearman([r.get("old_feasibility_label_id") for r in labels], [r.get("planner_failure") for r in labels]),
        }
    )
    rows.append(
        {
            "section": "spearman",
            "metric": "planner_failure_vs_current_min_distance_m",
            "value": _spearman([r.get("planner_failure") for r in labels], [r.get("current_min_distance_m") for r in labels]),
        }
    )
    rows.append(
        {
            "section": "spearman",
            "metric": "planner_failure_vs_current_ttc_s",
            "value": _spearman([r.get("planner_failure") for r in labels], [r.get("current_ttc_s") for r in labels]),
        }
    )

    for bucket_fn, section, source in [
        (_distance_bucket, "planner_failure_rate_by_distance_bucket", "current_min_distance_m"),
        (_ttc_bucket, "planner_failure_rate_by_ttc_bucket", "current_ttc_s"),
    ]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in labels:
            buckets[bucket_fn(row.get(source))].append(row)
        for bucket, subset in sorted(buckets.items()):
            n = len(subset)
            f = sum(_safe_int(r.get("planner_failure")) for r in subset)
            rows.append(
                {
                    "section": section,
                    "metric": bucket,
                    "value": _fmt(f / max(n, 1)),
                    "count": n,
                }
            )
    return rows


def _output_paths(work_dir: Path, out_name: str) -> Tuple[Path, Path, Path, Path, Path]:
    stem = Path(out_name).stem
    tag = "pilot1000" if "pilot1000" in stem else stem
    out_dir = work_dir / "results" / "commonroad_planner_feasibility" / tag
    return (
        out_dir,
        out_dir / f"commonroad_lattice_planner_labels_{tag}.csv",
        out_dir / f"commonroad_lattice_planner_candidates_{tag}.csv",
        out_dir / f"commonroad_lattice_planner_summary_{tag}.csv",
        out_dir / f"commonroad_lattice_planner_failures_{tag}.csv",
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    planner_family = args.planner_family or str((cfg.get("planner") or {}).get("family") or "lattice_base")
    args.planner_family = planner_family
    work_dir = Path(cfg["project"]["work_dir"])
    samples_dir = Path(args.samples_dir)
    manifest_rows = [r for r in _read_csv(Path(args.manifest_csv)) if r.get("export_status", "ok") == "ok"]
    old_rows = _read_csv(Path(args.feasibility_labels_csv)) if args.feasibility_labels_csv else []
    old_by_id = {str(r.get("sample_id", "")): r for r in old_rows}
    selected = _select_samples(manifest_rows, old_rows, int(args.sample_size), int(args.seed))
    candidates = _make_candidates(planner_family)
    out_dir, label_path, candidate_path, summary_path, failure_path = _output_paths(work_dir, args.out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, manifest in enumerate(selected, start=1):
        sample_path = _sample_path(manifest, samples_dir)
        try:
            sample = _read_json_gz(sample_path)
            label_row, rows = _evaluate_sample(sample, manifest, old_by_id, cfg, args, candidates)
            label_rows.append(label_row)
            candidate_rows.extend(rows)
        except Exception as exc:
            failures.append(
                {
                    "sample_id": manifest.get("sample_id", ""),
                    "commonroad_scenario_id": manifest.get("commonroad_scenario_id", ""),
                    "json_gz_path": str(sample_path),
                    "error": "".join(traceback.format_exception_only(type(exc), exc)).strip()[:800],
                    "traceback": traceback.format_exc(),
                }
            )
        if idx == 1 or idx % 50 == 0 or idx == len(selected):
            print(f"[progress] processed={idx}/{len(selected)} success={len(label_rows)} failed={len(failures)}")

    runtime_s = time.perf_counter() - t0
    _write_csv(label_path, label_rows, LABEL_FIELDS)
    if bool(args.write_candidates):
        _write_csv(candidate_path, candidate_rows, CANDIDATE_FIELDS)
    _write_csv(summary_path, _summary_rows(label_rows, failures, runtime_s), ["section", "metric", "value", "count", "mean", "std", "min", "p25", "p50", "p75", "max"])
    _write_csv(failure_path, failures, FAILURE_FIELDS)
    print(f"[done] out_dir={out_dir}")
    print(f"[done] labels={label_path}")
    if bool(args.write_candidates):
        print(f"[done] candidates={candidate_path}")
    else:
        print("[done] candidates=not_written (--no-write-candidates)")
    print(f"[done] summary={summary_path}")
    print(f"[done] failures={failure_path}")
    print(f"[done] total={len(selected)} planner_success={sum(_safe_int(r.get('planner_success')) for r in label_rows)} planner_failure={sum(_safe_int(r.get('planner_failure')) for r in label_rows)} sample_errors={len(failures)} runtime_s={runtime_s:.3f}")


if __name__ == "__main__":
    main()
