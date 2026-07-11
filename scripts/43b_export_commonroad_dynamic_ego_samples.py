#!/usr/bin/env python
"""Export CommonRoad dynamic obstacles as ego-centric standard samples.

This exporter runs in the commonroad_io environment and does not import the
ROF/rtbev pipeline. It creates agent-centric CommonRoad samples for independent
candidate-action feasibility labeling.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import statistics
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml
from commonroad.common.file_reader import CommonRoadFileReader
from shapely.geometry import Polygon


MANIFEST_FIELDS = [
    "sample_id",
    "commonroad_scenario_id",
    "xml_path",
    "ego_obstacle_id",
    "current_time_step",
    "agent_count",
    "dynamic_agent_count",
    "static_agent_count",
    "lanelet_count",
    "ego_speed_mps",
    "current_min_distance_m",
    "current_ttc_s",
    "bucket",
    "export_status",
    "export_error",
    "json_gz_path",
]

FAILURE_FIELDS = [
    "scenario_id",
    "xml_path",
    "ego_obstacle_id",
    "current_time_step",
    "export_error",
    "traceback",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-csv", required=True)
    parser.add_argument("--out-name", default="commonroad_dynamic_ego_samples_pilot1000")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--time-stride", type=int, default=10)
    parser.add_argument("--horizon-steps", type=int, default=30)
    parser.add_argument("--min-neighbor-agents", type=int, default=3)
    parser.add_argument("--min-ego-speed-mps", type=float, default=0.5)
    parser.add_argument("--include-static-obstacles", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


def _load_work_dir(config_path: str) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = _expand(yaml.safe_load(f) or {})
    work_dir = (cfg.get("project") or {}).get("work_dir") or os.environ.get("ROF_WORK_DIR")
    if not work_dir:
        raise ValueError("project.work_dir missing in config and ROF_WORK_DIR is not set")
    return Path(str(work_dir))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _write_json_gz(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, default=_json_default, separators=(",", ":"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _as_xy(position: Any) -> np.ndarray:
    arr = np.asarray(position, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"position does not contain x/y: {position!r}")
    return arr[:2].astype(np.float64)


def _state_heading(state: Any, fallback: float = 0.0) -> float:
    value = getattr(state, "orientation", None)
    if value is None:
        return float(fallback)
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _state_speed(state: Any) -> float:
    try:
        return float(getattr(state, "velocity", 0.0))
    except Exception:
        return 0.0


def _state_velocity_xy(state: Any, heading: float | None = None) -> np.ndarray:
    hd = _state_heading(state, 0.0) if heading is None else float(heading)
    v_long = _state_speed(state)
    v_lat = 0.0
    if hasattr(state, "velocity_y"):
        try:
            v_lat = float(getattr(state, "velocity_y"))
        except Exception:
            v_lat = 0.0
    c, s = math.cos(hd), math.sin(hd)
    return np.asarray([v_long * c - v_lat * s, v_long * s + v_lat * c], dtype=np.float64)


def _points_global_to_local(points: np.ndarray, origin: np.ndarray, heading0: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    delta = pts - origin.reshape(1, 2)
    c, s = math.cos(float(heading0)), math.sin(float(heading0))
    local = np.empty_like(delta)
    local[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    local[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    return local


def _vectors_global_to_local(vectors: np.ndarray, heading0: float) -> np.ndarray:
    vec = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    c, s = math.cos(float(heading0)), math.sin(float(heading0))
    local = np.empty_like(vec)
    local[:, 0] = c * vec[:, 0] + s * vec[:, 1]
    local[:, 1] = -s * vec[:, 0] + c * vec[:, 1]
    return local


def _shape_size_lw(shape: Any, default_l: float = 4.8, default_w: float = 2.0) -> Tuple[float, float]:
    try:
        length = getattr(shape, "length", None)
        width = getattr(shape, "width", None)
        if length is not None and width is not None:
            return float(length), float(width)
        radius = getattr(shape, "radius", None)
        if radius is not None:
            r = float(radius)
            return 2.0 * r, 2.0 * r
    except Exception:
        pass
    return float(default_l), float(default_w)


def _state_time_step(state: Any) -> int | None:
    if state is None:
        return None
    value = getattr(state, "time_step", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return _safe_int(value, 0)


def _trajectory_state_at(trajectory: Any, time_step: int) -> Any | None:
    if trajectory is None:
        return None
    method = getattr(trajectory, "state_at_time_step", None)
    if callable(method):
        try:
            state = method(time_step)
            if state is not None:
                return state
        except Exception:
            pass
    for state in list(getattr(trajectory, "state_list", []) or []):
        if _state_time_step(state) == int(time_step):
            return state
    initial = getattr(trajectory, "initial_state", None)
    if _state_time_step(initial) == int(time_step):
        return initial
    return None


def _dynamic_state_at(obstacle: Any, time_step: int) -> Any | None:
    initial = getattr(obstacle, "initial_state", None)
    if _state_time_step(initial) == int(time_step):
        return initial
    pred = getattr(obstacle, "prediction", None)
    traj = getattr(pred, "trajectory", None) if pred is not None else None
    return _trajectory_state_at(traj, int(time_step))


def _available_time_steps(obstacle: Any) -> List[int]:
    out = set()
    t0 = _state_time_step(getattr(obstacle, "initial_state", None))
    if t0 is not None:
        out.add(t0)
    pred = getattr(obstacle, "prediction", None)
    traj = getattr(pred, "trajectory", None) if pred is not None else None
    for state in list(getattr(traj, "state_list", []) or []):
        t = _state_time_step(state)
        if t is not None:
            out.add(t)
    return sorted(out)


def _vehicle_like(obstacle: Any) -> bool:
    name = str(getattr(getattr(obstacle, "obstacle_type", ""), "value", getattr(obstacle, "obstacle_type", ""))).lower()
    return name in {"car", "truck", "bus", "motorcycle"}


def _commonroad_type_name(obstacle: Any) -> str:
    typ = getattr(obstacle, "obstacle_type", "unknown")
    return str(getattr(typ, "value", typ))


def _oriented_box(cx: float, cy: float, length: float, width: float, heading: float) -> Polygon:
    hl = 0.5 * float(length)
    hw = 0.5 * float(width)
    corners = np.asarray([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
    c, s = math.cos(float(heading)), math.sin(float(heading))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    pts = corners @ rot.T + np.asarray([cx, cy], dtype=np.float64)
    return Polygon(pts)


def _current_distance_ttc(ego_state: Any, ego_shape: Any, other_items: List[Tuple[Any, Any]]) -> Tuple[float, float]:
    ego_pos = _as_xy(getattr(ego_state, "position"))
    ego_heading = _state_heading(ego_state, 0.0)
    ego_vel = _state_velocity_xy(ego_state, ego_heading)
    ego_l, ego_w = _shape_size_lw(ego_shape)
    ego_poly = _oriented_box(ego_pos[0], ego_pos[1], ego_l, ego_w, ego_heading)
    min_dist = math.inf
    best_ttc = -1.0
    for obs, state in other_items:
        pos = _as_xy(getattr(state, "position"))
        heading = _state_heading(state, ego_heading)
        vel = _state_velocity_xy(state, heading)
        l, w = _shape_size_lw(getattr(obs, "obstacle_shape", None))
        poly = _oriented_box(pos[0], pos[1], l, w, heading)
        dist = float(ego_poly.distance(poly))
        if dist < min_dist:
            min_dist = dist
        rel_pos = pos - ego_pos
        center_dist = float(np.linalg.norm(rel_pos))
        if center_dist > 1e-6:
            rel_vel = vel - ego_vel
            closing = -float(np.dot(rel_pos, rel_vel)) / center_dist
            if closing > 1e-6:
                approx_gap = max(center_dist - 0.5 * math.hypot(ego_l, ego_w) - 0.5 * math.hypot(l, w), 0.0)
                ttc = approx_gap / closing
                if best_ttc < 0 or ttc < best_ttc:
                    best_ttc = float(ttc)
    if not math.isfinite(min_dist):
        min_dist = math.inf
    return float(min_dist), float(best_ttc)


def _extract_lane_centerlines(scenario: Any, origin: np.ndarray, heading0: float) -> List[List[List[float]]]:
    lanelet_network = getattr(scenario, "lanelet_network", None)
    lanelets = getattr(lanelet_network, "lanelets", []) if lanelet_network is not None else []
    lines: List[List[List[float]]] = []
    for lanelet in lanelets:
        vertices = getattr(lanelet, "center_vertices", None)
        if vertices is None:
            continue
        arr = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] < 2:
            continue
        local = _points_global_to_local(arr, origin, heading0)
        lines.append(np.round(local, 6).tolist())
    return lines


def _future_for_dynamic(
    obstacle: Any,
    current_state: Any,
    origin: np.ndarray,
    heading0: float,
    current_time_step: int,
    horizon_steps: int,
    dt: float,
    allow_cv_fallback: bool,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[bool], bool]:
    out_xy: List[List[float]] = []
    out_vel: List[List[float]] = []
    out_heading: List[float] = []
    out_valid: List[bool] = []
    cv_used = False
    current_pos = _as_xy(getattr(current_state, "position"))
    current_heading = _state_heading(current_state, heading0)
    current_vel = _state_velocity_xy(current_state, current_heading)
    for k in range(horizon_steps + 1):
        step = current_time_step + k
        state = _dynamic_state_at(obstacle, step)
        if state is None and allow_cv_fallback:
            cv_used = True
            pos = current_pos + current_vel * (k * dt)
            hd = current_heading
            vel = current_vel
            valid = True
        elif state is None:
            out_xy.append([0.0, 0.0])
            out_vel.append([0.0, 0.0])
            out_heading.append(0.0)
            out_valid.append(False)
            continue
        else:
            pos = _as_xy(getattr(state, "position"))
            hd = _state_heading(state, current_heading)
            vel = _state_velocity_xy(state, hd)
            valid = True
        local_xy = _points_global_to_local(pos.reshape(1, 2), origin, heading0)[0]
        local_vel = _vectors_global_to_local(vel.reshape(1, 2), heading0)[0]
        out_xy.append(np.round(local_xy, 6).tolist())
        out_vel.append(np.round(local_vel, 6).tolist())
        out_heading.append(round(_wrap_to_pi(hd - heading0), 6))
        out_valid.append(valid)
    return out_xy, out_vel, out_heading, out_valid, cv_used


def _append_agent(
    *,
    agent_ids: List[Any],
    agent_types: List[str],
    current_xy: List[List[float]],
    current_vel_xy: List[List[float]],
    current_heading: List[float],
    current_size_lw: List[List[float]],
    future_xy: List[List[List[float]]],
    future_vel_xy: List[List[List[float]]],
    future_heading: List[List[float]],
    future_valid: List[List[bool]],
    obstacle: Any,
    state: Any,
    origin: np.ndarray,
    heading0: float,
    current_time_step: int,
    horizon_steps: int,
    dt: float,
    static: bool = False,
) -> bool:
    pos = _as_xy(getattr(state, "position"))
    hd = _state_heading(state, heading0)
    vel = np.zeros(2, dtype=np.float64) if static else _state_velocity_xy(state, hd)
    local_xy = _points_global_to_local(pos.reshape(1, 2), origin, heading0)[0]
    local_vel = _vectors_global_to_local(vel.reshape(1, 2), heading0)[0]
    l, w = _shape_size_lw(getattr(obstacle, "obstacle_shape", None))
    if static:
        this_xy: List[List[float]] = []
        this_vel: List[List[float]] = []
        this_hd: List[float] = []
        this_valid: List[bool] = []
        for _ in range(horizon_steps + 1):
            this_xy.append(np.round(local_xy, 6).tolist())
            this_vel.append([0.0, 0.0])
            this_hd.append(round(_wrap_to_pi(hd - heading0), 6))
            this_valid.append(True)
    else:
        this_xy, this_vel, this_hd, this_valid, _ = _future_for_dynamic(
            obstacle,
            state,
            origin,
            heading0,
            current_time_step,
            horizon_steps,
            dt,
            allow_cv_fallback=False,
        )
    agent_ids.append(getattr(obstacle, "obstacle_id", len(agent_ids)))
    agent_types.append(_commonroad_type_name(obstacle))
    current_xy.append(np.round(local_xy, 6).tolist())
    current_vel_xy.append(np.round(local_vel, 6).tolist())
    current_heading.append(round(_wrap_to_pi(hd - heading0), 6))
    current_size_lw.append([round(l, 6), round(w, 6)])
    future_xy.append(this_xy)
    future_vel_xy.append(this_vel)
    future_heading.append(this_hd)
    future_valid.append(this_valid)
    return True


def _scenario_stats(scenario: Any) -> Tuple[List[Any], List[Any], List[Any], int]:
    dynamic = list(getattr(scenario, "dynamic_obstacles", []) or [])
    static = list(getattr(scenario, "static_obstacles", []) or [])
    lanelet_network = getattr(scenario, "lanelet_network", None)
    lanelets = list(getattr(lanelet_network, "lanelets", []) or []) if lanelet_network is not None else []
    return dynamic, static, lanelets, len(lanelets)


def _candidate_bucket(current_min_distance_m: float, neighbor_count: int) -> str:
    if current_min_distance_m < 3.0:
        return "close_interaction"
    if current_min_distance_m < 10.0:
        return "medium_interaction"
    if neighbor_count >= 5:
        return "far_but_interactive"
    return "random_fill"


def _discover_candidates(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    xml_path = str(row.get("xml_path", ""))
    try:
        scenario, _ = CommonRoadFileReader(xml_path).open()
        dynamic, static, lanelets, lanelet_count = _scenario_stats(scenario)
        dynamic_by_id = {str(getattr(o, "obstacle_id", "")): o for o in dynamic}
        scenario_id = str(getattr(scenario, "scenario_id", row.get("scenario_id", "")))
        dt = float(getattr(scenario, "dt", 0.1))
        for ego in dynamic:
            if not _vehicle_like(ego):
                continue
            ego_id = str(getattr(ego, "obstacle_id", ""))
            time_steps = _available_time_steps(ego)
            if not time_steps:
                continue
            selected_times = time_steps[:: max(1, int(args.time_stride))]
            for t in selected_times:
                ego_state = _dynamic_state_at(ego, t)
                if ego_state is None:
                    continue
                ego_speed = float(np.linalg.norm(_state_velocity_xy(ego_state, _state_heading(ego_state, 0.0))))
                if ego_speed < float(args.min_ego_speed_mps):
                    continue
                others: List[Tuple[Any, Any]] = []
                for obs in dynamic:
                    if str(getattr(obs, "obstacle_id", "")) == ego_id:
                        continue
                    state = _dynamic_state_at(obs, t)
                    if state is not None:
                        others.append((obs, state))
                if args.include_static_obstacles:
                    for obs in static:
                        state = getattr(obs, "initial_state", None)
                        if state is not None:
                            others.append((obs, state))
                agent_count = 1 + len(others)
                if agent_count < int(args.min_neighbor_agents) + 1:
                    continue
                dmin, ttc = _current_distance_ttc(ego_state, getattr(ego, "obstacle_shape", None), others)
                if not math.isfinite(dmin):
                    continue
                bucket = _candidate_bucket(dmin, agent_count - 1)
                candidates.append(
                    {
                        "commonroad_scenario_id": scenario_id,
                        "xml_path": xml_path,
                        "ego_obstacle_id": ego_id,
                        "current_time_step": int(t),
                        "agent_count": agent_count,
                        "dynamic_agent_count": len([1 for obs, _ in others if str(getattr(obs, "obstacle_id", "")) in dynamic_by_id]),
                        "static_agent_count": len(others) - len([1 for obs, _ in others if str(getattr(obs, "obstacle_id", "")) in dynamic_by_id]),
                        "lanelet_count": lanelet_count,
                        "ego_speed_mps": ego_speed,
                        "current_min_distance_m": dmin,
                        "current_ttc_s": ttc,
                        "bucket": bucket,
                        "dt": dt,
                    }
                )
    except Exception as exc:
        failures.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "xml_path": xml_path,
                "ego_obstacle_id": "",
                "current_time_step": "",
                "export_error": "".join(traceback.format_exception_only(type(exc), exc)).strip()[:800],
                "traceback": traceback.format_exc(),
            }
        )
    return candidates, failures


def _select_candidates(candidates: List[Dict[str, Any]], max_samples: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for cand in candidates:
        key = (cand["commonroad_scenario_id"], cand["ego_obstacle_id"], int(cand["current_time_step"]))
        if key in seen:
            continue
        seen.add(key)
        by_bucket[cand["bucket"]].append(cand)
    for rows in by_bucket.values():
        rng.shuffle(rows)
    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    def take(bucket: str, limit: int) -> None:
        for cand in by_bucket.get(bucket, []):
            if len(selected) >= max_samples or len([r for r in selected if r["bucket"] == bucket]) >= limit:
                break
            key = (cand["commonroad_scenario_id"], cand["ego_obstacle_id"], int(cand["current_time_step"]))
            if key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(key)

    take("medium_interaction", int(round(max_samples * 0.35)))
    take("far_but_interactive", int(round(max_samples * 0.20)))
    take("close_interaction", int(math.floor(max_samples * 0.25)))

    remaining = []
    for bucket, rows in by_bucket.items():
        for cand in rows:
            key = (cand["commonroad_scenario_id"], cand["ego_obstacle_id"], int(cand["current_time_step"]))
            if key not in selected_keys:
                remaining.append(cand)
    rng.shuffle(remaining)
    for cand in remaining:
        if len(selected) >= max_samples:
            break
        if cand["bucket"] == "close_interaction" and len([r for r in selected if r["bucket"] == "close_interaction"]) >= int(math.floor(max_samples * 0.25)):
            continue
        selected.append(cand)
    return selected[:max_samples]


def _build_sample(cand: Dict[str, Any], scenario: Any, args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    dynamic, static, lanelets, lanelet_count = _scenario_stats(scenario)
    ego_id = str(cand["ego_obstacle_id"])
    current_time_step = int(cand["current_time_step"])
    dt = float(getattr(scenario, "dt", cand.get("dt", 0.1)))
    ego = None
    for obs in dynamic:
        if str(getattr(obs, "obstacle_id", "")) == ego_id:
            ego = obs
            break
    if ego is None:
        raise ValueError(f"ego obstacle not found: {ego_id}")
    ego_state = _dynamic_state_at(ego, current_time_step)
    if ego_state is None:
        raise ValueError(f"ego state missing at time_step={current_time_step}")
    origin = _as_xy(getattr(ego_state, "position"))
    heading0 = _state_heading(ego_state, 0.0)
    scenario_id = str(cand["commonroad_scenario_id"])
    sample_id = f"crdyn_{scenario_id}_ego{ego_id}_t{current_time_step}"
    sample_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id)

    agent_ids: List[Any] = [getattr(ego, "obstacle_id", 0)]
    agent_types: List[str] = ["ego"]
    current_xy: List[List[float]] = [[0.0, 0.0]]
    ego_vel_local = _vectors_global_to_local(_state_velocity_xy(ego_state, heading0).reshape(1, 2), heading0)[0]
    current_vel_xy: List[List[float]] = [np.round(ego_vel_local, 6).tolist()]
    current_heading: List[float] = [0.0]
    ego_l, ego_w = _shape_size_lw(getattr(ego, "obstacle_shape", None))
    current_size_lw: List[List[float]] = [[round(ego_l, 6), round(ego_w, 6)]]
    future_xy: List[List[List[float]]] = []
    future_vel_xy: List[List[List[float]]] = []
    future_heading: List[List[float]] = []
    future_valid: List[List[bool]] = []
    ego_fx, ego_fv, ego_fh, ego_valid, ego_cv = _future_for_dynamic(
        ego,
        ego_state,
        origin,
        heading0,
        current_time_step,
        int(args.horizon_steps),
        dt,
        allow_cv_fallback=True,
    )
    future_xy.append(ego_fx)
    future_vel_xy.append(ego_fv)
    future_heading.append(ego_fh)
    future_valid.append(ego_valid)

    dynamic_agent_count = 0
    for obs in dynamic:
        if str(getattr(obs, "obstacle_id", "")) == ego_id:
            continue
        state = _dynamic_state_at(obs, current_time_step)
        if state is None:
            continue
        _append_agent(
            agent_ids=agent_ids,
            agent_types=agent_types,
            current_xy=current_xy,
            current_vel_xy=current_vel_xy,
            current_heading=current_heading,
            current_size_lw=current_size_lw,
            future_xy=future_xy,
            future_vel_xy=future_vel_xy,
            future_heading=future_heading,
            future_valid=future_valid,
            obstacle=obs,
            state=state,
            origin=origin,
            heading0=heading0,
            current_time_step=current_time_step,
            horizon_steps=int(args.horizon_steps),
            dt=dt,
            static=False,
        )
        dynamic_agent_count += 1

    static_agent_count = 0
    if args.include_static_obstacles:
        for obs in static:
            state = getattr(obs, "initial_state", None)
            if state is None:
                continue
            _append_agent(
                agent_ids=agent_ids,
                agent_types=agent_types,
                current_xy=current_xy,
                current_vel_xy=current_vel_xy,
                current_heading=current_heading,
                current_size_lw=current_size_lw,
                future_xy=future_xy,
                future_vel_xy=future_vel_xy,
                future_heading=future_heading,
                future_valid=future_valid,
                obstacle=obs,
                state=state,
                origin=origin,
                heading0=heading0,
                current_time_step=current_time_step,
                horizon_steps=int(args.horizon_steps),
                dt=dt,
                static=True,
            )
            static_agent_count += 1

    times_s = [round(k * dt, 6) for k in range(int(args.horizon_steps) + 1)]
    sample = {
        "sample_id": sample_id,
        "source_dataset": "CommonRoad",
        "scenario_id": sample_id,
        "commonroad_scenario_id": scenario_id,
        "xml_path": cand["xml_path"],
        "ego_obstacle_id": ego_id,
        "current_time_step": current_time_step,
        "dt": dt,
        "ego_index": 0,
        "agent_count": len(agent_ids),
        "agent_ids": agent_ids,
        "agent_types": agent_types,
        "times_s": times_s,
        "current_xy": current_xy,
        "current_vel_xy": current_vel_xy,
        "current_heading": current_heading,
        "current_size_lw": current_size_lw,
        "future_xy": future_xy,
        "future_vel_xy": future_vel_xy,
        "future_heading": future_heading,
        "future_valid": future_valid,
        "map_lane_centerlines": _extract_lane_centerlines(scenario, origin, heading0),
        "map_crosswalks": [],
        "map_driveways": [],
        "commonroad_metadata": {
            "bucket": cand["bucket"],
            "dynamic_agent_count": dynamic_agent_count,
            "static_agent_count": static_agent_count,
            "lanelet_count": lanelet_count,
            "ego_speed_mps": float(cand["ego_speed_mps"]),
            "current_min_distance_m": float(cand["current_min_distance_m"]),
            "current_ttc_s": float(cand["current_ttc_s"]),
            "ego_future_cv_fallback_used": bool(ego_cv),
        },
    }
    json_path = out_dir / "samples_json_gz" / f"{sample_id}.json.gz"
    _write_json_gz(json_path, sample)
    return {
        "sample_id": sample_id,
        "commonroad_scenario_id": scenario_id,
        "xml_path": cand["xml_path"],
        "ego_obstacle_id": ego_id,
        "current_time_step": current_time_step,
        "agent_count": len(agent_ids),
        "dynamic_agent_count": dynamic_agent_count,
        "static_agent_count": static_agent_count,
        "lanelet_count": lanelet_count,
        "ego_speed_mps": f"{float(cand['ego_speed_mps']):.6f}",
        "current_min_distance_m": f"{float(cand['current_min_distance_m']):.6f}",
        "current_ttc_s": f"{float(cand['current_ttc_s']):.6f}",
        "bucket": cand["bucket"],
        "export_status": "ok",
        "export_error": "",
        "json_gz_path": str(json_path),
    }


def _export_selected(selected: List[Dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    by_xml: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cand in selected:
        by_xml[str(cand["xml_path"])].append(cand)
    for xml_path, rows in by_xml.items():
        try:
            scenario, _ = CommonRoadFileReader(xml_path).open()
        except Exception as exc:
            for cand in rows:
                failures.append(
                    {
                        "scenario_id": cand.get("commonroad_scenario_id", ""),
                        "xml_path": xml_path,
                        "ego_obstacle_id": cand.get("ego_obstacle_id", ""),
                        "current_time_step": cand.get("current_time_step", ""),
                        "export_error": "".join(traceback.format_exception_only(type(exc), exc)).strip()[:800],
                        "traceback": traceback.format_exc(),
                    }
                )
            continue
        for cand in rows:
            try:
                manifest.append(_build_sample(cand, scenario, args, out_dir))
            except Exception as exc:
                failures.append(
                    {
                        "scenario_id": cand.get("commonroad_scenario_id", ""),
                        "xml_path": xml_path,
                        "ego_obstacle_id": cand.get("ego_obstacle_id", ""),
                        "current_time_step": cand.get("current_time_step", ""),
                        "export_error": "".join(traceback.format_exception_only(type(exc), exc)).strip()[:800],
                        "traceback": traceback.format_exc(),
                    }
                )
    return manifest, failures


def _describe(values: Iterable[float]) -> Dict[str, str]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
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


def _write_summary(path: Path, candidates: List[Dict[str, Any]], manifest: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    fields = ["section", "metric", "value", "count", "mean", "std", "min", "p25", "p50", "p75", "max", "sample_id"]
    rows: List[Dict[str, Any]] = [
        {"section": "counts", "metric": "candidate_count_before_sampling", "value": len(candidates)},
        {"section": "counts", "metric": "exported_count", "value": len(manifest)},
        {"section": "counts", "metric": "failed_count", "value": len(failures)},
        {"section": "counts", "metric": "scenario_count", "value": len(set(r.get("commonroad_scenario_id", "") for r in manifest))},
        {"section": "counts", "metric": "ego_obstacle_count", "value": len(set((r.get("commonroad_scenario_id", ""), r.get("ego_obstacle_id", "")) for r in manifest))},
    ]
    for bucket, count in Counter(r.get("bucket", "") for r in manifest).most_common():
        rows.append({"section": "bucket_distribution", "metric": "bucket", "value": bucket, "count": count})
    for metric in ["agent_count", "ego_speed_mps", "current_min_distance_m", "current_ttc_s", "lanelet_count"]:
        rec = {"section": "describe", "metric": metric, "value": ""}
        rec.update(_describe(_safe_float(r.get(metric), math.nan) for r in manifest))
        rows.append(rec)
    for row in manifest[:20]:
        rows.append({"section": "first20_sample_id", "metric": "sample_id", "sample_id": row.get("sample_id", "")})
    _write_csv(path, rows, fields)


def main() -> None:
    args = parse_args()
    work_dir = _load_work_dir(args.config)
    out_dir = work_dir / "results" / "commonroad_samples" / args.out_name
    (out_dir / "samples_json_gz").mkdir(parents=True, exist_ok=True)
    pilot_rows = _read_csv(Path(args.pilot_csv))
    if args.scenario_limit is not None:
        pilot_rows = pilot_rows[: max(0, int(args.scenario_limit))]

    all_candidates: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for idx, row in enumerate(pilot_rows, start=1):
        candidates, row_failures = _discover_candidates(row, args)
        all_candidates.extend(candidates)
        failures.extend(row_failures)
        if idx % 25 == 0 or idx == len(pilot_rows):
            print(f"[discover] scenarios={idx}/{len(pilot_rows)} candidates={len(all_candidates)} failures={len(failures)}")

    selected = _select_candidates(all_candidates, int(args.max_samples), int(args.seed))
    manifest, export_failures = _export_selected(selected, args, out_dir)
    failures.extend(export_failures)

    manifest_path = out_dir / "commonroad_dynamic_ego_samples_manifest.csv"
    summary_path = out_dir / "commonroad_dynamic_ego_export_summary.csv"
    failures_path = out_dir / "commonroad_dynamic_ego_export_failures.csv"
    _write_csv(manifest_path, manifest, MANIFEST_FIELDS)
    _write_csv(failures_path, failures, FAILURE_FIELDS)
    _write_summary(summary_path, all_candidates, manifest, failures)

    print(f"[done] samples_dir={out_dir / 'samples_json_gz'}")
    print(f"[done] manifest={manifest_path}")
    print(f"[done] summary={summary_path}")
    print(f"[done] failures={failures_path}")
    print(f"[done] candidates={len(all_candidates)} exported={len(manifest)} failed={len(failures)}")


if __name__ == "__main__":
    main()
