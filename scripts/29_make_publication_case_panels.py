#!/usr/bin/env python
"""Make publication-quality actionability case panels from selected cases.

This script only reads existing artifacts and sample pickles. It does not
train models, regenerate labels/features, or touch pipeline.py.
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Polygon

try:
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:  # pragma: no cover
    PdfPages = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rtbev.config import load_config
from rtbev.io_utils import ensure_dir


ACTIONS = [
    "keep",
    "mild_brake",
    "hard_brake",
    "left",
    "right",
    "brake_left",
    "brake_right",
]

ACTION_COLORS = {
    "feasible": "#1b9e77",
    "infeasible": "#b2182b",
}

INTENDED_USES = {
    "33e40c2133dc7ed8": (
        "recovered_positive",
        "baseline/CV missed critical actionability; enhanced model recovered.",
    ),
    "ab06a686a52a3fac": (
        "recovered_positive_partial",
        "partial feasible-action depletion; not purely all-or-nothing.",
    ),
    "2aedd6aa278acd2c": (
        "proximity_warning_high_actionability",
        "original warning but several actions remain feasible.",
    ),
    "17140261fe2db703": (
        "proximity_warning_high_actionability",
        "proximity warning with high actionability; baseline score reduced.",
    ),
    "cddf5c8291665dd7": (
        "baseline_false_positive_fixed",
        "baseline/CV high alert corrected by actionability.",
    ),
    "3b007871ad02fbca": (
        "proximity_safe_caution_but_critical",
        "original caution but no-map actionability critical; use as concept/supplement, not primary model success.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--proximity-labels-csv", required=True)
    parser.add_argument("--actionability-labels-csv", required=True)
    parser.add_argument("--out-name", default="publication_case_panels_v1")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--rollout-use-map", dest="rollout_use_map", action="store_true", default=True)
    parser.add_argument("--no-rollout-use-map", dest="rollout_use_map", action="store_false")
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--obstacle-mode", choices=["oracle_future", "cv_current"], default="oracle_future")
    return parser.parse_args()


def _config_get(cfg: Any, path: str, default: Any = None) -> Any:
    node = cfg
    for key in path.split("."):
        if isinstance(node, dict):
            if key not in node:
                return default
            node = node[key]
        else:
            return default
    return node


def _work_dir(cfg: Dict[str, Any]) -> Path:
    work = _config_get(cfg, "project.work_dir")
    if not work:
        raise ValueError("config project.work_dir is missing")
    return Path(str(work))


def _load_sample(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _as_array(x: Any, dtype: Any = float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _safe_float(row: pd.Series, names: Sequence[str], default: float = np.nan) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                return default
    return default


def _safe_str(row: pd.Series, names: Sequence[str], default: str = "") -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return str(row[name])
    return default


def _safe_int(row: pd.Series, names: Sequence[str], default: int = -1) -> int:
    value = _safe_float(row, names, np.nan)
    if np.isfinite(value):
        return int(value)
    return default


def _wrap_angle(theta: float) -> float:
    return (float(theta) + np.pi) % (2.0 * np.pi) - np.pi


def _ego_state(sample: Dict[str, Any]) -> Tuple[int, np.ndarray, float, float]:
    ego_idx = int(sample.get("ego_index", 0))
    current_xy = _as_array(sample.get("current_xy"))
    origin = current_xy[ego_idx].astype(float)

    heading = None
    if "current_velocity" in sample:
        vel = _as_array(sample.get("current_velocity"))
        speed = float(np.linalg.norm(vel[ego_idx]))
        if speed > 0.2:
            heading = float(np.arctan2(vel[ego_idx, 1], vel[ego_idx, 0]))
    if heading is None and "current_heading" in sample:
        heading = float(_as_array(sample.get("current_heading"))[ego_idx])
    if heading is None:
        heading = 0.0

    speed_mps = 0.0
    if "current_velocity" in sample:
        speed_mps = float(np.linalg.norm(_as_array(sample.get("current_velocity"))[ego_idx]))
    elif "ego_speed_mps" in sample:
        speed_mps = float(sample["ego_speed_mps"])
    return ego_idx, origin, heading, speed_mps


def _to_local(points: np.ndarray, origin: np.ndarray, ego_heading: float) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape((-1, 2))
    flat = arr.reshape((-1, 2))
    delta = flat - origin.reshape(1, 2)
    forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    left = np.array([-np.sin(ego_heading), np.cos(ego_heading)])
    out = np.stack([delta @ forward, delta @ left], axis=1)
    return out.reshape(arr.shape)


def _local_heading(heading: float, ego_heading: float) -> float:
    return _wrap_angle(float(heading) - float(ego_heading))


def _box_corners(center: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    c = np.array(center, dtype=float)
    l2 = max(float(length), 0.1) / 2.0
    w2 = max(float(width), 0.1) / 2.0
    base = np.array([[l2, w2], [l2, -w2], [-l2, -w2], [-l2, w2]], dtype=float)
    rot = np.array(
        [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]],
        dtype=float,
    )
    return c + base @ rot.T


def _agent_size(sample: Dict[str, Any], idx: int) -> Tuple[float, float]:
    for key in ("size", "agent_size", "current_size"):
        if key in sample:
            size = _as_array(sample[key])
            if size.ndim == 2 and idx < size.shape[0]:
                length = float(size[idx, 0])
                width = float(size[idx, 1]) if size.shape[1] > 1 else 2.0
                return max(length, 0.5), max(width, 0.4)
    return 4.5, 2.0


def _agent_heading(sample: Dict[str, Any], idx: int, fallback: float = 0.0) -> float:
    if "current_heading" in sample:
        arr = _as_array(sample["current_heading"])
        if idx < arr.shape[0] and np.isfinite(arr[idx]):
            return float(arr[idx])
    if "current_velocity" in sample:
        vel = _as_array(sample["current_velocity"])
        if idx < vel.shape[0] and np.linalg.norm(vel[idx]) > 0.1:
            return float(np.arctan2(vel[idx, 1], vel[idx, 0]))
    return fallback


def _valid_agents(sample: Dict[str, Any]) -> np.ndarray:
    current_xy = _as_array(sample.get("current_xy"))
    valid = np.ones(current_xy.shape[0], dtype=bool)
    if "current_valid" in sample:
        arr = np.asarray(sample["current_valid"]).astype(bool)
        if arr.shape[0] == valid.shape[0]:
            valid &= arr
    valid &= np.isfinite(current_xy).all(axis=1)
    return valid


def _future_xy(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("future_xy", "future_pos", "future_positions"):
        if key in sample:
            return _as_array(sample[key])
    return None


def _future_valid(sample: Dict[str, Any], n_agents: int, n_steps: int) -> np.ndarray:
    for key in ("future_valid", "future_mask"):
        if key in sample:
            arr = np.asarray(sample[key]).astype(bool)
            if arr.shape[:2] == (n_agents, n_steps):
                return arr
    return np.ones((n_agents, n_steps), dtype=bool)


def _future_heading(sample: Dict[str, Any], n_agents: int, n_steps: int) -> np.ndarray:
    if "future_heading" in sample:
        arr = _as_array(sample["future_heading"])
        if arr.shape[:2] == (n_agents, n_steps):
            return arr
    cur = np.array([_agent_heading(sample, i) for i in range(n_agents)], dtype=float)
    return np.repeat(cur[:, None], n_steps, axis=1)


def _dt_s(cfg: Dict[str, Any]) -> float:
    return float(_config_get(cfg, "labels.dt_s", _config_get(cfg, "features.dt_s", 0.1)) or 0.1)


def _rollout_action(
    sample: Dict[str, Any],
    action: str,
    cfg: Dict[str, Any],
    horizon_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego_idx, origin, heading, speed_mps = _ego_state(sample)
    del ego_idx

    dt = _dt_s(cfg)
    n_steps = max(1, int(np.ceil(horizon_s / dt)))
    times = np.arange(1, n_steps + 1, dtype=float) * dt

    accel = {
        "keep": 0.0,
        "mild_brake": -1.5,
        "hard_brake": -4.5,
        "left": 0.0,
        "right": 0.0,
        "brake_left": -3.0,
        "brake_right": -3.0,
    }[action]
    lat_target = {
        "keep": 0.0,
        "mild_brake": 0.0,
        "hard_brake": 0.0,
        "left": 3.2,
        "right": -3.2,
        "brake_left": 3.0,
        "brake_right": -3.0,
    }[action]

    speeds = np.maximum(0.0, speed_mps + accel * times)
    forward = np.cumsum(speeds * dt)
    lateral = lat_target * np.minimum(1.0, times / 1.5)

    local = np.stack([forward, lateral], axis=1)
    forward_vec = np.array([np.cos(heading), np.sin(heading)])
    left_vec = np.array([-np.sin(heading), np.cos(heading)])
    global_xy = origin[None, :] + local[:, 0:1] * forward_vec[None, :] + local[:, 1:2] * left_vec[None, :]
    headings = np.full(n_steps, heading, dtype=float)
    if abs(lat_target) > 1e-6:
        dy_dt = np.gradient(lateral, dt)
        headings = heading + np.arctan2(dy_dt, np.maximum(speeds, 0.1))
    return global_xy, headings, times


def _obstacle_pose(
    sample: Dict[str, Any],
    idx: int,
    step: int,
    dt: float,
    obstacle_mode: str,
) -> Optional[Tuple[np.ndarray, float]]:
    current_xy = _as_array(sample.get("current_xy"))
    if idx >= current_xy.shape[0] or not np.isfinite(current_xy[idx]).all():
        return None

    if obstacle_mode == "oracle_future":
        fut = _future_xy(sample)
        if fut is not None and fut.ndim == 3 and idx < fut.shape[0] and step < fut.shape[1]:
            fvalid = _future_valid(sample, fut.shape[0], fut.shape[1])
            if fvalid[idx, step] and np.isfinite(fut[idx, step]).all():
                fheading = _future_heading(sample, fut.shape[0], fut.shape[1])
                return fut[idx, step], float(fheading[idx, step])

    xy = current_xy[idx].astype(float).copy()
    heading = _agent_heading(sample, idx)
    if "current_velocity" in sample:
        vel = _as_array(sample["current_velocity"])
        if idx < vel.shape[0] and np.isfinite(vel[idx]).all():
            xy = xy + vel[idx] * ((step + 1) * dt)
            if np.linalg.norm(vel[idx]) > 0.1:
                heading = float(np.arctan2(vel[idx, 1], vel[idx, 0]))
    return xy, heading


def _rect_overlap_sat(
    c1: np.ndarray,
    h1: float,
    l1: float,
    w1: float,
    c2: np.ndarray,
    h2: float,
    l2: float,
    w2: float,
    buffer_m: float = 0.0,
) -> bool:
    p1 = _box_corners(c1, h1, l1 + 2 * buffer_m, w1 + 2 * buffer_m)
    p2 = _box_corners(c2, h2, l2 + 2 * buffer_m, w2 + 2 * buffer_m)
    axes = []
    for poly in (p1, p2):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            norm = np.linalg.norm(edge)
            if norm > 1e-9:
                axes.append(np.array([-edge[1], edge[0]]) / norm)
    for axis in axes:
        s1 = p1 @ axis
        s2 = p2 @ axis
        if s1.max() < s2.min() or s2.max() < s1.min():
            return False
    return True


def _analyze_action(
    sample: Dict[str, Any],
    action: str,
    cfg: Dict[str, Any],
    horizon_s: float,
    obstacle_mode: str,
    rollout_use_map: bool,
) -> Dict[str, Any]:
    # Map feasibility is intentionally conservative. The publication run uses
    # --no-rollout-use-map, so only agent collisions are considered there.
    del rollout_use_map
    ego_idx, _, _, _ = _ego_state(sample)
    ego_l, ego_w = _agent_size(sample, ego_idx)
    valid = _valid_agents(sample)
    global_xy, headings, times = _rollout_action(sample, action, cfg, horizon_s)
    dt = _dt_s(cfg)
    current_xy = _as_array(sample.get("current_xy"))

    for step, (xy, heading, t) in enumerate(zip(global_xy, headings, times)):
        for idx in range(current_xy.shape[0]):
            if idx == ego_idx or not valid[idx]:
                continue
            pose = _obstacle_pose(sample, idx, step, dt, obstacle_mode)
            if pose is None:
                continue
            obs_xy, obs_heading = pose
            obs_l, obs_w = _agent_size(sample, idx)
            if _rect_overlap_sat(xy, heading, ego_l, ego_w, obs_xy, obs_heading, obs_l, obs_w):
                return {
                    "feasible": False,
                    "first_unsafe_time_s": float(t),
                    "unsafe_reason": "collision",
                    "blocking_agent_idx": int(idx),
                    "trajectory_xy": global_xy,
                    "trajectory_heading": headings,
                    "times": times,
                }

    return {
        "feasible": True,
        "first_unsafe_time_s": np.nan,
        "unsafe_reason": "no_failure",
        "blocking_agent_idx": -1,
        "trajectory_xy": global_xy,
        "trajectory_heading": headings,
        "times": times,
    }


def _analyze_rollouts(
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    horizon_s: float,
    obstacle_mode: str,
    rollout_use_map: bool,
) -> Dict[str, Dict[str, Any]]:
    return {
        action: _analyze_action(sample, action, cfg, horizon_s, obstacle_mode, rollout_use_map)
        for action in ACTIONS
    }


def _nearest_or_blocking_agent(sample: Dict[str, Any], rollouts: Dict[str, Dict[str, Any]]) -> int:
    blocking = [v.get("blocking_agent_idx", -1) for v in rollouts.values() if int(v.get("blocking_agent_idx", -1)) >= 0]
    if blocking:
        return int(Counter(blocking).most_common(1)[0][0])

    ego_idx, origin, _, _ = _ego_state(sample)
    valid = _valid_agents(sample)
    current_xy = _as_array(sample.get("current_xy"))
    best_idx = -1
    best_dist = np.inf
    for idx, xy in enumerate(current_xy):
        if idx == ego_idx or not valid[idx] or not np.isfinite(xy).all():
            continue
        dist = float(np.linalg.norm(xy - origin))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _draw_lanes(ax: plt.Axes, sample: Dict[str, Any], origin: np.ndarray, ego_heading: float) -> None:
    lanes = sample.get("map_lane_centerlines", sample.get("lane_centerlines", None))
    if lanes is None:
        return
    try:
        iterable = lanes.values() if isinstance(lanes, dict) else lanes
        for lane in iterable:
            arr = _as_array(lane)
            if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
                continue
            pts = _to_local(arr[:, :2], origin, ego_heading)
            if not np.isfinite(pts).all():
                continue
            ax.plot(pts[:, 0], pts[:, 1], color="#c8c8c8", lw=0.6, alpha=0.55, zorder=0)
    except Exception:
        return


def _draw_vehicle(
    ax: plt.Axes,
    center_local: np.ndarray,
    heading_local: float,
    length: float,
    width: float,
    face: str,
    edge: str,
    lw: float,
    alpha: float,
    zorder: int,
) -> None:
    patch = Polygon(
        _box_corners(center_local, heading_local, length, width),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _draw_future(ax: plt.Axes, sample: Dict[str, Any], origin: np.ndarray, ego_heading: float) -> None:
    fut = _future_xy(sample)
    if fut is None or fut.ndim != 3:
        return
    ego_idx, _, _, _ = _ego_state(sample)
    fvalid = _future_valid(sample, fut.shape[0], fut.shape[1])
    for idx in range(fut.shape[0]):
        valid = fvalid[idx] & np.isfinite(fut[idx]).all(axis=1)
        if valid.sum() < 2:
            continue
        pts = _to_local(fut[idx, valid], origin, ego_heading)
        if idx == ego_idx:
            ax.plot(pts[:, 0], pts[:, 1], color="#2166ac", lw=1.2, alpha=0.35, zorder=1)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color="#666666", lw=0.8, alpha=0.25, zorder=1)


def _collect_plot_points(
    sample: Dict[str, Any],
    origin: np.ndarray,
    ego_heading: float,
    rollouts: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    pts: List[np.ndarray] = []
    current_xy = _as_array(sample.get("current_xy"))
    valid = _valid_agents(sample)
    if current_xy.size:
        pts.append(_to_local(current_xy[valid], origin, ego_heading).reshape((-1, 2)))
    fut = _future_xy(sample)
    if fut is not None and fut.ndim == 3:
        fvalid = _future_valid(sample, fut.shape[0], fut.shape[1])
        mask = fvalid & np.isfinite(fut).all(axis=2)
        if mask.any():
            pts.append(_to_local(fut[mask], origin, ego_heading).reshape((-1, 2)))
    for info in rollouts.values():
        pts.append(_to_local(info["trajectory_xy"], origin, ego_heading).reshape((-1, 2)))
    if not pts:
        return np.zeros((1, 2), dtype=float)
    out = np.concatenate(pts, axis=0)
    out = out[np.isfinite(out).all(axis=1)]
    return out if out.size else np.zeros((1, 2), dtype=float)


def _set_publication_limits(ax: plt.Axes, points_local: np.ndarray) -> None:
    pts = np.asarray(points_local, dtype=float)
    near = pts[(pts[:, 0] > -8) & (pts[:, 0] < 65) & (np.abs(pts[:, 1]) < 28)]
    if near.shape[0] == 0:
        near = pts
    x_min = min(-5.0, float(np.nanmin(near[:, 0])) - 3.0)
    x_max = max(45.0, float(np.nanmax(near[:, 0])) + 3.0)
    y_min = min(-15.0, float(np.nanmin(near[:, 1])) - 3.0)
    y_max = max(15.0, float(np.nanmax(near[:, 1])) + 3.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e8e8e8", linewidth=0.35)


def _draw_current_scene(
    ax: plt.Axes,
    sample: Dict[str, Any],
    rollouts: Dict[str, Dict[str, Any]],
    title: str,
    points_local: np.ndarray,
) -> None:
    ego_idx, origin, ego_heading, _ = _ego_state(sample)
    relevant_idx = _nearest_or_blocking_agent(sample, rollouts)
    valid = _valid_agents(sample)
    current_xy = _as_array(sample.get("current_xy"))

    _draw_lanes(ax, sample, origin, ego_heading)
    _draw_future(ax, sample, origin, ego_heading)

    for idx, xy in enumerate(current_xy):
        if not valid[idx]:
            continue
        center = _to_local(xy.reshape(1, 2), origin, ego_heading)[0]
        heading = _local_heading(_agent_heading(sample, idx, ego_heading), ego_heading)
        length, width = _agent_size(sample, idx)
        if idx == ego_idx:
            face, edge, lw, alpha, z = "#2166ac", "#053061", 1.0, 0.92, 5
        elif idx == relevant_idx:
            face, edge, lw, alpha, z = "#d95f02", "#b2182b", 1.8, 0.86, 4
        else:
            face, edge, lw, alpha, z = "#e69f00", "#8c510a", 0.55, 0.56, 3
        _draw_vehicle(ax, center, heading, length, width, face, edge, lw, alpha, z)

    if relevant_idx >= 0:
        rel = _to_local(current_xy[relevant_idx].reshape(1, 2), origin, ego_heading)[0]
        ax.add_patch(Circle(rel, radius=1.8, fill=False, edgecolor="#b2182b", lw=1.2, alpha=0.9, zorder=6))

    _set_publication_limits(ax, points_local)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("Forward distance from ego (m)", fontsize=8)
    ax.set_ylabel("Lateral offset (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def _draw_action_rollouts(
    ax: plt.Axes,
    sample: Dict[str, Any],
    rollouts: Dict[str, Dict[str, Any]],
    title: str,
    points_local: np.ndarray,
) -> None:
    ego_idx, origin, ego_heading, _ = _ego_state(sample)
    relevant_idx = _nearest_or_blocking_agent(sample, rollouts)
    current_xy = _as_array(sample.get("current_xy"))
    valid = _valid_agents(sample)

    _draw_lanes(ax, sample, origin, ego_heading)
    for idx, xy in enumerate(current_xy):
        if idx == ego_idx or not valid[idx]:
            continue
        center = _to_local(xy.reshape(1, 2), origin, ego_heading)[0]
        heading = _local_heading(_agent_heading(sample, idx, ego_heading), ego_heading)
        length, width = _agent_size(sample, idx)
        if idx == relevant_idx:
            _draw_vehicle(ax, center, heading, length, width, "#d95f02", "#b2182b", 1.3, 0.52, 3)
        else:
            _draw_vehicle(ax, center, heading, length, width, "#e69f00", "#8c510a", 0.4, 0.22, 2)

    ego_center = np.zeros(2, dtype=float)
    ego_heading_local = 0.0
    ego_l, ego_w = _agent_size(sample, ego_idx)
    _draw_vehicle(ax, ego_center, ego_heading_local, ego_l, ego_w, "#2166ac", "#053061", 1.0, 0.85, 5)

    for action, info in rollouts.items():
        pts = _to_local(info["trajectory_xy"], origin, ego_heading)
        feasible = bool(info["feasible"])
        color = ACTION_COLORS["feasible" if feasible else "infeasible"]
        lw = 1.6 if feasible else 1.4
        ls = "-" if feasible else "--"
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, alpha=0.95, zorder=6)
        if not feasible and pts.shape[0] > 0:
            t = float(info.get("first_unsafe_time_s", np.nan))
            if np.isfinite(t):
                ax.text(
                    pts[-1, 0],
                    pts[-1, 1],
                    f"{t:.1f}s",
                    fontsize=6.5,
                    color=color,
                    ha="left",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
                    zorder=7,
                )

    if relevant_idx >= 0:
        rel = _to_local(current_xy[relevant_idx].reshape(1, 2), origin, ego_heading)[0]
        ax.add_patch(Circle(rel, radius=1.9, fill=False, edgecolor="#b2182b", lw=1.2, alpha=0.9, zorder=8))

    _set_publication_limits(ax, points_local)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("Forward distance from ego (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    handles = [
        plt.Line2D([0], [0], color=ACTION_COLORS["feasible"], lw=1.8, label="Feasible"),
        plt.Line2D([0], [0], color=ACTION_COLORS["infeasible"], lw=1.5, ls="--", label="Infeasible"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True, framealpha=0.9)


def _format_label(label_id: int, label_name: str) -> str:
    if label_name:
        return f"{label_id}: {label_name}"
    return str(label_id)


def _draw_score_panel(ax: plt.Axes, row: pd.Series, sample_id: str, intended_use: str) -> None:
    ax.axis("off")
    original_id = _safe_int(row, ["original_label_id", "label_id"])
    original_name = _safe_str(row, ["original_label_name", "label_name"])
    action_id = _safe_int(row, ["actionability_label_id"])
    action_name = _safe_str(row, ["actionability_label_name"])

    values = [
        ("sample", sample_id[:8] + "..." + sample_id[-4:]),
        ("case", intended_use),
        ("original", _format_label(original_id, original_name)),
        ("actionability", _format_label(action_id, action_name)),
        ("distance", f"{_safe_float(row, ['current_min_distance_m']):.2f} m"),
        ("TTC", f"{_safe_float(row, ['current_ttc_s']):.2f} s"),
        ("baseline score", f"{_safe_float(row, ['baseline_score']):.3f}"),
        ("enhanced score", f"{_safe_float(row, ['enhanced_score']):.3f}"),
        ("score delta", f"{_safe_float(row, ['score_delta']):+.3f}"),
        ("ASR_cum", f"{_safe_float(row, ['asr_cum_final']):.3f}"),
        ("TTAD", f"{_safe_float(row, ['ttad_s']):.2f} s"),
        ("comfort_ASR", f"{_safe_float(row, ['comfort_asr']):.3f}"),
        ("emergency_ASR", f"{_safe_float(row, ['emergency_asr']):.3f}"),
    ]

    ax.text(
        0.02,
        0.97,
        "Case summary",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    y = 0.86
    for key, value in values:
        ax.text(0.02, y, key, transform=ax.transAxes, fontsize=8, color="#555555", va="top")
        ax.text(0.48, y, value, transform=ax.transAxes, fontsize=8, color="#111111", va="top")
        y -= 0.065


def _render_case(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    sample: Dict[str, Any],
    row: pd.Series,
    sample_id: str,
    cfg: Dict[str, Any],
    intended_use: str,
    horizon_s: float,
    obstacle_mode: str,
    rollout_use_map: bool,
    title_prefix: str = "",
) -> Dict[str, Dict[str, Any]]:
    rollouts = _analyze_rollouts(sample, cfg, horizon_s, obstacle_mode, rollout_use_map)
    _, origin, ego_heading, _ = _ego_state(sample)
    points_local = _collect_plot_points(sample, origin, ego_heading, rollouts)

    panel_a_title = f"{title_prefix}BEV interaction".strip()
    panel_b_title = f"{title_prefix}Candidate action rollouts".strip()
    _draw_current_scene(axes[0], sample, rollouts, panel_a_title, points_local)
    _draw_action_rollouts(axes[1], sample, rollouts, panel_b_title, points_local)
    _draw_score_panel(axes[2], row, sample_id, intended_use)
    fig.tight_layout(pad=0.9)
    return rollouts


def _augment_case_rows(
    cases_df: pd.DataFrame,
    features_df: pd.DataFrame,
    proximity_df: pd.DataFrame,
    action_df: pd.DataFrame,
) -> pd.DataFrame:
    df = cases_df.copy()
    for other, suffix in (
        (features_df, "_feat"),
        (proximity_df, "_prox"),
        (action_df, "_act"),
    ):
        if "sample_id" in other.columns:
            cols = [c for c in other.columns if c == "sample_id" or c not in df.columns]
            df = df.merge(other[cols], on="sample_id", how="left", suffixes=("", suffix))
    return df


def _choose_case_row(df: pd.DataFrame, sample_id: str) -> Optional[pd.Series]:
    sub = df[df["sample_id"].astype(str) == sample_id]
    if sub.empty:
        return None
    # Prefer rows with the requested no-map critical task, then the highest
    # absolute model-score separation.
    if "task" in sub.columns:
        preferred = sub[sub["task"].astype(str).str.contains("critical", case=False, na=False)]
        if not preferred.empty:
            sub = preferred
    if "score_delta" in sub.columns:
        order = sub["score_delta"].astype(float).abs().sort_values(ascending=False).index
        return sub.loc[order[0]]
    return sub.iloc[0]


def _write_single_case(
    sample_id: str,
    sample: Dict[str, Any],
    row: pd.Series,
    cfg: Dict[str, Any],
    out_dir: Path,
    index: int,
    dpi: int,
    horizon_s: float,
    obstacle_mode: str,
    rollout_use_map: bool,
) -> Tuple[Path, Path]:
    intended_use, _ = INTENDED_USES.get(sample_id, ("selected_case", "selected case"))
    png_path = out_dir / f"{index:02d}_{sample_id}_publication_panel.png"
    pdf_path = out_dir / f"{index:02d}_{sample_id}_publication_panel.pdf"

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 4.2),
        gridspec_kw={"width_ratios": [1.35, 1.35, 0.95]},
    )
    _render_case(
        fig,
        axes,
        sample,
        row,
        sample_id,
        cfg,
        intended_use,
        horizon_s,
        obstacle_mode,
        rollout_use_map,
    )
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _write_combined(
    rendered: List[Tuple[str, Dict[str, Any], pd.Series]],
    cfg: Dict[str, Any],
    out_dir: Path,
    dpi: int,
    horizon_s: float,
    obstacle_mode: str,
    rollout_use_map: bool,
) -> Tuple[Optional[Path], Optional[Path]]:
    if not rendered:
        return None, None
    n = len(rendered)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12.8, 3.35 * n),
        gridspec_kw={"width_ratios": [1.32, 1.32, 0.98]},
    )
    if n == 1:
        axes = np.asarray([axes])
    for row_i, (sample_id, sample, row) in enumerate(rendered):
        intended_use, _ = INTENDED_USES.get(sample_id, ("selected_case", "selected case"))
        prefix = f"{chr(65 + row_i)}. "
        _render_case(
            fig,
            axes[row_i],
            sample,
            row,
            sample_id,
            cfg,
            intended_use,
            horizon_s,
            obstacle_mode,
            rollout_use_map,
            title_prefix=prefix,
        )
    fig.subplots_adjust(hspace=0.32, wspace=0.18)
    png_path = out_dir / "combined_publication_cases.png"
    pdf_path = out_dir / "combined_publication_cases.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _metadata_row(sample_id: str, row: pd.Series, png_path: Path, pdf_path: Path) -> Dict[str, Any]:
    intended_use, reason = INTENDED_USES.get(sample_id, ("selected_case", "selected case"))
    original_id = _safe_int(row, ["original_label_id", "label_id"])
    original_name = _safe_str(row, ["original_label_name", "label_name"])
    action_id = _safe_int(row, ["actionability_label_id"])
    action_name = _safe_str(row, ["actionability_label_name"])
    return {
        "sample_id": sample_id,
        "intended_use": intended_use,
        "original_label": _format_label(original_id, original_name),
        "actionability_label": _format_label(action_id, action_name),
        "distance": _safe_float(row, ["current_min_distance_m"]),
        "TTC": _safe_float(row, ["current_ttc_s"]),
        "baseline_score": _safe_float(row, ["baseline_score"]),
        "enhanced_score": _safe_float(row, ["enhanced_score"]),
        "score_delta": _safe_float(row, ["score_delta"]),
        "reason_for_selection": reason,
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    work_dir = _work_dir(cfg)
    out_dir = ensure_dir(work_dir / "results" / "nc_actionability_cases" / args.out_name)

    sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    cases_df = pd.read_csv(args.cases_csv)
    features_df = pd.read_csv(args.features_csv)
    proximity_df = pd.read_csv(args.proximity_labels_csv)
    action_df = pd.read_csv(args.actionability_labels_csv)
    cases_full = _augment_case_rows(cases_df, features_df, proximity_df, action_df)

    metadata_rows: List[Dict[str, Any]] = []
    rendered: List[Tuple[str, Dict[str, Any], pd.Series]] = []
    failures: List[Dict[str, str]] = []

    for idx, sample_id in enumerate(sample_ids, start=1):
        sample_path = work_dir / "samples" / f"{sample_id}.pkl.gz"
        row = _choose_case_row(cases_full, sample_id)
        if row is None:
            failures.append({"sample_id": sample_id, "reason": "sample_id not found in cases CSV"})
            continue
        if not sample_path.exists():
            failures.append({"sample_id": sample_id, "reason": f"sample pickle not found: {sample_path}"})
            continue
        try:
            sample = _load_sample(sample_path)
            png_path, pdf_path = _write_single_case(
                sample_id,
                sample,
                row,
                cfg,
                out_dir,
                idx,
                args.dpi,
                args.horizon_s,
                args.obstacle_mode,
                args.rollout_use_map,
            )
            metadata_rows.append(_metadata_row(sample_id, row, png_path, pdf_path))
            rendered.append((sample_id, sample, row))
            print(f"[ok] {sample_id} -> {png_path.name}, {pdf_path.name}")
        except Exception as exc:
            failures.append({"sample_id": sample_id, "reason": repr(exc)})
            print(f"[failed] {sample_id}: {exc}", file=sys.stderr)

    combined_png, combined_pdf = _write_combined(
        rendered,
        cfg,
        out_dir,
        args.dpi,
        args.horizon_s,
        args.obstacle_mode,
        args.rollout_use_map,
    )
    if combined_png is not None and combined_pdf is not None:
        print(f"[ok] combined -> {combined_png.name}, {combined_pdf.name}")

    meta_path = out_dir / "publication_case_panel_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)
    print(f"[ok] metadata -> {meta_path}")

    if failures:
        fail_path = out_dir / "publication_case_panel_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"[warn] failures -> {fail_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
