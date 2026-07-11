from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtbev.external.metrics import normalize_score


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _speed_mps(df: pd.DataFrame) -> pd.Series:
    speed = _num(df, "ego_speed_mps")
    if speed.notna().any():
        return speed
    return _num(df, "ego_speed_kph") / 3.6


def _base_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["sample_id"] = df["sample_id"].astype(str) if "sample_id" in df.columns else pd.Series([str(i) for i in range(len(df))])
    if "scenario_id" in df.columns:
        out["scenario_id"] = df["scenario_id"].astype(str)
    elif "commonroad_scenario_id" in df.columns:
        out["scenario_id"] = df["commonroad_scenario_id"].astype(str)
    else:
        out["scenario_id"] = out["sample_id"]
    if "commonroad_scenario_id" in df.columns:
        out["commonroad_scenario_id"] = df["commonroad_scenario_id"].astype(str)
    return out.reset_index(drop=True)


def _arr(value: Any, ndim: int = 2) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != ndim:
        return np.empty((0, 2), dtype=float)
    return a


def _unit_from_heading_or_velocity(vel: np.ndarray, heading: float | None) -> np.ndarray:
    speed = float(np.linalg.norm(vel))
    if speed > 1e-3:
        return vel / speed
    if heading is not None and np.isfinite(heading):
        return np.asarray([math.cos(float(heading)), math.sin(float(heading))], dtype=float)
    return np.asarray([1.0, 0.0], dtype=float)


def _diag_radius(size_lw: np.ndarray, idx: int) -> float:
    if size_lw.size and idx < len(size_lw):
        length = float(size_lw[idx, 0]) if np.isfinite(size_lw[idx, 0]) else 4.5
        width = float(size_lw[idx, 1]) if size_lw.shape[1] > 1 and np.isfinite(size_lw[idx, 1]) else 1.8
    else:
        length, width = 4.5, 1.8
    return 0.5 * math.hypot(length, width)


def _nearest_lane_margin(sample: dict[str, Any], ego_xy: np.ndarray, lane_half_width_m: float) -> tuple[float, str]:
    lanelets = sample.get("map_lane_centerlines") or []
    best = float("inf")
    for lane in lanelets:
        pts = np.asarray(lane, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
            continue
        d = np.linalg.norm(pts[:, :2] - ego_xy[None, :], axis=1)
        local = float(np.nanmin(d)) if len(d) else float("inf")
        if local < best:
            best = local
    if not np.isfinite(best):
        return float("nan"), "map_lane_centerlines unavailable"
    return float(lane_half_width_m - best), ""


def _current_state_features_from_sample(sample: dict[str, Any], horizon_s: float, dt_s: float, lane_half_width_m: float) -> dict[str, Any]:
    pos = _arr(sample.get("current_xy", []), 2)
    vel = _arr(sample.get("current_vel_xy", []), 2)
    size_lw = _arr(sample.get("current_size_lw", []), 2)
    if len(pos) == 0 or len(vel) != len(pos):
        raise ValueError("missing current_xy/current_vel_xy arrays")
    ego_idx = int(sample.get("ego_index", 0))
    if ego_idx < 0 or ego_idx >= len(pos):
        ego_idx = 0
    headings = sample.get("current_heading") or []
    heading = None
    if ego_idx < len(headings):
        try:
            heading = float(headings[ego_idx])
        except Exception:
            heading = None
    ego_xy = pos[ego_idx]
    ego_vel = vel[ego_idx]
    ego_speed = float(np.linalg.norm(ego_vel))
    h = _unit_from_heading_or_velocity(ego_vel, heading)
    lat_unit = np.asarray([-h[1], h[0]], dtype=float)
    ego_radius = _diag_radius(size_lw, ego_idx)

    nearest_distance = float("inf")
    nearest_closing = float("nan")
    nearest_lateral_speed = float("nan")
    nearest_lateral_clearance = float("nan")
    headway = float("nan")
    headway_closing = float("nan")
    headway_lateral_speed = float("nan")
    min_sep = float("inf")
    min_collision_time = float("nan")
    overlap_integral = 0.0
    steps = max(1, int(round(float(horizon_s) / max(float(dt_s), 1e-6))))
    times = np.linspace(0.0, float(horizon_s), steps + 1)

    for idx in range(len(pos)):
        if idx == ego_idx:
            continue
        rel = pos[idx] - ego_xy
        rel_v = vel[idx] - ego_vel
        center_dist = float(np.linalg.norm(rel))
        other_radius = _diag_radius(size_lw, idx)
        footprint_gap = max(center_dist - ego_radius - other_radius, 0.0)
        if footprint_gap < nearest_distance:
            rel_unit = rel / center_dist if center_dist > 1e-6 else h
            nearest_distance = footprint_gap
            nearest_closing = max(0.0, -float(np.dot(rel_v, rel_unit)))
            nearest_lateral_speed = abs(float(np.dot(rel_v, lat_unit)))
            nearest_lateral_clearance = max(abs(float(np.dot(rel, lat_unit))) - ego_radius - other_radius, 0.0)

        lon = float(np.dot(rel, h))
        lat = float(np.dot(rel, lat_unit))
        longitudinal_gap = lon - ego_radius - other_radius
        lateral_gate = max(3.5, ego_radius + other_radius)
        if longitudinal_gap > 0.0 and abs(lat) <= lateral_gate:
            if not np.isfinite(headway) or longitudinal_gap < headway:
                headway = longitudinal_gap
                headway_closing = max(0.0, -float(np.dot(rel_v, h)))
                headway_lateral_speed = abs(float(np.dot(rel_v, lat_unit)))

        radius = ego_radius + other_radius
        rel_t = rel[None, :] + rel_v[None, :] * times[:, None]
        sep_t = np.linalg.norm(rel_t, axis=1) - radius
        local_min_idx = int(np.nanargmin(sep_t))
        local_min = float(sep_t[local_min_idx])
        if local_min < min_sep:
            min_sep = local_min
        hit = np.where(sep_t <= 0.0)[0]
        if len(hit):
            t_hit = float(times[int(hit[0])])
            if not np.isfinite(min_collision_time) or t_hit < min_collision_time:
                min_collision_time = t_hit
        overlap_integral += float(np.sum(np.clip(1.0 - sep_t / max(radius + 1.0, 1.0), 0.0, 1.0)) * (times[1] - times[0] if len(times) > 1 else 0.0))

    if not np.isfinite(headway):
        headway = nearest_distance
        headway_closing = nearest_closing
        headway_lateral_speed = nearest_lateral_speed
    road_margin, road_reason = _nearest_lane_margin(sample, ego_xy, lane_half_width_m)
    return {
        "ego_speed_mps_current": ego_speed,
        "current_state_min_distance_m": nearest_distance if np.isfinite(nearest_distance) else np.nan,
        "headway_m": headway,
        "nearest_agent_closing_speed_mps": headway_closing if np.isfinite(headway_closing) else nearest_closing,
        "nearest_agent_lateral_speed_mps": headway_lateral_speed if np.isfinite(headway_lateral_speed) else nearest_lateral_speed,
        "current_lateral_clearance_m": nearest_lateral_clearance,
        "cv_min_predicted_separation_3s_m": min_sep if np.isfinite(min_sep) else np.nan,
        "cv_min_collision_time_3s_s": min_collision_time,
        "cv_occupancy_overlap_integral_3s": overlap_integral,
        "road_margin_m": road_margin,
        "road_margin_unavailable_reason": road_reason,
        "current_accel_available": False,
        "current_accel_unavailable_reason": "current acceleration not available in exported sample; recorded future trajectory is forbidden",
        "current_state_json_status": "ok",
        "current_state_json_error": "",
    }


def current_state_kinematics_from_json(
    df: pd.DataFrame,
    horizon_s: float = 3.0,
    dt_s: float = 0.1,
    lane_half_width_m: float = 1.75,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out: dict[str, Any] = {"sample_id": str(row.get("sample_id", ""))}
        path_value = str(row.get("json_gz_path", "") or "")
        if not path_value:
            out.update({"current_state_json_status": "missing", "current_state_json_error": "json_gz_path missing"})
            rows.append(out)
            continue
        try:
            path = Path(os.path.expandvars(os.path.expanduser(path_value)))
            with gzip.open(path, "rt", encoding="utf-8") as f:
                sample = json.load(f)
            out.update(_current_state_features_from_sample(sample, horizon_s, dt_s, lane_half_width_m))
        except Exception as exc:
            out.update({"current_state_json_status": "failed", "current_state_json_error": f"{type(exc).__name__}: {exc}"})
        rows.append(out)
    return pd.DataFrame(rows)


def commonroad_crime_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = _base_ids(df)
    distance = _num(df, "headway_m")
    distance = distance.where(distance.notna(), _num(df, "current_min_distance_m"))
    speed = _speed_mps(df).where(_speed_mps(df).notna(), _num(df, "ego_speed_mps_current")).clip(lower=0.0)
    closing = _num(df, "nearest_agent_closing_speed_mps").clip(lower=0.0)
    ttc = _num(df, "current_ttc_s")
    ttc = ttc.where(ttc >= 0.0, _num(df, "cv_min_collision_time_3s_s"))
    ttc = ttc.where(ttc >= 0.0, np.nan)
    lateral_speed = _num(df, "nearest_agent_lateral_speed_mps", 0.0).abs()
    lateral_clearance = _num(df, "current_lateral_clearance_m")
    out["HW_m"] = distance
    out["THW_s"] = distance / speed.replace(0.0, np.nan)
    out["TTC_s"] = ttc
    out["HW"] = out["HW_m"]
    out["THW"] = out["THW_s"]
    out["TTC"] = out["TTC_s"]
    out["HW_inverse"] = -out["HW_m"]
    out["THW_inverse"] = -out["THW_s"]
    out["TTC_inverse_crime"] = -out["TTC_s"]
    out["TTR_s"] = np.nan
    out["TTB_s"] = np.nan
    out["TTS_s"] = np.nan
    out["TTR"] = out["TTR_s"]
    out["TTB"] = out["TTB_s"]
    out["TTS"] = out["TTS_s"]
    out["TTR_available"] = False
    out["TTB_available"] = False
    out["TTS_available"] = False
    unavailable = "CommonRoad-CriMe maneuver solver not available in this environment; recorded future and planner survival are forbidden"
    out["TTR_unavailable_reason"] = unavailable
    out["TTB_unavailable_reason"] = unavailable
    out["TTS_unavailable_reason"] = unavailable
    out["ALongReq_mps2"] = (closing**2) / (2.0 * distance.clip(lower=0.1))
    out["ALatReq_mps2"] = (lateral_speed**2) / (2.0 * lateral_clearance.clip(lower=0.1))
    out["ALongReq"] = out["ALongReq_mps2"]
    out["ALatReq"] = out["ALatReq_mps2"]
    risk_parts = [
        normalize_score(out["HW_m"], invert=True),
        normalize_score(out["THW_s"], invert=True),
        normalize_score(out["TTC_s"], invert=True),
        normalize_score(out["ALongReq_mps2"]),
        normalize_score(out["ALatReq_mps2"]),
    ]
    out["commonroad_crime_risk_score"] = pd.concat(risk_parts, axis=1).mean(axis=1, skipna=True)
    out["baseline_family"] = "commonroad_crime_style"
    out["recorded_future_access"] = False
    out["candidate_action_survival_access"] = False
    out["endpoint_intermediate_access"] = False
    return out.reset_index(drop=True)


def rss_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = _base_ids(df)
    distance = _num(df, "headway_m").where(_num(df, "headway_m").notna(), _num(df, "current_min_distance_m"))
    ego_speed = _speed_mps(df).where(_speed_mps(df).notna(), _num(df, "ego_speed_mps_current")).clip(lower=0.0)
    closing = _num(df, "nearest_agent_closing_speed_mps", 0.0).fillna(0.0).clip(lower=0.0)
    lead_speed = (ego_speed - closing).clip(lower=0.0)
    response_time = 1.0
    a_max = 2.0
    b_min = 4.0
    b_max = 8.0
    safe_long = ego_speed * response_time + 0.5 * a_max * response_time**2 + ((ego_speed + response_time * a_max) ** 2) / (2.0 * b_min) - (lead_speed**2) / (2.0 * b_max)
    out["rss_longitudinal_safe_distance_m"] = safe_long.clip(lower=0.0)
    out["rss_longitudinal_margin_m"] = distance - out["rss_longitudinal_safe_distance_m"]
    out["rss_longitudinal_margin"] = out["rss_longitudinal_margin_m"]
    lateral_clearance = _num(df, "current_lateral_clearance_m", 3.5)
    out["rss_lateral_safe_distance_m"] = 1.0 + 0.5 * _num(df, "nearest_agent_lateral_speed_mps", 0.0).abs()
    out["rss_lateral_margin_m"] = lateral_clearance - out["rss_lateral_safe_distance_m"]
    out["rss_lateral_margin"] = out["rss_lateral_margin_m"]
    out["rss_longitudinal_margin_inverse"] = -out["rss_longitudinal_margin_m"]
    out["rss_lateral_margin_inverse"] = -out["rss_lateral_margin_m"]
    out["rss_danger_score"] = pd.concat(
        [
            normalize_score(-out["rss_longitudinal_margin_m"]),
            normalize_score(-out["rss_lateral_margin_m"]),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["baseline_family"] = "rss_style_margin"
    out["not_full_rss_stack"] = True
    out["rss_scope_note"] = "RSS-style current-state safe-distance margins only; not a complete RSS stack."
    out["recorded_future_access"] = False
    out["candidate_action_survival_access"] = False
    out["endpoint_intermediate_access"] = False
    return out.reset_index(drop=True)


def drivability_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = _base_ids(df)
    speed = _speed_mps(df).where(_speed_mps(df).notna(), _num(df, "ego_speed_mps_current")).clip(lower=0.0)
    distance = _num(df, "headway_m").where(_num(df, "headway_m").notna(), _num(df, "current_min_distance_m"))
    ttc = _num(df, "cv_min_collision_time_3s_s")
    ttc = ttc.where(ttc >= 0.0, _num(df, "current_ttc_s").where(_num(df, "current_ttc_s") >= 0.0, np.nan))
    stopping_distance = speed**2 / (2.0 * 7.0)
    out["emergency_brake_margin_m"] = distance - stopping_distance
    out["emergency_brake_feasible"] = out["emergency_brake_margin_m"] >= 0.0
    out["emergency_brake_infeasible_score"] = (~out["emergency_brake_feasible"]).astype(float)
    road_margin = _num(df, "road_margin_m")
    out["min_road_margin_keep_lane"] = road_margin
    out["min_road_margin_keep_lane_unavailable_reason"] = df.get(
        "road_margin_unavailable_reason",
        pd.Series("", index=df.index),
    )
    out["min_collision_time_keep_lane"] = ttc
    out["keep_lane_cv_feasible"] = (ttc.isna() | (ttc > 3.0)) & (road_margin.fillna(0.0) > -0.25)
    out["keep_lane_cv_infeasible_score"] = (~out["keep_lane_cv_feasible"]).astype(float)
    out["min_collision_time_keep_lane_inverse"] = -out["min_collision_time_keep_lane"]
    out["min_road_margin_keep_lane_inverse"] = -out["min_road_margin_keep_lane"]
    out["full_planner_feasibility_count_secondary"] = _num(df, "feasible_candidate_count", np.nan)
    out["full_planner_feasibility_count_note"] = "secondary diagnostic only; not used in drivability_risk_score"
    out["drivability_risk_score"] = pd.concat(
        [
            normalize_score(-out["emergency_brake_margin_m"]),
            normalize_score(-out["min_road_margin_keep_lane"]),
            normalize_score(out["min_collision_time_keep_lane"], invert=True),
            normalize_score((~out["keep_lane_cv_feasible"]).astype(float)),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["baseline_family"] = "drivability_current_state"
    out["primary_uses_full_planner_count"] = False
    out["recorded_future_access"] = False
    out["candidate_action_survival_access"] = False
    out["endpoint_intermediate_access"] = False
    return out.reset_index(drop=True)


def forecast_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = _base_ids(df)
    distance = _num(df, "headway_m").where(_num(df, "headway_m").notna(), _num(df, "current_min_distance_m"))
    closing = _num(df, "nearest_agent_closing_speed_mps", 0.0).fillna(0.0).clip(lower=0.0)
    min_sep_3s = _num(df, "cv_min_predicted_separation_3s_m")
    min_sep_3s = min_sep_3s.where(min_sep_3s.notna(), distance - closing * 3.0)
    overlap = _num(df, "cv_occupancy_overlap_integral_3s")
    if overlap.isna().all():
        overlap = normalize_score(_num(df, "nearby_agent_count_10m", 0.0))
    out["cv_forecast_collision_risk"] = normalize_score(-min_sep_3s)
    out["ca_forecast_collision_risk"] = np.nan
    out["ca_forecast_available"] = False
    out["ca_forecast_unavailable_reason"] = "current acceleration unavailable; recorded future trajectory is forbidden"
    out["occupancy_overlap_integral_3s"] = overlap
    out["minimum_predicted_separation_3s_m"] = min_sep_3s
    out["minimum_predicted_separation_over_3s_m"] = min_sep_3s
    out["minimum_predicted_separation_3s"] = min_sep_3s
    out["minimum_predicted_separation_3s_inverse"] = -min_sep_3s
    out["forecast_risk_score"] = pd.concat(
        [
            out["cv_forecast_collision_risk"],
            normalize_score(out["occupancy_overlap_integral_3s"]),
            normalize_score(out["minimum_predicted_separation_3s_m"], invert=True),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["baseline_family"] = "current_state_cv_forecast"
    out["recorded_future_access"] = False
    out["candidate_action_survival_access"] = False
    out["endpoint_intermediate_access"] = False
    return out.reset_index(drop=True)


def compute_all_baseline_scores(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "commonroad_crime_scores": commonroad_crime_scores(df),
        "rss_scores": rss_scores(df),
        "drivability_baseline_scores": drivability_baseline_scores(df),
        "forecast_risk_scores": forecast_risk_scores(df),
    }
