from __future__ import annotations

from typing import Any

import numpy as np
from waymo_open_dataset.protos import scenario_pb2

from .utils import transform_points_global_to_local, transform_vectors_global_to_local, wrap_to_pi

TYPE_VEHICLE = int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE)


def _roi_contains(x: float, y: float, roi: dict) -> bool:
    return (roi["x_min"] <= x <= roi["x_max"]) and (roi["y_min"] <= y <= roi["y_max"])


def _object_type_name(t: int) -> str:
    if t == int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE):
        return "vehicle"
    if t == int(scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN):
        return "pedestrian"
    if t == int(scenario_pb2.Track.ObjectType.TYPE_CYCLIST):
        return "cyclist"
    return "other"


def _scenario_from_bytes(raw: bytes) -> scenario_pb2.Scenario:
    s = scenario_pb2.Scenario()
    s.ParseFromString(raw)
    return s


def quick_scan_scenario(raw: bytes, cfg: dict) -> dict:
    s = _scenario_from_bytes(raw)
    cur_idx = int(s.current_time_index)
    roi = cfg["roi"]
    min_count = int(cfg["dataset"].get("min_vehicle_count", 3))
    future_steps = int(cfg["dataset"].get("max_future_steps", 20))

    out = {
        "scenario_id": s.scenario_id,
        "keep": False,
        "reason": "unknown",
        "vehicle_count_roi": 0,
        "ego_track_id": -1,
        "has_future": False,
    }

    if s.sdc_track_index < 0 or s.sdc_track_index >= len(s.tracks):
        out["reason"] = "bad_sdc_track_index"
        return out

    ego_track = s.tracks[s.sdc_track_index]
    if ego_track.object_type != TYPE_VEHICLE:
        out["reason"] = "ego_not_vehicle"
        return out
    ego_state = ego_track.states[cur_idx]
    if not ego_state.valid:
        out["reason"] = "ego_current_invalid"
        return out
    if cur_idx + future_steps >= len(ego_track.states):
        out["reason"] = "future_too_short"
        return out

    origin = np.array([ego_state.center_x, ego_state.center_y], dtype=np.float64)
    heading0 = float(ego_state.heading)
    out["ego_track_id"] = int(ego_track.id)
    out["has_future"] = True

    count = 0
    for tr in s.tracks:
        if tr.object_type != TYPE_VEHICLE:
            continue
        st = tr.states[cur_idx]
        if not st.valid:
            continue
        p = np.array([[st.center_x, st.center_y]], dtype=np.float64)
        local_xy = transform_points_global_to_local(p, origin, heading0)[0]
        if _roi_contains(float(local_xy[0]), float(local_xy[1]), roi):
            count += 1

    out["vehicle_count_roi"] = count
    if count < min_count:
        out["reason"] = "too_few_vehicles_in_roi"
        return out

    out["keep"] = True
    out["reason"] = "ok"
    return out


def _extract_map_layers(s: scenario_pb2.Scenario, origin_xy: np.ndarray, ego_heading: float, roi: dict) -> dict:
    lanes = []
    crosswalks = []
    driveways = []

    for mf in s.map_features:
        feat_name = mf.WhichOneof("feature_data")
        if feat_name == "lane":
            pts = np.array([[p.x, p.y] for p in mf.lane.polyline], dtype=np.float64)
            if len(pts) < 2:
                continue
            local = transform_points_global_to_local(pts, origin_xy, ego_heading)
            mask = (
                (local[:, 0] >= roi["x_min"] - 10.0)
                & (local[:, 0] <= roi["x_max"] + 10.0)
                & (local[:, 1] >= roi["y_min"] - 10.0)
                & (local[:, 1] <= roi["y_max"] + 10.0)
            )
            if mask.any():
                lanes.append(local.astype(np.float32))
        elif feat_name == "crosswalk":
            pts = np.array([[p.x, p.y] for p in mf.crosswalk.polygon], dtype=np.float64)
            if len(pts) >= 3:
                local = transform_points_global_to_local(pts, origin_xy, ego_heading)
                crosswalks.append(local.astype(np.float32))
        elif feat_name == "driveway":
            pts = np.array([[p.x, p.y] for p in mf.driveway.polygon], dtype=np.float64)
            if len(pts) >= 3:
                local = transform_points_global_to_local(pts, origin_xy, ego_heading)
                driveways.append(local.astype(np.float32))
    return {
        "map_lane_centerlines": lanes,
        "map_crosswalks": crosswalks,
        "map_driveways": driveways,
    }


def scenario_bytes_to_sample(raw: bytes, cfg: dict, current_time_index: int | None = None, sample_id_suffix: str | None = None) -> dict[str, Any] | None:
    s = _scenario_from_bytes(raw)
    cur_idx = int(s.current_time_index if current_time_index is None else current_time_index)
    future_steps = int(cfg["dataset"].get("max_future_steps", 20))
    roi = cfg["roi"]
    min_count = int(cfg["dataset"].get("min_vehicle_count", 3))

    if s.sdc_track_index < 0 or s.sdc_track_index >= len(s.tracks):
        return None
    ego_track = s.tracks[s.sdc_track_index]
    if ego_track.object_type != TYPE_VEHICLE:
        return None
    ego_state = ego_track.states[cur_idx]
    if not ego_state.valid:
        return None
    if cur_idx + future_steps >= len(ego_track.states):
        return None

    origin = np.array([ego_state.center_x, ego_state.center_y], dtype=np.float64)
    heading0 = float(ego_state.heading)

    agent_ids = []
    agent_types = []
    current_xy = []
    current_vel_xy = []
    current_heading = []
    current_size = []
    future_xy = []
    future_vel = []
    future_heading = []
    future_valid = []
    ego_index = None

    wanted_types = set(str(x).lower() for x in cfg["dataset"].get("object_types", ["vehicle"]))

    for tr_idx, tr in enumerate(s.tracks):
        tname = _object_type_name(int(tr.object_type))
        if tname not in wanted_types:
            continue
        st = tr.states[cur_idx]
        if not st.valid:
            continue
        p_cur = np.array([[st.center_x, st.center_y]], dtype=np.float64)
        local_xy = transform_points_global_to_local(p_cur, origin, heading0)[0]
        if tr_idx != s.sdc_track_index and not _roi_contains(float(local_xy[0]), float(local_xy[1]), roi):
            continue
        v_cur = np.array([[st.velocity_x, st.velocity_y]], dtype=np.float64)
        local_v = transform_vectors_global_to_local(v_cur, heading0)[0]
        current_xy.append(local_xy.astype(np.float32))
        current_vel_xy.append(local_v.astype(np.float32))
        current_heading.append(np.float32(wrap_to_pi(float(st.heading) - heading0)))
        current_size.append(np.array([float(st.length), float(st.width)], dtype=np.float32))
        agent_ids.append(int(tr.id))
        agent_types.append(int(tr.object_type))

        this_future_xy = np.zeros((future_steps + 1, 2), dtype=np.float32)
        this_future_vel = np.zeros((future_steps + 1, 2), dtype=np.float32)
        this_future_heading = np.zeros((future_steps + 1,), dtype=np.float32)
        this_future_valid = np.zeros((future_steps + 1,), dtype=bool)

        for kk in range(future_steps + 1):
            st_k = tr.states[cur_idx + kk]
            if not st_k.valid:
                continue
            p = np.array([[st_k.center_x, st_k.center_y]], dtype=np.float64)
            v = np.array([[st_k.velocity_x, st_k.velocity_y]], dtype=np.float64)
            this_future_xy[kk] = transform_points_global_to_local(p, origin, heading0)[0].astype(np.float32)
            this_future_vel[kk] = transform_vectors_global_to_local(v, heading0)[0].astype(np.float32)
            this_future_heading[kk] = np.float32(wrap_to_pi(float(st_k.heading) - heading0))
            this_future_valid[kk] = True

        future_xy.append(this_future_xy)
        future_vel.append(this_future_vel)
        future_heading.append(this_future_heading)
        future_valid.append(this_future_valid)

        if tr_idx == s.sdc_track_index:
            ego_index = len(agent_ids) - 1

    if ego_index is None:
        return None
    if len(agent_ids) < min_count:
        return None

    map_layers = _extract_map_layers(s, origin, heading0, roi)
    times = np.asarray(s.timestamps_seconds[cur_idx : cur_idx + future_steps + 1], dtype=np.float64)
    times = times - times[0]

    sample_id = str(s.scenario_id) if not sample_id_suffix else f"{s.scenario_id}_{sample_id_suffix}"

    return {
        "sample_id": sample_id,
        "scenario_id": str(s.scenario_id),
        "current_time_index": int(cur_idx),
        "current_time_s_global": float(s.timestamps_seconds[cur_idx]) if cur_idx < len(s.timestamps_seconds) else float(cur_idx),
        "ego_track_id": int(ego_track.id),
        "ego_index": int(ego_index),
        "agent_ids": np.asarray(agent_ids, dtype=np.int64),
        "agent_types": np.asarray(agent_types, dtype=np.int64),
        "agent_count": int(len(agent_ids)),
        "times_s": times.astype(np.float32),
        "current_xy": np.asarray(current_xy, dtype=np.float32),
        "current_vel_xy": np.asarray(current_vel_xy, dtype=np.float32),
        "current_heading": np.asarray(current_heading, dtype=np.float32),
        "current_size_lw": np.asarray(current_size, dtype=np.float32),
        "future_xy": np.asarray(future_xy, dtype=np.float32),
        "future_vel_xy": np.asarray(future_vel, dtype=np.float32),
        "future_heading": np.asarray(future_heading, dtype=np.float32),
        "future_valid": np.asarray(future_valid, dtype=bool),
        **map_layers,
    }
