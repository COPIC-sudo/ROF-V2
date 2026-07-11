from __future__ import annotations

import numpy as np

from .geometry import oriented_box_polygon


def compute_future_min_distance(sample: dict, horizon_s: float | None = None):
    ego_idx = int(sample["ego_index"])
    agent_ids = np.asarray(sample["agent_ids"])
    future_xy = np.asarray(sample["future_xy"], dtype=np.float64)
    future_heading = np.asarray(sample["future_heading"], dtype=np.float64)
    future_valid = np.asarray(sample["future_valid"], dtype=bool)
    current_size = np.asarray(sample["current_size_lw"], dtype=np.float64)
    times_s = np.asarray(sample["times_s"], dtype=np.float64)

    ego_size = current_size[ego_idx]
    dmin = np.inf
    t_at = np.nan
    collision_any = False
    conflict_track_id = -1

    max_k = len(times_s)
    if horizon_s is not None:
        max_k = int(np.searchsorted(times_s, float(horizon_s) + 1e-9, side="right"))
        max_k = max(1, min(max_k, len(times_s)))

    for j in range(len(agent_ids)):
        if j == ego_idx:
            continue
        other_size = current_size[j]
        for k in range(1, max_k):
            if not future_valid[ego_idx, k] or not future_valid[j, k]:
                continue
            ego_poly = oriented_box_polygon(future_xy[ego_idx, k, 0], future_xy[ego_idx, k, 1], ego_size[0], ego_size[1], future_heading[ego_idx, k])
            oth_poly = oriented_box_polygon(future_xy[j, k, 0], future_xy[j, k, 1], other_size[0], other_size[1], future_heading[j, k])
            if ego_poly.intersects(oth_poly):
                collision_any = True
                # We use zero clearance for overlap in the main labels.  A signed penetration-depth
                # estimate can be added later, but label logic only needs overlap / non-overlap.
                d = 0.0
            else:
                d = float(ego_poly.distance(oth_poly))
            if d < dmin:
                dmin = d
                t_at = float(times_s[k])
                conflict_track_id = int(agent_ids[j])
    if not np.isfinite(dmin):
        dmin = 9999.0
    if not np.isfinite(t_at):
        t_at = -1.0
    return float(dmin), float(t_at), bool(collision_any), int(conflict_track_id)


def assign_label(sample: dict, cfg: dict) -> dict:
    lab = cfg["labels"]
    dmin, t_at, collision_any, conflict_track_id = compute_future_min_distance(sample, horizon_s=float(lab.get("future_horizon_s", 2.0)))
    emergency_d = float(lab["emergency_distance_m"])
    warning_d = float(lab["warning_distance_m"])
    caution_d = float(lab["caution_distance_m"])
    warning_t = float(lab.get("warning_time_s", 1.0))

    if collision_any or dmin <= emergency_d:
        label_id, label_name = 3, "emergency"
    elif dmin <= warning_d or (0.0 < t_at <= warning_t and dmin <= caution_d):
        label_id, label_name = 2, "warning"
    elif dmin <= caution_d:
        label_id, label_name = 1, "caution"
    else:
        label_id, label_name = 0, "safe"

    d_pos = max(dmin, 0.0)
    d_term = float(np.exp(-d_pos / 2.0))
    t_term = float(np.exp(-max(t_at, 0.0) / 1.0)) if t_at >= 0.0 else 0.0
    collision_term = 1.0 if collision_any else 0.0
    risk_score = min(1.0, 0.50 * d_term + 0.25 * t_term + 0.50 * collision_term)

    return {
        "dmin_future_m": round(dmin, 4),
        "t_at_dmin_s": round(t_at, 4),
        "collision_any": bool(collision_any),
        "label_id": int(label_id),
        "label_name": label_name,
        "risk_score": round(risk_score, 6),
        "conflict_track_id": int(conflict_track_id),
    }
