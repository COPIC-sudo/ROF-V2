from __future__ import annotations

import argparse
import gzip
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir

try:
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    LineString = None
    Point = None
    unary_union = None


ACTION_SPECS = [
    {"name": "keep", "accel": 0.0, "lat_m": 0.0},
    {"name": "mild_brake", "accel": -2.0, "lat_m": 0.0},
    {"name": "hard_brake", "accel": -5.0, "lat_m": 0.0},
    {"name": "left", "accel": 0.0, "lat_m": 3.0},
    {"name": "right", "accel": 0.0, "lat_m": -3.0},
    {"name": "brake_left", "accel": -4.0, "lat_m": 3.0},
    {"name": "brake_right", "accel": -4.0, "lat_m": -3.0},
]

UNSAFE_REASONS = [
    "collision_agent",
    "map_violation",
    "initial_overlap",
    "self_collision_or_leakage",
    "oracle_future_collision",
    "cv_current_collision",
    "no_failure",
    "unknown",
]


def _load_sample(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _split_semicolon(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def _safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _safe_int(value, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _rect_corners(cx: float, cy: float, heading: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(float(heading)), math.sin(float(heading))
    hl, hw = float(length) * 0.5, float(width) * 0.5
    local = np.asarray([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=float)
    rot = np.asarray([[c, -s], [s, c]], dtype=float)
    return local @ rot.T + np.asarray([float(cx), float(cy)], dtype=float)


def _quad_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.asarray([-edge[1], edge[0]], dtype=float)
            norm = np.linalg.norm(axis)
            if norm <= 1e-12:
                continue
            axis /= norm
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _smoothstep(u: float) -> tuple[float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    return 3.0 * u * u - 2.0 * u * u * u, 6.0 * u * (1.0 - u)


def _rollout_action(sample: dict, action: dict, horizon_s: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    ego = int(sample["ego_index"])
    xy0 = np.asarray(sample["current_xy"][ego], dtype=float)
    vxy = np.asarray(sample["current_vel_xy"][ego], dtype=float)
    heading0 = float(sample["current_heading"][ego])
    speed0 = float(np.linalg.norm(vxy))
    if speed0 > 0.2:
        heading0 = float(math.atan2(vxy[1], vxy[0]))
    c, s = math.cos(heading0), math.sin(heading0)
    forward = np.asarray([c, s], dtype=float)
    left = np.asarray([-s, c], dtype=float)

    times = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    poses = []
    headings = []
    for t in times:
        accel = float(action["accel"])
        if abs(accel) < 1e-12:
            long_s = speed0 * t
            long_v = speed0
        else:
            t_stop = max(speed0 / max(-accel, 1e-9), 0.0) if accel < 0 else np.inf
            tt = min(t, t_stop)
            long_s = max(speed0 * tt + 0.5 * accel * tt * tt, 0.0)
            long_v = max(speed0 + accel * tt, 0.0)
        u, du = _smoothstep(t / max(horizon_s, 1e-9))
        lat = float(action["lat_m"]) * u
        lat_v = float(action["lat_m"]) * du / max(horizon_s, 1e-9)
        xy = xy0 + forward * long_s + left * lat
        hd = heading0 + math.atan2(lat_v, max(long_v, 0.2))
        poses.append(xy)
        headings.append(hd)
    return np.asarray(poses, dtype=float), np.asarray(headings, dtype=float)


def _build_drivable(sample: dict, lane_buffer_m: float):
    if LineString is None or unary_union is None:
        return None
    lanes = sample.get("map_lane_centerlines", []) or []
    polys = []
    for line in lanes:
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or len(arr) < 2:
            continue
        try:
            polys.append(LineString(arr).buffer(float(lane_buffer_m), cap_style=2, join_style=2))
        except Exception:
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def _map_ok(drivable, xy: np.ndarray) -> bool:
    if drivable is None or Point is None:
        return True
    try:
        return bool(drivable.intersects(Point(float(xy[0]), float(xy[1]))))
    except Exception:
        return True


def _agent_ids(sample: dict) -> list[str] | None:
    ids = sample.get("agent_ids")
    if ids is None:
        return None
    try:
        return [str(x) for x in list(ids)]
    except Exception:
        return None


def _agent_value(sample: dict, key: str, index: int):
    values = sample.get(key)
    if values is None:
        return ""
    try:
        return values[index]
    except Exception:
        return ""


def _self_leakage_indices(sample: dict) -> list[int]:
    ids = _agent_ids(sample)
    if not ids:
        return []
    ego = int(sample["ego_index"])
    if ego >= len(ids):
        return []
    ego_id = ids[ego]
    return [i for i, agent_id in enumerate(ids) if i != ego and agent_id == ego_id]


def _iter_obstacle_indices(sample: dict) -> tuple[list[int], bool]:
    ego = int(sample["ego_index"])
    n_agents = int(sample.get("agent_count", len(sample.get("current_xy", []))))
    leakage = set(_self_leakage_indices(sample))
    indices = [i for i in range(n_agents) if i != ego and i not in leakage]
    return indices, bool(leakage)


def _obstacle_pose(sample: dict, agent_i: int, step_i: int, t: float, obstacle_mode: str) -> tuple[np.ndarray, float, bool]:
    if obstacle_mode == "oracle_future":
        fut_valid = np.asarray(sample["future_valid"])
        fut_xy = np.asarray(sample["future_xy"])
        fut_heading = np.asarray(sample["future_heading"])
        k = min(max(int(step_i), 0), fut_xy.shape[1] - 1)
        valid = bool(fut_valid[agent_i, k])
        return np.asarray(fut_xy[agent_i, k], dtype=float), float(fut_heading[agent_i, k]), valid
    xy = np.asarray(sample["current_xy"][agent_i], dtype=float) + np.asarray(sample["current_vel_xy"][agent_i], dtype=float) * float(t)
    v = np.asarray(sample["current_vel_xy"][agent_i], dtype=float)
    heading = float(sample["current_heading"][agent_i])
    if np.linalg.norm(v) > 0.2:
        heading = float(math.atan2(v[1], v[0]))
    return xy, heading, True


def _blocking_metadata(sample: dict, agent_i: int, ego_xy: np.ndarray, obs_xy: np.ndarray) -> dict:
    ids = _agent_ids(sample)
    agent_id = ids[agent_i] if ids is not None and agent_i < len(ids) else str(agent_i)
    agent_type = _agent_value(sample, "agent_types", agent_i)
    try:
        agent_type = int(agent_type)
    except Exception:
        agent_type = _safe_str(agent_type)
    distance = float(np.linalg.norm(np.asarray(obs_xy, dtype=float) - np.asarray(ego_xy, dtype=float)))
    return {
        "blocking_agent_id": agent_id,
        "blocking_agent_type": agent_type,
        "blocking_agent_distance_m": distance,
    }


def _find_collision(
    sample: dict,
    ego_xy: np.ndarray,
    ego_heading: float,
    step_i: int,
    t: float,
    obstacle_mode: str,
    safety_buffer_m: float,
) -> dict | None:
    ego = int(sample["ego_index"])
    sizes = np.asarray(sample["current_size_lw"], dtype=float)
    ego_l, ego_w = sizes[ego]
    ego_quad = _rect_corners(
        float(ego_xy[0]),
        float(ego_xy[1]),
        float(ego_heading),
        float(ego_l) + 2.0 * safety_buffer_m,
        float(ego_w) + 2.0 * safety_buffer_m,
    )
    obstacle_indices, self_leakage_flag = _iter_obstacle_indices(sample)
    for j in obstacle_indices:
        if t <= 0.0:
            obs_xy = np.asarray(sample["current_xy"][j], dtype=float)
            obs_heading = float(sample["current_heading"][j])
            valid = True
        else:
            obs_xy, obs_heading, valid = _obstacle_pose(sample, j, step_i, float(t), obstacle_mode)
        if not valid or not np.all(np.isfinite(obs_xy)):
            continue
        length, width = sizes[j]
        obs_quad = _rect_corners(
            float(obs_xy[0]),
            float(obs_xy[1]),
            float(obs_heading),
            float(length) + 2.0 * safety_buffer_m,
            float(width) + 2.0 * safety_buffer_m,
        )
        if _quad_overlap(ego_quad, obs_quad):
            out = _blocking_metadata(sample, j, ego_xy, obs_xy)
            out["self_leakage_flag"] = bool(self_leakage_flag)
            return out
    return None


def _collision_reason(obstacle_mode: str) -> str:
    if obstacle_mode == "oracle_future":
        return "oracle_future_collision"
    if obstacle_mode == "cv_current":
        return "cv_current_collision"
    return "collision_agent"


def _map_violation_fraction(poses: np.ndarray, times: np.ndarray, drivable, ignore_initial_s: float) -> float:
    if drivable is None:
        return np.nan
    checked = 0
    violations = 0
    for t, xy in zip(times, poses):
        if float(t) <= float(ignore_initial_s):
            continue
        checked += 1
        if not _map_ok(drivable, xy):
            violations += 1
    if checked == 0:
        return np.nan
    return float(violations / checked)


def _audit_action(
    sample: dict,
    action: dict,
    mode_name: str,
    obstacle_mode: str,
    use_map: bool,
    ignore_initial_s: float,
    horizon_s: float,
    dt_s: float,
    safety_buffer_m: float,
    drivable,
) -> dict:
    ego = int(sample["ego_index"])
    poses, headings = _rollout_action(sample, action, horizon_s, dt_s)
    times = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    current_xy = np.asarray(sample["current_xy"][ego], dtype=float)
    current_heading = float(sample["current_heading"][ego])
    _, self_leakage_flag = _iter_obstacle_indices(sample)
    map_fraction = _map_violation_fraction(poses, times, drivable if use_map else None, ignore_initial_s)

    base = {
        "action_name": action["name"],
        "checked_mode": mode_name,
        "feasible": True,
        "first_unsafe_time_s": np.nan,
        "unsafe_reason_primary": "no_failure",
        "blocking_agent_id": "",
        "blocking_agent_type": "",
        "blocking_agent_distance_m": np.nan,
        "map_violation_fraction": map_fraction,
        "self_leakage_flag": bool(self_leakage_flag),
    }

    if self_leakage_flag and len(_iter_obstacle_indices(sample)[0]) == 0:
        base.update({
            "feasible": False,
            "first_unsafe_time_s": 0.0,
            "unsafe_reason_primary": "self_collision_or_leakage",
        })
        return base

    if ignore_initial_s <= 0.0:
        hit = _find_collision(sample, current_xy, current_heading, 0, 0.0, obstacle_mode, safety_buffer_m)
        if hit is not None:
            base.update(hit)
            base.update({
                "feasible": False,
                "first_unsafe_time_s": 0.0,
                "unsafe_reason_primary": "initial_overlap",
            })
            return base

    for step_i, (t, xy, heading) in enumerate(zip(times, poses, headings), start=1):
        if float(t) <= float(ignore_initial_s):
            continue
        if use_map and drivable is not None and not _map_ok(drivable, xy):
            base.update({
                "feasible": False,
                "first_unsafe_time_s": float(t),
                "unsafe_reason_primary": "map_violation",
            })
            return base
        hit = _find_collision(sample, xy, float(heading), step_i, float(t), obstacle_mode, safety_buffer_m)
        if hit is not None:
            base.update(hit)
            base.update({
                "feasible": False,
                "first_unsafe_time_s": float(t),
                "unsafe_reason_primary": _collision_reason(obstacle_mode),
            })
            return base
    return base


def _modes(args) -> list[dict]:
    modes = [
        {
            "name": "original_oracle_map",
            "obstacle_mode": "oracle_future",
            "use_map": True,
            "ignore_initial_s": 0.0,
        },
        {
            "name": "oracle_map_ignore_initial",
            "obstacle_mode": "oracle_future",
            "use_map": True,
            "ignore_initial_s": float(args.ignore_initial_s),
        },
    ]
    if bool(args.also_no_map):
        modes.append({
            "name": "oracle_no_map",
            "obstacle_mode": "oracle_future",
            "use_map": False,
            "ignore_initial_s": 0.0,
        })
    if bool(args.also_cv_current):
        modes.append({
            "name": "cv_current_map",
            "obstacle_mode": "cv_current",
            "use_map": True,
            "ignore_initial_s": 0.0,
        })
        if bool(args.also_no_map):
            modes.append({
                "name": "cv_current_no_map",
                "obstacle_mode": "cv_current",
                "use_map": False,
                "ignore_initial_s": 0.0,
            })
    return modes


def _read_selected_cases(path: Path, top_per_category: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"selected cases CSV must include sample_id; columns={list(df.columns)}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    if "case_categories" not in df.columns:
        df["case_categories"] = df.get("case_category", "").astype(str)
    if "tasks" not in df.columns:
        df["tasks"] = df.get("task", "").astype(str)
    if top_per_category is None:
        return df.drop_duplicates("sample_id").copy()

    selected_ids: list[str] = []
    seen = set()
    exploded_rows = []
    for _, row in df.iterrows():
        for category in _split_semicolon(row.get("case_categories")):
            exploded_rows.append((category, str(row["sample_id"])))
    for category in sorted({x[0] for x in exploded_rows}):
        ids = [sample_id for cat, sample_id in exploded_rows if cat == category]
        count = 0
        for sample_id in ids:
            if count >= int(top_per_category):
                break
            count += 1
            if sample_id not in seen:
                selected_ids.append(sample_id)
                seen.add(sample_id)
    return df[df["sample_id"].isin(selected_ids)].drop_duplicates("sample_id").copy()


def _dominant_reason(reasons: pd.Series) -> str:
    bad = reasons[reasons.ne("no_failure")]
    if bad.empty:
        return "no_failure"
    counts = bad.value_counts()
    return str(counts.index[0])


def _sample_summary(cases: pd.DataFrame, per_action: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, case_row in cases.iterrows():
        sample_id = str(case_row["sample_id"])
        sub = per_action[per_action["sample_id"].astype(str).eq(sample_id)]
        counts = sub.groupby("checked_mode")["feasible"].sum().to_dict()
        original_reasons = sub[sub["checked_mode"].eq("original_oracle_map")]["unsafe_reason_primary"]
        orig_count = int(counts.get("original_oracle_map", 0))
        ignore_count = int(counts.get("oracle_map_ignore_initial", 0))
        no_map_count = int(counts.get("oracle_no_map", 0))
        cv_map_count = int(counts.get("cv_current_map", 0))
        cv_no_map_count = int(counts.get("cv_current_no_map", 0))
        actionability_label_id = _safe_int(case_row.get("actionability_label_id"), default=-1)
        suspicious = bool(
            actionability_label_id >= 2
            and orig_count == 0
            and max(ignore_count, no_map_count, cv_map_count) >= 3
        )
        rows.append({
            "sample_id": sample_id,
            "scenario_id": _safe_str(case_row.get("scenario_id", sample_id)),
            "case_categories": _safe_str(case_row.get("case_categories", case_row.get("case_category", ""))),
            "tasks": _safe_str(case_row.get("tasks", case_row.get("task", ""))),
            "original_label_id": _safe_int(case_row.get("original_label_id"), default=-1),
            "original_label_name": _safe_str(case_row.get("original_label_name")),
            "actionability_label_id": actionability_label_id,
            "actionability_label_name": _safe_str(case_row.get("actionability_label_name")),
            "total_actions": len(ACTION_SPECS),
            "feasible_actions_original_oracle_map": orig_count,
            "feasible_actions_oracle_map_ignore_initial": ignore_count,
            "feasible_actions_oracle_no_map": no_map_count,
            "feasible_actions_cv_current_map": cv_map_count,
            "feasible_actions_cv_current_no_map": cv_no_map_count,
            "dominant_unsafe_reason_original": _dominant_reason(original_reasons),
            "map_sensitive_flag": bool(orig_count == 0 and no_map_count > 0),
            "initial_time_sensitive_flag": bool(orig_count == 0 and ignore_count > 0),
            "obstacle_prediction_sensitive_flag": bool(orig_count == 0 and cv_map_count > 0),
            "suspicious_label_flag": suspicious,
            "self_leakage_flag": bool(sub["self_leakage_flag"].any()),
        })
    return pd.DataFrame(rows)


def _summary_by_category(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exploded = []
    for _, row in per_sample.iterrows():
        for category in _split_semicolon(row.get("case_categories")):
            out = row.to_dict()
            out["case_category"] = category
            exploded.append(out)
    if not exploded:
        return pd.DataFrame(columns=[
            "case_category",
            "n_samples",
            "fraction_map_sensitive",
            "fraction_initial_time_sensitive",
            "fraction_obstacle_prediction_sensitive",
            "fraction_suspicious_label",
            "fraction_self_leakage",
            "dominant_unsafe_reason_distribution",
        ])
    exp = pd.DataFrame(exploded)
    for category, sub in exp.groupby("case_category", dropna=False):
        n = max(int(len(sub)), 1)
        reason_counts = Counter(sub["dominant_unsafe_reason_original"].astype(str))
        dist = ";".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
        row = {
            "case_category": category,
            "n_samples": int(len(sub)),
            "fraction_map_sensitive": float(sub["map_sensitive_flag"].mean()),
            "fraction_initial_time_sensitive": float(sub["initial_time_sensitive_flag"].mean()),
            "fraction_obstacle_prediction_sensitive": float(sub["obstacle_prediction_sensitive_flag"].mean()),
            "fraction_suspicious_label": float(sub["suspicious_label_flag"].mean()),
            "fraction_self_leakage": float(sub["self_leakage_flag"].mean()),
            "dominant_unsafe_reason_distribution": dist,
        }
        for reason in UNSAFE_REASONS:
            row[f"dominant_reason_{reason}_count"] = int(reason_counts.get(reason, 0))
            row[f"dominant_reason_{reason}_fraction"] = float(reason_counts.get(reason, 0) / n)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("case_category")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit lightweight action feasibility failure reasons for selected actionability cases.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--selected-cases-csv", required=True)
    ap.add_argument("--out-name", default="actionability_case_feasibility_audit_v1")
    ap.add_argument("--horizon-s", type=float, default=3.0)
    ap.add_argument("--obstacle-mode", choices=["oracle_future", "cv_current"], default="oracle_future")
    ap.add_argument("--also-cv-current", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--also-no-map", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ignore-initial-s", type=float, default=0.3)
    ap.add_argument("--safety-buffer-m", type=float, default=0.0)
    ap.add_argument("--top-per-category", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out_dir = ensure_dir(work / "results" / "nc_actionability_cases" / args.out_name)
    samples_dir = work / "samples"
    dt_s = float(cfg.get("labels", {}).get("dt_s", 0.1))
    lane_buffer = max(float(cfg.get("bev", {}).get("lane_buffer_m", 2.0)), 3.0)

    cases = _read_selected_cases(Path(args.selected_cases_csv), args.top_per_category)
    modes = _modes(args)
    per_action_rows = []
    for _, case_row in cases.iterrows():
        sample_id = str(case_row["sample_id"])
        sample_path = samples_dir / f"{sample_id}.pkl.gz"
        if not sample_path.exists():
            raise FileNotFoundError(f"sample pkl not found: {sample_path}")
        sample = _load_sample(sample_path)
        drivable = _build_drivable(sample, lane_buffer)
        for mode in modes:
            mode_drivable = drivable if mode["use_map"] else None
            for action in ACTION_SPECS:
                audit = _audit_action(
                    sample,
                    action,
                    mode["name"],
                    mode["obstacle_mode"],
                    bool(mode["use_map"]),
                    float(mode["ignore_initial_s"]),
                    float(args.horizon_s),
                    dt_s,
                    float(args.safety_buffer_m),
                    mode_drivable,
                )
                audit.update({
                    "sample_id": sample_id,
                    "scenario_id": _safe_str(case_row.get("scenario_id", sample_id)),
                    "case_categories": _safe_str(case_row.get("case_categories", case_row.get("case_category", ""))),
                    "tasks": _safe_str(case_row.get("tasks", case_row.get("task", ""))),
                    "original_label_id": _safe_int(case_row.get("original_label_id"), default=-1),
                    "actionability_label_id": _safe_int(case_row.get("actionability_label_id"), default=-1),
                    "horizon_s": float(args.horizon_s),
                    "ignore_initial_s": float(mode["ignore_initial_s"]),
                    "use_map": bool(mode["use_map"]),
                    "obstacle_mode": mode["obstacle_mode"],
                    "safety_buffer_m": float(args.safety_buffer_m),
                })
                per_action_rows.append(audit)

    per_action = pd.DataFrame(per_action_rows)
    per_sample = _sample_summary(cases, per_action)
    by_category = _summary_by_category(per_sample)

    per_action.to_csv(out_dir / "action_feasibility_audit_per_action.csv", index=False)
    per_sample.to_csv(out_dir / "action_feasibility_audit_per_sample.csv", index=False)
    by_category.to_csv(out_dir / "action_feasibility_audit_summary_by_category.csv", index=False)
    print(f"[feasibility-audit] samples={len(per_sample)}")
    print(f"[feasibility-audit] per_action_rows={len(per_action)}")
    print(f"[feasibility-audit] output_dir={out_dir}")


if __name__ == "__main__":
    main()
