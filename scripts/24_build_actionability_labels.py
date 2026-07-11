from __future__ import annotations

import argparse
import gzip
import math
import pickle
import time
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

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ACTION_SPECS = [
    {"name": "keep", "cost": 0.0, "accel": 0.0, "lat_m": 0.0, "comfort": True},
    {"name": "mild_brake", "cost": 0.25, "accel": -2.0, "lat_m": 0.0, "comfort": True},
    {"name": "hard_brake", "cost": 1.0, "accel": -5.0, "lat_m": 0.0, "comfort": False},
    {"name": "left", "cost": 0.5, "accel": 0.0, "lat_m": 3.0, "comfort": True},
    {"name": "right", "cost": 0.5, "accel": 0.0, "lat_m": -3.0, "comfort": True},
    {"name": "brake_left", "cost": 1.25, "accel": -4.0, "lat_m": 3.0, "comfort": False},
    {"name": "brake_right", "cost": 1.25, "accel": -4.0, "lat_m": -3.0, "comfort": False},
]
COMFORT_ACTIONS = {a["name"] for a in ACTION_SPECS if a["comfort"]}
EMERGENCY_ACTIONS = {a["name"] for a in ACTION_SPECS}
LABEL_NAMES = {
    0: "high_actionability",
    1: "reduced_actionability",
    2: "critical_actionability",
    3: "infeasible_or_unavoidable",
}
DIAGNOSTIC_FEATURE_COLS = [
    "sample_id",
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "agent_count",
    "nearest_agent_rel_speed_mps",
    "nearest_agent_closing_speed_mps",
]


def _load_sample(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


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
        a = float(action["accel"])
        if abs(a) < 1e-12:
            long_s = speed0 * t
            long_v = speed0
        else:
            t_stop = max(speed0 / max(-a, 1e-9), 0.0) if a < 0 else np.inf
            tt = min(t, t_stop)
            long_s = max(speed0 * tt + 0.5 * a * tt * tt, 0.0)
            long_v = max(speed0 + a * tt, 0.0)
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
        if len(arr) >= 2:
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


def _action_failure_time(sample: dict, action: dict, horizon_s: float, dt_s: float, obstacle_mode: str, drivable) -> float | None:
    ego = int(sample["ego_index"])
    ego_l, ego_w = np.asarray(sample["current_size_lw"][ego], dtype=float)
    sizes = np.asarray(sample["current_size_lw"], dtype=float)
    poses, headings = _rollout_action(sample, action, horizon_s, dt_s)
    times = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    for step_i, (t, xy, hd) in enumerate(zip(times, poses, headings), start=1):
        if not _map_ok(drivable, xy):
            return float(t)
        ego_quad = _rect_corners(float(xy[0]), float(xy[1]), float(hd), float(ego_l), float(ego_w))
        for j in range(int(sample["agent_count"])):
            if j == ego:
                continue
            obs_xy, obs_hd, valid = _obstacle_pose(sample, j, step_i, float(t), obstacle_mode)
            if not valid or not np.all(np.isfinite(obs_xy)):
                continue
            L, W = sizes[j]
            obs_quad = _rect_corners(float(obs_xy[0]), float(obs_xy[1]), float(obs_hd), float(L), float(W))
            if _quad_overlap(ego_quad, obs_quad):
                return float(t)
    return None


def _time_to_no_action(failure_times: dict[str, float | None], action_names: set[str], horizon_s: float) -> float:
    if any(failure_times[name] is None for name in action_names):
        return float(horizon_s)
    return float(max(float(failure_times[name]) for name in action_names))


def _label_original(comfort_ratio: float, emergency_ratio: float) -> int:
    if emergency_ratio == 0:
        return 3
    if comfort_ratio == 0 and emergency_ratio > 0:
        return 2
    if comfort_ratio < 0.25:
        return 1
    return 0


def _label_moderate(comfort_ratio: float, emergency_ratio: float) -> int:
    if emergency_ratio <= 0.0:
        return 3
    if comfort_ratio <= 0.25 or emergency_ratio <= 0.35:
        return 2
    if comfort_ratio < 0.75 or emergency_ratio < 0.65:
        return 1
    return 0


def _label_strict(comfort_ratio: float, emergency_ratio: float) -> int:
    if emergency_ratio <= 0.0:
        return 3
    if comfort_ratio <= 0.50 or emergency_ratio <= 0.50:
        return 2
    if comfort_ratio < 1.00 or emergency_ratio < 0.85:
        return 1
    return 0


def _label_emergency_sensitive(comfort_ratio: float, emergency_ratio: float) -> int:
    if emergency_ratio <= 0.0:
        return 3
    if emergency_ratio <= 0.25:
        return 2
    if emergency_ratio <= 0.60 or comfort_ratio <= 0.50:
        return 1
    return 0


def _label_comfort_sensitive(comfort_ratio: float, emergency_ratio: float) -> int:
    if emergency_ratio <= 0.0:
        return 3
    if comfort_ratio <= 0.25:
        return 2
    if comfort_ratio <= 0.75:
        return 1
    return 0


def _label_quantile_like(comfort_ratio: float, emergency_ratio: float) -> int:
    if emergency_ratio <= 0.0:
        return 3
    if comfort_ratio <= 0.50 or emergency_ratio <= 0.4286:
        return 2
    if comfort_ratio <= 0.75 or emergency_ratio <= 0.7143:
        return 1
    return 0


LABEL_RULES = {
    "original": _label_original,
    "moderate": _label_moderate,
    "strict": _label_strict,
    "emergency_sensitive": _label_emergency_sensitive,
    "comfort_sensitive": _label_comfort_sensitive,
    "quantile_like": _label_quantile_like,
}


def _label_from_ratios(comfort_ratio: float, emergency_ratio: float, rule: str) -> tuple[int, str]:
    label_id = int(LABEL_RULES[rule](float(comfort_ratio), float(emergency_ratio)))
    return label_id, LABEL_NAMES[label_id]


def _process_one(
    row: pd.Series,
    sample: dict,
    cfg: dict,
    horizon_s: float,
    dt_s: float,
    obstacle_mode: str,
    use_map: bool,
    rule: str,
) -> dict:
    lane_buffer = max(float(cfg.get("bev", {}).get("lane_buffer_m", 2.0)), 3.0)
    drivable = _build_drivable(sample, lane_buffer) if use_map else None
    failure_times = {}
    feasible = {}
    for action in ACTION_SPECS:
        fail_t = _action_failure_time(sample, action, horizon_s, dt_s, obstacle_mode, drivable)
        failure_times[action["name"]] = fail_t
        feasible[action["name"]] = fail_t is None

    comfort_feasible = sum(1 for name in COMFORT_ACTIONS if feasible[name])
    emergency_feasible = sum(1 for name in EMERGENCY_ACTIONS if feasible[name])
    comfort_total = len(COMFORT_ACTIONS)
    emergency_total = len(EMERGENCY_ACTIONS)
    comfort_ratio = comfort_feasible / comfort_total
    emergency_ratio = emergency_feasible / emergency_total
    label_id, label_name = _label_from_ratios(comfort_ratio, emergency_ratio, rule)
    feasible_costs = [float(a["cost"]) for a in ACTION_SPECS if feasible[a["name"]]]
    min_cost = float(min(feasible_costs)) if feasible_costs else np.nan

    return {
        "sample_id": str(row["sample_id"]),
        "scenario_id": str(row.get("scenario_id", sample.get("scenario_id", row["sample_id"]))),
        "original_label_id": int(row["label_id"]),
        "original_label_name": str(row.get("label_name", sample.get("label", {}).get("label_name", ""))),
        "actionability_label_id": int(label_id),
        "actionability_label_name": label_name,
        "rule": rule,
        "actionability_binary_degraded": int(label_id >= 1),
        "actionability_binary_critical": int(label_id >= 2),
        "actionability_binary_infeasible": int(label_id == 3),
        "comfort_feasible_count": int(comfort_feasible),
        "comfort_total_count": int(comfort_total),
        "comfort_feasible_ratio": float(comfort_ratio),
        "emergency_feasible_count": int(emergency_feasible),
        "emergency_total_count": int(emergency_total),
        "emergency_feasible_ratio": float(emergency_ratio),
        "min_required_action_cost": min_cost,
        "time_to_no_comfort_s": _time_to_no_action(failure_times, COMFORT_ACTIONS, horizon_s),
        "time_to_no_emergency_s": _time_to_no_action(failure_times, EMERGENCY_ACTIONS, horizon_s),
        "obstacle_mode": obstacle_mode,
        "horizon_s": float(horizon_s),
        "use_map": bool(use_map),
    }


def _load_diagnostic_features(path_arg: str | None, sample_ids: pd.Series) -> pd.DataFrame | None:
    if not path_arg:
        return None
    path = Path(path_arg)
    if not path.exists():
        raise FileNotFoundError(f"diagnostic features CSV not found: {path}")
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in DIAGNOSTIC_FEATURE_COLS if c in header.columns]
    if "sample_id" not in cols:
        raise ValueError(f"diagnostic features CSV must include sample_id: {path}")
    feat = pd.read_csv(path, usecols=cols)
    feat["sample_id"] = feat["sample_id"].astype(str)
    ids = set(sample_ids.astype(str))
    return feat[feat["sample_id"].isin(ids)].drop_duplicates("sample_id").copy()


def _safe_mean(s: pd.Series) -> float:
    value = s.mean()
    return float(value) if pd.notna(value) else np.nan


def _safe_median(s: pd.Series) -> float:
    value = s.median()
    return float(value) if pd.notna(value) else np.nan


def _summary_rows(labels: pd.DataFrame, diagnostics: pd.DataFrame | None) -> list[dict]:
    rows: list[dict] = []
    n = max(len(labels), 1)
    labels_for_summary = labels.copy()
    labels_for_summary["sample_id"] = labels_for_summary["sample_id"].astype(str)
    if diagnostics is not None and not diagnostics.empty:
        diag = diagnostics.copy()
        diag["sample_id"] = diag["sample_id"].astype(str)
        merged = labels_for_summary.merge(diag, on="sample_id", how="left")
    else:
        merged = labels_for_summary

    dist = labels["actionability_label_id"].value_counts().reindex([0, 1, 2, 3], fill_value=0).sort_index()
    for label_id, count in dist.items():
        rows.append({
            "section": "actionability_label_distribution",
            "metric": "count",
            "actionability_label_id": int(label_id),
            "actionability_label_name": LABEL_NAMES[int(label_id)],
            "count": int(count),
            "fraction": float(count / n),
        })

    for col in [
        "actionability_binary_degraded",
        "actionability_binary_critical",
        "actionability_binary_infeasible",
    ]:
        s = pd.to_numeric(labels[col], errors="coerce")
        rows.append({
            "section": "binary_positive_rate",
            "metric": "positive_rate",
            "variable": col,
            "count": int(s.notna().sum()),
            "positive_count": int(s.fillna(0).sum()),
            "value": _safe_mean(s),
        })

    ct = pd.crosstab(labels["original_label_id"], labels["actionability_label_id"])
    original_ids = sorted(labels["original_label_id"].dropna().astype(int).unique())
    for original_id in original_ids:
        for action_id in [0, 1, 2, 3]:
            value = int(ct.loc[original_id, action_id]) if original_id in ct.index and action_id in ct.columns else 0
            rows.append({
                "section": "original_vs_actionability_crosstab",
                "metric": "count",
                "original_label_id": int(original_id),
                "actionability_label_id": int(action_id),
                "actionability_label_name": LABEL_NAMES[int(action_id)],
                "count": value,
            })

    for col in ["current_min_distance_m", "current_ttc_s"]:
        if col in merged.columns:
            label_s = pd.to_numeric(merged["actionability_label_id"], errors="coerce")
            diag_s = pd.to_numeric(merged[col], errors="coerce")
            ok = label_s.notna() & diag_s.notna()
            corr = label_s[ok].corr(diag_s[ok], method="spearman") if int(ok.sum()) >= 3 else np.nan
            rows.append({
                "section": "spearman",
                "metric": f"actionability_label_id_vs_{col}",
                "variable": col,
                "count": int(ok.sum()),
                "value": float(corr) if pd.notna(corr) else np.nan,
            })

    stat_cols = ["comfort_feasible_ratio", "emergency_feasible_ratio"]
    for col in ["current_min_distance_m", "current_ttc_s"]:
        if col in merged.columns:
            stat_cols.append(col)
    for label_id in [0, 1, 2, 3]:
        sub = merged[merged["actionability_label_id"] == label_id]
        for col in stat_cols:
            s = pd.to_numeric(sub[col], errors="coerce") if col in sub.columns else pd.Series(dtype=float)
            rows.append({
                "section": "per_actionability_label_stats",
                "metric": "mean_median",
                "actionability_label_id": int(label_id),
                "actionability_label_name": LABEL_NAMES[int(label_id)],
                "variable": col,
                "count": int(s.notna().sum()),
                "mean": _safe_mean(s),
                "median": _safe_median(s),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build actionability labels from sample-level kinematic avoidance rollouts.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out-name", default="labels_actionability.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon-s", type=float, default=3.0)
    ap.add_argument("--dt-s", type=float, default=None)
    ap.add_argument("--use-map", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--obstacle-mode", choices=["oracle_future", "cv_current"], default="oracle_future")
    ap.add_argument("--rule", choices=sorted(LABEL_RULES), default="moderate")
    ap.add_argument("--diagnostic-features-csv", default=None)
    args = ap.parse_args()

    started = time.perf_counter()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    dt_s = float(args.dt_s if args.dt_s is not None else cfg.get("labels", {}).get("dt_s", 0.1))
    labels = pd.read_csv(args.labels_csv)
    if args.sample_size is not None:
        sample_n = min(int(args.sample_size), len(labels))
        labels = labels.sample(n=sample_n, random_state=int(args.seed)).copy()
    elif args.limit is not None:
        labels = labels.head(int(args.limit)).copy()

    rows = []
    iterator = labels.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(labels), desc="actionability-labels")
    samples_dir = work / "samples"
    for _, row in iterator:
        sample_id = str(row["sample_id"])
        sample_path = samples_dir / f"{sample_id}.pkl.gz"
        if not sample_path.exists():
            raise FileNotFoundError(f"sample pkl not found: {sample_path}")
        sample = _load_sample(sample_path)
        rows.append(
            _process_one(
                row,
                sample,
                cfg,
                float(args.horizon_s),
                dt_s,
                args.obstacle_mode,
                bool(args.use_map),
                args.rule,
            )
        )

    out_labels = pd.DataFrame(rows)
    out_dir = ensure_dir(work / "labels")
    out_path = out_dir / args.out_name
    out_labels.to_csv(out_path, index=False)

    diagnostics = _load_diagnostic_features(args.diagnostic_features_csv, out_labels["sample_id"])
    summary = pd.DataFrame(_summary_rows(out_labels, diagnostics))
    summary_dir = ensure_dir(work / "results" / "nc_actionability_labels")
    summary_path = summary_dir / f"actionability_label_summary_{Path(args.out_name).stem}.csv"
    summary.to_csv(summary_path, index=False)
    elapsed = time.perf_counter() - started
    print(f"[actionability-labels] wrote {out_path}")
    print(f"[actionability-labels] wrote {summary_path}")
    print(f"[actionability-labels] elapsed={elapsed:.1f}s, samples={len(out_labels)}")
    print(out_labels["actionability_label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
