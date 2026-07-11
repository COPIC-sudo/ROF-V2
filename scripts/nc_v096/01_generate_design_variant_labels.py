#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _utils import append_blockers, load_yaml, output_dir, resolve_path, write_csv

try:
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    LineString = None
    Point = None
    unary_union = None


LABEL_NAMES = {
    0: "high_actionability",
    1: "reduced_actionability",
    2: "critical_actionability",
    3: "infeasible_or_unavoidable",
}

BASE7_ACTIONS = [
    {"name": "keep", "cost": 0.0, "accel": 0.0, "lat_m": 0.0, "comfort": True},
    {"name": "mild_brake", "cost": 0.25, "accel": -2.0, "lat_m": 0.0, "comfort": True},
    {"name": "hard_brake", "cost": 1.0, "accel": -5.0, "lat_m": 0.0, "comfort": False},
    {"name": "left", "cost": 0.5, "accel": 0.0, "lat_m": 3.0, "comfort": True},
    {"name": "right", "cost": 0.5, "accel": 0.0, "lat_m": -3.0, "comfort": True},
    {"name": "brake_left", "cost": 1.25, "accel": -4.0, "lat_m": 3.0, "comfort": False},
    {"name": "brake_right", "cost": 1.25, "accel": -4.0, "lat_m": -3.0, "comfort": False},
]

EXTENDED_ADDITIONS = [
    {"name": "accelerate", "cost": 0.35, "accel": 1.5, "lat_m": 0.0, "comfort": True},
    {"name": "strong_brake", "cost": 1.4, "accel": -7.0, "lat_m": 0.0, "comfort": False},
    {"name": "mild_left", "cost": 0.35, "accel": 0.0, "lat_m": 1.5, "comfort": True},
    {"name": "mild_right", "cost": 0.35, "accel": 0.0, "lat_m": -1.5, "comfort": True},
    {"name": "accelerate_left", "cost": 0.75, "accel": 1.0, "lat_m": 3.0, "comfort": False},
    {"name": "accelerate_right", "cost": 0.75, "accel": 1.0, "lat_m": -3.0, "comfort": False},
    {"name": "strong_brake_left", "cost": 1.6, "accel": -6.0, "lat_m": 3.0, "comfort": False},
    {"name": "strong_brake_right", "cost": 1.6, "accel": -6.0, "lat_m": -3.0, "comfort": False},
]


def action_library(name: str) -> list[dict[str, Any]]:
    if name == "base7":
        return [dict(a) for a in BASE7_ACTIONS]
    if name == "extended":
        return [dict(a) for a in BASE7_ACTIONS + EXTENDED_ADDITIONS]
    raise ValueError(f"unknown action library: {name}")


def load_sample(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def rect_corners(cx: float, cy: float, heading: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(float(heading)), math.sin(float(heading))
    hl, hw = float(length) * 0.5, float(width) * 0.5
    local = np.asarray([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=float)
    rot = np.asarray([[c, -s], [s, c]], dtype=float)
    return local @ rot.T + np.asarray([float(cx), float(cy)], dtype=float)


def quad_overlap(a: np.ndarray, b: np.ndarray) -> bool:
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


def smoothstep(u: float) -> tuple[float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    return 3.0 * u * u - 2.0 * u * u * u, 6.0 * u * (1.0 - u)


def rollout_action(sample: dict[str, Any], action: dict[str, Any], horizon_s: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
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
    poses: list[np.ndarray] = []
    headings: list[float] = []
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
        u, du = smoothstep(t / max(horizon_s, 1e-9))
        lat = float(action["lat_m"]) * u
        lat_v = float(action["lat_m"]) * du / max(horizon_s, 1e-9)
        xy = xy0 + forward * long_s + left * lat
        hd = heading0 + math.atan2(lat_v, max(long_v, 0.2))
        poses.append(xy)
        headings.append(hd)
    return np.asarray(poses, dtype=float), np.asarray(headings, dtype=float)


def build_drivable(sample: dict[str, Any], lane_buffer_m: float):
    if LineString is None or unary_union is None:
        return None
    polys = []
    for line in sample.get("map_lane_centerlines", []) or []:
        arr = np.asarray(line, dtype=float)
        if len(arr) < 2:
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


def map_ok(drivable, xy: np.ndarray) -> bool:
    if drivable is None or Point is None:
        return True
    try:
        return bool(drivable.intersects(Point(float(xy[0]), float(xy[1]))))
    except Exception:
        return True


def cv_pose(sample: dict[str, Any], agent_i: int, t: float) -> tuple[np.ndarray, float]:
    xy = np.asarray(sample["current_xy"][agent_i], dtype=float) + np.asarray(sample["current_vel_xy"][agent_i], dtype=float) * float(t)
    v = np.asarray(sample["current_vel_xy"][agent_i], dtype=float)
    heading = float(sample["current_heading"][agent_i])
    if np.linalg.norm(v) > 0.2:
        heading = float(math.atan2(v[1], v[0]))
    return xy, heading


def obstacle_pose(
    sample: dict[str, Any],
    agent_i: int,
    step_i: int,
    t: float,
    future_handling: str,
    counters: Counter,
) -> tuple[np.ndarray, float, bool]:
    fut_xy = np.asarray(sample.get("future_xy"))
    fut_valid = np.asarray(sample.get("future_valid"))
    fut_heading = np.asarray(sample.get("future_heading"))
    valid = False
    xy = np.asarray([np.nan, np.nan], dtype=float)
    heading = float(sample["current_heading"][agent_i])
    if fut_xy.ndim == 3 and fut_valid.ndim == 2 and agent_i < fut_xy.shape[0] and step_i < fut_xy.shape[1]:
        valid = bool(fut_valid[agent_i, step_i])
        xy = np.asarray(fut_xy[agent_i, step_i], dtype=float)
        if fut_heading.ndim == 2 and step_i < fut_heading.shape[1]:
            heading = float(fut_heading[agent_i, step_i])
    else:
        counters["future_step_out_of_cache"] += 1

    finite = bool(np.all(np.isfinite(xy)) and np.isfinite(heading))
    if valid and finite:
        return xy, heading, True
    counters["invalid_or_nonfinite_future_state"] += 1
    if future_handling == "cv_fallback":
        xy_cv, hd_cv = cv_pose(sample, agent_i, t)
        return xy_cv, hd_cv, bool(np.all(np.isfinite(xy_cv)) and np.isfinite(hd_cv))
    return xy, heading, False


def action_failure_time(
    sample: dict[str, Any],
    action: dict[str, Any],
    horizon_s: float,
    dt_s: float,
    future_handling: str,
    drivable,
    counters: Counter,
) -> float | None:
    ego = int(sample["ego_index"])
    ego_l, ego_w = np.asarray(sample["current_size_lw"][ego], dtype=float)
    sizes = np.asarray(sample["current_size_lw"], dtype=float)
    poses, headings = rollout_action(sample, action, horizon_s, dt_s)
    times = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    for step_i, (t, xy, hd) in enumerate(zip(times, poses, headings), start=1):
        if not map_ok(drivable, xy):
            counters["map_violation"] += 1
            return float(t)
        ego_quad = rect_corners(float(xy[0]), float(xy[1]), float(hd), float(ego_l), float(ego_w))
        for j in range(int(sample["agent_count"])):
            if j == ego:
                continue
            obs_xy, obs_hd, valid = obstacle_pose(sample, j, step_i, float(t), future_handling, counters)
            if not valid or not np.all(np.isfinite(obs_xy)):
                continue
            length, width = sizes[j]
            obs_quad = rect_corners(float(obs_xy[0]), float(obs_xy[1]), float(obs_hd), float(length), float(width))
            if quad_overlap(ego_quad, obs_quad):
                counters["collision_agent"] += 1
                return float(t)
    return None


def label_moderate(comfort_ratio: float, emergency_ratio: float) -> int:
    if emergency_ratio <= 0.0:
        return 3
    if comfort_ratio <= 0.25 or emergency_ratio <= 0.35:
        return 2
    if comfort_ratio < 0.75 or emergency_ratio < 0.65:
        return 1
    return 0


def time_to_no_action(failure_times: dict[str, float | None], names: set[str], horizon_s: float) -> float:
    if any(failure_times[name] is None for name in names):
        return float(horizon_s)
    return float(max(float(failure_times[name]) for name in names))


def process_variant(
    row: pd.Series,
    sample: dict[str, Any],
    variant: dict[str, Any],
    drivable,
    dt_s: float,
    counters: Counter,
) -> dict[str, Any]:
    actions = action_library(str(variant["action_library"]))
    comfort_names = {a["name"] for a in actions if bool(a["comfort"])}
    emergency_names = {a["name"] for a in actions}
    failure_times: dict[str, float | None] = {}
    feasible: dict[str, bool] = {}
    for action in actions:
        ft = action_failure_time(
            sample,
            action,
            float(variant["horizon_s"]),
            dt_s,
            str(variant["future_handling"]),
            drivable,
            counters,
        )
        failure_times[action["name"]] = ft
        feasible[action["name"]] = ft is None
    comfort_feasible = sum(1 for name in comfort_names if feasible[name])
    emergency_feasible = sum(1 for name in emergency_names if feasible[name])
    comfort_ratio = comfort_feasible / max(len(comfort_names), 1)
    emergency_ratio = emergency_feasible / max(len(emergency_names), 1)
    label_id = label_moderate(comfort_ratio, emergency_ratio)
    feasible_costs = [float(a["cost"]) for a in actions if feasible[a["name"]]]
    return {
        "sample_id": str(row["sample_id"]),
        "scenario_id": str(row.get("scenario_id", sample.get("scenario_id", row["sample_id"]))),
        "original_label_id": int(row["label_id"]) if "label_id" in row and pd.notna(row["label_id"]) else "",
        "original_label_name": str(row.get("label_name", "")),
        "actionability_label_id": int(label_id),
        "actionability_label_name": LABEL_NAMES[int(label_id)],
        "comfort_feasible_ratio": float(comfort_ratio),
        "emergency_feasible_ratio": float(emergency_ratio),
        "n_candidates": int(len(actions)),
        "n_feasible_comfort": int(comfort_feasible),
        "n_feasible_emergency": int(emergency_feasible),
        "comfort_total_count": int(len(comfort_names)),
        "emergency_total_count": int(len(emergency_names)),
        "min_required_action_cost": float(min(feasible_costs)) if feasible_costs else np.nan,
        "time_to_no_comfort_s": time_to_no_action(failure_times, comfort_names, float(variant["horizon_s"])),
        "time_to_no_emergency_s": time_to_no_action(failure_times, emergency_names, float(variant["horizon_s"])),
        "horizon_s": float(variant["horizon_s"]),
        "lane_buffer_m": float(variant["lane_buffer_m"]),
        "action_library": str(variant["action_library"]),
        "future_handling": str(variant["future_handling"]),
        "threshold_rule": "moderate",
        "map_constraint": "map_constrained",
        "use_map": True,
        "variant_id": str(variant["variant_id"]),
        "variant_family": str(variant.get("family", "")),
    }


def write_action_manifest(path: Path, library: str) -> None:
    rows = []
    for i, action in enumerate(action_library(library), start=1):
        row = {"order": i, "action_library": library, **action}
        rows.append(row)
    write_csv(path, rows)


def process_sample_job(
    row_dict: dict[str, Any],
    sample_path_text: str,
    variants: list[dict[str, Any]],
    dt_s: float,
) -> dict[str, Any]:
    sid = str(row_dict["sample_id"])
    sample_path = Path(sample_path_text)
    out: dict[str, Any] = {
        "sample_id": sid,
        "rows": {},
        "counters": {str(v["variant_id"]): {} for v in variants},
        "errors": [],
    }
    if not sample_path.exists():
        out["errors"].append({"sample_id": sid, "status": "MISSING_SAMPLE_PKL", "path": str(sample_path)})
        return out
    try:
        sample = load_sample(sample_path)
        row_s = pd.Series(row_dict)
        by_buffer: dict[float, list[dict[str, Any]]] = {}
        for v in variants:
            by_buffer.setdefault(float(v["lane_buffer_m"]), []).append(v)
        drivable_by_buffer = {b: build_drivable(sample, b) for b in by_buffer}
        for b, vlist in by_buffer.items():
            drivable = drivable_by_buffer[b]
            for v in vlist:
                vid = str(v["variant_id"])
                counter = Counter()
                if drivable is None:
                    counter["missing_or_empty_drivable_geometry"] += 1
                row = process_variant(row_s, sample, v, drivable, dt_s, counter)
                out["rows"][vid] = row
                out["counters"][vid] = dict(counter)
    except Exception as exc:
        out["errors"].append({"sample_id": sid, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_endpoint_design_robustness.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--variants", default=None, help="Optional comma-separated variant IDs.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    labels_path = resolve_path(cfg["inputs"]["waymo_proximity_labels_csv"])
    samples_dir = resolve_path(cfg["inputs"]["waymo_samples_dir"])
    dt_s = float(cfg["actionability_labels"]["dt_s"])
    variants = list(cfg["actionability_labels"]["variants"])
    if args.variants:
        wanted = {v.strip() for v in str(args.variants).split(",") if v.strip()}
        variants = [v for v in variants if str(v["variant_id"]) in wanted]
    if not variants:
        raise ValueError("no variants selected")
    if LineString is None or unary_union is None:
        raise RuntimeError("shapely is required for map-constrained label variants")

    labels = pd.read_csv(labels_path)
    labels["sample_id"] = labels["sample_id"].astype(str)
    if args.limit:
        labels = labels.head(int(args.limit)).copy()

    variant_rows: dict[str, list[dict[str, Any]]] = {str(v["variant_id"]): [] for v in variants}
    variant_counters: dict[str, Counter] = {str(v["variant_id"]): Counter() for v in variants}
    missing = 0
    errors: list[dict[str, Any]] = []

    by_buffer: dict[float, list[dict[str, Any]]] = {}
    for v in variants:
        by_buffer.setdefault(float(v["lane_buffer_m"]), []).append(v)

    if int(args.n_jobs) == 1:
        for idx, row in enumerate(labels.itertuples(index=False), start=1):
            sid = str(getattr(row, "sample_id"))
            result = process_sample_job(row._asdict(), str(samples_dir / f"{sid}.pkl.gz"), variants, dt_s)
            errors.extend(result["errors"])
            missing += sum(1 for e in result["errors"] if e.get("status") == "MISSING_SAMPLE_PKL")
            for vid, row_out in result["rows"].items():
                variant_rows[vid].append(row_out)
            for vid, counts in result["counters"].items():
                variant_counters[vid].update(counts)
            if args.progress_every and idx % int(args.progress_every) == 0:
                print(f"[v096-labels] processed {idx}/{len(labels)} elapsed_s={time.perf_counter() - t0:.1f}")
    else:
        jobs = [
            (row._asdict(), str(samples_dir / f"{str(getattr(row, 'sample_id'))}.pkl.gz"))
            for row in labels.itertuples(index=False)
        ]
        results = Parallel(n_jobs=int(args.n_jobs), verbose=10, batch_size=8)(
            delayed(process_sample_job)(row_dict, sample_path_text, variants, dt_s)
            for row_dict, sample_path_text in jobs
        )
        for result in results:
            errors.extend(result["errors"])
            missing += sum(1 for e in result["errors"] if e.get("status") == "MISSING_SAMPLE_PKL")
            for vid, row_out in result["rows"].items():
                variant_rows[vid].append(row_out)
            for vid, counts in result["counters"].items():
                variant_counters[vid].update(counts)

    root = out_dir / "variant_labels"
    for v in variants:
        vid = str(v["variant_id"])
        vdir = root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        csv_path = vdir / f"labels_actionability_{vid}.csv"
        if csv_path.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite existing labels without --force: {csv_path}")
        df = pd.DataFrame(variant_rows[vid])
        df.to_csv(csv_path, index=False)
        write_action_manifest(vdir / "action_library_manifest.csv", str(v["action_library"]))
        manifest = {
            "variant": v,
            "status": "GENERATED",
            "label_csv": str(csv_path),
            "rows": int(len(df)),
            "unique_sample_id": int(df["sample_id"].nunique()) if "sample_id" in df else 0,
            "dt_s": dt_s,
            "source_labels_csv": str(labels_path),
            "samples_dir": str(samples_dir),
            "scientific_definition": "full candidate-action feasibility relabeling; not threshold-only remapping",
            "geometry_reuse": "adapted from scripts/24_build_actionability_labels.py",
            "counters": dict(variant_counters[vid]),
        }
        (vdir / "label_generation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        log = [
            f"variant_id={vid}",
            f"rows={len(df)}",
            f"unique_sample_id={df['sample_id'].nunique() if 'sample_id' in df else 0}",
            f"elapsed_s={time.perf_counter() - t0:.1f}",
            f"counters={dict(variant_counters[vid])}",
            "future_handling skip_invalid skips invalid or out-of-cache oracle-future obstacle states.",
            "future_handling cv_fallback replaces invalid or out-of-cache oracle-future obstacle states with current-state constant-velocity extrapolation.",
        ]
        if float(v["horizon_s"]) > 3.0:
            log.append("caveat=horizon exceeds typical 3.0s sample future cache; out-of-cache obstacle states are counted and handled per future_handling.")
        (vdir / "label_generation_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    if errors:
        write_csv(out_dir / "label_generation_errors.csv", errors)
    if missing or errors:
        append_blockers(
            out_dir,
            [
                {
                    "category": "label_generation",
                    "item": "missing_or_error_samples",
                    "status": "PARTIAL" if variant_rows else "BLOCKED",
                    "details": f"missing={missing}, errors={len(errors)}",
                }
            ],
        )
    print(f"[v096-labels] wrote {len(variants)} variants to {root} elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
