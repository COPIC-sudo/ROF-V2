#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rtbev.config import load_config  # noqa: E402
from rtbev import pipeline as rtpipe  # noqa: E402
from _utils import load_yaml, output_dir, resolve_path, sha256, write_csv  # noqa: E402

try:
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    LineString = None
    Point = None
    unary_union = None


BASE7_ACTIONS = [
    {"name": "keep", "cost": 0.0, "accel": 0.0, "lat_m": 0.0, "comfort": True, "family": "keep"},
    {"name": "mild_brake", "cost": 0.25, "accel": -2.0, "lat_m": 0.0, "comfort": True, "family": "brake"},
    {"name": "hard_brake", "cost": 1.0, "accel": -5.0, "lat_m": 0.0, "comfort": False, "family": "hard_brake"},
    {"name": "left", "cost": 0.5, "accel": 0.0, "lat_m": 3.0, "comfort": True, "family": "left"},
    {"name": "right", "cost": 0.5, "accel": 0.0, "lat_m": -3.0, "comfort": True, "family": "right"},
    {"name": "brake_left", "cost": 1.25, "accel": -4.0, "lat_m": 3.0, "comfort": False, "family": "brake_left"},
    {"name": "brake_right", "cost": 1.25, "accel": -4.0, "lat_m": -3.0, "comfort": False, "family": "brake_right"},
]
EXTENDED_ADDITIONS = [
    {"name": "accelerate", "cost": 0.35, "accel": 1.5, "lat_m": 0.0, "comfort": True, "family": "accelerate"},
    {"name": "strong_brake", "cost": 1.4, "accel": -7.0, "lat_m": 0.0, "comfort": False, "family": "hard_brake"},
    {"name": "mild_left", "cost": 0.35, "accel": 0.0, "lat_m": 1.5, "comfort": True, "family": "left"},
    {"name": "mild_right", "cost": 0.35, "accel": 0.0, "lat_m": -1.5, "comfort": True, "family": "right"},
    {"name": "accelerate_left", "cost": 0.75, "accel": 1.0, "lat_m": 3.0, "comfort": False, "family": "left"},
    {"name": "accelerate_right", "cost": 0.75, "accel": 1.0, "lat_m": -3.0, "comfort": False, "family": "right"},
    {"name": "strong_brake_left", "cost": 1.6, "accel": -6.0, "lat_m": 3.0, "comfort": False, "family": "brake_left"},
    {"name": "strong_brake_right", "cost": 1.6, "accel": -6.0, "lat_m": -3.0, "comfort": False, "family": "brake_right"},
]

CURRENT_REUSE = [
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "agent_count",
    "nearest_agent_rel_speed_mps",
    "nearest_agent_closing_speed_mps",
    "ttc_closing_speed_mps",
    "nearby_agent_count_10m",
    "nearby_agent_count_20m",
]
CV_FIELDS = ["cv_rcr", "cv_rfr_drv", "cv_c_time", "cv_gtoa_norm_union", "cv_oce_norm", "cv_c_density", "cv_max_overlap_count"]
TEMPORAL_FIELDS = ["ttad_s", "time_to_first_conflict_s", "early_blocking_ratio", "collapse_rate_max_per_s", "collapse_rate_mean_per_s"]


def action_library(name: str) -> list[dict[str, Any]]:
    if name == "base7":
        return [dict(a) for a in BASE7_ACTIONS]
    if name == "extended":
        return [dict(a) for a in BASE7_ACTIONS + EXTENDED_ADDITIONS]
    raise ValueError(f"unknown action_library={name}")


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


def make_variant_cfg(base_cfg: dict[str, Any], horizon_s: float, lane_buffer_m: float) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    dt_s = float(cfg.get("labels", {}).get("dt_s", 0.1))
    cfg.setdefault("bev", {})["lane_buffer_m"] = float(lane_buffer_m)
    cfg.setdefault("dataset", {})["max_future_steps"] = int(round(float(horizon_s) / dt_s))
    cfg.setdefault("tube", {})["horizon_s"] = float(horizon_s)
    cfg.setdefault("tube", {})["query_dt_s"] = dt_s
    cfg.setdefault("runtime", {})["gpu_rasterization"] = False
    return cfg


def compute_cv_features(
    sample: dict[str, Any],
    cfg: dict[str, Any],
    horizon_s: float,
    dt_s: float,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    lane_mask: np.ndarray | None = None,
) -> dict[str, float]:
    xs, ys, xx, yy = grid if grid is not None else rtpipe.make_grid(cfg)
    _ = xs, ys
    times = np.arange(dt_s, float(horizon_s) + 1e-9, dt_s, dtype=float)
    weights = np.exp(-times / float(cfg.get("metrics", {}).get("tau_time_s", 1.0)))
    if lane_mask is None:
        lane_mask = rtpipe._make_lane_mask(sample, cfg, xx, yy)
    ego, oth, cnt = rtpipe._cv_masks_cpu(sample, times, xx, yy, cfg)
    feats, _maps = rtpipe._basic_reachability_metrics("cv_", ego, oth, cnt, lane_mask, times, weights, float(cfg["bev"]["resolution_m"]) ** 2, cfg)
    out = {field: feats.get(field, np.nan) for field in CV_FIELDS}
    return {k: (round(float(v), 6) if np.isfinite(float(v)) else np.nan) for k, v in out.items()}


def compute_temporal_features(
    sample: dict[str, Any],
    cfg: dict[str, Any],
    horizon_s: float,
    lane_buffer_m: float,
    library_name: str,
    dt_s: float,
    drivable=None,
) -> dict[str, float]:
    actions = action_library(library_name)
    times = np.arange(dt_s, float(horizon_s) + 1e-9, dt_s, dtype=float)
    safe_slice = np.ones((len(actions), len(times)), dtype=bool)
    if drivable is None:
        drivable = build_drivable(sample, lane_buffer_m)
    ego = int(sample["ego_index"])
    ego_l, ego_w = np.asarray(sample["current_size_lw"][ego], dtype=float)
    sizes = np.asarray(sample["current_size_lw"], dtype=float)
    for m, action in enumerate(actions):
        poses, headings = rollout_action(sample, action, horizon_s, dt_s)
        for k, (t, xy, hd) in enumerate(zip(times, poses, headings)):
            if not map_ok(drivable, xy):
                safe_slice[m, k] = False
                continue
            ego_quad = rect_corners(float(xy[0]), float(xy[1]), float(hd), float(ego_l), float(ego_w))
            for j in range(int(sample["agent_count"])):
                if j == ego:
                    continue
                obs_xy, obs_hd = cv_pose(sample, j, float(t))
                if not np.all(np.isfinite(obs_xy)):
                    continue
                length, width = sizes[j]
                obs_quad = rect_corners(float(obs_xy[0]), float(obs_xy[1]), float(obs_hd), float(length), float(width))
                if quad_overlap(ego_quad, obs_quad):
                    safe_slice[m, k] = False
                    break
    safe_cum = np.minimum.accumulate(safe_slice.astype(np.uint8), axis=1).astype(bool)
    prim = {
        "action_family": np.asarray([a["family"] for a in actions], dtype=str),
        "action_cost": np.asarray([float(a["cost"]) for a in actions], dtype=float),
    }
    weights = np.ones(len(actions), dtype=float)
    feats = rtpipe._summarize_actionability(safe_slice, safe_cum, weights, times, prim, cfg)
    out = {field: feats.get(field, np.nan) for field in TEMPORAL_FIELDS}
    return {k: (round(float(v), 6) if np.isfinite(float(v)) else np.nan) for k, v in out.items()}


def process_sample_job(row: dict[str, Any], sample_path_text: str, variant_specs: list[dict[str, Any]], base_cfg: dict[str, Any], dt_s: float) -> dict[str, Any]:
    sid = str(row["sample_id"])
    sample_path = Path(sample_path_text)
    if not sample_path.exists():
        return {"sample_id": sid, "error": f"missing sample: {sample_path}", "rows": {}}
    try:
        sample = load_sample(sample_path)
        cv_cache: dict[tuple[float, float], dict[str, float]] = {}
        temporal_cache: dict[tuple[float, float, str], dict[str, float]] = {}
        grid = rtpipe.make_grid(base_cfg)
        lane_cache: dict[float, np.ndarray] = {}
        drivable_cache: dict[float, Any] = {}
        rows = {}
        for v in variant_specs:
            vid = str(v["variant_id"])
            h = float(v["horizon_s"])
            b = float(v["lane_buffer_m"])
            lib = str(v["action_library"])
            cfg_v = make_variant_cfg(base_cfg, h, b)
            cv_key = (h, b)
            if cv_key not in cv_cache:
                if b not in lane_cache:
                    lane_cache[b] = rtpipe._make_lane_mask(sample, cfg_v, grid[2], grid[3])
                cv_cache[cv_key] = compute_cv_features(sample, cfg_v, h, dt_s, grid=grid, lane_mask=lane_cache[b])
            temp_key = (h, b, lib)
            if temp_key not in temporal_cache:
                if b not in drivable_cache:
                    drivable_cache[b] = build_drivable(sample, b)
                temporal_cache[temp_key] = compute_temporal_features(sample, cfg_v, h, b, lib, dt_s, drivable=drivable_cache[b])
            out = {
                "sample_id": sid,
                "scenario_id": str(row.get("scenario_id", sample.get("scenario_id", sid))),
                "variant_id": vid,
                "horizon_s": h,
                "lane_buffer_m": b,
                "action_library": lib,
                "future_handling": str(v["future_handling"]),
                "feature_alignment_status": "ALIGNED_LABEL_AND_FEATURE_VARIANT",
            }
            for col in CURRENT_REUSE:
                out[col] = row.get(col, np.nan)
            out.update(cv_cache[cv_key])
            out.update(temporal_cache[temp_key])
            rows[vid] = out
        return {"sample_id": sid, "error": "", "rows": rows}
    except Exception as exc:
        return {"sample_id": sid, "error": f"{type(exc).__name__}: {exc}", "rows": {}}


def load_variant_specs(manifest: pd.DataFrame, cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    wanted = set(cfg["aligned_features"]["variants"])
    specs = []
    for row in manifest.to_dict("records"):
        vid = str(row["variant_id"])
        if vid not in wanted:
            continue
        label_csv = Path(str(row["label_csv"]))
        if not label_csv.exists():
            label_csv = out_dir.parent / "nc_v096_endpoint_design_robustness" / "variant_labels" / vid / f"labels_actionability_{vid}.csv"
        specs.append(
            {
                "variant_id": vid,
                "family": row.get("family", ""),
                "horizon_s": float(row["horizon_s"]),
                "lane_buffer_m": float(row["lane_buffer_m"]),
                "action_library": str(row["action_library"]),
                "future_handling": str(row["future_handling"]),
                "label_csv": str(label_csv),
            }
        )
    if len(specs) != len(wanted):
        got = {s["variant_id"] for s in specs}
        raise ValueError(f"variant manifest mismatch missing={sorted(wanted-got)}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v097/nc_v097_aligned_feature_robustness.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    feature_root = out_dir / "aligned_features"
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve_path(cfg["inputs"]["v096_variant_manifest_csv"]))
    specs = load_variant_specs(manifest, cfg, out_dir)
    ref_cols = ["sample_id", "scenario_id"] + CURRENT_REUSE
    ref = pd.read_csv(resolve_path(cfg["inputs"]["reference_features_csv"]), usecols=lambda c: c in set(ref_cols))
    ref["sample_id"] = ref["sample_id"].astype(str)
    if args.max_samples:
        ref = ref.sort_values("sample_id").head(int(args.max_samples)).copy()
    output_paths = [feature_root / s["variant_id"] / f"features_aligned_{s['variant_id']}.csv" for s in specs]
    if all(p.exists() for p in output_paths) and not args.force:
        print("[v097-features] all aligned feature files already exist; use --force to regenerate")
        return
    base_cfg = load_config(str(resolve_path(cfg["inputs"]["base_config"])))
    dt_s = float(cfg["aligned_features"]["dt_s"])
    samples_dir = resolve_path(cfg["inputs"]["waymo_samples_dir"])
    jobs = [(row._asdict(), str(samples_dir / f"{str(getattr(row, 'sample_id'))}.pkl.gz")) for row in ref.itertuples(index=False)]
    if int(args.n_jobs) == 1:
        results = [process_sample_job(row, path, specs, base_cfg, dt_s) for row, path in jobs]
    else:
        results = Parallel(n_jobs=int(args.n_jobs), verbose=10, batch_size=4)(
            delayed(process_sample_job)(row, path, specs, base_cfg, dt_s) for row, path in jobs
        )
    errors = [r for r in results if r.get("error")]
    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for res in results:
        for vid, row in res.get("rows", {}).items():
            rows_by_variant[vid].append(row)
    audit = []
    checksum_rows = []
    for spec in specs:
        vid = spec["variant_id"]
        vdir = feature_root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / f"features_aligned_{vid}.csv"
        df = pd.DataFrame(rows_by_variant[vid])
        df.to_csv(path, index=False)
        audit.append(
            {
                **spec,
                "feature_csv": str(path),
                "rows": int(len(df)),
                "unique_sample_id": int(df["sample_id"].nunique()) if "sample_id" in df else 0,
                "current_state_fields": ";".join(CURRENT_REUSE),
                "current_state_action": "reused_from_reference_features",
                "cv_fields": ";".join(CV_FIELDS),
                "cv_action": "regenerated_current_state_cv_occupancy",
                "strict_temporal_fields": ";".join(TEMPORAL_FIELDS),
                "strict_temporal_action": "regenerated_candidate_action_cv_survival",
                "elapsed_s_total": time.perf_counter() - t0,
            }
        )
        checksum_rows.append({"variant_id": vid, "artifact": "aligned_feature_csv", "path": str(path), "sha256": sha256(path), "rows": int(len(df))})
    if errors:
        write_csv(out_dir / "aligned_feature_generation_errors.csv", errors)
    write_csv(out_dir / "aligned_feature_generation_audit.csv", audit)
    checksum_rows.append({"variant_id": "reference_features_input", "artifact": "reference_features_csv", "path": str(resolve_path(cfg["inputs"]["reference_features_csv"])), "sha256": sha256(resolve_path(cfg["inputs"]["reference_features_csv"])), "rows": len(ref)})
    write_csv(out_dir / "aligned_feature_checksum_manifest.csv", checksum_rows)

    comparison = []
    ref_full = pd.read_csv(resolve_path(cfg["inputs"]["reference_features_csv"]), usecols=["sample_id"] + CV_FIELDS + TEMPORAL_FIELDS)
    ref_full["sample_id"] = ref_full["sample_id"].astype(str)
    for spec in specs:
        vid = spec["variant_id"]
        aligned = pd.read_csv(feature_root / vid / f"features_aligned_{vid}.csv")
        merged = aligned.merge(ref_full, on="sample_id", suffixes=("_aligned", "_reference"), how="inner")
        for field in CV_FIELDS + TEMPORAL_FIELDS:
            a = pd.to_numeric(merged[f"{field}_aligned"], errors="coerce")
            r = pd.to_numeric(merged[f"{field}_reference"], errors="coerce")
            diff = (a - r).abs()
            comparison.append(
                {
                    "variant_id": vid,
                    "feature": field,
                    "n": int(diff.notna().sum()),
                    "mean_abs_diff": float(diff.mean()) if diff.notna().any() else np.nan,
                    "max_abs_diff": float(diff.max()) if diff.notna().any() else np.nan,
                    "same_as_reference_fraction": float((diff.fillna(np.inf) <= 1e-12).mean()) if len(diff) else np.nan,
                }
            )
    write_csv(out_dir / "aligned_vs_reference_feature_comparison.csv", comparison)
    print(f"[v097-features] wrote {len(specs)} aligned feature tables elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
