#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _utils import append_blockers, load_yaml, output_dir, resolve_path, write_csv


LABEL_NAMES = {
    0: "high_actionability",
    1: "reduced_actionability",
    2: "critical_actionability",
    3: "infeasible_or_unavoidable",
}


@lru_cache(maxsize=1)
def label_mod():
    path = Path(__file__).resolve().parent / "01_generate_design_variant_labels.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("nc_v096_design_labels", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def state_valid(xy: Any, heading: Any) -> bool:
    arr = np.asarray(xy, dtype=float)
    return bool(arr.shape == (2,) and np.all(np.isfinite(arr)) and np.isfinite(float(heading)))


def future_state(sample: dict[str, Any], agent_i: int, k: int) -> tuple[np.ndarray, float, bool]:
    fut_xy = np.asarray(sample.get("future_xy"))
    fut_valid = np.asarray(sample.get("future_valid"))
    fut_heading = np.asarray(sample.get("future_heading"))
    if fut_xy.ndim != 3 or fut_valid.ndim != 2 or agent_i >= fut_xy.shape[0] or k < 0 or k >= fut_xy.shape[1]:
        return np.asarray([np.nan, np.nan], dtype=float), np.nan, False
    valid = bool(fut_valid[agent_i, k])
    xy = np.asarray(fut_xy[agent_i, k], dtype=float)
    heading = float(fut_heading[agent_i, k]) if fut_heading.ndim == 2 and k < fut_heading.shape[1] else np.nan
    return xy, heading, bool(valid and state_valid(xy, heading))


def future_velocity(sample: dict[str, Any], agent_i: int, k: int, dt_s: float) -> np.ndarray:
    fut_vel = np.asarray(sample.get("future_vel_xy"))
    if fut_vel.ndim == 3 and agent_i < fut_vel.shape[0] and 0 <= k < fut_vel.shape[1]:
        v = np.asarray(fut_vel[agent_i, k], dtype=float)
        if v.shape == (2,) and np.all(np.isfinite(v)):
            return v
    xy_k, _, ok_k = future_state(sample, agent_i, k)
    if ok_k:
        for j in range(k - 1, -1, -1):
            xy_j, _, ok_j = future_state(sample, agent_i, j)
            if ok_j:
                return (xy_k - xy_j) / max((k - j) * dt_s, 1e-9)
    cur_v = np.asarray(sample["current_vel_xy"][agent_i], dtype=float)
    if cur_v.shape == (2,) and np.all(np.isfinite(cur_v)):
        return cur_v
    return np.zeros(2, dtype=float)


def current_state(sample: dict[str, Any], agent_i: int) -> tuple[np.ndarray, float, bool]:
    xy = np.asarray(sample["current_xy"][agent_i], dtype=float)
    heading = float(sample["current_heading"][agent_i])
    return xy, heading, state_valid(xy, heading)


def most_recent_valid_anchor(sample: dict[str, Any], agent_i: int, step_i: int) -> tuple[int, np.ndarray, float, bool]:
    fut_xy = np.asarray(sample.get("future_xy"))
    max_k = min(int(step_i), fut_xy.shape[1] - 1) if fut_xy.ndim == 3 else -1
    for k in range(max_k, -1, -1):
        xy, heading, ok = future_state(sample, agent_i, k)
        if ok:
            return k, xy, heading, True
    xy, heading, ok = current_state(sample, agent_i)
    return 0, xy, heading, ok


def obstacle_pose(
    sample: dict[str, Any],
    agent_i: int,
    step_i: int,
    t: float,
    mode: str,
    dt_s: float,
    heading_speed_threshold_mps: float,
    pose_cache: dict[tuple[int, int, str], tuple[np.ndarray, float, bool, str]],
) -> tuple[np.ndarray, float, bool, str]:
    key = (int(agent_i), int(step_i), str(mode))
    if key in pose_cache:
        return pose_cache[key]
    xy, heading, ok = future_state(sample, agent_i, step_i)
    if ok:
        out = (xy, heading, True, "observed_oracle_future")
        pose_cache[key] = out
        return out
    if mode == "skip_invalid_oracle_future":
        out = (xy, heading, False, "invalid_skipped")
        pose_cache[key] = out
        return out
    if mode != "cv_fallback_invalid_future":
        raise ValueError(f"unknown future handling mode: {mode}")

    anchor_k, anchor_xy, anchor_heading, anchor_ok = most_recent_valid_anchor(sample, agent_i, step_i)
    if not anchor_ok:
        out = (anchor_xy, anchor_heading, False, "non_imputable")
        pose_cache[key] = out
        return out
    v = future_velocity(sample, agent_i, anchor_k, dt_s)
    anchor_t = float(anchor_k) * float(dt_s)
    pred_xy = np.asarray(anchor_xy, dtype=float) + np.asarray(v, dtype=float) * max(float(t) - anchor_t, 0.0)
    pred_heading = float(anchor_heading)
    if np.linalg.norm(v) > float(heading_speed_threshold_mps):
        pred_heading = float(math.atan2(float(v[1]), float(v[0])))
    out = (pred_xy, pred_heading, bool(state_valid(pred_xy, pred_heading)), "cv_fallback_imputed")
    pose_cache[key] = out
    return out


def future_imputation_stats(sample: dict[str, Any], horizon_s: float, dt_s: float) -> dict[str, Any]:
    ego = int(sample["ego_index"])
    agent_count = int(sample["agent_count"])
    steps = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    total = invalid = imputed = non_imputable = 0
    actors_with_invalid = set()
    for j in range(agent_count):
        if j == ego:
            continue
        for step_i, _t in enumerate(steps, start=1):
            total += 1
            _xy, _hd, ok = future_state(sample, j, step_i)
            if ok:
                continue
            invalid += 1
            actors_with_invalid.add(j)
            _ak, _axy, _ahd, anchor_ok = most_recent_valid_anchor(sample, j, step_i)
            if anchor_ok:
                imputed += 1
            else:
                non_imputable += 1
    return {
        "total_future_slots": int(total),
        "invalid_future_slots": int(invalid),
        "imputed_future_slots": int(imputed),
        "non_imputable_future_slots": int(non_imputable),
        "imputed_fraction": float(imputed / max(total, 1)),
        "invalid_fraction": float(invalid / max(total, 1)),
        "non_imputable_fraction": float(non_imputable / max(total, 1)),
        "actors_with_invalid_future": int(len(actors_with_invalid)),
    }


def action_failure_time(
    sample: dict[str, Any],
    action: dict[str, Any],
    mode: str,
    horizon_s: float,
    dt_s: float,
    drivable: Any,
    heading_speed_threshold_mps: float,
    pose_cache: dict[tuple[int, int, str], tuple[np.ndarray, float, bool, str]],
) -> float | None:
    mod = label_mod()
    ego = int(sample["ego_index"])
    ego_l, ego_w = np.asarray(sample["current_size_lw"][ego], dtype=float)
    sizes = np.asarray(sample["current_size_lw"], dtype=float)
    poses, headings = mod.rollout_action(sample, action, horizon_s, dt_s)
    times = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    for step_i, (t, xy, hd) in enumerate(zip(times, poses, headings), start=1):
        if not mod.map_ok(drivable, xy):
            return float(t)
        ego_quad = mod.rect_corners(float(xy[0]), float(xy[1]), float(hd), float(ego_l), float(ego_w))
        for j in range(int(sample["agent_count"])):
            if j == ego:
                continue
            obs_xy, obs_hd, valid, _source = obstacle_pose(sample, j, step_i, float(t), mode, dt_s, heading_speed_threshold_mps, pose_cache)
            if not valid or not np.all(np.isfinite(obs_xy)):
                continue
            length, width = sizes[j]
            obs_quad = mod.rect_corners(float(obs_xy[0]), float(obs_xy[1]), float(obs_hd), float(length), float(width))
            if mod.quad_overlap(ego_quad, obs_quad):
                return float(t)
    return None


def time_to_no_action(failure_times: dict[str, float | None], names: set[str], horizon_s: float) -> float:
    if any(failure_times[name] is None for name in names):
        return float(horizon_s)
    return float(max(float(failure_times[name]) for name in names))


def process_label(
    row_dict: dict[str, Any],
    sample: dict[str, Any],
    mode: str,
    drivable: Any,
    cfg: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    mod = label_mod()
    actions = mod.action_library(str(cfg["actionability_labels"]["action_library"]))
    comfort_names = {a["name"] for a in actions if bool(a["comfort"])}
    emergency_names = {a["name"] for a in actions}
    horizon_s = float(cfg["actionability_labels"]["horizon_s"])
    dt_s = float(cfg["actionability_labels"]["dt_s"])
    threshold = float(cfg["actionability_labels"].get("heading_speed_threshold_mps", 0.5))
    pose_cache: dict[tuple[int, int, str], tuple[np.ndarray, float, bool, str]] = {}
    failure_times: dict[str, float | None] = {}
    feasible: dict[str, bool] = {}
    for action in actions:
        ft = action_failure_time(sample, action, mode, horizon_s, dt_s, drivable, threshold, pose_cache)
        failure_times[action["name"]] = ft
        feasible[action["name"]] = ft is None
    comfort_feasible = sum(1 for name in comfort_names if feasible[name])
    emergency_feasible = sum(1 for name in emergency_names if feasible[name])
    comfort_ratio = comfort_feasible / max(len(comfort_names), 1)
    emergency_ratio = emergency_feasible / max(len(emergency_names), 1)
    label_id = int(mod.label_moderate(comfort_ratio, emergency_ratio))
    feasible_costs = [float(a["cost"]) for a in actions if feasible[a["name"]]]
    return {
        "sample_id": str(row_dict["sample_id"]),
        "scenario_id": str(row_dict.get("scenario_id", sample.get("scenario_id", row_dict["sample_id"]))),
        "original_label_id": row_dict.get("label_id", ""),
        "original_label_name": row_dict.get("label_name", ""),
        "actionability_label_id": label_id,
        "actionability_label_name": LABEL_NAMES[label_id],
        "actionability_binary_degraded": int(label_id >= 1),
        "actionability_binary_critical": int(label_id >= 2),
        "actionability_binary_infeasible": int(label_id == 3),
        "comfort_feasible_count": int(comfort_feasible),
        "comfort_total_count": int(len(comfort_names)),
        "comfort_feasible_ratio": float(comfort_ratio),
        "emergency_feasible_count": int(emergency_feasible),
        "emergency_total_count": int(len(emergency_names)),
        "emergency_feasible_ratio": float(emergency_ratio),
        "n_candidates": int(len(actions)),
        "n_feasible_comfort": int(comfort_feasible),
        "n_feasible_emergency": int(emergency_feasible),
        "min_required_action_cost": float(min(feasible_costs)) if feasible_costs else np.nan,
        "time_to_no_comfort_s": time_to_no_action(failure_times, comfort_names, horizon_s),
        "time_to_no_emergency_s": time_to_no_action(failure_times, emergency_names, horizon_s),
        "obstacle_mode": "oracle_future",
        "future_handling": mode,
        "horizon_s": horizon_s,
        "dt_s": dt_s,
        "lane_buffer_m": float(cfg["actionability_labels"]["lane_buffer_m"]),
        "action_library": str(cfg["actionability_labels"]["action_library"]),
        "threshold_rule": str(cfg["actionability_labels"]["threshold_rule"]),
        "use_map": True,
        **stats,
    }


def process_sample_job(row_dict: dict[str, Any], sample_path_text: str, cfg: dict[str, Any], modes: list[str]) -> dict[str, Any]:
    sid = str(row_dict["sample_id"])
    sample_path = Path(sample_path_text)
    out: dict[str, Any] = {"sample_id": sid, "rows": {}, "error": None}
    if not sample_path.exists():
        out["error"] = {"sample_id": sid, "status": "MISSING_SAMPLE_PKL", "path": str(sample_path)}
        return out
    try:
        mod = label_mod()
        sample = mod.load_sample(sample_path)
        drivable = mod.build_drivable(sample, float(cfg["actionability_labels"]["lane_buffer_m"]))
        stats = future_imputation_stats(sample, float(cfg["actionability_labels"]["horizon_s"]), float(cfg["actionability_labels"]["dt_s"]))
        for mode in modes:
            out["rows"][mode] = process_label(row_dict, sample, mode, drivable, cfg, stats)
    except Exception as exc:
        out["error"] = {"sample_id": sid, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    return out


def deterministic_pilot(labels: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return labels.sample(n=min(int(n), len(labels)), random_state=int(seed)).sort_values("sample_id").copy()


def write_shift_outputs(out_dir: Path, cv: pd.DataFrame, ref: pd.DataFrame) -> None:
    cv = cv.copy()
    ref = ref.copy()
    cv["sample_id"] = cv["sample_id"].astype(str)
    ref["sample_id"] = ref["sample_id"].astype(str)
    merged = ref[["sample_id", "actionability_label_id", "actionability_label_name"]].rename(
        columns={"actionability_label_id": "reference_label_id", "actionability_label_name": "reference_label_name"}
    ).merge(
        cv[["sample_id", "scenario_id", "actionability_label_id", "actionability_label_name", "comfort_feasible_ratio", "emergency_feasible_ratio", "imputed_fraction", "invalid_fraction", "non_imputable_fraction"]],
        on="sample_id",
        how="inner",
    ).rename(columns={"actionability_label_id": "cv_label_id", "actionability_label_name": "cv_label_name"})
    ref_label = pd.to_numeric(merged["reference_label_id"], errors="coerce").astype(int)
    cv_label = pd.to_numeric(merged["cv_label_id"], errors="coerce").astype(int)
    changed = ref_label != cv_label
    severe_ref = ref_label >= 2
    severe_cv = cv_label >= 2
    severe_union = set(merged.loc[severe_ref | severe_cv, "sample_id"])
    severe_inter = set(merged.loc[severe_ref & severe_cv, "sample_id"])
    summary = [
        {
            "comparison": "skip_invalid_oracle_future_vs_cv_fallback_invalid_future",
            "n_common": int(len(merged)),
            "label_changed_count": int(changed.sum()),
            "label_changed_fraction": float(changed.mean()) if len(changed) else np.nan,
            "reference_critical_or_worse_prevalence": float(severe_ref.mean()) if len(merged) else np.nan,
            "cv_critical_or_worse_prevalence": float(severe_cv.mean()) if len(merged) else np.nan,
            "reference_infeasible_prevalence": float((ref_label == 3).mean()) if len(merged) else np.nan,
            "cv_infeasible_prevalence": float((cv_label == 3).mean()) if len(merged) else np.nan,
            "severe_set_jaccard": float(len(severe_inter) / len(severe_union)) if severe_union else 1.0,
            "mean_imputed_fraction": float(pd.to_numeric(merged["imputed_fraction"], errors="coerce").mean()),
            "p95_imputed_fraction": float(pd.to_numeric(merged["imputed_fraction"], errors="coerce").quantile(0.95)),
        }
    ]
    write_csv(out_dir / "cv_fallback_label_shift_summary.csv", summary)
    ct = pd.crosstab(ref_label, cv_label)
    rows = []
    for r in [0, 1, 2, 3]:
        for c in [0, 1, 2, 3]:
            rows.append(
                {
                    "reference_label_id": r,
                    "reference_label_name": LABEL_NAMES[r],
                    "cv_label_id": c,
                    "cv_label_name": LABEL_NAMES[c],
                    "count": int(ct.loc[r, c]) if r in ct.index and c in ct.columns else 0,
                }
            )
    write_csv(out_dir / "cv_fallback_transition_matrix.csv", rows)
    prev = []
    for source, series in [("reference_skip_invalid", ref_label), ("cv_fallback_invalid_future", cv_label)]:
        counts = series.value_counts().reindex([0, 1, 2, 3], fill_value=0)
        for lid, count in counts.items():
            prev.append({"source": source, "label_id": int(lid), "label_name": LABEL_NAMES[int(lid)], "count": int(count), "fraction": float(count / max(len(series), 1))})
    write_csv(out_dir / "cv_fallback_prevalence_by_label.csv", prev)
    write_csv(
        out_dir / "cv_fallback_severe_overlap.csv",
        [
            {
                "reference_severe_count": int(severe_ref.sum()),
                "cv_severe_count": int(severe_cv.sum()),
                "intersection_count": int((severe_ref & severe_cv).sum()),
                "union_count": int((severe_ref | severe_cv).sum()),
                "jaccard": float(len(severe_inter) / len(severe_union)) if severe_union else 1.0,
                "cv_added_severe": int((~severe_ref & severe_cv).sum()),
                "cv_removed_severe": int((severe_ref & ~severe_cv).sum()),
            }
        ],
    )


def write_imputation_outputs(out_dir: Path, cv: pd.DataFrame) -> None:
    stat_cols = ["total_future_slots", "invalid_future_slots", "imputed_future_slots", "non_imputable_future_slots"]
    totals = {c: int(pd.to_numeric(cv[c], errors="coerce").fillna(0).sum()) for c in stat_cols}
    summary = {
        "n_samples": int(len(cv)),
        **totals,
        "invalid_fraction": float(totals["invalid_future_slots"] / max(totals["total_future_slots"], 1)),
        "imputed_fraction": float(totals["imputed_future_slots"] / max(totals["total_future_slots"], 1)),
        "non_imputable_fraction": float(totals["non_imputable_future_slots"] / max(totals["total_future_slots"], 1)),
        "mean_sample_imputed_fraction": float(pd.to_numeric(cv["imputed_fraction"], errors="coerce").mean()),
        "p95_sample_imputed_fraction": float(pd.to_numeric(cv["imputed_fraction"], errors="coerce").quantile(0.95)),
        "max_sample_imputed_fraction": float(pd.to_numeric(cv["imputed_fraction"], errors="coerce").max()),
    }
    write_csv(out_dir / "cv_fallback_imputation_summary.csv", [summary])
    by_label = []
    for lid, sub in cv.groupby("actionability_label_id", dropna=False):
        row = {"cv_label_id": int(lid), "cv_label_name": LABEL_NAMES.get(int(lid), str(lid)), "n_samples": int(len(sub))}
        for c in stat_cols:
            row[c] = int(pd.to_numeric(sub[c], errors="coerce").fillna(0).sum())
        row["mean_sample_imputed_fraction"] = float(pd.to_numeric(sub["imputed_fraction"], errors="coerce").mean())
        row["median_sample_imputed_fraction"] = float(pd.to_numeric(sub["imputed_fraction"], errors="coerce").median())
        by_label.append(row)
    write_csv(out_dir / "cv_fallback_imputation_by_label.csv", by_label)
    threshold = float(pd.to_numeric(cv["imputed_fraction"], errors="coerce").quantile(0.95))
    low = cv[(pd.to_numeric(cv["imputed_fraction"], errors="coerce") >= threshold) | (pd.to_numeric(cv["non_imputable_future_slots"], errors="coerce") > 0)].copy()
    low = low.sort_values(["imputed_fraction", "invalid_future_slots"], ascending=[False, False]).head(1000)
    low.to_csv(out_dir / "cv_fallback_low_imputation_samples.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_cv_fallback.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.pilot and not args.full:
        raise ValueError("Specify --pilot or --full")
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    samples_dir = resolve_path(cfg["inputs"]["waymo_samples_dir"])
    prox = pd.read_csv(resolve_path(cfg["inputs"]["waymo_proximity_labels_csv"]))
    prox["sample_id"] = prox["sample_id"].astype(str)
    ref = pd.read_csv(resolve_path(cfg["inputs"]["waymo_actionability_reference_labels_csv"]))
    ref["sample_id"] = ref["sample_id"].astype(str)

    if args.pilot:
        n = int(args.max_samples if args.max_samples is not None else cfg["actionability_labels"]["pilot_sample_size"])
        labels = deterministic_pilot(prox, n, int(cfg["actionability_labels"]["pilot_seed"]))
        modes = [cfg["actionability_labels"]["reference_future_handling"], cfg["actionability_labels"]["cv_future_handling"]]
    else:
        labels = prox.copy()
        if args.max_samples:
            labels = labels.head(int(args.max_samples)).copy()
        modes = [cfg["actionability_labels"]["cv_future_handling"]]
        out_label = out_dir / "labels_actionability_moderate_cv_fallback_full.csv"
        if out_label.exists() and not args.force:
            print(f"[cv-relabel] full labels already exist: {out_label}")
            return

    jobs = [(row._asdict(), str(samples_dir / f"{str(getattr(row, 'sample_id'))}.pkl.gz")) for row in labels.itertuples(index=False)]
    if int(args.n_jobs) == 1:
        results = [process_sample_job(row, path, cfg, modes) for row, path in jobs]
    else:
        results = Parallel(n_jobs=int(args.n_jobs), verbose=10, batch_size=8, prefer="threads")(
            delayed(process_sample_job)(row, path, cfg, modes) for row, path in jobs
        )
    errors = [r["error"] for r in results if r.get("error")]
    rows_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    for result in results:
        for mode, row in result["rows"].items():
            rows_by_mode[mode].append(row)
    if errors:
        write_csv(out_dir / ("cv_fallback_relabel_errors_pilot.csv" if args.pilot else "cv_fallback_relabel_errors_full.csv"), errors)

    if args.pilot:
        skip = pd.DataFrame(rows_by_mode[cfg["actionability_labels"]["reference_future_handling"]])
        cv = pd.DataFrame(rows_by_mode[cfg["actionability_labels"]["cv_future_handling"]])
        skip.to_csv(out_dir / "pilot_skip_invalid_oracle_future_labels.csv", index=False)
        cv.to_csv(out_dir / "pilot_cv_fallback_invalid_future_labels.csv", index=False)
        existing = ref[ref["sample_id"].isin(skip["sample_id"].astype(str))].copy()
        merged = skip[["sample_id", "actionability_label_id", "comfort_feasible_ratio", "emergency_feasible_ratio"]].rename(
            columns={
                "actionability_label_id": "regenerated_label_id",
                "comfort_feasible_ratio": "regenerated_comfort_feasible_ratio",
                "emergency_feasible_ratio": "regenerated_emergency_feasible_ratio",
            }
        ).merge(
            existing[["sample_id", "actionability_label_id", "comfort_feasible_ratio", "emergency_feasible_ratio"]].rename(
                columns={
                    "actionability_label_id": "existing_label_id",
                    "comfort_feasible_ratio": "existing_comfort_feasible_ratio",
                    "emergency_feasible_ratio": "existing_emergency_feasible_ratio",
                }
            ),
            on="sample_id",
            how="left",
        )
        merged["label_match"] = merged["regenerated_label_id"].astype(int) == merged["existing_label_id"].astype(int)
        merged["comfort_ratio_abs_diff"] = (merged["regenerated_comfort_feasible_ratio"] - merged["existing_comfort_feasible_ratio"]).abs()
        merged["emergency_ratio_abs_diff"] = (merged["regenerated_emergency_feasible_ratio"] - merged["existing_emergency_feasible_ratio"]).abs()
        merged["parity_status"] = np.where(merged["label_match"] & (merged["comfort_ratio_abs_diff"] <= 1e-12) & (merged["emergency_ratio_abs_diff"] <= 1e-12), "PASS", "FAIL")
        merged.to_csv(out_dir / "reference_relabel_parity_audit.csv", index=False)
        write_shift_outputs(out_dir, cv, existing)
        write_imputation_outputs(out_dir, cv)
        fail_count = int((merged["parity_status"] != "PASS").sum())
        if fail_count:
            append_blockers(
                out_dir,
                [
                    {
                        "category": "cv_fallback_reference_parity",
                        "item": "pilot_reference_relabel",
                        "status": "BLOCKED",
                        "details": f"{fail_count}/{len(merged)} pilot samples differed from existing reference labels/ratios.",
                        "resume_command": f"conda run -n waymo_rt_bev python scripts/nc_v096/01_cv_fallback_relabel.py --config {args.config} --pilot --max-samples {len(merged)}",
                    }
                ],
            )
            raise SystemExit(f"reference parity failed for {fail_count}/{len(merged)} samples")
        print(f"[cv-relabel] pilot parity PASS rows={len(merged)} elapsed_s={time.perf_counter() - t0:.1f}")
        return

    cv = pd.DataFrame(rows_by_mode[cfg["actionability_labels"]["cv_future_handling"]])
    cv.to_csv(out_dir / "labels_actionability_moderate_cv_fallback_full.csv", index=False)
    write_shift_outputs(out_dir, cv, ref)
    write_imputation_outputs(out_dir, cv)
    print(f"[cv-relabel] full CV-fallback labels rows={len(cv)} elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
