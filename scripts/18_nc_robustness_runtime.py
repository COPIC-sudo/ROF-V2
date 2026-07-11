from __future__ import annotations
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir, read_gzip_pickle
from rtbev.pipeline import sample_to_bev_tensor_and_features
from rtbev.tube.rt_library import PrimitiveLibrary, TubeLibrary
from rtbev.nc_eval import binary_score_metrics, normalize_score, y_for_task


def _perturb_sample(sample: dict, rng: np.random.Generator, kind: str, level: float) -> dict:
    s = copy.deepcopy(sample)
    if level <= 0:
        return s
    if kind == "position_noise_m":
        s["current_xy"] = np.asarray(s["current_xy"], dtype=np.float32) + rng.normal(0, level, size=np.asarray(s["current_xy"]).shape).astype(np.float32)
    elif kind == "heading_noise_deg":
        s["current_heading"] = np.asarray(s["current_heading"], dtype=np.float32) + rng.normal(0, np.deg2rad(level), size=np.asarray(s["current_heading"]).shape).astype(np.float32)
    elif kind == "velocity_noise_mps":
        s["current_vel_xy"] = np.asarray(s["current_vel_xy"], dtype=np.float32) + rng.normal(0, level, size=np.asarray(s["current_vel_xy"]).shape).astype(np.float32)
    elif kind == "missed_agent_prob":
        ego = int(s["ego_index"])
        keep = np.ones(int(s["agent_count"]), dtype=bool)
        for i in range(len(keep)):
            if i != ego and rng.random() < level:
                keep[i] = False
        # Keep at least ego + one other if possible.
        if keep.sum() >= 2:
            old_indices = np.where(keep)[0]
            new_ego = int(np.where(old_indices == ego)[0][0])
            for key in ["agent_ids", "agent_types", "current_xy", "current_vel_xy", "current_heading", "current_size_lw", "future_xy", "future_vel_xy", "future_heading", "future_valid"]:
                if key in s:
                    s[key] = np.asarray(s[key])[keep]
            s["ego_index"] = new_ego
            s["agent_count"] = int(keep.sum())
    return s


def _risk_score_from_feats(feats: dict) -> float:
    for c in ["redi_actionability", "redi_full", "redi_no_msr", "rcr"]:
        if c in feats:
            try:
                v = float(feats[c])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    if "asr" in feats:
        try:
            return 1.0 - float(feats["asr"])
        except Exception:
            pass
    return np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="NC-minimal robustness and runtime profiling for final ROF-v2.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "nc_robustness_runtime")
    device = args.device or cfg.get("runtime", {}).get("device", "cpu")
    labels = pd.read_csv(args.labels_csv)
    if args.max_samples is not None:
        labels = labels.iloc[: args.max_samples].copy()
    lib = TubeLibrary.from_workdir(work / "tube_library")
    prim = PrimitiveLibrary.from_workdir(work / "tube_library")
    rng = np.random.default_rng(args.seed)

    perturbations = [("none", 0.0)]
    perturbations += [("position_noise_m", v) for v in [0.1, 0.3, 0.5, 1.0]]
    perturbations += [("heading_noise_deg", v) for v in [1.0, 3.0, 5.0]]
    perturbations += [("velocity_noise_mps", v) for v in [0.2, 0.5, 1.0]]
    perturbations += [("missed_agent_prob", v) for v in [0.05, 0.10, 0.20]]

    y = y_for_task(labels["label_id"].astype(int).to_numpy(), "warning_or_above")
    rows = []
    runtime_rows = []
    for kind, level in perturbations:
        scores = []
        lat = []
        for _, row in tqdm(labels.iterrows(), total=len(labels), desc=f"robust {kind}={level}"):
            sid = row["sample_id"]
            sample = read_gzip_pickle(work / "samples" / f"{sid}.pkl.gz")
            sample_p = _perturb_sample(sample, rng, kind, float(level))
            t0 = time.perf_counter()
            _, feats = sample_to_bev_tensor_and_features(sample_p, lib, cfg, device=device, primitive_lib=prim, return_tensors=False)
            lat.append(time.perf_counter() - t0)
            scores.append(_risk_score_from_feats(feats))
        scores = np.asarray(scores, dtype=float)
        # Normalize within each perturbation for threshold-free robustness summaries.
        score_norm = normalize_score(pd.Series(scores), invert=False)
        rows.append({"perturbation": kind, "level": level, **binary_score_metrics(y[: len(score_norm)], score_norm, fpr_levels=[0.01, 0.05])})
        lat = np.asarray(lat, dtype=float)
        runtime_rows.append({
            "perturbation": kind,
            "level": level,
            "n": int(len(lat)),
            "latency_mean_s": float(np.mean(lat)),
            "latency_p50_s": float(np.percentile(lat, 50)),
            "latency_p90_s": float(np.percentile(lat, 90)),
            "latency_p95_s": float(np.percentile(lat, 95)),
            "latency_p99_s": float(np.percentile(lat, 99)),
        })
    pd.DataFrame(rows).to_csv(out / "robustness_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(out / "runtime_profile.csv", index=False)
    print(f"[nc-robustness] wrote {out}")


if __name__ == "__main__":
    main()
