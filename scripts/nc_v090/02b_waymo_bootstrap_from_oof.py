#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score


BASELINE = "strong_baseline_cv"
PRIMARY_ENHANCED = "strong_baseline_cv_plus_strict_temporal_dynamics"
METRICS = ["auprc", "auroc", "recall_at_5pct_fpr"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_value(y: np.ndarray, score: np.ndarray, alert: np.ndarray, metric: str) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    alert = np.asarray(alert, dtype=bool)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    alert = alert[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    if metric == "auprc":
        return float(average_precision_score(y, score))
    if metric == "auroc":
        return float(roc_auc_score(y, score))
    if metric == "recall_at_5pct_fpr":
        pos = y == 1
        return float(np.mean(alert[pos])) if np.any(pos) else np.nan
    raise ValueError(metric)


def seed_for(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_one(job: dict[str, Any], n_bootstrap: int, seed: int) -> dict[str, Any]:
    merged: pd.DataFrame = job.pop("merged")
    metric = job["metric"]
    y = merged["y_true"].to_numpy(int)
    b_score = merged["score_baseline"].to_numpy(float)
    e_score = merged["score_enhanced"].to_numpy(float)
    b_alert = merged["alert_baseline"].to_numpy(bool)
    e_alert = merged["alert_enhanced"].to_numpy(bool)
    groups = merged["scenario_id"].astype(str).to_numpy()
    uniq = np.unique(groups)
    group_indices = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        b = metric_value(yy, b_score[idx], b_alert[idx], metric)
        e = metric_value(yy, e_score[idx], e_alert[idx], metric)
        if np.isfinite(b) and np.isfinite(e):
            vals.append(float(e - b))
    arr = np.asarray(vals, dtype=float)
    point = metric_value(y, e_score, e_alert, metric) - metric_value(y, b_score, b_alert, metric)
    out = dict(job)
    out.update({
        "delta": float(point),
        "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
        "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
        "n_bootstrap_valid": int(len(arr)),
        "bootstrap_status": "BOOTSTRAPPED_2000" if int(n_bootstrap) >= 2000 else f"BOOTSTRAPPED_{n_bootstrap}",
        "n_samples": int(len(y)),
        "n_scenarios": int(len(uniq)),
        "positive_count": int(y.sum()),
        "positive_rate": float(np.mean(y)),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v090/nc_v090_audit.yaml")
    parser.add_argument("--predictions-csv", default="results/nc_v090_scientific_audit/waymo_oof_predictions.csv")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    t0 = time.perf_counter()
    repo = Path.cwd()
    cfg = load_yaml(repo / args.config)
    out_dir = repo / cfg["project"]["output_dir"]
    pred = pd.read_csv(args.predictions_csv)
    pred["sample_id"] = pred["sample_id"].astype(str)
    pred["scenario_id"] = pred["scenario_id"].astype(str)
    pred["alert_at_calibrated_5pct_fpr"] = pred["alert_at_calibrated_5pct_fpr"].astype(str).str.lower().isin(["true", "1"])

    rows: list[dict[str, Any]] = []
    boot_jobs: list[dict[str, Any]] = []
    for (endpoint, model, seed), sub in pred.groupby(["endpoint", "model", "seed"], dropna=False):
        base = sub[sub["feature_set"] == BASELINE]
        if base.empty:
            continue
        for enhanced in sorted(set(sub["feature_set"]) - {BASELINE}):
            enh = sub[sub["feature_set"] == enhanced]
            if enh.empty:
                continue
            merged = base[["sample_id", "scenario_id", "y_true", "score", "alert_at_calibrated_5pct_fpr"]].rename(
                columns={"score": "score_baseline", "alert_at_calibrated_5pct_fpr": "alert_baseline"}
            ).merge(
                enh[["sample_id", "score", "alert_at_calibrated_5pct_fpr"]].rename(
                    columns={"score": "score_enhanced", "alert_at_calibrated_5pct_fpr": "alert_enhanced"}
                ),
                on="sample_id",
                how="inner",
            )
            for metric in METRICS:
                y = merged["y_true"].to_numpy(int)
                b = metric_value(y, merged["score_baseline"].to_numpy(float), merged["alert_baseline"].to_numpy(bool), metric)
                e = metric_value(y, merged["score_enhanced"].to_numpy(float), merged["alert_enhanced"].to_numpy(bool), metric)
                row = {
                    "endpoint": endpoint,
                    "model": model,
                    "seed": int(seed),
                    "baseline_feature_set": BASELINE,
                    "enhanced_feature_set": enhanced,
                    "metric": metric,
                    "baseline_point": b,
                    "enhanced_point": e,
                    "delta": e - b if np.isfinite(e) and np.isfinite(b) else np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_bootstrap_valid": 0,
                    "bootstrap_status": "POINT_ONLY_NOT_BOOTSTRAPPED",
                    "n_samples": int(len(merged)),
                    "n_scenarios": int(merged["scenario_id"].nunique()),
                    "positive_count": int(y.sum()),
                    "positive_rate": float(np.mean(y)),
                }
                if enhanced == PRIMARY_ENHANCED and model == "rf":
                    boot_jobs.append({**row, "merged": merged.copy(), "metric": metric})
                else:
                    rows.append(row)
    boot_rows = Parallel(n_jobs=int(args.n_jobs))(
        delayed(bootstrap_one)(job, int(args.n_bootstrap), seed_for(str(job["endpoint"]), str(job["seed"]), str(job["metric"])))
        for job in boot_jobs
    )
    rows.extend(boot_rows)
    out = pd.DataFrame(rows)
    out = out.sort_values(["endpoint", "model", "seed", "enhanced_feature_set", "metric"]).reset_index(drop=True)
    out.to_csv(out_dir / "waymo_paired_deltas.csv", index=False)

    stability = []
    for (endpoint, model, feature_set), sub in pred.groupby(["endpoint", "model", "feature_set"], dropna=False):
        vals = []
        for seed, seed_df in sub.groupby("seed"):
            vals.append(metric_value(seed_df["y_true"].to_numpy(int), seed_df["score"].to_numpy(float), seed_df["alert_at_calibrated_5pct_fpr"].to_numpy(bool), "auprc"))
        vals = np.asarray(vals, dtype=float)
        stability.append({
            "endpoint": endpoint,
            "model": model,
            "feature_set": feature_set,
            "seed_count": int(np.isfinite(vals).sum()),
            "auprc_min": float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan,
            "auprc_max": float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan,
            "auprc_range": float(np.nanmax(vals) - np.nanmin(vals)) if np.isfinite(vals).any() else np.nan,
        })
    pd.DataFrame(stability).to_csv(out_dir / "waymo_fold_seed_stability.csv", index=False)

    completeness = []
    for (endpoint, model, seed, feature_set), sub in pred.groupby(["endpoint", "model", "seed", "feature_set"], dropna=False):
        completeness.append({
            "endpoint": endpoint,
            "model": model,
            "seed": int(seed),
            "feature_set": feature_set,
            "rows": int(len(sub)),
            "unique_sample_id": int(sub["sample_id"].nunique()),
            "duplicate_sample_id_count": int(sub["sample_id"].duplicated().sum()),
            "fold_count": int(sub["outer_fold"].nunique()),
        })
    pd.DataFrame(completeness).to_csv(out_dir / "waymo_oof_completeness_audit.csv", index=False)
    blocker = {
        "category": "runtime_scope",
        "item": "non_primary_comparison_bootstrap",
        "status": "PARTIAL",
        "details": "Full all-comparison 2000-replicate bootstrap exceeded the execution timeout after model checkpoints completed. v0.9.0 primary strict-temporal RF comparisons were bootstrapped with 2000 replicates; secondary/context comparisons are point-only.",
        "resume_command": f"conda run -n waymo_rt_bev python scripts/nc_v090/02b_waymo_bootstrap_from_oof.py --config {args.config} --predictions-csv {args.predictions_csv} --n-bootstrap {args.n_bootstrap} --n-jobs {args.n_jobs}",
    }
    existing = []
    block_path = out_dir / "BLOCKERS.csv"
    if block_path.exists():
        existing = pd.read_csv(block_path).to_dict("records")
    write_csv(block_path, existing + [blocker])
    print(f"[bootstrap-oof] wrote {out_dir / 'waymo_paired_deltas.csv'}")
    print(f"[bootstrap-oof] primary bootstrap jobs={len(boot_jobs)} elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
