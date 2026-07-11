#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from _utils import load_yaml, output_dir


BASELINE = "strong_baseline_cv"
ENHANCED = "strong_baseline_cv_plus_strict_temporal_dynamics"


def bool_series(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy(bool)
    return s.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(bool)


def metric_value(y: np.ndarray, score: np.ndarray, alert_1: np.ndarray, alert_5: np.ndarray, metric: str) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    if metric == "auprc":
        return float(average_precision_score(y, score))
    if metric == "auroc":
        return float(roc_auc_score(y, score))
    if metric == "recall_at_1pct_fpr":
        pos = y == 1
        return float(np.mean(np.asarray(alert_1, dtype=bool)[pos])) if np.any(pos) else np.nan
    if metric == "recall_at_5pct_fpr":
        pos = y == 1
        return float(np.mean(np.asarray(alert_5, dtype=bool)[pos])) if np.any(pos) else np.nan
    raise ValueError(metric)


def seed_for(*parts: Any) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_one(job: dict[str, Any], n_bootstrap: int, base_seed: int) -> dict[str, Any]:
    merged: pd.DataFrame = job.pop("merged")
    metric = str(job["metric"])
    y = merged["y_true"].to_numpy(int)
    groups = merged["scenario_id"].astype(str).to_numpy()
    uniq = np.unique(groups)
    group_indices = [np.where(groups == g)[0] for g in uniq]
    b_score = merged["score_baseline"].to_numpy(float)
    e_score = merged["score_enhanced"].to_numpy(float)
    b_alert_1 = bool_series(merged["alert_1_baseline"])
    e_alert_1 = bool_series(merged["alert_1_enhanced"])
    b_alert_5 = bool_series(merged["alert_5_baseline"])
    e_alert_5 = bool_series(merged["alert_5_enhanced"])
    rng = np.random.default_rng(base_seed)
    vals: list[float] = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        b = metric_value(yy, b_score[idx], b_alert_1[idx], b_alert_5[idx], metric)
        e = metric_value(yy, e_score[idx], e_alert_1[idx], e_alert_5[idx], metric)
        if np.isfinite(b) and np.isfinite(e):
            vals.append(float(e - b))
    arr = np.asarray(vals, dtype=float)
    baseline_point = metric_value(y, b_score, b_alert_1, b_alert_5, metric)
    enhanced_point = metric_value(y, e_score, e_alert_1, e_alert_5, metric)
    delta = float(enhanced_point - baseline_point) if np.isfinite(baseline_point) and np.isfinite(enhanced_point) else np.nan
    out = dict(job)
    out.update(
        {
            "baseline_point": float(baseline_point) if np.isfinite(baseline_point) else np.nan,
            "enhanced_point": float(enhanced_point) if np.isfinite(enhanced_point) else np.nan,
            "delta": delta,
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
            "bootstrap_prob_delta_gt_0": float(np.mean(arr > 0.0)) if len(arr) else np.nan,
            "p_value_two_sided": float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))) if len(arr) else np.nan,
            "n_bootstrap_valid": int(len(arr)),
            "bootstrap_n_requested": int(n_bootstrap),
            "n_samples": int(len(y)),
            "n_scenarios": int(len(uniq)),
            "positive_count": int(y.sum()),
            "positive_rate": float(np.mean(y)),
            "ci_type": "percentile",
            "resampling_unit": "scenario_id",
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_endpoint_design_robustness.yaml")
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--metrics", default="auprc,auroc,recall_at_1pct_fpr,recall_at_5pct_fpr")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    pred_path = Path(args.predictions_csv) if args.predictions_csv else out_dir / "waymo_design_variant_oof_predictions.csv"
    pred = pd.read_csv(pred_path)
    pred["sample_id"] = pred["sample_id"].astype(str)
    pred["scenario_id"] = pred["scenario_id"].astype(str)
    n_boot = int(args.n_bootstrap if args.n_bootstrap is not None else cfg["evaluation"]["bootstrap_replicates"])
    n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg["evaluation"]["n_jobs"])
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    jobs: list[dict[str, Any]] = []
    for (vid, seed), sub in pred.groupby(["variant_id", "seed"], dropna=False):
        base = sub[sub["feature_set"] == BASELINE]
        enh = sub[sub["feature_set"] == ENHANCED]
        if base.empty or enh.empty:
            continue
        merged = base[["sample_id", "scenario_id", "y_true", "score", "alert_at_calibrated_1pct_fpr", "alert_at_calibrated_5pct_fpr"]].rename(
            columns={
                "score": "score_baseline",
                "alert_at_calibrated_1pct_fpr": "alert_1_baseline",
                "alert_at_calibrated_5pct_fpr": "alert_5_baseline",
            }
        ).merge(
            enh[["sample_id", "score", "alert_at_calibrated_1pct_fpr", "alert_at_calibrated_5pct_fpr"]].rename(
                columns={
                    "score": "score_enhanced",
                    "alert_at_calibrated_1pct_fpr": "alert_1_enhanced",
                    "alert_at_calibrated_5pct_fpr": "alert_5_enhanced",
                }
            ),
            on="sample_id",
            how="inner",
        )
        for metric in metrics:
            jobs.append(
                {
                    "variant_id": vid,
                    "endpoint": "actionability_critical_or_worse",
                    "model": "rf",
                    "seed": int(seed),
                    "baseline_feature_set": BASELINE,
                    "enhanced_feature_set": ENHANCED,
                    "metric": metric,
                    "feature_mode": cfg["evaluation"]["feature_mode"],
                    "merged": merged.copy(),
                }
            )
    rows = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_one)(job, n_boot, seed_for(cfg["evaluation"]["bootstrap_seed"], job["variant_id"], job["seed"], job["metric"]))
        for job in jobs
    )
    pd.DataFrame(rows).sort_values(["variant_id", "seed", "metric"]).to_csv(out_dir / "waymo_design_variant_bootstrap_deltas.csv", index=False)
    config = {
        "predictions_csv": str(pred_path),
        "n_bootstrap": n_boot,
        "n_jobs": n_jobs,
        "metrics": metrics,
        "baseline_feature_set": BASELINE,
        "enhanced_feature_set": ENHANCED,
        "seed": cfg["evaluation"]["bootstrap_seed"],
        "resampling_unit": "scenario_id",
        "elapsed_s": time.perf_counter() - t0,
    }
    (out_dir / "waymo_design_variant_bootstrap_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[v096-bootstrap] wrote {out_dir / 'waymo_design_variant_bootstrap_deltas.csv'} elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
