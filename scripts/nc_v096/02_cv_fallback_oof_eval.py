#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from _utils import load_yaml, output_dir, resolve_path, seed_for


BASELINE = "strong_baseline_cv"
ENHANCED = "strong_baseline_cv_plus_strict_temporal_dynamics"


def load_oof_mod():
    path = Path(__file__).resolve().parent / "03_model_eval_variants.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("nc_v096_oof_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def endpoint_y(label: pd.Series, rule: str) -> pd.Series:
    lab = pd.to_numeric(label, errors="coerce").fillna(-1).astype(int)
    if rule == "ge2":
        return (lab >= 2).astype(int)
    if rule == "eq3":
        return (lab == 3).astype(int)
    raise ValueError(rule)


def bool_array(s: pd.Series) -> np.ndarray:
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
        return float(np.mean(alert_1[pos])) if np.any(pos) else np.nan
    if metric == "recall_at_5pct_fpr":
        pos = y == 1
        return float(np.mean(alert_5[pos])) if np.any(pos) else np.nan
    raise ValueError(metric)


def bootstrap_delta(job: dict[str, Any], n_bootstrap: int, seed: int) -> dict[str, Any]:
    merged: pd.DataFrame = job.pop("merged")
    metric = str(job["metric"])
    y = merged["y_true"].to_numpy(int)
    groups = merged["scenario_id"].astype(str).to_numpy()
    uniq = np.unique(groups)
    group_indices = [np.where(groups == g)[0] for g in uniq]
    b_score = merged["score_baseline"].to_numpy(float)
    e_score = merged["score_enhanced"].to_numpy(float)
    b_alert_1 = bool_array(merged["alert_1_baseline"])
    e_alert_1 = bool_array(merged["alert_1_enhanced"])
    b_alert_5 = bool_array(merged["alert_5_baseline"])
    e_alert_5 = bool_array(merged["alert_5_enhanced"])
    rng = np.random.default_rng(int(seed))
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
    out = dict(job)
    out.update(
        {
            "baseline_point": float(baseline_point) if np.isfinite(baseline_point) else np.nan,
            "enhanced_point": float(enhanced_point) if np.isfinite(enhanced_point) else np.nan,
            "delta": float(enhanced_point - baseline_point) if np.isfinite(enhanced_point) and np.isfinite(baseline_point) else np.nan,
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
            "bootstrap_prob_delta_gt_0": float(np.mean(arr > 0.0)) if len(arr) else np.nan,
            "p_value_two_sided": float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))) if len(arr) else np.nan,
            "n_bootstrap_valid": int(len(arr)),
            "bootstrap_n_requested": int(n_bootstrap),
            "resampling_unit": "scenario_id",
            "ci_type": "percentile",
            "n_samples": int(len(y)),
            "n_scenarios": int(len(uniq)),
            "positive_count": int(y.sum()),
            "prevalence": float(np.mean(y)),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_cv_fallback.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    pred_path = out_dir / "cv_fallback_oof_predictions.csv"
    if pred_path.exists() and not args.force:
        print(f"[cv-oof] using existing predictions: {pred_path}")
        return
    oof = load_oof_mod()
    features = pd.read_csv(resolve_path(cfg["inputs"]["waymo_features_csv"]))
    features["sample_id"] = features["sample_id"].astype(str)
    labels_path = out_dir / "labels_actionability_moderate_cv_fallback_full.csv"
    labels = pd.read_csv(labels_path)
    labels["sample_id"] = labels["sample_id"].astype(str)
    keep = ["sample_id", "scenario_id", "actionability_label_id", "actionability_label_name"]
    df0 = features.merge(labels[keep], on="sample_id", how="inner", suffixes=("", "_label"))
    if len(df0) != len(features):
        raise ValueError(f"join mismatch features={len(features)} labels_join={len(df0)}")
    if "scenario_id" not in df0.columns and "scenario_id_label" in df0.columns:
        df0["scenario_id"] = df0["scenario_id_label"]
    if "scenario_id" not in df0.columns:
        df0["scenario_id"] = df0["sample_id"]

    seeds = [int(x) for x in cfg["evaluation"]["rf_seeds"]]
    n_folds = int(cfg["evaluation"]["outer_folds"])
    fold_seed = int(cfg["evaluation"]["scenario_hash_seed"])
    cal_frac = float(cfg["evaluation"]["calibration_fraction_within_outer_train"])
    fpr_levels = [float(x) for x in cfg["evaluation"]["fixed_fpr_levels"]]
    n_boot = int(args.bootstrap_n if args.bootstrap_n is not None else cfg["evaluation"]["bootstrap_replicates"])
    n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg["evaluation"]["n_jobs"])

    all_preds: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for endpoint_cfg in cfg["evaluation"]["endpoints"]:
        endpoint = str(endpoint_cfg["endpoint"])
        df = df0.copy()
        df["y"] = endpoint_y(df["actionability_label_id"], str(endpoint_cfg["positive_rule"]))
        df["outer_fold"] = oof.assign_outer_folds(df, n_folds, fold_seed)
        feature_sets = {name: oof.available_cols(oof.FEATURE_SETS[name], df) for name in [BASELINE, ENHANCED]}
        for fs_name, cols in feature_sets.items():
            feature_rows.append({"endpoint": endpoint, "feature_set": fs_name, "n_features": len(cols), "features": ";".join(cols)})
        for seed in seeds:
            for fold in sorted(df["outer_fold"].unique()):
                test_mask = df["outer_fold"].to_numpy(int) == int(fold)
                train_cal_df = df.loc[~test_mask].copy()
                test_df = df.loc[test_mask].copy()
                cal_mask = oof.fit_calibration_mask(train_cal_df, int(seed), int(fold), cal_frac)
                fit_df = train_cal_df.loc[~cal_mask].copy()
                cal_df = train_cal_df.loc[cal_mask].copy()
                for fs_name, cols in feature_sets.items():
                    y_fit = fit_df["y"].to_numpy(int)
                    y_cal = cal_df["y"].to_numpy(int)
                    y_test = test_df["y"].to_numpy(int)
                    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                        continue
                    pre = oof.TrainOnlyPreprocessor(cols).fit(fit_df)
                    model = oof.make_model(seed, cfg)
                    model.fit(pre.transform(fit_df), y_fit)
                    score_cal = oof.positive_score(model, pre.transform(cal_df))
                    score_test = oof.positive_score(model, pre.transform(test_df))
                    thresholds: dict[str, float] = {}
                    cal_fprs: dict[str, float] = {}
                    for fpr in fpr_levels:
                        label = "1pct_fpr" if abs(fpr - 0.01) < 1e-9 else "5pct_fpr" if abs(fpr - 0.05) < 1e-9 else f"{fpr:g}_fpr"
                        thr, cal_fpr = oof.threshold_from_calibration(y_cal, score_cal, fpr)
                        thresholds[label] = thr
                        cal_fprs[label] = cal_fpr
                    fold_metrics = oof.metrics_at_thresholds(y_test, score_test, thresholds)
                    metrics_rows.append(
                        {
                            "level": "fold",
                            "endpoint": endpoint,
                            "future_handling": cfg["actionability_labels"]["cv_future_handling"],
                            "model": "rf",
                            "seed": int(seed),
                            "outer_fold": int(fold),
                            "feature_set": fs_name,
                            "n_fit": int(len(fit_df)),
                            "n_calibration": int(len(cal_df)),
                            **fold_metrics,
                        }
                    )
                    for name, thr in thresholds.items():
                        operating_rows.append(
                            {
                                "endpoint": endpoint,
                                "future_handling": cfg["actionability_labels"]["cv_future_handling"],
                                "model": "rf",
                                "seed": int(seed),
                                "outer_fold": int(fold),
                                "feature_set": fs_name,
                                "nominal_fpr": 0.01 if name == "1pct_fpr" else 0.05,
                                "threshold_source": "calibration_negatives_only",
                                "threshold_operator": ">=",
                                "threshold": thr,
                                "calibration_fpr": cal_fprs[name],
                                "outer_test_achieved_fpr": fold_metrics[f"achieved_fpr_at_{name}"],
                                "outer_test_recall": fold_metrics[f"recall_at_{name}"],
                                "outer_test_precision": fold_metrics[f"precision_at_{name}"],
                            }
                        )
                    pred = pd.DataFrame(
                        {
                            "sample_id": test_df["sample_id"].astype(str).to_numpy(),
                            "scenario_id": test_df["scenario_id"].astype(str).to_numpy(),
                            "endpoint": endpoint,
                            "future_handling": cfg["actionability_labels"]["cv_future_handling"],
                            "model": "rf",
                            "seed": int(seed),
                            "outer_fold": int(fold),
                            "feature_set": fs_name,
                            "y_true": y_test,
                            "score": score_test,
                            "threshold_at_1pct_fpr": thresholds.get("1pct_fpr", np.nan),
                            "threshold_at_5pct_fpr": thresholds.get("5pct_fpr", np.nan),
                            "alert_at_calibrated_1pct_fpr": score_test >= thresholds.get("1pct_fpr", np.nan),
                            "alert_at_calibrated_5pct_fpr": score_test >= thresholds.get("5pct_fpr", np.nan),
                        }
                    )
                    all_preds.append(pred)
                    print(f"[cv-oof] {endpoint} seed={seed} fold={fold} feature_set={fs_name} elapsed_s={time.perf_counter()-t0:.1f}")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(pred_path, index=False)
    pooled_rows = []
    for (endpoint, seed, fs), sub in pred_df.groupby(["endpoint", "seed", "feature_set"], dropna=False):
        m = oof.pooled_metrics(
            sub["y_true"].to_numpy(int),
            sub["score"].to_numpy(float),
            bool_array(sub["alert_at_calibrated_1pct_fpr"]),
            bool_array(sub["alert_at_calibrated_5pct_fpr"]),
        )
        pooled_rows.append(
            {
                "level": "pooled_oof",
                "endpoint": endpoint,
                "future_handling": cfg["actionability_labels"]["cv_future_handling"],
                "model": "rf",
                "seed": int(seed),
                "feature_set": fs,
                "n_fit": np.nan,
                "n_calibration": np.nan,
                **m,
            }
        )
    pd.DataFrame(metrics_rows + pooled_rows).to_csv(out_dir / "cv_fallback_primary_oof_metrics.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(out_dir / "cv_fallback_calibrated_operating_points.csv", index=False)
    pd.DataFrame(feature_rows).drop_duplicates(["endpoint", "feature_set"]).to_csv(out_dir / "cv_fallback_feature_sets.csv", index=False)

    delta_jobs: list[dict[str, Any]] = []
    for (endpoint, seed), sub in pred_df.groupby(["endpoint", "seed"], dropna=False):
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
        for metric in ["auprc", "auroc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"]:
            delta_jobs.append(
                {
                    "endpoint": endpoint,
                    "future_handling": cfg["actionability_labels"]["cv_future_handling"],
                    "model": "rf",
                    "seed": int(seed),
                    "baseline_feature_set": BASELINE,
                    "enhanced_feature_set": ENHANCED,
                    "metric": metric,
                    "merged": merged.copy(),
                }
            )
        completeness_rows.append(
            {
                "endpoint": endpoint,
                "seed": int(seed),
                "baseline_rows": int(len(base)),
                "enhanced_rows": int(len(enh)),
                "baseline_unique_sample_id": int(base["sample_id"].nunique()),
                "enhanced_unique_sample_id": int(enh["sample_id"].nunique()),
                "baseline_fold_count": int(base["outer_fold"].nunique()),
                "enhanced_fold_count": int(enh["outer_fold"].nunique()),
            }
        )
    delta_rows = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_delta)(job, n_boot, seed_for(cfg["evaluation"]["bootstrap_seed"], job["endpoint"], job["seed"], job["metric"]))
        for job in delta_jobs
    )
    pd.DataFrame(delta_rows).sort_values(["endpoint", "seed", "metric"]).to_csv(out_dir / "cv_fallback_primary_paired_deltas.csv", index=False)
    pd.DataFrame(completeness_rows).to_csv(out_dir / "cv_fallback_oof_completeness_audit.csv", index=False)
    (out_dir / "cv_fallback_bootstrap_config.json").write_text(
        json.dumps({"n_bootstrap": n_boot, "n_jobs": n_jobs, "resampling_unit": "scenario_id", "ci_type": "percentile"}, indent=2),
        encoding="utf-8",
    )
    print(f"[cv-oof] wrote outputs to {out_dir} elapsed_s={time.perf_counter()-t0:.1f}")


if __name__ == "__main__":
    main()
