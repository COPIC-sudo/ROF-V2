#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

from _utils import load_yaml, output_dir, resolve_path, seed_for, stable_unit_float


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
BASELINE = "strong_baseline_cv_aligned"
ENHANCED = "strong_baseline_cv_plus_strict_temporal_dynamics_aligned"
FEATURE_SETS = {BASELINE: CURRENT_REUSE + CV_FIELDS, ENHANCED: CURRENT_REUSE + CV_FIELDS + TEMPORAL_FIELDS}


class TrainOnlyPreprocessor:
    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.imputer = SimpleImputer(strategy="median")

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            if "ttc" in col.lower():
                s = s.mask((s < 0) | (~np.isfinite(s)), np.nan)
            out[col] = s.replace([np.inf, -np.inf], np.nan)
        return out

    def fit(self, df: pd.DataFrame) -> "TrainOnlyPreprocessor":
        self.imputer.fit(self._frame(df))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.imputer.transform(self._frame(df)), dtype=float)


def assign_outer_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    keys = df["scenario_id"].astype(str)
    mapping = {k: int(math.floor(stable_unit_float(f"outer:{seed}:{k}") * n_folds)) for k in keys.unique()}
    mapping = {k: min(v, n_folds - 1) for k, v in mapping.items()}
    return keys.map(mapping).astype(int)


def fit_calibration_mask(train_df: pd.DataFrame, seed: int, fold: int, calibration_fraction: float) -> np.ndarray:
    keys = train_df["scenario_id"].astype(str)
    unique = list(keys.unique())
    cal_keys = {k for k in unique if stable_unit_float(f"cal:{seed}:{fold}:{k}") < calibration_fraction}
    if not cal_keys:
        cal_keys = {min(unique, key=lambda k: stable_unit_float(f"cal:{seed}:{fold}:{k}"))}
    mask = keys.isin(cal_keys).to_numpy(bool)
    if mask.all():
        mask[:] = False
        mask[0] = True
    return mask


def make_model(seed: int, cfg: dict[str, Any]) -> RandomForestClassifier:
    mcfg = cfg["models"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=int(mcfg["n_estimators"]),
        criterion=str(mcfg.get("criterion", "gini")),
        max_depth=mcfg.get("max_depth"),
        min_samples_split=int(mcfg.get("min_samples_split", 2)),
        min_samples_leaf=int(mcfg.get("min_samples_leaf", 2)),
        max_features=mcfg.get("max_features", "sqrt"),
        bootstrap=bool(mcfg.get("bootstrap", True)),
        class_weight=mcfg.get("class_weight", "balanced_subsample"),
        random_state=int(seed),
        n_jobs=int(mcfg.get("n_jobs", -1)),
    )


def positive_score(model: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(x.shape[0], dtype=float)


def threshold_from_calibration(y_cal: np.ndarray, score_cal: np.ndarray, nominal_fpr: float) -> tuple[float, float]:
    neg = score_cal[y_cal == 0]
    if len(neg) == 0:
        return np.nan, np.nan
    thr = float(np.quantile(neg, 1.0 - float(nominal_fpr)))
    return thr, float(np.mean(neg >= thr))


def metrics_at_thresholds(y: np.ndarray, score: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    out = {"n": int(len(y)), "positive_count": int(y.sum()), "positive_rate": float(np.mean(y)) if len(y) else np.nan}
    if len(np.unique(y)) >= 2:
        out["auprc"] = float(average_precision_score(y, score))
        out["auroc"] = float(roc_auc_score(y, score))
    else:
        out["auprc"] = np.nan
        out["auroc"] = np.nan
    for name, threshold in thresholds.items():
        if not np.isfinite(threshold):
            out[f"threshold_at_{name}"] = np.nan
            out[f"achieved_fpr_at_{name}"] = np.nan
            out[f"recall_at_{name}"] = np.nan
            out[f"precision_at_{name}"] = np.nan
            continue
        pred = score >= float(threshold)
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        tn = int(np.sum((~pred) & (y == 0)))
        out[f"threshold_at_{name}"] = float(threshold)
        out[f"achieved_fpr_at_{name}"] = float(fp / max(fp + tn, 1))
        out[f"recall_at_{name}"] = float(tp / max(tp + fn, 1))
        out[f"precision_at_{name}"] = float(tp / max(tp + fp, 1)) if tp + fp else np.nan
    return out


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
        return float(np.mean(np.asarray(alert_1, dtype=bool)[pos])) if np.any(pos) else np.nan
    if metric == "recall_at_5pct_fpr":
        pos = y == 1
        return float(np.mean(np.asarray(alert_5, dtype=bool)[pos])) if np.any(pos) else np.nan
    raise ValueError(metric)


def bootstrap_one(job: dict[str, Any], n_bootstrap: int, seed: int) -> dict[str, Any]:
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
    vals = []
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
    baseline = metric_value(y, b_score, b_alert_1, b_alert_5, metric)
    enhanced = metric_value(y, e_score, e_alert_1, e_alert_5, metric)
    out = dict(job)
    out.update(
        {
            "baseline_point": float(baseline) if np.isfinite(baseline) else np.nan,
            "enhanced_point": float(enhanced) if np.isfinite(enhanced) else np.nan,
            "delta": float(enhanced - baseline) if np.isfinite(enhanced) and np.isfinite(baseline) else np.nan,
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
            "bootstrap_prob_delta_gt_0": float(np.mean(arr > 0.0)) if len(arr) else np.nan,
            "n_bootstrap_valid": int(len(arr)),
            "bootstrap_n_requested": int(n_bootstrap),
            "n_samples": int(len(y)),
            "n_scenarios": int(len(uniq)),
            "positive_count": int(y.sum()),
            "positive_rate": float(np.mean(y)),
        }
    )
    return out


def variant_feature_path(out_dir: Path, variant_id: str) -> Path:
    return out_dir / "aligned_features" / variant_id / f"features_aligned_{variant_id}.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v097/nc_v097_aligned_feature_robustness.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    pred_path = out_dir / "aligned_feature_oof_predictions.csv"
    if pred_path.exists() and not args.force:
        print(f"[v097-oof] using existing predictions: {pred_path}")
        return
    v_manifest = pd.read_csv(resolve_path(cfg["inputs"]["v096_variant_manifest_csv"]))
    variants = [v for v in cfg["aligned_features"]["variants"]]
    seeds = [int(x) for x in cfg["evaluation"]["rf_seeds"]]
    n_folds = int(cfg["evaluation"]["outer_folds"])
    cal_frac = float(cfg["evaluation"]["calibration_fraction_within_outer_train"])
    fpr_levels = [float(x) for x in cfg["evaluation"]["fixed_fpr_levels"]]
    n_boot = int(args.bootstrap_n if args.bootstrap_n is not None else cfg["evaluation"]["bootstrap_replicates"])
    n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg["evaluation"]["n_jobs"])

    pred_frames = []
    metric_rows = []
    op_rows = []
    completeness = []
    for vid in variants:
        feat = pd.read_csv(variant_feature_path(out_dir, vid))
        feat["sample_id"] = feat["sample_id"].astype(str)
        label_csv = Path(str(v_manifest.loc[v_manifest["variant_id"] == vid, "label_csv"].iloc[0]))
        labels = pd.read_csv(label_csv)
        labels["sample_id"] = labels["sample_id"].astype(str)
        df = feat.merge(labels[["sample_id", "actionability_label_id", "actionability_label_name"]], on="sample_id", how="inner")
        if len(df) != len(feat):
            raise ValueError(f"join mismatch for {vid}: features={len(feat)} joined={len(df)}")
        df["scenario_id"] = df["scenario_id"].astype(str) if "scenario_id" in df else df["sample_id"]
        df["y"] = (pd.to_numeric(df["actionability_label_id"], errors="coerce").fillna(-1).astype(int) >= 2).astype(int)
        df["outer_fold"] = assign_outer_folds(df, n_folds, int(cfg["evaluation"]["scenario_hash_seed"]))
        for seed in seeds:
            for fold in sorted(df["outer_fold"].unique()):
                test_mask = df["outer_fold"].to_numpy(int) == int(fold)
                train_cal_df = df.loc[~test_mask].copy()
                test_df = df.loc[test_mask].copy()
                cal_mask = fit_calibration_mask(train_cal_df, seed, int(fold), cal_frac)
                fit_df = train_cal_df.loc[~cal_mask].copy()
                cal_df = train_cal_df.loc[cal_mask].copy()
                for fs_name, cols in FEATURE_SETS.items():
                    y_fit = fit_df["y"].to_numpy(int)
                    y_cal = cal_df["y"].to_numpy(int)
                    y_test = test_df["y"].to_numpy(int)
                    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                        continue
                    pre = TrainOnlyPreprocessor(cols).fit(fit_df)
                    model = make_model(seed, cfg)
                    model.fit(pre.transform(fit_df), y_fit)
                    score_cal = positive_score(model, pre.transform(cal_df))
                    score_test = positive_score(model, pre.transform(test_df))
                    thresholds = {}
                    cal_fprs = {}
                    for fpr in fpr_levels:
                        label = "1pct_fpr" if abs(fpr - 0.01) < 1e-9 else "5pct_fpr" if abs(fpr - 0.05) < 1e-9 else f"{fpr:g}_fpr"
                        thr, cal_fpr = threshold_from_calibration(y_cal, score_cal, fpr)
                        thresholds[label] = thr
                        cal_fprs[label] = cal_fpr
                    m = metrics_at_thresholds(y_test, score_test, thresholds)
                    metric_rows.append(
                        {
                            "level": "fold",
                            "variant_id": vid,
                            "endpoint": cfg["evaluation"]["endpoint"],
                            "model": "rf",
                            "seed": seed,
                            "outer_fold": int(fold),
                            "feature_set": fs_name,
                            "n_fit": int(len(fit_df)),
                            "n_calibration": int(len(cal_df)),
                            **m,
                        }
                    )
                    for name, thr in thresholds.items():
                        op_rows.append(
                            {
                                "variant_id": vid,
                                "model": "rf",
                                "seed": seed,
                                "outer_fold": int(fold),
                                "feature_set": fs_name,
                                "nominal_fpr": 0.01 if name == "1pct_fpr" else 0.05,
                                "threshold_source": "calibration_negatives_only",
                                "threshold_operator": ">=",
                                "threshold": thr,
                                "calibration_fpr": cal_fprs[name],
                                "outer_test_achieved_fpr": m[f"achieved_fpr_at_{name}"],
                                "outer_test_recall": m[f"recall_at_{name}"],
                                "outer_test_precision": m[f"precision_at_{name}"],
                            }
                        )
                    pred_frames.append(
                        pd.DataFrame(
                            {
                                "sample_id": test_df["sample_id"].astype(str).to_numpy(),
                                "scenario_id": test_df["scenario_id"].astype(str).to_numpy(),
                                "variant_id": vid,
                                "endpoint": cfg["evaluation"]["endpoint"],
                                "model": "rf",
                                "seed": seed,
                                "outer_fold": int(fold),
                                "feature_set": fs_name,
                                "y_true": y_test,
                                "score": score_test,
                                "alert_at_calibrated_1pct_fpr": score_test >= thresholds.get("1pct_fpr", np.nan),
                                "alert_at_calibrated_5pct_fpr": score_test >= thresholds.get("5pct_fpr", np.nan),
                            }
                        )
                    )
                    print(f"[v097-oof] {vid} seed={seed} fold={fold} fs={fs_name} elapsed_s={time.perf_counter()-t0:.1f}")

    pred = pd.concat(pred_frames, ignore_index=True)
    pred.to_csv(pred_path, index=False)
    pooled = []
    for (vid, seed, fs), sub in pred.groupby(["variant_id", "seed", "feature_set"], dropna=False):
        y = sub["y_true"].to_numpy(int)
        score = sub["score"].to_numpy(float)
        a1 = bool_array(sub["alert_at_calibrated_1pct_fpr"])
        a5 = bool_array(sub["alert_at_calibrated_5pct_fpr"])
        pooled.append(
            {
                "level": "pooled_oof",
                "variant_id": vid,
                "endpoint": cfg["evaluation"]["endpoint"],
                "model": "rf",
                "seed": int(seed),
                "feature_set": fs,
                "n": int(len(y)),
                "positive_count": int(y.sum()),
                "positive_rate": float(np.mean(y)),
                "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) >= 2 else np.nan,
                "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) >= 2 else np.nan,
                "recall_at_1pct_fpr": metric_value(y, score, a1, a5, "recall_at_1pct_fpr"),
                "recall_at_5pct_fpr": metric_value(y, score, a1, a5, "recall_at_5pct_fpr"),
            }
        )
    pd.DataFrame(metric_rows + pooled).to_csv(out_dir / "aligned_feature_model_metrics.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out_dir / "aligned_feature_calibrated_operating_points.csv", index=False)

    delta_jobs = []
    point_rows = []
    for (vid, seed), sub in pred.groupby(["variant_id", "seed"], dropna=False):
        base = sub[sub["feature_set"] == BASELINE]
        enh = sub[sub["feature_set"] == ENHANCED]
        merged = base[["sample_id", "scenario_id", "y_true", "score", "alert_at_calibrated_1pct_fpr", "alert_at_calibrated_5pct_fpr"]].rename(
            columns={"score": "score_baseline", "alert_at_calibrated_1pct_fpr": "alert_1_baseline", "alert_at_calibrated_5pct_fpr": "alert_5_baseline"}
        ).merge(
            enh[["sample_id", "score", "alert_at_calibrated_1pct_fpr", "alert_at_calibrated_5pct_fpr"]].rename(
                columns={"score": "score_enhanced", "alert_at_calibrated_1pct_fpr": "alert_1_enhanced", "alert_at_calibrated_5pct_fpr": "alert_5_enhanced"}
            ),
            on="sample_id",
            how="inner",
        )
        y = merged["y_true"].to_numpy(int)
        for metric in ["auprc", "auroc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"]:
            job = {
                "variant_id": vid,
                "endpoint": cfg["evaluation"]["endpoint"],
                "model": "rf",
                "seed": int(seed),
                "baseline_feature_set": BASELINE,
                "enhanced_feature_set": ENHANCED,
                "metric": metric,
                "merged": merged.copy(),
            }
            delta_jobs.append(job)
            b = metric_value(y, merged["score_baseline"], bool_array(merged["alert_1_baseline"]), bool_array(merged["alert_5_baseline"]), metric)
            e = metric_value(y, merged["score_enhanced"], bool_array(merged["alert_1_enhanced"]), bool_array(merged["alert_5_enhanced"]), metric)
            point_rows.append({k: v for k, v in job.items() if k != "merged"} | {"baseline_point": b, "enhanced_point": e, "delta": e - b, "n_samples": int(len(y)), "positive_count": int(y.sum()), "positive_rate": float(np.mean(y))})
        completeness.append({"variant_id": vid, "seed": int(seed), "baseline_rows": int(len(base)), "enhanced_rows": int(len(enh)), "fold_count": int(base["outer_fold"].nunique())})
    pd.DataFrame(point_rows).to_csv(out_dir / "aligned_feature_paired_deltas.csv", index=False)
    boot_rows = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_one)(job, n_boot, seed_for(cfg["evaluation"]["bootstrap_seed"], job["variant_id"], job["seed"], job["metric"]))
        for job in delta_jobs
    )
    pd.DataFrame(boot_rows).sort_values(["variant_id", "seed", "metric"]).to_csv(out_dir / "aligned_feature_bootstrap_deltas.csv", index=False)
    pd.DataFrame(completeness).to_csv(out_dir / "aligned_feature_oof_completeness_audit.csv", index=False)
    print(f"[v097-oof] wrote aligned OOF outputs elapsed_s={time.perf_counter()-t0:.1f}")


if __name__ == "__main__":
    main()
