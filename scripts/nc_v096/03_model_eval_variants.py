#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

from _utils import load_yaml, output_dir, resolve_path, stable_unit_float, write_csv


BASELINE_COLS = [
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "agent_count",
    "nearest_agent_rel_speed_mps",
    "nearest_agent_closing_speed_mps",
    "ttc_closing_speed_mps",
    "nearby_agent_count_10m",
    "nearby_agent_count_20m",
    "cv_rcr",
    "cv_rfr_drv",
    "cv_c_time",
    "cv_gtoa_norm_union",
    "cv_oce_norm",
    "cv_c_density",
    "cv_max_overlap_count",
]
STRICT_TEMPORAL = [
    "ttad_s",
    "time_to_first_conflict_s",
    "early_blocking_ratio",
    "collapse_rate_max_per_s",
    "collapse_rate_mean_per_s",
]
FEATURE_SETS = {
    "strong_baseline_cv": BASELINE_COLS,
    "strong_baseline_cv_plus_strict_temporal_dynamics": BASELINE_COLS + STRICT_TEMPORAL,
}


class TrainOnlyPreprocessor:
    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.imputer = SimpleImputer(strategy="median")
        self.train_medians_: dict[str, float] = {}
        self.invalid_counts_: dict[str, int] = {}

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            if "ttc" in col.lower():
                s = s.mask((s < 0) | (~np.isfinite(s)), np.nan)
            out[col] = s.replace([np.inf, -np.inf], np.nan)
        return out

    def fit(self, df: pd.DataFrame) -> "TrainOnlyPreprocessor":
        x = self._frame(df)
        arr = self.imputer.fit_transform(x)
        self.train_medians_ = {c: float(v) if np.isfinite(v) else np.nan for c, v in zip(x.columns, self.imputer.statistics_)}
        self.invalid_counts_ = {c: int(x[c].isna().sum()) for c in x.columns}
        _ = arr
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.imputer.transform(self._frame(df)), dtype=float)


def label_path(out_dir: Path, variant_id: str) -> Path:
    return out_dir / "variant_labels" / variant_id / f"labels_actionability_{variant_id}.csv"


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
    fpr = float(np.mean(neg >= thr))
    return thr, fpr


def metrics_at_thresholds(y: np.ndarray, score: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positive_count": int(np.sum(y == 1)),
        "positive_rate": float(np.mean(y)) if len(y) else np.nan,
    }
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


def pooled_metrics(y: np.ndarray, score: np.ndarray, alert_1: np.ndarray, alert_5: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(np.mean(y)) if len(y) else np.nan,
    }
    if len(np.unique(y)) >= 2:
        out["auprc"] = float(average_precision_score(y, score))
        out["auroc"] = float(roc_auc_score(y, score))
    else:
        out["auprc"] = np.nan
        out["auroc"] = np.nan
    for name, alert in [("1pct_fpr", alert_1), ("5pct_fpr", alert_5)]:
        alert = np.asarray(alert, dtype=bool)
        tp = int(np.sum(alert & (y == 1)))
        fp = int(np.sum(alert & (y == 0)))
        fn = int(np.sum((~alert) & (y == 1)))
        tn = int(np.sum((~alert) & (y == 0)))
        out[f"achieved_fpr_at_{name}"] = float(fp / max(fp + tn, 1))
        out[f"recall_at_{name}"] = float(tp / max(tp + fn, 1))
        out[f"precision_at_{name}"] = float(tp / max(tp + fp, 1)) if tp + fp else np.nan
    return out


def available_cols(cols: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [c for c in dict.fromkeys(cols) if c in df.columns]


def merge_variant_dataset(features: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    labels["sample_id"] = labels["sample_id"].astype(str)
    keep = ["sample_id", "scenario_id", "actionability_label_id", "actionability_label_name"]
    keep = [c for c in keep if c in labels.columns]
    df = features.merge(labels[keep], on="sample_id", how="inner", suffixes=("", "_label"))
    if len(df) != len(features):
        raise ValueError(f"join cardinality mismatch for {labels_path}: features={len(features)} joined={len(df)}")
    if "scenario_id" not in df.columns and "scenario_id_label" in df.columns:
        df["scenario_id"] = df["scenario_id_label"]
    if "scenario_id" not in df.columns:
        df["scenario_id"] = df["sample_id"]
    label = pd.to_numeric(df["actionability_label_id"], errors="coerce").fillna(-1).astype(int)
    df["y"] = (label >= 2).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_endpoint_design_robustness.yaml")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds; default from config.")
    parser.add_argument("--limit-variants", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    pred_path = out_dir / "waymo_design_variant_oof_predictions.csv"
    if pred_path.exists() and not args.force:
        print(f"[v096-oof] using existing predictions: {pred_path}")
        return

    features = pd.read_csv(resolve_path(cfg["inputs"]["waymo_features_csv"]))
    features["sample_id"] = features["sample_id"].astype(str)
    variants = list(cfg["actionability_labels"]["variants"])
    if args.variants:
        wanted = {v.strip() for v in str(args.variants).split(",") if v.strip()}
        variants = [v for v in variants if str(v["variant_id"]) in wanted]
    if args.limit_variants:
        variants = variants[: int(args.limit_variants)]
    seeds = [int(x) for x in (args.seeds.split(",") if args.seeds else cfg["evaluation"]["rf_seeds"])]
    fpr_levels = [float(x) for x in cfg["evaluation"]["fixed_fpr_levels"]]
    n_folds = int(cfg["evaluation"]["outer_folds"])
    fold_seed = int(cfg["evaluation"]["scenario_hash_seed"])
    cal_frac = float(cfg["evaluation"]["calibration_fraction_within_outer_train"])

    all_preds: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for v in variants:
        vid = str(v["variant_id"])
        df = merge_variant_dataset(features, label_path(out_dir, vid))
        if args.max_samples:
            df = df.sort_values("sample_id").head(int(args.max_samples)).copy()
        df["outer_fold"] = assign_outer_folds(df, n_folds, fold_seed)
        feature_sets = {name: available_cols(cols, df) for name, cols in FEATURE_SETS.items()}
        for fs_name, cols in feature_sets.items():
            feature_rows.append(
                {
                    "variant_id": vid,
                    "feature_set": fs_name,
                    "n_features": len(cols),
                    "features": ";".join(cols),
                    "missing": ";".join([c for c in FEATURE_SETS[fs_name] if c not in cols]),
                    "feature_mode": cfg["evaluation"]["feature_mode"],
                }
            )
        for seed in seeds:
            for fold in sorted(df["outer_fold"].unique()):
                test_mask = df["outer_fold"].to_numpy(int) == int(fold)
                train_cal_df = df.loc[~test_mask].copy()
                test_df = df.loc[test_mask].copy()
                cal_mask = fit_calibration_mask(train_cal_df, int(seed), int(fold), cal_frac)
                fit_df = train_cal_df.loc[~cal_mask].copy()
                cal_df = train_cal_df.loc[cal_mask].copy()
                for fs_name, cols in feature_sets.items():
                    if not cols:
                        continue
                    y_fit = fit_df["y"].to_numpy(int)
                    y_cal = cal_df["y"].to_numpy(int)
                    y_test = test_df["y"].to_numpy(int)
                    if len(np.unique(y_fit)) < 2 or len(np.unique(y_test)) < 2 or len(np.unique(y_cal)) < 2:
                        continue
                    pre = TrainOnlyPreprocessor(cols).fit(fit_df)
                    model = make_model(seed, cfg)
                    model.fit(pre.transform(fit_df), y_fit)
                    score_cal = positive_score(model, pre.transform(cal_df))
                    score_test = positive_score(model, pre.transform(test_df))
                    thresholds: dict[str, float] = {}
                    cal_fprs: dict[str, float] = {}
                    for fpr in fpr_levels:
                        label = "1pct_fpr" if abs(fpr - 0.01) < 1e-9 else "5pct_fpr" if abs(fpr - 0.05) < 1e-9 else f"{fpr:g}_fpr"
                        thr, cal_fpr = threshold_from_calibration(y_cal, score_cal, fpr)
                        thresholds[label] = thr
                        cal_fprs[label] = cal_fpr
                    fold_metrics = metrics_at_thresholds(y_test, score_test, thresholds)
                    metrics_rows.append(
                        {
                            "level": "fold",
                            "variant_id": vid,
                            "variant_family": v.get("family", ""),
                            "endpoint": "actionability_critical_or_worse",
                            "model": "rf",
                            "seed": int(seed),
                            "outer_fold": int(fold),
                            "feature_set": fs_name,
                            "feature_mode": cfg["evaluation"]["feature_mode"],
                            "n_fit": int(len(fit_df)),
                            "n_calibration": int(len(cal_df)),
                            **fold_metrics,
                        }
                    )
                    for name, thr in thresholds.items():
                        operating_rows.append(
                            {
                                "variant_id": vid,
                                "model": "rf",
                                "seed": int(seed),
                                "outer_fold": int(fold),
                                "feature_set": fs_name,
                                "nominal_fpr": 0.01 if name == "1pct_fpr" else 0.05 if name == "5pct_fpr" else name,
                                "threshold_source": "calibration_negatives_only",
                                "threshold_operator": ">=",
                                "threshold": thr,
                                "calibration_fpr": cal_fprs[name],
                                "outer_test_achieved_fpr": fold_metrics[f"achieved_fpr_at_{name}"],
                                "outer_test_recall": fold_metrics[f"recall_at_{name}"],
                                "outer_test_precision": fold_metrics[f"precision_at_{name}"],
                                "feature_mode": cfg["evaluation"]["feature_mode"],
                            }
                        )
                    pred = pd.DataFrame(
                        {
                            "sample_id": test_df["sample_id"].astype(str).to_numpy(),
                            "scenario_id": test_df["scenario_id"].astype(str).to_numpy(),
                            "variant_id": vid,
                            "variant_family": v.get("family", ""),
                            "endpoint": "actionability_critical_or_worse",
                            "model": "rf",
                            "seed": int(seed),
                            "outer_fold": int(fold),
                            "feature_set": fs_name,
                            "feature_mode": cfg["evaluation"]["feature_mode"],
                            "y_true": y_test,
                            "score": score_test,
                            "threshold_at_1pct_fpr": thresholds.get("1pct_fpr", np.nan),
                            "threshold_at_5pct_fpr": thresholds.get("5pct_fpr", np.nan),
                            "alert_at_calibrated_1pct_fpr": score_test >= thresholds.get("1pct_fpr", np.nan),
                            "alert_at_calibrated_5pct_fpr": score_test >= thresholds.get("5pct_fpr", np.nan),
                        }
                    )
                    all_preds.append(pred)
                    preprocessing_rows.append(
                        {
                            "variant_id": vid,
                            "model": "rf",
                            "seed": int(seed),
                            "outer_fold": int(fold),
                            "feature_set": fs_name,
                            "fit_sample_count": int(len(fit_df)),
                            "calibration_sample_count": int(len(cal_df)),
                            "test_sample_count": int(len(test_df)),
                            "imputer_strategy": "median",
                            "ttc_invalid_handling": "values < 0 or non-finite set to NaN before fit-split median imputation",
                            "train_medians": pre.train_medians_,
                            "fit_missing_counts": pre.invalid_counts_,
                        }
                    )
                    print(f"[v096-oof] {vid} seed={seed} fold={fold} feature_set={fs_name} elapsed_s={time.perf_counter() - t0:.1f}")

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if pred_df.empty:
        raise ValueError("no OOF predictions generated")
    pred_df.to_csv(pred_path, index=False)

    pooled = []
    for (vid, seed, fs), sub in pred_df.groupby(["variant_id", "seed", "feature_set"], dropna=False):
        m = pooled_metrics(
            sub["y_true"].to_numpy(int),
            sub["score"].to_numpy(float),
            sub["alert_at_calibrated_1pct_fpr"].astype(str).str.lower().isin(["true", "1"]).to_numpy(bool),
            sub["alert_at_calibrated_5pct_fpr"].astype(str).str.lower().isin(["true", "1"]).to_numpy(bool),
        )
        pooled.append(
            {
                "level": "pooled_oof",
                "variant_id": vid,
                "endpoint": "actionability_critical_or_worse",
                "model": "rf",
                "seed": int(seed),
                "feature_set": fs,
                "feature_mode": cfg["evaluation"]["feature_mode"],
                "n_fit": np.nan,
                "n_calibration": np.nan,
                **m,
            }
        )
    metrics = pd.DataFrame(metrics_rows + pooled)
    metrics.to_csv(out_dir / "waymo_design_variant_model_metrics.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(out_dir / "waymo_design_variant_calibrated_operating_points.csv", index=False)
    pd.DataFrame(feature_rows).drop_duplicates(["variant_id", "feature_set"]).to_csv(out_dir / "waymo_design_variant_feature_sets.csv", index=False)
    (out_dir / "waymo_design_variant_preprocessing_audit.json").write_text(json.dumps(preprocessing_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    deltas = []
    for (vid, seed), sub in pred_df.groupby(["variant_id", "seed"], dropna=False):
        base = sub[sub["feature_set"] == "strong_baseline_cv"]
        enh = sub[sub["feature_set"] == "strong_baseline_cv_plus_strict_temporal_dynamics"]
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
        y = merged["y_true"].to_numpy(int)
        b = merged["score_baseline"].to_numpy(float)
        e = merged["score_enhanced"].to_numpy(float)
        for metric in ["auprc", "auroc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"]:
            if metric == "auprc":
                bval = average_precision_score(y, b) if len(np.unique(y)) >= 2 else np.nan
                eval_ = average_precision_score(y, e) if len(np.unique(y)) >= 2 else np.nan
            elif metric == "auroc":
                bval = roc_auc_score(y, b) if len(np.unique(y)) >= 2 else np.nan
                eval_ = roc_auc_score(y, e) if len(np.unique(y)) >= 2 else np.nan
            elif metric == "recall_at_1pct_fpr":
                bval = pooled_metrics(y, b, merged["alert_1_baseline"], merged["alert_5_baseline"])["recall_at_1pct_fpr"]
                eval_ = pooled_metrics(y, e, merged["alert_1_enhanced"], merged["alert_5_enhanced"])["recall_at_1pct_fpr"]
            else:
                bval = pooled_metrics(y, b, merged["alert_1_baseline"], merged["alert_5_baseline"])["recall_at_5pct_fpr"]
                eval_ = pooled_metrics(y, e, merged["alert_1_enhanced"], merged["alert_5_enhanced"])["recall_at_5pct_fpr"]
            deltas.append(
                {
                    "variant_id": vid,
                    "endpoint": "actionability_critical_or_worse",
                    "model": "rf",
                    "seed": int(seed),
                    "baseline_feature_set": "strong_baseline_cv",
                    "enhanced_feature_set": "strong_baseline_cv_plus_strict_temporal_dynamics",
                    "metric": metric,
                    "baseline_point": float(bval) if np.isfinite(bval) else np.nan,
                    "enhanced_point": float(eval_) if np.isfinite(eval_) else np.nan,
                    "delta": float(eval_ - bval) if np.isfinite(eval_) and np.isfinite(bval) else np.nan,
                    "n_samples": int(len(merged)),
                    "n_scenarios": int(merged["scenario_id"].nunique()),
                    "positive_count": int(y.sum()),
                    "positive_rate": float(np.mean(y)),
                    "feature_mode": cfg["evaluation"]["feature_mode"],
                }
            )
        completeness_rows.append(
            {
                "variant_id": vid,
                "model": "rf",
                "seed": int(seed),
                "rows_per_feature_set": int(base["sample_id"].nunique()) if not base.empty else 0,
                "baseline_rows": int(len(base)),
                "enhanced_rows": int(len(enh)),
                "unique_sample_id_baseline": int(base["sample_id"].nunique()),
                "unique_sample_id_enhanced": int(enh["sample_id"].nunique()),
                "fold_count_baseline": int(base["outer_fold"].nunique()),
                "fold_count_enhanced": int(enh["outer_fold"].nunique()),
            }
        )
    pd.DataFrame(deltas).to_csv(out_dir / "waymo_design_variant_paired_deltas.csv", index=False)
    pd.DataFrame(completeness_rows).to_csv(out_dir / "waymo_design_variant_oof_completeness_audit.csv", index=False)
    stability = []
    for (vid, fs), sub in metrics[metrics["level"] == "pooled_oof"].groupby(["variant_id", "feature_set"], dropna=False):
        vals = pd.to_numeric(sub["auprc"], errors="coerce")
        stability.append(
            {
                "variant_id": vid,
                "feature_set": fs,
                "seed_count": int(vals.notna().sum()),
                "auprc_min": float(vals.min()) if vals.notna().any() else np.nan,
                "auprc_max": float(vals.max()) if vals.notna().any() else np.nan,
                "auprc_range": float(vals.max() - vals.min()) if vals.notna().any() else np.nan,
            }
        )
    pd.DataFrame(stability).to_csv(out_dir / "waymo_design_variant_seed_stability.csv", index=False)
    print(f"[v096-oof] wrote outputs to {out_dir} elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
