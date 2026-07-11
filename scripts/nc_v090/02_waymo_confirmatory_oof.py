#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Endpoint:
    name: str
    label_path_key: str
    positive_rule: str
    report_name: str


ENDPOINTS = [
    Endpoint("map_critical_or_worse", "waymo_actionability_map_labels_csv", "ge2", "map-constrained critical-or-worse actionability"),
    Endpoint("map_candidate_set_infeasible", "waymo_actionability_map_labels_csv", "eq3", "map-constrained candidate-set infeasible"),
    Endpoint("nomap_critical_or_worse", "waymo_actionability_nomap_labels_csv", "ge2", "no-map critical-or-worse actionability"),
    Endpoint("nomap_candidate_set_infeasible", "waymo_actionability_nomap_labels_csv", "eq3", "no-map candidate-set infeasible exploratory"),
]

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
STRICT_SPATIAL = ["rcr", "rfr_drv", "c_time", "gtoa_norm_union", "oce_norm", "c_density"]
EXPLICIT_RATIO_EXCLUDED_CURRENT = [
    "redi_actionability",
    "redi_actionability_delta",
    "ttad_s",
    "time_to_first_conflict_s",
    "early_blocking_ratio",
    "collapse_rate_max_per_s",
    "collapse_rate_mean_per_s",
    "slice_survival_keep",
    "slice_survival_brake",
    "slice_survival_left",
    "slice_survival_right",
    "slice_survival_brake_left",
    "slice_survival_brake_right",
]
DIRECT_ACTION_RATIOS = [
    "comfort_asr",
    "emergency_asr",
    "comfort_to_emergency_gap",
    "asr_slice_final",
    "asr_slice_min",
    "asr_cum_final",
    "asr_cum_min",
]
FEATURE_SETS = {
    "strong_baseline_cv": BASELINE_COLS,
    "strong_baseline_cv_plus_strict_temporal_dynamics": BASELINE_COLS + STRICT_TEMPORAL,
    "strong_baseline_cv_plus_strict_spatial_no_action": BASELINE_COLS + STRICT_SPATIAL,
    "strong_baseline_cv_plus_explicit_ratio_field_excluded_current": BASELINE_COLS + EXPLICIT_RATIO_EXCLUDED_CURRENT,
    "strong_baseline_cv_plus_direct_action_ratios_only": BASELINE_COLS + DIRECT_ACTION_RATIOS,
}


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


def stable_unit_float(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)


def assign_outer_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    keys = df["scenario_id"].astype(str)
    mapping = {k: int(math.floor(stable_unit_float(f"outer:{seed}:{k}") * n_folds)) for k in keys.unique()}
    mapping = {k: min(v, n_folds - 1) for k, v in mapping.items()}
    return keys.map(mapping).astype(int)


def fit_calibration_mask(train_df: pd.DataFrame, seed: int, fold: int, calibration_fraction: float) -> np.ndarray:
    keys = train_df["scenario_id"].astype(str)
    cal_keys = {k for k in keys.unique() if stable_unit_float(f"cal:{seed}:{fold}:{k}") < calibration_fraction}
    if not cal_keys:
        cal_keys = {min(keys.unique(), key=lambda k: stable_unit_float(f"cal:{seed}:{fold}:{k}"))}
    mask = keys.isin(cal_keys).to_numpy(bool)
    if mask.all():
        mask[:] = False
        mask[0] = True
    return mask


def load_dataset(cfg: dict[str, Any], endpoint: Endpoint) -> pd.DataFrame:
    features = pd.read_csv(cfg["inputs"]["waymo_features_csv"])
    labels = pd.read_csv(cfg["inputs"][endpoint.label_path_key])
    features["sample_id"] = features["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)
    if features["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id in features")
    if labels["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample_id in labels for {endpoint.name}")
    keep = ["sample_id", "scenario_id", "actionability_label_id", "actionability_label_name"]
    keep = [c for c in keep if c in labels.columns]
    df = features.merge(labels[keep], on="sample_id", how="inner", suffixes=("", "_label"))
    if len(df) != len(features):
        raise ValueError(f"join cardinality mismatch for {endpoint.name}: features={len(features)} joined={len(df)}")
    if "scenario_id" not in df.columns and "scenario_id_label" in df.columns:
        df["scenario_id"] = df["scenario_id_label"]
    if "scenario_id" not in df.columns:
        df["scenario_id"] = df["sample_id"]
    label = pd.to_numeric(df["actionability_label_id"], errors="coerce").fillna(-1).astype(int)
    if endpoint.positive_rule == "ge2":
        df["y"] = (label >= 2).astype(int)
    elif endpoint.positive_rule == "eq3":
        df["y"] = (label == 3).astype(int)
    else:
        raise ValueError(endpoint.positive_rule)
    return df


class TrainOnlyPreprocessor:
    def __init__(self, columns: list[str], scale: bool = False):
        self.columns = list(columns)
        self.scale = bool(scale)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if self.scale else None
        self.feature_names_: list[str] = []
        self.train_medians_: dict[str, float] = {}
        self.invalid_counts_: dict[str, int] = {}

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            if "ttc" in col.lower():
                invalid = (s < 0) | (~np.isfinite(s))
                s = s.mask(invalid, np.nan)
            out[col] = s.replace([np.inf, -np.inf], np.nan)
        return out

    def fit(self, df: pd.DataFrame) -> "TrainOnlyPreprocessor":
        x = self._frame(df)
        self.feature_names_ = list(x.columns)
        self.invalid_counts_ = {c: int(x[c].isna().sum()) for c in x.columns}
        arr = self.imputer.fit_transform(x)
        self.train_medians_ = {c: float(v) if np.isfinite(v) else np.nan for c, v in zip(x.columns, self.imputer.statistics_)}
        if self.scaler is not None:
            self.scaler.fit(arr)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._frame(df)
        arr = self.imputer.transform(x)
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return np.asarray(arr, dtype=float)


def make_model(model_name: str, seed: int, cfg: dict[str, Any]):
    if model_name == "rf":
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
    if model_name == "logreg":
        lcfg = cfg["models"]["logistic_regression"]
        return LogisticRegression(
            penalty=str(lcfg.get("penalty", "l2")),
            C=float(lcfg.get("C", 1.0)),
            solver=str(lcfg.get("solver", "lbfgs")),
            class_weight=lcfg.get("class_weight", "balanced"),
            max_iter=int(lcfg.get("max_iter", 2000)),
            tol=float(lcfg.get("tol", 1e-4)),
            fit_intercept=bool(lcfg.get("fit_intercept", True)),
            random_state=int(seed),
        )
    raise ValueError(model_name)


def positive_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        if p.ndim == 2 and 1 in classes:
            return p[:, int(np.where(classes == 1)[0][0])]
    z = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def threshold_from_calibration(y_cal: np.ndarray, score_cal: np.ndarray, nominal_fpr: float) -> tuple[float, float]:
    neg = score_cal[y_cal == 0]
    if len(neg) == 0:
        return np.nan, np.nan
    thr = float(np.quantile(neg, 1.0 - float(nominal_fpr)))
    fpr = float(np.mean(neg >= thr))
    return thr, fpr


def metrics_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    out = {
        "n": int(len(y)),
        "positive_count": int(np.sum(y == 1)),
        "prevalence": float(np.mean(y)) if len(y) else np.nan,
    }
    if len(np.unique(y)) >= 2:
        out["auprc"] = float(average_precision_score(y, score))
        out["auroc"] = float(roc_auc_score(y, score))
    else:
        out["auprc"] = np.nan
        out["auroc"] = np.nan
    if not np.isfinite(threshold):
        out.update({"threshold": np.nan, "achieved_fpr": np.nan, "recall": np.nan, "precision": np.nan, "specificity": np.nan, "alert_rate": np.nan})
        return out
    pred = score >= float(threshold)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    out.update({
        "threshold": float(threshold),
        "achieved_fpr": float(fp / max(fp + tn, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)) if (tp + fp) else np.nan,
        "specificity": float(tn / max(tn + fp, 1)),
        "alert_rate": float(np.mean(pred)) if len(pred) else np.nan,
    })
    return out


def pooled_metrics_from_predictions(y: np.ndarray, score: np.ndarray, alert: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    alert = np.asarray(alert, dtype=bool)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    alert = alert[ok]
    out = {
        "n": int(len(y)),
        "positive_count": int(np.sum(y == 1)),
        "prevalence": float(np.mean(y)) if len(y) else np.nan,
        "threshold": np.nan,
    }
    if len(np.unique(y)) >= 2:
        out["auprc"] = float(average_precision_score(y, score))
        out["auroc"] = float(roc_auc_score(y, score))
    else:
        out["auprc"] = np.nan
        out["auroc"] = np.nan
    tp = int(np.sum(alert & (y == 1)))
    fp = int(np.sum(alert & (y == 0)))
    fn = int(np.sum((~alert) & (y == 1)))
    tn = int(np.sum((~alert) & (y == 0)))
    out.update({
        "achieved_fpr": float(fp / max(fp + tn, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)) if (tp + fp) else np.nan,
        "specificity": float(tn / max(tn + fp, 1)),
        "alert_rate": float(np.mean(alert)) if len(alert) else np.nan,
    })
    return out


def bootstrap_delta_job(task: tuple[str, str, str, str, str], merged: pd.DataFrame, metric: str, n_boot: int, seed: int) -> dict[str, Any]:
    endpoint, model, baseline, enhanced, seed_text = task
    y = merged["y_true"].to_numpy(int)
    b = merged["score_baseline"].to_numpy(float)
    e = merged["score_enhanced"].to_numpy(float)
    b_alert = merged["alert_baseline"].to_numpy(bool)
    e_alert = merged["alert_enhanced"].to_numpy(bool)
    groups = merged["scenario_id"].astype(str).to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "auprc":
            vals.append(float(average_precision_score(yy, e[idx]) - average_precision_score(yy, b[idx])))
        elif metric == "auroc":
            vals.append(float(roc_auc_score(yy, e[idx]) - roc_auc_score(yy, b[idx])))
        elif metric == "recall_at_5pct_fpr":
            vals.append(float(pooled_metrics_from_predictions(yy, e[idx], e_alert[idx])["recall"] - pooled_metrics_from_predictions(yy, b[idx], b_alert[idx])["recall"]))
    arr = np.asarray(vals, dtype=float)
    point = np.nan
    if len(np.unique(y)) >= 2:
        if metric == "auprc":
            point = float(average_precision_score(y, e) - average_precision_score(y, b))
        elif metric == "auroc":
            point = float(roc_auc_score(y, e) - roc_auc_score(y, b))
        elif metric == "recall_at_5pct_fpr":
            point = float(pooled_metrics_from_predictions(y, e, e_alert)["recall"] - pooled_metrics_from_predictions(y, b, b_alert)["recall"])
    return {
        "endpoint": endpoint,
        "model": model,
        "seed": seed_text,
        "baseline_feature_set": baseline,
        "enhanced_feature_set": enhanced,
        "metric": metric,
        "delta": point,
        "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
        "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
        "n_bootstrap_valid": int(len(arr)),
        "n_scenarios": int(len(uniq)),
        "n_samples": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(np.mean(y)),
    }


def available_cols(cols: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [c for c in dict.fromkeys(cols) if c in df.columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v090/nc_v090_audit.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    repo = Path.cwd()
    cfg = load_yaml(repo / args.config)
    out_dir = repo / cfg["project"]["output_dir"]
    ckpt_dir = out_dir / "checkpoints" / ("waymo_oof_smoke" if args.smoke else "waymo_oof_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_folds = int(cfg["splits"]["outer_folds"])
    rf_seeds = list(cfg["splits"]["rf_seeds"])
    if args.smoke:
        rf_seeds = [42]
        n_folds = min(2, n_folds)
    bootstrap_n = int(args.bootstrap_n if args.bootstrap_n is not None else cfg["evaluation"]["bootstrap_replicates"])
    if args.smoke:
        bootstrap_n = min(20, bootstrap_n)
    nominal_fpr = float(cfg["evaluation"]["fixed_fpr_nominal"])

    fold_assignments_written = False
    all_prediction_paths: list[Path] = []
    metrics_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for endpoint in ENDPOINTS:
        df = load_dataset(cfg, endpoint)
        if args.max_samples:
            df = df.sort_values("sample_id").head(int(args.max_samples)).copy()
        df["outer_fold"] = assign_outer_folds(df, n_folds, int(cfg["splits"]["scenario_hash_seed"]))
        if not fold_assignments_written and not args.smoke:
            df[["sample_id", "scenario_id", "outer_fold"]].to_csv(out_dir / "waymo_fold_assignments.csv", index=False)
            fold_assignments_written = True
        if args.smoke:
            df[["sample_id", "scenario_id", "outer_fold"]].to_csv(out_dir / "waymo_fold_assignments_smoke.csv", index=False)
        feature_sets = {name: available_cols(cols, df) for name, cols in FEATURE_SETS.items()}
        for fs_name, cols in feature_sets.items():
            feature_rows.append({"feature_set": fs_name, "n_features": len(cols), "features": ";".join(cols), "missing": ";".join([c for c in FEATURE_SETS[fs_name] if c not in cols])})
        jobs: list[tuple[str, int, int, str, str]] = []
        for seed in rf_seeds:
            for fold in sorted(df["outer_fold"].unique()):
                for fs_name, cols in feature_sets.items():
                    if not cols:
                        continue
                    jobs.append((endpoint.name, int(seed), int(fold), "rf", fs_name))
        # Logistic regression sensitivity only for primary endpoint and strict temporal comparison.
        if endpoint.name == "map_critical_or_worse":
            for fold in sorted(df["outer_fold"].unique()):
                for fs_name in ["strong_baseline_cv", "strong_baseline_cv_plus_strict_temporal_dynamics"]:
                    jobs.append((endpoint.name, 42, int(fold), "logreg", fs_name))

        for _, seed, fold, model_name, fs_name in jobs:
            pred_path = ckpt_dir / f"{endpoint.name}__seed{seed}__fold{fold}__{model_name}__{fs_name}.csv"
            meta_path = ckpt_dir / f"{endpoint.name}__seed{seed}__fold{fold}__{model_name}__{fs_name}.json"
            if pred_path.exists() and meta_path.exists() and not args.force:
                all_prediction_paths.append(pred_path)
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                metrics_rows.extend(meta.get("metrics_rows", []))
                operating_rows.extend(meta.get("operating_rows", []))
                preprocessing_rows.extend(meta.get("preprocessing_rows", []))
                continue
            test_mask = df["outer_fold"].to_numpy(int) == int(fold)
            train_cal_df = df.loc[~test_mask].copy()
            test_df = df.loc[test_mask].copy()
            cal_mask = fit_calibration_mask(train_cal_df, int(seed), int(fold), float(cfg["splits"]["calibration_fraction_within_outer_train"]))
            fit_df = train_cal_df.loc[~cal_mask].copy()
            cal_df = train_cal_df.loc[cal_mask].copy()
            cols = feature_sets[fs_name]
            pre = TrainOnlyPreprocessor(cols, scale=(model_name == "logreg")).fit(fit_df)
            x_fit = pre.transform(fit_df)
            x_cal = pre.transform(cal_df)
            x_test = pre.transform(test_df)
            y_fit = fit_df["y"].to_numpy(int)
            y_cal = cal_df["y"].to_numpy(int)
            y_test = test_df["y"].to_numpy(int)
            if len(np.unique(y_fit)) < 2 or len(np.unique(y_test)) < 2:
                continue
            model = make_model(model_name, int(seed), cfg)
            model.fit(x_fit, y_fit)
            score_cal = positive_score(model, x_cal)
            score_test = positive_score(model, x_test)
            threshold, cal_fpr = threshold_from_calibration(y_cal, score_cal, nominal_fpr)
            fold_metrics = metrics_at_threshold(y_test, score_test, threshold)
            metric_row = {
                "level": "fold",
                "endpoint": endpoint.name,
                "endpoint_report_name": endpoint.report_name,
                "model": model_name,
                "seed": int(seed),
                "outer_fold": int(fold),
                "feature_set": fs_name,
                "n_fit": int(len(fit_df)),
                "n_calibration": int(len(cal_df)),
                **fold_metrics,
            }
            metrics_rows.append(metric_row)
            operating_rows.append({
                "endpoint": endpoint.name,
                "model": model_name,
                "seed": int(seed),
                "outer_fold": int(fold),
                "feature_set": fs_name,
                "nominal_fpr": nominal_fpr,
                "threshold_source": "calibration_negatives_only",
                "threshold_operator": ">=",
                "threshold": threshold,
                "calibration_fpr": cal_fpr,
                "outer_test_achieved_fpr": fold_metrics["achieved_fpr"],
                "outer_test_recall": fold_metrics["recall"],
                "outer_test_precision": fold_metrics["precision"],
            })
            preprocessing_rows.append({
                "endpoint": endpoint.name,
                "model": model_name,
                "seed": int(seed),
                "outer_fold": int(fold),
                "feature_set": fs_name,
                "fit_sample_count": int(len(fit_df)),
                "calibration_sample_count": int(len(cal_df)),
                "test_sample_count": int(len(test_df)),
                "imputer_strategy": "median",
                "scaler": "StandardScaler" if model_name == "logreg" else "none",
                "ttc_invalid_handling": "values < 0 or non-finite set to NaN before fit-split median imputation",
                "train_medians": pre.train_medians_,
                "fit_missing_counts": pre.invalid_counts_,
            })
            pred = pd.DataFrame({
                "sample_id": test_df["sample_id"].astype(str).to_numpy(),
                "scenario_id": test_df["scenario_id"].astype(str).to_numpy(),
                "endpoint": endpoint.name,
                "endpoint_report_name": endpoint.report_name,
                "model": model_name,
                "seed": int(seed),
                "outer_fold": int(fold),
                "feature_set": fs_name,
                "y_true": y_test,
                "score": score_test,
                "threshold": threshold,
                "alert_at_calibrated_5pct_fpr": score_test >= threshold,
            })
            pred.to_csv(pred_path, index=False)
            meta_path.write_text(json.dumps({"metrics_rows": [metric_row], "operating_rows": operating_rows[-1:], "preprocessing_rows": preprocessing_rows[-1:]}, indent=2), encoding="utf-8")
            all_prediction_paths.append(pred_path)
            print(f"[oof] wrote {pred_path.relative_to(repo)}")

    pred_df = pd.concat([pd.read_csv(p) for p in all_prediction_paths], ignore_index=True) if all_prediction_paths else pd.DataFrame()
    if pred_df.empty:
        raise ValueError("no OOF predictions were generated")

    pooled_rows = []
    for (endpoint, model, seed, fs), sub in pred_df.groupby(["endpoint", "model", "seed", "feature_set"], dropna=False):
        m = pooled_metrics_from_predictions(
            sub["y_true"].to_numpy(int),
            sub["score"].to_numpy(float),
            sub["alert_at_calibrated_5pct_fpr"].to_numpy(bool),
        )
        pooled_rows.append({
            "level": "pooled_oof",
            "endpoint": endpoint,
            "model": model,
            "seed": int(seed),
            "feature_set": fs,
            "n_fit": np.nan,
            "n_calibration": np.nan,
            **m,
        })
    metrics = pd.DataFrame(metrics_rows + pooled_rows)
    metrics.to_csv(out_dir / ("waymo_confirmatory_metrics_smoke.csv" if args.smoke else "waymo_confirmatory_metrics.csv"), index=False)
    pd.DataFrame(operating_rows).to_csv(out_dir / ("waymo_calibrated_operating_points_smoke.csv" if args.smoke else "waymo_calibrated_operating_points.csv"), index=False)
    pd.DataFrame(feature_rows).drop_duplicates("feature_set").to_csv(out_dir / ("waymo_feature_sets_smoke.csv" if args.smoke else "waymo_feature_sets.csv"), index=False)
    (out_dir / ("waymo_preprocessing_audit_smoke.json" if args.smoke else "waymo_preprocessing_audit.json")).write_text(json.dumps(preprocessing_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    pred_csv = out_dir / ("waymo_oof_predictions_smoke.csv" if args.smoke else "waymo_oof_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    parquet_path = out_dir / ("waymo_oof_predictions_smoke.parquet" if args.smoke else "waymo_oof_predictions.parquet")
    blockers = []
    try:
        pred_df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        blockers.append({
            "category": "dependency",
            "item": "parquet_engine",
            "status": "BLOCKED",
            "details": f"{type(exc).__name__}: {exc}",
            "resume_command": f"conda install -n waymo_rt_bev pyarrow; conda run -n waymo_rt_bev python scripts/nc_v090/02_waymo_confirmatory_oof.py --config {args.config}",
        })
        (out_dir / f"{parquet_path.name}.BLOCKED.txt").write_text(json.dumps(blockers[-1], indent=2), encoding="utf-8")

    # Paired deltas on pooled OOF predictions.
    delta_rows: list[dict[str, Any]] = []
    comparisons = [(b, e) for b in ["strong_baseline_cv"] for e in FEATURE_SETS if e != b]
    delta_jobs = []
    for (endpoint, model, seed), sub in pred_df.groupby(["endpoint", "model", "seed"], dropna=False):
        if model == "logreg" and endpoint != "map_critical_or_worse":
            continue
        for baseline, enhanced in comparisons:
            base = sub[sub["feature_set"] == baseline]
            enh = sub[sub["feature_set"] == enhanced]
            if base.empty or enh.empty:
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
            if len(merged) != len(base) or len(merged) != len(enh):
                continue
            for metric in ["auprc", "auroc", "recall_at_5pct_fpr"]:
                delta_jobs.append(((str(endpoint), str(model), str(baseline), str(enhanced), str(seed)), merged, metric, bootstrap_n, int(cfg["evaluation"]["bootstrap_seed"]) + int(seed)))
    if args.n_jobs == 1 or len(delta_jobs) <= 1:
        delta_rows = [bootstrap_delta_job(*job) for job in delta_jobs]
    else:
        delta_rows = Parallel(n_jobs=int(args.n_jobs))(delayed(bootstrap_delta_job)(*job) for job in delta_jobs)
    pd.DataFrame(delta_rows).to_csv(out_dir / ("waymo_paired_deltas_smoke.csv" if args.smoke else "waymo_paired_deltas.csv"), index=False)

    stability = []
    for (endpoint, model, fs), sub in metrics[metrics["level"] == "pooled_oof"].groupby(["endpoint", "model", "feature_set"], dropna=False):
        vals = pd.to_numeric(sub["auprc"], errors="coerce")
        stability.append({
            "endpoint": endpoint,
            "model": model,
            "feature_set": fs,
            "seed_count": int(vals.notna().sum()),
            "auprc_min": float(vals.min()) if vals.notna().any() else np.nan,
            "auprc_max": float(vals.max()) if vals.notna().any() else np.nan,
            "auprc_range": float(vals.max() - vals.min()) if vals.notna().any() else np.nan,
        })
    pd.DataFrame(stability).to_csv(out_dir / ("waymo_fold_seed_stability_smoke.csv" if args.smoke else "waymo_fold_seed_stability.csv"), index=False)

    if blockers:
        write_csv(out_dir / "BLOCKERS.csv", blockers)
    print(f"[oof] wrote outputs to {out_dir}")
    print(f"[oof] predictions_csv={pred_csv}")
    print(f"[oof] parquet={'ok' if parquet_path.exists() else 'blocked'}")
    print(f"[oof] elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
