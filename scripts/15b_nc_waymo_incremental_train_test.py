from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.analysis_utils import load_merged
from rtbev.nc_eval import available_feature_sets, binary_score_metrics, bootstrap_metric_ci, diagnostic_subset_masks, numeric_frame, y_for_task


def _parse_csv_arg(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _models(seed: int, model_names: list[str], rf_n_estimators: int, rf_n_jobs: int):
    all_models = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "rf": RandomForestClassifier(n_estimators=int(rf_n_estimators), random_state=seed, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=int(rf_n_jobs)),
    }
    missing = [m for m in model_names if m not in all_models]
    if missing:
        raise SystemExit(f"unknown --models entries: {missing}; valid={sorted(all_models)}")
    return {name: all_models[name] for name in model_names}


def _positive_score(model, X_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_test)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        if p.ndim == 2 and 1 in classes:
            return p[:, int(np.where(classes == 1)[0][0])]
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X_test), dtype=float)
        return 1.0 / (1.0 + np.exp(-z))
    return np.zeros(X_test.shape[0], dtype=float)


def _load_all_features(work: Path, features_csv: str | None) -> pd.DataFrame:
    return pd.read_csv(features_csv) if features_csv else load_merged(work)


def _filter_by_labels(features: pd.DataFrame, labels_csv: str | None) -> pd.DataFrame:
    if labels_csv is None:
        return features.copy()
    lab = pd.read_csv(labels_csv)
    ids = set(lab["sample_id"].astype(str))
    out = features[features["sample_id"].astype(str).isin(ids)].copy()
    return out


def _bootstrap_delta_job(
    task: str,
    subset_name: str,
    metric_name: str,
    yv: np.ndarray,
    score_base: np.ndarray,
    score_rof: np.ndarray,
    groups: np.ndarray,
    boot_n: int,
    seed: int,
    n: int,
    model_name: str,
) -> dict | None:
    fn = average_precision_score if metric_name == "auprc" else roc_auc_score
    try:
        b_point, b_lo, b_hi = bootstrap_metric_ci(yv, score_base, groups, fn, n_boot=boot_n, seed=seed)
        r_point, r_lo, r_hi = bootstrap_metric_ci(yv, score_rof, groups, fn, n_boot=boot_n, seed=seed + 1)
    except Exception:
        return None
    return {
        "task": task,
        "model": model_name,
        "subset": subset_name,
        "metric": metric_name,
        "baseline_point": b_point,
        "rof_v2_point": r_point,
        "delta_point": r_point - b_point,
        "baseline_ci_lo": b_lo,
        "baseline_ci_hi": b_hi,
        "rof_v2_ci_lo": r_lo,
        "rof_v2_ci_hi": r_hi,
        "n": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="NC v1.1 train/test incremental evaluation using stratified train and natural/hard test feature sets.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--features-csv", default=None, help="single all-features CSV to filter by labels")
    ap.add_argument("--train-features-csv", default=None)
    ap.add_argument("--test-features-csv", default=None)
    ap.add_argument("--train-labels-csv", default=None, help="filter all-features CSV to stratified train sample_ids")
    ap.add_argument("--test-labels-csv", default=None, help="filter all-features CSV to natural/hard test sample_ids")
    ap.add_argument("--out-name", default="train_test")
    ap.add_argument("--bootstrap-n", type=int, default=None)
    ap.add_argument("--bootstrap-n-jobs", type=int, default=-1, help="joblib n_jobs for bootstrap delta jobs")
    ap.add_argument("--feature-sets", default=None, help="comma-separated feature sets to run")
    ap.add_argument("--models", default="logreg,rf", help="comma-separated models to run: logreg,rf")
    ap.add_argument("--rf-n-estimators", type=int, default=160)
    ap.add_argument("--rf-n-jobs", type=int, default=-1)
    ap.add_argument("--skip-bootstrap", action="store_true", help="skip bootstrap delta computation and output only metrics/predictions")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "nc_waymo_incremental" / args.out_name)
    seed = int(cfg.get("analysis", {}).get("random_seed", 42))
    fpr_levels = [float(x) for x in cfg.get("nc_experiments", {}).get("fixed_fpr_levels", [0.01, 0.05])]
    boot_n = int(args.bootstrap_n if args.bootstrap_n is not None else cfg.get("nc_experiments", {}).get("bootstrap_n", 500))
    model_names = _parse_csv_arg(args.models, ["logreg", "rf"])
    selected_models = _models(seed, model_names, args.rf_n_estimators, args.rf_n_jobs)

    if args.train_features_csv and args.test_features_csv:
        train_df = pd.read_csv(args.train_features_csv)
        test_df = pd.read_csv(args.test_features_csv)
    else:
        all_df = _load_all_features(work, args.features_csv)
        train_df = _filter_by_labels(all_df, args.train_labels_csv)
        test_df = _filter_by_labels(all_df, args.test_labels_csv)
        if args.train_labels_csv is None and args.test_labels_csv is None:
            raise SystemExit("Provide train/test feature CSVs or train/test label CSVs for filtering all features.")
    if train_df.empty or test_df.empty:
        raise SystemExit(f"empty train/test after filtering: train={len(train_df)}, test={len(test_df)}")
    train_df = train_df.copy(); test_df = test_df.copy()
    train_df["label_id"] = train_df["label_id"].astype(int)
    test_df["label_id"] = test_df["label_id"].astype(int)
    # Determine feature sets using union of available columns in both frames.
    common_cols = sorted(set(train_df.columns).intersection(test_df.columns))
    fsets = available_feature_sets(train_df[common_cols])
    requested_fsets = _parse_csv_arg(args.feature_sets)
    if requested_fsets:
        missing = [name for name in requested_fsets if name not in fsets]
        if missing:
            raise SystemExit(f"unknown --feature-sets entries: {missing}; available={sorted(fsets)}")
        fsets = {name: fsets[name] for name in requested_fsets}
    tasks = ["warning_or_above", "emergency_only", "safe_vs_risky"]
    rows, pred_rows, feature_rows = [], [], []
    for fs_name, cols in fsets.items():
        cols = [c for c in cols if c in test_df.columns]
        if not cols:
            continue
        Xtr = numeric_frame(train_df, cols).to_numpy(float)
        Xte = numeric_frame(test_df, cols).to_numpy(float)
        feature_rows.append({"feature_set": fs_name, "n_features": len(cols), "features": ";".join(cols)})
        for task in tasks:
            ytr = y_for_task(train_df["label_id"].to_numpy(), task)
            yte = y_for_task(test_df["label_id"].to_numpy(), task)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                rows.append({"task": task, "feature_set": fs_name, "model": "all", "error": "single-class train/test"})
                continue
            for model_name, model in selected_models.items():
                model.fit(Xtr, ytr)
                score = _positive_score(model, Xte)
                rows.append({"task": task, "feature_set": fs_name, "model": model_name, "n_train": int(len(train_df)), "n_test": int(len(test_df)), **binary_score_metrics(yte, score, fpr_levels)})
                for idx, sid, sc, yt in zip(np.arange(len(test_df)), test_df["sample_id"].astype(str), score, yte):
                    pred_rows.append({"row_index": int(idx), "sample_id": sid, "scenario_id": str(test_df.iloc[idx].get("scenario_id", sid)), "task": task, "feature_set": fs_name, "model": model_name, "y_true": int(yt), "score": float(sc), "label_id": int(test_df.iloc[idx]["label_id"])})
    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    pd.DataFrame(feature_rows).to_csv(out / "feature_sets.csv", index=False)
    metrics.to_csv(out / "nc_classification_metrics.csv", index=False)
    preds.to_csv(out / "nc_predictions.csv", index=False)

    subset_rows = []
    if not preds.empty:
        masks = diagnostic_subset_masks(test_df)
        for (task, fs, model_name), sub in preds.groupby(["task", "feature_set", "model"]):
            idx = sub["row_index"].to_numpy(int)
            for subset_name, mask_all in masks.items():
                keep = mask_all[idx]
                if keep.sum() < 10 or len(np.unique(sub.loc[keep, "y_true"])) < 2:
                    continue
                subset_rows.append({"task": task, "feature_set": fs, "model": model_name, "subset": subset_name, "n_subset": int(keep.sum()), **binary_score_metrics(sub.loc[keep, "y_true"].to_numpy(int), sub.loc[keep, "score"].to_numpy(float), fpr_levels)})
    pd.DataFrame(subset_rows).to_csv(out / "nc_classification_metrics_by_subset.csv", index=False)

    delta_rows = []
    if args.skip_bootstrap:
        print("[nc-train-test-v1.1] skipped bootstrap deltas (--skip-bootstrap)")
    elif not preds.empty:
        key_model = "rf" if (preds["model"] == "rf").any() else str(preds["model"].iloc[0])
        masks = diagnostic_subset_masks(test_df)
        jobs = []
        for task in tasks:
            base = preds[(preds.task == task) & (preds.feature_set == "strong_baseline") & (preds.model == key_model)]
            rof = preds[(preds.task == task) & (preds.feature_set == "strong_baseline_rof_v2") & (preds.model == key_model)]
            if base.empty or rof.empty:
                continue
            merged = base[["row_index", "sample_id", "scenario_id", "y_true", "score"]].rename(columns={"score":"score_base"}).merge(rof[["sample_id","score"]].rename(columns={"score":"score_rof"}), on="sample_id", how="inner")
            for subset_name, mask_all in masks.items():
                keep = mask_all[merged["row_index"].to_numpy(int)]
                sub = merged.loc[keep]
                if len(sub) < 20 or len(np.unique(sub["y_true"])) < 2:
                    continue
                yv = sub["y_true"].to_numpy(int)
                groups = sub["scenario_id"].astype(str).to_numpy()
                score_base = sub["score_base"].to_numpy(float)
                score_rof = sub["score_rof"].to_numpy(float)
                for metric_name in ["auprc", "auroc"]:
                    jobs.append(delayed(_bootstrap_delta_job)(
                        task, subset_name, metric_name, yv, score_base, score_rof, groups, boot_n, seed, len(sub), key_model
                    ))
        delta_rows = [row for row in Parallel(n_jobs=int(args.bootstrap_n_jobs), prefer="processes")(jobs) if row is not None]
    if not args.skip_bootstrap:
        pd.DataFrame(delta_rows).to_csv(out / "strong_baseline_vs_rof_v2_bootstrap_deltas_by_subset.csv", index=False)
    print(f"[nc-train-test-v1.1] wrote {out}")


if __name__ == "__main__":
    main()
