from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.analysis_utils import load_merged, scenario_hash_split, ordinal_mae, high_emergency_binary


def _feature_sets(df: pd.DataFrame):
    sets = {
        "distance_only": ["current_min_distance_m"],
        "ttc_only": ["current_ttc_s"],
        "distance_ttc": ["current_min_distance_m", "current_ttc_s"],
        # Revised CV baseline: swept / inflated constant-velocity occupancy.
        "cv_occupancy": ["cv_rcr", "cv_rfr_drv", "cv_c_time", "cv_gtoa_norm_union", "cv_oce_norm", "cv_max_overlap_count"],
        "global_overlap": ["gtoa_norm_union", "oce_norm", "max_overlap_count"],
        "rt_space_time": ["rcr", "rfr_drv", "c_time", "gtoa_norm_union", "oce_norm"],
        "rt_space_time_msr": ["rcr", "rfr_drv", "c_time", "msr", "c_maneuver", "gtoa_norm_union", "oce_norm"],
        "redi_no_msr": ["redi_no_msr"],
        "redi_full": ["redi_full"],
    }
    out = {}
    for k, cols in sets.items():
        cols2 = [c for c in cols if c in df.columns]
        if cols2:
            # If all values in a feature set are nan, skip.
            if df[cols2].apply(pd.to_numeric, errors="coerce").notna().any().any():
                out[k] = cols2
    return out


def _prep_X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # For TTC features, -1 means no closing TTC in the current implementation.
    # Treat it as a large, low-risk value for classifiers rather than as an
    # extremely small/urgent TTC.
    for c in X.columns:
        if "ttc" in c.lower():
            x = X[c].copy()
            finite_nonneg = x[(x >= 0) & np.isfinite(x)]
            large_ttc = max(float(finite_nonneg.quantile(0.95)) if len(finite_nonneg) else 10.0, 10.0)
            x[(x < 0) | (~np.isfinite(x))] = large_ttc
            X[c] = x
    for c in X.columns:
        med = X[c].median()
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(med)
    return X.to_numpy(dtype=float)


def _eval(y_true, y_pred, proba, positive_min_label: int, classes=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
    }
    pos_true = high_emergency_binary(y_true, positive_min_label)
    pos_pred = high_emergency_binary(y_pred, positive_min_label)
    out["emergency_recall"] = recall_score(pos_true, pos_pred, zero_division=0)
    out["emergency_precision"] = precision_score(pos_true, pos_pred, zero_division=0)
    if proba is not None and len(np.unique(pos_true)) == 2:
        if classes is None:
            classes = np.arange(proba.shape[1])
        classes = np.asarray(classes, dtype=int)
        mask = classes >= positive_min_label
        score = proba[:, mask].sum(axis=1) if mask.any() else np.zeros(proba.shape[0])
        try:
            out["emergency_auc"] = roc_auc_score(pos_true, score)
        except Exception:
            out["emergency_auc"] = np.nan
    else:
        out["emergency_auc"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "classification")
    df = load_merged(work)
    if len(df) < 10:
        raise SystemExit("样本太少，至少建议 >=10；正式实验应更多。")
    train_mask, test_mask = scenario_hash_split(df, float(cfg.get("analysis", {}).get("test_fraction", 0.25)), int(cfg.get("analysis", {}).get("random_seed", 42)))
    y = df["label_id"].astype(int).to_numpy()
    pos_min = int(cfg.get("analysis", {}).get("emergency_positive_label", cfg.get("labels", {}).get("high_emergency_min_label", 2)))
    rows = []
    pred_rows = []
    for name, cols in _feature_sets(df).items():
        X = _prep_X(df, cols)
        models = {
            "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
            "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2),
        }
        for mname, model in models.items():
            try:
                model.fit(X[train_mask], y[train_mask])
                pred = model.predict(X[test_mask])
                proba = model.predict_proba(X[test_mask]) if hasattr(model, "predict_proba") else None
                classes = getattr(model, "classes_", None)
                metrics = _eval(y[test_mask], pred, proba, pos_min, classes=classes)
                rows.append({"feature_set": name, "model": mname, "features": ";".join(cols), "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()), **metrics})
                for sid, yt, yp in zip(df.loc[test_mask, "sample_id"], y[test_mask], pred):
                    pred_rows.append({"sample_id": sid, "feature_set": name, "model": mname, "y_true": int(yt), "y_pred": int(yp)})
            except Exception as e:
                rows.append({"feature_set": name, "model": mname, "features": ";".join(cols), "error": str(e)})
    pd.DataFrame(rows).to_csv(out / "classification_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "classification_predictions.csv", index=False)
    print(f"[clf] wrote {out / 'classification_metrics.csv'}")

if __name__ == "__main__":
    main()
