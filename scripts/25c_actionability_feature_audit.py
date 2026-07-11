from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.nc_eval import binary_score_metrics, numeric_frame


STRONG_BASELINE_COLS = [
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
CV_COLS = [
    "cv_rcr",
    "cv_rfr_drv",
    "cv_c_time",
    "cv_gtoa_norm_union",
    "cv_oce_norm",
    "cv_c_density",
    "cv_max_overlap_count",
]
DIRECT_ACTION_RATIO_COLS = [
    "comfort_asr",
    "emergency_asr",
    "comfort_to_emergency_gap",
    "asr_slice_final",
    "asr_slice_min",
    "asr_cum_final",
    "asr_cum_min",
]
TEMPORAL_ACTIONABILITY_COLS = [
    "ttad_s",
    "time_to_first_conflict_s",
    "early_blocking_ratio",
    "collapse_rate_max_per_s",
    "collapse_rate_mean_per_s",
]
SPATIAL_ROF_COLS = [
    "rcr",
    "rfr_drv",
    "c_time",
    "gtoa_norm_union",
    "oce_norm",
    "c_density",
    "msr",
    "c_maneuver",
    "redi_full",
    "redi_no_msr",
]
REDI_ACTIONABILITY_COLS = [
    "redi_actionability",
    "redi_actionability_delta",
]
SURVIVAL_COLS = [
    "slice_survival_keep",
    "slice_survival_brake",
    "slice_survival_left",
    "slice_survival_right",
    "slice_survival_brake_left",
    "slice_survival_brake_right",
]
ACTIONABILITY_NO_DIRECT_COLS = (
    REDI_ACTIONABILITY_COLS
    + TEMPORAL_ACTIONABILITY_COLS
    + SURVIVAL_COLS
)
ACTIONABILITY_ALL_COLS = (
    DIRECT_ACTION_RATIO_COLS
    + TEMPORAL_ACTIONABILITY_COLS
    + SPATIAL_ROF_COLS
    + REDI_ACTIONABILITY_COLS
    + SURVIVAL_COLS
)


def _parse_csv_arg(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _unique(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _stable_row_id(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]


def _scenario_hash_split(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    key_col = "scenario_id" if "scenario_id" in df.columns else "sample_id"
    keys = df[key_col].astype(str)
    unique_keys = keys.drop_duplicates().to_numpy()
    if len(unique_keys) <= 1:
        test = np.zeros(len(df), dtype=bool)
        return ~test, test
    denom = float(16**16)
    scores = {}
    for key in unique_keys:
        digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()[:16]
        scores[key] = int(digest, 16) / denom
    test_keys = {key for key, value in scores.items() if value < float(test_fraction)}
    if not test_keys:
        test_keys = {min(scores, key=scores.get)}
    if len(test_keys) == len(unique_keys):
        test_keys = set(sorted(scores, key=scores.get)[: max(1, len(unique_keys) - 1)])
    test = keys.isin(test_keys).to_numpy(bool)
    return ~test, test


def _feature_groups(columns: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    definitions = {
        "strong_baseline": STRONG_BASELINE_COLS,
        "strong_baseline_cv": STRONG_BASELINE_COLS + CV_COLS,
        "direct_action_ratios_only": DIRECT_ACTION_RATIO_COLS,
        "temporal_actionability_only": TEMPORAL_ACTIONABILITY_COLS,
        "spatial_rof_only": SPATIAL_ROF_COLS,
        "redi_actionability_only": REDI_ACTIONABILITY_COLS,
        "actionability_no_direct_ratios": ACTIONABILITY_NO_DIRECT_COLS,
        "strong_baseline_cv_direct_ratios": STRONG_BASELINE_COLS + CV_COLS + DIRECT_ACTION_RATIO_COLS,
        "strong_baseline_cv_temporal_actionability": STRONG_BASELINE_COLS + CV_COLS + TEMPORAL_ACTIONABILITY_COLS,
        "strong_baseline_cv_spatial_rof": STRONG_BASELINE_COLS + CV_COLS + SPATIAL_ROF_COLS,
        "strong_baseline_cv_actionability_no_direct_ratios": STRONG_BASELINE_COLS + CV_COLS + ACTIONABILITY_NO_DIRECT_COLS,
        "strong_baseline_cv_actionability_all": STRONG_BASELINE_COLS + CV_COLS + ACTIONABILITY_ALL_COLS,
    }
    actual: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for name, cols in definitions.items():
        cols = _unique(cols)
        actual[name] = [c for c in cols if c in columns]
        missing[name] = [c for c in cols if c not in columns]
    return actual, missing


def _load_inputs(features_csv: str, labels_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)
    if "sample_id" not in features.columns or "sample_id" not in labels.columns:
        raise ValueError("both features and actionability labels must include sample_id")
    features = features.copy()
    labels = labels.copy()
    features["sample_id"] = features["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)
    return features, labels


def _merge_inputs(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "sample_id",
        "scenario_id",
        "original_label_id",
        "original_label_name",
        "actionability_label_id",
        "actionability_label_name",
    ]
    keep = [c for c in keep if c in labels.columns]
    merged = features.merge(labels[keep], on="sample_id", how="inner", suffixes=("", "_actionability"))
    if "scenario_id" not in merged.columns and "scenario_id_actionability" in merged.columns:
        merged["scenario_id"] = merged["scenario_id_actionability"]
    if "scenario_id" not in merged.columns:
        merged["scenario_id"] = merged["sample_id"].map(_stable_row_id)
    merged["actionability_label_id"] = pd.to_numeric(merged["actionability_label_id"], errors="coerce").astype(int)
    return merged


def _task_labels(df: pd.DataFrame, task: str) -> np.ndarray:
    label = df["actionability_label_id"].to_numpy(int)
    if task == "actionability_critical":
        return (label >= 2).astype(int)
    if task == "actionability_infeasible":
        return (label == 3).astype(int)
    raise ValueError(f"unknown task={task}")


def _make_model(seed: int, rf_n_estimators: int, rf_n_jobs: int, model_name: str) -> RandomForestClassifier:
    if model_name != "rf":
        raise SystemExit(f"unsupported --model={model_name}; this audit currently supports rf")
    return RandomForestClassifier(
        n_estimators=int(rf_n_estimators),
        n_jobs=int(rf_n_jobs),
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=int(seed),
    )


def _positive_score(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    if p.ndim == 2 and 1 in classes:
        return p[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(X.shape[0], dtype=float)


def _precision_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> float:
    if not np.isfinite(threshold):
        return np.nan
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    pred = s >= float(threshold)
    denom = int(np.sum(pred))
    if denom == 0:
        return np.nan
    return float(np.sum(pred & (y == 1)) / denom)


def _score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    metrics = binary_score_metrics(y_true, score, fpr_levels=[0.01, 0.05])
    metrics.pop("n_eval", None)
    for pct in [1, 5]:
        threshold = float(metrics.get(f"threshold_at_{pct}pct_fpr", np.nan))
        metrics[f"precision_at_{pct}pct_fpr"] = _precision_at_threshold(y_true, score, threshold)
    return metrics


def _importance_rows(task: str, group: str, cols: list[str], importances: np.ndarray, kind: str) -> list[dict]:
    order = np.argsort(np.asarray(importances, dtype=float))[::-1]
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append({
            "task": task,
            "feature_group": group,
            "feature": cols[int(idx)],
            "importance_kind": kind,
            "rank": int(rank),
            "importance": float(importances[int(idx)]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit actionability feature groups and non-circularity risks.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--actionability-labels-csv", required=True)
    ap.add_argument("--out-name", default="actionability_feature_audit")
    ap.add_argument("--tasks", default="actionability_critical,actionability_infeasible")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--rf-n-estimators", type=int, default=120)
    ap.add_argument("--rf-n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permutation-repeats", type=int, default=5)
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args()

    started = time.perf_counter()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out_dir = ensure_dir(work / "results" / "nc_actionability_classification" / args.out_name)
    test_fraction = float(cfg.get("analysis", {}).get("test_fraction", 0.25))
    tasks = _parse_csv_arg(args.tasks, ["actionability_critical", "actionability_infeasible"])

    features, labels = _load_inputs(args.features_csv, args.actionability_labels_csv)
    df = _merge_inputs(features, labels)
    if df.empty:
        raise ValueError("empty dataset after merging features and actionability labels")
    train_mask, test_mask = _scenario_hash_split(df, test_fraction=test_fraction, seed=int(args.seed))
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    groups, missing = _feature_groups(set(df.columns))

    definition_rows = []
    for name, cols in groups.items():
        definition_rows.append({
            "feature_group": name,
            "n_features": int(len(cols)),
            "features": ";".join(cols),
            "missing_fields": ";".join(missing[name]),
            "selected": bool(cols),
        })
    pd.DataFrame(definition_rows).to_csv(out_dir / "feature_group_definitions.csv", index=False)

    metric_rows = []
    impurity_rows = []
    permutation_rows = []
    prediction_rows = []
    total = len(tasks) * len(groups)
    done = 0
    for task in tasks:
        y_train = _task_labels(train_df, task)
        y_test = _task_labels(test_df, task)
        for group_name, cols in groups.items():
            done += 1
            print(f"[feature-audit] {done}/{total} task={task} group={group_name} n_features={len(cols)}")
            base_row = {
                "task": task,
                "feature_group": group_name,
                "n_features": int(len(cols)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "positive_rate_train": float(np.mean(y_train)) if len(y_train) else np.nan,
                "positive_rate_test": float(np.mean(y_test)) if len(y_test) else np.nan,
            }
            if not cols:
                metric_rows.append({**base_row, "error": "no available features"})
                continue
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                metric_rows.append({**base_row, "error": "single-class train/test"})
                continue
            X_train = numeric_frame(train_df, cols).to_numpy(float)
            X_test = numeric_frame(test_df, cols).to_numpy(float)
            model = _make_model(int(args.seed), int(args.rf_n_estimators), int(args.rf_n_jobs), args.model)
            model.fit(X_train, y_train)
            score = _positive_score(model, X_test)
            metric_rows.append({**base_row, **_score_metrics(y_test, score)})
            for local_i, (_, row) in enumerate(test_df.iterrows()):
                prediction_rows.append({
                    "sample_id": str(row["sample_id"]),
                    "scenario_id": str(row.get("scenario_id", row["sample_id"])),
                    "task": task,
                    "group": group_name,
                    "feature_group": group_name,
                    "feature_set": group_name,
                    "model": args.model,
                    "y_true": int(y_test[local_i]),
                    "score": float(score[local_i]),
                })
            impurity_rows.extend(_importance_rows(task, group_name, cols, model.feature_importances_, "rf_impurity"))
            if not args.skip_permutation:
                perm = permutation_importance(
                    model,
                    X_test,
                    y_test,
                    scoring="average_precision",
                    n_repeats=int(args.permutation_repeats),
                    random_state=int(args.seed),
                    n_jobs=1,
                )
                for row in _importance_rows(task, group_name, cols, perm.importances_mean, "permutation_auprc"):
                    idx = cols.index(row["feature"])
                    row["importance_std"] = float(perm.importances_std[idx])
                    row["n_repeats"] = int(args.permutation_repeats)
                    permutation_rows.append(row)

    pd.DataFrame(metric_rows).to_csv(out_dir / "feature_group_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / "feature_group_predictions.csv", index=False)
    pd.DataFrame(impurity_rows).to_csv(out_dir / "feature_importance_impurity.csv", index=False)
    if not args.skip_permutation:
        pd.DataFrame(permutation_rows).to_csv(out_dir / "feature_importance_permutation.csv", index=False)
    elapsed = time.perf_counter() - started
    print(f"[feature-audit] wrote {out_dir}")
    print(f"[feature-audit] elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
