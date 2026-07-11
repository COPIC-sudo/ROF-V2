from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.nc_eval import binary_score_metrics, numeric_frame
from rtbev.progress import ProgressReporter


ACTIONABILITY_LABEL_NAMES = {
    0: "high_actionability",
    1: "reduced_actionability",
    2: "critical_actionability",
    3: "infeasible_or_unavoidable",
}

TASKS = {
    "actionability_degraded": lambda s: (s >= 1).astype(int),
    "actionability_critical": lambda s: (s >= 2).astype(int),
    "actionability_infeasible": lambda s: (s == 3).astype(int),
}

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
ROF_SPACE_COLS = [
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
ACTIONABILITY_COLS = [
    "redi_actionability",
    "redi_actionability_delta",
    "asr_slice_final",
    "asr_slice_min",
    "asr_cum_final",
    "asr_cum_min",
    "comfort_asr",
    "emergency_asr",
    "comfort_to_emergency_gap",
    "ttad_s",
    "time_to_first_conflict_s",
    "early_blocking_ratio",
    "collapse_rate_max_per_s",
    "min_safe_action_cost",
    "slice_survival_keep",
    "slice_survival_brake",
    "slice_survival_left",
    "slice_survival_right",
    "slice_survival_brake_left",
    "slice_survival_brake_right",
]
DEFAULT_FEATURE_SET_ORDER = [
    "distance_ttc",
    "strong_baseline",
    "rof_v2_only",
    "actionability_only",
    "strong_baseline_rof_v2",
    "strong_baseline_actionability",
    "strong_baseline_cv",
    "strong_baseline_cv_actionability",
]


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


def _feature_sets(columns: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    requested = {
        "distance_ttc": ["current_min_distance_m", "current_ttc_s"],
        "strong_baseline": STRONG_BASELINE_COLS,
        "rof_v2_only": ROF_SPACE_COLS + ACTIONABILITY_COLS,
        "actionability_only": ACTIONABILITY_COLS,
        "strong_baseline_rof_v2": STRONG_BASELINE_COLS + ROF_SPACE_COLS + ACTIONABILITY_COLS,
        "strong_baseline_actionability": STRONG_BASELINE_COLS + ACTIONABILITY_COLS,
        "strong_baseline_cv": STRONG_BASELINE_COLS + CV_COLS,
        "strong_baseline_cv_actionability": STRONG_BASELINE_COLS + CV_COLS + ACTIONABILITY_COLS,
    }
    actual: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for name, cols in requested.items():
        cols = _unique(cols)
        actual[name] = [c for c in cols if c in columns]
        missing[name] = [c for c in cols if c not in columns]
    return actual, missing


def _models(seed: int, model_names: list[str], rf_n_estimators: int, rf_n_jobs: int):
    all_models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=int(rf_n_estimators),
            n_jobs=int(rf_n_jobs),
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=seed,
        ),
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


def _score_metrics(y_true: np.ndarray, score: np.ndarray, fpr_levels: list[float]) -> dict:
    metrics = binary_score_metrics(y_true, score, fpr_levels=fpr_levels)
    positive_rate = metrics.pop("positive_rate", np.nan)
    metrics.pop("n_eval", None)
    out = {"positive_rate": positive_rate, **metrics}
    for fpr in fpr_levels:
        pct = int(round(float(fpr) * 100))
        thr = out.get(f"threshold_at_{pct}pct_fpr", np.nan)
        out[f"precision_at_{pct}pct_fpr"] = _precision_at_threshold(y_true, score, float(thr))
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
    denom = float(16 ** 16)
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
    keep_label_cols = [
        "sample_id",
        "scenario_id",
        "original_label_id",
        "original_label_name",
        "actionability_label_id",
        "actionability_label_name",
        "actionability_binary_degraded",
        "actionability_binary_critical",
        "actionability_binary_infeasible",
    ]
    keep_label_cols = [c for c in keep_label_cols if c in labels.columns]
    merged = features.merge(labels[keep_label_cols], on="sample_id", how="inner", suffixes=("", "_actionability"))
    if "scenario_id" not in merged.columns and "scenario_id_actionability" in merged.columns:
        merged["scenario_id"] = merged["scenario_id_actionability"]
    if "scenario_id" not in merged.columns:
        merged["scenario_id"] = merged["sample_id"].map(_stable_row_id)
    if "original_label_id" not in merged.columns:
        if "label_id" in merged.columns:
            merged["original_label_id"] = pd.to_numeric(merged["label_id"], errors="coerce").fillna(0).astype(int)
        else:
            merged["original_label_id"] = 0
    merged["actionability_label_id"] = pd.to_numeric(merged["actionability_label_id"], errors="coerce").astype(int)
    merged["original_label_id"] = pd.to_numeric(merged["original_label_id"], errors="coerce").fillna(0).astype(int)
    return merged


def _subset_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    def num(name: str, default=np.nan) -> pd.Series:
        if name not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)

    original = num("original_label_id", 0).fillna(0).astype(int)
    action = num("actionability_label_id", 0).fillna(0).astype(int)
    speed = num("ego_speed_kph")
    agents = num("agent_count")
    dist = num("current_min_distance_m")
    ttc = num("current_ttc_s")
    masks: dict[str, np.ndarray] = {
        "all": np.ones(n, dtype=bool),
        "original_safe": (original == 0).to_numpy(bool),
        "original_warning": (original == 2).to_numpy(bool),
        "original_emergency": (original == 3).to_numpy(bool),
        "low_speed_lt15kph": (speed < 15.0).fillna(False).to_numpy(bool),
        "close_distance_lt5m": (dist < 5.0).fillna(False).to_numpy(bool),
        "large_or_no_ttc": ((ttc < 0) | (~np.isfinite(ttc)) | (ttc > 3.0)).to_numpy(bool),
        "high_actionability": (action == 0).to_numpy(bool),
        "reduced_actionability": (action == 1).to_numpy(bool),
        "critical_actionability": (action == 2).to_numpy(bool),
        "infeasible_or_unavoidable": (action == 3).to_numpy(bool),
    }
    if agents.notna().any():
        p75 = float(agents.quantile(0.75))
        masks["dense_agents_p75"] = (agents >= p75).fillna(False).to_numpy(bool)
    else:
        masks["dense_agents_p75"] = np.zeros(n, dtype=bool)
    ordered = [
        "all",
        "original_safe",
        "original_warning",
        "original_emergency",
        "low_speed_lt15kph",
        "dense_agents_p75",
        "close_distance_lt5m",
        "large_or_no_ttc",
        "high_actionability",
        "reduced_actionability",
        "critical_actionability",
        "infeasible_or_unavoidable",
    ]
    return {k: masks[k] for k in ordered}


def _label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    counts = df["actionability_label_id"].value_counts().reindex([0, 1, 2, 3], fill_value=0).sort_index()
    for label_id, count in counts.items():
        rows.append({
            "section": "actionability_label_distribution",
            "actionability_label_id": int(label_id),
            "actionability_label_name": ACTIONABILITY_LABEL_NAMES[int(label_id)],
            "count": int(count),
            "fraction": float(count / n),
        })
    for name, mask in [
        ("actionability_degraded", df["actionability_label_id"].to_numpy(int) >= 1),
        ("actionability_critical", df["actionability_label_id"].to_numpy(int) >= 2),
        ("actionability_infeasible", df["actionability_label_id"].to_numpy(int) == 3),
    ]:
        rows.append({
            "section": "binary_positive_rate",
            "task": name,
            "count": int(np.sum(mask)),
            "fraction": float(np.mean(mask)),
        })
    ct = pd.crosstab(df["original_label_id"], df["actionability_label_id"])
    for original_id in sorted(df["original_label_id"].dropna().astype(int).unique()):
        for action_id in [0, 1, 2, 3]:
            value = int(ct.loc[original_id, action_id]) if original_id in ct.index and action_id in ct.columns else 0
            rows.append({
                "section": "original_vs_actionability_crosstab",
                "original_label_id": int(original_id),
                "actionability_label_id": int(action_id),
                "count": value,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train first-pass classifiers for moderate actionability labels.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--actionability-labels-csv", required=True)
    ap.add_argument("--out-name", default="actionability_full")
    ap.add_argument("--models", default="logreg,rf")
    ap.add_argument("--feature-sets", default=None, help="comma-separated feature set names")
    ap.add_argument("--rf-n-estimators", type=int, default=120)
    ap.add_argument("--rf-n-jobs", type=int, default=-1)
    ap.add_argument("--test-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-bootstrap", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--progress-task", default="actionability_classification")
    args = ap.parse_args()

    progress: ProgressReporter | None = None
    try:
        cfg = load_config(args.config)
        work = Path(cfg["project"]["work_dir"])
        out_dir = ensure_dir(work / "results" / "nc_actionability_classification" / args.out_name)
        fpr_levels = [0.01, 0.05]
        test_fraction = float(args.test_fraction if args.test_fraction is not None else cfg.get("analysis", {}).get("test_fraction", 0.25))
        model_names = _parse_csv_arg(args.models, ["logreg", "rf"])
        selected_models = _models(int(args.seed), model_names, int(args.rf_n_estimators), int(args.rf_n_jobs))
        progress = ProgressReporter(work, args.progress_task, enabled=True)

        progress.update("loading", message="reading features and actionability labels")
        features, labels = _load_inputs(args.features_csv, args.actionability_labels_csv)

        progress.update("merging", message="merging by sample_id")
        df = _merge_inputs(features, labels)
        if df.empty:
            raise ValueError("empty dataset after merging features and actionability labels")

        progress.update("splitting", message=f"scenario-level split, test_fraction={test_fraction:.3f}")
        train_mask, test_mask = _scenario_hash_split(df, test_fraction=test_fraction, seed=int(args.seed))
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(f"empty train/test split: train={len(train_df)}, test={len(test_df)}")

        feature_sets_all, missing_fields = _feature_sets(set(df.columns))
        requested_fsets = _parse_csv_arg(args.feature_sets, DEFAULT_FEATURE_SET_ORDER)
        unknown = [name for name in requested_fsets if name not in feature_sets_all]
        if unknown:
            raise SystemExit(f"unknown --feature-sets entries: {unknown}; valid={sorted(feature_sets_all)}")
        feature_sets = {name: feature_sets_all[name] for name in requested_fsets if feature_sets_all[name]}
        if not feature_sets:
            raise ValueError("no usable feature sets after filtering missing columns")

        feature_rows = []
        for name in requested_fsets:
            feature_rows.append({
                "feature_set": name,
                "n_features": int(len(feature_sets_all.get(name, []))),
                "features": ";".join(feature_sets_all.get(name, [])),
                "missing_fields": ";".join(missing_fields.get(name, [])),
                "selected": bool(name in feature_sets),
            })

        total_jobs = len(TASKS) * len(selected_models) * len(feature_sets)
        progress.update("training", step=0, total=total_jobs, message=f"starting {total_jobs} model jobs")
        metrics_rows = []
        subset_rows = []
        pred_rows = []
        job_i = 0
        for task_name, y_fn in TASKS.items():
            y_train = y_fn(train_df["actionability_label_id"].to_numpy(int))
            y_test = y_fn(test_df["actionability_label_id"].to_numpy(int))
            for feature_set_name, cols in feature_sets.items():
                X_train = numeric_frame(train_df, cols).to_numpy(float)
                X_test = numeric_frame(test_df, cols).to_numpy(float)
                for model_name, model in selected_models.items():
                    job_i += 1
                    combo = f"{task_name}/{model_name}/{feature_set_name}"
                    progress.update("training", step=job_i, total=total_jobs, message=combo)
                    row_base = {
                        "task": task_name,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "positive_rate_train": float(np.mean(y_train)),
                        "positive_rate_test": float(np.mean(y_test)),
                    }
                    if len(np.unique(y_train)) < 2:
                        metrics_rows.append({**row_base, "error": "single-class train"})
                        continue
                    model.fit(X_train, y_train)
                    progress.update("evaluating", step=job_i, total=total_jobs, message=combo)
                    score = _positive_score(model, X_test)
                    metric = _score_metrics(y_test, score, fpr_levels)
                    pos_test = metric.pop("positive_rate", np.nan)
                    metrics_rows.append({**row_base, "positive_rate_test": pos_test, **metric})
                    for local_i, (_, row) in enumerate(test_df.iterrows()):
                        pred_rows.append({
                            "row_index": int(local_i),
                            "sample_id": str(row["sample_id"]),
                            "scenario_id": str(row.get("scenario_id", row["sample_id"])),
                            "task": task_name,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "y_true": int(y_test[local_i]),
                            "score": float(score[local_i]),
                            "original_label_id": int(row["original_label_id"]),
                            "actionability_label_id": int(row["actionability_label_id"]),
                        })
                    masks = _subset_masks(test_df)
                    for subset_name, mask in masks.items():
                        sub_y = y_test[mask]
                        sub_score = score[mask]
                        sub_metrics = _score_metrics(sub_y, sub_score, fpr_levels) if len(sub_y) else {}
                        subset_rows.append({
                            "task": task_name,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "subset": subset_name,
                            "n_subset": int(np.sum(mask)),
                            **sub_metrics,
                        })

        if not args.skip_bootstrap:
            progress.event("--no-skip-bootstrap was requested, but bootstrap is not implemented in this first-pass script.", level="warning")

        progress.update("writing_outputs", message=str(out_dir))
        pd.DataFrame(metrics_rows).to_csv(out_dir / "actionability_classification_metrics.csv", index=False)
        pd.DataFrame(subset_rows).to_csv(out_dir / "actionability_classification_metrics_by_subset.csv", index=False)
        pd.DataFrame(pred_rows).to_csv(out_dir / "actionability_predictions.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(out_dir / "feature_sets.csv", index=False)
        _label_distribution(df).to_csv(out_dir / "label_distribution.csv", index=False)
        progress.complete(f"wrote {out_dir}")
        print(f"[actionability-classification] wrote {out_dir}")
    except Exception as exc:
        if progress is not None:
            progress.fail(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
