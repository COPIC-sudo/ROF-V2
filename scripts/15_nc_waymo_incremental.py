from __future__ import annotations
import argparse
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
from rtbev.analysis_utils import load_merged
from rtbev.nc_eval import (
    available_feature_sets,
    binary_score_metrics,
    bootstrap_metric_ci,
    numeric_frame,
    scenario_hash_split,
    y_for_task,
)
from sklearn.metrics import average_precision_score, roc_auc_score


def _load_features(work: Path, features_csv: str | None) -> pd.DataFrame:
    if features_csv:
        return pd.read_csv(features_csv)
    return load_merged(work)


def _models(seed: int, fast: bool = False, model_names: str = "logreg,rf"):
    wanted = {x.strip().lower() for x in str(model_names).split(",") if x.strip()}
    out = {}
    if "logreg" in wanted:
        out["logreg"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ])
    if "rf" in wanted:
        out["rf"] = RandomForestClassifier(
            n_estimators=40 if fast else 160,
            random_state=seed,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=1,
        )
    return out


def _positive_score(model, X_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_test)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        if p.ndim == 2 and 1 in classes:
            return p[:, int(np.where(classes == 1)[0][0])]
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X_test)
        z = np.asarray(z, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))
    return np.zeros(X_test.shape[0], dtype=float)


def _subset_definitions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    def num(c, default=np.nan):
        if c not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    label = df["label_id"].astype(int).to_numpy()
    dist = num("current_min_distance_m", np.inf).fillna(np.inf).to_numpy(float)
    ttc = num("current_ttc_s", np.nan).to_numpy(float)
    speed = num("ego_speed_kph", 0.0).fillna(0.0).to_numpy(float)
    agent = num("agent_count", 0.0).fillna(0.0).to_numpy(float)
    no_ttc = (~np.isfinite(ttc)) | (ttc < 0)
    large_ttc = no_ttc | (ttc > 3.0)
    dense_thr = np.nanpercentile(agent, 75) if np.isfinite(agent).any() else np.inf
    return {
        "all": np.ones(n, dtype=bool),
        "no_ttc": no_ttc,
        "large_ttc_or_no_ttc": large_ttc,
        "low_speed_lt15kph": speed < 15.0,
        "dense_agents_p75": agent >= dense_thr,
        "close_distance_lt5m": dist < 5.0,
        "hard_safe_close": (label <= 1) & (dist < 5.0),
        "future_risk_no_ttc": (label >= 2) & no_ttc,
        "future_risk_large_ttc": (label >= 2) & large_ttc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="NC-minimal Waymo experiments: strong baseline + ROF-v2 increment and emergency-only evaluation.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--features-csv", default=None)
    ap.add_argument("--test-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bootstrap-n", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="debug mode: fewer RF trees")
    ap.add_argument("--models", default="logreg,rf", help="comma-separated model list: logreg,rf")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "nc_waymo_incremental")
    df = _load_features(work, args.features_csv)
    if len(df) < 20:
        raise SystemExit("Need at least ~20 feature rows for this experiment.")
    df = df.copy()
    df["label_id"] = df["label_id"].astype(int)

    acfg = cfg.get("analysis", {})
    ncfg = cfg.get("nc_experiments", {})
    test_fraction = float(args.test_fraction if args.test_fraction is not None else acfg.get("test_fraction", 0.25))
    seed = int(args.seed if args.seed is not None else acfg.get("random_seed", 42))
    boot_n = int(args.bootstrap_n if args.bootstrap_n is not None else ncfg.get("bootstrap_n", 500))
    fpr_levels = [float(x) for x in ncfg.get("fixed_fpr_levels", [0.01, 0.05])]

    train_mask, test_mask = scenario_hash_split(df, test_fraction=test_fraction, seed=seed)
    fsets = available_feature_sets(df)
    rows = []
    pred_rows = []
    feature_rows = []

    tasks = ["warning_or_above", "emergency_only", "safe_vs_risky"]
    for fs_name, cols in fsets.items():
        X_df = numeric_frame(df, cols)
        X = X_df.to_numpy(dtype=float)
        feature_rows.append({"feature_set": fs_name, "n_features": len(cols), "features": ";".join(cols)})
        for task in tasks:
            y = y_for_task(df["label_id"].to_numpy(), task)
            if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
                rows.append({"task": task, "feature_set": fs_name, "model": "all", "error": "single-class train/test split"})
                continue
            for model_name, model in _models(seed, fast=args.fast, model_names=args.models).items():
                try:
                    model.fit(X[train_mask], y[train_mask])
                    score = _positive_score(model, X[test_mask])
                    metrics = binary_score_metrics(y[test_mask], score, fpr_levels=fpr_levels)
                    rows.append({
                        "task": task,
                        "feature_set": fs_name,
                        "model": model_name,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        **metrics,
                    })
                    for idx, sid, sc, yt in zip(np.where(test_mask)[0], df.loc[test_mask, "sample_id"].astype(str), score, y[test_mask]):
                        pred_rows.append({
                            "row_index": int(idx),
                            "sample_id": sid,
                            "scenario_id": str(df.iloc[idx].get("scenario_id", sid)),
                            "task": task,
                            "feature_set": fs_name,
                            "model": model_name,
                            "y_true": int(yt),
                            "score": float(sc),
                            "label_id": int(df.iloc[idx]["label_id"]),
                        })
                except Exception as e:
                    rows.append({"task": task, "feature_set": fs_name, "model": model_name, "error": str(e)})

    metrics_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    pd.DataFrame(feature_rows).to_csv(out / "feature_sets.csv", index=False)
    metrics_df.to_csv(out / "nc_classification_metrics.csv", index=False)
    pred_df.to_csv(out / "nc_predictions.csv", index=False)

    # Diagnostic subset metrics.  These are crucial when all-sample labels are
    # dominated by distance/TTC and can hide actionability gains.
    subset_rows = []
    if not pred_df.empty:
        subsets = _subset_definitions(df)
        for (task, fs, model_name), sub in pred_df.groupby(["task", "feature_set", "model"]):
            idx = sub["row_index"].to_numpy(int)
            yv = sub["y_true"].to_numpy(int)
            sc = sub["score"].to_numpy(float)
            for subset_name, mask_all in subsets.items():
                m = mask_all[idx]
                if int(np.sum(m)) < 10 or len(np.unique(yv[m])) < 2:
                    continue
                met = binary_score_metrics(yv[m], sc[m], fpr_levels=fpr_levels)
                subset_rows.append({
                    "task": task, "feature_set": fs, "model": model_name,
                    "subset": subset_name, "n_subset": int(np.sum(m)), **met
                })
    pd.DataFrame(subset_rows).to_csv(out / "nc_classification_metrics_by_subset.csv", index=False)

    # Bootstrap CIs for the key comparison: strong_baseline_rof_v2 vs strong_baseline.
    delta_rows = []
    if not pred_df.empty:
        key_model = "rf" if (pred_df["model"] == "rf").any() else str(pred_df["model"].iloc[0])
        for task in tasks:
            base = pred_df[(pred_df.task == task) & (pred_df.feature_set == "strong_baseline") & (pred_df.model == key_model)]
            rof = pred_df[(pred_df.task == task) & (pred_df.feature_set == "strong_baseline_rof_v2") & (pred_df.model == key_model)]
            if base.empty or rof.empty:
                continue
            merged = base[["sample_id", "scenario_id", "y_true", "score"]].rename(columns={"score": "score_base"}).merge(
                rof[["sample_id", "score"]].rename(columns={"score": "score_rof"}), on="sample_id", how="inner"
            )
            if merged.empty or len(np.unique(merged["y_true"])) < 2:
                continue
            yv = merged["y_true"].to_numpy(dtype=int)
            groups = merged["scenario_id"].astype(str).to_numpy()
            for metric_name, fn in [
                ("auprc", average_precision_score),
                ("auroc", roc_auc_score),
            ]:
                b_point, b_lo, b_hi = bootstrap_metric_ci(yv, merged["score_base"].to_numpy(float), groups, fn, n_boot=boot_n, seed=seed)
                r_point, r_lo, r_hi = bootstrap_metric_ci(yv, merged["score_rof"].to_numpy(float), groups, fn, n_boot=boot_n, seed=seed + 1)
                # paired bootstrap for delta
                rng = np.random.default_rng(seed + 2)
                uniq = np.unique(groups)
                deltas = []
                for _ in range(boot_n):
                    gs = rng.choice(uniq, size=len(uniq), replace=True)
                    idxs = np.concatenate([np.where(groups == g)[0] for g in gs if np.any(groups == g)])
                    yy = yv[idxs]
                    if len(np.unique(yy)) < 2:
                        continue
                    try:
                        deltas.append(float(fn(yy, merged["score_rof"].to_numpy(float)[idxs]) - fn(yy, merged["score_base"].to_numpy(float)[idxs])))
                    except Exception:
                        pass
                delta_rows.append({
                    "task": task,
                    "model": key_model,
                    "metric": metric_name,
                    "baseline_point": b_point,
                    "baseline_ci_lo": b_lo,
                    "baseline_ci_hi": b_hi,
                    "rof_v2_point": r_point,
                    "rof_v2_ci_lo": r_lo,
                    "rof_v2_ci_hi": r_hi,
                    "delta_point": float(r_point - b_point),
                    "delta_ci_lo": float(np.percentile(deltas, 2.5)) if deltas else np.nan,
                    "delta_ci_hi": float(np.percentile(deltas, 97.5)) if deltas else np.nan,
                })
    pd.DataFrame(delta_rows).to_csv(out / "strong_baseline_vs_rof_v2_bootstrap_deltas.csv", index=False)

    # v1.1: paired bootstrap deltas by diagnostic subset.  This is often where
    # actionability helps even when the all-sample label is dominated by proximity.
    subset_delta_rows = []
    if not pred_df.empty:
        subsets = _subset_definitions(df)
        key_model = "rf" if (pred_df["model"] == "rf").any() else str(pred_df["model"].iloc[0])
        for task in tasks:
            base = pred_df[(pred_df.task == task) & (pred_df.feature_set == "strong_baseline") & (pred_df.model == key_model)]
            rof = pred_df[(pred_df.task == task) & (pred_df.feature_set == "strong_baseline_rof_v2") & (pred_df.model == key_model)]
            if base.empty or rof.empty:
                continue
            merged = base[["row_index", "sample_id", "scenario_id", "y_true", "score"]].rename(columns={"score": "score_base"}).merge(
                rof[["sample_id", "score"]].rename(columns={"score": "score_rof"}), on="sample_id", how="inner"
            )
            for subset_name, mask_all in subsets.items():
                keep = mask_all[merged["row_index"].to_numpy(int)]
                subm = merged.loc[keep]
                if len(subm) < 20 or len(np.unique(subm["y_true"])) < 2:
                    continue
                yv = subm["y_true"].to_numpy(int)
                groups = subm["scenario_id"].astype(str).to_numpy()
                for metric_name, fn in [("auprc", average_precision_score), ("auroc", roc_auc_score)]:
                    try:
                        b = float(fn(yv, subm["score_base"].to_numpy(float)))
                        r = float(fn(yv, subm["score_rof"].to_numpy(float)))
                    except Exception:
                        continue
                    rng = np.random.default_rng(seed + 17)
                    uniq = np.unique(groups)
                    deltas = []
                    for _ in range(boot_n):
                        gs = rng.choice(uniq, size=len(uniq), replace=True)
                        idxs = np.concatenate([np.where(groups == g)[0] for g in gs if np.any(groups == g)])
                        yy = yv[idxs]
                        if len(np.unique(yy)) < 2:
                            continue
                        try:
                            deltas.append(float(fn(yy, subm["score_rof"].to_numpy(float)[idxs]) - fn(yy, subm["score_base"].to_numpy(float)[idxs])))
                        except Exception:
                            pass
                    subset_delta_rows.append({
                        "task": task, "model": key_model, "subset": subset_name, "metric": metric_name,
                        "baseline_point": b, "rof_v2_point": r, "delta_point": r - b,
                        "delta_ci_lo": float(np.percentile(deltas, 2.5)) if deltas else np.nan,
                        "delta_ci_hi": float(np.percentile(deltas, 97.5)) if deltas else np.nan,
                        "n": int(len(subm)),
                    })
    pd.DataFrame(subset_delta_rows).to_csv(out / "strong_baseline_vs_rof_v2_bootstrap_deltas_by_subset.csv", index=False)
    print(f"[nc-incremental-v1.1] wrote {out}")


if __name__ == "__main__":
    main()
