#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score


RULE_NAMES = ["original", "moderate", "strict", "emergency_sensitive", "comfort_sensitive", "quantile_like"]
LABEL_NAMES = {
    0: "high_actionability",
    1: "reduced_actionability",
    2: "critical_actionability",
    3: "candidate_set_infeasible",
}
BASELINE = "strong_baseline_cv"
ENHANCED = "strong_baseline_cv_plus_strict_temporal_dynamics"


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


def rule_label(rule: str, comfort: pd.Series, emergency: pd.Series) -> np.ndarray:
    c = pd.to_numeric(comfort, errors="coerce").fillna(0).to_numpy(float)
    e = pd.to_numeric(emergency, errors="coerce").fillna(0).to_numpy(float)
    out = np.zeros(len(c), dtype=int)
    if rule == "original":
        out[e == 0] = 3
        out[(c == 0) & (e > 0)] = 2
        out[(c < 0.25) & ~((e == 0) | ((c == 0) & (e > 0)))] = 1
        return out
    if rule == "moderate":
        out[e <= 0.0] = 3
        out[((c <= 0.25) | (e <= 0.35)) & (out != 3)] = 2
        out[((c < 0.75) | (e < 0.65)) & (out == 0)] = 1
        return out
    if rule == "strict":
        out[e <= 0.0] = 3
        out[((c <= 0.50) | (e <= 0.50)) & (out != 3)] = 2
        out[((c < 1.00) | (e < 0.85)) & (out == 0)] = 1
        return out
    if rule == "emergency_sensitive":
        out[e <= 0.0] = 3
        out[(e <= 0.25) & (out != 3)] = 2
        out[((e <= 0.60) | (c <= 0.50)) & (out == 0)] = 1
        return out
    if rule == "comfort_sensitive":
        out[e <= 0.0] = 3
        out[(c <= 0.25) & (out != 3)] = 2
        out[(c <= 0.75) & (out == 0)] = 1
        return out
    if rule == "quantile_like":
        out[e <= 0.0] = 3
        out[((c <= 0.50) | (e <= 0.4286)) & (out != 3)] = 2
        out[((c <= 0.75) | (e <= 0.7143)) & (out == 0)] = 1
        return out
    raise ValueError(rule)


def severe_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a) >= 2
    bb = np.asarray(b) >= 2
    union = aa | bb
    return float(np.sum(aa & bb) / max(np.sum(union), 1))


def load_oof_module(repo: Path):
    path = repo / "scripts/nc_v090/02_waymo_confirmatory_oof.py"
    spec = importlib.util.spec_from_file_location("nc_v090_oof", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def model_metric(y: np.ndarray, score: np.ndarray, alert: np.ndarray) -> dict[str, float]:
    out = {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "prevalence": float(np.mean(y)),
        "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "recall_at_calibrated_5pct_fpr": float(np.mean(alert[y == 1])) if np.any(y == 1) else np.nan,
        "achieved_fpr": float(np.mean(alert[y == 0])) if np.any(y == 0) else np.nan,
        "precision": float(np.sum(alert & (y == 1)) / max(np.sum(alert), 1)) if np.any(alert) else np.nan,
    }
    return out


def threshold_rule_oof(repo: Path, cfg: dict[str, Any], label_df: pd.DataFrame, rule_labels: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    mod = load_oof_module(repo)
    features = pd.read_csv(cfg["inputs"]["waymo_features_csv"])
    features["sample_id"] = features["sample_id"].astype(str)
    labels = label_df[["sample_id", "scenario_id"]].copy()
    labels["sample_id"] = labels["sample_id"].astype(str)
    df = features.merge(labels, on="sample_id", how="inner", suffixes=("", "_label"))
    if "scenario_id" not in df.columns and "scenario_id_label" in df.columns:
        df["scenario_id"] = df["scenario_id_label"]
    df["outer_fold"] = mod.assign_outer_folds(df, int(cfg["splits"]["outer_folds"]), int(cfg["splits"]["scenario_hash_seed"]))
    rf_cfg = cfg["models"]["random_forest"]
    rows = []
    for rule, lab in rule_labels.items():
        df["y"] = (np.asarray(lab, dtype=int) >= 2).astype(int)
        for fs_name, cols in {
            BASELINE: [c for c in mod.FEATURE_SETS[BASELINE] if c in df.columns],
            ENHANCED: [c for c in mod.FEATURE_SETS[ENHANCED] if c in df.columns],
        }.items():
            preds = []
            for fold in sorted(df["outer_fold"].unique()):
                test_mask = df["outer_fold"].to_numpy(int) == int(fold)
                train_cal = df.loc[~test_mask].copy()
                test = df.loc[test_mask].copy()
                cal_mask = mod.fit_calibration_mask(train_cal, 42, int(fold), float(cfg["splits"]["calibration_fraction_within_outer_train"]))
                fit = train_cal.loc[~cal_mask].copy()
                cal = train_cal.loc[cal_mask].copy()
                if len(np.unique(fit["y"])) < 2 or len(np.unique(test["y"])) < 2:
                    continue
                pre = mod.TrainOnlyPreprocessor(cols, scale=False).fit(fit)
                model = RandomForestClassifier(
                    n_estimators=int(rf_cfg["n_estimators"]),
                    criterion=str(rf_cfg.get("criterion", "gini")),
                    max_depth=rf_cfg.get("max_depth"),
                    min_samples_split=int(rf_cfg.get("min_samples_split", 2)),
                    min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
                    max_features=rf_cfg.get("max_features", "sqrt"),
                    bootstrap=bool(rf_cfg.get("bootstrap", True)),
                    class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
                    random_state=42,
                    n_jobs=int(rf_cfg.get("n_jobs", -1)),
                )
                model.fit(pre.transform(fit), fit["y"].to_numpy(int))
                cal_score = mod.positive_score(model, pre.transform(cal))
                threshold, cal_fpr = mod.threshold_from_calibration(cal["y"].to_numpy(int), cal_score, float(cfg["evaluation"]["fixed_fpr_nominal"]))
                score = mod.positive_score(model, pre.transform(test))
                preds.append(pd.DataFrame({
                    "sample_id": test["sample_id"].to_numpy(str),
                    "scenario_id": test["scenario_id"].to_numpy(str),
                    "y": test["y"].to_numpy(int),
                    "score": score,
                    "alert": score >= threshold,
                    "fold": int(fold),
                    "threshold": threshold,
                    "calibration_fpr": cal_fpr,
                }))
            if not preds:
                continue
            pred = pd.concat(preds, ignore_index=True)
            rows.append({"variant": f"rule_{rule}", "rule": rule, "feature_set": fs_name, "seed": 42, **model_metric(pred["y"].to_numpy(int), pred["score"].to_numpy(float), pred["alert"].to_numpy(bool))})
    # Add paired deltas per rule.
    by_rule = pd.DataFrame(rows)
    delta_rows = []
    for rule in by_rule["rule"].dropna().unique():
        base = by_rule[(by_rule["rule"] == rule) & (by_rule["feature_set"] == BASELINE)]
        enh = by_rule[(by_rule["rule"] == rule) & (by_rule["feature_set"] == ENHANCED)]
        if not base.empty and not enh.empty:
            for metric in ["auprc", "auroc", "recall_at_calibrated_5pct_fpr"]:
                delta_rows.append({
                    "variant": f"rule_{rule}",
                    "rule": rule,
                    "feature_set": f"{BASELINE}_vs_{ENHANCED}",
                    "metric": metric,
                    "delta": float(enh.iloc[0][metric] - base.iloc[0][metric]),
                    "directionally_consistent": bool(enh.iloc[0][metric] - base.iloc[0][metric] > 0),
                })
    return rows + delta_rows


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    groups = groups[ok]
    rx = rankdata(x)
    ry = rankdata(y)
    point = float(np.corrcoef(rx, ry)[0, 1]) if len(rx) >= 3 else np.nan
    uniq = np.unique(groups)
    idxs = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        sampled = rng.integers(0, len(idxs), size=len(idxs))
        idx = np.concatenate([idxs[i] for i in sampled])
        if len(np.unique(rx[idx])) < 2 or len(np.unique(ry[idx])) < 2:
            continue
        vals.append(float(np.corrcoef(rx[idx], ry[idx])[0, 1]))
    arr = np.asarray(vals, dtype=float)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), int(len(arr))


def ttc_audit(cfg: dict[str, Any], out_dir: Path) -> None:
    features = pd.read_csv(cfg["inputs"]["waymo_features_csv"], usecols=["sample_id", "scenario_id", "current_min_distance_m", "current_ttc_s", "ego_speed_kph", "agent_count"])
    labels = pd.read_csv(cfg["inputs"]["waymo_actionability_map_labels_csv"], usecols=["sample_id", "actionability_label_id"])
    nomap = pd.read_csv(cfg["inputs"]["waymo_actionability_nomap_labels_csv"], usecols=["sample_id", "actionability_label_id"]).rename(columns={"actionability_label_id": "nomap_actionability_label_id"})
    df = features.merge(labels, on="sample_id").merge(nomap, on="sample_id")
    ttc = pd.to_numeric(df["current_ttc_s"], errors="coerce")
    rows = [
        {"category": "valid_ttc", "count": int(((ttc >= 0) & np.isfinite(ttc)).sum()), "rate": float(((ttc >= 0) & np.isfinite(ttc)).mean())},
        {"category": "invalid_or_no_closing_ttc", "count": int((ttc < 0).sum()), "rate": float((ttc < 0).mean())},
        {"category": "non_finite_ttc", "count": int((~np.isfinite(ttc)).sum()), "rate": float((~np.isfinite(ttc)).mean())},
    ]
    write_csv(out_dir / "ttc_missingness_summary.csv", rows)
    sens = []
    groups = df["scenario_id"].astype(str).to_numpy()
    label_map = pd.to_numeric(df["actionability_label_id"], errors="coerce").to_numpy(float)
    label_nomap = pd.to_numeric(df["nomap_actionability_label_id"], errors="coerce").to_numpy(float)
    dist = pd.to_numeric(df["current_min_distance_m"], errors="coerce").to_numpy(float)
    ttc_arr = ttc.to_numpy(float)
    reps = {
        "valid_ttc_only": np.where((ttc_arr >= 0) & np.isfinite(ttc_arr), ttc_arr, np.nan),
        "no_ttc_as_category": np.where((ttc_arr >= 0) & np.isfinite(ttc_arr), ttc_arr, 999.0),
        "capped_ttc_prespecified_10s": np.clip(np.where((ttc_arr >= 0) & np.isfinite(ttc_arr), ttc_arr, 10.0), 0, 10.0),
        "inverse_ttc_prespecified": 1.0 / np.maximum(np.where((ttc_arr >= 0) & np.isfinite(ttc_arr), ttc_arr, 10.0), 0.1),
        "legacy_sentinel": ttc_arr,
    }
    for label_name, label in [("map_actionability_label_id", label_map), ("nomap_actionability_label_id", label_nomap)]:
        for var_name, values in [("current_min_distance_m", dist), *reps.items()]:
            point, lo, hi, n_valid = bootstrap_spearman(label, values, groups, n_boot=2000, seed=42)
            sens.append({"label": label_name, "variable": var_name, "spearman": point, "ci_low": lo, "ci_high": hi, "n_bootstrap_valid": n_valid})
    write_csv(out_dir / "ttc_sensitivity.csv", sens)
    # Distribution within simple strata.
    df["distance_stratum"] = pd.cut(pd.to_numeric(df["current_min_distance_m"], errors="coerce"), [-np.inf, 2, 5, 10, np.inf], labels=["lt2m", "2to5m", "5to10m", "ge10m"])
    df["ttc_stratum"] = pd.cut(reps["capped_ttc_prespecified_10s"], [-np.inf, 1, 3, 10, np.inf], labels=["lt1s", "1to3s", "3to10s", "no_or_ge10s"])
    strata_rows = []
    for col in ["distance_stratum", "ttc_stratum"]:
        for key, sub in df.groupby(col, observed=False):
            for label_col in ["actionability_label_id", "nomap_actionability_label_id"]:
                counts = sub[label_col].value_counts().reindex([0, 1, 2, 3], fill_value=0)
                row = {"stratum_type": col, "stratum": str(key), "label": label_col, "n": int(len(sub))}
                for i, count in counts.items():
                    row[f"label_{i}_count"] = int(count)
                    row[f"label_{i}_fraction"] = float(count / max(len(sub), 1))
                strata_rows.append(row)
    write_csv(out_dir / "ttc_actionability_strata.csv", strata_rows)
    md = [
        "# TTC Audit",
        "",
        "TTC representations were audited without retraining the main OOF models.",
        "The v0.9.0 confirmatory OOF script uses train-only median imputation after setting negative/non-finite TTC values to missing.",
        "The legacy sentinel representation is retained only as a labelled diagnostic.",
    ]
    (out_dir / "TTC_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v090/nc_v090_audit.yaml")
    args = parser.parse_args()
    repo = Path.cwd()
    cfg = load_yaml(repo / args.config)
    out_dir = repo / cfg["project"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    label_df = pd.read_csv(cfg["inputs"]["waymo_actionability_map_labels_csv"])
    ref = label_df["actionability_label_id"].to_numpy(int)
    rule_labels = {rule: rule_label(rule, label_df["comfort_feasible_ratio"], label_df["emergency_feasible_ratio"]) for rule in RULE_NAMES}
    summary = []
    transition_dir = out_dir / "label_transition_matrices"
    transition_dir.mkdir(parents=True, exist_ok=True)
    for rule, lab in rule_labels.items():
        counts = pd.Series(lab).value_counts().reindex([0, 1, 2, 3], fill_value=0)
        for label_id, count in counts.items():
            summary.append({"variant": f"rule_{rule}", "rule": rule, "label_id": int(label_id), "label_name": LABEL_NAMES[int(label_id)], "count": int(count), "fraction": float(count / len(lab))})
        summary.append({"variant": f"rule_{rule}", "rule": rule, "metric": "critical_or_worse_prevalence", "value": float(np.mean(lab >= 2))})
        summary.append({"variant": f"rule_{rule}", "rule": rule, "metric": "candidate_set_infeasible_prevalence", "value": float(np.mean(lab == 3))})
        summary.append({"variant": f"rule_{rule}", "rule": rule, "metric": "weighted_cohen_kappa_vs_moderate", "value": float(cohen_kappa_score(ref, lab, weights="quadratic"))})
        summary.append({"variant": f"rule_{rule}", "rule": rule, "metric": "severe_class_jaccard_vs_moderate", "value": severe_jaccard(ref, lab)})
        summary.append({"variant": f"rule_{rule}", "rule": rule, "metric": "fraction_changed_vs_moderate", "value": float(np.mean(ref != lab))})
        ref_severe = ref >= 2
        summary.append({"variant": f"rule_{rule}", "rule": rule, "metric": "reference_severe_changed_fraction", "value": float(np.mean(ref[ref_severe] != lab[ref_severe])) if np.any(ref_severe) else np.nan})
        pd.crosstab(pd.Series(ref, name="reference_moderate"), pd.Series(lab, name=f"rule_{rule}")).to_csv(transition_dir / f"transition_reference_moderate_to_rule_{rule}.csv")
    write_csv(out_dir / "label_robustness_summary.csv", summary)
    manifest = [{"variant": f"rule_{rule}", "family": "threshold_rule", "status": "SUPPORTED_FROM_EXISTING_FULL_RATIO_FIELDS", "notes": "No rollout rerun; labels recomputed from existing comfort/emergency feasible ratios."} for rule in RULE_NAMES]
    for family, variants, reason in [
        ("rollout_horizon_s", "2.0,3.0,4.0", "Requires full actionability label rollout regeneration."),
        ("map_lane_centerline_buffer_m", "2.0,3.0,4.0", "Current label script does not expose lane buffer as a CLI parameter independent of config."),
        ("action_library", "base7,extended", "Current label script action library is hard-coded; extending it would change scientific definition."),
        ("invalid_future_handling", "skip_invalid,cv_fallback", "Current label script skips invalid oracle-future obstacle states; CV fallback variant is not implemented."),
    ]:
        manifest.append({"variant": variants, "family": family, "status": "BLOCKED_NOT_RUN", "notes": reason})
    write_csv(out_dir / "label_variant_manifest.csv", manifest)
    model_rows = threshold_rule_oof(repo, cfg, label_df, rule_labels)
    write_csv(out_dir / "waymo_robustness_model_metrics.csv", model_rows)
    write_csv(out_dir / "future_validity_audit.csv", [{"status": "BLOCKED_NOT_RUN", "reason": "Full future_validity audit requires reading all sample pkl.gz files or rerunning label generator; not needed for threshold-rule robustness from existing ratios."}])
    ttc_audit(cfg, out_dir)
    print(f"[robustness-ttc] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
