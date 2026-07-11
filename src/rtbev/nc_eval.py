from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if "ttc" in c.lower():
            finite_nonneg = X[c][(X[c] >= 0) & np.isfinite(X[c])]
            large = max(float(finite_nonneg.quantile(0.95)) if len(finite_nonneg) else 10.0, 10.0)
            X.loc[(X[c] < 0) | (~np.isfinite(X[c])), c] = large
        med = X[c].median()
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(float(med))
    return X


def available_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    base = [
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
    cv = ["cv_rcr", "cv_rfr_drv", "cv_c_time", "cv_gtoa_norm_union", "cv_oce_norm", "cv_c_density", "cv_max_overlap_count"]
    old_rof = ["rcr", "rfr_drv", "c_time", "gtoa_norm_union", "oce_norm", "c_density", "msr", "c_maneuver", "redi_full", "redi_no_msr"]
    act = [
        # backward-compatible aliases
        "asr", "asr_final", "asr_min", "asr_initial", "c_actionability",
        # v1.1 explicit actionability metrics
        "asr_cum_initial", "asr_cum_final", "asr_cum_min",
        "asr_slice_initial", "asr_slice_final", "asr_slice_min",
        "comfort_asr", "emergency_asr", "comfort_to_emergency_gap",
        "ttad_s", "time_to_first_conflict_s", "collapse_rate_per_s",
        "collapse_rate_mean_per_s", "collapse_rate_max_per_s",
        "early_blocking_ratio", "min_safe_action_cost",
        "survival_keep", "survival_accelerate", "survival_brake", "survival_hard_brake",
        "survival_left", "survival_right", "survival_brake_left", "survival_brake_right",
        "slice_survival_keep", "slice_survival_brake", "slice_survival_hard_brake",
        "slice_survival_left", "slice_survival_right", "slice_survival_brake_left", "slice_survival_brake_right",
        "redi_actionability", "redi_actionability_delta",
    ]
    sets = {
        "strong_baseline": base,
        "strong_baseline_cv": base + cv,
        "old_rof_only": old_rof,
        "rof_v2_only": old_rof + act,
        "actionability_only": act,
        "strong_baseline_old_rof": base + old_rof,
        "strong_baseline_rof_v2": base + old_rof + act,
        "distance_ttc": ["current_min_distance_m", "current_ttc_s"],
        "redi_actionability_scalar": ["redi_actionability"],
        "redi_full_scalar": ["redi_full"],
        "asr_cum_scalar": ["asr_cum_final"],
        "asr_slice_scalar": ["asr_slice_min"],
        "ttad_scalar": ["ttad_s"],
        "collapse_scalar": ["collapse_rate_max_per_s"],
    }
    out: dict[str, list[str]] = {}
    for name, cols in sets.items():
        cols2 = [c for c in cols if c in df.columns]
        if cols2 and df[cols2].apply(pd.to_numeric, errors="coerce").notna().any().any():
            out[name] = cols2
    return out


def scenario_hash_split(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42):
    key_col = "scenario_id" if "scenario_id" in df.columns else "sample_id"
    keys = df[key_col].astype(str).unique()
    rng = np.random.default_rng(seed)
    keys = keys.copy()
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_fraction))) if len(keys) > 1 else 0
    test_keys = set(keys[:n_test])
    test = df[key_col].astype(str).isin(test_keys).to_numpy()
    return ~test, test


def y_for_task(labels: np.ndarray, task: str) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    if task == "warning_or_above":
        return (y >= 2).astype(int)
    if task == "emergency_only":
        return (y == 3).astype(int)
    if task == "safe_vs_risky":
        return (y >= 1).astype(int)
    raise ValueError(f"unknown task={task}")


def recall_at_fpr(y_true: np.ndarray, score: np.ndarray, fpr_level: float) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    neg = s[y == 0]
    if len(neg) == 0:
        return np.nan, np.nan, np.nan
    # threshold such that at most fpr_level of negatives are above threshold.
    thr = float(np.quantile(neg, max(0.0, min(1.0, 1.0 - fpr_level))))
    pred = s >= thr
    tp = np.sum(pred & (y == 1))
    fn = np.sum((~pred) & (y == 1))
    fp = np.sum(pred & (y == 0))
    tn = np.sum((~pred) & (y == 0))
    recall = float(tp / max(tp + fn, 1))
    fpr = float(fp / max(fp + tn, 1))
    return recall, fpr, thr


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(score, dtype=float)
    ok = np.isfinite(p)
    y = y[ok]
    p = np.clip(p[ok], 0.0, 1.0)
    if len(y) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not np.any(m):
            continue
        ece += float(np.mean(m) * abs(np.mean(p[m]) - np.mean(y[m])))
    return ece


def binary_score_metrics(y_true: np.ndarray, score: np.ndarray, fpr_levels=(0.01, 0.05)) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    out = {"n_eval": int(len(y)), "positive_rate": float(np.mean(y)) if len(y) else np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "ece": np.nan, "best_f1": np.nan})
    else:
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
        p = np.clip(s, 0.0, 1.0)
        out["brier"] = float(brier_score_loss(y, p))
        out["ece"] = float(expected_calibration_error(y, p))
        prec, rec, thr = precision_recall_curve(y, s)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        out["best_f1"] = float(np.nanmax(f1))
    for fpr in fpr_levels:
        rec_v, fpr_v, thr_v = recall_at_fpr(y, s, float(fpr))
        pct = int(round(float(fpr) * 100))
        out[f"recall_at_{pct}pct_fpr"] = rec_v
        out[f"actual_fpr_at_{pct}pct"] = fpr_v
        out[f"threshold_at_{pct}pct_fpr"] = thr_v
    return out


def bootstrap_metric_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray | None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        mask_idx = []
        for g in sampled:
            idx = np.where(groups == g)[0]
            if len(idx):
                mask_idx.append(idx)
        if not mask_idx:
            continue
        idx = np.concatenate(mask_idx)
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            vals.append(float(metric_fn(yy, ss)))
        except Exception:
            continue
    point = float(metric_fn(y, s)) if len(np.unique(y)) >= 2 else np.nan
    if not vals:
        return point, np.nan, np.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def normalize_score(series: pd.Series, invert: bool = False) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if invert:
        finite = np.isfinite(x)
        if finite.any():
            maxv = np.nanpercentile(x[finite], 99)
            x = maxv - x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(len(x), dtype=float)
    lo, hi = np.nanpercentile(x[finite], [1, 99])
    if hi <= lo:
        lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi <= lo:
        return np.zeros(len(x), dtype=float)
    y = (x - lo) / (hi - lo)
    y[~np.isfinite(y)] = 0.0
    return np.clip(y, 0.0, 1.0)


def score_definitions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    def add(name: str, col: str, invert: bool = False):
        if col in df.columns:
            scores[name] = normalize_score(df[col], invert=invert)
    add("REDI_actionability", "redi_actionability")
    add("REDI_full", "redi_full")
    add("REDI_delta", "redi_actionability_delta")
    add("RCR", "rcr")
    for col, name in [
        ("asr_cum_final", "ASR_cum_inverse"),
        ("asr_cum_min", "ASR_cum_min_inverse"),
        ("asr_slice_min", "ASR_slice_min_inverse"),
        ("comfort_asr", "comfort_ASR_inverse"),
        ("emergency_asr", "emergency_ASR_inverse"),
    ]:
        if col in df.columns:
            scores[name] = normalize_score(1.0 - pd.to_numeric(df[col], errors="coerce"), invert=False)
    # Backward compatibility if only v1.0 exists.
    if "ASR_cum_inverse" not in scores and "asr" in df.columns:
        scores["ASR_inverse"] = normalize_score(1.0 - pd.to_numeric(df["asr"], errors="coerce"), invert=False)
    add("TTAD_inverse", "ttad_s", invert=True)
    add("time_to_first_conflict_inverse", "time_to_first_conflict_s", invert=True)
    add("collapse_rate", "collapse_rate_per_s")
    add("collapse_rate_max", "collapse_rate_max_per_s")
    add("early_blocking", "early_blocking_ratio")
    add("distance_inverse", "current_min_distance_m", invert=True)
    if "current_ttc_s" in df.columns:
        t = pd.to_numeric(df["current_ttc_s"], errors="coerce")
        t = t.where(t >= 0, np.nan)
        scores["TTC_inverse"] = normalize_score(t, invert=True)
    # A robust scalar composite that does not require model training.
    parts = []
    for key in ["REDI_actionability", "ASR_cum_min_inverse", "comfort_ASR_inverse", "TTAD_inverse", "collapse_rate_max", "early_blocking"]:
        if key in scores:
            parts.append(scores[key])
    if parts:
        scores["ROF_v2_composite"] = np.nanmean(np.vstack(parts), axis=0)
    return scores


def diagnostic_subset_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Subsets designed to reveal value beyond raw proximity/TTC.

    These masks are intentionally diagnostic; always report the natural `all`
    result alongside them.
    """
    n = len(df)
    mask_all = np.ones(n, dtype=bool)
    def col(name, default=np.nan):
        if name not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dist = col("current_min_distance_m")
    ttc = col("current_ttc_s")
    speed = col("ego_speed_kph")
    agents = col("agent_count")
    labels = pd.to_numeric(df.get("label_id", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    masks = {"all": mask_all}
    masks["no_ttc"] = ((ttc < 0) | (~np.isfinite(ttc))).to_numpy(bool)
    masks["large_or_no_ttc"] = ((ttc < 0) | (~np.isfinite(ttc)) | (ttc > 3.0)).to_numpy(bool)
    masks["low_speed_lt15kph"] = (speed < 15.0).fillna(False).to_numpy(bool)
    if agents.notna().any():
        p75 = float(agents.quantile(0.75))
        masks["dense_agents_p75"] = (agents >= p75).fillna(False).to_numpy(bool)
    masks["close_distance_lt5m"] = (dist < 5.0).fillna(False).to_numpy(bool)
    masks["hard_safe_close"] = ((dist < 5.0) & (labels <= 1)).fillna(False).to_numpy(bool)
    masks["future_risk_no_ttc"] = (((ttc < 0) | (~np.isfinite(ttc))) & (labels >= 2)).fillna(False).to_numpy(bool)
    masks["future_risk_large_ttc"] = (((ttc > 3.0) | (ttc < 0) | (~np.isfinite(ttc))) & (labels >= 2)).fillna(False).to_numpy(bool)
    return {k: np.asarray(v, dtype=bool) for k, v in masks.items()}
