from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.analysis_utils import load_merged, high_emergency_binary

LABEL_NAMES = {0: "safe", 1: "caution", 2: "warning", 3: "emergency"}


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _risk_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Scores where larger means more urgent."""
    scores: dict[str, np.ndarray] = {}

    def norm(x, invert=False):
        x = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        if invert:
            finite = np.isfinite(x)
            if finite.any():
                x = np.nanmax(x[finite]) - x
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x, dtype=float)
        lo, hi = np.nanpercentile(x[finite], [1, 99])
        if hi <= lo:
            lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
        if hi <= lo:
            return np.zeros_like(x, dtype=float)
        y = (x - lo) / (hi - lo)
        y[~np.isfinite(y)] = 0.0
        return np.clip(y, 0.0, 1.0)

    if "redi_full" in df.columns:
        scores["REDI_full"] = norm(df["redi_full"])
    if "redi_no_msr" in df.columns:
        scores["REDI_no_MSR"] = norm(df["redi_no_msr"])
    if "rcr" in df.columns:
        scores["RCR"] = norm(df["rcr"])
    if "rfr_drv" in df.columns:
        scores["RFR_drv_inverse"] = norm(df["rfr_drv"], invert=True)
    if "c_time" in df.columns:
        scores["C_time"] = norm(df["c_time"])
    if "msr" in df.columns:
        scores["MSR_inverse"] = norm(df["msr"], invert=True)
    if "gtoa_norm_union" in df.columns:
        scores["Global_overlap"] = norm(df["gtoa_norm_union"])

    # Revised CV composite: swept/inflated CV occupancy terms.
    cv_rcr = _num(df, "cv_rcr", 0.0).fillna(0.0)
    cv_ctime = _num(df, "cv_c_time", 0.0).fillna(0.0)
    cv_cdensity = _num(df, "cv_c_density", 0.0).fillna(0.0)
    if any(c in df.columns for c in ["cv_rcr", "cv_c_time", "cv_c_density"]):
        scores["CV_composite"] = norm(0.5 * cv_rcr + 0.3 * cv_ctime + 0.2 * cv_cdensity)
        scores["CV_RCR"] = norm(cv_rcr)

    if "current_min_distance_m" in df.columns:
        scores["Current_distance_inverse"] = norm(df["current_min_distance_m"], invert=True)
    if "current_ttc_s" in df.columns:
        ttc = _num(df, "current_ttc_s")
        # no TTC (<0) means low current-risk for inverse-TTC scoring.
        finite_nonneg = ttc[(ttc >= 0) & np.isfinite(ttc)]
        large_ttc = max(float(finite_nonneg.quantile(0.95)) if len(finite_nonneg) else 10.0, 10.0)
        ttc = ttc.copy()
        ttc[(ttc < 0) | (~np.isfinite(ttc))] = large_ttc
        scores["Current_TTC_inverse"] = norm(ttc, invert=True)
    return scores


def _make_subsets(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    acfg = cfg.get("analysis", {})
    d1 = float(acfg.get("early_warning_distance_m", 4.0))
    d2 = float(acfg.get("early_warning_distance_strict_m", 6.0))
    t1 = float(acfg.get("early_warning_ttc_s", 2.0))
    t2 = float(acfg.get("early_warning_ttc_strict_s", 3.0))

    dist = _num(df, "current_min_distance_m", np.nan)
    ttc = _num(df, "current_ttc_s", np.nan)
    no_ttc_or_ge_t1 = (ttc < 0) | (ttc >= t1) | (~np.isfinite(ttc))
    no_ttc_or_ge_t2 = (ttc < 0) | (ttc >= t2) | (~np.isfinite(ttc))

    subsets = {
        "all": np.ones(len(df), dtype=bool),
        f"distance_ge_{d1:g}m": (dist >= d1).fillna(False).to_numpy(),
        f"distance_ge_{d2:g}m": (dist >= d2).fillna(False).to_numpy(),
        f"ttc_ge_{t1:g}s_or_none": no_ttc_or_ge_t1.to_numpy(),
        f"ttc_ge_{t2:g}s_or_none": no_ttc_or_ge_t2.to_numpy(),
        f"non_obvious_d{d1:g}_ttc{t1:g}": ((dist >= d1) & no_ttc_or_ge_t1).fillna(False).to_numpy(),
        f"strict_non_obvious_d{d2:g}_ttc{t2:g}": ((dist >= d2) & no_ttc_or_ge_t2).fillna(False).to_numpy(),
        f"current_close_dist_lt_{d1:g}m": (dist < d1).fillna(False).to_numpy(),
    }
    return subsets


def _subset_summary(df: pd.DataFrame, subsets: dict[str, np.ndarray], pos_min: int) -> pd.DataFrame:
    rows = []
    y = df["label_id"].astype(int).to_numpy()
    for name, mask in subsets.items():
        sub = df.loc[mask]
        if len(sub) == 0:
            rows.append({"subset": name, "n": 0})
            continue
        counts = sub["label_id"].astype(int).value_counts().to_dict()
        row = {
            "subset": name,
            "n": int(len(sub)),
            "high_urgent_n": int(np.sum(y[mask] >= pos_min)),
            "high_urgent_fraction": float(np.mean(y[mask] >= pos_min)),
        }
        for i in range(4):
            row[f"n_{LABEL_NAMES[i]}"] = int(counts.get(i, 0))
        for c in ["current_min_distance_m", "current_ttc_s", "rcr", "rfr_drv", "c_time", "msr", "redi_full", "redi_no_msr", "cv_rcr", "cv_c_time"]:
            if c in sub.columns:
                x = _num(sub, c)
                row[f"{c}_median"] = float(x.median()) if x.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _score_metrics(df: pd.DataFrame, subsets: dict[str, np.ndarray], scores: dict[str, np.ndarray], pos_min: int, min_n: int) -> pd.DataFrame:
    y_label = df["label_id"].astype(int).to_numpy()
    y_pos_all = high_emergency_binary(y_label, pos_min)
    rows = []
    for subset_name, mask in subsets.items():
        if int(mask.sum()) < min_n:
            continue
        y_pos = y_pos_all[mask]
        y_ord = y_label[mask]
        for score_name, score_all in scores.items():
            s = np.asarray(score_all, dtype=float)[mask]
            ok = np.isfinite(s)
            if int(ok.sum()) < min_n or len(np.unique(s[ok])) < 2:
                rows.append({"subset": subset_name, "score": score_name, "n": int(ok.sum()), "note": "constant_or_too_few"})
                continue
            y_pos_ok = y_pos[ok]
            y_ord_ok = y_ord[ok]
            row = {"subset": subset_name, "score": score_name, "n": int(ok.sum()), "note": ""}
            if len(np.unique(y_pos_ok)) == 2:
                row["high_urgent_auc"] = float(roc_auc_score(y_pos_ok, s[ok]))
                row["high_urgent_ap"] = float(average_precision_score(y_pos_ok, s[ok]))
                # Threshold-free-ish operating summary: best F1 over observed score thresholds.
                best = {"f1": -1.0}
                for thr in np.unique(np.quantile(s[ok], np.linspace(0, 1, 101))):
                    pred = (s[ok] >= thr).astype(int)
                    f1 = f1_score(y_pos_ok, pred, zero_division=0)
                    if f1 > best["f1"]:
                        best = {
                            "f1": float(f1),
                            "threshold": float(thr),
                            "precision": float(precision_score(y_pos_ok, pred, zero_division=0)),
                            "recall": float(recall_score(y_pos_ok, pred, zero_division=0)),
                        }
                row.update({f"best_{k}": v for k, v in best.items()})
            else:
                row["high_urgent_auc"] = np.nan
                row["high_urgent_ap"] = np.nan
            if len(np.unique(y_ord_ok)) >= 2:
                sp = spearmanr(s[ok], y_ord_ok)
                kt = kendalltau(s[ok], y_ord_ok)
                row["spearman_rho_label"] = float(sp.statistic)
                row["spearman_p_label"] = float(sp.pvalue)
                row["kendall_tau_label"] = float(kt.statistic)
                row["kendall_p_label"] = float(kt.pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_subset_metrics(df: pd.DataFrame, subsets: dict[str, np.ndarray], work: Path, pos_min: int, min_n: int) -> pd.DataFrame:
    pred_csv = work / "results" / "classification" / "classification_predictions.csv"
    if not pred_csv.exists():
        return pd.DataFrame()
    pred = pd.read_csv(pred_csv)
    if pred.empty:
        return pd.DataFrame()
    meta_cols = ["sample_id", "label_id", "label_name", "current_min_distance_m", "current_ttc_s"]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()
    out_rows = []
    for subset_name, mask in subsets.items():
        subset_ids = set(df.loc[mask, "sample_id"].astype(str))
        pp = pred[pred["sample_id"].astype(str).isin(subset_ids)].copy()
        if len(pp) < min_n:
            continue
        for (fs, model), sub in pp.groupby(["feature_set", "model"]):
            y_true = sub["y_true"].astype(int).to_numpy()
            y_pred = sub["y_pred"].astype(int).to_numpy()
            row = {
                "subset": subset_name,
                "feature_set": fs,
                "model": model,
                "n_test_subset": int(len(sub)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
            pos_true = high_emergency_binary(y_true, pos_min)
            pos_pred = high_emergency_binary(y_pred, pos_min)
            row["emergency_recall"] = float(recall_score(pos_true, pos_pred, zero_division=0))
            row["emergency_precision"] = float(precision_score(pos_true, pos_pred, zero_division=0))
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def _strata_tables(df: pd.DataFrame, out: Path):
    dist = _num(df, "current_min_distance_m")
    ttc = _num(df, "current_ttc_s")
    dd = df.copy()
    dd["distance_stratum"] = pd.cut(dist, bins=[-np.inf, 2, 4, 6, np.inf], labels=["<2m", "2-4m", "4-6m", ">=6m"])
    # -1 / no TTC is kept as a separate stratum.
    ttc_str = pd.Series("no_ttc", index=df.index, dtype=object)
    ttc_str[(ttc >= 0) & (ttc < 1)] = "0-1s"
    ttc_str[(ttc >= 1) & (ttc < 2)] = "1-2s"
    ttc_str[(ttc >= 2) & (ttc < 3)] = "2-3s"
    ttc_str[(ttc >= 3)] = ">=3s"
    dd["ttc_stratum"] = ttc_str
    for col in ["distance_stratum", "ttc_stratum"]:
        rows = []
        for name, sub in dd.groupby(col, dropna=False, observed=False):
            row = {"stratum": str(name), "n": int(len(sub))}
            for i in range(4):
                row[f"n_{LABEL_NAMES[i]}"] = int((sub["label_id"].astype(int) == i).sum())
            for m in ["rcr", "rfr_drv", "c_time", "msr", "redi_full", "cv_rcr", "cv_c_time"]:
                if m in sub.columns:
                    row[f"{m}_median"] = float(_num(sub, m).median())
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"{col}_summary.csv", index=False)


def _plot_score_bars(score_df: pd.DataFrame, out: Path):
    if score_df.empty or "high_urgent_auc" not in score_df.columns:
        return
    keep_scores = ["Current_distance_inverse", "Current_TTC_inverse", "CV_composite", "RCR", "RFR_drv_inverse", "MSR_inverse", "REDI_full", "REDI_no_MSR"]
    df = score_df[score_df["score"].isin(keep_scores)].copy()
    df = df[np.isfinite(pd.to_numeric(df["high_urgent_auc"], errors="coerce"))]
    if df.empty:
        return
    # Plot only the most paper-relevant subsets to avoid clutter.
    subset_order = [s for s in ["all"] + [x for x in df["subset"].unique() if "non_obvious" in str(x)] if s in set(df["subset"])]
    if not subset_order:
        subset_order = list(df["subset"].unique())[:4]
    for subset in subset_order:
        sub = df[df["subset"] == subset].sort_values("high_urgent_auc", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.barh(sub["score"], sub["high_urgent_auc"])
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("High-urgent AUROC")
        ax.set_title(f"Score comparison: {subset}")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        safe_name = subset.replace("/", "_").replace(" ", "_")
        fig.savefig(out / f"fig_score_auc_{safe_name}.png", bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "stratified_early_warning")
    df = load_merged(work)
    pos_min = int(cfg.get("analysis", {}).get("emergency_positive_label", cfg.get("labels", {}).get("high_emergency_min_label", 2)))
    min_n = int(cfg.get("analysis", {}).get("stratified_min_subset_n", 20))

    subsets = _make_subsets(df, cfg)
    summary = _subset_summary(df, subsets, pos_min)
    summary.to_csv(out / "subset_summary.csv", index=False)

    scores = _risk_scores(df)
    score_df = _score_metrics(df, subsets, scores, pos_min, min_n)
    score_df.to_csv(out / "subset_score_metrics.csv", index=False)

    pred_df = _prediction_subset_metrics(df, subsets, work, pos_min, min_n)
    if not pred_df.empty:
        pred_df.to_csv(out / "subset_classifier_metrics.csv", index=False)

    _strata_tables(df, out)
    _plot_score_bars(score_df, out)
    print(f"[stratified] wrote {out}")


if __name__ == "__main__":
    main()
