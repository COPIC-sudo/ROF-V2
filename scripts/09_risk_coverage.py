from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.analysis_utils import load_merged, high_emergency_binary


def _norm_score(s: pd.Series, invert: bool = False):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if invert:
        # smaller is riskier; convert to high-risk score
        finite = np.isfinite(x)
        if finite.any():
            maxv = np.nanmax(x[finite])
            x = maxv - x
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
    return np.clip(y, 0, 1)


def _curves(score: np.ndarray, y_pos: np.ndarray, n_thr: int):
    rows = []
    for thr in np.linspace(0, 1, n_thr):
        alert = score >= thr
        TP = int(np.sum(alert & (y_pos == 1)))
        FP = int(np.sum(alert & (y_pos == 0)))
        FN = int(np.sum((~alert) & (y_pos == 1)))
        TN = int(np.sum((~alert) & (y_pos == 0)))
        alert_rate = float(np.mean(alert))
        emergency_recall = TP / max(TP + FN, 1)
        false_alert_risk = FP / max(TP + FP, 1)
        safe = score < thr
        safe_coverage = float(np.mean(safe))
        missed_emergency_risk = FN / max(int(np.sum(safe)), 1)
        rows.append({"threshold": thr, "alert_rate": alert_rate, "emergency_recall": emergency_recall, "false_alert_risk": false_alert_risk, "safe_coverage": safe_coverage, "missed_emergency_risk": missed_emergency_risk, "TP": TP, "FP": FP, "FN": FN, "TN": TN})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "risk_coverage")
    df = load_merged(work)
    pos_min = int(cfg.get("analysis", {}).get("emergency_positive_label", cfg.get("labels", {}).get("high_emergency_min_label", 2)))
    y_pos = high_emergency_binary(df["label_id"].astype(int).to_numpy(), pos_min)

    # Composite score for the revised constant-velocity occupancy baseline.
    # It mirrors REDI structure loosely but only uses CV occupancy terms.
    for c in ["cv_rcr", "cv_c_time", "cv_c_density"]:
        if c not in df.columns:
            df[c] = np.nan
    df["_cv_composite_score"] = (
        0.5 * pd.to_numeric(df["cv_rcr"], errors="coerce").fillna(0.0)
        + 0.3 * pd.to_numeric(df["cv_c_time"], errors="coerce").fillna(0.0)
        + 0.2 * pd.to_numeric(df["cv_c_density"], errors="coerce").fillna(0.0)
    )

    score_defs = {
        "REDI_full": ("redi_full", False),
        "REDI_no_MSR": ("redi_no_msr", False),
        "RCR": ("rcr", False),
        "CV_composite": ("_cv_composite_score", False),
        "CV_RCR": ("cv_rcr", False),
        "CV_C_time": ("cv_c_time", False),
        "distance_inverse": ("current_min_distance_m", True),
    }
    if "current_ttc_s" in df.columns:
        # use inverse only for finite nonnegative TTC; negative no TTC => low risk.
        ttc = pd.to_numeric(df["current_ttc_s"], errors="coerce")
        tmp = ttc.copy()
        tmp[tmp < 0] = np.nan
        df["_ttc_for_score"] = tmp
        score_defs["TTC_inverse"] = ("_ttc_for_score", True)
    all_rows = []
    n_thr = int(cfg.get("analysis", {}).get("n_thresholds", 101))
    for name, (col, inv) in score_defs.items():
        if col not in df.columns:
            continue
        score = _norm_score(df[col], invert=inv)
        curves = _curves(score, y_pos, n_thr)
        curves["score_name"] = name
        all_rows.append(curves)
    res = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    res.to_csv(out / "risk_coverage_curves.csv", index=False)
    if not res.empty:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        for name, sub in res.groupby("score_name"):
            ax.plot(sub["alert_rate"], sub["false_alert_risk"], label=name)
        ax.set_xlabel("AlertRate = flagged / N")
        ax.set_ylabel("FalseAlertRisk = FP / alerted")
        ax.set_title("Emergency-alert view")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "fig_alert_risk_coverage.png", bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        for name, sub in res.groupby("score_name"):
            ax.plot(sub["safe_coverage"], sub["missed_emergency_risk"], label=name)
        ax.set_xlabel("SafeCoverage = accepted-safe / N")
        ax.set_ylabel("MissedEmergencyRisk = urgent in safe-accepted / safe-accepted")
        ax.set_title("Safe-accept triage view")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "fig_safe_accept_risk_coverage.png", bbox_inches="tight")
        plt.close(fig)
    print(f"[risk] wrote {out}")

if __name__ == "__main__":
    main()
