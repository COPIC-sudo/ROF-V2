from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.analysis_utils import load_merged
from rtbev.nc_eval import normalize_score, y_for_task


def _load_features(work: Path, features_csv: str | None) -> pd.DataFrame:
    if features_csv:
        return pd.read_csv(features_csv)
    return load_merged(work)


def _score_auc_ap(df: pd.DataFrame, col: str, task: str, invert: bool = False) -> dict:
    y = y_for_task(df["label_id"].to_numpy(), task)
    s = normalize_score(df[col], invert=invert)
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def main() -> None:
    ap = argparse.ArgumentParser(description="NC-minimal: ROF-v2 low-speed tube/actionability sanity checks.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--features-csv", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "nc_rof_v2_sanity")
    df = _load_features(work, args.features_csv)
    if df.empty:
        raise SystemExit("features csv is empty")

    rows = []
    if {"ego_speed_kph", "ego_speed_bin_kph"}.issubset(df.columns):
        x = pd.to_numeric(df["ego_speed_kph"], errors="coerce")
        b = pd.to_numeric(df["ego_speed_bin_kph"], errors="coerce")
        err = (x - b).abs()
        rows.append({"check": "speed_bin_abs_error_mean_kph", "value": float(err.mean())})
        rows.append({"check": "speed_bin_abs_error_p95_kph", "value": float(err.quantile(0.95))})
        rows.append({"check": "speed_bin_abs_error_max_kph", "value": float(err.max())})
        bins = [-np.inf, 5, 15, 30, 60, np.inf]
        labels = ["<5", "5-15", "15-30", "30-60", ">=60"]
        speed_stratum = pd.cut(x, bins=bins, labels=labels)
        tab = []
        for name, sub in df.groupby(speed_stratum, observed=False):
            if len(sub) == 0:
                continue
            row = {"speed_stratum_kph": str(name), "n": int(len(sub))}
            for c in ["ego_speed_bin_abs_error_kph", "rcr", "redi_full", "asr", "ttad_s", "collapse_rate_per_s"]:
                if c in sub.columns:
                    row[f"{c}_median"] = float(pd.to_numeric(sub[c], errors="coerce").median())
            tab.append(row)
        pd.DataFrame(tab).to_csv(out / "speed_stratum_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.hist(err.dropna().to_numpy(), bins=40)
        ax.set_xlabel("|ego speed - selected tube speed bin| (km/h)")
        ax.set_ylabel("count")
        ax.set_title("Speed-bin mismatch after ROF-v2 tube selection")
        fig.tight_layout()
        fig.savefig(out / "fig_speed_bin_mismatch.png", bbox_inches="tight")
        plt.close(fig)

    # v1.1 actionability non-degeneracy checks.
    def _max_abs_diff(a, b):
        if a not in df.columns or b not in df.columns:
            return np.nan
        x = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")
        return float(x.abs().max()) if x.notna().any() else np.nan
    for a, b, label in [
        ("redi_actionability", "redi_full", "max_abs_redi_actionability_minus_redi_full"),
        ("asr_slice_final", "asr_cum_final", "max_abs_asr_slice_final_minus_cum_final"),
        ("asr_slice_min", "asr_cum_min", "max_abs_asr_slice_min_minus_cum_min"),
    ]:
        rows.append({"check": label, "value": _max_abs_diff(a, b)})
    for col in ["ttad_s", "collapse_rate_max_per_s", "early_blocking_ratio", "comfort_to_emergency_gap"]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            rows.append({"check": f"{col}_finite_fraction", "value": float(x.notna().mean())})
            rows.append({"check": f"{col}_median", "value": float(x.median()) if x.notna().any() else np.nan})

    metric_rows = []
    for task in ["warning_or_above", "emergency_only"]:
        for name, col, inv in [
            ("distance_inverse", "current_min_distance_m", True),
            ("TTC_inverse", "current_ttc_s", True),
            ("REDI_full", "redi_full", False),
            ("REDI_actionability", "redi_actionability", False),
            ("ASR_cum_inverse", "asr_cum_final", True),
            ("ASR_slice_min_inverse", "asr_slice_min", True),
            ("TTAD_inverse", "ttad_s", True),
            ("collapse_rate", "collapse_rate_per_s", False),
            ("early_blocking", "early_blocking_ratio", False),
        ]:
            if col not in df.columns:
                continue
            m = _score_auc_ap(df, col, task, invert=inv)
            metric_rows.append({"task": task, "score": name, "column": col, "n": int(len(df)), **m})
    pd.DataFrame(metric_rows).to_csv(out / "rof_v2_scalar_sanity_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "rof_v2_sanity_checks.csv", index=False)
    print(f"[nc-sanity] wrote {out}")


if __name__ == "__main__":
    main()
