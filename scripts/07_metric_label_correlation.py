from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.analysis_utils import load_merged, label_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "metric_label_correlation")
    df = load_merged(work)


    horizon = float(cfg.get("tube", {}).get("horizon_s", cfg.get("labels", {}).get("future_horizon_s", 2.0)))
    dt = float(cfg.get("tube", {}).get("query_dt_s", cfg.get("labels", {}).get("dt_s", 0.1)))
    no_conflict_time = horizon + dt

    if "tfrc_s" in df.columns:
        df["tfrc_eval_s"] = pd.to_numeric(df["tfrc_s"], errors="coerce")
        df.loc[df["tfrc_eval_s"] < 0, "tfrc_eval_s"] = no_conflict_time

    if "cv_tfrc_s" in df.columns:
        df["cv_tfrc_eval_s"] = pd.to_numeric(df["cv_tfrc_s"], errors="coerce")
        df.loc[df["cv_tfrc_eval_s"] < 0, "cv_tfrc_eval_s"] = no_conflict_time


    # metrics = [c for c in ["rcr", "rfr", "rfr_drv", "tfrc_s", "msr", "weighted_overlap_area_m2", "gtoa_norm_union", "oce_norm", "redi_full", "redi_no_msr", "cv_rcr", "cv_tfrc_s"] if c in df.columns]
    metrics = [
        c for c in [
            "rcr",
            "rfr",
            "rfr_drv",
            "tfrc_eval_s",
            "c_time",
            "msr",
            "weighted_overlap_area_m2",
            "gtoa_norm_union",
            "oce_norm",
            "redi_full",
            "redi_no_msr",
            "cv_rcr",
            "cv_tfrc_eval_s",
            "cv_c_time",
        ]
        if c in df.columns
    ]
    rows = []
    y = df["label_id"].astype(int).to_numpy()
    for m in metrics:
        x = pd.to_numeric(df[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ok = x.notna().to_numpy()
        if ok.sum() < 3:
            continue

        #revise
        x_ok = x[ok].to_numpy(dtype=float)
        y_ok = y[ok]

        # Skip correlation for constant metrics, e.g., degenerate CV baseline.
        if len(np.unique(x_ok)) < 2 or len(np.unique(y_ok)) < 2:
            rows.append({
                "metric": m,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "kendall_tau": np.nan,
                "kendall_p": np.nan,
                "n": int(ok.sum()),
                "note": "constant input; correlation undefined",
            })
            continue

        sp = spearmanr(x_ok, y_ok)
        kt = kendalltau(x_ok, y_ok)
        rows.append({
            "metric": m,
            "spearman_rho": sp.statistic,
            "spearman_p": sp.pvalue,
            "kendall_tau": kt.statistic,
            "kendall_p": kt.pvalue,
            "n": int(ok.sum()),
            "note": "",
        })


        # sp = spearmanr(x[ok], y[ok])
        # kt = kendalltau(x[ok], y[ok])
        # rows.append({"metric": m, "spearman_rho": sp.statistic, "spearman_p": sp.pvalue, "kendall_tau": kt.statistic, "kendall_p": kt.pvalue, "n": int(ok.sum())})
    pd.DataFrame(rows).to_csv(out / "metric_label_correlation.csv", index=False)

    for m in metrics:
        x = pd.to_numeric(df[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
        data = [x[df["label_name"] == lab].dropna().to_numpy() for lab in label_order()]
        if sum(len(a) for a in data) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        # ax.boxplot(data, labels=label_order(), showfliers=False)
        try:
            ax.boxplot(data, tick_labels=label_order(), showfliers=False)
        except TypeError:
            # For older matplotlib versions.
            ax.boxplot(data, labels=label_order(), showfliers=False)

        ax.set_title(f"{m} vs future label")
        ax.set_xlabel("future-trajectory label")
        ax.set_ylabel(m)
        fig.tight_layout()
        fig.savefig(out / f"fig_box_{m}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"[corr] wrote {out}")

if __name__ == "__main__":
    main()
