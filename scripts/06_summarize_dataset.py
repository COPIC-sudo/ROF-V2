from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
    out = ensure_dir(work / "results" / "summary")
    df = load_merged(work)

    horizon = float(cfg.get("tube", {}).get("horizon_s", cfg.get("labels", {}).get("future_horizon_s", 2.0)))
    dt = float(cfg.get("tube", {}).get("query_dt_s", cfg.get("labels", {}).get("dt_s", 0.1)))
    no_conflict_time = horizon + dt

    if "tfrc_s" in df.columns:
        df["tfrc_eval_s"] = pd.to_numeric(df["tfrc_s"], errors="coerce")
        df.loc[df["tfrc_eval_s"] < 0, "tfrc_eval_s"] = no_conflict_time



    counts = df["label_name"].value_counts().reindex(label_order(), fill_value=0).reset_index()
    counts.columns = ["label_name", "count"]
    counts["fraction"] = counts["count"] / max(counts["count"].sum(), 1)
    counts.to_csv(out / "label_distribution.csv", index=False)
    # num_cols = [c for c in ["ego_speed_kph", "ego_speed_mismatch_abs_kph", "agent_count", "dmin_future_m", "rcr", "rfr", "rfr_drv", "tfrc_s", "msr", "redi_full", "redi_no_msr"] if c in df.columns]
    num_cols = [
        c for c in [
            "ego_speed_kph",
            "ego_speed_mismatch_abs_kph",
            "agent_count",
            "dmin_future_m",
            "rcr",
            "rfr",
            "rfr_drv",
            "tfrc_s",
            "tfrc_eval_s",
            "c_time",
            "msr",
            "redi_full",
            "redi_no_msr",
        ]
        if c in df.columns
    ]

    desc = df[num_cols].describe().T if num_cols else pd.DataFrame()
    desc.to_csv(out / "feature_describe.csv")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(counts["label_name"], counts["count"])
    ax.set_title("Label distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / "fig_label_distribution.png", bbox_inches="tight")
    plt.close(fig)
    if "ego_speed_mismatch_abs_kph" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.hist(df["ego_speed_mismatch_abs_kph"].dropna(), bins=30)
        ax.set_xlabel("ego speed-bin mismatch [km/h]")
        ax.set_ylabel("count")
        ax.set_title("Tube speed-bin mismatch")
        fig.tight_layout()
        fig.savefig(out / "fig_speed_mismatch.png", bbox_inches="tight")
        plt.close(fig)
    print(f"[summary] wrote {out}")
    print(counts.to_string(index=False))

if __name__ == "__main__":
    main()
