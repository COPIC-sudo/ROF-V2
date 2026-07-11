from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir, read_gzip_pickle
from rtbev.tube.rt_library import TubeLibrary
from rtbev.visualize import render_sample_overlay, render_tensor_maps
from rtbev.analysis_utils import load_merged, label_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-cases", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "case_visualizations")
    df = load_merged(work)
    lib = TubeLibrary.from_workdir(work / "tube_library")
    chosen = []
    for lab in label_order():
        sub = df[df["label_name"] == lab]
        if not sub.empty:
            if "redi_full" in sub.columns and sub["redi_full"].notna().any():
                row = sub.sort_values("redi_full", ascending=False).iloc[0]
            else:
                row = sub.iloc[0]
            chosen.append((f"label_{lab}", row["sample_id"]))
    for col, asc, tag in [("redi_full", False, "highest_redi"), ("redi_no_msr", False, "highest_redi_no_msr"), ("rfr_drv", True, "lowest_rfr_drv"), ("msr", True, "lowest_msr")]:
        if col in df.columns and df[col].notna().any():
            row = df.sort_values(col, ascending=asc).iloc[0]
            chosen.append((tag, row["sample_id"]))
    seen = set()
    count = 0
    for tag, sid in chosen:
        if sid in seen or count >= args.max_cases:
            continue
        seen.add(sid); count += 1
        sample = read_gzip_pickle(work / "samples" / f"{sid}.pkl.gz")
        overlay = out / f"{tag}_{sid}_overlay.png"
        render_sample_overlay(sample, lib, cfg, overlay)
        tensor_path = work / "bev_tensors" / f"{sid}.npz"
        if tensor_path.exists():
            render_tensor_maps(tensor_path, cfg, out / f"{tag}_{sid}_maps.png", title=f"{tag} | {sid}")
    pd.DataFrame([{"tag": t, "sample_id": s} for t, s in chosen]).drop_duplicates().to_csv(out / "selected_cases.csv", index=False)
    print(f"[cases] wrote {out}")

if __name__ == "__main__":
    main()
