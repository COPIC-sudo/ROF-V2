from __future__ import annotations
# This script is a thin wrapper around 08_train_classifiers.py; it extracts the feature-set
# comparison table as the ablation table used by the paper.
import argparse
from pathlib import Path
import runpy
import pandas as pd

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    # Ensure classification table exists.
    import sys
    old_argv = sys.argv[:]
    sys.argv = ["08_train_classifiers.py", "--config", args.config]
    try:
        runpy.run_path(str(Path(__file__).with_name("08_train_classifiers.py")), run_name="__main__")
    finally:
        sys.argv = old_argv
    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    cls_csv = work / "results" / "classification" / "classification_metrics.csv"
    df = pd.read_csv(cls_csv)
    # Keep RandomForest rows by default for a compact ablation table.
    abl = df[df.get("model", "") == "rf"].copy() if "model" in df.columns else df.copy()
    out = ensure_dir(work / "results" / "ablation")
    abl.to_csv(out / "ablation_table_rf.csv", index=False)
    print(f"[ablation] wrote {out / 'ablation_table_rf.csv'}")

if __name__ == "__main__":
    main()
