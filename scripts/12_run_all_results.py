from __future__ import annotations
import argparse
from pathlib import Path
import runpy
import sys

from _bootstrap import ROOT


def _run(script: str, config: str):
    print("\n" + "=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    old = sys.argv[:]
    sys.argv = [script, "--config", config]
    try:
        runpy.run_path(str(Path(__file__).with_name(script)), run_name="__main__")
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    for s in [
        "06_summarize_dataset.py",
        "07_metric_label_correlation.py",
        "08_train_classifiers.py",
        "09_risk_coverage.py",
        "10_ablation.py",
        "11_visualize_cases.py",
        "13_stratified_early_warning.py",
    ]:
        _run(s, args.config)

if __name__ == "__main__":
    main()
