#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v112/nc_v112_field_baselines.yaml")
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--planner-labels-csv", default=None)
    parser.add_argument("--rof-scores-csv", default=None)
    parser.add_argument("--bootstrap-n", type=int, default=None)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    cmd0 = [sys.executable, "scripts/nc_v112/00_compute_field_baselines.py", "--config", args.config]
    if args.features_csv:
        cmd0 += ["--features-csv", args.features_csv]
    _run(cmd0)
    cmd1 = [sys.executable, "scripts/nc_v112/01_evaluate_field_baselines.py", "--config", args.config]
    if args.planner_labels_csv:
        cmd1 += ["--planner-labels-csv", args.planner_labels_csv]
    if args.rof_scores_csv:
        cmd1 += ["--rof-scores-csv", args.rof_scores_csv]
    if args.bootstrap_n is not None:
        cmd1 += ["--bootstrap-n", str(args.bootstrap_n)]
    _run(cmd1)


if __name__ == "__main__":
    main()
