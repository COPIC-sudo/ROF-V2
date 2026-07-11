#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v110/nc_v110_commonroad_scaleup.yaml")
    parser.add_argument("--sample-candidates-csv", default=None)
    parser.add_argument("--scenario-manifest-csv", default=None)
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--planner-labels-csv", default=None)
    parser.add_argument("--bootstrap-n", type=int, default=None)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    cohort_cmd = [sys.executable, "scripts/nc_v110/00_select_commonroad_cohort.py", "--config", args.config]
    if args.sample_candidates_csv:
        cohort_cmd += ["--sample-candidates-csv", args.sample_candidates_csv]
    if args.scenario_manifest_csv:
        cohort_cmd += ["--scenario-manifest-csv", args.scenario_manifest_csv]
    _run(cohort_cmd)
    metrics_cmd = [sys.executable, "scripts/nc_v110/01_external_metrics.py", "--config", args.config]
    if args.features_csv:
        metrics_cmd += ["--features-csv", args.features_csv]
    if args.planner_labels_csv:
        metrics_cmd += ["--planner-labels-csv", args.planner_labels_csv]
    if args.bootstrap_n is not None:
        metrics_cmd += ["--bootstrap-n", str(args.bootstrap_n)]
    _run(metrics_cmd)


if __name__ == "__main__":
    main()
