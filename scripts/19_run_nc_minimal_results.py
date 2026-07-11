from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


def _run(script: str, argv: list[str]):
    print("\n" + "=" * 80)
    print("Running", script, " ".join(argv))
    print("=" * 80)
    old = sys.argv[:]
    sys.argv = [script] + argv
    try:
        runpy.run_path(str(Path(__file__).with_name(script)), run_name="__main__")
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser(description="Run NC-minimal Waymo analysis scripts after ROF-v2 features exist.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--features-csv", default=None, help="default: <work>/features/rof_features.csv")
    ap.add_argument("--rolling-features-csv", default=None, help="optional: <work>/features/rof_features_rolling.csv")
    ap.add_argument("--skip-rolling", action="store_true")
    args = ap.parse_args()
    common = ["--config", args.config]
    if args.features_csv:
        common += ["--features-csv", args.features_csv]
    _run("14_nc_rof_v2_sanity.py", common)
    _run("15_nc_waymo_incremental.py", common)
    _run("16_nc_matched_ttc_distance.py", common + ["--task", "warning_or_above"])
    _run("16_nc_matched_ttc_distance.py", common + ["--task", "emergency_only"])
    if not args.skip_rolling and args.rolling_features_csv:
        _run("17_nc_rolling_lead_time.py", ["--config", args.config, "--features-csv", args.rolling_features_csv, "--task", "warning_or_above"])
        _run("17_nc_rolling_lead_time.py", ["--config", args.config, "--features-csv", args.rolling_features_csv, "--task", "emergency_only"])


if __name__ == "__main__":
    main()
