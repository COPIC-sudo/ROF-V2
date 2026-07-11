#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.baselines.scores import compute_all_baseline_scores, current_state_kinematics_from_json
from rtbev.external.common import (
    add_config_hash,
    artifact_manifest_rows,
    config_hash,
    experiment_out_dir,
    load_yaml_config,
    resolve_input_path,
    run_manifest,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v112/nc_v112_field_baselines.yaml")
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--cohort-sample-manifest-csv", default=None)
    parser.add_argument("--skip-current-state-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v112_field_baselines")
    inputs = cfg.get("inputs", {})
    features_path = resolve_input_path(args.features_csv or inputs.get("features_csv"), cfg)
    cohort_path = resolve_input_path(args.cohort_sample_manifest_csv or inputs.get("cohort_sample_manifest_csv"), cfg)
    if features_path is None or not features_path.exists():
        raise FileNotFoundError("features CSV is required for v112 baseline score computation")
    features = pd.read_csv(features_path)
    features["sample_id"] = features["sample_id"].astype(str)
    if cohort_path and cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
        cohort["sample_id"] = cohort["sample_id"].astype(str)
        keep_cols = [
            c
            for c in cohort.columns
            if c == "sample_id" or c not in features.columns
        ]
        features = features.merge(cohort[keep_cols].drop_duplicates("sample_id"), on="sample_id", how="inner")
    else:
        cohort = pd.DataFrame()
    json_cfg = ((cfg.get("baselines") or {}).get("current_state_json") or {})
    use_json = bool(json_cfg.get("enabled", True)) and not args.skip_current_state_json and "json_gz_path" in features.columns
    if use_json:
        print(f"[v112-baselines] extracting current-state JSON features for {len(features)} fixed-cohort samples")
        kin = current_state_kinematics_from_json(
            features,
            horizon_s=float(json_cfg.get("forecast_horizon_s", 3.0)),
            dt_s=float(json_cfg.get("forecast_dt_s", 0.1)),
            lane_half_width_m=float(json_cfg.get("lane_half_width_m", 1.75)),
        )
        features = features.merge(kin, on="sample_id", how="left")
    cfg_hash = config_hash(args.config)
    scores = compute_all_baseline_scores(features)
    paths = {
        "commonroad_crime_scores": out_dir / "commonroad_crime_scores.csv",
        "rss_scores": out_dir / "rss_scores.csv",
        "drivability_baseline_scores": out_dir / "drivability_baseline_scores.csv",
        "forecast_risk_scores": out_dir / "forecast_risk_scores.csv",
    }
    for name, df in scores.items():
        write_csv(paths[name], add_config_hash(df.to_dict("records"), cfg_hash))
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"
    outputs = list(paths.values())
    write_csv(manifest_path, artifact_manifest_rows(args.config, outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*outputs, manifest_path]))
    print(f"[v112-baselines] rows={len(features)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
