#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.baselines.feature_sets import feature_lineage_rows
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
    parser.add_argument("--config", default="configs/nc_v111/nc_v111_decoupling_audit.yaml")
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--extra-features", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v111_decoupling_audit")
    features: set[str] = set()
    features_path = resolve_input_path(args.features_csv or (cfg.get("inputs") or {}).get("features_csv"), cfg)
    if features_path and features_path.exists():
        features.update(pd.read_csv(features_path, nrows=5).columns.astype(str))
    for value in (args.extra_features or "").split(","):
        value = value.strip()
        if value:
            features.add(value)
    for item in (cfg.get("feature_lineage") or {}).get("expected_features", []):
        features.add(str(item))
    if not features:
        raise ValueError("no features supplied for lineage audit")
    cfg_hash = config_hash(args.config)
    out_path = out_dir / "feature_lineage_v111.csv"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"
    write_csv(out_path, add_config_hash(feature_lineage_rows(features), cfg_hash))
    write_csv(manifest_path, artifact_manifest_rows(args.config, [out_path]))
    write_json(run_path, run_manifest(args.config, cfg, [out_path, manifest_path]))
    print(f"[v111-lineage] features={len(features)} path={out_path}")


if __name__ == "__main__":
    main()
