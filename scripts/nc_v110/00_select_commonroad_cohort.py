#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.cohort import sample_candidates_from_scenarios, select_neutral_stratified_cohort
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
    parser.add_argument("--config", default="configs/nc_v110/nc_v110_commonroad_scaleup.yaml")
    parser.add_argument("--sample-candidates-csv", default=None)
    parser.add_argument("--scenario-manifest-csv", default=None)
    parser.add_argument("--target-min", type=int, default=None)
    parser.add_argument("--target-max", type=int, default=None)
    parser.add_argument("--min-scenarios", type=int, default=None)
    parser.add_argument("--max-per-scenario", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v110_commonroad_scaleup")
    ccfg = cfg.get("cohort", {})
    inputs = cfg.get("inputs", {})
    sample_path = resolve_input_path(args.sample_candidates_csv or inputs.get("sample_candidates_csv"), cfg)
    scenario_path = resolve_input_path(args.scenario_manifest_csv or inputs.get("scenario_manifest_csv"), cfg)
    if sample_path and sample_path.exists():
        candidates = pd.read_csv(sample_path)
    elif scenario_path and scenario_path.exists():
        candidates = sample_candidates_from_scenarios(pd.read_csv(scenario_path))
    else:
        raise FileNotFoundError(
            "Provide --sample-candidates-csv or --scenario-manifest-csv, or set inputs.sample_candidates_csv / inputs.scenario_manifest_csv"
        )
    cohort, samples, diagnostics = select_neutral_stratified_cohort(
        candidates,
        target_samples_min=int(args.target_min or ccfg.get("target_samples_min", 10000)),
        target_samples_max=int(args.target_max or ccfg.get("target_samples_max", 15000)),
        min_unique_scenarios=int(args.min_scenarios or ccfg.get("min_unique_scenarios", 1000)),
        max_samples_per_scenario=int(args.max_per_scenario or ccfg.get("max_samples_per_scenario", 10)),
        seed=int(args.seed if args.seed is not None else ccfg.get("seed", 42)),
    )
    cfg_hash = config_hash(args.config)
    cohort_path = out_dir / "cohort_manifest.csv"
    sample_path_out = out_dir / "sample_manifest.csv"
    diag_path = out_dir / "cohort_selection_diagnostics.csv"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"
    write_csv(cohort_path, add_config_hash(cohort.to_dict("records"), cfg_hash))
    write_csv(sample_path_out, add_config_hash(samples.to_dict("records"), cfg_hash))
    write_csv(diag_path, add_config_hash(diagnostics, cfg_hash))
    outputs = [cohort_path, sample_path_out, diag_path]
    write_csv(manifest_path, artifact_manifest_rows(args.config, outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*outputs, manifest_path]))
    print(f"[v110-cohort] samples={len(samples)} scenarios={cohort['scenario_id'].nunique()} out_dir={out_dir}")


if __name__ == "__main__":
    main()
