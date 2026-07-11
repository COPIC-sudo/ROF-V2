#!/usr/bin/env python3
"""Verify that the public repository contains the complete v1.1 paper code chain."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REQUIRED_GROUPS = {
    "waymo_configs": [
        "configs/nc_v090/nc_v090_audit.yaml",
        "configs/nc_v096/nc_v096_endpoint_design_robustness.yaml",
        "configs/nc_v097/nc_v097_aligned_feature_robustness.yaml",
    ],
    "commonroad_configs": [
        "configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml",
        "configs/nc_v110/nc_v110b_lattice_extended_full_10k.yaml",
        "configs/nc_v111/nc_v111_decoupling_full.yaml",
        "configs/nc_v112/nc_v112_field_baselines_full_10k.yaml",
        "configs/nc_v112/nc_v112b_field_baselines_extended_label.yaml",
    ],
    "core_modules": [
        "src/rtbev/pipeline.py",
        "src/rtbev/external/taxonomy.py",
        "src/rtbev/external/metrics.py",
        "src/rtbev/baselines/scores.py",
        "src/rtbev/baselines/feature_sets.py",
    ],
    "paper_scripts": [
        "scripts/nc_v090/02_waymo_confirmatory_oof.py",
        "scripts/nc_v096/01_generate_design_variant_labels.py",
        "scripts/nc_v097/01_generate_aligned_features.py",
        "scripts/nc_v110/03_dryrun_commonroad_scaleup.py",
        "scripts/nc_v110/08_recompute_strict_fpr_metrics.py",
        "scripts/nc_v110/09_stratum_boundary_analysis.py",
        "scripts/nc_v111/02_decoupling_audit_full.py",
        "scripts/nc_v112/01_evaluate_field_baselines.py",
        "scripts/nc_v112/02_evaluate_extended_label_baselines_strict.py",
    ],
    "figure_scripts": [
        "figure_tools/make_rof_figures_2_to_6.py",
        "figure_tools/plot_v100_redesigned_figures_refined.py",
        "figure_tools/make_supplementary_figures_v100.py",
        "figure_tools/plot_nc_v11_figures_4_5_final_v2_2.py",
        "figure_tools/plot_supplementary_figures.py",
    ],
    "tests": [
        "tests/nc_v110/test_v110_fixed_taxonomy.py",
        "tests/nc_v111/test_v111_smoke.py",
        "tests/nc_v112/test_v112_smoke.py",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--report", default="paper_release_verification.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    missing: dict[str, list[str]] = {}
    for group, paths in REQUIRED_GROUPS.items():
        absent = [p for p in paths if not (root / p).is_file()]
        if absent:
            missing[group] = absent

    import_checks = {}
    for module in ["numpy", "pandas", "sklearn", "shapely", "yaml", "matplotlib"]:
        import_checks[module] = importlib.util.find_spec(module) is not None

    report = {
        "status": "passed" if not missing else "failed",
        "root": "<REPOSITORY_ROOT>",
        "version": "1.1.0",
        "missing": missing,
        "import_checks": import_checks,
        "required_group_counts": {k: len(v) for k, v in REQUIRED_GROUPS.items()},
    }
    (root / args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
