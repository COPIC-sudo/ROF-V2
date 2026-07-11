# Reproducing the v1.1 paper analyses

This document separates two reproducibility modes:

1. **Figure/table regeneration from the deposited derived-data evidence lock.** This is the fastest exact route to the manuscript display items.
2. **Full raw-data rerun.** This requires third-party Waymo/CommonRoad data, local storage and substantial compute.

The canonical derived-data archive is `ROF_results_v1_1_integrated_evidence_lock.zip`.

## 1. Environment

Use Python 3.10.

```bash
conda env create -f environment.yml
conda activate rof-actionability
```

For CommonRoad stages, use the separate environment:

```bash
conda env create -f environment-commonroad.yml
conda activate rof-actionability-commonroad
```

Set the three path variables:

```bash
export ROF_WORK_DIR=/path/to/rof_work
export WAYMO_SCENARIO_ROOT=/path/to/waymo/scenario
export COMMONROAD_SCENARIO_ROOT=/path/to/commonroad/scenarios
```

All public YAML files use `${ROF_WORK_DIR}`, `${WAYMO_SCENARIO_ROOT}` and `${COMMONROAD_SCENARIO_ROOT}`. Do not commit a local absolute-path configuration.

## 2. Exact figure regeneration from derived data

Extract the evidence-lock archive to `<EVIDENCE_ROOT>`.

### Main Figures 2–3

```bash
python figure_tools/make_rof_figures_2_to_6.py \
  --input <EVIDENCE_ROOT>/99_cleanup_QA/legacy_v100_reference \
  --out figures/generated/v100 \
  --figures 2 3 \
  --formats pdf svg png \
  --dpi 600
```

### Main Figures 4–5

```bash
python figure_tools/plot_nc_v11_figures_4_5_final_v2_2.py \
  --root <EVIDENCE_ROOT> \
  --outdir figures/generated/v11 \
  --skip-supplementary
```

### Supplementary Figures S1–S5

```bash
python figure_tools/make_supplementary_figures_v100.py \
  --data <EVIDENCE_ROOT>/99_cleanup_QA/legacy_v100_reference \
  --out figures/generated/supp_v100 \
  --formats pdf png svg
```

### Supplementary Figures S6–S9

This script accepts the ZIP directly:

```bash
python figure_tools/plot_supplementary_figures.py \
  --evidence-lock ROF_results_v1_1_integrated_evidence_lock.zip \
  --output-dir figures/generated/supp_v11 \
  --dpi 600
```

Figure 1 is a conceptual information-access schematic. Its source tables are in `03_main_figure_source_data/Figure1_information_access/`; the final vector artwork is maintained in the manuscript repository.

## 3. Waymo raw-data chain

The following sequence generates the principal Waymo inputs. Paths and output names may be overridden through the public configuration template.

```bash
python scripts/01_scan_waymo.py \
  --config configs/actionability_main_example.yaml \
  --split validation \
  --out-name val_full

python scripts/03_extract_and_label.py \
  --config configs/actionability_main_example.yaml \
  --candidate-csv ${ROF_WORK_DIR}/manifests/val_full_candidates_kept.csv \
  --out-name labels_val_full.csv

python scripts/04_generate_rof_features.py \
  --config configs/actionability_main_example.yaml \
  --labels-csv ${ROF_WORK_DIR}/labels/labels_val_full.csv \
  --out-name rof_features_val_full.csv \
  --device cpu \
  --no-tensors

python scripts/24_build_actionability_labels.py \
  --config configs/actionability_main_example.yaml \
  --labels-csv ${ROF_WORK_DIR}/labels/labels_val_full.csv \
  --out-name labels_actionability_moderate_full.csv \
  --rule moderate
```

The locked Waymo evidence stages are then run in order:

```text
scripts/nc_v090  confirmatory OOF, paired deltas and information-access audit
scripts/nc_v095  secondary context and future-validity checks
scripts/nc_v096  CV fallback and endpoint-design variants
scripts/nc_v097  aligned label-feature robustness
scripts/nc_v100  Figure 3 PR-curve derivation
```

Each versioned directory exposes CLI scripts with `--config`. Start with the inventory/run-plan script in each directory and use the frozen public YAML under `configs/`.

## 4. CommonRoad full 10k chain

The final CommonRoad endpoint is generated with an outcome-blind cohort and fixed planner taxonomy.

### 4.1 Scenario inventory

```bash
python scripts/40_scan_commonroad_scenarios.py \
  --config configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml \
  --commonroad-root ${COMMONROAD_SCENARIO_ROOT} \
  --out-name commonroad_scenario_manifest.csv
```

### 4.2 Outcome-blind scenario selection

```bash
python scripts/nc_v110/05_select_commonroad_full_scenarios.py \
  --config configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml \
  --manifest-csv ${ROF_WORK_DIR}/results/commonroad_pilot_selection/commonroad_scenario_manifest.csv \
  --out-name nc_v110_full_fixed_taxonomy_scenarios.csv \
  --target-scenarios 1500 \
  --seed 42
```

### 4.3 Dynamic-ego sample export

```bash
python scripts/43b_export_commonroad_dynamic_ego_samples.py \
  --config configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml \
  --pilot-csv ${ROF_WORK_DIR}/results/commonroad_pilot_selection/nc_v110_full_fixed_taxonomy_scenarios.csv \
  --out-name nc_v110_full_fixed_taxonomy_lattice_base_pool \
  --max-samples 15000 \
  --time-stride 5 \
  --horizon-steps 30 \
  --seed 42
```

### 4.4 Lattice-base planner labels

```bash
python scripts/51_commonroad_lattice_planner_feasibility.py \
  --config configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml \
  --samples-dir <SAMPLES_JSON_GZ_DIR> \
  --manifest-csv <SAMPLE_MANIFEST_CSV> \
  --out-name nc_v110_full_fixed_taxonomy_lattice_base \
  --planner-family lattice_base \
  --sample-size 10000 \
  --seed 42 \
  --horizon-s 3.0 \
  --lane-buffer-m 4.0
```

### 4.5 ROF feature generation and fixed-taxonomy evaluation

```bash
python scripts/nc_v110/06_commonroad_rtbev_features_parallel.py \
  --config configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml \
  --samples-dir <SAMPLES_JSON_GZ_DIR> \
  --manifest-csv <SAMPLE_MANIFEST_CSV> \
  --out-csv <FEATURE_SCORE_TABLE_CSV> \
  --workers 4 \
  --device cpu \
  --resume

python scripts/nc_v110/03_dryrun_commonroad_scaleup.py \
  --config configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml \
  --sample-candidates-csv <SAMPLE_MANIFEST_CSV> \
  --planner-labels-csv <PLANNER_LABELS_CSV> \
  --features-csv <FEATURE_SCORE_TABLE_CSV> \
  --bootstrap-n 2000 \
  --seed 42
```

The fixed taxonomy is implemented in `src/rtbev/external/taxonomy.py` and covered by `tests/nc_v110/test_v110_fixed_taxonomy.py`.

### 4.6 Strict fixed-FPR and boundary analyses

```bash
python scripts/nc_v110/08_recompute_strict_fpr_metrics.py \
  --v110-bootstrap-n 2000 \
  --v112-bootstrap-n 2000 \
  --seed 42

python scripts/nc_v110/09_stratum_boundary_analysis.py \
  --config configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml \
  --bootstrap-n 1000 \
  --seed 42 \
  --min-positives 10
```

## 5. Lattice-extended action-library sensitivity

Use the same 10k cohort and the extended configuration:

```bash
python scripts/51_commonroad_lattice_planner_feasibility.py \
  --config configs/nc_v110/nc_v110b_lattice_extended_full_10k.yaml \
  --samples-dir <SAMPLES_JSON_GZ_DIR> \
  --manifest-csv <LOCKED_SAMPLE_MANIFEST_CSV> \
  --out-name nc_v110_full_10k_lattice_extended \
  --planner-family lattice_extended \
  --sample-size 10000 \
  --seed 42 \
  --horizon-s 3.0 \
  --lane-buffer-m 4.0 \
  --no-write-candidates \
  --no-store-trajectory-json

python scripts/nc_v110/07_lattice_extended_sensitivity_report.py \
  --config configs/nc_v110/nc_v110b_lattice_extended_full_10k.yaml
```

This stage tests expansion within the lattice-planner family; it is not a native-planner robustness experiment.

## 6. Field baselines

```bash
python scripts/nc_v112/00_compute_field_baselines.py \
  --config configs/nc_v112/nc_v112_field_baselines_full_10k.yaml

python scripts/nc_v112/01_evaluate_field_baselines.py \
  --config configs/nc_v112/nc_v112_field_baselines_full_10k.yaml \
  --bootstrap-n 2000

python scripts/nc_v112/02_evaluate_extended_label_baselines_strict.py \
  --config configs/nc_v112/nc_v112b_field_baselines_extended_label.yaml \
  --bootstrap-n 2000 \
  --seed 42
```

Primary baseline calculations must not read recorded future trajectories, candidate-action survival fields or planner labels.

## 7. Decoupling audit

```bash
python scripts/nc_v111/02_decoupling_audit_full.py \
  --config configs/nc_v111/nc_v111_decoupling_full.yaml \
  --bootstrap-n 1000 \
  --seed 42 \
  --n-folds 5 \
  --model rf

python scripts/nc_v111/03_consistency_fix.py \
  --config configs/nc_v111/nc_v111_decoupling_full.yaml
```

The final audit compares the base, strict non-action, strict temporal and full-actionability feature sets, and constructs the six-by-six label-feature mismatch matrix.

## 8. Verification

```bash
python -m compileall -q src scripts tests figure_tools
pytest -q
python scripts/99_smoke_test.py
python scripts/99_check_github_readiness.py --root .
python scripts/99_verify_paper_release.py --root .
```

## 9. Scope

Exact numerical reproduction requires the same third-party dataset releases, the frozen configuration files and the deposited derived-data archive. Differences in CommonRoad package versions, scenario inventories or operating-system geometry backends may affect runtime and low-level candidate diagnostics; record environment manifests for any rerun.
