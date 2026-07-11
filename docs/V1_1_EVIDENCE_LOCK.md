# V1.1 Evidence Lock

Created at: 2026-07-07T16:11:01.400740+00:00

## Canonical Directories

- v110_lattice_base: <RESULTS_ROOT>/nc_v110_commonroad_scaleup/full_10k_fixed_taxonomy_lattice_base
- v110b_lattice_extended: <RESULTS_ROOT>/nc_v110_commonroad_scaleup/full_10k_lattice_extended_fixed_taxonomy
- v110_stratum_boundary: <RESULTS_ROOT>/nc_v110_commonroad_scaleup/full_10k_stratum_boundary_analysis
- v112_lattice_base: <RESULTS_ROOT>/nc_v112_field_baselines/full_10k_fixed_taxonomy_lattice_base
- v112b_lattice_extended: <RESULTS_ROOT>/nc_v112_field_baselines/full_10k_lattice_extended_fixed_taxonomy
- v111_full: <RESULTS_ROOT>/nc_v111_decoupling_audit/full
- v111_consistency_fixed: <RESULTS_ROOT>/nc_v111_decoupling_audit/full_consistency_fixed

## Locked Source Data

- derived-data archive: `ROF_results_v1_1_integrated_evidence_lock.zip` (deposited separately)
- figure/table source mapping: contained in the external evidence lock
- this code repository does not embed result tables or raw data

## Manuscript Citation Policy

- Use `external_metrics_strict_fpr.csv`, `external_bootstrap_deltas_strict_fpr.csv`, `field_baseline_metrics_strict_fpr.csv`, and `field_baseline_bootstrap_deltas_strict_fpr.csv` for primary Recall@FPR and delta statements.
- Do not cite old non-strict Recall@FPR outputs as primary manuscript results.
- v112 lattice_base `best_baseline_summary_strict_fpr.csv` in this lock is derived from `field_baseline_metrics_strict_fpr.csv` because the canonical source directory did not contain that strict summary file.
- v111 manuscript references should use `full_consistency_fixed/` for lineage/provenance-corrected tables and report.
- v1.0.1 evidence lock was not modified.

## Claim Boundary

- Current CommonRoad external validation is lattice_base expanded validation plus lattice_extended action-library sensitivity.
- Native-planner robustness is not locked here.
- Every-stratum superiority is not supported; boundary analyses must be cited for low-speed and near-overlap strata.
