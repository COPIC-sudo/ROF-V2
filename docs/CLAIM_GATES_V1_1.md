# Claim Gates V1.1

This file defines the minimum gates before using v1.1 evidence in a Nature Communications submission.

## Gate 1: CommonRoad External Validation

- Primary cohort must be outcome-blind and neutral-stratified.
- Target: 10000-15000 samples, at least 1000 unique scenarios, max 10 samples per scenario.
- Primary endpoint: `known_failure` only.
- `unknown_failure` must be reported and excluded from the primary endpoint.
- Bootstrap unit must be `scenario_id`.
- Primary metrics: AUPRC and Recall@5%FPR.
- Required outputs: `cohort_manifest.csv`, `sample_manifest.csv`, `external_metrics.csv`, `external_bootstrap_deltas.csv`, `failure_taxonomy.csv`, `stratum_metrics.csv`, `unknown_failure_sensitivity.csv`.

## Gate 2: Label-Feature Decoupling

- `strict_non_action_current_cv` must exclude candidate-action-derived features.
- Feature lineage must mark `uses_action_library`, `uses_candidate_survival`, `uses_label_horizon`, `uses_label_lane_buffer`, `reads_recorded_future`, and `reads_label`.
- Label-feature transfer matrix must include off-diagonal variants across horizon, lane buffer, and action library.
- OOF evaluation must be grouped by `scenario_id`; `segment_id` is required when present.
- Required outputs: `feature_lineage_v111.csv`, `label_feature_mismatch_matrix.csv`, `non_action_feature_oof_metrics.csv`, `grouped_oof_metrics.csv`.

## Gate 3: Stronger Field Baselines

- Primary baselines must not read recorded future.
- RSS-style results must be described as margin heuristics, not a full RSS stack.
- Full planner feasibility count is secondary diagnostic only.
- Forecast-risk baseline must not read recorded future or candidate-action survival.
- Required outputs: `commonroad_crime_scores.csv`, `rss_scores.csv`, `drivability_baseline_scores.csv`, `forecast_risk_scores.csv`, `field_baseline_metrics.csv`, `field_baseline_bootstrap_deltas.csv`, `best_baseline_summary.csv`.

## Gate 4: Reproducibility

- All scripts must expose CLI flags.
- Outputs must live under `project.work_dir`.
- Each run must write an artifact manifest with the config hash.
- Smoke tests must not require Waymo or CommonRoad raw data.
- CommonRoad package work may use `environment-commonroad.yml` if protobuf dependencies conflict with the main Waymo environment.
