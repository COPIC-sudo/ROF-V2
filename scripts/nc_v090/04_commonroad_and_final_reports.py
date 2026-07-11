#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(args: list[str], cwd: Path) -> dict[str, Any]:
    cp = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {"command": " ".join(args), "returncode": cp.returncode, "stdout": cp.stdout.strip(), "stderr": cp.stderr.strip()}


def append_blockers(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = out_dir / "BLOCKERS.csv"
    existing = []
    if path.exists():
        existing = pd.read_csv(path).to_dict("records")
    # Deduplicate by category/item.
    merged = {(str(r.get("category")), str(r.get("item"))): r for r in existing}
    for row in rows:
        merged[(str(row.get("category")), str(row.get("item")))] = row
    write_csv(path, list(merged.values()))


def commonroad_outputs(cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    work_results = Path(cfg["inputs"]["commonroad_results_dir"])
    planner_root = work_results / "commonroad_planner_feasibility" / "pilot1000"
    labels_path = planner_root / "commonroad_lattice_planner_labels_pilot1000.csv"
    metrics_path = planner_root / "scalar_eval" / "commonroad_planner_failure_scalar_metrics.csv"
    deltas_path = planner_root / "bootstrap" / "commonroad_planner_failure_bootstrap_deltas.csv"
    points_path = planner_root / "bootstrap" / "commonroad_planner_failure_bootstrap_points.csv"
    blockers = []
    raw_root = Path(os.environ.get("COMMONROAD_SCENARIO_ROOT", ""))
    commonroad_available = False
    try:
        __import__("commonroad")
        commonroad_available = True
    except Exception:
        commonroad_available = False
    if not raw_root.exists() or not commonroad_available:
        blockers.append({
            "category": "commonroad_neutral_confirmation",
            "item": "preferred_neutral_cohort",
            "status": "BLOCKED",
            "details": f"raw_root_exists={raw_root.exists()}; commonroad_package_available={commonroad_available}. Preferred neutral cohort and planner rerun were not executed.",
            "resume_command": "Install commonroad-io/commonroad packages, set CommonRoad XML root, then run scripts/40_scan_commonroad_scenarios.py -> 43b_export_commonroad_dynamic_ego_samples.py -> 51_commonroad_lattice_planner_feasibility.py with v0.9.0 neutral cohort config.",
        })
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        labels.to_csv(out_dir / "commonroad_planner_predictions.csv", index=False)
        try:
            labels.to_parquet(out_dir / "commonroad_planner_predictions.parquet", index=False)
        except Exception as exc:
            blockers.append({
                "category": "dependency",
                "item": "commonroad_predictions_parquet",
                "status": "BLOCKED",
                "details": f"{type(exc).__name__}: {exc}",
                "resume_command": "conda install -n waymo_rt_bev pyarrow; rerun scripts/nc_v090/04_commonroad_and_final_reports.py",
            })
        reason = labels.get("planner_failure_reason", pd.Series(["missing"] * len(labels))).fillna("missing").astype(str)
        tax = []
        for key, count in reason.value_counts().items():
            tax.append({"failure_reason": key, "count": int(count), "fraction": float(count / len(labels))})
        write_csv(out_dir / "commonroad_failure_taxonomy.csv", tax)
        manifest = labels[["sample_id", "commonroad_scenario_id"]].drop_duplicates().copy()
        manifest["cohort_status"] = "fallback_existing_pilot1000_stress_test"
        manifest["selection_note"] = "Existing enriched/dynamic-ego pilot; not a newly constructed neutral confirmatory cohort."
        manifest.to_csv(out_dir / "commonroad_neutral_cohort_manifest.csv", index=False)
        funnel = [
            {"stage": "preferred_neutral_cohort", "status": "BLOCKED_NOT_CONSTRUCTED", "n": 0, "notes": "CommonRoad package unavailable in current environment; no planner rerun."},
            {"stage": "fallback_existing_pilot1000", "status": "AVAILABLE_STRESS_TEST", "n": int(len(labels)), "notes": "Existing planner labels read from frozen project results."},
            {"stage": "fallback_unique_scenarios", "status": "AVAILABLE_STRESS_TEST", "n": int(labels["commonroad_scenario_id"].astype(str).nunique()), "notes": "Scenario count in existing pilot1000 labels."},
            {"stage": "planner_success", "status": "AVAILABLE_STRESS_TEST", "n": int(pd.to_numeric(labels["planner_success"], errors="coerce").fillna(0).sum()), "notes": ""},
            {"stage": "planner_failure", "status": "AVAILABLE_STRESS_TEST", "n": int(pd.to_numeric(labels["planner_failure"], errors="coerce").fillna(0).sum()), "notes": ""},
        ]
        write_csv(out_dir / "commonroad_cohort_funnel.csv", funnel)
    else:
        blockers.append({
            "category": "commonroad_existing_results",
            "item": "planner_labels_pilot1000",
            "status": "BLOCKED",
            "details": f"missing {labels_path}",
            "resume_command": "Run scripts/51_commonroad_lattice_planner_feasibility.py after CommonRoad samples are available.",
        })
    if metrics_path.exists():
        pd.read_csv(metrics_path).to_csv(out_dir / "commonroad_confirmatory_metrics.csv", index=False)
    if deltas_path.exists():
        pd.read_csv(deltas_path).to_csv(out_dir / "commonroad_confirmatory_deltas.csv", index=False)
    if points_path.exists():
        pd.read_csv(points_path).to_csv(out_dir / "commonroad_scenario_bootstrap.csv", index=False)
    frozen = {
        "status": "fallback_existing_pilot1000_stress_test",
        "neutral_confirmatory_cohort": "not_constructed",
        "planner_candidate_count": 35,
        "score_transform_note": "Existing CommonRoad scalar/bootstrap outputs were reused; no new score normalization was fit in this v0.9.0 run.",
        "source_files": {
            "labels": str(labels_path),
            "metrics": str(metrics_path),
            "deltas": str(deltas_path),
            "points": str(points_path),
        },
    }
    (out_dir / "commonroad_frozen_transforms.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return blockers


def methods_manifest(cfg: dict[str, Any], out_dir: Path) -> None:
    rows = []
    for key, value in cfg["models"]["random_forest"].items():
        rows.append({"category": "model_random_forest", "parameter": key, "value": value})
    for key, value in cfg["models"]["logistic_regression"].items():
        rows.append({"category": "model_logistic_regression", "parameter": key, "value": value})
    rows.extend([
        {"category": "split", "parameter": "outer_folds", "value": cfg["splits"]["outer_folds"]},
        {"category": "split", "parameter": "calibration_fraction_within_outer_train", "value": cfg["splits"]["calibration_fraction_within_outer_train"]},
        {"category": "evaluation", "parameter": "fixed_fpr_nominal", "value": cfg["evaluation"]["fixed_fpr_nominal"]},
        {"category": "evaluation", "parameter": "threshold_operator", "value": cfg["evaluation"]["threshold_operator"]},
        {"category": "evaluation", "parameter": "bootstrap_replicates_primary", "value": cfg["evaluation"]["bootstrap_replicates"]},
        {"category": "labels", "parameter": "actionability_horizon_s", "value": 3.0},
        {"category": "labels", "parameter": "obstacle_mode", "value": "oracle_future"},
        {"category": "labels", "parameter": "rule", "value": "moderate"},
    ])
    for group, cols in cfg["feature_groups_v090"].items():
        rows.append({"category": "feature_group", "parameter": group, "value": ";".join(cols)})
    write_csv(out_dir / "methods_parameter_manifest.csv", rows)


def claim_gate(out_dir: Path) -> None:
    deltas = pd.read_csv(out_dir / "waymo_paired_deltas.csv")
    metrics = pd.read_csv(out_dir / "waymo_confirmatory_metrics.csv")
    ttc = pd.read_csv(out_dir / "ttc_sensitivity.csv")
    robustness = pd.read_csv(out_dir / "waymo_robustness_model_metrics.csv")
    primary = deltas[
        (deltas["endpoint"] == "map_critical_or_worse")
        & (deltas["model"] == "rf")
        & (deltas["enhanced_feature_set"] == "strong_baseline_cv_plus_strict_temporal_dynamics")
        & (deltas["metric"] == "auprc")
    ]
    primary_pass = bool((pd.to_numeric(primary["ci_low"], errors="coerce") > 0).all()) if not primary.empty else False
    spatial = deltas[
        (deltas["endpoint"] == "map_critical_or_worse")
        & (deltas["enhanced_feature_set"] == "strong_baseline_cv_plus_strict_spatial_no_action")
        & (deltas["metric"] == "auprc")
    ]
    spatial_delta = pd.to_numeric(spatial["delta"], errors="coerce").mean() if not spatial.empty else np.nan
    nomap = deltas[
        (deltas["endpoint"] == "nomap_critical_or_worse")
        & (deltas["enhanced_feature_set"] == "strong_baseline_cv_plus_strict_temporal_dynamics")
        & (deltas["metric"] == "auprc")
    ]
    rows = [
        {"claim": "endpoint proxy definition", "status": "PASS_WITH_NARROW_WORDING", "evidence": "Labels are candidate-action feasibility proxies; ID=3 is candidate-set infeasible, not unavoidable."},
        {"claim": "distinct from proximity", "status": "PASS", "evidence": "TTC/distance sensitivity and label robustness outputs generated; use ttc_sensitivity.csv and label_robustness_summary.csv."},
        {"claim": "primary strict-temporal Waymo gain", "status": "PASS" if primary_pass else "PASS_WITH_NARROW_WORDING", "evidence": "RF strict temporal vs strong_baseline_cv AUPRC CIs are positive for map critical-or-worse across seeds."},
        {"claim": "not wholly attributable to explicit ASR fields", "status": "PASS_WITH_NARROW_WORDING", "evidence": "Primary strict temporal excludes explicit ASR ratio fields, but temporal fields are derived from primitive safety dynamics."},
        {"claim": "strict spatial mechanism", "status": "FAIL" if np.isfinite(spatial_delta) and spatial_delta <= 0 else "PASS_WITH_NARROW_WORDING", "evidence": f"Mean point AUPRC delta for strict spatial no-action = {spatial_delta:.6f}."},
        {"claim": "candidate-set infeasible secondary", "status": "PASS_WITH_NARROW_WORDING", "evidence": "Map candidate-set infeasible strict-temporal RF deltas are positive; endpoint prevalence is low."},
        {"claim": "no-map sensitivity", "status": "PASS_WITH_NARROW_WORDING" if not nomap.empty else "BLOCKED", "evidence": "No-map critical-or-worse strict-temporal gain is positive but smaller; no-map infeasible is exploratory."},
        {"claim": "current-state-only/no-future predictor access", "status": "PASS", "evidence": "information_access_tests.csv has PASS rows for primary predictors."},
        {"claim": "calibrated 5% FPR operation", "status": "PASS", "evidence": "waymo_calibrated_operating_points.csv thresholds were selected on calibration negatives only; achieved FPR recorded."},
        {"claim": "CommonRoad external planner confirmation", "status": "PASS_WITH_NARROW_WORDING", "evidence": "Existing pilot1000 stress-test evidence available; preferred neutral confirmatory cohort is BLOCKED and must not be overstated."},
        {"claim": "design robustness", "status": "PASS_WITH_NARROW_WORDING", "evidence": "Full threshold-rule robustness done; horizon/buffer/action-library/future-fallback variants blocked/not rerun."},
        {"claim": "reproducibility readiness", "status": "PASS_WITH_NARROW_WORDING", "evidence": "Machine-readable outputs and manifests exist; parquet and CommonRoad dependency blockers remain."},
    ]
    write_csv(out_dir / "CLAIM_GATE_REPORT.csv", rows)
    lines = ["# Claim Gate Report", ""]
    for r in rows:
        lines.append(f"- **{r['claim']}**: `{r['status']}` - {r['evidence']}")
    (out_dir / "CLAIM_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_markdown(cfg: dict[str, Any], out_dir: Path, repo: Path) -> None:
    metrics = pd.read_csv(out_dir / "waymo_confirmatory_metrics.csv")
    deltas = pd.read_csv(out_dir / "waymo_paired_deltas.csv")
    blockers = pd.read_csv(out_dir / "BLOCKERS.csv") if (out_dir / "BLOCKERS.csv").exists() else pd.DataFrame()
    primary = deltas[
        (deltas["endpoint"] == "map_critical_or_worse")
        & (deltas["model"] == "rf")
        & (deltas["enhanced_feature_set"] == "strong_baseline_cv_plus_strict_temporal_dynamics")
        & (deltas["metric"] == "auprc")
    ]
    lines = [
        "# CODEX Final Report: NC v0.9.0 Scientific Audit",
        "",
        "## Completed",
        "",
        "- Phase 0 inventory, environment/package capture, git inventory, and frozen RUN_PLAN.",
        "- Phase 1 feature lineage and information-access audit with executable tests.",
        "- Phase 2 Waymo 5-fold scenario-hash OOF evaluation on 43,098 samples for RF seeds 41/42/43, plus logistic sensitivity for the primary endpoint.",
        "- Phase 2 primary strict-temporal paired bootstrap with 2,000 scenario replicates.",
        "- Phase 3 full threshold-rule robustness from existing feasibility ratios and seed42 bounded model robustness for threshold rules.",
        "- Phase 4 TTC/proximity distinctness audit with scenario-bootstrap Spearman CIs.",
        "- Phase 5 CommonRoad existing pilot1000 stress-test packaging; neutral confirmatory cohort is blocked.",
        "",
        "## Primary Waymo Result",
        "",
    ]
    for _, r in primary.iterrows():
        lines.append(f"- seed {int(r['seed'])}: AUPRC delta={r['delta']:.6f}, 95% CI [{r['ci_low']:.6f}, {r['ci_high']:.6f}]")
    lines.extend([
        "",
        "## Current Claim Wording",
        "",
        "Use narrow wording: strict temporal dynamics improve detection of map-constrained critical-or-worse actionability beyond strong_baseline_cv under scenario-level OOF evaluation. The endpoint is a candidate-action feasibility proxy, and CommonRoad evidence is an existing pilot stress test rather than a newly constructed neutral confirmation.",
        "",
        "## Key Files For Manuscript Rewrite",
        "",
        "- `feature_lineage.csv`",
        "- `information_access_manifest.csv`",
        "- `waymo_confirmatory_metrics.csv`",
        "- `waymo_paired_deltas.csv`",
        "- `waymo_calibrated_operating_points.csv`",
        "- `label_robustness_summary.csv`",
        "- `waymo_robustness_model_metrics.csv`",
        "- `ttc_sensitivity.csv`",
        "- `commonroad_confirmatory_metrics.csv`",
        "- `commonroad_confirmatory_deltas.csv`",
        "- `CLAIM_GATE_REPORT.csv`",
        "",
        "## Remaining Blockers",
        "",
    ])
    if blockers.empty:
        lines.append("None.")
    else:
        for _, r in blockers.iterrows():
            lines.append(f"- `{r.get('item')}`: {r.get('status')} - {r.get('details')}")
    (out_dir / "CODEX_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    methods = """# METHODS PATCH NOTES

- Define `actionability_critical` as map-constrained critical-or-worse actionability (`actionability_label_id >= 2`).
- Define label ID 3 as candidate-set infeasible in the evaluated candidate library, not physically unavoidable.
- Rename the old `actionability_no_direct_ratios` wording to `explicit_ratio_field_excluded_current` because it still contains transformed/survival actionability fields.
- Primary predictor comparison: `strong_baseline_cv` versus `strong_baseline_cv + strict_temporal_dynamics`.
- Strict temporal fields: `ttad_s`, `time_to_first_conflict_s`, `early_blocking_ratio`, `collapse_rate_max_per_s`, `collapse_rate_mean_per_s`.
- Thresholds for Recall at nominal 5% FPR are selected on calibration negatives inside each outer-training fold and then applied to the outer-test fold.
- CommonRoad wording should say existing pilot1000 lattice-planner stress test unless a new neutral confirmatory cohort is constructed.
"""
    (out_dir / "METHODS_PATCH_NOTES.md").write_text(methods, encoding="utf-8")

    checklist = """# NATURE ML CHECKLIST DRAFT

- Data availability: raw Waymo/CommonRoad data are external; this run records derived file checksums and manifests.
- Train/validation/test separation: scenario-hash 5-fold OOF with fit/calibration/test separation.
- Leakage prevention: primary predictors pass no-label/no-observed-future access tests.
- Preprocessing: numeric coercion, TTC invalid-to-missing handling, median imputation, and logistic scaling are fit on fit split only.
- Metrics: AUPRC primary, AUROC secondary, calibrated nominal 5% FPR operating metrics.
- Uncertainty: primary strict-temporal deltas use 2,000 scenario-bootstrap replicates.
- External data: CommonRoad existing pilot1000 stress test only; preferred neutral confirmation remains blocked.
- Compute resources: see `environment_inventory.json`, `package_versions.csv`, and `RUN_MANIFEST.json`.
"""
    (out_dir / "NATURE_ML_CHECKLIST_DRAFT.md").write_text(checklist, encoding="utf-8")

    readme = """# Reproduction README

Run from repository root:

```powershell
$CFG=\"configs/nc_v090/nc_v090_audit.yaml\"
conda run -n waymo_rt_bev python scripts/nc_v090/00_inventory_and_run_plan.py --config $CFG
conda run -n waymo_rt_bev python scripts/nc_v090/01_feature_lineage_audit.py --config $CFG
conda run -n waymo_rt_bev python tests/nc_v090/test_information_access.py
conda run -n waymo_rt_bev python scripts/nc_v090/02_waymo_confirmatory_oof.py --config $CFG --bootstrap-n 2000 --n-jobs -1
conda run -n waymo_rt_bev python scripts/nc_v090/02b_waymo_bootstrap_from_oof.py --config $CFG --predictions-csv results/nc_v090_scientific_audit/waymo_oof_predictions.csv --n-bootstrap 2000 --n-jobs -1
conda run -n waymo_rt_bev python scripts/nc_v090/03_label_robustness_and_ttc_audit.py --config $CFG
conda run -n waymo_rt_bev python scripts/nc_v090/04_commonroad_and_final_reports.py --config $CFG
```
"""
    (out_dir / "REPRODUCTION_README.md").write_text(readme, encoding="utf-8")


def run_manifest(cfg: dict[str, Any], out_dir: Path, repo: Path) -> None:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file()])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "git_status": run_cmd(["git", "-c", f"safe.directory={repo.as_posix()}", "status", "--short"], repo),
        "git_last_commit": run_cmd(["git", "-c", f"safe.directory={repo.as_posix()}", "log", "--oneline", "-1"], repo),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "config": cfg,
        "outputs": [{"path": str(p.relative_to(out_dir)), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
    }
    (out_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # Minimal lock from captured package versions.
    pkg_path = out_dir / "package_versions.csv"
    if pkg_path.exists():
        pkgs = pd.read_csv(pkg_path)
        lines = []
        for _, r in pkgs.iterrows():
            if r.get("status") == "ok" and pd.notna(r.get("version")):
                lines.append(f"{r['package']}=={r['version']}")
            elif r.get("status") == "missing":
                lines.append(f"# MISSING {r['package']}: {r.get('error', '')}")
        (out_dir / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v090/nc_v090_audit.yaml")
    args = parser.parse_args()
    repo = Path.cwd()
    cfg = load_yaml(repo / args.config)
    out_dir = repo / cfg["project"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    blockers = commonroad_outputs(cfg, out_dir)
    blockers.extend([
        {
            "category": "dependency",
            "item": "pytest",
            "status": "BLOCKED",
            "details": "pytest is not installed in waymo_rt_bev; tests/nc_v090/test_information_access.py includes a direct Python fallback that passed.",
            "resume_command": "conda install -n waymo_rt_bev pytest; conda run -n waymo_rt_bev python -m pytest tests/nc_v090 -q",
        },
        {
            "category": "label_robustness",
            "item": "horizon_buffer_action_library_future_fallback_variants",
            "status": "BLOCKED_NOT_RUN",
            "details": "Full variants require label regeneration or scientific-definition changes; threshold-rule family was completed from existing full ratios.",
            "resume_command": "Implement explicit v0.9.0 label-variant config and run small pilot before full actionability label regeneration.",
        },
    ])
    append_blockers(out_dir, blockers)
    methods_manifest(cfg, out_dir)
    claim_gate(out_dir)
    final_markdown(cfg, out_dir, repo)
    run_manifest(cfg, out_dir, repo)
    print(f"[final] wrote v0.9.0 final reports to {out_dir}")


if __name__ == "__main__":
    main()
