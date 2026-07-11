#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.baselines.feature_sets import feature_lineage_rows, lineage_flags
from rtbev.external.common import artifact_manifest_rows, config_hash, load_yaml_config, sha256_file, write_csv, write_json


STRICT_ROW = "strong_baseline_cv_plus_strict_non_action_current_cv"
CURRENT_STATE_OVERLAP_FEATURES = {
    "current_collision",
    "max_overlap_count",
    "mean_overlap_count_nonzero",
    "overlap_count_entropy_norm",
}
UNCHANGED_REFERENCE_ARTIFACTS = [
    "non_action_feature_oof_metrics.csv",
    "non_action_feature_bootstrap_deltas.csv",
    "label_feature_mismatch_matrix.csv",
    "label_feature_mismatch_pairs.csv",
    "label_feature_mismatch_bootstrap.csv",
    "grouped_oof_metrics.csv",
    "leave_family_out_metrics.csv",
    "external_label_decoupling_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix v111 lineage/provenance consistency without rerunning OOF.")
    parser.add_argument("--config", default="configs/nc_v111/nc_v111_decoupling_full.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--script-path", default="scripts/nc_v111/02_decoupling_audit_full.py")
    parser.add_argument("--no-copy-unchanged", action="store_true")
    return parser.parse_args()


def _default_input_dir() -> Path:
    rel = Path("results/nc_v111_decoupling_audit/full")
    if rel.exists():
        return rel
    work_dir = os.environ.get("ROF_WORK_DIR")
    if work_dir:
        return Path(work_dir) / "results" / "nc_v111_decoupling_audit" / "full"
    raise FileNotFoundError("input dir not provided and neither local results nor ROF_WORK_DIR is available")


def _read_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _strict_features(non_action: pd.DataFrame) -> list[str]:
    sub = non_action[non_action["feature_set"].astype(str) == STRICT_ROW]
    if sub.empty:
        raise ValueError(f"missing feature_set row: {STRICT_ROW}")
    raw = str(sub.iloc[0].get("features", ""))
    features = [x.strip() for x in raw.split(";") if x.strip()]
    expected = int(float(sub.iloc[0].get("n_features", len(features))))
    if len(features) != expected:
        raise ValueError(f"strict feature list length {len(features)} != n_features {expected}")
    return features


def _lineage_by_feature(lineage: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in lineage.to_dict("records"):
        feature = str(row.get("feature_name") or row.get("feature") or "")
        if feature:
            out[feature] = row
    return out


def _correct_lineage(lineage: pd.DataFrame, actual_features: list[str], cfg_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    actual_set = set(actual_features)
    by_feature = _lineage_by_feature(lineage)
    missing = [feature for feature in actual_features if feature not in by_feature]
    for row in feature_lineage_rows(missing):
        by_feature[str(row["feature_name"])] = row

    original_allowed = {
        feature
        for feature, row in by_feature.items()
        if _bool_value(row.get("allowed_in_strict_non_action")) or _bool_value(row.get("allowed_in_strict_non_action_current_cv"))
    }
    unsafe_actual: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    for feature in actual_features:
        flags = lineage_flags(feature)
        safe = not any(flags.values())
        if feature in CURRENT_STATE_OVERLAP_FEATURES:
            classification = "current_state_overlap_or_collision_scalar"
        elif feature.startswith("cv_"):
            classification = "current_cv_occupancy_or_conflict_scalar"
        else:
            classification = "current_state_kinematic_or_proximity_scalar"
        if not safe:
            unsafe_actual.append(feature)
        audit_rows.append(
            {
                "row_type": "actual_strict_feature",
                "feature_name": feature,
                "classification": classification,
                "in_original_lineage": feature in _lineage_by_feature(lineage),
                "original_allowed_in_strict_non_action": feature in original_allowed,
                "corrected_allowed_in_strict_non_action": safe,
                "reads_recorded_future": flags["reads_recorded_future"],
                "reads_label": flags["reads_label"],
                "uses_action_library": flags["uses_action_library"],
                "uses_candidate_survival": flags["uses_candidate_survival"],
                "uses_label_horizon": flags["uses_label_horizon"],
                "uses_label_lane_buffer": flags["uses_label_lane_buffer"],
                "uses_endpoint_intermediate": flags["uses_endpoint_intermediate"],
                "action": "mark_allowed" if safe else "requires_feature_set_removal_and_oof_recompute",
                "rationale": "feature is current-state/non-action and has no label/future/action-survival flags" if safe else "lineage flags indicate forbidden information access",
            }
        )

    if unsafe_actual:
        raise RuntimeError(
            "unsafe strict features found; stop instead of silently rewriting lineage: " + ", ".join(sorted(unsafe_actual))
        )

    all_features = sorted(by_feature)
    corrected_rows: list[dict[str, Any]] = []
    for feature in all_features:
        row = dict(by_feature[feature])
        flags = lineage_flags(feature)
        allowed = feature in actual_set
        row["feature_name"] = feature
        row["feature"] = row.get("feature") or feature
        row["feature_set"] = "strict_non_action_current_cv" if allowed else "excluded_or_diagnostic"
        row["allowed_in_strict_non_action"] = bool(allowed)
        row["allowed_in_strict_non_action_current_cv"] = bool(allowed)
        for key, value in flags.items():
            row[key] = bool(value)
        row["config_hash"] = cfg_hash
        corrected_rows.append(row)

    corrected_allowed = {row["feature_name"] for row in corrected_rows if _bool_value(row.get("allowed_in_strict_non_action"))}
    audit_rows.append(
        {
            "row_type": "summary",
            "feature_name": "",
            "classification": "",
            "in_original_lineage": "",
            "original_allowed_in_strict_non_action": len(original_allowed),
            "corrected_allowed_in_strict_non_action": len(corrected_allowed),
            "reads_recorded_future": "",
            "reads_label": "",
            "uses_action_library": "",
            "uses_candidate_survival": "",
            "uses_label_horizon": "",
            "uses_label_lane_buffer": "",
            "uses_endpoint_intermediate": "",
            "action": "lineage_allowed_set_matches_actual_strict_feature_set",
            "rationale": f"actual={len(actual_set)} corrected_allowed={len(corrected_allowed)} mismatch={sorted(actual_set ^ corrected_allowed)}",
        }
    )
    summary = {
        "actual_count": len(actual_set),
        "original_allowed_count": len(original_allowed),
        "corrected_allowed_count": len(corrected_allowed),
        "missing_from_original_lineage": missing,
        "not_allowed_but_used_original": sorted(actual_set - original_allowed),
        "allowed_not_used_original": sorted(original_allowed - actual_set),
        "corrected_matches_actual": corrected_allowed == actual_set,
    }
    return corrected_rows, audit_rows, summary


def _model_values(df: pd.DataFrame) -> list[str]:
    if "model" not in df.columns:
        return []
    return sorted(str(x) for x in df["model"].dropna().astype(str).unique())


def _model_provenance(
    input_dir: Path,
    config_path: Path,
    script_path: Path,
    non_action: pd.DataFrame,
    deltas: pd.DataFrame,
    grouped: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = load_yaml_config(config_path)
    cfg_model = str((cfg.get("evaluation") or {}).get("model", ""))
    manifest_path = input_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest_cfg_hash = str(manifest.get("config_hash", ""))
    current_cfg_hash = config_hash(config_path)
    script_text = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
    script_model_logic = "--model" in script_text and "eval_cfg.get(\"model\", \"rf\")" in script_text
    observed = {
        "non_action_feature_oof_metrics.csv": _model_values(non_action),
        "non_action_feature_bootstrap_deltas.csv": _model_values(deltas),
        "grouped_oof_metrics.csv": _model_values(grouped),
    }
    observed_models = sorted({model for models in observed.values() for model in models})
    final_model = observed_models[0] if len(observed_models) == 1 else ""
    provenance_inconsistent = not final_model or (cfg_model and final_model != cfg_model) or (manifest_cfg_hash and manifest_cfg_hash != current_cfg_hash)
    status = "OK_FINAL_ARTIFACTS_USE_RF" if final_model == "rf" and not provenance_inconsistent else "INCONSISTENT_OR_BLOCKED"
    rows: list[dict[str, Any]] = []
    for source, models in observed.items():
        rows.append(
            {
                "evidence_source": source,
                "observed_model": ";".join(models) if models else "",
                "expected_model": cfg_model,
                "status": "matches_config" if len(models) == 1 and models[0] == cfg_model else "does_not_match_config",
                "detail": "model column in final artifact",
            }
        )
    rows.extend(
        [
            {
                "evidence_source": str(config_path),
                "observed_model": cfg_model,
                "expected_model": cfg_model,
                "status": "config_model",
                "detail": "evaluation.model from config",
            },
            {
                "evidence_source": str(manifest_path),
                "observed_model": final_model,
                "expected_model": cfg_model,
                "status": "config_hash_matches" if manifest_cfg_hash == current_cfg_hash else "config_hash_mismatch",
                "detail": f"run_manifest config_hash={manifest_cfg_hash}; current config_hash={current_cfg_hash}",
            },
            {
                "evidence_source": str(script_path),
                "observed_model": final_model,
                "expected_model": cfg_model,
                "status": "script_uses_cli_or_config_model" if script_model_logic else "script_model_logic_not_detected",
                "detail": "script resolves model as args.model or evaluation.model; no final artifact evidence supports logreg",
            },
            {
                "evidence_source": "final_decision",
                "observed_model": final_model,
                "expected_model": cfg_model,
                "status": status,
                "detail": "final artifacts, config, and run manifest support RF; external log text is not used to relabel artifacts",
            },
        ]
    )
    summary = {
        "config_model": cfg_model,
        "observed_models": observed_models,
        "final_model": final_model,
        "provenance_inconsistent": provenance_inconsistent,
        "manifest_config_hash_matches": manifest_cfg_hash == current_cfg_hash,
        "script_model_logic_detected": script_model_logic,
    }
    return rows, summary


def _report(
    input_dir: Path,
    output_dir: Path,
    lineage_summary: dict[str, Any],
    model_summary: dict[str, Any],
    non_action: pd.DataFrame,
) -> str:
    strict_row = non_action[non_action["feature_set"].astype(str) == STRICT_ROW].iloc[0]
    metrics_changed = False
    lines = [
        "# v111 Decoupling Audit Consistency Fixed Report",
        "",
        "## Scope",
        "",
        f"- input_dir: {input_dir}",
        f"- output_dir: {output_dir}",
        "- heavy OOF rerun: not performed",
        "- planner rerun / cohort resampling: not performed",
        "- v110/v112/v112b raw outputs: not modified",
        "",
        "## Feature Lineage Consistency",
        "",
        f"- strict_non_action_current_cv actual feature count: {lineage_summary['actual_count']}",
        f"- original allowed_in_strict_non_action=True feature count: {lineage_summary['original_allowed_count']}",
        f"- corrected allowed_in_strict_non_action=True feature count: {lineage_summary['corrected_allowed_count']}",
        f"- corrected allowed set exactly matches actual strict feature set: {lineage_summary['corrected_matches_actual']}",
        f"- features reclassified as allowed: {', '.join(lineage_summary['not_allowed_but_used_original']) or 'none'}",
        "- reclassification rationale: the four overlap/collision scalars are current-state non-action features and have no label, future, action-library, candidate-survival, horizon, lane-buffer, or endpoint-intermediate flags.",
        "",
        "## Model Provenance",
        "",
        f"- v111 non-action audit final model: {model_summary['final_model']}",
        f"- config model: {model_summary['config_model']}",
        f"- observed artifact models: {', '.join(model_summary['observed_models'])}",
        f"- provenance inconsistency: {model_summary['provenance_inconsistent']}",
        f"- run manifest config hash matches current config: {model_summary['manifest_config_hash_matches']}",
        "- action: kept CSV model=rf and corrected only lineage/provenance reporting.",
        "",
        "## Metric Impact",
        "",
        f"- AUPRC / strict Recall@5%FPR values changed: {metrics_changed}",
        f"- unchanged strict feature row AUPRC: {float(strict_row['AUPRC']):.6g}",
        f"- unchanged strict feature row strict Recall@5%FPR: {float(strict_row['Recall@5%FPR_strict']):.6g}",
        "- reason: the actual feature set and final RF predictions were unchanged; only the lineage table and provenance audit were corrected.",
        "",
        "## Paper Citation Recommendation",
        "",
        "- recommended citation directory: full_consistency_fixed/",
        "- use this directory for v111 lineage/provenance/report references; copied metric CSVs are byte-identical references to the original full/ metrics except for the corrected lineage/report/audit files.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else _default_input_dir()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_dir.parent / "full_consistency_fixed"
    config_path = Path(args.config)
    script_path = Path(args.script_path)
    cfg_hash = config_hash(config_path)

    non_action = _read_required(input_dir / "non_action_feature_oof_metrics.csv", "non-action OOF metrics")
    deltas = _read_required(input_dir / "non_action_feature_bootstrap_deltas.csv", "non-action bootstrap deltas")
    grouped = _read_required(input_dir / "grouped_oof_metrics.csv", "grouped OOF metrics")
    lineage = _read_required(input_dir / "feature_lineage_v111.csv", "feature lineage")
    actual_features = _strict_features(non_action)
    corrected_lineage, lineage_audit, lineage_summary = _correct_lineage(lineage, actual_features, cfg_hash)
    model_rows, model_summary = _model_provenance(input_dir, config_path, script_path, non_action, deltas, grouped)

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []
    if not args.no_copy_unchanged:
        for artifact in UNCHANGED_REFERENCE_ARTIFACTS:
            src = input_dir / artifact
            if src.exists():
                dst = output_dir / artifact
                shutil.copyfile(src, dst)
                copied_paths.append(dst)

    lineage_path = output_dir / "feature_lineage_v111.csv"
    lineage_audit_path = output_dir / "feature_lineage_consistency_audit.csv"
    model_audit_path = output_dir / "model_provenance_audit.csv"
    report_path = output_dir / "v111_decoupling_report_consistency_fixed.md"
    manifest_path = output_dir / "artifact_manifest.csv"
    run_path = output_dir / "run_manifest.json"

    write_csv(lineage_path, corrected_lineage)
    write_csv(lineage_audit_path, lineage_audit)
    write_csv(model_audit_path, model_rows)
    report_path.write_text(_report(input_dir, output_dir, lineage_summary, model_summary, non_action), encoding="utf-8")

    outputs = [lineage_path, lineage_audit_path, model_audit_path, report_path, *copied_paths]
    write_csv(manifest_path, artifact_manifest_rows(config_path, outputs))
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "config_path": str(config_path),
        "config_hash": cfg_hash,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": "consistency_fix_no_heavy_oof_rerun",
        "strict_feature_count": lineage_summary["actual_count"],
        "corrected_allowed_count": lineage_summary["corrected_allowed_count"],
        "final_model": model_summary["final_model"],
        "provenance_inconsistent": model_summary["provenance_inconsistent"],
        "input_run_manifest_sha256": sha256_file(input_dir / "run_manifest.json") if (input_dir / "run_manifest.json").exists() else "",
        "outputs": artifact_manifest_rows(config_path, [*outputs, manifest_path]),
    }
    write_json(run_path, run)
    print(
        "[v111-consistency-fix] "
        f"out_dir={output_dir} "
        f"strict_features={lineage_summary['actual_count']} "
        f"corrected_allowed={lineage_summary['corrected_allowed_count']} "
        f"final_model={model_summary['final_model']} "
        f"metrics_changed=False"
    )


if __name__ == "__main__":
    main()
