#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PRIMARY_GROUPS = {"strong_baseline_cv", "strict_temporal_dynamics"}
LABEL_LIKE_RE = re.compile(r"(label|actionability_label|planner_failure|future_valid|future_xy|future_heading)", re.I)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def literal_list_assignments(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        continue
                    if isinstance(value, list) and all(isinstance(x, str) for x in value):
                        out[target.id] = list(value)
    return out


def line_ref(path: Path, pattern: str) -> str:
    if not path.exists():
        return f"{path.as_posix()}:MISSING"
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if pattern in line:
            return f"{path.as_posix()}:{i}"
    return f"{path.as_posix()}:not_found:{pattern}"


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


def csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def read_ids(path: Path) -> pd.DataFrame:
    usecols = ["sample_id"]
    header = csv_header(path)
    if "scenario_id" in header:
        usecols.append("scenario_id")
    df = pd.read_csv(path, usecols=usecols)
    df["sample_id"] = df["sample_id"].astype(str)
    if "scenario_id" in df.columns:
        df["scenario_id"] = df["scenario_id"].astype(str)
    return df


def membership(feature: str, groups: dict[str, list[str]]) -> tuple[str, str]:
    current = [name for name, cols in groups.items() if feature in cols]
    v090 = []
    for name in [
        "strong_baseline_cv",
        "strict_temporal_dynamics",
        "strict_spatial_no_action",
        "explicit_ratio_field_excluded_current",
        "direct_action_ratios_only",
    ]:
        if feature in groups.get(name, []):
            v090.append(name)
    return ";".join(current), ";".join(v090)


def lineage_for_feature(feature: str, repo: Path) -> dict[str, Any]:
    pipeline = repo / "src/rtbev/pipeline.py"
    nc_eval = repo / "src/rtbev/nc_eval.py"
    audit = repo / "scripts/25c_actionability_feature_audit.py"
    train = repo / "scripts/25_train_actionability_classifiers.py"
    row: dict[str, Any] = {
        "feature_name": feature,
        "raw_input_fields": "current_xy;current_vel_xy;current_heading;current_size_lw;agent_count",
        "maximum_time_index_accessed": "current",
        "uses_current_state": True,
        "uses_map": False,
        "uses_constant_velocity_extrapolation": False,
        "uses_observed_future": False,
        "uses_label_file": False,
        "uses_primitive_safety_matrix": False,
        "uses_explicit_asr_field": False,
        "uses_transformed_or_composite_asr": False,
        "uses_maneuver_survival": False,
        "formula_or_code_reference": "",
        "allowed_in_primary_predictor": False,
        "rationale": "",
    }
    if feature.startswith("cv_"):
        row.update({
            "uses_constant_velocity_extrapolation": True,
            "uses_map": feature in {"cv_rfr_drv", "cv_c_time", "cv_gtoa_norm_union", "cv_oce_norm", "cv_c_density"},
            "maximum_time_index_accessed": "current + CV rollout to configured horizon",
            "formula_or_code_reference": line_ref(pipeline, "def _cv_masks_cpu"),
            "rationale": "CV occupancy baseline from current state only.",
        })
    elif feature in {
        "current_min_distance_m", "current_ttc_s", "ego_speed_kph", "agent_count",
        "nearest_agent_rel_speed_mps", "nearest_agent_closing_speed_mps",
        "ttc_closing_speed_mps", "nearby_agent_count_10m", "nearby_agent_count_20m",
    }:
        row.update({
            "formula_or_code_reference": line_ref(pipeline, "def _compute_current_distance_ttc"),
            "allowed_in_primary_predictor": True,
            "rationale": "Strong baseline kinematics computed from current state only.",
        })
    elif feature in {"rcr", "rfr_drv", "c_time", "gtoa_norm_union", "oce_norm", "c_density", "msr", "c_maneuver", "redi_full", "redi_no_msr"}:
        row.update({
            "uses_map": feature in {"rfr_drv", "c_time", "gtoa_norm_union", "oce_norm", "c_density", "redi_full", "redi_no_msr"},
            "maximum_time_index_accessed": "current state plus reachability tube horizon",
            "formula_or_code_reference": line_ref(pipeline, "def _basic_reachability_metrics") if feature not in {"redi_full", "redi_no_msr"} else line_ref(pipeline, "def _redi"),
            "uses_primitive_safety_matrix": feature in {"msr", "c_maneuver"},
            "uses_transformed_or_composite_asr": feature in {"msr", "c_maneuver", "redi_full"},
            "rationale": "ROF spatial/reachability feature; not part of primary strict-temporal comparison.",
        })
    elif feature in {"ttad_s", "time_to_first_conflict_s", "early_blocking_ratio", "collapse_rate_max_per_s", "collapse_rate_mean_per_s"}:
        row.update({
            "uses_map": True,
            "uses_primitive_safety_matrix": True,
            "uses_transformed_or_composite_asr": True,
            "maximum_time_index_accessed": "current state plus primitive safety horizon",
            "formula_or_code_reference": line_ref(pipeline, "def _summarize_actionability"),
            "allowed_in_primary_predictor": True,
            "rationale": "Locked strict temporal dynamics fields; no label file or observed future trajectory access in feature code.",
        })
    elif feature in {"comfort_asr", "emergency_asr", "comfort_to_emergency_gap", "asr_slice_final", "asr_slice_min", "asr_cum_final", "asr_cum_min"}:
        row.update({
            "uses_map": True,
            "uses_primitive_safety_matrix": True,
            "uses_explicit_asr_field": True,
            "maximum_time_index_accessed": "current state plus primitive safety horizon",
            "formula_or_code_reference": line_ref(pipeline, "def _summarize_actionability"),
            "rationale": "Explicit action-ratio field; context only due endpoint coupling concern.",
        })
    elif feature in {"redi_actionability", "redi_actionability_delta"}:
        row.update({
            "uses_map": True,
            "uses_primitive_safety_matrix": True,
            "uses_transformed_or_composite_asr": True,
            "maximum_time_index_accessed": "current state plus primitive safety horizon",
            "formula_or_code_reference": line_ref(pipeline, "redi_actionability ="),
            "rationale": "Composite actionability score includes transformed cumulative ASR terms; renamed secondary context.",
        })
    elif feature.startswith("slice_survival_") or feature.startswith("survival_"):
        row.update({
            "uses_map": True,
            "uses_primitive_safety_matrix": True,
            "uses_maneuver_survival": True,
            "maximum_time_index_accessed": "current state plus primitive safety horizon",
            "formula_or_code_reference": line_ref(pipeline, "slice_survival_"),
            "rationale": "Maneuver-family survival from primitive safety matrix; not strict primary field.",
        })
    else:
        row.update({
            "formula_or_code_reference": f"{nc_eval.as_posix()};{audit.as_posix()};{train.as_posix()}",
            "rationale": "Unclassified predictor column included for completeness; not allowed in primary predictor until manually reviewed.",
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v090/nc_v090_audit.yaml")
    args = parser.parse_args()
    repo = Path.cwd()
    cfg = load_yaml(repo / args.config)
    out_dir = repo / cfg["project"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lists = literal_list_assignments(repo / "scripts/25_train_actionability_classifiers.py")
    audit_lists = literal_list_assignments(repo / "scripts/25c_actionability_feature_audit.py")
    cfg_groups = cfg.get("feature_groups_v090", {})
    groups: dict[str, list[str]] = {
        "strong_baseline": train_lists.get("STRONG_BASELINE_COLS", []),
        "cv_cols": train_lists.get("CV_COLS", []),
        "strong_baseline_cv": train_lists.get("STRONG_BASELINE_COLS", []) + train_lists.get("CV_COLS", []),
        "direct_action_ratios_only": audit_lists.get("DIRECT_ACTION_RATIO_COLS", []),
        "explicit_ratio_field_excluded_current": (
            audit_lists.get("REDI_ACTIONABILITY_COLS", [])
            + audit_lists.get("TEMPORAL_ACTIONABILITY_COLS", [])
            + audit_lists.get("SURVIVAL_COLS", [])
        ),
        "strict_temporal_dynamics": list(cfg_groups.get("strict_temporal_dynamics", [])),
        "strict_spatial_no_action": list(cfg_groups.get("strict_spatial_no_action", [])),
    }

    features_header = csv_header(Path(cfg["inputs"]["waymo_features_csv"]))
    feature_union = sorted({f for cols in groups.values() for f in cols if f in features_header})
    rows: list[dict[str, Any]] = []
    for feature in feature_union:
        current, v090 = membership(feature, groups)
        row = lineage_for_feature(feature, repo)
        row["feature_group_current"] = current
        row["feature_group_v090"] = v090
        if current == "strong_baseline_cv" or "strong_baseline_cv" in current.split(";"):
            row["allowed_in_primary_predictor"] = True
        if any(group in PRIMARY_GROUPS for group in v090.split(";")) and (
            row["uses_observed_future"] or row["uses_label_file"] or LABEL_LIKE_RE.search(feature)
        ):
            row["allowed_in_primary_predictor"] = False
            row["rationale"] += " BLOCKED: primary predictor cannot use label/future fields."
        rows.append(row)
    write_csv(out_dir / "feature_lineage.csv", rows)

    access_rows = [
        {
            "component": "map_constrained_actionability_label_generation",
            "source_file": "scripts/24_build_actionability_labels.py",
            "uses_observed_future": True,
            "uses_constant_velocity_extrapolation": False,
            "uses_map": True,
            "uses_label_file": True,
            "notes": "oracle_future mode uses sample future_xy/future_heading/future_valid to generate labels; label file supplies sample IDs and original labels.",
        },
        {
            "component": "no_map_actionability_label_generation",
            "source_file": "scripts/24_build_actionability_labels.py",
            "uses_observed_future": True,
            "uses_constant_velocity_extrapolation": False,
            "uses_map": False,
            "uses_label_file": True,
            "notes": "same oracle_future obstacle source; --no-use-map disables drivable/map constraint.",
        },
        {
            "component": "strong_baseline_cv_predictors",
            "source_file": "src/rtbev/pipeline.py",
            "uses_observed_future": False,
            "uses_constant_velocity_extrapolation": True,
            "uses_map": True,
            "uses_label_file": False,
            "notes": "current-state kinematics plus CV occupancy features.",
        },
        {
            "component": "strict_temporal_dynamics_predictors",
            "source_file": "src/rtbev/pipeline.py",
            "uses_observed_future": False,
            "uses_constant_velocity_extrapolation": False,
            "uses_map": True,
            "uses_label_file": False,
            "notes": "derived from primitive safety matrix and reachability/occupancy, not from observed future trajectories.",
        },
        {
            "component": "model_fitting_and_calibration",
            "source_file": "scripts/nc_v090/02_waymo_confirmatory_oof.py",
            "uses_observed_future": False,
            "uses_constant_velocity_extrapolation": False,
            "uses_map": False,
            "uses_label_file": True,
            "notes": "labels define supervised endpoint only; preprocessing and thresholds are fit inside training/calibration splits.",
        },
        {
            "component": "commonroad_planner_labels",
            "source_file": "scripts/51_commonroad_lattice_planner_feasibility.py",
            "uses_observed_future": False,
            "uses_constant_velocity_extrapolation": False,
            "uses_map": True,
            "uses_label_file": False,
            "notes": "separately implemented 35-candidate lattice planner; not fully independent in assumptions.",
        },
    ]
    write_csv(out_dir / "information_access_manifest.csv", access_rows)

    f_ids = read_ids(Path(cfg["inputs"]["waymo_features_csv"]))
    map_ids = read_ids(Path(cfg["inputs"]["waymo_actionability_map_labels_csv"]))
    nomap_ids = read_ids(Path(cfg["inputs"]["waymo_actionability_nomap_labels_csv"]))
    prox_ids = read_ids(Path(cfg["inputs"]["waymo_proximity_labels_csv"]))
    merge_rows = []
    for name, df in [("features", f_ids), ("map_labels", map_ids), ("nomap_labels", nomap_ids), ("proximity_labels", prox_ids)]:
        merge_rows.append({
            "artifact": name,
            "rows": len(df),
            "unique_sample_id": df["sample_id"].nunique(),
            "duplicate_sample_id_count": int(df["sample_id"].duplicated().sum()),
            "unique_scenario_id": int(df["scenario_id"].nunique()) if "scenario_id" in df.columns else None,
        })
    base = f_ids[["sample_id"]].merge(map_ids[["sample_id"]], on="sample_id", how="inner")
    base = base.merge(nomap_ids[["sample_id"]], on="sample_id", how="inner")
    base = base.merge(prox_ids[["sample_id"]], on="sample_id", how="inner")
    merge_rows.append({
        "artifact": "features_map_nomap_proximity_inner_join",
        "rows": len(base),
        "unique_sample_id": base["sample_id"].nunique(),
        "duplicate_sample_id_count": int(base["sample_id"].duplicated().sum()),
    })
    write_csv(out_dir / "merge_cardinality_audit.csv", merge_rows)

    test_rows = []
    primary = [r for r in rows if any(g in PRIMARY_GROUPS for g in str(r["feature_group_v090"]).split(";"))]
    for r in primary:
        passed = not (r["uses_observed_future"] or r["uses_label_file"] or LABEL_LIKE_RE.search(str(r["feature_name"])))
        test_rows.append({
            "test": "primary_predictor_no_future_or_label_access",
            "feature_name": r["feature_name"],
            "status": "PASS" if passed else "FAIL",
            "details": r["rationale"],
        })
    for row in merge_rows:
        if row["artifact"] != "features_map_nomap_proximity_inner_join":
            passed = int(row["duplicate_sample_id_count"]) == 0
            test_rows.append({
                "test": "duplicate_sample_id",
                "artifact": row["artifact"],
                "status": "PASS" if passed else "FAIL",
                "details": f"duplicates={row['duplicate_sample_id_count']}",
            })
    joined_ok = len(base) == len(f_ids) == map_ids["sample_id"].nunique() == nomap_ids["sample_id"].nunique()
    test_rows.append({
        "test": "one_to_one_merge_cardinality",
        "artifact": "features_map_nomap_proximity",
        "status": "PASS" if joined_ok else "FAIL",
        "details": f"join_rows={len(base)} features_rows={len(f_ids)}",
    })
    write_csv(out_dir / "information_access_tests.csv", test_rows)
    print(f"[nc-v090] wrote lineage/access audit to {out_dir}")
    print(f"[nc-v090] tests: {pd.Series([r['status'] for r in test_rows]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
