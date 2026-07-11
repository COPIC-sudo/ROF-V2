#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _utils import count_csv_rows, detect_gpu, environment_row, load_yaml, output_dir, package_status, resolve_path, sha256, write_csv


CODE_PATHS = [
    "scripts/04_generate_rof_features.py",
    "src/rtbev/pipeline.py",
    "scripts/nc_v090/02_waymo_confirmatory_oof.py",
    "scripts/nc_v096/01_generate_design_variant_labels.py",
    "scripts/nc_v096/03_model_eval_variants.py",
    "scripts/nc_v097/01_generate_aligned_features.py",
    "scripts/nc_v097/02_aligned_oof_eval.py",
]


def artifact_row(item: str, path: Path) -> dict[str, Any]:
    return {
        "item": item,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else "",
        "rows": count_csv_rows(path) if path.exists() and path.suffix.lower() == ".csv" else "",
        "sha256": sha256(path) if path.exists() and path.is_file() else "",
    }


def write_variant_manifest(cfg: dict[str, Any], out_dir: Path) -> None:
    import pandas as pd

    src = resolve_path(cfg["inputs"]["v096_variant_manifest_csv"])
    wanted = list(cfg["aligned_features"]["variants"])
    src_df = pd.read_csv(src)
    rows = []
    for variant_id in wanted:
        hit = src_df[src_df["variant_id"].astype(str) == variant_id]
        if hit.empty:
            rows.append(
                {
                    "variant_id": variant_id,
                    "status": "MISSING_IN_V096_MANIFEST",
                    "feature_alignment_mode": "BLOCKED",
                }
            )
            continue
        r = hit.iloc[0].to_dict()
        feature_mode = "ALIGNED_LABEL_AND_FEATURE_VARIANT"
        no_op_reason = ""
        if str(r.get("future_handling", "")) == "cv_fallback":
            no_op_reason = "future_handling affects label generation; predictor features do not read observed future trajectories"
        rows.append(
            {
                "variant_id": variant_id,
                "family": r.get("family", ""),
                "horizon_s": r.get("horizon_s", ""),
                "lane_buffer_m": r.get("lane_buffer_m", ""),
                "action_library": r.get("action_library", ""),
                "future_handling": r.get("future_handling", ""),
                "label_csv": r.get("label_csv", ""),
                "label_rows": count_csv_rows(Path(str(r.get("label_csv", "")))) if Path(str(r.get("label_csv", ""))).exists() else "",
                "feature_alignment_mode": feature_mode,
                "current_state_fields_action": "reuse_reference_features_with_checksum",
                "cv_fields_action": "regenerate_horizon_lane_buffer_aligned",
                "strict_temporal_fields_action": "regenerate_candidate_action_cv_survival_aligned",
                "future_handling_no_op_reason": no_op_reason,
                "status": "PLANNED",
            }
        )
    write_csv(out_dir / "aligned_feature_variant_manifest.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v097/nc_v097_aligned_feature_robustness.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    write_variant_manifest(cfg, out_dir)

    artifacts = []
    for key, value in cfg.get("inputs", {}).items():
        artifacts.append(artifact_row(key, resolve_path(value)))
    for path in CODE_PATHS:
        artifacts.append(artifact_row(path, resolve_path(path)))
    write_csv(out_dir / "input_artifact_inventory_v097.csv", artifacts)

    deps = [environment_row()]
    for pkg in ["numpy", "pandas", "sklearn", "shapely", "joblib", "yaml", "torch"]:
        deps.append(package_status(pkg))
    write_csv(out_dir / "dependency_inventory_v097.csv", deps)

    gpu = detect_gpu()
    (out_dir / "GPU_USAGE_REPORT_V097.md").write_text(
        "# GPU Usage Report v0.9.7\n\n"
        f"Decision: `{gpu.get('decision')}`\n\n"
        f"Reason: {gpu.get('reason')}\n\n"
        "```json\n" + json.dumps(gpu, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )

    lineage = [
        {
            "feature": "current_min_distance_m,current_ttc_s,ego_speed_kph,agent_count,relative/closing speed,nearby counts",
            "feature_group": "current_state_baseline",
            "design_dependency": "independent",
            "v097_action": "reuse_reference_features_with_checksum",
            "justification": "Computed from current sample state only; does not depend on rollout horizon, lane buffer, action library, or future handling.",
        },
        {
            "feature": "cv_rcr,cv_rfr_drv,cv_c_time,cv_gtoa_norm_union,cv_oce_norm,cv_c_density,cv_max_overlap_count",
            "feature_group": "cv_occupancy_baseline",
            "design_dependency": "horizon_and_lane_buffer",
            "v097_action": "regenerate_for_variant_horizon_and_lane_buffer",
            "justification": "CV occupancy uses constant-velocity masks over query times and lane/drivable mask.",
        },
        {
            "feature": "ttad_s,time_to_first_conflict_s,early_blocking_ratio,collapse_rate_max_per_s,collapse_rate_mean_per_s",
            "feature_group": "strict_temporal_dynamics",
            "design_dependency": "horizon_lane_buffer_action_library",
            "v097_action": "regenerate_candidate_action_cv_survival",
            "justification": "Regenerated without observed future labels using endpoint candidate-action library, current-state CV obstacles, and variant map buffer.",
        },
        {
            "feature": "future_handling=cv_fallback predictor features",
            "feature_group": "future_handling",
            "design_dependency": "none_for_predictor",
            "v097_action": "no_op_same_as_reference_parameters",
            "justification": "Predictor features do not read observed future trajectories; future handling is a label-generation policy only.",
        },
    ]
    write_csv(out_dir / "aligned_feature_lineage.csv", lineage)

    plan = [
        "# NC v0.9.7 Run Plan Locked",
        "",
        "Goal: upgrade v0.9.6 endpoint-design robustness from reference-feature evaluation to the strongest feasible aligned label+feature analysis.",
        "",
        "Feature strategy:",
        "- Current-state baseline fields are reused from the reference feature table with checksum documentation.",
        "- `cv_*` baseline fields are regenerated for each unique horizon/lane-buffer pair using current-state constant-velocity occupancy and the variant lane buffer.",
        "- Strict temporal dynamics are regenerated using endpoint candidate-action CV-survival dynamics. This path does not use oracle future trajectories, labels, or planner outcomes.",
        "- Future-handling variant is a predictor no-op because predictor features use current-state CV, not observed future.",
        "",
        "Important limitation:",
        "- The original v0.9 strict-temporal fields came from the primitive-survival pipeline. The v0.9.7 aligned path uses candidate-action CV-survival dynamics so that the extended action library can be represented without modifying `pipeline.py` or rebuilding primitive libraries.",
        "- Therefore claim wording should be `aligned label-and-feature endpoint-design sensitivity`, with Methods noting the regenerated strict-temporal feature path.",
    ]
    (out_dir / "RUN_PLAN_LOCKED_V097.md").write_text("\n".join(plan) + "\n", encoding="utf-8")
    print(f"[v097-inventory] wrote {out_dir}")


if __name__ == "__main__":
    main()
