#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _utils import count_csv_rows, detect_gpu, environment_row, load_yaml, output_dir, package_status, resolve_path, sha256, write_csv


PRIOR_ARTIFACTS = [
    "results/nc_v090_scientific_audit/CODEX_FINAL_REPORT.md",
    "results/nc_v090_scientific_audit/BLOCKERS.csv",
    "results/nc_v090_scientific_audit/waymo_confirmatory_metrics.csv",
    "results/nc_v090_scientific_audit/waymo_paired_deltas.csv",
    "results/nc_v095_p0_extension_envswitch/P0_EXTENSION_FINAL_REPORT_ENVSWITCH.md",
    "results/nc_v095_p0_extension_envswitch/CLAIM_GATE_REPORT_V095_ENVSWITCH.csv",
    "results/nc_v095_p0_extension_envswitch/waymo_design_variant_manifest.csv",
    "results/nc_v095_p0_extension_envswitch/future_validity_summary.csv",
]


def artifact_row(name: str, path: Path) -> dict[str, Any]:
    return {
        "artifact": name,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else "",
        "rows": count_csv_rows(path) if path.exists() and path.suffix.lower() == ".csv" else "",
        "sha256": sha256(path) if path.exists() and path.is_file() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_endpoint_design_robustness.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)

    artifact_rows: list[dict[str, Any]] = []
    for rel in PRIOR_ARTIFACTS:
        artifact_rows.append(artifact_row(rel, resolve_path(rel)))
    for key, value in cfg.get("inputs", {}).items():
        if str(key).endswith("_csv") or str(key).endswith("_dir"):
            artifact_rows.append(artifact_row(key, resolve_path(value)))
    write_csv(out_dir / "input_artifact_inventory.csv", artifact_rows)

    code_paths = [
        "scripts/24_build_actionability_labels.py",
        "scripts/nc_v090/02_waymo_confirmatory_oof.py",
        "scripts/nc_v090/02b_waymo_bootstrap_from_oof.py",
        "scripts/nc_v095/02_endpoint_design_robustness.py",
        "scripts/nc_v095/03_future_validity_audit.py",
        "scripts/nc_v096/01_generate_design_variant_labels.py",
        "scripts/nc_v096/03_model_eval_variants.py",
        "scripts/nc_v096/04_bootstrap_variants.py",
    ]
    write_csv(out_dir / "code_path_inventory.csv", [artifact_row(p, resolve_path(p)) for p in code_paths])

    packages = [environment_row()]
    for name in ["numpy", "pandas", "sklearn", "shapely", "joblib", "yaml", "torch"]:
        packages.append(package_status(name))
    write_csv(out_dir / "package_versions.csv", packages)

    gpu = detect_gpu()
    (out_dir / "GPU_USAGE_REPORT.md").write_text(
        "# GPU Usage Report (v0.9.6)\n\n"
        f"Decision: `{gpu.get('decision')}`\n\n"
        f"Reason: {gpu.get('reason')}\n\n"
        "Detected backends:\n\n"
        "```json\n" + json.dumps(gpu, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )

    variants = cfg["actionability_labels"]["variants"]
    plan = [
        "# NC v0.9.6 Run Plan Locked",
        "",
        "Scope: endpoint-design robustness full relabeling plus model reevaluation for Waymo actionability labels.",
        "",
        "Code reuse:",
        "- Reuse the geometry, candidate-action rollout, rectangle-overlap, map lane-buffer and moderate label-rule definitions from `scripts/24_build_actionability_labels.py`.",
        "- Reuse the v0.9.0 confirmatory OOF protocol from `scripts/nc_v090/02_waymo_confirmatory_oof.py`: scenario-hash outer folds, train-only median imputation, calibration-negative fixed-FPR thresholds, RandomForestClassifier.",
        "",
        "Design parameters changed:",
    ]
    for v in variants:
        plan.append(
            f"- `{v['variant_id']}`: horizon={v['horizon_s']}s, lane_buffer={v['lane_buffer_m']}m, "
            f"actions={v['action_library']}, future_handling={v['future_handling']}, threshold=moderate, map-constrained."
        )
    plan.extend(
        [
            "",
            "Feature handling:",
            f"- Model evaluation status: `{cfg['evaluation']['feature_mode']}`.",
            "- The full relabeling variants regenerate feasible-action ratios and endpoint labels. ROF/reference feature CSV is reused because per-variant strict-temporal features are not safely regenerated in this run.",
            "- Therefore model conclusions are label-variant robustness with reference features, not fully aligned label+feature robustness.",
            "",
            "Execution order:",
            "1. Generate variant labels with `scripts/nc_v096/01_generate_design_variant_labels.py`.",
            "2. Build label stability and CV-fallback transition tables with `scripts/nc_v096/02_label_stability.py`.",
            "3. Run OOF model reevaluation with `scripts/nc_v096/03_model_eval_variants.py`.",
            "4. Run scenario bootstrap with `scripts/nc_v096/04_bootstrap_variants.py`.",
            "5. Build claim gates/final package with `scripts/nc_v096/05_final_package.py`.",
            "",
            "Runtime caveats:",
            "- Seed 42 is mandatory for every variant; seeds 41/43 are optional runtime extensions under the locked plan.",
            "- Bootstrap defaults to 1000 replicates in this run to keep the P0 extension tractable.",
        ]
    )
    (out_dir / "RUN_PLAN_LOCKED.md").write_text("\n".join(plan) + "\n", encoding="utf-8")
    print(f"[v096-inventory] wrote {out_dir}")


if __name__ == "__main__":
    main()
