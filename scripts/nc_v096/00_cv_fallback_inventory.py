#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _utils import count_csv_rows, detect_gpu, environment_row, load_yaml, output_dir, package_status, resolve_path, sha256, write_csv


REQUIRED_CODE = [
    "configs/my_nc_v11.yaml",
    "scripts/24_build_actionability_labels.py",
    "src/rtbev/pipeline.py",
    "scripts/nc_v090/02_waymo_confirmatory_oof.py",
    "scripts/nc_v090/02b_waymo_bootstrap_from_oof.py",
    "scripts/nc_v096/01_generate_design_variant_labels.py",
    "scripts/nc_v096/03_model_eval_variants.py",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_cv_fallback.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)

    artifacts: list[dict[str, Any]] = []
    for key, value in cfg.get("inputs", {}).items():
        artifacts.append(artifact_row(key, resolve_path(value)))
    for rel in REQUIRED_CODE:
        artifacts.append(artifact_row(rel, resolve_path(rel)))
    write_csv(out_dir / "input_artifact_inventory.csv", artifacts)

    deps = [environment_row()]
    for name in ["numpy", "pandas", "sklearn", "shapely", "joblib", "yaml", "torch"]:
        deps.append(package_status(name))
    deps.append({"package": "gpu_detection", **detect_gpu()})
    write_csv(out_dir / "dependency_inventory.csv", deps)

    plan = [
        "# RUN PLAN: CV-fallback future-handling sensitivity",
        "",
        "Scope: targeted full Waymo actionability label sensitivity for invalid non-ego future states.",
        "",
        "Reference mode: `skip_invalid_oracle_future`.",
        "Variant mode: `cv_fallback_invalid_future`.",
        "",
        "Fixed endpoint settings:",
        f"- horizon_s: {cfg['actionability_labels']['horizon_s']}",
        f"- dt_s: {cfg['actionability_labels']['dt_s']}",
        f"- lane_buffer_m: {cfg['actionability_labels']['lane_buffer_m']}",
        f"- threshold_rule: {cfg['actionability_labels']['threshold_rule']}",
        f"- action_library: {cfg['actionability_labels']['action_library']}",
        "- map constrained: true",
        "",
        "Implementation:",
        "- Reuse v0.9.6 geometry and base7 action rollout helpers from `scripts/nc_v096/01_generate_design_variant_labels.py`, which were adapted from `scripts/24_build_actionability_labels.py`.",
        "- `skip_invalid_oracle_future` reproduces the current label behavior: invalid future obstacle states are excluded.",
        "- `cv_fallback_invalid_future` imputes invalid non-ego future states from the most recent valid future state at or before the requested time; if none exists, it uses the current state when finite; otherwise the slot remains non-imputable.",
        "- Heading follows the configured CV baseline convention: use velocity heading when speed exceeds threshold, otherwise keep the most recent valid heading.",
        "- Model evaluation reuses the frozen reference feature CSV. CV-fallback labels are not used as input features.",
        "",
        "Execution:",
        "1. Inventory inputs and dependencies.",
        "2. Pilot parity: regenerate skip-invalid labels on 500 deterministic samples and compare to existing full labels.",
        "3. Full CV-fallback relabeling for all 43,098 samples.",
        "4. OOF RF evaluation for critical-or-worse and candidate-set-infeasible CV-fallback endpoints.",
        "5. Scenario-level paired bootstrap, claim gate and package.",
    ]
    (out_dir / "RUN_PLAN_CV_FALLBACK.md").write_text("\n".join(plan) + "\n", encoding="utf-8")
    (out_dir / "GPU_USAGE_REPORT_CV_FALLBACK.md").write_text(
        "# GPU Usage Report\n\n"
        "GPU acceleration was not used. The geometry path is Shapely/NumPy CPU code and no validated CPU-vs-GPU parity implementation exists for this label generator.\n\n"
        "```json\n" + json.dumps(detect_gpu(), indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    print(f"[cv-inventory] wrote {out_dir}")


if __name__ == "__main__":
    main()
