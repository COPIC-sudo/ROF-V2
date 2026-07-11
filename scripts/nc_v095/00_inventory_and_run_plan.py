#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from _utils import count_csv_rows, detect_torch_gpu, environment_row, load_yaml, output_dir, package_status, resolve_path, write_csv, write_json


INPUT_KEYS = [
    "waymo_features_csv",
    "waymo_proximity_labels_csv",
    "waymo_actionability_map_labels_csv",
    "waymo_actionability_nomap_labels_csv",
    "waymo_samples_dir",
    "waymo_oof_predictions_csv",
    "commonroad_raw_root",
    "commonroad_results_dir",
]

PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "yaml",
    "shapely",
    "torch",
    "pyarrow",
    "fastparquet",
    "pytest",
    "commonroad",
    "commonroad_io",
]


def artifact_row(key: str, value: str) -> dict[str, Any]:
    p = resolve_path(value)
    row: dict[str, Any] = {
        "artifact_key": key,
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else "",
        "row_count": "",
        "notes": "",
    }
    if p.exists() and p.is_file() and p.suffix.lower() == ".csv":
        row["row_count"] = count_csv_rows(p)
    if p.exists() and p.is_dir():
        try:
            row["direct_child_count"] = len(list(p.iterdir()))
        except Exception as exc:
            row["notes"] = f"cannot list directory: {type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v095/nc_v095_p0_extension.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    os.environ.setdefault("NC_V095_USE_GPU", "auto")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    rows = []
    for key in INPUT_KEYS:
        value = cfg["inputs"].get(key, "")
        rows.append(artifact_row(key, str(value)))
    v090_dir = resolve_path(cfg["inputs"]["v090_output_dir"])
    if v090_dir.exists():
        for name in [
            "waymo_confirmatory_metrics.csv",
            "waymo_paired_deltas.csv",
            "waymo_oof_predictions.csv",
            "label_robustness_summary.csv",
            "future_validity_audit.csv",
            "commonroad_confirmatory_metrics.csv",
            "commonroad_confirmatory_deltas.csv",
        ]:
            rows.append(artifact_row(f"v090_{name}", str(v090_dir / name)))
    write_csv(out_dir / "input_artifact_inventory.csv", rows)

    dep_rows = [package_status(pkg) for pkg in PACKAGES]
    torch_gpu = detect_torch_gpu()
    for row in dep_rows:
        if row["package"] == "torch":
            row.update(torch_gpu)
    write_csv(out_dir / "dependency_inventory.csv", dep_rows)
    write_json(out_dir / "environment_inventory_v095.json", environment_row() | {"gpu": torch_gpu})

    gpu_rows = []
    for component in [
        "inventory",
        "commonroad_neutral_confirmation",
        "endpoint_design_robustness",
        "future_validity_audit",
        "secondary_context_bootstrap",
        "final_decision_package",
    ]:
        gpu_rows.append(
            {
                "component": component,
                "gpu_available": bool(torch_gpu.get("gpu_available", False)),
                "gpu_used": False,
                "backend": torch_gpu.get("backend", "torch"),
                "cpu_parity_subset_n": 0,
                "max_abs_diff": "",
                "reason_if_not_used": "Component reads frozen CSV/pkl metadata or runs CPU sklearn metrics; GPU acceleration is not required for bitwise-identical results.",
            }
        )
    write_csv(out_dir / "gpu_usage_report.csv", gpu_rows)
    parity_row = {
        "component": "torch_vector_add",
        "gpu_available": bool(torch_gpu.get("gpu_available", False)),
        "gpu_used": False,
        "backend": "torch",
        "cpu_parity_subset_n": 0,
        "max_abs_diff": "",
        "status": "NOT_RUN",
        "notes": "GPU not available or torch import failed.",
    }
    if torch_gpu.get("gpu_available", False):
        try:
            import torch

            x_cpu = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float32)
            y_cpu = torch.sin(x_cpu) + x_cpu * x_cpu
            x_gpu = x_cpu.cuda()
            y_gpu = (torch.sin(x_gpu) + x_gpu * x_gpu).cpu()
            parity_row.update(
                {
                    "gpu_used": True,
                    "cpu_parity_subset_n": int(x_cpu.numel()),
                    "max_abs_diff": float(torch.max(torch.abs(y_cpu - y_gpu)).item()),
                    "status": "PASS",
                    "notes": "Diagnostic parity only; v0.9.5 CSV/bootstrap tasks remain CPU for reproducibility.",
                }
            )
        except Exception as exc:
            parity_row.update({"status": f"FAILED_{type(exc).__name__}", "notes": str(exc)})
    write_csv(out_dir / "gpu_parity_check.csv", [parity_row])

    plan = [
        "# NC v0.9.5 P0 Extension Run Plan",
        "",
        "Scope: complete the residual P0 checks without rerunning completed Waymo OOF training, feature generation, rolling evaluation, or pipeline.py.",
        "",
        "## Phase Status",
        "",
        "1. Inventory and dependency/GPU report: run in this namespace.",
        "2. CommonRoad neutral confirmation: construct outcome-blind XML manifest if raw XMLs are visible; planner rerun is blocked unless CommonRoad parser/planner dependencies are available.",
        "3. Endpoint robustness: reuse v0.9 threshold-rule outputs; horizon/buffer/action-library variants require full label regeneration and will be reported as blocked unless already present.",
        "4. Future-validity audit: read frozen Waymo sample pkl.gz files and summarize oracle-future validity without changing labels.",
        "5. Secondary/context bootstrap: reuse v0.9 OOF predictions and compute scenario-bootstrap CIs for RF context comparisons.",
        "6. Final decision package: write claim gates, methods notes, reproduction notes, and zip.",
        "",
        "## Non-goals",
        "",
        "- No full rolling.",
        "- No ROF feature regeneration.",
        "- No actionability label regeneration unless a requested variant already exists.",
        "- No pipeline.py changes.",
    ]
    (out_dir / "RUN_PLAN.md").write_text("\n".join(plan) + "\n", encoding="utf-8")
    print(f"[v095-inventory] wrote {out_dir}")


if __name__ == "__main__":
    main()
