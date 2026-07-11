#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "shapely",
    "matplotlib",
    "joblib",
    "yaml",
    "pyarrow",
    "fastparquet",
    "torch",
    "commonroad",
    "commonroad_io",
]

CODE_TARGETS = {
    "waymo_label_generation": [
        "scripts/03_extract_and_label.py",
        "src/rtbev/labels.py",
        "src/rtbev/waymo_reader.py",
        "scripts/24_build_actionability_labels.py",
    ],
    "feature_generation": [
        "scripts/04_generate_rof_features.py",
        "src/rtbev/pipeline.py",
        "src/rtbev/nc_eval.py",
    ],
    "feature_group_definitions": [
        "src/rtbev/nc_eval.py",
        "scripts/25_train_actionability_classifiers.py",
        "scripts/25c_actionability_feature_audit.py",
    ],
    "preprocessing_split_threshold_bootstrap": [
        "src/rtbev/nc_eval.py",
        "scripts/25_train_actionability_classifiers.py",
        "scripts/25b_actionability_bootstrap_from_predictions.py",
        "scripts/25c_actionability_feature_audit.py",
    ],
    "commonroad_sampling_planner_eval": [
        "scripts/40_scan_commonroad_scenarios.py",
        "scripts/41_select_commonroad_pilot.py",
        "scripts/41b_refine_commonroad_pilot.py",
        "scripts/43b_export_commonroad_dynamic_ego_samples.py",
        "scripts/51_commonroad_lattice_planner_feasibility.py",
        "scripts/52_evaluate_commonroad_planner_failure_scalars.py",
        "scripts/53_bootstrap_commonroad_planner_failure_scalars.py",
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_cmd(args: list[str], cwd: Path) -> dict[str, Any]:
    cp = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {
        "command": " ".join(args),
        "returncode": int(cp.returncode),
        "stdout": cp.stdout.strip(),
        "stderr": cp.stderr.strip(),
    }


def package_versions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in PACKAGE_NAMES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "OK")
            status = "ok"
            error = ""
        except Exception as exc:
            version = ""
            status = "missing"
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"package": name, "status": status, "version": version, "error": error})
    return rows


def csv_row_count(path: Path, usecols: list[str] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "kind": "missing", "rows": 0, "columns": 0, "unique_sample_id": None, "sha256": None}
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        return {
            "exists": True,
            "kind": "directory",
            "rows": None,
            "columns": None,
            "unique_sample_id": None,
            "sha256": None,
            "size_bytes": int(sum(p.stat().st_size for p in files)),
            "file_count": int(len(files)),
        }
    if path.suffix.lower() != ".csv":
        return {
            "exists": True,
            "kind": "file",
            "rows": None,
            "columns": None,
            "unique_sample_id": None,
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    df = pd.read_csv(path, usecols=usecols)
    unique_sample = int(df["sample_id"].astype(str).nunique()) if "sample_id" in df.columns else None
    if usecols is not None:
        header = pd.read_csv(path, nrows=0)
        n_cols = int(len(header.columns))
    else:
        n_cols = int(len(df.columns))
    return {
        "exists": True,
        "kind": "csv",
        "rows": int(len(df)),
        "columns": n_cols,
        "unique_sample_id": unique_sample,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_run_plan(cfg: dict[str, Any], out_dir: Path, inventory: dict[str, Any]) -> str:
    inputs = cfg["inputs"]
    folds = cfg["splits"]["outer_folds"]
    rf_seeds = cfg["splits"]["rf_seeds"]
    return f"""# NC v0.9.0 Scientific Audit RUN_PLAN

Generated: {datetime.now(timezone.utc).isoformat()}

## Scope

This run writes only to:

- `results/nc_v090_scientific_audit/`
- `configs/nc_v090/`
- `scripts/nc_v090/`
- `tests/nc_v090/`

It must not overwrite v0.8.1/v0.9 manuscript artifacts or edit LaTeX.

## Inputs

- Base config: `{inputs['base_config']}`
- Waymo features: `{inputs['waymo_features_csv']}`
- Waymo proximity labels: `{inputs['waymo_proximity_labels_csv']}`
- Map-constrained actionability labels: `{inputs['waymo_actionability_map_labels_csv']}`
- No-map actionability labels: `{inputs['waymo_actionability_nomap_labels_csv']}`
- CommonRoad results root: `{inputs['commonroad_results_dir']}`

## Observed Input Counts

- Waymo features rows: {inventory.get('waymo_features_rows')}
- Map actionability label rows: {inventory.get('map_actionability_rows')}
- No-map actionability label rows: {inventory.get('nomap_actionability_rows')}
- Proximity label rows: {inventory.get('proximity_label_rows')}

## Checkpoint Order

1. Phase 0 inventory and smoke tests.
2. Phase 1 feature lineage and information-access audit.
3. Phase 2 Waymo confirmatory OOF evaluation:
   - {folds}-fold scenario-hash outer CV.
   - RF seeds: {rf_seeds}.
   - Fit/calibration split inside each outer train fold.
   - Threshold at nominal 5% FPR chosen on calibration negatives only.
4. Phase 3 endpoint robustness:
   - Start with metadata/manifest and only run supported label variants.
5. Phase 4 TTC/proximity distinctness and preprocessing sensitivity.
6. Phase 5 CommonRoad:
   - Prefer neutral cohort. If unavailable, label existing enriched/pilot analysis as fallback stress test.
7. Phase 6 final manifests, claim gate, blockers, and patch notes.

## Smoke Commands

```powershell
$CFG=\"configs/nc_v090/nc_v090_audit.yaml\"
conda run -n waymo_rt_bev python scripts/nc_v090/00_inventory_and_run_plan.py --config $CFG
conda run -n waymo_rt_bev python -m py_compile scripts/nc_v090/*.py
```

Additional smoke tests will be added under `tests/nc_v090/` before full jobs.

## Full Commands To Be Added After Lineage Audit

```powershell
conda run -n waymo_rt_bev python scripts/nc_v090/01_feature_lineage_audit.py --config $CFG
conda run -n waymo_rt_bev python scripts/nc_v090/02_waymo_confirmatory_oof.py --config $CFG
conda run -n waymo_rt_bev python scripts/nc_v090/03_ttc_sensitivity.py --config $CFG
```

## Blocker Policy

If a dataset or dependency is missing, continue unaffected tasks and write `BLOCKERS.md` with the exact missing path/dependency and resume command.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v090/nc_v090_audit.yaml")
    args = parser.parse_args()

    repo = Path.cwd()
    cfg = load_yaml(repo / args.config)
    out_dir = repo / cfg["project"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }
    (out_dir / "environment_inventory.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "package_versions.csv", env["packages"])

    git_rows = [
        {"name": "status_short", **run_cmd(["git", "-c", f"safe.directory={repo.as_posix()}", "status", "--short"], repo)},
        {"name": "last_commit", **run_cmd(["git", "-c", f"safe.directory={repo.as_posix()}", "log", "--oneline", "-1"], repo)},
    ]
    write_csv(out_dir / "git_inventory.csv", git_rows)

    input_rows: list[dict[str, Any]] = []
    for key, value in cfg["inputs"].items():
        path = Path(str(value))
        if not path.is_absolute():
            path = repo / path
        row = {"input_key": key, "path": str(path), **csv_row_count(path, usecols=["sample_id"] if path.suffix.lower() == ".csv" else None)}
        input_rows.append(row)
    write_csv(out_dir / "input_artifact_inventory.csv", input_rows)

    code_rows: list[dict[str, Any]] = []
    for category, paths in CODE_TARGETS.items():
        for path_text in paths:
            path = repo / path_text
            code_rows.append({
                "category": category,
                "path": path_text,
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            })
    write_csv(out_dir / "code_location_inventory.csv", code_rows)

    counts = {row["input_key"]: row for row in input_rows}
    inventory = {
        "waymo_features_rows": counts.get("waymo_features_csv", {}).get("rows"),
        "map_actionability_rows": counts.get("waymo_actionability_map_labels_csv", {}).get("rows"),
        "nomap_actionability_rows": counts.get("waymo_actionability_nomap_labels_csv", {}).get("rows"),
        "proximity_label_rows": counts.get("waymo_proximity_labels_csv", {}).get("rows"),
    }
    (out_dir / "RUN_PLAN.md").write_text(build_run_plan(cfg, out_dir, inventory), encoding="utf-8")
    print(f"[nc-v090] wrote inventory and RUN_PLAN to {out_dir}")


if __name__ == "__main__":
    main()
