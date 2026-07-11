from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config


ARTIFACTS = [
    ("legacy proximity labels", "labels/labels_val_full.csv"),
    ("frozen ROF features", "features/rof_features_val_full.csv"),
    ("map-constrained actionability labels", "labels/labels_actionability_moderate_full.csv"),
    ("no-map actionability labels", "labels/labels_actionability_moderate_nomap_full.csv"),
    (
        "map expanded baseline metrics",
        "results/nc_actionability_classification/actionability_moderate_expanded_baseline/actionability_classification_metrics.csv",
    ),
    (
        "map expanded baseline bootstrap",
        "results/nc_actionability_classification/actionability_moderate_expanded_baseline/actionability_bootstrap_deltas.csv",
    ),
    (
        "map feature audit metrics",
        "results/nc_actionability_classification/actionability_feature_audit_fast/feature_group_metrics.csv",
    ),
    (
        "map selected cases",
        "results/nc_actionability_cases/actionability_case_selection_v1/selected_cases_all.csv",
    ),
    (
        "map case panel index",
        "results/nc_actionability_cases/actionability_case_panels_v1/index.html",
    ),
    (
        "map feasibility audit summary",
        "results/nc_actionability_cases/actionability_case_feasibility_audit_v1/action_feasibility_audit_summary_by_category.csv",
    ),
]


def _count_csv_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            count = sum(1 for _ in reader)
        return max(count - 1, 0)
    except Exception:
        return None


def _fmt_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Check key actionability-main artifacts without running experiments.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(args.work_dir) if args.work_dir else Path(cfg["project"]["work_dir"])
    rows = []
    for name, rel in ARTIFACTS:
        path = work / rel
        exists = path.exists()
        rows.append({
            "name": name,
            "status": "exists" if exists else "missing",
            "size_bytes": path.stat().st_size if exists else "",
            "csv_rows": _count_csv_rows(path) if exists else "",
            "modified_at": _fmt_mtime(path) if exists else "",
            "path": str(path),
        })

    headers = ["status", "name", "size_bytes", "csv_rows", "modified_at", "path"]
    print(f"work_dir: {work}")
    print(",".join(headers))
    for row in rows:
        print(",".join(str(row[h]) for h in headers))
    missing = [r for r in rows if r["status"] == "missing"]
    print(f"summary: exists={len(rows) - len(missing)}, missing={len(missing)}, total={len(rows)}")
    if missing:
        print("missing_artifacts:")
        for row in missing:
            print(f"- {row['name']}: {row['path']}")


if __name__ == "__main__":
    main()

