#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _utils import append_blockers, load_yaml, output_dir, write_csv


BASELINE = "strong_baseline_cv"
ENHANCED = "strong_baseline_cv_plus_strict_temporal_dynamics"


def read_csv_if(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def metric_lookup(metrics: pd.DataFrame, endpoint: str, seed: int, feature_set: str) -> dict[str, Any]:
    sub = metrics[
        (metrics["level"] == "pooled_oof")
        & (metrics["endpoint"] == endpoint)
        & (metrics["seed"].astype(int) == int(seed))
        & (metrics["feature_set"] == feature_set)
    ]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def delta_lookup(deltas: pd.DataFrame, endpoint: str, seed: int, metric: str) -> dict[str, Any]:
    sub = deltas[(deltas["endpoint"] == endpoint) & (deltas["seed"].astype(int) == int(seed)) & (deltas["metric"] == metric)]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_cv_fallback.yaml")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    labels_path = out_dir / "labels_actionability_moderate_cv_fallback_full.csv"
    metrics_path = out_dir / "cv_fallback_primary_oof_metrics.csv"
    deltas_path = out_dir / "cv_fallback_primary_paired_deltas.csv"
    shift_path = out_dir / "cv_fallback_label_shift_summary.csv"
    imp_path = out_dir / "cv_fallback_imputation_summary.csv"

    blockers: list[dict[str, Any]] = []
    for path, item in [
        (labels_path, "full_cv_fallback_labels"),
        (metrics_path, "oof_metrics"),
        (deltas_path, "paired_deltas"),
        (shift_path, "label_shift_summary"),
        (imp_path, "imputation_summary"),
    ]:
        if not path.exists():
            blockers.append({"category": "cv_fallback", "item": item, "status": "BLOCKED", "details": f"Missing {path}"})

    labels = read_csv_if(labels_path)
    metrics = read_csv_if(metrics_path)
    deltas = read_csv_if(deltas_path)
    shift = read_csv_if(shift_path)
    imp = read_csv_if(imp_path)

    if not labels.empty and len(labels) != 43098:
        blockers.append({"category": "cv_fallback", "item": "full_label_row_count", "status": "PARTIAL", "details": f"Expected 43098 rows, found {len(labels)}"})

    write_rows: list[dict[str, Any]] = []
    for endpoint_cfg in cfg["evaluation"]["endpoints"]:
        endpoint = str(endpoint_cfg["endpoint"])
        for seed in cfg["evaluation"]["rf_seeds"]:
            base = metric_lookup(metrics, endpoint, int(seed), BASELINE) if not metrics.empty else {}
            enh = metric_lookup(metrics, endpoint, int(seed), ENHANCED) if not metrics.empty else {}
            au = delta_lookup(deltas, endpoint, int(seed), "auprc") if not deltas.empty else {}
            rec = delta_lookup(deltas, endpoint, int(seed), "recall_at_5pct_fpr") if not deltas.empty else {}
            write_rows.append(
                {
                    "endpoint": endpoint,
                    "future_handling": cfg["actionability_labels"]["cv_future_handling"],
                    "seed": int(seed),
                    "baseline_auprc": safe_float(base.get("auprc")),
                    "enhanced_auprc": safe_float(enh.get("auprc")),
                    "delta_auprc": safe_float(au.get("delta")),
                    "ci_low": safe_float(au.get("ci_low")),
                    "ci_high": safe_float(au.get("ci_high")),
                    "baseline_recall_at_nominal5fpr": safe_float(base.get("recall_at_5pct_fpr")),
                    "enhanced_recall_at_nominal5fpr": safe_float(enh.get("recall_at_5pct_fpr")),
                    "delta_recall": safe_float(rec.get("delta")),
                    "recall_ci_low": safe_float(rec.get("ci_low")),
                    "recall_ci_high": safe_float(rec.get("ci_high")),
                    "achieved_fpr_baseline": safe_float(base.get("achieved_fpr_at_5pct_fpr")),
                    "achieved_fpr_enhanced": safe_float(enh.get("achieved_fpr_at_5pct_fpr")),
                    "positive_count": int(base.get("positive_count", 0)) if base else "",
                    "prevalence": safe_float(base.get("positive_rate", base.get("prevalence", np.nan))),
                    "manuscript_status": "",
                }
            )

    primary_rows = [r for r in write_rows if r["endpoint"] == "map_critical_or_worse_cv_fallback"]
    primary_auprc_pos = all(np.isfinite(r["delta_auprc"]) and r["delta_auprc"] > 0 for r in primary_rows)
    primary_auprc_ci_pos = all(np.isfinite(r["ci_low"]) and r["ci_low"] > 0 for r in primary_rows)
    primary_recall_pos = all(np.isfinite(r["delta_recall"]) and r["delta_recall"] > 0 for r in primary_rows)
    label_shift = safe_float(shift["label_changed_fraction"].iloc[0]) if not shift.empty and "label_changed_fraction" in shift else np.nan
    impute_p95 = safe_float(imp["p95_sample_imputed_fraction"].iloc[0]) if not imp.empty and "p95_sample_imputed_fraction" in imp else np.nan

    if blockers:
        status = "BLOCKED"
    elif primary_auprc_pos and primary_auprc_ci_pos and primary_recall_pos and np.isfinite(label_shift) and label_shift < 0.01:
        status = "PASS"
    elif primary_auprc_pos and primary_recall_pos:
        status = "PASS_WITH_NARROW_WORDING"
    else:
        status = "FAIL"

    for row in write_rows:
        row["manuscript_status"] = status
    write_csv(out_dir / "RESULTS_WRITE_IN_TABLE_CV_FALLBACK.csv", write_rows)

    if blockers:
        append_blockers(out_dir, blockers)
    else:
        write_csv(
            out_dir / "BLOCKERS_CV_FALLBACK.csv",
            [{"category": "cv_fallback", "item": "none", "status": "NONE", "details": "No blocking condition detected."}],
        )

    if status == "PASS":
        sentence = "Future-validity sensitivity using recent-valid-state constant-velocity fallback preserved the primary Waymo effect: strict temporal actionability dynamics improved detection of map-constrained critical-or-worse actionability beyond the strong geometry + CV baseline."
    elif status == "PASS_WITH_NARROW_WORDING":
        sentence = "Future-validity sensitivity using CV fallback preserved the primary Waymo effect, although CV fallback changed some endpoint assignments and should be presented as a sensitivity analysis."
    elif status == "FAIL":
        sentence = "CV-fallback future handling materially weakened or reversed the primary strict-temporal gain; the manuscript should not claim robustness to this endpoint-design choice."
    else:
        sentence = "CV-fallback future-handling sensitivity is blocked; the manuscript should not claim CV-fallback robustness until full labels and model evaluation are completed."

    claim = [
        "# CV-fallback Future-handling Claim Gate",
        "",
        f"Status: `{status}`",
        "",
        f"Full CV-fallback label rows: {len(labels) if not labels.empty else 'MISSING'}",
        f"Label changed fraction vs skip-invalid: {label_shift if np.isfinite(label_shift) else 'MISSING'}",
        f"P95 sample imputed fraction: {impute_p95 if np.isfinite(impute_p95) else 'MISSING'}",
        "",
        "Decision:",
        f"- Can the manuscript claim CV-fallback robustness? {'yes' if status == 'PASS' else 'narrow' if status == 'PASS_WITH_NARROW_WORDING' else 'no'}",
        f"- Manuscript sentence: {sentence}",
        "- Suggested location: endpoint-design robustness / sensitivity table and Methods future-handling paragraph.",
        "",
        "Primary endpoint seed-level results are in `RESULTS_WRITE_IN_TABLE_CV_FALLBACK.csv`.",
    ]
    (out_dir / "cv_fallback_claim_gate.md").write_text("\n".join(claim) + "\n", encoding="utf-8")

    (out_dir / "METHODS_PATCH_NOTES_CV_FALLBACK.md").write_text(
        "# Methods Patch Notes: CV-fallback sensitivity\n\n"
        "- Reference labels use `skip_invalid_oracle_future`: invalid non-ego future obstacle states are excluded from collision checks.\n"
        "- CV-fallback labels use `cv_fallback_invalid_future`: invalid non-ego future states are synthesized from the most recent valid future state at or before the evaluated time; if no future state is available, the current finite state is used; otherwise the slot remains non-imputable.\n"
        "- Position extrapolation uses constant velocity from `future_vel_xy` when available, else finite differencing of valid future positions, else current velocity.\n"
        "- Heading uses velocity direction above the configured speed threshold; otherwise it keeps the most recent valid heading.\n"
        "- Ego future states are not imputed. Candidate ego rollouts remain generated by the existing lightweight action rollout model.\n"
        "- CV-fallback label IDs are not used as model input features.\n",
        encoding="utf-8",
    )

    commands = [
        "conda run -n waymo_rt_bev python scripts/nc_v096/00_cv_fallback_inventory.py --config configs/nc_v096/nc_v096_cv_fallback.yaml",
        "conda run -n waymo_rt_bev python scripts/nc_v096/01_cv_fallback_relabel.py --config configs/nc_v096/nc_v096_cv_fallback.yaml --pilot --max-samples 500",
        "conda run -n waymo_rt_bev python scripts/nc_v096/01_cv_fallback_relabel.py --config configs/nc_v096/nc_v096_cv_fallback.yaml --full --n-jobs -1 --force",
        "conda run -n waymo_rt_bev python scripts/nc_v096/02_cv_fallback_oof_eval.py --config configs/nc_v096/nc_v096_cv_fallback.yaml --bootstrap-n 2000 --n-jobs -1 --force",
        "conda run -n waymo_rt_bev python scripts/nc_v096/03_cv_fallback_final_report.py --config configs/nc_v096/nc_v096_cv_fallback.yaml",
    ]
    report = [
        "# P0 CV-fallback Final Report",
        "",
        f"Status: `{status}`",
        f"Elapsed report-build seconds: {time.perf_counter() - t0:.1f}",
        "",
        "Commands:",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "Core files:",
        "- `labels_actionability_moderate_cv_fallback_full.csv`",
        "- `cv_fallback_label_shift_summary.csv`",
        "- `cv_fallback_imputation_summary.csv`",
        "- `cv_fallback_primary_oof_metrics.csv`",
        "- `cv_fallback_primary_paired_deltas.csv`",
        "- `RESULTS_WRITE_IN_TABLE_CV_FALLBACK.csv`",
        "- `cv_fallback_claim_gate.md`",
    ]
    (out_dir / "P0_CV_FALLBACK_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    zip_path = out_dir / "nc_v096_cv_fallback_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(out_dir))
    print(f"[cv-final] status={status} zip={zip_path}")


if __name__ == "__main__":
    main()
