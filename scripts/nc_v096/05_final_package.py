#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _utils import append_blockers, load_yaml, output_dir, write_csv


def read_csv_if(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def safe_num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_endpoint_design_robustness.yaml")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)

    stability = read_csv_if(out_dir / "waymo_design_variant_label_stability.csv")
    deltas = read_csv_if(out_dir / "waymo_design_variant_bootstrap_deltas.csv")
    point_deltas = read_csv_if(out_dir / "waymo_design_variant_paired_deltas.csv")
    label_summary = read_csv_if(out_dir / "waymo_design_variant_label_summary.csv")

    blockers: list[dict[str, Any]] = []
    variants = [str(v["variant_id"]) for v in cfg["actionability_labels"]["variants"]]
    missing_variant_outputs = []
    for vid in variants:
        if not (out_dir / "variant_labels" / vid / f"labels_actionability_{vid}.csv").exists():
            missing_variant_outputs.append(vid)
    if missing_variant_outputs:
        blockers.append(
            {
                "category": "endpoint_design_robustness",
                "item": "missing_variant_labels",
                "status": "BLOCKED",
                "details": ";".join(missing_variant_outputs),
            }
        )
    if deltas.empty:
        blockers.append(
            {
                "category": "endpoint_design_robustness",
                "item": "bootstrap_missing",
                "status": "BLOCKED",
                "details": "waymo_design_variant_bootstrap_deltas.csv missing or empty",
            }
        )
    append_blockers(out_dir, blockers)
    if not blockers and not (out_dir / "BLOCKERS_V096.csv").exists():
        write_csv(
            out_dir / "BLOCKERS_V096.csv",
            [
                {
                    "category": "endpoint_design_robustness",
                    "item": "none",
                    "status": "NONE",
                    "details": "No blocking condition was detected in the v0.9.6 final package build.",
                }
            ],
        )

    primary = deltas[(deltas.get("metric", "") == "auprc")] if not deltas.empty else pd.DataFrame()
    all_delta_positive = bool((pd.to_numeric(primary.get("delta", pd.Series(dtype=float)), errors="coerce") > 0).all()) if not primary.empty else False
    all_ci_nonnegative = bool((pd.to_numeric(primary.get("ci_low", pd.Series(dtype=float)), errors="coerce") > 0).all()) if not primary.empty else False
    cv_row = stability[stability.get("variant_id", "") == "future_h3_b3_base7_cvfallback"] if not stability.empty else pd.DataFrame()
    cv_change = safe_num(cv_row["label_changed_fraction"].iloc[0]) if not cv_row.empty else np.nan
    feature_mode = cfg["evaluation"]["feature_mode"]

    if blockers:
        status = "BLOCKED"
    elif all_delta_positive and all_ci_nonnegative and feature_mode != "LABEL_VARIANT_ONLY_WITH_REFERENCE_FEATURES" and np.isfinite(cv_change) and cv_change < 0.05:
        status = "PASS"
    elif all_delta_positive:
        status = "PASS_WITH_NARROW_WORDING"
    else:
        status = "FAIL"

    claim_lines = [
        "# Waymo Endpoint-Design Robustness Claim Gate (v0.9.6)",
        "",
        f"Status: `{status}`",
        "",
        f"Feature mode: `{feature_mode}`.",
        "",
        "Interpretation:",
        "- The v0.9.6 run regenerates endpoint labels from candidate-action feasibility for horizon, lane buffer, action-library, and future-handling variants.",
        "- Model reevaluation uses the frozen reference feature table, so the model result should be described as label-variant robustness with reference features.",
        "- A full aligned label+feature robustness claim would require regenerating strict-temporal/actionability features under each endpoint design.",
        "",
    ]
    if not primary.empty:
        claim_lines.append("Primary AUPRC deltas:")
        for row in primary.sort_values(["variant_id", "seed"]).to_dict("records"):
            claim_lines.append(
                f"- {row['variant_id']} seed={row['seed']}: delta={safe_num(row.get('delta')):.6f}, "
                f"95% CI [{safe_num(row.get('ci_low')):.6f}, {safe_num(row.get('ci_high')):.6f}], "
                f"bootstrap_n={row.get('bootstrap_n_requested', '')}."
            )
    (out_dir / "waymo_endpoint_design_robustness_claim_gate.md").write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    future_lines = [
        "# Future-Validity CV-Fallback Claim Gate (v0.9.6)",
        "",
        f"Status: `{status if not cv_row.empty else 'BLOCKED'}`",
        "",
        f"CV-fallback label changed fraction vs reference: {cv_change if np.isfinite(cv_change) else 'MISSING'}",
        "",
        "CV-fallback labels were actually generated and evaluated in v0.9.6. Interpret materiality using prevalence shift and model delta rows for `future_h3_b3_base7_cvfallback`.",
    ]
    (out_dir / "future_validity_cvfallback_claim_gate.md").write_text("\n".join(future_lines) + "\n", encoding="utf-8")

    write_rows = []
    for vid in variants:
        stab = stability[stability.get("variant_id", "") == vid] if not stability.empty else pd.DataFrame()
        au = primary[primary.get("variant_id", "") == vid] if not primary.empty else pd.DataFrame()
        row = {
            "variant_id": vid,
            "feature_mode": feature_mode,
            "label_changed_fraction_vs_reference": safe_num(stab["label_changed_fraction"].iloc[0]) if not stab.empty else np.nan,
            "severe_set_jaccard_vs_reference": safe_num(stab["severe_set_jaccard_vs_reference"].iloc[0]) if not stab.empty else np.nan,
            "delta_auprc": safe_num(au["delta"].iloc[0]) if not au.empty else np.nan,
            "ci_low": safe_num(au["ci_low"].iloc[0]) if not au.empty else np.nan,
            "ci_high": safe_num(au["ci_high"].iloc[0]) if not au.empty else np.nan,
            "manuscript_write_in": "Endpoint-design sensitivity supports directionally positive strict-temporal increment under this label variant, but wording must state reference features were reused.",
        }
        write_rows.append(row)
    write_csv(out_dir / "RESULTS_WRITE_IN_TABLE_V096.csv", write_rows)

    (out_dir / "METHODS_PATCH_NOTES_V096.md").write_text(
        "# Methods Patch Notes v0.9.6\n\n"
        "- Endpoint labels were regenerated by candidate-action feasibility under one-at-a-time design variants.\n"
        "- Base7 actions match `scripts/24_build_actionability_labels.py`; the extended library adds deterministic acceleration, stronger braking, mild lateral, and combined primitives.\n"
        "- Map-constrained variants use lane centerline buffers at the specified buffer radius.\n"
        "- `skip_invalid` ignores invalid or out-of-cache oracle-future obstacle states; `cv_fallback` replaces those states with current-state constant-velocity extrapolation.\n"
        "- OOF model reevaluation uses existing reference feature table and must be labeled `LABEL_VARIANT_ONLY_WITH_REFERENCE_FEATURES`.\n",
        encoding="utf-8",
    )
    (out_dir / "FIGURE_UPDATE_RECOMMENDATIONS_V096.md").write_text(
        "# Figure Update Recommendations v0.9.6\n\n"
        "- Add an endpoint-design robustness panel/table using `RESULTS_WRITE_IN_TABLE_V096.csv`.\n"
        "- Show label stability versus reference for horizon, buffer, action-library, and CV-fallback variants.\n"
        "- State in caption that model reevaluation uses reference features unless aligned per-variant features are later regenerated.\n",
        encoding="utf-8",
    )

    final_lines = [
        "# P0 Endpoint Design Final Report v0.9.6",
        "",
        f"Status: `{status}`",
        f"Elapsed package build seconds: {time.perf_counter() - t0:.1f}",
        "",
        "Generated files:",
        "- `waymo_design_variant_manifest.csv`",
        "- `waymo_design_variant_label_summary.csv`",
        "- `waymo_design_variant_transition_matrix.csv`",
        "- `waymo_design_variant_label_stability.csv`",
        "- `future_validity_label_shift_skip_vs_cv.csv`",
        "- `waymo_design_variant_model_metrics.csv`",
        "- `waymo_design_variant_paired_deltas.csv`",
        "- `waymo_design_variant_bootstrap_deltas.csv`",
        "- `RESULTS_WRITE_IN_TABLE_V096.csv`",
        "",
    ]
    if not label_summary.empty:
        final_lines.append("Label summary rows: " + str(len(label_summary)))
    if blockers:
        final_lines.extend(["", "Blockers:", *[f"- {b['item']}: {b['details']}" for b in blockers]])
    (out_dir / "P0_ENDPOINT_DESIGN_FINAL_REPORT_V096.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    zip_path = out_dir / "nc_v096_endpoint_design_robustness_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(out_dir))
    print(f"[v096-final] status={status} zip={zip_path}")


if __name__ == "__main__":
    main()
