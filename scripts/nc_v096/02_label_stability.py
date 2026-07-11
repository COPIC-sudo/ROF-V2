#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from _utils import load_yaml, output_dir, write_csv


LABEL_NAMES = {
    0: "high_actionability",
    1: "reduced_actionability",
    2: "critical_actionability",
    3: "infeasible_or_unavoidable",
}


def label_path(out_dir: Path, variant_id: str) -> Path:
    return out_dir / "variant_labels" / variant_id / f"labels_actionability_{variant_id}.csv"


def severe_jaccard(a: pd.Series, b: pd.Series) -> float:
    aa = set(a[a >= 2].index.astype(str))
    bb = set(b[b >= 2].index.astype(str))
    union = aa | bb
    return float(len(aa & bb) / len(union)) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v096/nc_v096_endpoint_design_robustness.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    variants = cfg["actionability_labels"]["variants"]
    ref_id = str(cfg["actionability_labels"]["reference_variant"])
    ref_path = label_path(out_dir, ref_id)
    if not ref_path.exists():
        raise FileNotFoundError(f"reference label CSV missing: {ref_path}")
    ref = pd.read_csv(ref_path)
    ref["sample_id"] = ref["sample_id"].astype(str)
    ref = ref.drop_duplicates("sample_id").set_index("sample_id")
    ref_label = pd.to_numeric(ref["actionability_label_id"], errors="coerce").astype(int)

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []

    for v in variants:
        vid = str(v["variant_id"])
        path = label_path(out_dir, vid)
        exists = path.exists()
        manifest_rows.append(
            {
                "variant_id": vid,
                "family": v.get("family", ""),
                "horizon_s": v["horizon_s"],
                "lane_buffer_m": v["lane_buffer_m"],
                "action_library": v["action_library"],
                "future_handling": v["future_handling"],
                "threshold_rule": cfg["actionability_labels"]["threshold_rule"],
                "map_constraint": cfg["actionability_labels"]["map_constraint"],
                "label_csv": str(path),
                "exists": bool(exists),
                "evaluation_feature_mode": cfg["evaluation"]["feature_mode"],
            }
        )
        if not exists:
            stability_rows.append({"variant_id": vid, "status": "MISSING_LABEL_CSV", "label_csv": str(path)})
            continue
        df = pd.read_csv(path)
        df["sample_id"] = df["sample_id"].astype(str)
        df = df.drop_duplicates("sample_id").set_index("sample_id")
        label = pd.to_numeric(df["actionability_label_id"], errors="coerce").astype(int)
        n = int(len(label))
        counts = label.value_counts().reindex([0, 1, 2, 3], fill_value=0)
        for lid, count in counts.items():
            summary_rows.append(
                {
                    "variant_id": vid,
                    "label_id": int(lid),
                    "label_name": LABEL_NAMES[int(lid)],
                    "count": int(count),
                    "fraction": float(count / max(n, 1)),
                    "n_samples": n,
                    "n_unique_sample_id": int(label.index.nunique()),
                }
            )
        common = sorted(set(ref_label.index.astype(str)) & set(label.index.astype(str)))
        r = ref_label.loc[common]
        x = label.loc[common]
        changed = r != x
        severe_ref = r >= 2
        severe_var = x >= 2
        kappa = cohen_kappa_score(r.to_numpy(int), x.to_numpy(int), weights="quadratic") if len(common) else np.nan
        stability = {
            "variant_id": vid,
            "reference_variant_id": ref_id,
            "family": v.get("family", ""),
            "n_common": int(len(common)),
            "n_variant": n,
            "n_unique_sample_id": int(label.index.nunique()),
            "critical_or_worse_prevalence": float(np.mean(x >= 2)) if len(common) else np.nan,
            "candidate_set_infeasible_prevalence": float(np.mean(x == 3)) if len(common) else np.nan,
            "reference_critical_or_worse_prevalence": float(np.mean(r >= 2)) if len(common) else np.nan,
            "severe_set_jaccard_vs_reference": severe_jaccard(r, x),
            "weighted_kappa_vs_reference": float(kappa) if np.isfinite(kappa) else np.nan,
            "label_changed_count": int(changed.sum()),
            "label_changed_fraction": float(np.mean(changed)) if len(common) else np.nan,
            "severe_added_count": int((~severe_ref & severe_var).sum()),
            "severe_removed_count": int((severe_ref & ~severe_var).sum()),
        }
        stability_rows.append(stability)
        ct = pd.crosstab(r, x)
        for ref_lid in [0, 1, 2, 3]:
            for var_lid in [0, 1, 2, 3]:
                transition_rows.append(
                    {
                        "variant_id": vid,
                        "reference_label_id": ref_lid,
                        "reference_label_name": LABEL_NAMES[ref_lid],
                        "variant_label_id": var_lid,
                        "variant_label_name": LABEL_NAMES[var_lid],
                        "count": int(ct.loc[ref_lid, var_lid]) if ref_lid in ct.index and var_lid in ct.columns else 0,
                    }
                )
        if str(v["future_handling"]) == "cv_fallback":
            for row in transition_rows:
                if row["variant_id"] == vid:
                    cv_rows.append(dict(row))
            cv_rows.append({**stability, "section": "cv_fallback_stability"})

    write_csv(out_dir / "waymo_design_variant_manifest.csv", manifest_rows)
    write_csv(out_dir / "waymo_design_variant_label_summary.csv", summary_rows)
    write_csv(out_dir / "waymo_design_variant_transition_matrix.csv", transition_rows)
    write_csv(out_dir / "waymo_design_variant_label_stability.csv", stability_rows)
    write_csv(out_dir / "future_validity_label_shift_skip_vs_cv.csv", cv_rows)
    print(f"[v096-stability] wrote label summaries to {out_dir}")


if __name__ == "__main__":
    main()
