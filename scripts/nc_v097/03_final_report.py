#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _utils import count_csv_rows, load_yaml, output_dir, resolve_path, sha256, write_csv


REQUIRED_OUTPUTS = [
    "RUN_PLAN_LOCKED_V097.md",
    "GPU_USAGE_REPORT_V097.md",
    "aligned_feature_variant_manifest.csv",
    "aligned_feature_lineage.csv",
    "aligned_feature_generation_audit.csv",
    "aligned_feature_checksum_manifest.csv",
    "aligned_feature_model_metrics.csv",
    "aligned_feature_calibrated_operating_points.csv",
    "aligned_feature_paired_deltas.csv",
    "aligned_feature_bootstrap_deltas.csv",
    "aligned_vs_reference_feature_comparison.csv",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def format_num(value: Any, digits: int = 4) -> str:
    val = finite_float(value)
    return "NA" if val is None else f"{val:.{digits}f}"


def build_results_table(out_dir: Path) -> pd.DataFrame:
    boot = safe_read_csv(out_dir / "aligned_feature_bootstrap_deltas.csv")
    if boot.empty:
        rows = [{"status": "MISSING", "notes": "aligned_feature_bootstrap_deltas.csv not found"}]
        write_csv(out_dir / "RESULTS_WRITE_IN_TABLE_V097.csv", rows)
        return pd.DataFrame(rows)
    rows = []
    for r in boot.to_dict("records"):
        rows.append(
            {
                "variant_id": r.get("variant_id", ""),
                "seed": r.get("seed", ""),
                "metric": r.get("metric", ""),
                "baseline_feature_set": r.get("baseline_feature_set", ""),
                "enhanced_feature_set": r.get("enhanced_feature_set", ""),
                "baseline_point": r.get("baseline_point", ""),
                "enhanced_point": r.get("enhanced_point", ""),
                "delta": r.get("delta", ""),
                "ci_low": r.get("ci_low", ""),
                "ci_high": r.get("ci_high", ""),
                "bootstrap_prob_delta_gt_0": r.get("bootstrap_prob_delta_gt_0", ""),
                "n_bootstrap_valid": r.get("n_bootstrap_valid", ""),
                "n_samples": r.get("n_samples", ""),
                "positive_count": r.get("positive_count", ""),
                "positive_rate": r.get("positive_rate", ""),
                "write_location": "Results" if r.get("metric") == "auprc" else "Supplementary",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "RESULTS_WRITE_IN_TABLE_V097.csv", index=False)
    return df


def evaluate_claims(cfg: dict[str, Any], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = list(cfg["aligned_features"]["variants"])
    audit = safe_read_csv(out_dir / "aligned_feature_generation_audit.csv")
    boot = safe_read_csv(out_dir / "aligned_feature_bootstrap_deltas.csv")
    blockers: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []

    missing = [name for name in REQUIRED_OUTPUTS if not (out_dir / name).exists()]
    for name in missing:
        blockers.append({"category": "missing_output", "item": name, "severity": "blocking", "notes": "required v0.9.7 output is absent"})

    if audit.empty:
        blockers.append({"category": "feature_generation", "item": "aligned_feature_generation_audit.csv", "severity": "blocking", "notes": "cannot verify aligned feature generation"})
        all_rows_complete = False
    else:
        row_map = {str(r.get("variant_id")): int(r.get("rows", 0)) for r in audit.to_dict("records")}
        all_rows_complete = all(row_map.get(v, 0) == 43098 for v in variants)
        for v in variants:
            if row_map.get(v, 0) != 43098:
                blockers.append({"category": "feature_generation", "item": v, "severity": "blocking", "notes": f"expected 43098 rows, got {row_map.get(v, 0)}"})

    primary_ok = False
    primary_mixed = False
    if boot.empty:
        blockers.append({"category": "bootstrap", "item": "aligned_feature_bootstrap_deltas.csv", "severity": "blocking", "notes": "bootstrap deltas missing"})
    else:
        primary = boot[boot["metric"].astype(str) == "auprc"].copy()
        if primary.empty:
            blockers.append({"category": "bootstrap", "item": "auprc", "severity": "blocking", "notes": "no AUPRC bootstrap rows"})
        else:
            primary_ok = bool(((primary["delta"] > 0) & (primary["ci_low"] > 0)).all())
            primary_mixed = bool(((primary["delta"] > 0) & (primary["ci_low"] > 0)).mean() >= 0.8)
            for row in primary.to_dict("records"):
                if not (finite_float(row.get("delta")) is not None and row["delta"] > 0 and finite_float(row.get("ci_low")) is not None and row["ci_low"] > 0):
                    blockers.append(
                        {
                            "category": "primary_metric",
                            "item": f"{row.get('variant_id')} seed={row.get('seed')}",
                            "severity": "claim_limiting",
                            "notes": f"AUPRC delta={row.get('delta')} ci_low={row.get('ci_low')}",
                        }
                    )

    future_noop = "future_h3_b3_base7_cvfallback" in variants
    if not missing and all_rows_complete and primary_ok and not future_noop:
        overall = "PASS"
        notes = "All design-dependent aligned features complete and all primary AUPRC CIs are above zero."
    elif not missing and all_rows_complete and primary_ok:
        overall = "PASS_WITH_NARROW_WORDING"
        notes = "Primary AUPRC passes; future-handling predictor is documented as a no-op because predictor features do not use observed future."
    elif primary_mixed and all_rows_complete:
        overall = "PASS_WITH_NARROW_WORDING"
        notes = "Most primary AUPRC deltas pass, but at least one row limits broad wording."
    elif boot.empty or audit.empty:
        overall = "BLOCKED"
        notes = "Required aligned feature or bootstrap outputs are missing."
    else:
        overall = "FAIL"
        notes = "At least one key aligned strict-temporal AUPRC delta is non-positive or its CI crosses zero."

    gates.append({"claim_gate": "aligned_label_feature_endpoint_design_robustness", "status": overall, "notes": notes})

    families = {
        "horizon_aligned_robustness": ["reference_h3_b3_base7_skip", "horizon_h2_b3_base7_skip", "horizon_h4_b3_base7_skip"],
        "lane_buffer_aligned_robustness": ["buffer_h3_b2_base7_skip", "buffer_h3_b4_base7_skip"],
        "action_library_aligned_robustness": ["action_h3_b3_extended_skip"],
        "future_handling_aligned_no_op_robustness": ["future_h3_b3_base7_cvfallback"],
        "low_fpr_operating_point_sensitivity": variants,
    }
    for gate, vids in families.items():
        if boot.empty:
            status = "BLOCKED"
            note = "bootstrap deltas missing"
        else:
            metric = "auprc" if gate != "low_fpr_operating_point_sensitivity" else "recall_at_5pct_fpr"
            sub = boot[(boot["variant_id"].isin(vids)) & (boot["metric"].astype(str) == metric)]
            if sub.empty:
                status = "BLOCKED"
                note = f"no {metric} rows found"
            elif ((sub["delta"] > 0) & (sub["ci_low"] > 0)).all():
                status = "PASS_WITH_NARROW_WORDING" if "future" in gate else "PASS"
                note = f"{metric} deltas positive with lower CI above zero"
            elif (sub["delta"] > 0).all():
                status = "PASS_WITH_NARROW_WORDING"
                note = f"{metric} point deltas positive but at least one CI crosses zero"
            else:
                status = "FAIL"
                note = f"at least one {metric} delta is non-positive"
        gates.append({"claim_gate": gate, "status": status, "notes": note})
    return gates, blockers


def write_markdown_reports(cfg: dict[str, Any], out_dir: Path, gates: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> None:
    audit = safe_read_csv(out_dir / "aligned_feature_generation_audit.csv")
    boot = safe_read_csv(out_dir / "aligned_feature_bootstrap_deltas.csv")
    comparison = safe_read_csv(out_dir / "aligned_vs_reference_feature_comparison.csv")
    results = safe_read_csv(out_dir / "RESULTS_WRITE_IN_TABLE_V097.csv")
    gate_map = {r["claim_gate"]: r["status"] for r in gates}

    methods = [
        "# Methods Patch Notes v0.9.7",
        "",
        "This analysis upgrades v0.9.6 from label-variant-only evaluation with reference features to an aligned label-and-feature endpoint-design sensitivity analysis where feasible.",
        "",
        "- Current-state strong-baseline fields are reused from the frozen reference feature table because they are independent of horizon, lane buffer, action library, and future-handling policy.",
        "- CV occupancy fields are regenerated for each variant horizon and lane-buffer setting using current-state constant-velocity occupancy.",
        "- Strict temporal dynamics are regenerated using endpoint candidate-action CV survival under the variant action library, horizon, and lane buffer.",
        "- Predictor features do not use observed future trajectories, actionability labels, or planner outcomes.",
        "- The future-handling variant is a predictor no-op: `cv_fallback` is a label-generation policy, not a predictor feature input.",
        "- Fixed-FPR thresholds are selected from calibration negatives with the `>=` alert operator and frozen for outer-test scoring.",
        "- Scenario-level paired bootstrap uses percentile CIs.",
        "",
        "Limitation: the original v0.9 primitive-survival fields are not rewritten in `pipeline.py`; v0.9.7 uses a new candidate-action CV-survival strict-temporal path to align the endpoint action-library design.",
    ]
    (out_dir / "METHODS_PATCH_NOTES_V097.md").write_text("\n".join(methods) + "\n", encoding="utf-8")

    figure = [
        "# Figure Update Recommendations v0.9.7",
        "",
        "- Main text: report aligned AUPRC deltas for `strong_baseline_cv_aligned` versus `strong_baseline_cv_plus_strict_temporal_dynamics_aligned`.",
        "- Supplement: keep per-variant, per-seed AUROC and low-FPR recall rows.",
        "- Do not merge v0.9.6 reference-feature rows into the aligned primary claim.",
        "- Use wording: `aligned label-and-feature endpoint-design sensitivity`.",
        "- Avoid wording: unavoidable collision, deployment-controlled FPR, fully ratio-free, proof of closed-loop safety.",
    ]
    (out_dir / "FIGURE_UPDATE_RECOMMENDATIONS_V097.md").write_text("\n".join(figure) + "\n", encoding="utf-8")

    top_results = []
    if not results.empty and "metric" in results:
        primary = results[results["metric"].astype(str) == "auprc"].copy()
        for r in primary.head(21).to_dict("records"):
            top_results.append(
                f"- {r.get('variant_id')} seed {r.get('seed')}: AUPRC delta {format_num(r.get('delta'))} "
                f"[{format_num(r.get('ci_low'))}, {format_num(r.get('ci_high'))}]"
            )
    if not top_results:
        top_results = ["- MISSING: aligned bootstrap AUPRC results were not available when the report was written."]

    complete_rows = "MISSING"
    if not audit.empty and "rows" in audit:
        complete_rows = str(int((audit["rows"] == 43098).sum())) + f"/{len(audit)} variants complete"

    changed = "MISSING"
    if not comparison.empty and "mean_abs_diff" in comparison:
        changed = "Aligned CV/temporal features differ from reference where horizon, lane buffer, or action library changes; see `aligned_vs_reference_feature_comparison.csv`."

    final = [
        "# P0 Aligned Feature Final Report v0.9.7",
        "",
        f"Overall claim gate: `{gate_map.get('aligned_label_feature_endpoint_design_robustness', 'MISSING')}`",
        "",
        "## 1. Regenerated vs Reused Features",
        "",
        f"Feature generation completeness: {complete_rows}.",
        "",
        "- Reused with checksum: current distance/TTC, ego speed, agent counts, relative/closing speed, and nearby-agent counts.",
        "- Regenerated: CV occupancy fields under variant horizon/lane buffer.",
        "- Regenerated: strict temporal dynamics under variant horizon/lane buffer/action library using candidate-action CV survival.",
        "- No-op with justification: future-handling predictor features, because the predictor path does not read observed future.",
        "",
        "## 2. Did Aligned Features Change v0.9.6 Conclusion?",
        "",
        changed,
        "",
        "Primary aligned AUPRC rows:",
        *top_results,
        "",
        "## 3. Manuscript Upgrade",
        "",
        "The manuscript can upgrade only if the overall gate is `PASS` or `PASS_WITH_NARROW_WORDING`. With narrow wording, describe the result as aligned label-and-feature endpoint-design sensitivity, not proof of closed-loop safety.",
        "",
        "## 4. Results vs Supplement",
        "",
        "- Results: AUPRC aligned deltas and claim-gate summary.",
        "- Supplement: per-seed AUROC, low-FPR recall, calibrated operating points, feature-lineage, and reference-feature comparison.",
        "",
        "## 5. Remaining Limitations",
        "",
        "- h4 extends beyond the original 3 s reference horizon and relies on regenerated aligned features plus existing v0.9.6 labels.",
        "- Extended-action strict-temporal features are regenerated with candidate-action CV survival, not by changing the original primitive library in `pipeline.py`.",
        "- Future-handling aligned predictor features are a documented no-op because predictor features are current-state/CV based.",
    ]
    if blockers:
        final.extend(["", "## Blockers", ""])
        final.extend([f"- {b.get('category')} / {b.get('item')}: {b.get('notes')}" for b in blockers])
    (out_dir / "P0_ALIGNED_FEATURE_FINAL_REPORT_V097.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def make_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "nc_v097_aligned_feature_robustness_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(out_dir))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v097/nc_v097_aligned_feature_robustness.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)

    results = build_results_table(out_dir)
    gates, blockers = evaluate_claims(cfg, out_dir)
    write_csv(out_dir / "CLAIM_GATE_REPORT_V097.csv", gates)
    if not blockers:
        blockers = [{"category": "none", "item": "none", "severity": "none", "notes": "No blocking issue recorded by final report script."}]
    write_csv(out_dir / "BLOCKERS_V097.csv", blockers)
    write_markdown_reports(cfg, out_dir, gates, blockers if blockers[0]["category"] != "none" else [])

    checksum_rows = []
    for name in REQUIRED_OUTPUTS + [
        "RESULTS_WRITE_IN_TABLE_V097.csv",
        "METHODS_PATCH_NOTES_V097.md",
        "FIGURE_UPDATE_RECOMMENDATIONS_V097.md",
        "CLAIM_GATE_REPORT_V097.csv",
        "BLOCKERS_V097.csv",
        "P0_ALIGNED_FEATURE_FINAL_REPORT_V097.md",
    ]:
        path = out_dir / name
        checksum_rows.append(
            {
                "artifact": name,
                "exists": path.exists(),
                "rows": count_csv_rows(path) if path.exists() and path.suffix.lower() == ".csv" else "",
                "sha256": sha256(path) if path.exists() and path.is_file() else "",
            }
        )
    existing_checksum = safe_read_csv(out_dir / "aligned_feature_checksum_manifest.csv")
    if not existing_checksum.empty:
        combined = pd.concat([existing_checksum, pd.DataFrame(checksum_rows)], ignore_index=True, sort=False)
        combined.to_csv(out_dir / "aligned_feature_checksum_manifest.csv", index=False)
    zip_path = make_zip(out_dir)
    print(f"[v097-final] wrote {out_dir}")
    print(f"[v097-final] results_rows={len(results)} zip={zip_path} zip_bytes={zip_path.stat().st_size}")


if __name__ == "__main__":
    main()
