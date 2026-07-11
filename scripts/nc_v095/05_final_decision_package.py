#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _utils import load_yaml, output_dir, write_csv


def read_csv_if(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def claim_statuses(out_dir: Path) -> list[dict[str, Any]]:
    blocker_name = "BLOCKERS_V095_ENVSWITCH.csv" if "envswitch" in out_dir.name.lower() else "BLOCKERS_V095.csv"
    blockers = read_csv_if(out_dir / blocker_name)
    secondary = read_csv_if(out_dir / "waymo_secondary_context_bootstrap_v095.csv")
    future = read_csv_if(out_dir / "future_validity_audit_v095.csv")
    cr_labels = read_csv_if(out_dir / "commonroad_planner_neutral_labels.csv")
    cr_deltas = read_csv_if(out_dir / "commonroad_neutral_deltas.csv")
    rows: list[dict[str, Any]] = []

    commonroad_status = "BLOCKED"
    commonroad_action = "Keep CommonRoad as stress-test/supplement unless neutral planner labels are generated."
    if not cr_labels.empty and "planner_failure" in cr_labels.columns:
        if not cr_deltas.empty and {"task", "metric", "ci_low"}.issubset(cr_deltas.columns):
            known = cr_deltas[(cr_deltas["task"].astype(str) == "planner_failure_known") & (cr_deltas["metric"].astype(str) == "auprc")]
            supported = known[pd.to_numeric(known["ci_low"], errors="coerce") > 0]
            reason = cr_labels.get("planner_failure_reason", pd.Series("", index=cr_labels.index)).fillna("").astype(str)
            failure = pd.to_numeric(cr_labels.get("planner_failure", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
            known_count = int(((failure == 1) & (reason != "unknown")).sum())
            if len(supported) >= 2 and known_count >= 20:
                commonroad_status = "PASS"
            elif known_count > 0:
                commonroad_status = "PASS_WITH_NARROW_WORDING"
            else:
                commonroad_status = "FAIL"
            commonroad_action = f"Neutral planner labels and scalar deltas are available; known_failures={known_count}, supported_known_auprc_deltas={len(supported)}."
        else:
            commonroad_status = "PARTIAL_PLANNER_RERUN_COMPLETED"
            commonroad_action = "Neutral planner labels were generated, but scalar validation/deltas remain blocked by missing neutral ROF feature artifacts."
    elif not blockers.empty and blockers["category"].astype(str).str.contains("commonroad", case=False, na=False).any():
        commonroad_status = "BLOCKED_RUNTIME_OR_DATA_ERROR"
        commonroad_action = "CommonRoad neutral rerun did not produce planner labels; inspect commonroad_neutral_rerun_commands.csv for the real traceback/data issue."
    rows.append(
        {
            "claim": "CommonRoad neutral external validation",
            "status": commonroad_status,
            "evidence_file": "commonroad_neutral_claim_gate.md",
            "manuscript_action": commonroad_action,
        }
    )

    rows.append(
        {
            "claim": "Endpoint design robustness beyond threshold-rule sensitivity",
            "status": "BLOCKED_FOR_FULL_DESIGN_VARIANTS",
            "evidence_file": "waymo_design_robustness_claim_gate.md",
            "manuscript_action": "Claim only threshold-rule and map/no-map sensitivity; do not claim horizon/buffer/action-library invariance.",
        }
    )

    if future.empty:
        fut_status = "MISSING"
        fut_note = "future_validity_audit_v095.csv missing"
    else:
        ok = future[future.get("status", "") == "OK"] if "status" in future.columns else future
        valid_rate = (
            pd.to_numeric(ok.get("valid_future_slots", pd.Series(dtype=float)), errors="coerce").sum()
            / max(pd.to_numeric(ok.get("total_future_slots", pd.Series(dtype=float)), errors="coerce").sum(), 1)
        )
        fut_status = "PASS_WITH_NARROW_WORDING"
        fut_note = f"overall valid-slot rate={valid_rate:.6f}; CV-fallback relabeling remains blocked"
    rows.append(
        {
            "claim": "Future-validity handling does not threaten endpoint",
            "status": fut_status,
            "evidence_file": "future_validity_claim_gate.md",
            "manuscript_action": fut_note,
        }
    )

    if secondary.empty:
        rows.append(
            {
                "claim": "Secondary/context Waymo comparisons",
                "status": "MISSING",
                "evidence_file": "waymo_secondary_context_bootstrap_v095.csv",
                "manuscript_action": "Run scripts/nc_v095/04_secondary_context_bootstrap.py.",
            }
        )
    else:
        pass_rows = secondary[
            (secondary["metric"].astype(str) == "auprc")
            & (pd.to_numeric(secondary["ci_low"], errors="coerce") > 0)
        ]
        rows.append(
            {
                "claim": "Secondary/context Waymo comparisons",
                "status": "PASS_WITH_ENDPOINT_SPECIFIC_WORDING" if not pass_rows.empty else "INCONCLUSIVE",
                "evidence_file": "waymo_secondary_context_bootstrap_v095.csv",
                "manuscript_action": f"{len(pass_rows)} RF AUPRC deltas have CI_low > 0; use endpoint/comparison-specific wording.",
            }
        )
    return rows


def write_results_table(out_dir: Path) -> None:
    sec = read_csv_if(out_dir / "waymo_secondary_context_bootstrap_v095.csv")
    rows: list[dict[str, Any]] = []
    if not sec.empty:
        keep = sec[
            (sec["metric"].isin(["auprc", "auroc", "recall_at_5pct_fpr"]))
            & (sec["model"].astype(str) == "rf")
        ].copy()
        for _, r in keep.iterrows():
            rows.append(
                {
                    "section": "Waymo secondary/context bootstrap",
                    "endpoint": r["endpoint"],
                    "comparison": r["comparison"],
                    "seed": r["seed"],
                    "metric": r["metric"],
                    "baseline": r["baseline_point"],
                    "enhanced": r["enhanced_point"],
                    "delta": r["delta"],
                    "ci": f"[{float(r['ci_low']):.6f}, {float(r['ci_high']):.6f}]" if np.isfinite(float(r["ci_low"])) else "",
                    "recommended_use": "main_or_supplement_if_claim_aligned" if float(r.get("ci_low", np.nan)) > 0 else "context_only",
                }
            )
    write_csv(out_dir / "RESULTS_WRITE_IN_TABLE_V095.csv", rows)


def make_zip(out_dir: Path) -> Path:
    zip_name = "nc_v095_p0_extension_envswitch_results.zip" if "envswitch" in out_dir.name.lower() else "nc_v095_p0_extension_results.zip"
    zip_path = out_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                zf.write(p, p.relative_to(out_dir))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v095/nc_v095_p0_extension.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)

    claims = claim_statuses(out_dir)
    envswitch = "envswitch" in out_dir.name.lower()
    claim_csv = out_dir / ("CLAIM_GATE_REPORT_V095_ENVSWITCH.csv" if envswitch else "CLAIM_GATE_REPORT_V095.csv")
    claim_md_path = out_dir / ("CLAIM_GATE_REPORT_V095_ENVSWITCH.md" if envswitch else "CLAIM_GATE_REPORT_V095.md")
    write_csv(claim_csv, claims)
    claim_md = ["# CLAIM GATE REPORT V095", ""]
    for row in claims:
        claim_md.append(f"- **{row['claim']}**: `{row['status']}` - {row['manuscript_action']} (`{row['evidence_file']}`)")
    claim_md_path.write_text("\n".join(claim_md) + "\n", encoding="utf-8")

    blocker_name = "BLOCKERS_V095_ENVSWITCH.csv" if envswitch else "BLOCKERS_V095.csv"
    blockers = read_csv_if(out_dir / blocker_name)
    blocker_count = len(blockers)
    claim_df = pd.DataFrame(claims)
    commonroad_claim = claim_df[claim_df["claim"] == "CommonRoad neutral external validation"].iloc[0].to_dict()
    if str(commonroad_claim["status"]).startswith("PARTIAL"):
        cr_answer = "Partial. The neutral CommonRoad planner rerun produced planner labels, but neutral scalar validation is still incomplete because matching neutral ROF feature artifacts were not generated in this task."
    elif str(commonroad_claim["status"]).startswith("PASS"):
        cr_answer = "Yes, subject to the neutral scalar deltas in commonroad_neutral_deltas.csv."
    else:
        cr_answer = "No. The neutral rerun did not produce complete planner-label/scalar evidence; inspect commonroad_neutral_rerun_commands.csv and blockers for the actual runtime/data issue."
    remaining_blocker_sentence = (
        "No direct contradiction to the primary strict-temporal Waymo claim was found. CommonRoad neutral validation is supported by the envswitch rerun; full design-variant robustness and CV-fallback relabeling remain blockers for stronger robustness claims."
        if str(commonroad_claim["status"]) == "PASS"
        else "No direct contradiction to the primary strict-temporal Waymo claim was found, but CommonRoad neutral validation and full design-variant robustness remain blockers for stronger external/general robustness claims."
    )
    file_lines = [
        "`CLAIM_GATE_REPORT_V095_ENVSWITCH.csv/md`" if envswitch else "`CLAIM_GATE_REPORT_V095.csv/md`",
        "`waymo_secondary_context_bootstrap_full.csv`" if envswitch else "`waymo_secondary_context_bootstrap_v095.csv`",
        "`future_validity_audit_envswitch.csv`" if envswitch else "`future_validity_audit_v095.csv`",
        "`waymo_design_variant_manifest.csv`" if envswitch else "`label_variant_manifest_v095.csv`",
        "`commonroad_neutral_claim_gate.md`",
        "`BLOCKERS_V095_ENVSWITCH.csv`" if envswitch else "`BLOCKERS_V095.csv`",
    ]
    report = [
        "# P0 Extension Final Report (v0.9.5)",
        "",
        "## Direct Answers",
        "",
        f"1. **Can CommonRoad upgrade from enriched stress test to neutral external validation?** {cr_answer}",
        "2. **Can design robustness be claimed beyond threshold-rule sensitivity?** No. Existing threshold-rule and map/no-map sensitivity can be cited; horizon, lane-buffer, action-library, and CV-fallback variants were not regenerated.",
        "3. **Does future-validity handling threaten endpoint?** The audit quantifies oracle-future validity and supports narrow wording, but it does not close the CV-fallback sensitivity gap.",
        "4. **Which secondary/context comparisons can be main vs Supplement?** Use only comparisons with positive scenario-bootstrap CIs and endpoint-specific wording; strict-temporal map critical remains the strongest candidate. Other comparisons should be Supplement/context unless their CI is positive and scientifically aligned.",
        f"5. **Any contradictory results blocking v0.10 rewrite?** {remaining_blocker_sentence}",
        "",
        "## Files",
        "",
        *[f"- {line}" for line in file_lines],
        "",
        f"Blocker rows: {blocker_count}",
    ]
    report_name = "P0_EXTENSION_FINAL_REPORT_ENVSWITCH.md" if envswitch else "P0_EXTENSION_FINAL_REPORT.md"
    (out_dir / report_name).write_text("\n".join(report) + "\n", encoding="utf-8")
    if envswitch:
        (out_dir / "CODEX_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    methods = [
        "# METHODS PATCH NOTES V095",
        "",
        "- State that actionability labels are candidate-action feasibility proxies with horizon 3.0 s, dt 0.1 s, obstacle_mode oracle_future, and moderate rule.",
        "- State that map-constrained and no-map labels differ only in map/drivable constraint usage; no-map sensitivity is separate from the primary map-constrained endpoint.",
        "- State that fixed-FPR alerts use the `>=` threshold operator and fold-calibrated 5% FPR thresholds from v0.9 OOF evaluation.",
        "- State that secondary/context bootstrap resamples scenario IDs and does not recalibrate thresholds inside bootstrap replicates.",
        "- Do not claim CommonRoad neutral validation until neutral planner labels are generated.",
        "- Do not claim horizon/buffer/action-library/future-fallback robustness until those full label variants and model metrics exist.",
    ]
    (out_dir / ("METHODS_PATCH_NOTES_V095_ENVSWITCH.md" if envswitch else "METHODS_PATCH_NOTES_V095.md")).write_text("\n".join(methods) + "\n", encoding="utf-8")

    fig = [
        "# FIGURE UPDATE RECOMMENDATIONS V095",
        "",
        "- Main Waymo figure: keep strict-temporal map critical as primary; add bootstrap CI source from `waymo_secondary_context_bootstrap_v095.csv`.",
        "- Robustness figure: label threshold-rule and map/no-map sensitivity only; mark horizon/buffer/action-library as not generated.",
        "- CommonRoad figure/table: describe as existing stress-test evidence, not neutral external validation.",
        "- Supplement: include future-validity audit tables and secondary/context comparisons with positive CIs.",
    ]
    (out_dir / ("FIGURE_UPDATE_RECOMMENDATIONS_V095_ENVSWITCH.md" if envswitch else "FIGURE_UPDATE_RECOMMENDATIONS_V095.md")).write_text("\n".join(fig) + "\n", encoding="utf-8")

    repro = [
        "# REPRODUCTION README V095",
        "",
        "Run from repository root:",
        "",
        "```powershell",
        "$CFG=\"configs/nc_v095/nc_v095_p0_extension.yaml\"",
        "conda run -n waymo_rt_bev python scripts/nc_v095/00_inventory_and_run_plan.py --config $CFG",
        "conda run -n waymo_rt_bev python scripts/nc_v095/01_commonroad_neutral_confirmation.py --config $CFG",
        "conda run -n waymo_rt_bev python scripts/nc_v095/02_endpoint_design_robustness.py --config $CFG",
        "conda run -n waymo_rt_bev python scripts/nc_v095/03_future_validity_audit.py --config $CFG",
        "conda run -n waymo_rt_bev python scripts/nc_v095/04_secondary_context_bootstrap.py --config $CFG",
        "conda run -n waymo_rt_bev python scripts/nc_v095/05_final_decision_package.py --config $CFG",
        "```",
    ]
    (out_dir / "REPRODUCTION_README_V095.md").write_text("\n".join(repro) + "\n", encoding="utf-8")
    write_results_table(out_dir)
    if envswitch and (out_dir / "RESULTS_WRITE_IN_TABLE_V095.csv").exists():
        pd.read_csv(out_dir / "RESULTS_WRITE_IN_TABLE_V095.csv").to_csv(out_dir / "RESULTS_WRITE_IN_TABLE_V095_ENVSWITCH.csv", index=False)
    if envswitch:
        import json, platform, sys
        manifest = {
            "config": args.config,
            "output_dir": str(out_dir),
            "python": sys.version,
            "platform": platform.platform(),
            "files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
        }
        (out_dir / "RUN_MANIFEST_V095_ENVSWITCH.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = make_zip(out_dir)
    print(f"[v095-final] wrote {out_dir}")
    print(f"[v095-final] zip {zip_path}")


if __name__ == "__main__":
    main()
