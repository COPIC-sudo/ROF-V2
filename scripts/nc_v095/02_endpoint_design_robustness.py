#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from _utils import append_blockers, load_yaml, output_dir, resolve_path, sha256, write_csv


def file_entry(path: Path, variant: str, family: str, status: str, notes: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "family": family,
        "status": status,
        "source_path": str(path),
        "exists": path.exists(),
        "rows": max(sum(1 for _ in path.open("rb")) - 1, 0) if path.exists() and path.suffix.lower() == ".csv" else "",
        "sha256": sha256(path) if path.exists() and path.is_file() else "",
        "notes": notes,
    }


def label_counts(path: Path, label_col: str, variant: str) -> list[dict[str, Any]]:
    if not path.exists():
        return [{"variant": variant, "status": "MISSING", "source_path": str(path)}]
    df = pd.read_csv(path, usecols=[label_col])
    counts = df[label_col].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    names = {0: "high_actionability", 1: "reduced_actionability", 2: "critical_actionability", 3: "candidate_set_infeasible"}
    rows = []
    for label_id, count in counts.items():
        rows.append(
            {
                "variant": variant,
                "status": "AVAILABLE_EXISTING_FULL_LABELS",
                "label_id": int(label_id),
                "label_name": names[int(label_id)],
                "count": int(count),
                "fraction": float(count / len(df)),
                "source_path": str(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v095/nc_v095_p0_extension.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    v090_dir = resolve_path(cfg["inputs"]["v090_output_dir"])
    transition_out = out_dir / "label_transition_matrices_v095"
    transition_out.mkdir(parents=True, exist_ok=True)

    manifest = [
        file_entry(resolve_path(cfg["inputs"]["waymo_actionability_map_labels_csv"]), "moderate_h3_map", "existing_full_label", "AVAILABLE", "Original map-constrained full moderate labels; no regeneration."),
        file_entry(resolve_path(cfg["inputs"]["waymo_actionability_nomap_labels_csv"]), "moderate_h3_nomap", "existing_full_label", "AVAILABLE", "Original no-map full moderate labels; no regeneration."),
    ]
    blockers = []
    for h in cfg["actionability_labels"]["requested_horizon_variants_s"]:
        status = "AVAILABLE_EXISTING_REFERENCE" if float(h) == 3.0 else "BLOCKED_NOT_RUN"
        notes = "Reference existing full label horizon." if status.startswith("AVAILABLE") else "Requires full actionability label rollout regeneration; not run in v0.9.5 because the task forbids regenerating labels unless explicitly staged."
        manifest.append({"variant": f"horizon_{h:g}s", "family": "rollout_horizon_s", "status": status, "notes": notes})
    for b in cfg["actionability_labels"]["requested_lane_buffer_variants_m"]:
        status = "AVAILABLE_EXISTING_REFERENCE" if float(b) == 3.0 else "BLOCKED_NOT_RUN"
        notes = "Reference behavior from existing map-constrained label generator summary; exact historical map buffer depends on generator code/config." if status.startswith("AVAILABLE") else "Requires config-driven map-buffer label regeneration; not run."
        manifest.append({"variant": f"lane_buffer_{b:g}m", "family": "map_lane_buffer_m", "status": status, "notes": notes})
    for lib in cfg["actionability_labels"]["requested_action_libraries"]:
        status = "AVAILABLE_EXISTING_REFERENCE" if str(lib) == "base7" else "BLOCKED_NOT_RUN"
        notes = "Base-7 action library is the existing label-generator action set." if status.startswith("AVAILABLE") else "Extended action library would change candidate-action definition and requires a staged pilot/full rerun."
        manifest.append({"variant": str(lib), "family": "action_library", "status": status, "notes": notes})
    for handling in cfg["actionability_labels"]["requested_future_handling"]:
        status = "AVAILABLE_EXISTING_REFERENCE" if str(handling) == "skip_invalid" else "BLOCKED_NOT_RUN"
        notes = "Existing generator skips invalid oracle-future obstacle states." if status.startswith("AVAILABLE") else "CV fallback requires instrumented relabeling and model reevaluation; not run."
        manifest.append({"variant": str(handling), "family": "future_handling", "status": status, "notes": notes})
    write_csv(out_dir / "label_variant_manifest_v095.csv", manifest)
    if "envswitch" in out_dir.name.lower():
        write_csv(out_dir / "waymo_design_variant_manifest.csv", manifest)

    summary = []
    summary.extend(label_counts(resolve_path(cfg["inputs"]["waymo_actionability_map_labels_csv"]), "actionability_label_id", "moderate_h3_map"))
    summary.extend(label_counts(resolve_path(cfg["inputs"]["waymo_actionability_nomap_labels_csv"]), "actionability_label_id", "moderate_h3_nomap"))
    v090_summary = v090_dir / "label_robustness_summary.csv"
    if v090_summary.exists():
        old = pd.read_csv(v090_summary)
        old["v095_source"] = "v090_threshold_rule_reuse"
        old.to_csv(out_dir / "label_threshold_rule_robustness_reused_from_v090.csv", index=False)
        for row in old.to_dict("records"):
            row = dict(row)
            row["status"] = "REUSED_FROM_V090_THRESHOLD_RULE_SENSITIVITY"
            summary.append(row)
    write_csv(out_dir / "label_robustness_summary_v095.csv", summary)
    if "envswitch" in out_dir.name.lower():
        write_csv(out_dir / "waymo_design_variant_label_summary.csv", summary)

    old_transition_dir = v090_dir / "label_transition_matrices"
    if old_transition_dir.exists():
        transition_summary = []
        for path in old_transition_dir.glob("*.csv"):
            shutil.copy2(path, transition_out / path.name)
            transition_summary.append({"transition_matrix": path.name, "status": "REUSED_FROM_V090_THRESHOLD_RULE_SENSITIVITY", "source_path": str(path)})
        if "envswitch" in out_dir.name.lower():
            write_csv(out_dir / "waymo_design_variant_transition_summary.csv", transition_summary)
    else:
        write_csv(transition_out / "MISSING_transition_matrices.csv", [{"status": "MISSING", "source_dir": str(old_transition_dir)}])
        if "envswitch" in out_dir.name.lower():
            write_csv(out_dir / "waymo_design_variant_transition_summary.csv", [{"status": "MISSING", "source_dir": str(old_transition_dir)}])

    old_metrics = v090_dir / "waymo_robustness_model_metrics.csv"
    if old_metrics.exists():
        metrics = pd.read_csv(old_metrics)
        metrics["v095_status"] = "REUSED_FROM_V090_THRESHOLD_RULE_MODEL_ROBUSTNESS"
        metrics.to_csv(out_dir / "waymo_design_robustness_metrics_v095.csv", index=False)
        delta_like = metrics[metrics.get("metric", pd.Series([""] * len(metrics))).notna()].copy()
        delta_like.to_csv(out_dir / "waymo_design_robustness_deltas_v095.csv", index=False)
        if "envswitch" in out_dir.name.lower():
            metrics.to_csv(out_dir / "waymo_design_variant_model_metrics.csv", index=False)
            delta_like.to_csv(out_dir / "waymo_design_variant_paired_deltas.csv", index=False)
    else:
        write_csv(out_dir / "waymo_design_robustness_metrics_v095.csv", [{"status": "MISSING", "source_path": str(old_metrics)}])
        write_csv(out_dir / "waymo_design_robustness_deltas_v095.csv", [{"status": "MISSING", "source_path": str(old_metrics)}])
        if "envswitch" in out_dir.name.lower():
            write_csv(out_dir / "waymo_design_variant_model_metrics.csv", [{"status": "MISSING", "source_path": str(old_metrics)}])
            write_csv(out_dir / "waymo_design_variant_paired_deltas.csv", [{"status": "MISSING", "source_path": str(old_metrics)}])

    blocked_families = [r for r in manifest if str(r.get("status")) == "BLOCKED_NOT_RUN"]
    if blocked_families:
        blockers.append(
            {
                "category": "endpoint_design_robustness",
                "item": "full_horizon_buffer_action_future_variants",
                "status": "BLOCKED_NOT_RUN",
                "details": f"{len(blocked_families)} requested design variants require full relabeling/model reevaluation and were not executed.",
                "resume_command": "Implement config-driven v0.9.5 label generator variants, run small pilot, then full label/model evaluation under scripts/nc_v095.",
            }
        )
    append_blockers(out_dir, blockers)

    claim = [
        "# Waymo Design Robustness Claim Gate (v0.9.5)",
        "",
        "Status: `PASS_WITH_NARROW_WORDING` for threshold-rule sensitivity; `BLOCKED` for horizon/map-buffer/action-library/future-fallback design robustness.",
        "",
        "The v0.9.5 package reuses completed v0.9 threshold-rule robustness tables and existing full map/no-map moderate labels.",
        "It does not claim robustness to rollout horizon, map buffer, action-library expansion, or CV fallback handling because those variants were not regenerated and re-evaluated.",
    ]
    (out_dir / "waymo_design_robustness_claim_gate.md").write_text("\n".join(claim) + "\n", encoding="utf-8")
    if "envswitch" in out_dir.name.lower():
        (out_dir / "waymo_design_variant_claim_gate.md").write_text("\n".join(claim) + "\n", encoding="utf-8")
    print(f"[v095-robustness] wrote {out_dir}")


if __name__ == "__main__":
    main()
