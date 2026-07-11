#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from _utils import append_blockers, load_yaml, output_dir, package_status, resolve_path, write_csv


def stable_unit_interval(text: str) -> float:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12 - 1)


def discover_xml_roots(cfg: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for key in ["commonroad_raw_root"]:
        value = cfg["inputs"].get(key)
        if value:
            roots.append(resolve_path(value))
    prior_manifest = resolve_path(cfg["inputs"].get("prior_v095_output_dir", "results/nc_v095_p0_extension")) / "commonroad_neutral_cohort_manifest.csv"
    if prior_manifest.exists():
        try:
            prior = pd.read_csv(prior_manifest, usecols=["xml_path"])
            for path in prior["xml_path"].dropna().astype(str).head(50):
                p = Path(path)
                if p.exists():
                    roots.append(p.parents[5] if len(p.parents) > 5 else p.parent)
        except Exception:
            pass
    unique: list[Path] = []
    seen = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def scan_commonroad_xml(root: Path, target_n: int) -> tuple[list[dict[str, Any]], int]:
    if not root.exists():
        return [], 0
    xmls = sorted([p for p in root.rglob("*.xml") if p.is_file()])
    rows = []
    for p in xmls:
        scenario_id = p.stem
        rel = str(p.relative_to(root)).replace("\\", "/")
        rows.append(
            {
                "sample_id": scenario_id,
                "commonroad_scenario_id": scenario_id,
                "scenario_id": scenario_id,
                "xml_path": str(p),
                "source_root": str(root),
                "xml_relative_path": rel,
                "selection_score": stable_unit_interval(rel),
                "selection_seed": 42,
                "neutral_sampling_rule": "sha1(xml_relative_path), outcome-blind; no actionability/ROF/planner labels",
                "previous_pilot_overlap_flag": False,
                "selection_source": "xml_path_hash_outcome_blind",
                "cohort_role": "neutral_candidate",
            }
        )
    rows = sorted(rows, key=lambda r: (r["selection_score"], r["xml_relative_path"]))[: int(target_n)]
    return rows, len(xmls)


def blocked_rows(kind: str, reason: str, resume_command: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "status": "BLOCKED_NOT_RUN",
            "artifact": kind,
            "reason": reason,
            "resume_command": resume_command or "Inspect commonroad_neutral_rerun_commands.csv, fix the reported runtime/data issue, and rerun scripts/nc_v095/01_commonroad_neutral_confirmation.py in commonroad_io.",
        }
    ]


def clear_commonroad_blockers(out_dir: Path) -> None:
    path = out_dir / ("BLOCKERS_V095_ENVSWITCH.csv" if "envswitch" in out_dir.name.lower() else "BLOCKERS_V095.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "category" not in df.columns:
        return
    keep = ~df["category"].astype(str).eq("commonroad_neutral_confirmation")
    write_csv(path, df[keep].to_dict("records"))


def run_subprocess(command: list[str], cwd: Path) -> dict[str, Any]:
    started = pd.Timestamp.utcnow().isoformat()
    try:
        cp = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        return {
            "started_at": started,
            "command": " ".join(command),
            "returncode": int(cp.returncode),
            "stdout_tail": cp.stdout[-6000:],
            "stderr_tail": cp.stderr[-6000:],
            "status": "OK" if cp.returncode == 0 else "FAILED",
        }
    except Exception as exc:
        return {
            "started_at": started,
            "command": " ".join(command),
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": traceback.format_exc(),
            "status": f"EXCEPTION_{type(exc).__name__}",
        }


def work_dir_from_base_config(cfg: dict[str, Any]) -> Path:
    base = resolve_path(cfg["inputs"]["base_config"])
    base_cfg = load_yaml(base)
    work_dir = (base_cfg.get("project") or {}).get("work_dir")
    if not work_dir:
        raise ValueError(f"project.work_dir missing from {base}")
    return Path(str(work_dir))


def summarize_planner_labels(label_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = pd.read_csv(label_path)
    labels["planner_success"] = pd.to_numeric(labels.get("planner_success", 0), errors="coerce").fillna(0).astype(int)
    labels["planner_failure"] = pd.to_numeric(labels.get("planner_failure", 0), errors="coerce").fillna(0).astype(int)
    reason_col = "planner_failure_reason" if "planner_failure_reason" in labels.columns else None
    metrics = [
        {"metric": "neutral_planner_label_rows", "value": int(len(labels)), "status": "PLANNER_RERUN_COMPLETED"},
        {"metric": "neutral_planner_success_count", "value": int(labels["planner_success"].sum()), "status": "PLANNER_RERUN_COMPLETED"},
        {"metric": "neutral_planner_failure_count", "value": int(labels["planner_failure"].sum()), "status": "PLANNER_RERUN_COMPLETED"},
        {"metric": "neutral_planner_failure_rate", "value": float(labels["planner_failure"].mean()) if len(labels) else float("nan"), "status": "PLANNER_RERUN_COMPLETED"},
        {"metric": "neutral_unique_scenario_count", "value": int(labels.get("commonroad_scenario_id", labels["sample_id"]).astype(str).nunique()), "status": "PLANNER_RERUN_COMPLETED"},
    ]
    taxonomy = []
    if reason_col:
        for reason, count in labels[reason_col].fillna("missing").astype(str).value_counts().items():
            taxonomy.append({"failure_reason": reason, "count": int(count), "fraction": float(count / max(len(labels), 1))})
    else:
        taxonomy.append({"failure_reason": "MISSING_REASON_COLUMN", "count": int(len(labels)), "fraction": 1.0})
    return metrics, taxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v095/nc_v095_p0_extension.yaml")
    parser.add_argument("--skip-rerun", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    clear_commonroad_blockers(out_dir)
    repo = Path.cwd()

    roots = discover_xml_roots(cfg)
    root_inventory = []
    for root in roots:
        try:
            count = len([p for p in root.rglob("*.xml") if p.is_file()]) if root.exists() else 0
        except Exception as exc:
            count = 0
            root_inventory.append({"source_root": str(root), "exists": root.exists(), "xml_count": count, "status": f"ERROR_{type(exc).__name__}: {exc}"})
            continue
        root_inventory.append({"source_root": str(root), "exists": root.exists(), "xml_count": count, "status": "OK" if count else "NO_XML"})
    write_csv(out_dir / "commonroad_xml_root_inventory.csv", root_inventory)

    raw_root = roots[0] if roots else resolve_path(cfg["inputs"]["commonroad_raw_root"])
    target_n = int(cfg["commonroad"]["neutral_target_n"])
    manifest, xml_count = scan_commonroad_xml(raw_root, target_n)
    if manifest:
        write_csv(out_dir / "commonroad_neutral_cohort_manifest.csv", manifest)
        write_csv(out_dir / "commonroad_neutral_candidate_manifest_envswitch.csv", manifest)
    else:
        write_csv(out_dir / "commonroad_neutral_cohort_manifest.csv", blocked_rows("neutral_cohort_manifest", f"CommonRoad XML root missing or empty: {raw_root}"))

    commonroad_pkg = package_status("commonroad")
    commonroad_io_pkg = package_status("commonroad_io")
    try:
        from commonroad.common.file_reader import CommonRoadFileReader  # noqa: F401

        reader_available = True
        reader_error = ""
    except Exception as exc:
        reader_available = False
        reader_error = traceback.format_exc()
    planner_available = reader_available
    reason = (
        f"xml_root={raw_root}; xml_count={xml_count}; "
        f"commonroad={commonroad_pkg['status']}; commonroad_io={commonroad_io_pkg['status']}; "
        f"CommonRoadFileReader_available={reader_available}; "
        "this script does not fabricate planner outcomes from existing enriched pilot1000 results."
    )

    protocol = [
        "# CommonRoad Neutral Sampling Protocol (v0.9.5)",
        "",
        "The neutral candidate cohort is selected without reading planner labels, scalar scores, or failure outcomes.",
        "When raw XML files are visible, XML relative paths are sorted by a deterministic SHA1-derived unit-interval score and the first target-N scenarios are written as candidates.",
        "This only defines an outcome-blind candidate set. Planner labels and scalar metrics require CommonRoad parsing and the independent lattice-planner backend.",
        "",
        f"- Raw XML root: `{raw_root}`",
        f"- XML files found: {xml_count}",
        f"- Target neutral candidate count: {target_n}",
        f"- CommonRoad package status: {commonroad_pkg['status']}",
        f"- commonroad_io package status: {commonroad_io_pkg['status']}",
        f"- CommonRoadFileReader available in this conda environment: {reader_available}",
        "",
        "Existing pilot1000 CommonRoad planner results are not reused as neutral-confirmatory evidence because that cohort was previously selected as a dynamic-ego stress test.",
    ]
    (out_dir / "commonroad_neutral_sampling_protocol.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")

    command_rows: list[dict[str, Any]] = []
    label_path: Path | None = None
    export_manifest: Path | None = None
    if planner_available and not args.skip_rerun and manifest:
        work_dir = work_dir_from_base_config(cfg)
        sample_out_name = str(cfg.get("commonroad", {}).get("neutral_sample_out_name", "commonroad_neutral_v095"))
        planner_out_name = str(cfg.get("commonroad", {}).get("neutral_planner_out_name", "neutral_v095"))
        export_dir = work_dir / "results" / "commonroad_samples" / sample_out_name
        planner_dir = work_dir / "results" / "commonroad_planner_feasibility" / planner_out_name
        export_manifest = export_dir / "commonroad_dynamic_ego_samples_manifest.csv"
        label_path = planner_dir / f"commonroad_lattice_planner_labels_{planner_out_name}.csv"
        export_cmd = [
            sys.executable,
            "scripts/43b_export_commonroad_dynamic_ego_samples.py",
            "--config",
            str(resolve_path(cfg["inputs"]["base_config"])),
            "--pilot-csv",
            str(out_dir / "commonroad_neutral_cohort_manifest.csv"),
            "--out-name",
            sample_out_name,
            "--max-samples",
            str(target_n),
            "--scenario-limit",
            str(target_n),
            "--time-stride",
            "10",
            "--horizon-steps",
            "30",
            "--min-neighbor-agents",
            "3",
            "--min-ego-speed-mps",
            "0.5",
            "--seed",
            "42",
        ]
        command_rows.append({"stage": "export_neutral_dynamic_ego_samples", **run_subprocess(export_cmd, repo)})
        if command_rows[-1]["returncode"] == 0 and export_manifest.exists():
            exported = pd.read_csv(export_manifest)
            exported_ok = exported[exported.get("export_status", "ok").fillna("ok").astype(str) == "ok"].copy()
            sample_size = min(target_n, int(len(exported_ok)))
            planner_cmd = [
                sys.executable,
                "scripts/51_commonroad_lattice_planner_feasibility.py",
                "--config",
                str(resolve_path(cfg["inputs"]["base_config"])),
                "--samples-dir",
                str(export_dir / "samples_json_gz"),
                "--manifest-csv",
                str(export_manifest),
                "--out-name",
                planner_out_name,
                "--sample-size",
                str(sample_size),
                "--seed",
                "42",
                "--horizon-s",
                "3.0",
                "--dt-s",
                "0.1",
            ]
            command_rows.append({"stage": "neutral_lattice_planner", **run_subprocess(planner_cmd, repo)})
        else:
            command_rows.append(
                {
                    "stage": "neutral_lattice_planner",
                    "command": "SKIPPED",
                    "returncode": "",
                    "status": "SKIPPED_EXPORT_FAILED_OR_NO_MANIFEST",
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
    write_csv(out_dir / "commonroad_neutral_rerun_commands.csv", command_rows)

    if not planner_available:
        resume = "Run this script inside the conda environment that can import commonroad.common.file_reader.CommonRoadFileReader."
        write_csv(out_dir / "commonroad_planner_neutral_labels.csv", blocked_rows("planner_labels", reason, resume))
        write_csv(out_dir / "commonroad_neutral_failure_taxonomy.csv", blocked_rows("failure_taxonomy", reason, resume))
        write_csv(out_dir / "commonroad_neutral_metrics.csv", blocked_rows("neutral_metrics", reason, resume))
        write_csv(out_dir / "commonroad_neutral_deltas.csv", blocked_rows("neutral_deltas", reason, resume))
        write_csv(out_dir / "commonroad_neutral_scenario_bootstrap.csv", blocked_rows("scenario_bootstrap", reason, resume))
        append_blockers(
            out_dir,
            [
                {
                    "category": "commonroad_neutral_confirmation",
                    "item": "planner_rerun",
                    "status": "BLOCKED",
                    "details": reason + ("; traceback=" + reader_error[-1000:] if reader_error else ""),
                    "resume_command": "Run this script inside the conda environment that can import commonroad.common.file_reader.CommonRoadFileReader.",
                }
            ],
        )
        gate_status = "BLOCKED"
    else:
        if label_path is not None and label_path.exists():
            labels = pd.read_csv(label_path)
            labels.to_csv(out_dir / "commonroad_planner_neutral_labels.csv", index=False)
            metrics, taxonomy = summarize_planner_labels(label_path)
            if export_manifest is not None and export_manifest.exists():
                exported = pd.read_csv(export_manifest)
                metrics.extend(
                    [
                        {"metric": "neutral_export_manifest_rows", "value": int(len(exported)), "status": "PLANNER_RERUN_COMPLETED"},
                        {"metric": "neutral_export_unique_scenarios", "value": int(exported.get("commonroad_scenario_id", pd.Series(dtype=str)).astype(str).nunique()), "status": "PLANNER_RERUN_COMPLETED"},
                    ]
                )
            metrics.append(
                {
                    "metric": "scalar_validation_status",
                    "value": "BLOCKED_MISSING_NEUTRAL_ROF_FEATURES",
                    "status": "PARTIAL",
                    "notes": "Neutral planner labels were generated, but this task forbids regenerating ROF features; scalar metrics/deltas need a neutral feature CSV matching these sample_id values.",
                }
            )
            write_csv(out_dir / "commonroad_neutral_metrics.csv", metrics)
            write_csv(out_dir / "commonroad_neutral_failure_taxonomy.csv", taxonomy)
            missing_feature_reason = "Neutral planner labels exist, but no neutral ROF feature CSV was generated in this task; scalar validation/deltas require matching neutral sample features."
            resume = "Generate neutral CommonRoad ROF scalar features for commonroad_planner_neutral_labels.csv sample_id values, then run scalar metrics/bootstrap."
            write_csv(out_dir / "commonroad_neutral_deltas.csv", blocked_rows("neutral_deltas", missing_feature_reason, resume))
            write_csv(out_dir / "commonroad_neutral_scenario_bootstrap.csv", blocked_rows("scenario_bootstrap", missing_feature_reason, resume))
            append_blockers(
                out_dir,
                [
                    {
                        "category": "commonroad_neutral_confirmation",
                        "item": "neutral_scalar_features",
                        "status": "PARTIAL_BLOCKED",
                        "details": missing_feature_reason,
                        "resume_command": "Generate neutral CommonRoad ROF scalar features for commonroad_planner_neutral_labels.csv sample_id values, then run scalar metrics/bootstrap.",
                    }
                ],
            )
            gate_status = "PARTIAL_PLANNER_RERUN_COMPLETED"
        else:
            failed_detail = "CommonRoadFileReader import succeeded, but neutral planner label output was not produced."
            failed_cmd = next((r for r in command_rows if str(r.get("status")) not in {"OK", "SKIPPED_EXPORT_FAILED_OR_NO_MANIFEST"}), command_rows[-1] if command_rows else {})
            if failed_cmd:
                failed_detail += " Failed stage: " + str(failed_cmd.get("stage", "")) + ". Stderr tail: " + str(failed_cmd.get("stderr_tail", ""))[-1800:]
            resume = "Inspect commonroad_neutral_rerun_commands.csv stderr_tail and rerun the failed command in commonroad_io with write access to project.work_dir."
            write_csv(out_dir / "commonroad_planner_neutral_labels.csv", blocked_rows("planner_labels", failed_detail, resume))
            write_csv(out_dir / "commonroad_neutral_failure_taxonomy.csv", blocked_rows("failure_taxonomy", failed_detail, resume))
            write_csv(out_dir / "commonroad_neutral_metrics.csv", blocked_rows("neutral_metrics", failed_detail, resume))
            write_csv(out_dir / "commonroad_neutral_deltas.csv", blocked_rows("neutral_deltas", failed_detail, resume))
            write_csv(out_dir / "commonroad_neutral_scenario_bootstrap.csv", blocked_rows("scenario_bootstrap", failed_detail, resume))
            append_blockers(
                out_dir,
                [
                    {
                        "category": "commonroad_neutral_confirmation",
                        "item": "planner_rerun_runtime",
                        "status": "BLOCKED_RUNTIME_ERROR",
                        "details": failed_detail,
                        "resume_command": "Inspect commonroad_neutral_rerun_commands.csv stderr/stdout tails and rerun the failing command in commonroad_io.",
                    }
                ],
            )
            gate_status = "BLOCKED_RUNTIME_ERROR"

    claim = [
        "# CommonRoad Neutral Claim Gate",
        "",
        f"Status: `{gate_status}`",
        "",
        "The v0.9.5 run defines an outcome-blind neutral candidate manifest and attempts the independent CommonRoad dynamic-ego export plus lattice-planner rerun in the current conda environment.",
        "",
        "Do not cite the old pilot1000 stress-test metrics as neutral confirmation. If this gate is PARTIAL, neutral planner labels exist but scalar validation still needs matching neutral ROF features and bootstrap deltas.",
    ]
    (out_dir / "commonroad_neutral_claim_gate.md").write_text("\n".join(claim) + "\n", encoding="utf-8")
    print(f"[v095-commonroad] xml_count={xml_count} planner_available={planner_available} wrote {out_dir}")


if __name__ == "__main__":
    main()
