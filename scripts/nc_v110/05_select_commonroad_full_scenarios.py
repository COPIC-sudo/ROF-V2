#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.common import add_config_hash, config_hash, load_yaml_config, require_work_dir, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-name", default="nc_v110_full_fixed_taxonomy_scenarios.csv")
    parser.add_argument("--target-scenarios", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _stable_float(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _bucket(values: pd.Series, edges: list[float], labels: list[str], missing: str) -> pd.Series:
    out = pd.cut(values, [-float("inf"), *edges, float("inf")], labels=labels).astype("object")
    out[pd.isna(out)] = missing
    return out.astype(str)


def _add_neutral_strata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_hint"] = out.get("source_hint", "unknown").fillna("unknown").astype(str)
    out["scenario_family"] = out.get("scenario_family", "unknown").fillna("unknown").astype(str)
    out["lanelet_stratum"] = _bucket(_num(out, "lanelet_count"), [20, 60], ["lt20_lanelets", "20to60_lanelets", "gte60_lanelets"], "missing_lanelets")
    out["density_stratum"] = _bucket(_num(out, "dynamic_obstacle_count"), [5, 15], ["lt5_dyn", "5to15_dyn", "gte15_dyn"], "missing_density")
    out["horizon_stratum"] = _bucket(_num(out, "max_time_step"), [30, 90], ["lt30_steps", "30to90_steps", "gte90_steps"], "missing_horizon")
    out["neutral_scenario_stratum"] = (
        out["source_hint"]
        + "|"
        + out["scenario_family"]
        + "|"
        + out["lanelet_stratum"]
        + "|"
        + out["density_stratum"]
        + "|"
        + out["horizon_stratum"]
    )
    return out


def _filter_commonroad_scenario_xml(manifest: pd.DataFrame, min_max_time_step: int) -> pd.DataFrame:
    """Keep scenario XML files and drop SUMO helper XML files from extracted archives."""
    out = manifest.copy()
    if "parse_status" in out.columns:
        out = out[out["parse_status"].fillna("").astype(str).str.lower().eq("ok")].copy()
    if "xml_path" not in out.columns:
        return out

    xml_path = out["xml_path"].fillna("").astype(str)
    base = xml_path.map(lambda s: Path(s).name.lower())
    rel = out.get("xml_rel_path", xml_path).fillna("").astype(str).str.replace("\\", "/", regex=False).str.lower()
    aux_suffix = (
        base.str.endswith(".net.xml")
        | base.str.endswith(".rou.xml")
        | base.str.endswith(".add.xml")
        | base.str.endswith(".poly.xml")
        | base.str.endswith(".sumocfg")
        | base.isin({"edges.net.xml", "nodes.net.xml", "_connections.net.xml", "_tll.net.xml"})
        | rel.str.contains("/sumo/", regex=False)
    )
    planning = _num(out, "planning_problem_count").fillna(0)
    lanelets = _num(out, "lanelet_count").fillna(0)
    dynamics = _num(out, "dynamic_obstacle_count").fillna(0)
    max_time_step = _num(out, "max_time_step").fillna(0)
    scenario_like = (
        (~aux_suffix)
        & (planning >= 1)
        & (lanelets >= 1)
        & (dynamics >= 1)
        & (max_time_step >= int(min_max_time_step))
    )
    return out[scenario_like].copy()


def _select_round_robin(rows: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    work = _add_neutral_strata(rows)
    work["_score"] = work["scenario_id"].astype(str).map(lambda s: _stable_float(f"{seed}|{s}"))
    work = work.sort_values(["neutral_scenario_stratum", "_score", "scenario_id"]).reset_index(drop=True)
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, row in work.iterrows():
        buckets[str(row["neutral_scenario_stratum"])].append(int(idx))
    bucket_keys = sorted(buckets, key=lambda k: _stable_float(f"{seed}|bucket|{k}"))
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < target and bucket_keys:
        next_keys: list[str] = []
        for key in bucket_keys:
            if len(selected) >= target:
                break
            if not buckets[key]:
                continue
            idx = buckets[key].pop(0)
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if buckets[key]:
                next_keys.append(key)
        bucket_keys = next_keys
    out = work.iloc[selected].copy().reset_index(drop=True)
    out["selected_order"] = range(1, len(out) + 1)
    out["selection_protocol"] = "outcome_blind_neutral_scenario_stratified"
    out["selection_seed"] = int(seed)
    return out.drop(columns=["_score"], errors="ignore")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    work_dir = require_work_dir(cfg)
    full_cfg = cfg.get("full") or {}
    min_max_time_step = int((cfg.get("dataset") or {}).get("max_future_steps", 30))
    target = int(args.target_scenarios or full_cfg.get("n_scenarios_target_max") or full_cfg.get("n_scenarios_target") or 1500)
    seed = int(args.seed if args.seed is not None else full_cfg.get("seed", 42))
    manifest = pd.read_csv(os.path.expanduser(os.path.expandvars(args.manifest_csv)))
    manifest = _filter_commonroad_scenario_xml(manifest, min_max_time_step)
    manifest = manifest.drop_duplicates(subset=["scenario_id", "xml_path"]).copy()
    target = min(target, len(manifest))
    selected = _select_round_robin(manifest, target, seed)
    out_dir = work_dir / "results" / "commonroad_pilot_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    summary_path = out_dir / f"{Path(args.out_name).stem}_summary.csv"
    notes_path = out_dir / f"{Path(args.out_name).stem}_selection_notes.md"
    cfg_hash = config_hash(args.config)
    write_csv(out_path, add_config_hash(selected.to_dict("records"), cfg_hash))
    summary_rows: list[dict[str, Any]] = [
        {"metric": "selected_scenarios", "value": int(len(selected))},
        {"metric": "eligible_scenarios", "value": int(len(manifest))},
        {"metric": "auxiliary_xml_excluded", "value": "true"},
        {"metric": "min_max_time_step", "value": int(min_max_time_step)},
        {"metric": "neutral_scenario_strata", "value": int(selected["neutral_scenario_stratum"].nunique())},
        {"metric": "scenario_families", "value": int(selected["scenario_family"].nunique())},
        {"metric": "source_hints", "value": int(selected["source_hint"].nunique())},
    ]
    for key, count in Counter(selected["source_hint"].astype(str)).most_common(20):
        summary_rows.append({"metric": "source_hint_count", "value": int(count), "stratum": key})
    write_csv(summary_path, add_config_hash(summary_rows, cfg_hash))
    notes_path.write_text(
        "\n".join(
            [
                "# nc_v110 Full Scenario Selection",
                "",
                f"- selected_scenarios: {len(selected)}",
                f"- eligible_scenarios: {len(manifest)}",
                f"- seed: {seed}",
                "- protocol: outcome_blind_neutral_scenario_stratified",
                "- fields used: source_hint, scenario_family, lanelet_count, dynamic_obstacle_count, max_time_step",
                f"- scenario XML filter: parse_status=ok, planning_problem_count>=1, lanelet_count>=1, dynamic_obstacle_count>=1, max_time_step>={min_max_time_step}, excludes SUMO/net/rou/add helper XML",
                "- forbidden: planner label, actionability score, ROF score, recorded future outcome, fixed-taxonomy outcome",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        out_dir / f"{Path(args.out_name).stem}_run_manifest.json",
        {"config_hash": cfg_hash, "manifest_csv": args.manifest_csv, "out_csv": str(out_path), "rows": int(len(selected))},
    )
    print(f"[v110-full-scenarios] selected={len(selected)} out={out_path}")


if __name__ == "__main__":
    main()
