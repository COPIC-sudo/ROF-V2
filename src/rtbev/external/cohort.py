from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd


def stable_unit_float(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _bucket(values: pd.Series, edges: list[float], labels: list[str], missing: str) -> pd.Series:
    out = pd.cut(values, [-np.inf, *edges, np.inf], labels=labels).astype("object")
    out[pd.isna(out)] = missing
    return out.astype(str)


def add_neutral_strata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dist = _num(out, "current_min_distance_m")
    ttc = _num(out, "current_ttc_s")
    ttc = ttc.where(ttc >= 0.0, np.nan)
    speed = _num(out, "ego_speed_mps")
    if speed.isna().all() and "ego_speed_kph" in out.columns:
        speed = _num(out, "ego_speed_kph") / 3.6
    agents = _num(out, "agent_count")
    out["distance_stratum"] = _bucket(dist, [3.0, 10.0], ["lt3m", "3to10m", "gte10m"], "missing_distance")
    out["ttc_stratum"] = _bucket(ttc, [1.0, 3.0], ["lt1s", "1to3s", "gte3s"], "missing_ttc")
    out["speed_stratum"] = _bucket(speed, [5.0, 15.0], ["lt5mps", "5to15mps", "gte15mps"], "missing_speed")
    out["agent_count_stratum"] = _bucket(agents, [5.0, 15.0], ["lt5", "5to15", "gte15"], "missing_agents")
    source = out["source_hint"].fillna("unknown").astype(str) if "source_hint" in out.columns else pd.Series("unknown", index=out.index)
    out["neutral_stratum"] = (
        out["distance_stratum"]
        + "|"
        + out["ttc_stratum"]
        + "|"
        + out["speed_stratum"]
        + "|"
        + out["agent_count_stratum"]
        + "|"
        + source
    )
    return out


def _scenario_col(df: pd.DataFrame) -> str:
    for col in ["scenario_id", "commonroad_scenario_id"]:
        if col in df.columns:
            return col
    raise ValueError("sample candidates require scenario_id or commonroad_scenario_id")


def sample_candidates_from_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    out = scenarios.copy()
    if "scenario_id" not in out.columns and "commonroad_scenario_id" in out.columns:
        out["scenario_id"] = out["commonroad_scenario_id"].astype(str)
    if "commonroad_scenario_id" not in out.columns:
        out["commonroad_scenario_id"] = out["scenario_id"].astype(str)
    if "sample_id" not in out.columns:
        out["sample_id"] = out["scenario_id"].astype(str)
    return out


def select_neutral_stratified_cohort(
    sample_candidates: pd.DataFrame,
    target_samples_min: int = 10000,
    target_samples_max: int = 15000,
    min_unique_scenarios: int = 1000,
    max_samples_per_scenario: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if sample_candidates.empty:
        raise ValueError("sample candidate table is empty")
    work = sample_candidates_from_scenarios(sample_candidates)
    scenario_col = _scenario_col(work)
    work["sample_id"] = work["sample_id"].astype(str)
    work["scenario_id"] = work[scenario_col].astype(str)
    if "commonroad_scenario_id" not in work.columns:
        work["commonroad_scenario_id"] = work["scenario_id"]
    for status_col in ["parse_status", "export_status"]:
        if status_col in work.columns:
            work = work[work[status_col].fillna("ok").astype(str).str.lower().isin(["ok", "success", "true", "1"])].copy()
    if work.empty:
        raise ValueError("no eligible samples after status filtering")
    work = add_neutral_strata(work)
    work["_selection_score"] = work["sample_id"].map(lambda s: stable_unit_float(f"{seed}|{s}"))
    work = work.sort_values(["scenario_id", "_selection_score", "sample_id"]).copy()
    work["_within_scenario_rank"] = work.groupby("scenario_id").cumcount() + 1
    work = work[work["_within_scenario_rank"] <= int(max_samples_per_scenario)].copy()
    target = int(target_samples_max)
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, row in work.reset_index(drop=True).iterrows():
        buckets[str(row["neutral_stratum"])].append(int(idx))
    bucket_keys = sorted(buckets, key=lambda k: stable_unit_float(f"{seed}|bucket|{k}"))
    selected_idx: list[int] = []
    seen: set[int] = set()
    while len(selected_idx) < target and bucket_keys:
        next_keys: list[str] = []
        for key in bucket_keys:
            if len(selected_idx) >= target:
                break
            if not buckets[key]:
                continue
            idx = buckets[key].pop(0)
            if idx not in seen:
                seen.add(idx)
                selected_idx.append(idx)
            if buckets[key]:
                next_keys.append(key)
        bucket_keys = next_keys
    selected = work.reset_index(drop=True).iloc[selected_idx].copy()
    selected = selected.sort_values(["scenario_id", "_within_scenario_rank", "sample_id"]).reset_index(drop=True)
    selected["cohort_role"] = "primary_neutral"
    selected["selection_protocol"] = "outcome_blind_neutral_stratified"
    selected["selection_seed"] = int(seed)
    selected["max_samples_per_scenario"] = int(max_samples_per_scenario)
    selected["target_samples_min"] = int(target_samples_min)
    selected["target_samples_max"] = int(target_samples_max)
    selected["target_min_unique_scenarios"] = int(min_unique_scenarios)
    selected["selected_sample_order"] = np.arange(1, len(selected) + 1)
    selected = selected.drop(columns=[c for c in ["_selection_score"] if c in selected.columns])

    scenario_rows: list[dict[str, Any]] = []
    for sid, group in selected.groupby("scenario_id", sort=True):
        counts = Counter(group["neutral_stratum"].astype(str))
        scenario_rows.append(
            {
                "scenario_id": sid,
                "commonroad_scenario_id": group["commonroad_scenario_id"].astype(str).iloc[0],
                "selected_sample_count": int(len(group)),
                "neutral_strata": ";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                "cohort_role": "primary_neutral",
                "selection_protocol": "outcome_blind_neutral_stratified",
                "selection_seed": int(seed),
            }
        )
    diagnostics = [
        {"metric": "selected_samples", "value": int(len(selected)), "target_min": int(target_samples_min), "target_max": int(target_samples_max)},
        {"metric": "unique_scenarios", "value": int(selected["scenario_id"].nunique()), "target_min": int(min_unique_scenarios)},
        {
            "metric": "max_samples_per_scenario_observed",
            "value": int(selected.groupby("scenario_id").size().max()) if len(selected) else 0,
            "target_max": int(max_samples_per_scenario),
        },
        {"metric": "neutral_strata_count", "value": int(selected["neutral_stratum"].nunique())},
        {"metric": "eligible_samples_after_filters", "value": int(len(work))},
    ]
    return pd.DataFrame(scenario_rows), selected, diagnostics
