from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.external.cohort import select_neutral_stratified_cohort
from rtbev.external.metrics import (
    evaluate_external_scores,
    failure_taxonomy_rows,
    merge_scores_labels,
    recall_at_fpr_strict,
    scenario_bootstrap_deltas,
    scenario_bootstrap_deltas_strict,
    unknown_failure_sensitivity,
)


def synthetic_samples(n_scenarios: int = 12, per_scenario: int = 4) -> pd.DataFrame:
    rows = []
    for s in range(n_scenarios):
        for k in range(per_scenario):
            rows.append(
                {
                    "sample_id": f"s{s:03d}_{k}",
                    "scenario_id": f"s{s:03d}",
                    "commonroad_scenario_id": f"s{s:03d}",
                    "current_min_distance_m": 1.0 + (s % 5) + k,
                    "current_ttc_s": 0.5 + (k % 4),
                    "ego_speed_mps": 3.0 + (s % 6),
                    "agent_count": 3 + (k % 5),
                    "source_hint": "urban" if s % 2 else "highway",
                    "export_status": "ok",
                }
            )
    return pd.DataFrame(rows)


def test_neutral_stratified_cohort_caps_per_scenario() -> None:
    cohort, samples, diagnostics = select_neutral_stratified_cohort(
        synthetic_samples(),
        target_samples_min=20,
        target_samples_max=30,
        min_unique_scenarios=6,
        max_samples_per_scenario=2,
        seed=7,
    )
    assert len(samples) <= 30
    assert samples.groupby("scenario_id").size().max() <= 2
    assert cohort["scenario_id"].nunique() >= 6
    assert {d["metric"] for d in diagnostics} >= {"selected_samples", "unique_scenarios"}
    assert "neutral_stratum" in samples.columns


def test_known_unknown_external_metrics_and_bootstrap() -> None:
    samples = synthetic_samples(16, 3)
    labels = samples[["sample_id", "scenario_id", "commonroad_scenario_id"]].copy()
    idx = np.arange(len(labels))
    labels["planner_failure"] = ((idx % 5) == 0).astype(int)
    labels["planner_failure_reason"] = np.where(labels["planner_failure"].eq(0), "no_failure", np.where(idx % 10 == 0, "collision_only", "unknown"))
    scores = samples[["sample_id", "scenario_id", "current_min_distance_m", "current_ttc_s"]].copy()
    scores["ROF_v2_composite"] = labels["planner_failure"].to_numpy() * 0.8 + (idx % 7) * 0.01
    scores["distance_inverse"] = -scores["current_min_distance_m"]
    merged = merge_scores_labels(scores, labels, samples)
    taxonomy = failure_taxonomy_rows(merged)
    assert {r["failure_taxonomy"] for r in taxonomy} == {"known_failure", "no_failure", "unknown_failure"}
    metrics = evaluate_external_scores(merged, ["ROF_v2_composite", "distance_inverse"])
    assert {r["score"] for r in metrics} == {"ROF_v2_composite", "distance_inverse"}
    assert all(r["endpoint"] == "known_failure" for r in metrics)
    sens = unknown_failure_sensitivity(merged, ["ROF_v2_composite"])
    assert {r["sensitivity_mode"] for r in sens} == {"known_failure", "all_failures_positive", "unknown_as_negative"}
    deltas = scenario_bootstrap_deltas(
        merged,
        [("ROF_v2_composite", "distance_inverse")],
        metrics=("auprc", "recall_at_5pct_fpr"),
        n_bootstrap=20,
        seed=3,
    )
    assert {r["bootstrap_unit"] for r in deltas} == {"scenario_id"}
    assert {r["metric"] for r in deltas} == {"auprc", "recall_at_5pct_fpr"}


def test_strict_recall_at_fpr_respects_discrete_ties() -> None:
    y = np.array([1, 1, *([0] * 10)])
    score = np.array([2.0, 1.0, *([1.0] * 3), *([0.0] * 7)])
    strict = recall_at_fpr_strict(y, score, 0.05)
    assert strict["actual_fpr"] <= 0.05
    assert strict["recall"] == 0.5

    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(y))],
            "scenario_id": [f"g{i // 2}" for i in range(len(y))],
            "failure_taxonomy": np.where(y == 1, "known_failure", "no_failure"),
            "a": score,
            "b": np.arange(len(y), dtype=float),
        }
    )
    deltas = scenario_bootstrap_deltas_strict(frame, [("a", "b")], n_bootstrap=5, seed=1)
    assert {r["metric"] for r in deltas} == {"auprc", "recall_at_5pct_fpr_strict"}
    assert all("pairwise_n" in r and "pairwise_positive_count" in r for r in deltas)
