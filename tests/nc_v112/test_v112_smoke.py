from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.baselines.scores import compute_all_baseline_scores
from rtbev.external.metrics import evaluate_external_scores, merge_scores_labels, scenario_bootstrap_deltas


def make_features(n: int = 48) -> pd.DataFrame:
    rows = []
    for i in range(n):
        risky = int(i % 6 == 0 or i % 11 == 0)
        rows.append(
            {
                "sample_id": f"cr_{i:03d}",
                "scenario_id": f"scenario_{i // 3:03d}",
                "commonroad_scenario_id": f"scenario_{i // 3:03d}",
                "current_min_distance_m": 1.0 + (i % 8) - 0.5 * risky,
                "current_ttc_s": 0.4 + (i % 5),
                "ego_speed_mps": 4.0 + (i % 9),
                "nearest_agent_closing_speed_mps": 0.5 + risky * 5.0,
                "nearest_agent_rel_speed_mps": -1.0 * risky,
                "nearest_agent_lateral_speed_mps": 0.2 + 0.3 * risky,
                "nearby_agent_count_10m": 1 + risky * 4,
                "agent_count": 4 + i % 5,
                "cv_oce_norm": 0.1 + 0.7 * risky,
                "cv_rcr": 0.1 + 0.6 * risky,
                "ROF_v2_composite": 0.2 + 0.7 * risky + (i % 3) * 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_field_baseline_scores_and_metrics_do_not_require_future() -> None:
    features = make_features()
    scores = compute_all_baseline_scores(features)
    assert set(scores) == {"commonroad_crime_scores", "rss_scores", "drivability_baseline_scores", "forecast_risk_scores"}
    for df in scores.values():
        assert "recorded_future_access" in df.columns
        assert not df["recorded_future_access"].astype(bool).any()
    merged_scores = scores["commonroad_crime_scores"].merge(
        scores["rss_scores"][["sample_id", "rss_danger_score"]],
        on="sample_id",
    ).merge(
        scores["drivability_baseline_scores"][["sample_id", "drivability_risk_score"]],
        on="sample_id",
    ).merge(
        scores["forecast_risk_scores"][["sample_id", "forecast_risk_score"]],
        on="sample_id",
    )
    merged_scores = merged_scores.merge(features[["sample_id", "ROF_v2_composite"]], on="sample_id")
    labels = features[["sample_id", "scenario_id", "commonroad_scenario_id"]].copy()
    idx = np.arange(len(labels))
    labels["planner_failure"] = ((idx % 6 == 0) | (idx % 11 == 0)).astype(int)
    labels["planner_failure_reason"] = np.where(labels["planner_failure"].eq(1), "collision_only", "no_failure")
    merged = merge_scores_labels(merged_scores, labels)
    metrics = evaluate_external_scores(merged, ["commonroad_crime_risk_score", "rss_danger_score", "drivability_risk_score", "forecast_risk_score", "ROF_v2_composite"])
    assert {r["score"] for r in metrics} >= {"forecast_risk_score", "ROF_v2_composite"}
    deltas = scenario_bootstrap_deltas(
        merged,
        [("ROF_v2_composite", "forecast_risk_score")],
        metrics=("auprc",),
        n_bootstrap=20,
        seed=5,
    )
    assert len(deltas) == 1
    assert deltas[0]["bootstrap_unit"] == "scenario_id"
