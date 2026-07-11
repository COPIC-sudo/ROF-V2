from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.baselines.feature_sets import feature_lineage_rows, strict_non_action_current_cv_columns
from rtbev.baselines.oof import grouped_oof_predictions, oof_metrics


def make_table(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        y = int(i % 4 == 0 or i % 7 == 0)
        rows.append(
            {
                "sample_id": f"id_{i:03d}",
                "scenario_id": f"scenario_{i // 4:03d}",
                "segment_id": f"segment_{i // 8:03d}",
                "current_min_distance_m": 2.0 + (i % 9) - y * 1.5,
                "current_ttc_s": 0.5 + (i % 5),
                "ego_speed_kph": 10.0 + i % 12,
                "agent_count": 3 + i % 6,
                "cv_rcr": 0.1 + 0.6 * y + (i % 3) * 0.02,
                "cv_oce_norm": 0.2 + 0.5 * y,
                "redi_actionability": 0.9 * y,
                "asr_cum_final": 1.0 - 0.7 * y,
                "actionability_label_id": 2 if y else 0,
            }
        )
    return pd.DataFrame(rows)


def test_feature_lineage_excludes_action_survival_from_strict_set() -> None:
    rows = feature_lineage_rows(
        [
            "current_min_distance_m",
            "cv_rcr",
            "current_collision",
            "max_overlap_count",
            "mean_overlap_count_nonzero",
            "overlap_count_entropy_norm",
            "redi_actionability",
            "asr_cum_final",
            "label_id",
        ]
    )
    by_feature = {r["feature"]: r for r in rows}
    assert by_feature["current_min_distance_m"]["feature_name"] == "current_min_distance_m"
    assert by_feature["current_min_distance_m"]["allowed_in_strict_non_action_current_cv"] is True
    assert by_feature["current_min_distance_m"]["allowed_in_strict_non_action"] is True
    assert by_feature["cv_rcr"]["allowed_in_strict_non_action_current_cv"] is True
    for feature in [
        "current_collision",
        "max_overlap_count",
        "mean_overlap_count_nonzero",
        "overlap_count_entropy_norm",
    ]:
        assert by_feature[feature]["allowed_in_strict_non_action_current_cv"] is True
        assert by_feature[feature]["uses_action_library"] is False
        assert by_feature[feature]["uses_candidate_survival"] is False
        assert by_feature[feature]["reads_label"] is False
        assert by_feature[feature]["reads_recorded_future"] is False
    assert by_feature["redi_actionability"]["uses_action_library"] is True
    assert by_feature["asr_cum_final"]["uses_candidate_survival"] is True
    assert by_feature["label_id"]["reads_label"] is True
    assert "uses_endpoint_intermediate" in by_feature["label_id"]


def test_grouped_oof_runs_with_strict_non_action_features() -> None:
    df = make_table()
    df["_y"] = (df["actionability_label_id"] >= 2).astype(int)
    cols = strict_non_action_current_cv_columns(df)
    assert "redi_actionability" not in cols
    assert {"current_min_distance_m", "cv_rcr"} <= set(cols)
    pred = grouped_oof_predictions(df, cols, "_y", group_col="scenario_id", n_folds=4, seed=11, model="rf")
    metrics = oof_metrics(pred)
    assert metrics["n"] > 0
    assert metrics["positive_count"] > 0
    assert "AUPRC" in metrics
