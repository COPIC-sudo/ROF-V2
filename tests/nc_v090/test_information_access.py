from __future__ import annotations

import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "nc_v090_scientific_audit"


def read_rows(name: str):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"optional generated audit output is not present: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_primary_predictors_do_not_use_future_or_labels():
    rows = read_rows("feature_lineage.csv")
    primary_groups = {"strong_baseline_cv", "strict_temporal_dynamics"}
    primary = [
        r for r in rows
        if primary_groups.intersection(set(str(r.get("feature_group_v090", "")).split(";")))
    ]
    assert primary, "no primary predictor lineage rows found"
    bad = [
        r["feature_name"] for r in primary
        if str(r.get("uses_observed_future")).lower() == "true"
        or str(r.get("uses_label_file")).lower() == "true"
        or "label" in r["feature_name"].lower()
    ]
    assert not bad, f"primary predictors have forbidden information access: {bad}"


def test_merge_cardinality_and_duplicate_ids():
    rows = read_rows("merge_cardinality_audit.csv")
    duplicates = [
        (r.get("artifact"), r.get("duplicate_sample_id_count"))
        for r in rows
        if r.get("artifact") != "features_map_nomap_proximity_inner_join"
        and int(float(r.get("duplicate_sample_id_count") or 0)) != 0
    ]
    assert not duplicates, f"duplicate sample IDs found: {duplicates}"
    join = [r for r in rows if r.get("artifact") == "features_map_nomap_proximity_inner_join"]
    assert join, "missing inner join audit row"
    assert int(float(join[0]["rows"])) == int(float(join[0]["unique_sample_id"]))


if __name__ == "__main__":
    test_primary_predictors_do_not_use_future_or_labels()
    test_merge_cardinality_and_duplicate_ids()
    print("information_access_tests=PASS")
