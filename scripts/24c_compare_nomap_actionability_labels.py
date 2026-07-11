from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir


LABEL_NAMES = {
    0: "high_actionability",
    1: "reduced_actionability",
    2: "critical_actionability",
    3: "infeasible_or_unavoidable",
}

FEATURE_COLS = [
    "sample_id",
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "agent_count",
]


def _read_labels(path: str | Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns or "actionability_label_id" not in df.columns:
        raise ValueError(f"labels CSV must include sample_id and actionability_label_id: {path}; columns={list(df.columns)}")
    keep = [
        "sample_id",
        "actionability_label_id",
        "actionability_label_name",
        "comfort_feasible_ratio",
        "emergency_feasible_ratio",
        "original_label_id",
        "original_label_name",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    out["sample_id"] = out["sample_id"].astype(str)
    rename = {c: f"{prefix}_{c}" for c in cols if c != "sample_id"}
    return out.rename(columns=rename)


def _read_features(path: str | Path, sample_ids: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in FEATURE_COLS if c in header.columns]
    if "sample_id" not in cols:
        return pd.DataFrame({"sample_id": list(sample_ids)})
    df = pd.read_csv(path, usecols=cols)
    df["sample_id"] = df["sample_id"].astype(str)
    return df[df["sample_id"].isin(sample_ids)].drop_duplicates("sample_id").copy()


def _distribution_rows(df: pd.DataFrame, col: str, section: str) -> list[dict]:
    n = max(int(df[col].notna().sum()), 1)
    counts = pd.to_numeric(df[col], errors="coerce").value_counts().reindex([0, 1, 2, 3], fill_value=0).sort_index()
    rows = []
    for label_id, count in counts.items():
        rows.append({
            "section": section,
            "metric": "label_distribution",
            "label_id": int(label_id),
            "label_name": LABEL_NAMES[int(label_id)],
            "count": int(count),
            "fraction": float(count / n),
            "value": np.nan,
        })
    return rows


def _safe_spearman(df: pd.DataFrame, a: str, b: str) -> tuple[float, int]:
    if a not in df.columns or b not in df.columns:
        return np.nan, 0
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    ok = x.notna() & y.notna()
    if int(ok.sum()) < 3:
        return np.nan, int(ok.sum())
    value = x[ok].corr(y[ok], method="spearman")
    return float(value) if pd.notna(value) else np.nan, int(ok.sum())


def _median_rows(df: pd.DataFrame, prefix: str) -> list[dict]:
    rows = []
    cols = [
        f"{prefix}_comfort_feasible_ratio",
        f"{prefix}_emergency_feasible_ratio",
        "current_min_distance_m",
        "current_ttc_s",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "section": f"{prefix}_medians",
            "metric": col,
            "label_id": np.nan,
            "label_name": "",
            "count": int(values.notna().sum()),
            "fraction": np.nan,
            "value": float(values.median()) if values.notna().any() else np.nan,
        })
    return rows


def _nomap_label_median_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    cols = [
        "nomap_comfort_feasible_ratio",
        "nomap_emergency_feasible_ratio",
        "current_min_distance_m",
        "current_ttc_s",
    ]
    for label_id in [0, 1, 2, 3]:
        sub = df[pd.to_numeric(df["nomap_actionability_label_id"], errors="coerce").eq(label_id)]
        for col in cols:
            if col not in sub.columns:
                continue
            values = pd.to_numeric(sub[col], errors="coerce")
            rows.append({
                "section": "nomap_per_label_medians",
                "metric": col,
                "label_id": int(label_id),
                "label_name": LABEL_NAMES[int(label_id)],
                "count": int(values.notna().sum()),
                "fraction": float(len(sub) / max(len(df), 1)),
                "value": float(values.median()) if values.notna().any() else np.nan,
            })
    return rows


def _shift_summary_rows(df: pd.DataFrame) -> list[dict]:
    n = max(len(df), 1)
    changed = df["map_actionability_label_id"].ne(df["nomap_actionability_label_id"])
    map_crit_inf = pd.to_numeric(df["map_actionability_label_id"], errors="coerce") >= 2
    nomap_high_red = pd.to_numeric(df["nomap_actionability_label_id"], errors="coerce") <= 1
    map_inf = pd.to_numeric(df["map_actionability_label_id"], errors="coerce").eq(3)
    nomap_non_inf = pd.to_numeric(df["nomap_actionability_label_id"], errors="coerce").ne(3)
    orig_safe = pd.to_numeric(df.get("nomap_original_label_id", df.get("map_original_label_id")), errors="coerce").isin([0, 1])
    safe_map_crit_inf = orig_safe & map_crit_inf
    rows = []
    specs = [
        ("fraction_changed", changed, np.ones(len(df), dtype=bool)),
        ("fraction_map_critical_infeasible_to_nomap_high_reduced", map_crit_inf & nomap_high_red, map_crit_inf),
        ("fraction_map_infeasible_to_nomap_non_infeasible", map_inf & nomap_non_inf, map_inf),
        (
            "fraction_original_safe_map_critical_infeasible_to_nomap_high_reduced",
            safe_map_crit_inf & nomap_high_red,
            safe_map_crit_inf,
        ),
    ]
    for metric, numerator_mask, denominator_mask in specs:
        denom = int(np.asarray(denominator_mask).sum())
        numer = int(np.asarray(numerator_mask).sum())
        rows.append({
            "section": "label_shift",
            "metric": metric,
            "label_id": np.nan,
            "label_name": "",
            "count": numer,
            "denominator": denom,
            "fraction": float(numer / denom) if denom else np.nan,
            "value": float(numer / denom) if denom else np.nan,
        })
    rows.append({
        "section": "label_shift",
        "metric": "n_pilot_samples",
        "label_id": np.nan,
        "label_name": "",
        "count": int(len(df)),
        "denominator": int(n),
        "fraction": 1.0,
        "value": float(len(df)),
    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare map-constrained and no-map actionability labels on a pilot subset.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--map-labels-csv", required=True)
    ap.add_argument("--nomap-labels-csv", required=True)
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.out_name:
        out_dir = work / "results" / "nc_actionability_labels" / "nomap_sensitivity" / args.out_name
    else:
        out_dir = work / "results" / "nc_actionability_labels" / "nomap_sensitivity"
    out_dir = ensure_dir(out_dir)

    nomap = _read_labels(args.nomap_labels_csv, "nomap")
    sample_ids = set(nomap["sample_id"].astype(str))
    map_labels = _read_labels(args.map_labels_csv, "map")
    features = _read_features(args.features_csv, sample_ids)
    merged = nomap.merge(map_labels, on="sample_id", how="left", validate="one_to_one")
    merged = merged.merge(features, on="sample_id", how="left")
    missing_map = int(merged["map_actionability_label_id"].isna().sum())
    if missing_map:
        raise ValueError(f"{missing_map} no-map pilot samples were not found in map labels")

    map_ids = pd.to_numeric(merged["map_actionability_label_id"], errors="coerce").astype(int)
    nomap_ids = pd.to_numeric(merged["nomap_actionability_label_id"], errors="coerce").astype(int)
    crosstab = pd.crosstab(map_ids, nomap_ids).reindex(index=[0, 1, 2, 3], columns=[0, 1, 2, 3], fill_value=0)
    crosstab.index.name = "map_actionability_label_id"
    crosstab.columns = [f"nomap_{c}" for c in crosstab.columns]
    crosstab = crosstab.reset_index()
    crosstab["map_actionability_label_name"] = crosstab["map_actionability_label_id"].map(LABEL_NAMES)
    crosstab.to_csv(out_dir / "map_vs_nomap_crosstab_on_pilot.csv", index=False)

    shift = merged.copy()
    shift["label_changed"] = map_ids.ne(nomap_ids)
    shift["map_critical_or_infeasible"] = map_ids.ge(2)
    shift["nomap_high_or_reduced"] = nomap_ids.le(1)
    shift["map_critical_infeasible_to_nomap_high_reduced"] = shift["map_critical_or_infeasible"] & shift["nomap_high_or_reduced"]
    shift["map_infeasible_to_nomap_non_infeasible"] = map_ids.eq(3) & nomap_ids.ne(3)
    original_col = "nomap_original_label_id" if "nomap_original_label_id" in shift.columns else "map_original_label_id"
    shift["original_safe"] = pd.to_numeric(shift[original_col], errors="coerce").isin([0, 1])
    shift["original_safe_map_critical_infeasible_to_nomap_high_reduced"] = (
        shift["original_safe"]
        & shift["map_critical_or_infeasible"]
        & shift["nomap_high_or_reduced"]
    )
    shift.to_csv(out_dir / "map_sensitive_label_shift.csv", index=False)

    rows: list[dict] = []
    rows.extend(_distribution_rows(merged, "nomap_actionability_label_id", "nomap_pilot"))
    rows.extend(_distribution_rows(merged, "map_actionability_label_id", "map_on_nomap_pilot"))
    rows.extend(_shift_summary_rows(merged))
    for prefix in ["nomap", "map"]:
        for col in ["current_min_distance_m", "current_ttc_s"]:
            corr, count = _safe_spearman(merged, f"{prefix}_actionability_label_id", col)
            rows.append({
                "section": "spearman",
                "metric": f"{prefix}_actionability_label_id_vs_{col}",
                "label_id": np.nan,
                "label_name": "",
                "count": count,
                "fraction": np.nan,
                "value": corr,
            })
    rows.extend(_median_rows(merged, "nomap"))
    rows.extend(_median_rows(merged, "map"))
    rows.extend(_nomap_label_median_rows(merged))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "nomap_pilot_summary.csv", index=False)
    print(f"[nomap-compare] wrote {out_dir}")
    print(summary.to_string(index=False))
    print(crosstab.to_string(index=False))


if __name__ == "__main__":
    main()
