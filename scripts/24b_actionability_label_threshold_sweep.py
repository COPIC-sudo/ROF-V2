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


def _rule_moderate(c: float, e: float) -> int:
    if e <= 0.0:
        return 3
    if c <= 0.25 or e <= 0.35:
        return 2
    if c < 0.75 or e < 0.65:
        return 1
    return 0


def _rule_strict(c: float, e: float) -> int:
    if e <= 0.0:
        return 3
    if c <= 0.50 or e <= 0.50:
        return 2
    if c < 1.00 or e < 0.85:
        return 1
    return 0


def _rule_emergency_sensitive(c: float, e: float) -> int:
    if e <= 0.0:
        return 3
    if e <= 0.25:
        return 2
    if e <= 0.60 or c <= 0.50:
        return 1
    return 0


def _rule_comfort_sensitive(c: float, e: float) -> int:
    if e <= 0.0:
        return 3
    if c <= 0.25:
        return 2
    if c <= 0.75:
        return 1
    return 0


def _rule_quantile_like(c: float, e: float) -> int:
    if e <= 0.0:
        return 3
    if c <= 0.50 or e <= 0.4286:
        return 2
    if c <= 0.75 or e <= 0.7143:
        return 1
    return 0


RULES = {
    "moderate": _rule_moderate,
    "strict": _rule_strict,
    "emergency_sensitive": _rule_emergency_sensitive,
    "comfort_sensitive": _rule_comfort_sensitive,
    "quantile_like": _rule_quantile_like,
}


def _assign_rule(df: pd.DataFrame, rule_name: str) -> pd.Series:
    fn = RULES[rule_name]
    comfort = pd.to_numeric(df["comfort_feasible_ratio"], errors="coerce").fillna(0.0).to_numpy(float)
    emergency = pd.to_numeric(df["emergency_feasible_ratio"], errors="coerce").fillna(0.0).to_numpy(float)
    return pd.Series([fn(c, e) for c, e in zip(comfort, emergency)], index=df.index, dtype=int)


def _spearman(df: pd.DataFrame, new_label: pd.Series) -> float:
    if "current_min_distance_m" not in df.columns:
        return np.nan
    dist = pd.to_numeric(df["current_min_distance_m"], errors="coerce")
    ok = dist.notna() & new_label.notna()
    if ok.sum() < 3:
        return np.nan
    return float(new_label[ok].corr(dist[ok], method="spearman"))


def _criteria(fractions: dict[int, float], spearman: float) -> tuple[bool, str]:
    checks = []
    high = fractions.get(0, 0.0)
    reduced = fractions.get(1, 0.0)
    critical = fractions.get(2, 0.0)
    infeasible = fractions.get(3, 0.0)
    checks.append((0.30 <= high <= 0.80, f"high_fraction={high:.4f}"))
    checks.append((reduced >= 0.05, f"reduced_fraction={reduced:.4f}"))
    checks.append((critical >= 0.02, f"critical_fraction={critical:.4f}"))
    checks.append((0.002 <= infeasible <= 0.15, f"infeasible_fraction={infeasible:.4f}"))
    if np.isfinite(spearman):
        checks.append((abs(float(spearman)) < 0.6, f"abs_spearman={abs(float(spearman)):.4f}"))
    return all(ok for ok, _ in checks), "; ".join(msg for _, msg in checks)


def _build_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    dist_rows = []
    crosstab_rows = []
    ratio_rows = []
    n = max(len(df), 1)

    for rule_name in RULES:
        labels = _assign_rule(df, rule_name)
        counts = labels.value_counts().reindex([0, 1, 2, 3], fill_value=0).sort_index()
        fractions = {int(k): float(v / n) for k, v in counts.items()}
        spearman = _spearman(df, labels)
        ok, detail = _criteria(fractions, spearman)
        summary_rows.append({
            "rule_name": rule_name,
            "n": int(len(df)),
            "non_degenerate": bool(ok),
            "criteria_detail": detail,
            "spearman_current_min_distance_m": spearman,
            "high_fraction": fractions.get(0, 0.0),
            "reduced_fraction": fractions.get(1, 0.0),
            "critical_fraction": fractions.get(2, 0.0),
            "infeasible_fraction": fractions.get(3, 0.0),
        })
        for label_id, count in counts.items():
            dist_rows.append({
                "rule_name": rule_name,
                "new_actionability_label_id": int(label_id),
                "new_actionability_label_name": LABEL_NAMES[int(label_id)],
                "count": int(count),
                "fraction": float(count / n),
            })
        ct = pd.crosstab(df["original_label_id"].astype(int), labels)
        for original_id in sorted(df["original_label_id"].dropna().astype(int).unique()):
            for label_id in [0, 1, 2, 3]:
                value = int(ct.loc[original_id, label_id]) if original_id in ct.index and label_id in ct.columns else 0
                crosstab_rows.append({
                    "rule_name": rule_name,
                    "original_label_id": int(original_id),
                    "new_actionability_label_id": int(label_id),
                    "count": value,
                })
        tmp = df.copy()
        tmp["new_actionability_label_id"] = labels
        for label_id, sub in tmp.groupby("new_actionability_label_id"):
            for variable in ["comfort_feasible_ratio", "emergency_feasible_ratio"]:
                s = pd.to_numeric(sub[variable], errors="coerce")
                ratio_rows.append({
                    "rule_name": rule_name,
                    "new_actionability_label_id": int(label_id),
                    "new_actionability_label_name": LABEL_NAMES[int(label_id)],
                    "variable": variable,
                    "count": int(s.notna().sum()),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                })
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(dist_rows),
        pd.DataFrame(crosstab_rows),
        pd.DataFrame(ratio_rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep threshold rules for existing actionability-label pilot outputs.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-name", default="actionability_threshold_sweep")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    df = pd.read_csv(args.input_csv)
    required = ["comfort_feasible_ratio", "emergency_feasible_ratio", "original_label_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns in input CSV: {missing}")

    out_dir = ensure_dir(work / "results" / "nc_actionability_labels" / "threshold_sweep")
    summary, distributions, crosstabs, ratio_stats = _build_outputs(df)
    summary.to_csv(out_dir / "threshold_sweep_summary.csv", index=False)
    distributions.to_csv(out_dir / "threshold_sweep_distributions.csv", index=False)
    crosstabs.to_csv(out_dir / "threshold_sweep_crosstabs.csv", index=False)
    ratio_stats.to_csv(out_dir / "threshold_sweep_ratio_stats.csv", index=False)
    print(f"[actionability-threshold-sweep] wrote {out_dir}; out_name={args.out_name}; rules={len(RULES)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
