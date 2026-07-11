from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_merged(work_dir: str | Path) -> pd.DataFrame:
    work = Path(work_dir)
    feat = work / "features" / "rof_features.csv"
    if not feat.exists():
        feat = work / "features" / "rt_features.csv"
    if not feat.exists():
        raise FileNotFoundError(f"features csv not found under {work / 'features'}")
    df = pd.read_csv(feat)
    return df


def label_order():
    return ["safe", "caution", "warning", "emergency"]


def finite_series(s: pd.Series, fill: float = 0.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.fillna(fill)


def scenario_hash_split(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42):
    # deterministic split by scenario_id/sample_id to avoid random frame leakage
    key_col = "scenario_id" if "scenario_id" in df.columns else "sample_id"
    keys = df[key_col].astype(str).unique()
    rng = np.random.default_rng(seed)
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction))) if len(shuffled) > 1 else 0
    test_keys = set(shuffled[:n_test])
    test_mask = df[key_col].astype(str).isin(test_keys).to_numpy()
    return ~test_mask, test_mask


def ordinal_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def high_emergency_binary(y, positive_min_label: int = 2):
    return (np.asarray(y, dtype=int) >= int(positive_min_label)).astype(int)
