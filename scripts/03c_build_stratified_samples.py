from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.nc_eval import scenario_hash_split


def _merge_features(labels: pd.DataFrame, features_csv: str | None) -> pd.DataFrame:
    df = labels.copy()
    if features_csv:
        feat = pd.read_csv(features_csv)
        keep_cols = [c for c in [
            "sample_id", "current_min_distance_m", "current_ttc_s", "ego_speed_kph", "agent_count",
            "nearest_agent_rel_speed_mps", "nearest_agent_closing_speed_mps",
            "rcr", "redi_full", "redi_actionability", "asr_cum_final", "asr_slice_min", "ttad_s",
        ] if c in feat.columns]
        df = df.merge(feat[keep_cols], on="sample_id", how="left", suffixes=("", "_feat"))
    return df


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _hard_case_mask(df: pd.DataFrame) -> pd.Series:
    labels = pd.to_numeric(df["label_id"], errors="coerce").fillna(0).astype(int)
    dist = _num(df, "current_min_distance_m")
    ttc = _num(df, "current_ttc_s")
    speed = _num(df, "ego_speed_kph")
    agents = _num(df, "agent_count")
    dense_thr = float(agents.quantile(0.75)) if agents.notna().any() else np.inf
    hard = pd.Series(False, index=df.index)
    hard |= ((dist < 5.0) & (labels <= 1)).fillna(False)       # close but non-critical
    hard |= (((ttc < 0) | (~np.isfinite(ttc))) & (labels >= 2)).fillna(False)
    hard |= (((ttc > 3.0) | (ttc < 0) | (~np.isfinite(ttc))) & (labels >= 2)).fillna(False)
    hard |= ((speed < 15.0) & (labels >= 2)).fillna(False)
    hard |= ((agents >= dense_thr) & (labels >= 2)).fillna(False)
    return hard.astype(bool)


def _sample_cap(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or cap <= 0 or len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stratified/hard-case label CSVs after labels/features generation.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--features-csv", default=None, help="optional; enables hard-case selection based on distance/TTC/ROF features")
    ap.add_argument("--test-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--safe-cap", type=int, default=3000)
    ap.add_argument("--caution-cap", type=int, default=3000)
    ap.add_argument("--warning-cap", type=int, default=0, help="0 means keep all")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "labels")
    seed = int(args.seed if args.seed is not None else cfg.get("analysis", {}).get("random_seed", 42))
    test_fraction = float(args.test_fraction if args.test_fraction is not None else cfg.get("analysis", {}).get("test_fraction", 0.25))
    labels = pd.read_csv(args.labels_csv)
    if "label_id" not in labels.columns:
        raise SystemExit("labels CSV must contain label_id")
    df = _merge_features(labels, args.features_csv)
    df["label_id"] = df["label_id"].astype(int)
    train_mask, test_mask = scenario_hash_split(df, test_fraction=test_fraction, seed=seed)
    train_pool = df[train_mask].copy()
    natural_test = df[test_mask].copy()

    hard = df[_hard_case_mask(df)].copy()
    # Enriched train: keep all emergency and warning by default, cap lower-risk classes.
    parts = []
    for lab, cap in [(0, args.safe_cap), (1, args.caution_cap), (2, args.warning_cap), (3, 0)]:
        sub = train_pool[train_pool["label_id"] == lab]
        parts.append(_sample_cap(sub, cap, seed + lab))
    enriched_train = pd.concat(parts, ignore_index=True).drop_duplicates("sample_id") if parts else train_pool.iloc[0:0]
    # Preserve hard train cases even if class caps sampled them out.
    hard_train = hard[hard["sample_id"].isin(set(train_pool["sample_id"].astype(str)))].copy()
    enriched_train = pd.concat([enriched_train, hard_train], ignore_index=True).drop_duplicates("sample_id")
    emergency_enriched = pd.concat([df[df["label_id"] == 3], df[df["label_id"] == 2], hard], ignore_index=True).drop_duplicates("sample_id")

    prefix = args.out_prefix
    def name(base: str) -> Path:
        return out / f"{prefix}{base}" if prefix else out / base
    enriched_train.to_csv(name("labels_stratified_train.csv"), index=False)
    natural_test.to_csv(name("labels_natural_test.csv"), index=False)
    hard.to_csv(name("labels_hard_cases.csv"), index=False)
    emergency_enriched.to_csv(name("labels_emergency_enriched.csv"), index=False)
    summary_rows = []
    for split_name, sub in [("all", df), ("stratified_train", enriched_train), ("natural_test", natural_test), ("hard_cases", hard), ("emergency_enriched", emergency_enriched)]:
        vc = sub["label_id"].value_counts().sort_index()
        row = {"split": split_name, "n": int(len(sub)), "n_scenarios": int(sub.get("scenario_id", sub["sample_id"]).astype(str).nunique())}
        for lab in [0, 1, 2, 3]:
            row[f"label_{lab}_n"] = int(vc.get(lab, 0))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(name("label_sampling_summary.csv"), index=False)
    print(f"[stratified] wrote under {out}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
