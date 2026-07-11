from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from _bootstrap import ROOT  # noqa: F401
from rtbev.analysis_utils import load_merged
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.nc_eval import score_definitions, y_for_task


SCORE_NAMES = [
    "distance_inverse",
    "TTC_inverse",
    "REDI_full",
    "REDI_actionability",
    "ROF_v2_composite",
    "TTAD_inverse",
    "early_blocking",
    "ASR_cum_inverse",
    "ASR_slice_min_inverse",
    "comfort_ASR_inverse",
]


def _load_features(work: Path, features_csv: str | None) -> pd.DataFrame:
    return pd.read_csv(features_csv) if features_csv else load_merged(work)


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)


def _settings() -> list[dict]:
    return [
        {
            "setting_name": "distance_only_0p25m",
            "family": "distance_only",
            "distance_m": 0.25,
            "dims": [("current_min_distance_m", "distance_m")],
        },
        {
            "setting_name": "distance_only_0p5m",
            "family": "distance_only",
            "distance_m": 0.5,
            "dims": [("current_min_distance_m", "distance_m")],
        },
        {
            "setting_name": "distance_only_1p0m",
            "family": "distance_only",
            "distance_m": 1.0,
            "dims": [("current_min_distance_m", "distance_m")],
        },
        {
            "setting_name": "distance_speed_agent",
            "family": "distance_speed_agent",
            "distance_m": 0.5,
            "speed_kph": 2.0,
            "agent_count": 2.0,
            "dims": [
                ("current_min_distance_m", "distance_m"),
                ("ego_speed_kph", "speed_kph"),
                ("agent_count", "agent_count"),
            ],
        },
        {
            "setting_name": "finite_ttc_strict",
            "family": "finite_ttc_strict",
            "finite_ttc": True,
            "distance_m": 0.5,
            "ttc_s": 1.0,
            "speed_kph": 2.0,
            "dims": [
                ("current_min_distance_m", "distance_m"),
                ("current_ttc_s", "ttc_s"),
                ("ego_speed_kph", "speed_kph"),
            ],
        },
        {
            "setting_name": "finite_ttc_loose",
            "family": "finite_ttc_loose",
            "finite_ttc": True,
            "distance_m": 1.0,
            "ttc_s": 2.0,
            "speed_kph": 5.0,
            "dims": [
                ("current_min_distance_m", "distance_m"),
                ("current_ttc_s", "ttc_s"),
                ("ego_speed_kph", "speed_kph"),
            ],
        },
        {
            "setting_name": "no_ttc_pair",
            "family": "no_ttc_pair",
            "no_ttc": True,
            "distance_m": 1.0,
            "speed_kph": 5.0,
            "dims": [
                ("current_min_distance_m", "distance_m"),
                ("ego_speed_kph", "speed_kph"),
            ],
        },
        {
            "setting_name": "large_or_no_ttc",
            "family": "large_or_no_ttc",
            "large_or_no_ttc": True,
            "distance_m": 1.0,
            "speed_kph": 5.0,
            "dims": [
                ("current_min_distance_m", "distance_m"),
                ("ego_speed_kph", "speed_kph"),
            ],
        },
    ]


def _candidate_mask(values: dict[str, np.ndarray], pos_i: int, neg_idx: np.ndarray, setting: dict) -> np.ndarray:
    dist = values["current_min_distance_m"]
    ttc = values["current_ttc_s"]
    speed = values["ego_speed_kph"]
    agents = values["agent_count"]

    mask = np.isfinite(dist[neg_idx]) & np.isfinite(dist[pos_i])
    mask &= np.abs(dist[neg_idx] - dist[pos_i]) <= float(setting.get("distance_m", np.inf))

    if "speed_kph" in setting:
        mask &= np.isfinite(speed[neg_idx]) & np.isfinite(speed[pos_i])
        mask &= np.abs(speed[neg_idx] - speed[pos_i]) <= float(setting["speed_kph"])

    if "agent_count" in setting:
        mask &= np.isfinite(agents[neg_idx]) & np.isfinite(agents[pos_i])
        mask &= np.abs(agents[neg_idx] - agents[pos_i]) <= float(setting["agent_count"])

    if setting.get("finite_ttc"):
        mask &= np.isfinite(ttc[neg_idx]) & np.isfinite(ttc[pos_i])
        mask &= (ttc[neg_idx] >= 0.0) & (ttc[pos_i] >= 0.0)
        mask &= np.abs(ttc[neg_idx] - ttc[pos_i]) <= float(setting["ttc_s"])

    if setting.get("no_ttc"):
        mask &= np.isfinite(ttc[neg_idx]) & np.isfinite(ttc[pos_i])
        mask &= (ttc[neg_idx] < 0.0) & (ttc[pos_i] < 0.0)

    if setting.get("large_or_no_ttc"):
        mask &= np.isfinite(ttc[neg_idx]) & np.isfinite(ttc[pos_i])
        mask &= ((ttc[neg_idx] < 0.0) | (ttc[neg_idx] > 3.0))
        mask &= ((ttc[pos_i] < 0.0) | (ttc[pos_i] > 3.0))

    return mask


def _standardized_distance(values: dict[str, np.ndarray], pos_i: int, cand_idx: np.ndarray, setting: dict) -> np.ndarray:
    parts = []
    for col, scale_key in setting["dims"]:
        scale = max(float(setting[scale_key]), 1e-9)
        parts.append((values[col][cand_idx] - values[col][pos_i]) / scale)
    X = np.vstack(parts)
    return np.sqrt(np.sum(X * X, axis=0))


def _build_pairs(df: pd.DataFrame, y: np.ndarray, setting: dict, max_pairs: int) -> pd.DataFrame:
    values = {
        "current_min_distance_m": _num(df, "current_min_distance_m"),
        "current_ttc_s": _num(df, "current_ttc_s"),
        "ego_speed_kph": _num(df, "ego_speed_kph"),
        "agent_count": _num(df, "agent_count"),
    }
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rows = []
    for pos_i in pos_idx:
        mask = _candidate_mask(values, int(pos_i), neg_idx, setting)
        if not np.any(mask):
            continue
        cand_idx = neg_idx[mask]
        dist_z = _standardized_distance(values, int(pos_i), cand_idx, setting)
        best = int(cand_idx[int(np.nanargmin(dist_z))])
        rows.append({
            "setting_name": setting["setting_name"],
            "family": setting["family"],
            "pos_index": int(pos_i),
            "neg_index": best,
            "match_distance_z": float(np.nanmin(dist_z)),
            "pos_sample_id": str(df.iloc[int(pos_i)].get("sample_id", pos_i)),
            "neg_sample_id": str(df.iloc[best].get("sample_id", best)),
        })
        if len(rows) >= int(max_pairs):
            break
    return pd.DataFrame(rows)


def _add_pair_columns(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    cols = [
        "current_min_distance_m",
        "current_ttc_s",
        "ego_speed_kph",
        "agent_count",
        "label_id",
        "label_name",
        "redi_actionability",
        "redi_full",
        "asr_cum_final",
        "asr_slice_min",
        "comfort_asr",
        "ttad_s",
        "early_blocking_ratio",
    ]
    for col in cols:
        if col in df.columns:
            out[f"pos_{col}"] = df.iloc[out["pos_index"].to_numpy(int)][col].to_numpy()
            out[f"neg_{col}"] = df.iloc[out["neg_index"].to_numpy(int)][col].to_numpy()
    return out


def _balance_rows(df: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    rows = []
    for setting_name, sub in pairs.groupby("setting_name"):
        pos_idx = sub["pos_index"].to_numpy(int)
        neg_idx = sub["neg_index"].to_numpy(int)
        for col in ["current_min_distance_m", "current_ttc_s", "ego_speed_kph", "agent_count"]:
            vals = _num(df, col)
            pos = vals[pos_idx]
            neg = vals[neg_idx]
            diff = np.abs(pos - neg)
            rows.append({
                "setting_name": setting_name,
                "variable": col,
                "pos_mean": float(np.nanmean(pos)),
                "neg_mean": float(np.nanmean(neg)),
                "mean_abs_diff": float(np.nanmean(diff)),
                "p95_abs_diff": float(np.nanpercentile(diff, 95)),
            })
    return rows


def _metric_rows(df: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    scores = score_definitions(df)
    rows = []
    balance = pd.DataFrame(_balance_rows(df, pairs))
    balance_wide = {}
    if not balance.empty:
        for setting_name, sub in balance.groupby("setting_name"):
            balance_wide[setting_name] = {
                f"{r.variable}_mean_abs_diff": float(r.mean_abs_diff)
                for r in sub.itertuples(index=False)
            }
    for setting_name, sub in pairs.groupby("setting_name"):
        pos_idx = sub["pos_index"].to_numpy(int)
        neg_idx = sub["neg_index"].to_numpy(int)
        base = {
            "setting_name": setting_name,
            "n_pairs": int(len(sub)),
            "unique_positive_count": int(sub["pos_index"].nunique()),
            "unique_negative_count": int(sub["neg_index"].nunique()),
            "unstable": bool(len(sub) < 100),
        }
        base.update(balance_wide.get(setting_name, {}))
        for score_name in SCORE_NAMES:
            if score_name not in scores:
                continue
            score = scores[score_name]
            pos_s = score[pos_idx]
            neg_s = score[neg_idx]
            valid = np.isfinite(pos_s) & np.isfinite(neg_s)
            if not valid.any():
                continue
            win = pos_s[valid] > neg_s[valid]
            tie = np.isclose(pos_s[valid], neg_s[valid])
            y_pair = np.concatenate([np.ones(valid.sum(), dtype=int), np.zeros(valid.sum(), dtype=int)])
            s_pair = np.concatenate([pos_s[valid], neg_s[valid]])
            row = dict(base)
            row.update({
                "score_name": score_name,
                "score_valid_pairs": int(valid.sum()),
                "matched_win_rate": float(np.mean(win)),
                "matched_tie_rate": float(np.mean(tie)),
                "matched_loss_rate": float(np.mean(~win & ~tie)),
                "auroc": float(roc_auc_score(y_pair, s_pair)),
                "auprc": float(average_precision_score(y_pair, s_pair)),
            })
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Emergency-only caliper matched sensitivity analysis.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--features-csv", default=None)
    ap.add_argument("--task", default="emergency_only", choices=["emergency_only", "warning_or_above", "safe_vs_risky"])
    ap.add_argument("--max-pairs", type=int, default=50000)
    ap.add_argument("--out-name", default="emergency_caliper_sensitivity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "nc_matched_ttc_distance")
    df = _load_features(work, args.features_csv).copy().reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)
    y = y_for_task(df["label_id"].to_numpy(), args.task)
    if len(np.unique(y)) < 2:
        raise SystemExit(f"Need both positive and negative examples for task={args.task}")

    pair_frames = []
    for setting in _settings():
        pairs = _build_pairs(df, y, setting, args.max_pairs)
        if not pairs.empty:
            pair_frames.append(pairs)
    if not pair_frames:
        raise SystemExit("No caliper matched pairs found.")

    pairs_all = pd.concat(pair_frames, ignore_index=True)
    pairs_all = _add_pair_columns(df, pairs_all)
    balance = pd.DataFrame(_balance_rows(df, pairs_all))
    metrics = pd.DataFrame(_metric_rows(df, pairs_all))

    prefix = args.out_name
    pairs_all.to_csv(out / f"{prefix}_pairs.csv", index=False)
    metrics.to_csv(out / f"{prefix}_metrics.csv", index=False)
    balance.to_csv(out / f"{prefix}_balance.csv", index=False)
    print(f"[nc-caliper-sensitivity] wrote {out}; pairs={len(pairs_all)}; settings={pairs_all['setting_name'].nunique()}")


if __name__ == "__main__":
    main()
