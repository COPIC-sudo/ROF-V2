from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.analysis_utils import load_merged
from rtbev.nc_eval import binary_score_metrics, score_definitions, y_for_task, numeric_frame


def _load_features(work: Path, features_csv: str | None) -> pd.DataFrame:
    return pd.read_csv(features_csv) if features_csv else load_merged(work)


def _num(df: pd.DataFrame, c: str, default=np.nan) -> pd.Series:
    if c not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _prep_match_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = ["current_min_distance_m", "current_ttc_s", "ego_speed_kph", "agent_count", "nearby_agent_count_20m", "nearest_agent_rel_speed_mps"]
    cols = [c for c in cols if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "current_ttc_s" in X.columns:
        t = X["current_ttc_s"]
        finite = t[(t >= 0) & np.isfinite(t)]
        large = max(float(finite.quantile(0.95)) if len(finite) else 10.0, 10.0)
        X.loc[(t < 0) | (~np.isfinite(t)), "current_ttc_s"] = large
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(float(med if np.isfinite(med) else 0.0))
    return X.to_numpy(float), cols


def _balance_table(df: pd.DataFrame, pairs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    if pairs.empty:
        return pd.DataFrame()
    for c in cols:
        if c not in df.columns:
            continue
        pos = _num(df.iloc[pairs["pos_index"].to_numpy(int)], c).to_numpy(float)
        neg = _num(df.iloc[pairs["neg_index"].to_numpy(int)], c).to_numpy(float)
        rows.append({
            "variable": c,
            "pos_mean": float(np.nanmean(pos)), "neg_mean": float(np.nanmean(neg)),
            "mean_abs_diff": float(np.nanmean(np.abs(pos - neg))),
            "p95_abs_diff": float(np.nanpercentile(np.abs(pos - neg), 95)),
        })
    return pd.DataFrame(rows)


def _nearest_pairs(df: pd.DataFrame, y: np.ndarray, max_pairs: int, neighbors: int, seed: int) -> pd.DataFrame:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return pd.DataFrame()
    X, match_cols = _prep_match_matrix(df)
    Xs = StandardScaler().fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=min(neighbors, len(neg_idx)), algorithm="auto").fit(Xs[neg_idx])
    dist, nn = nbrs.kneighbors(Xs[pos_idx])
    rows = []
    rng = np.random.default_rng(seed)
    order = np.arange(len(pos_idx)); rng.shuffle(order)
    for p_local in order:
        ii = int(pos_idx[p_local])
        for jj_local, dd in zip(nn[p_local], dist[p_local]):
            jj = int(neg_idx[jj_local])
            rows.append({"method": "nearest_neighbor", "pos_index": ii, "neg_index": jj, "match_distance_z": float(dd), "pos_sample_id": df.iloc[ii]["sample_id"], "neg_sample_id": df.iloc[jj]["sample_id"]})
            break
        if len(rows) >= max_pairs:
            break
    return pd.DataFrame(rows)


def _relaxed_bin_pairs(df: pd.DataFrame, y: np.ndarray, cfg: dict, max_pairs: int, seed: int) -> pd.DataFrame:
    ncfg = cfg.get("nc_experiments", {})
    d_bin = float(ncfg.get("relaxed_distance_bin_m", 3.0))
    t_bin = float(ncfg.get("relaxed_ttc_bin_s", 2.0))
    s_bin = float(ncfg.get("relaxed_speed_bin_kph", 10.0))
    a_bin = float(ncfg.get("relaxed_agent_count_bin", 10.0))
    dist = _num(df, "current_min_distance_m").fillna(999.0)
    ttc = _num(df, "current_ttc_s")
    ttc_b = ttc.copy(); ttc_b[(ttc_b < 0) | (~np.isfinite(ttc_b))] = -1.0
    speed = _num(df, "ego_speed_kph").fillna(0.0)
    agents = _num(df, "agent_count").fillna(0.0)
    bin_key = (
        np.floor(dist / d_bin).clip(0, 50).astype(int).astype(str) + "|" +
        np.where(ttc_b < 0, "no", np.floor(ttc_b / t_bin).clip(0, 50).astype(int).astype(str)) + "|" +
        np.floor(speed / s_bin).clip(0, 30).astype(int).astype(str) + "|" +
        np.floor(agents / a_bin).clip(0, 20).astype(int).astype(str)
    )
    rng = np.random.default_rng(seed)
    rows = []
    for key, idxs in pd.Series(np.arange(len(df))).groupby(bin_key).groups.items():
        idxs = np.asarray(list(idxs), dtype=int)
        pos = idxs[y[idxs] == 1]
        neg = idxs[y[idxs] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        rng.shuffle(pos); rng.shuffle(neg)
        n = min(len(pos), len(neg), max_pairs - len(rows))
        for ii, jj in zip(pos[:n], neg[:n]):
            rows.append({"method": "relaxed_bin", "match_bin": key, "pos_index": int(ii), "neg_index": int(jj), "pos_sample_id": df.iloc[ii]["sample_id"], "neg_sample_id": df.iloc[jj]["sample_id"]})
        if len(rows) >= max_pairs:
            break
    return pd.DataFrame(rows)


def _evaluate_pairs(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    scores = score_definitions(df)
    rows = []
    if pairs.empty:
        return pd.DataFrame()
    for score_name, score in scores.items():
        pos_s = score[pairs["pos_index"].to_numpy(int)]
        neg_s = score[pairs["neg_index"].to_numpy(int)]
        valid = np.isfinite(pos_s) & np.isfinite(neg_s)
        if not valid.any():
            continue
        win = pos_s[valid] > neg_s[valid]
        tie = np.isclose(pos_s[valid], neg_s[valid])
        pair_score = np.concatenate([pos_s[valid], neg_s[valid]])
        pair_y = np.concatenate([np.ones(valid.sum(), dtype=int), np.zeros(valid.sum(), dtype=int)])
        row = {
            "score_name": score_name,
            "n_pairs": int(valid.sum()),
            "matched_win_rate": float(np.mean(win)),
            "matched_tie_rate": float(np.mean(tie)),
            "matched_loss_rate": float(np.mean(~win & ~tie)),
        }
        if len(np.unique(pair_y)) == 2:
            row["conditional_auroc"] = float(roc_auc_score(pair_y, pair_score))
            row["conditional_auprc"] = float(average_precision_score(pair_y, pair_score))
        rows.append(row)
    return pd.DataFrame(rows)


def _residual_analysis(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    scores = score_definitions(df)
    base_cols = [c for c in ["current_min_distance_m", "current_ttc_s", "ego_speed_kph", "agent_count", "nearest_agent_rel_speed_mps"] if c in df.columns]
    if not base_cols or len(np.unique(y)) < 2:
        return pd.DataFrame()
    X = numeric_frame(df, base_cols).to_numpy(float)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    residual = y - p
    rows = []
    for name, s in scores.items():
        ok = np.isfinite(s) & np.isfinite(residual)
        if ok.sum() < 20:
            continue
        rows.append({
            "score_name": name,
            "n": int(ok.sum()),
            "pearson_with_baseline_residual": float(np.corrcoef(s[ok], residual[ok])[0, 1]) if np.std(s[ok]) > 0 else np.nan,
            **binary_score_metrics(y, 0.5 * p + 0.5 * s, fpr_levels=(0.01, 0.05)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="NC v1.1 mechanism test: relaxed/nearest matched TTC-distance analysis plus residual analysis.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--features-csv", default=None)
    ap.add_argument("--task", choices=["warning_or_above", "emergency_only", "safe_vs_risky"], default="warning_or_above")
    ap.add_argument("--max-pairs", type=int, default=50000)
    ap.add_argument("--neighbors", type=int, default=10)
    ap.add_argument("--method", choices=["both", "nearest", "relaxed_bin"], default="both")
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
    if args.method in ["both", "nearest"]:
        pair_frames.append(_nearest_pairs(df, y, args.max_pairs, args.neighbors, int(cfg.get("analysis", {}).get("random_seed", 42))))
    if args.method in ["both", "relaxed_bin"]:
        pair_frames.append(_relaxed_bin_pairs(df, y, cfg, args.max_pairs, int(cfg.get("analysis", {}).get("random_seed", 42)) + 1))
    pairs = pd.concat([p for p in pair_frames if p is not None and not p.empty], ignore_index=True) if pair_frames else pd.DataFrame()
    if pairs.empty:
        raise SystemExit("No matched pairs found; enlarge sample size or loosen relaxed bins in nc_experiments.")

    for col in ["current_min_distance_m", "current_ttc_s", "ego_speed_kph", "agent_count", "label_id", "label_name", "redi_actionability", "redi_full", "asr_cum_final", "asr_slice_min", "ttad_s"]:
        if col in df.columns:
            pairs[f"pos_{col}"] = df.iloc[pairs["pos_index"].to_numpy(int)][col].to_numpy()
            pairs[f"neg_{col}"] = df.iloc[pairs["neg_index"].to_numpy(int)][col].to_numpy()
    pairs.to_csv(out / f"matched_pairs_{args.task}.csv", index=False)
    score_rows = []
    for method, sub in pairs.groupby("method"):
        m = _evaluate_pairs(df, sub)
        if not m.empty:
            m.insert(0, "method", method)
            m.insert(0, "task", args.task)
            score_rows.append(m)
        X, cols = _prep_match_matrix(df)
        _balance_table(df, sub, cols).assign(task=args.task, method=method).to_csv(out / f"matched_balance_{args.task}_{method}.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(out / f"matched_score_metrics_{args.task}.csv", index=False)
    _residual_analysis(df, y).to_csv(out / f"residual_risk_analysis_{args.task}.csv", index=False)
    print(f"[nc-matched-v1.1] wrote {out}; pairs={len(pairs)}")


if __name__ == "__main__":
    main()
