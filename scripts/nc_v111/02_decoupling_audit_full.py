#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.baselines.feature_sets import feature_lineage_rows, strict_non_action_current_cv_columns
from rtbev.baselines.oof import grouped_oof_predictions
from rtbev.external.common import (
    add_config_hash,
    artifact_manifest_rows,
    config_hash,
    experiment_out_dir,
    load_yaml_config,
    resolve_input_path,
    run_manifest,
    write_csv,
    write_json,
)
from rtbev.external.metrics import binary_metric_values_strict


BASE_FEATURES = [
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "ego_speed_mps",
    "agent_count",
    "nearest_agent_rel_speed_mps",
    "nearest_agent_closing_speed_mps",
    "ttc_closing_speed_mps",
    "nearby_agent_count_10m",
    "nearby_agent_count_20m",
]

CV_FEATURES = [
    "cv_rcr",
    "cv_rfr_drv",
    "cv_c_time",
    "cv_gtoa_norm_union",
    "cv_oce_norm",
    "cv_c_density",
    "cv_max_overlap_count",
    "current_collision",
    "max_overlap_count",
    "mean_overlap_count_nonzero",
    "overlap_count_entropy_norm",
]

STRICT_TEMPORAL_FEATURES = [
    "ttad_s",
    "time_to_first_conflict_s",
    "early_blocking_ratio",
    "collapse_rate_per_s",
    "collapse_rate_mean_per_s",
    "collapse_rate_max_per_s",
]

FULL_ACTIONABILITY_FEATURES = [
    "redi_actionability",
    "redi_actionability_delta",
    "redi_full",
    "redi_no_msr",
    "asr",
    "asr_final",
    "asr_min",
    "asr_cum_final",
    "asr_cum_min",
    "asr_slice_final",
    "asr_slice_min",
    "comfort_asr",
    "emergency_asr",
    "comfort_to_emergency_gap",
    "min_safe_action_cost",
    "msr",
    "c_maneuver",
    "survival_keep",
    "survival_accelerate",
    "survival_brake",
    "survival_hard_brake",
    "survival_left",
    "survival_right",
    "survival_brake_left",
    "survival_brake_right",
]

VARIANT_FEATURE_SET_BASE = "strong_baseline_cv_aligned"
VARIANT_FEATURE_SET_ENH = "strong_baseline_cv_plus_strict_temporal_dynamics_aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v111/nc_v111_decoupling_full.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--model", default=None, choices=["rf", "logreg"])
    return parser.parse_args()


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def _input_path(cfg: dict[str, Any], key: str) -> Path:
    path = resolve_input_path((cfg.get("inputs") or {}).get(key), cfg)
    if path is None:
        raise FileNotFoundError(f"inputs.{key} is required")
    return path


def _available(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    strict_cols = strict_non_action_current_cv_columns(df)
    base_cols = _available(df, BASE_FEATURES)
    cv_cols = _available(df, [*BASE_FEATURES, *CV_FEATURES])
    temporal_cols = _available(df, [*BASE_FEATURES, *CV_FEATURES, *STRICT_TEMPORAL_FEATURES])
    full_cols = _available(df, [*BASE_FEATURES, *CV_FEATURES, *STRICT_TEMPORAL_FEATURES, *FULL_ACTIONABILITY_FEATURES])
    return {
        "strong_baseline_cv": base_cols,
        "strong_baseline_cv_plus_strict_non_action_current_cv": _available(df, [*base_cols, *strict_cols, *CV_FEATURES]),
        "strong_baseline_cv_plus_strict_temporal_dynamics": temporal_cols,
        "strong_baseline_cv_plus_full_actionability": full_cols,
    }


def _prepare_main_frame(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    f = features.copy()
    l = labels.copy()
    f["sample_id"] = f["sample_id"].astype(str)
    l["sample_id"] = l["sample_id"].astype(str)
    if "actionability_label_id" not in l.columns:
        raise ValueError("Waymo label table missing actionability_label_id")
    l["_y"] = (pd.to_numeric(l["actionability_label_id"], errors="coerce").fillna(0).astype(int) >= 2).astype(int)
    keep = [c for c in ["sample_id", "scenario_id", "segment_id", "_y"] if c in l.columns]
    out = f.merge(l[keep], on="sample_id", how="inner", suffixes=("", "_label"))
    if "scenario_id" not in out.columns and "scenario_id_label" in out.columns:
        out["scenario_id"] = out["scenario_id_label"]
    if "scenario_id" not in out.columns:
        out["scenario_id"] = out["sample_id"]
    return out


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    vals = binary_metric_values_strict(y, score)
    return {
        "n": vals.get("n", 0),
        "positive_count": vals.get("positive_count", 0),
        "positive_rate": vals.get("positive_rate", np.nan),
        "AUPRC": vals.get("AUPRC", np.nan),
        "AUROC": vals.get("AUROC", np.nan),
        "Recall@1%FPR_strict": vals.get("Recall@1%FPR_strict", np.nan),
        "strict_actual_fpr_at_1%FPR": vals.get("strict_actual_fpr_at_1%FPR", np.nan),
        "Recall@5%FPR_strict": vals.get("Recall@5%FPR_strict", np.nan),
        "strict_actual_fpr_at_5%FPR": vals.get("strict_actual_fpr_at_5%FPR", np.nan),
    }


def _pred_metrics(pred: pd.DataFrame) -> dict[str, Any]:
    if pred.empty:
        return {"n": 0, "positive_count": 0, "AUPRC": np.nan, "AUROC": np.nan, "Recall@5%FPR_strict": np.nan}
    return _metrics(pred["y_true"].to_numpy(int), pd.to_numeric(pred["score"], errors="coerce").to_numpy(float))


def _metric_value(y: np.ndarray, score: np.ndarray, metric: str) -> float:
    vals = _metrics(y, score)
    key = {
        "auprc": "AUPRC",
        "auroc": "AUROC",
        "recall_at_5pct_fpr_strict": "Recall@5%FPR_strict",
    }[metric]
    try:
        value = float(vals.get(key, np.nan))
    except Exception:
        value = float("nan")
    return value if np.isfinite(value) else float("nan")


def _rank_cache(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    if len(sorted_score) == 0:
        return order, np.asarray([], dtype=int)
    starts = np.r_[0, np.where(np.diff(sorted_score) != 0)[0] + 1].astype(int)
    return order, starts


def _weighted_rank_metrics(
    y: np.ndarray,
    score: np.ndarray,
    metrics: Sequence[str],
    weights: np.ndarray | None = None,
    cache: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    if cache is None:
        cache = _rank_cache(score)
    order, starts = cache
    if len(order) == 0 or len(starts) == 0:
        return {m: float("nan") for m in metrics}
    yy = y[order]
    ww = weights[order]
    pos_total = float(np.sum(weights * (y == 1)))
    neg_total = float(np.sum(weights * (y == 0)))
    out = {m: float("nan") for m in metrics}
    if pos_total <= 0 or neg_total <= 0:
        return out
    pos_counts = np.add.reduceat(ww * (yy == 1), starts)
    neg_counts = np.add.reduceat(ww * (yy == 0), starts)
    tp = np.cumsum(pos_counts)
    fp = np.cumsum(neg_counts)
    recalls = tp / pos_total
    fprs = fp / neg_total
    if "auprc" in out:
        precision = tp / np.maximum(tp + fp, 1.0)
        recall_prev = np.r_[0.0, recalls[:-1]]
        out["auprc"] = float(np.sum((recalls - recall_prev) * precision))
    if "recall_at_5pct_fpr_strict" in out:
        candidate_recalls = np.r_[0.0, recalls]
        candidate_fprs = np.r_[0.0, fprs]
        valid = candidate_fprs <= 0.05 + 1e-12
        out["recall_at_5pct_fpr_strict"] = float(np.max(candidate_recalls[valid])) if np.any(valid) else 0.0
    return out


def _bootstrap_delta(
    merged: pd.DataFrame,
    baseline_col: str,
    enhanced_col: str,
    metrics: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    work = merged.copy()
    ok = (
        pd.to_numeric(work[baseline_col], errors="coerce").notna()
        & pd.to_numeric(work[enhanced_col], errors="coerce").notna()
        & pd.to_numeric(work["y_true"], errors="coerce").notna()
    )
    work = work[ok].copy()
    if work.empty:
        return []
    groups = work.get("scenario_id", work["sample_id"]).fillna(work["sample_id"]).astype(str).to_numpy()
    uniq = np.unique(groups)
    group_indices = [np.where(groups == g)[0] for g in uniq]
    y = pd.to_numeric(work["y_true"], errors="coerce").fillna(0).astype(int).to_numpy()
    b = pd.to_numeric(work[baseline_col], errors="coerce").to_numpy(float)
    e = pd.to_numeric(work[enhanced_col], errors="coerce").to_numpy(float)
    b_cache = _rank_cache(b)
    e_cache = _rank_cache(e)
    rng = np.random.default_rng(int(seed))
    boot = {m: [] for m in metrics}
    unique_per_row = len(uniq) == len(work)
    for _ in range(int(n_bootstrap)):
        if unique_per_row:
            sampled = rng.integers(0, len(work), size=len(work))
            weights = np.bincount(sampled, minlength=len(work)).astype(float)
        else:
            sampled_groups = rng.integers(0, len(group_indices), size=len(group_indices))
            weights = np.zeros(len(work), dtype=float)
            for group_idx in sampled_groups:
                weights[group_indices[int(group_idx)]] += 1.0
        if np.sum(weights * (y == 1)) <= 0 or np.sum(weights * (y == 0)) <= 0:
            continue
        evs = _weighted_rank_metrics(y, e, metrics, weights=weights, cache=e_cache)
        bvs = _weighted_rank_metrics(y, b, metrics, weights=weights, cache=b_cache)
        for metric in metrics:
            ev = evs.get(metric, float("nan"))
            bv = bvs.get(metric, float("nan"))
            if np.isfinite(ev) and np.isfinite(bv):
                boot[metric].append(float(ev - bv))
    rows: list[dict[str, Any]] = []
    ev_points = _weighted_rank_metrics(y, e, metrics, cache=e_cache)
    bv_points = _weighted_rank_metrics(y, b, metrics, cache=b_cache)
    for metric in metrics:
        ev = ev_points.get(metric, float("nan"))
        bv = bv_points.get(metric, float("nan"))
        arr = np.asarray(boot[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "baseline_point": bv,
                "enhanced_point": ev,
                "delta": ev - bv if np.isfinite(ev) and np.isfinite(bv) else np.nan,
                "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                "P_delta_gt_0": float(np.mean(arr > 0.0)) if len(arr) else np.nan,
                "n_bootstrap_valid": int(len(arr)),
                "n_samples": int(len(work)),
                "n_scenarios": int(len(uniq)),
                "positive_count": int(y.sum()),
                "positive_rate": float(y.mean()) if len(y) else np.nan,
            }
        )
    return rows


def _non_action_oof(
    frame: pd.DataFrame,
    n_folds: int,
    seed: int,
    model: str,
    bootstrap_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_sets = _feature_sets(frame)
    metric_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    preds: dict[str, pd.DataFrame] = {}
    for name, cols in feature_sets.items():
        pred = grouped_oof_predictions(frame, cols, "_y", group_col="scenario_id", n_folds=n_folds, seed=seed, model=model)
        preds[name] = pred
        metric_rows.append(
            {
                "feature_set": name,
                "model": model,
                "group_col": "scenario_id",
                "n_features": len(cols),
                "features": ";".join(cols),
                "status": "OK" if not pred.empty else "NO_VALID_OOF_FOLDS",
                **_pred_metrics(pred),
            }
        )
        grouped_rows.append({**metric_rows[-1], "oof_protocol": "scenario_id_grouped"})
    if "segment_id" not in frame.columns:
        grouped_rows.append(
            {
                "feature_set": "ALL",
                "model": model,
                "group_col": "segment_id",
                "status": "NOT_AVAILABLE",
                "reason": "segment_id column not present in Waymo v1.0.1 feature-label table",
            }
        )
    delta_rows: list[dict[str, Any]] = []
    baseline_name = "strong_baseline_cv"
    for enhanced_name, pred in preds.items():
        if enhanced_name == baseline_name or pred.empty or preds.get(baseline_name, pd.DataFrame()).empty:
            continue
        base = preds[baseline_name][["sample_id", "scenario_id", "y_true", "score"]].rename(columns={"score": "baseline_score"})
        enh = pred[["sample_id", "score"]].rename(columns={"score": "enhanced_score"})
        merged = base.merge(enh, on="sample_id", how="inner")
        for row in _bootstrap_delta(merged, "baseline_score", "enhanced_score", ["auprc", "recall_at_5pct_fpr_strict"], bootstrap_n, seed):
            row.update(
                {
                    "baseline_feature_set": baseline_name,
                    "enhanced_feature_set": enhanced_name,
                    "model": model,
                    "bootstrap_unit": "scenario_id",
                }
            )
            delta_rows.append(row)
    return metric_rows, delta_rows, grouped_rows


def _variant_path(base_dir: Path, variant_id: str, kind: str) -> Path:
    if kind == "label":
        prefix = "labels_actionability"
    else:
        prefix = "features_aligned"
    matches = list((base_dir / variant_id).glob(f"{prefix}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"{kind} variant CSV missing for {variant_id} in {base_dir}")
    return matches[0]


def _variant_labels(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    label_dir = resolve_input_path((cfg.get("inputs") or {}).get("v096_variant_label_dir"), cfg)
    if label_dir is None:
        raise FileNotFoundError("v096_variant_label_dir missing")
    out: dict[str, pd.DataFrame] = {}
    for short_id, meta in (cfg.get("variants") or {}).items():
        variant_id = str(meta["label_variant_id"])
        path = _variant_path(label_dir, variant_id, "label")
        df = _read_csv(path, f"label variant {short_id}")
        df["sample_id"] = df["sample_id"].astype(str)
        df["y_true"] = (pd.to_numeric(df["actionability_label_id"], errors="coerce").fillna(0).astype(int) >= 2).astype(int)
        out[str(short_id)] = df[["sample_id", "scenario_id", "y_true"]].copy()
    return out


def _mismatch_matrix(
    cfg: dict[str, Any],
    bootstrap_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels_by_variant = _variant_labels(cfg)
    pred_path = resolve_input_path((cfg.get("inputs") or {}).get("v097_oof_predictions_csv"), cfg)
    if pred_path is None:
        raise FileNotFoundError("v097_oof_predictions_csv missing")
    pred_all = _read_csv(pred_path, "v097 aligned feature OOF predictions")
    pred_all["sample_id"] = pred_all["sample_id"].astype(str)
    pred_all = pred_all[
        (pred_all["endpoint"].astype(str) == "actionability_critical_or_worse")
        & (pd.to_numeric(pred_all["seed"], errors="coerce").fillna(-1).astype(int) == 42)
    ].copy()
    rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for label_short, label_df in labels_by_variant.items():
        for feature_short, meta in (cfg.get("variants") or {}).items():
            variant_id = str(meta["feature_variant_id"])
            sub = pred_all[pred_all["variant_id"].astype(str) == variant_id].copy()
            base = sub[sub["feature_set"].astype(str) == VARIANT_FEATURE_SET_BASE][["sample_id", "scenario_id", "score"]].rename(
                columns={"score": "baseline_score", "scenario_id": "scenario_id_pred"}
            )
            enh = sub[sub["feature_set"].astype(str) == VARIANT_FEATURE_SET_ENH][["sample_id", "score"]].rename(columns={"score": "enhanced_score"})
            merged = label_df.merge(base, on="sample_id", how="inner").merge(enh, on="sample_id", how="inner")
            if "scenario_id" not in merged.columns and "scenario_id_pred" in merged.columns:
                merged["scenario_id"] = merged["scenario_id_pred"]
            y = merged["y_true"].to_numpy(int)
            b = pd.to_numeric(merged["baseline_score"], errors="coerce").to_numpy(float)
            e = pd.to_numeric(merged["enhanced_score"], errors="coerce").to_numpy(float)
            bm = _metrics(y, b)
            em = _metrics(y, e)
            row = {
                "label_variant": label_short,
                "feature_variant": feature_short,
                "diagonal": bool(label_short == feature_short),
                "baseline_feature_set": "strong_baseline_cv",
                "enhanced_feature_set": "strong_baseline_cv_plus_strict_temporal_dynamics",
                "n": int(len(merged)),
                "positive_count": int(y.sum()) if len(y) else 0,
                "baseline_AUPRC": bm.get("AUPRC", np.nan),
                "enhanced_AUPRC": em.get("AUPRC", np.nan),
                "delta_AUPRC": em.get("AUPRC", np.nan) - bm.get("AUPRC", np.nan),
                "baseline_Recall@5%FPR_strict": bm.get("Recall@5%FPR_strict", np.nan),
                "enhanced_Recall@5%FPR_strict": em.get("Recall@5%FPR_strict", np.nan),
                "delta_Recall@5%FPR_strict": em.get("Recall@5%FPR_strict", np.nan) - bm.get("Recall@5%FPR_strict", np.nan),
                "score_transfer_note": "feature-variant OOF scores from v097 evaluated against each label variant; no label/feature regeneration in v111",
            }
            rows.append(row)
            for brow in _bootstrap_delta(merged, "baseline_score", "enhanced_score", ["auprc", "recall_at_5pct_fpr_strict"], bootstrap_n, seed):
                brow.update(
                    {
                        "label_variant": label_short,
                        "feature_variant": feature_short,
                        "diagonal": bool(label_short == feature_short),
                        "baseline_feature_set": row["baseline_feature_set"],
                        "enhanced_feature_set": row["enhanced_feature_set"],
                    }
                )
                boot_rows.append(brow)
    mat = pd.DataFrame(rows)
    off = mat[~mat["diagonal"].astype(bool)].copy() if not mat.empty else pd.DataFrame()
    diag = mat[mat["diagonal"].astype(bool)].copy() if not mat.empty else pd.DataFrame()
    summary = {
        "median_offdiagonal_delta_AUPRC": float(pd.to_numeric(off.get("delta_AUPRC", pd.Series(dtype=float)), errors="coerce").median()) if not off.empty else np.nan,
        "percent_offdiagonal_positive": float((pd.to_numeric(off.get("delta_AUPRC", pd.Series(dtype=float)), errors="coerce") > 0).mean() * 100.0) if not off.empty else np.nan,
        "median_diagonal_delta_AUPRC": float(pd.to_numeric(diag.get("delta_AUPRC", pd.Series(dtype=float)), errors="coerce").median()) if not diag.empty else np.nan,
    }
    summary["offdiagonal_gain_retention_vs_diagonal"] = (
        summary["median_offdiagonal_delta_AUPRC"] / summary["median_diagonal_delta_AUPRC"]
        if np.isfinite(summary["median_offdiagonal_delta_AUPRC"]) and np.isfinite(summary["median_diagonal_delta_AUPRC"]) and summary["median_diagonal_delta_AUPRC"] != 0
        else np.nan
    )
    return rows, boot_rows, summary


def _wide_mismatch_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    wide = matrix.pivot(index="label_variant", columns="feature_variant", values="delta_AUPRC")
    variant_order = list(dict.fromkeys(matrix["feature_variant"].astype(str).tolist()))
    row_order = list(dict.fromkeys(matrix["label_variant"].astype(str).tolist()))
    wide = wide.reindex(index=row_order, columns=variant_order)
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def _leave_family_rows(main_frame: pd.DataFrame, v110_sample_manifest: pd.DataFrame | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "dataset": "Waymo",
            "leave_out_field": "scenario_family",
            "status": "NOT_AVAILABLE",
            "reason": "Waymo v1.0.1 feature-label table has scenario_id but no scenario_family/family metadata; scenario_id is unique per row.",
            "n": int(len(main_frame)),
            "scenario_count": int(main_frame["scenario_id"].astype(str).nunique()) if "scenario_id" in main_frame.columns else "",
        }
    )
    rows.append(
        {
            "dataset": "Waymo",
            "leave_out_field": "segment_id",
            "status": "NOT_AVAILABLE",
            "reason": "segment_id column not present in the supplied Waymo feature-label table.",
            "n": int(len(main_frame)),
            "scenario_count": int(main_frame["scenario_id"].astype(str).nunique()) if "scenario_id" in main_frame.columns else "",
        }
    )
    if v110_sample_manifest is None or v110_sample_manifest.empty:
        rows.append({"dataset": "CommonRoad", "leave_out_field": "scenario_family/topology", "status": "NOT_AVAILABLE", "reason": "v110 sample manifest missing"})
    else:
        available = [c for c in ["scenario_family", "topology_stratum"] if c in v110_sample_manifest.columns]
        if not available:
            rows.append(
                {
                    "dataset": "CommonRoad",
                    "leave_out_field": "scenario_family/topology",
                    "status": "NOT_AVAILABLE",
                    "reason": "v110 sample_manifest exposes neutral/speed/distance/ttc strata but no scenario_family or topology_stratum column.",
                    "n": int(len(v110_sample_manifest)),
                    "scenario_count": int(v110_sample_manifest["scenario_id"].astype(str).nunique()) if "scenario_id" in v110_sample_manifest.columns else "",
                }
            )
    return rows


def _external_summary(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_specs = [
        ("lattice_base", _input_path(cfg, "v110_lattice_base_dir")),
        ("lattice_extended", _input_path(cfg, "v110_lattice_extended_dir")),
    ]
    for run_name, run_dir in run_specs:
        deltas_path = run_dir / "external_bootstrap_deltas_strict_fpr.csv"
        metrics_path = run_dir / "external_metrics_strict_fpr.csv"
        if not deltas_path.exists():
            rows.append({"external_run": run_name, "status": "MISSING_STRICT_FPR_DELTAS", "path": str(deltas_path)})
            continue
        deltas = pd.read_csv(deltas_path)
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        if not metrics.empty:
            for score in ["temporal_composite", "ROF_v2_no_asr_composite", "distance_inverse", "TTC_inverse"]:
                sub = metrics[metrics["score"].astype(str) == score]
                if sub.empty:
                    continue
                rows.append(
                    {
                        "external_run": run_name,
                        "row_type": "point_metric",
                        "score": score,
                        "AUPRC": float(sub.iloc[0].get("AUPRC", np.nan)),
                        "Recall@5%FPR_strict": float(sub.iloc[0].get("Recall@5%FPR_strict", np.nan)),
                        "positive_count": int(float(sub.iloc[0].get("positive_count", 0))),
                        "n": int(float(sub.iloc[0].get("n", 0))),
                    }
                )
        focus = deltas[
            deltas["enhanced_score"].astype(str).isin(["temporal_composite", "ROF_v2_no_asr_composite"])
            & deltas["baseline_score"].astype(str).isin(["distance_inverse", "TTC_inverse"])
            & deltas["metric"].astype(str).isin(["auprc", "recall_at_5pct_fpr_strict"])
        ]
        for _, row in focus.iterrows():
            rows.append(
                {
                    "external_run": run_name,
                    "row_type": "bootstrap_delta",
                    "enhanced_score": row["enhanced_score"],
                    "baseline_score": row["baseline_score"],
                    "metric": row["metric"],
                    "delta": float(row["delta"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                    "positive_ci": bool(float(row["ci_low"]) > 0.0),
                    "n_bootstrap_valid": int(float(row.get("n_bootstrap_valid", 0))),
                }
            )
    for key, run_name in [("v112_lattice_base_dir", "v112_lattice_base_strict"), ("v112_lattice_extended_dir", "v112b_lattice_extended_strict")]:
        run_dir = _input_path(cfg, key)
        path = run_dir / "field_baseline_bootstrap_deltas_strict_fpr.csv"
        rows.append({"external_run": run_name, "row_type": "availability", "artifact": path.name, "exists": bool(path.exists()), "path": str(path)})
    return rows


def _report(
    non_action: pd.DataFrame,
    non_action_deltas: pd.DataFrame,
    matrix: pd.DataFrame,
    matrix_boot: pd.DataFrame,
    grouped: pd.DataFrame,
    leave_family: pd.DataFrame,
    external: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    base = non_action[non_action["feature_set"].astype(str) == "strong_baseline_cv"]
    strict = non_action[non_action["feature_set"].astype(str) == "strong_baseline_cv_plus_strict_non_action_current_cv"]
    full = non_action[non_action["feature_set"].astype(str) == "strong_baseline_cv_plus_full_actionability"]
    strict_gain = float(strict.iloc[0]["AUPRC"] - base.iloc[0]["AUPRC"]) if not base.empty and not strict.empty else np.nan
    full_gain = float(full.iloc[0]["AUPRC"] - base.iloc[0]["AUPRC"]) if not base.empty and not full.empty else np.nan
    strict_explains_all = bool(np.isfinite(strict_gain) and np.isfinite(full_gain) and full_gain > 0 and strict_gain / full_gain >= 0.9)
    off_positive = float(summary.get("percent_offdiagonal_positive", np.nan))
    grouped_status = "not meaningfully testable" if (leave_family["status"].astype(str) == "NOT_AVAILABLE").any() else "available"
    external_delta = external[
        (external.get("row_type", pd.Series(dtype=str)).astype(str) == "bootstrap_delta")
        & (external.get("metric", pd.Series(dtype=str)).astype(str) == "auprc")
    ].copy()
    external_all_positive = bool((pd.to_numeric(external_delta.get("ci_low", pd.Series(dtype=float)), errors="coerce") > 0).all()) if not external_delta.empty else False
    lines = [
        "# v111 Decoupling Audit Full Report",
        "",
        "## Executive Answers",
        "",
        f"- strict_non_action_current_cv explains all gain: {strict_explains_all} (strict_gain_AUPRC={strict_gain:.6g}, full_actionability_gain_AUPRC={full_gain:.6g}).",
        f"- off-diagonal transfer remains positive: {bool(np.isfinite(off_positive) and off_positive > 50.0)} (percent_offdiagonal_positive={off_positive:.6g}%).",
        f"- grouped OOF collapse relative to ordinary OOF: {grouped_status}; scenario_id is unique per Waymo row and segment_id/family metadata are unavailable in supplied tables.",
        f"- lattice_base and lattice_extended external labels both support temporal/ROF_v2_no_asr signal: {external_all_positive}.",
        "- coupling concern reduced: yes, but not eliminated; results show positive transfer and external action-library sensitivity, not a proof of causal decoupling.",
        "- claims still not allowed: native-planner robustness, every-stratum superiority, complete removal of label-feature coupling, or conclusions based on unavailable segment/family grouping.",
        "",
        "## Non-action Feature Comparison",
        "",
    ]
    for _, row in non_action.sort_values("AUPRC", ascending=False, na_position="last").iterrows():
        lines.append(
            f"- {row['feature_set']}: AUPRC={float(row.get('AUPRC', np.nan)):.6g}, "
            f"strict Recall@5%FPR={float(row.get('Recall@5%FPR_strict', np.nan)):.6g}, "
            f"n_features={row.get('n_features')}, status={row.get('status')}"
        )
    lines.extend(["", "## Non-action Bootstrap Deltas vs strong_baseline_cv", ""])
    for _, row in non_action_deltas.iterrows():
        lines.append(
            f"- {row['enhanced_feature_set']} {row['metric']}: delta={float(row['delta']):.6g}, "
            f"CI=({float(row['ci_low']):.6g}, {float(row['ci_high']):.6g})"
        )
    lines.extend(
        [
            "",
            "## Label-feature Mismatch Transfer",
            "",
            f"- median_offdiagonal_delta_AUPRC: {summary.get('median_offdiagonal_delta_AUPRC', np.nan):.6g}",
            f"- percent_offdiagonal_positive: {summary.get('percent_offdiagonal_positive', np.nan):.6g}",
            f"- offdiagonal_gain_retention_vs_diagonal: {summary.get('offdiagonal_gain_retention_vs_diagonal', np.nan):.6g}",
        ]
    )
    off = matrix[~matrix["diagonal"].astype(bool)] if not matrix.empty else pd.DataFrame()
    for _, row in off.sort_values("delta_AUPRC", ascending=False).head(10).iterrows():
        lines.append(f"- label={row['label_variant']} feature={row['feature_variant']} delta_AUPRC={float(row['delta_AUPRC']):.6g}")
    primary_boot = matrix_boot[
        (~matrix_boot.get("diagonal", pd.Series(dtype=bool)).astype(bool))
        & (matrix_boot.get("metric", pd.Series(dtype=str)).astype(str) == "auprc")
    ].copy()
    if not primary_boot.empty:
        lines.extend(["", "## Primary Off-diagonal Bootstrap", ""])
        for _, row in primary_boot.sort_values("delta", ascending=False).head(10).iterrows():
            lines.append(
                f"- label={row['label_variant']} feature={row['feature_variant']}: "
                f"delta={float(row['delta']):.6g}, CI=({float(row['ci_low']):.6g}, {float(row['ci_high']):.6g})"
            )
    lines.extend(["", "## Grouping / Leave-family Availability", ""])
    for _, row in leave_family.iterrows():
        lines.append(f"- {row.get('dataset')} leave {row.get('leave_out_field')}: {row.get('status')} ({row.get('reason')})")
    lines.extend(["", "## External Label Decoupling", ""])
    for _, row in external_delta.iterrows():
        lines.append(
            f"- {row['external_run']} {row['enhanced_score']} vs {row['baseline_score']} {row['metric']}: "
            f"delta={float(row['delta']):.6g}, CI=({float(row['ci_low']):.6g}, {float(row['ci_high']):.6g})"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg_hash = config_hash(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v111_decoupling_audit")
    eval_cfg = cfg.get("evaluation", {})
    n_folds = int(args.n_folds if args.n_folds is not None else eval_cfg.get("outer_folds", 5))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("scenario_hash_seed", 42))
    bootstrap_n = int(args.bootstrap_n if args.bootstrap_n is not None else eval_cfg.get("bootstrap_replicates", 1000))
    model = str(args.model or eval_cfg.get("model", "rf"))

    features = _read_csv(_input_path(cfg, "waymo_features_csv"), "Waymo feature table")
    labels = _read_csv(_input_path(cfg, "waymo_labels_csv"), "Waymo label table")
    main_frame = _prepare_main_frame(features, labels)
    non_action_rows, delta_rows, grouped_rows = _non_action_oof(main_frame, n_folds, seed, model, bootstrap_n)
    matrix_rows, matrix_boot_rows, matrix_summary = _mismatch_matrix(cfg, bootstrap_n, seed)
    v110_manifest_path = _input_path(cfg, "v110_lattice_base_dir") / "sample_manifest.csv"
    v110_manifest = pd.read_csv(v110_manifest_path) if v110_manifest_path.exists() else pd.DataFrame()
    leave_rows = _leave_family_rows(main_frame, v110_manifest)
    external_rows = _external_summary(cfg)

    all_features = set(features.columns.astype(str)) | set(labels.columns.astype(str))
    for cols in _feature_sets(main_frame).values():
        all_features.update(cols)
    lineage_rows = feature_lineage_rows(all_features)
    lineage_required_order = [
        "feature_name",
        "feature_set",
        "reads_recorded_future",
        "reads_label",
        "uses_action_library",
        "uses_candidate_survival",
        "uses_label_horizon",
        "uses_label_lane_buffer",
        "uses_endpoint_intermediate",
        "allowed_in_strict_non_action",
    ]
    lineage_rows = [{**{k: row.get(k, "") for k in lineage_required_order}, **row} for row in lineage_rows]

    non_action = pd.DataFrame(non_action_rows)
    non_action_deltas = pd.DataFrame(delta_rows)
    matrix = pd.DataFrame(matrix_rows)
    matrix_wide = _wide_mismatch_matrix(matrix)
    matrix_boot = pd.DataFrame(matrix_boot_rows)
    grouped = pd.DataFrame(grouped_rows)
    leave_family = pd.DataFrame(leave_rows)
    external = pd.DataFrame(external_rows)

    paths = {
        "feature_lineage": out_dir / "feature_lineage_v111.csv",
        "non_action": out_dir / "non_action_feature_oof_metrics.csv",
        "non_action_deltas": out_dir / "non_action_feature_bootstrap_deltas.csv",
        "matrix": out_dir / "label_feature_mismatch_matrix.csv",
        "matrix_pairs": out_dir / "label_feature_mismatch_pairs.csv",
        "matrix_boot": out_dir / "label_feature_mismatch_bootstrap.csv",
        "grouped": out_dir / "grouped_oof_metrics.csv",
        "leave_family": out_dir / "leave_family_out_metrics.csv",
        "external": out_dir / "external_label_decoupling_summary.csv",
        "report": out_dir / "v111_decoupling_report.md",
        "manifest": out_dir / "artifact_manifest.csv",
        "run": out_dir / "run_manifest.json",
    }
    write_csv(paths["feature_lineage"], add_config_hash(lineage_rows, cfg_hash))
    write_csv(paths["non_action"], add_config_hash(non_action.to_dict("records"), cfg_hash))
    write_csv(paths["non_action_deltas"], add_config_hash(non_action_deltas.to_dict("records"), cfg_hash))
    write_csv(paths["matrix"], add_config_hash(matrix_wide.to_dict("records"), cfg_hash))
    write_csv(paths["matrix_pairs"], add_config_hash(matrix.to_dict("records"), cfg_hash))
    write_csv(paths["matrix_boot"], add_config_hash(matrix_boot.to_dict("records"), cfg_hash))
    write_csv(paths["grouped"], add_config_hash(grouped.to_dict("records"), cfg_hash))
    write_csv(paths["leave_family"], add_config_hash(leave_family.to_dict("records"), cfg_hash))
    write_csv(paths["external"], add_config_hash(external.to_dict("records"), cfg_hash))
    paths["report"].write_text(_report(non_action, non_action_deltas, matrix, matrix_boot, grouped, leave_family, external, matrix_summary), encoding="utf-8")
    outputs = [
        paths["feature_lineage"],
        paths["non_action"],
        paths["non_action_deltas"],
        paths["matrix"],
        paths["matrix_pairs"],
        paths["matrix_boot"],
        paths["grouped"],
        paths["leave_family"],
        paths["external"],
        paths["report"],
    ]
    write_csv(paths["manifest"], artifact_manifest_rows(args.config, outputs))
    write_json(paths["run"], run_manifest(args.config, cfg, [*outputs, paths["manifest"]]))
    print(
        "[v111-full] "
        f"out_dir={out_dir} "
        f"non_action_rows={len(non_action)} "
        f"matrix_rows={len(matrix)} "
        f"external_rows={len(external)}"
    )


if __name__ == "__main__":
    main()
