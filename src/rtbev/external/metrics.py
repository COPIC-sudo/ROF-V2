from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from rtbev.external.taxonomy import annotate_planner_failure_taxonomy


UNKNOWN_REASON_TOKENS = {
    "",
    "unknown",
    "no_candidate_generated",
    "parser_error",
    "parse_error",
    "runtime_error",
    "software_error",
    "numerical_error",
    "sample_error",
    "missing",
    "nan",
}


def _num(values: pd.Series | Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.Series(dtype=float)


def _fmt_float(value: Any) -> float:
    try:
        f = float(value)
    except Exception:
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def classify_failure_taxonomy(labels: pd.DataFrame) -> pd.Series:
    diagnostic_cols = {
        "collision_flag",
        "road_boundary_flag",
        "lane_buffer_flag",
        "kinematic_flag",
        "candidate_any_feasible",
        "candidate_all_invalid",
        "taxonomy_rule_id",
    }
    if diagnostic_cols.intersection(labels.columns):
        return annotate_planner_failure_taxonomy(labels)["failure_taxonomy"]
    if "failure_taxonomy" in labels.columns:
        raw = labels["failure_taxonomy"].fillna("").astype(str)
        mapped = raw.where(raw.isin(["known_failure", "no_failure", "unknown_failure"]), "")
        if (mapped != "").any():
            return mapped.replace("", "unknown_failure")
    failure_col = "planner_failure" if "planner_failure" in labels.columns else "planner_failed"
    failure = _num(labels.get(failure_col, pd.Series(0, index=labels.index))).fillna(0).astype(int)
    reason_col = next((c for c in ["planner_failure_reason", "failure_reason", "dominant_failure_reason"] if c in labels.columns), None)
    reason = labels[reason_col].fillna("").astype(str).str.lower() if reason_col else pd.Series("", index=labels.index)
    out = pd.Series("no_failure", index=labels.index, dtype=object)
    failure_mask = failure == 1
    unknown = reason.isin(UNKNOWN_REASON_TOKENS) | reason.str.contains("unknown|parser|runtime|software|numerical|missing", regex=True)
    out.loc[failure_mask & unknown] = "unknown_failure"
    out.loc[failure_mask & ~unknown] = "known_failure"
    return out


def normalize_score(values: pd.Series, invert: bool = False) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    if invert:
        finite = x[np.isfinite(x)]
        safe_max = float(finite.quantile(0.99)) if len(finite) else 0.0
        x = safe_max - x
    finite_mask = np.isfinite(x.to_numpy(dtype=float))
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if not finite_mask.any():
        return out
    finite = x[finite_mask]
    lo = float(finite.quantile(0.01))
    hi = float(finite.quantile(0.99))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    if hi <= lo:
        out.loc[finite_mask] = 0.0
    else:
        out.loc[finite_mask] = np.clip((finite - lo) / (hi - lo), 0.0, 1.0)
    return out


def default_score_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "ROF_v2_composite",
        "ROF_v2_no_asr_composite",
        "temporal_composite",
        "REDI_actionability",
        "TTAD_inverse",
        "collapse_rate",
        "distance_inverse",
        "TTC_inverse",
        "commonroad_crime_risk_score",
        "rss_danger_score",
        "drivability_risk_score",
        "forecast_risk_score",
    ]
    return [c for c in preferred if c in df.columns]


def add_common_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "distance_inverse" not in out.columns and "current_min_distance_m" in out.columns:
        out["distance_inverse"] = -_num(out["current_min_distance_m"])
    if "TTC_inverse" not in out.columns and "current_ttc_s" in out.columns:
        ttc = _num(out["current_ttc_s"])
        finite = ttc[(ttc >= 0.0) & np.isfinite(ttc)]
        safe = float(finite.max() + 10.0) if len(finite) else 999.0
        out["TTC_inverse"] = -ttc.where(ttc >= 0.0, safe)
    if "TTAD_inverse" not in out.columns and "ttad_s" in out.columns:
        ttad = _num(out["ttad_s"])
        finite = ttad[np.isfinite(ttad)]
        safe = float(finite.max() + 1.0) if len(finite) else 999.0
        out["TTAD_inverse"] = -ttad.where(np.isfinite(ttad), safe)
    if "collapse_rate" not in out.columns:
        for col in ["collapse_rate_max_per_s", "collapse_rate_per_s"]:
            if col in out.columns:
                out["collapse_rate"] = _num(out[col])
                break
    if "ROF_v2_no_asr_composite" not in out.columns:
        parts = []
        for col in ["redi_actionability", "TTAD_inverse", "collapse_rate", "early_blocking_ratio"]:
            if col in out.columns:
                parts.append(normalize_score(out[col]))
        if parts:
            out["ROF_v2_no_asr_composite"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    if "ROF_v2_composite" not in out.columns:
        parts = []
        for col in ["redi_actionability", "asr_cum_final", "TTAD_inverse", "collapse_rate", "early_blocking_ratio"]:
            if col == "asr_cum_final" and col in out.columns:
                parts.append(normalize_score(1.0 - _num(out[col])))
            elif col in out.columns:
                parts.append(normalize_score(out[col]))
        if parts:
            out["ROF_v2_composite"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    if "temporal_composite" not in out.columns:
        parts = [normalize_score(out[c]) for c in ["TTAD_inverse", "collapse_rate", "early_blocking_ratio"] if c in out.columns]
        if parts:
            out["temporal_composite"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return out


def merge_scores_labels(scores: pd.DataFrame, labels: pd.DataFrame, sample_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    scores = add_common_scores(scores.copy())
    labels = labels.copy()
    scores["sample_id"] = scores["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)
    if sample_manifest is not None and not sample_manifest.empty:
        manifest = sample_manifest.copy()
        manifest["sample_id"] = manifest["sample_id"].astype(str)
        keep = [c for c in manifest.columns if c not in scores.columns or c == "sample_id"]
        scores = scores.merge(manifest[keep], on="sample_id", how="inner")
    merged = scores.merge(labels, on="sample_id", suffixes=("", "_label"), how="inner")
    if "scenario_id" not in merged.columns:
        if "commonroad_scenario_id" in merged.columns:
            merged["scenario_id"] = merged["commonroad_scenario_id"].astype(str)
        elif "commonroad_scenario_id_label" in merged.columns:
            merged["scenario_id"] = merged["commonroad_scenario_id_label"].astype(str)
        else:
            merged["scenario_id"] = merged["sample_id"].astype(str)
    if "commonroad_scenario_id" not in merged.columns:
        merged["commonroad_scenario_id"] = merged["scenario_id"].astype(str)
    merged["failure_taxonomy"] = classify_failure_taxonomy(merged)
    return merged


def binary_metric_values(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positive_count": int(y.sum()) if len(y) else 0,
        "positive_rate": float(y.mean()) if len(y) else np.nan,
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"AUPRC": np.nan, "AUROC": np.nan})
        for pct in [1, 5]:
            out[f"Recall@{pct}%FPR"] = np.nan
            out[f"threshold_at_{pct}%FPR"] = np.nan
            out[f"actual_fpr_at_{pct}%FPR"] = np.nan
        return out
    out["AUPRC"] = float(average_precision_score(y, s))
    out["AUROC"] = float(roc_auc_score(y, s))
    neg = s[y == 0]
    pos = s[y == 1]
    for pct, fpr in [(1, 0.01), (5, 0.05)]:
        threshold = float(np.quantile(neg, 1.0 - fpr))
        out[f"Recall@{pct}%FPR"] = float(np.mean(pos >= threshold)) if len(pos) else np.nan
        out[f"threshold_at_{pct}%FPR"] = threshold
        out[f"actual_fpr_at_{pct}%FPR"] = float(np.mean(neg >= threshold)) if len(neg) else np.nan
    return out


def recall_at_fpr_strict(y_true: np.ndarray, score: np.ndarray, target_fpr: float, interpolate: bool = True) -> dict[str, float]:
    """Maximize recall subject to an achieved FPR not exceeding target_fpr.

    Thresholds are evaluated at tied-score boundaries. This avoids the optimistic
    quantile behavior where a discrete score can jump past the requested FPR.
    """

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "recall": float("nan"),
            "actual_fpr": float("nan"),
            "threshold": float("nan"),
            "interpolated_recall": float("nan"),
        }
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return {
            "recall": float("nan"),
            "actual_fpr": float("nan"),
            "threshold": float("nan"),
            "interpolated_recall": float("nan"),
        }

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    unique_scores, first_idx = np.unique(s_sorted, return_index=True)
    # np.unique sorts ascending; convert to descending tied-score boundaries.
    desc_scores = unique_scores[::-1]
    starts = first_idx[::-1]
    ends = np.r_[starts[1:], len(s_sorted)]
    pos_counts = np.asarray([np.sum(y_sorted[start:end] == 1) for start, end in zip(starts, ends)], dtype=float)
    neg_counts = np.asarray([np.sum(y_sorted[start:end] == 0) for start, end in zip(starts, ends)], dtype=float)
    tp = np.cumsum(pos_counts)
    fp = np.cumsum(neg_counts)
    recalls = tp / float(n_pos)
    fprs = fp / float(n_neg)

    no_alert_threshold = float(np.nextafter(np.nanmax(s), np.inf))
    candidate_thresholds = np.r_[no_alert_threshold, desc_scores.astype(float)]
    candidate_recalls = np.r_[0.0, recalls]
    candidate_fprs = np.r_[0.0, fprs]
    valid = candidate_fprs <= float(target_fpr) + 1e-12
    if not np.any(valid):
        return {
            "recall": 0.0,
            "actual_fpr": 0.0,
            "threshold": no_alert_threshold,
            "interpolated_recall": float("nan"),
        }
    valid_idx = np.where(valid)[0]
    best_recall = np.max(candidate_recalls[valid_idx])
    best_idx = valid_idx[candidate_recalls[valid_idx] == best_recall]
    # For identical recall, use the least conservative threshold that remains strict.
    chosen = best_idx[np.argmax(candidate_fprs[best_idx])]

    interp = float("nan")
    if interpolate:
        # Optional diagnostic interpolation on the upper recall envelope.
        order_fpr = np.argsort(candidate_fprs, kind="mergesort")
        fpr_sorted = candidate_fprs[order_fpr]
        rec_sorted = candidate_recalls[order_fpr]
        uniq_fpr, first = np.unique(fpr_sorted, return_index=True)
        max_recalls = np.maximum.reduceat(rec_sorted, first)
        interp = float(np.interp(float(target_fpr), uniq_fpr, max_recalls))
    return {
        "recall": float(candidate_recalls[chosen]),
        "actual_fpr": float(candidate_fprs[chosen]),
        "threshold": float(candidate_thresholds[chosen]),
        "interpolated_recall": interp,
    }


def binary_metric_values_strict(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positive_count": int(y.sum()) if len(y) else 0,
        "positive_rate": float(y.mean()) if len(y) else np.nan,
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"AUPRC": np.nan, "AUROC": np.nan})
        for pct in [1, 5]:
            out[f"Recall@{pct}%FPR"] = np.nan
            out[f"Recall@{pct}%FPR_strict"] = np.nan
            out[f"strict_actual_fpr_at_{pct}%FPR"] = np.nan
            out[f"strict_threshold_at_{pct}%FPR"] = np.nan
            out[f"interpolated_recall_at_{pct}%FPR"] = np.nan
        return out
    out["AUPRC"] = float(average_precision_score(y, s))
    out["AUROC"] = float(roc_auc_score(y, s))
    for pct, fpr in [(1, 0.01), (5, 0.05)]:
        strict = recall_at_fpr_strict(y, s, fpr)
        out[f"Recall@{pct}%FPR"] = strict["recall"]
        out[f"Recall@{pct}%FPR_strict"] = strict["recall"]
        out[f"strict_actual_fpr_at_{pct}%FPR"] = strict["actual_fpr"]
        out[f"strict_threshold_at_{pct}%FPR"] = strict["threshold"]
        out[f"interpolated_recall_at_{pct}%FPR"] = strict["interpolated_recall"]
    return out


def _ranked_metric_values(y_true: np.ndarray, score: np.ndarray, metrics: Sequence[str]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    out = {metric: float("nan") for metric in metrics}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return out
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return out

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    unique_scores, first_idx = np.unique(s_sorted, return_index=True)
    desc_scores = unique_scores[::-1].astype(float)
    starts = first_idx[::-1]
    ends = np.r_[starts[1:], len(s_sorted)]
    pos_counts = np.asarray([np.sum(y_sorted[start:end] == 1) for start, end in zip(starts, ends)], dtype=float)
    neg_counts = np.asarray([np.sum(y_sorted[start:end] == 0) for start, end in zip(starts, ends)], dtype=float)
    tp = np.cumsum(pos_counts)
    fp = np.cumsum(neg_counts)
    recalls = tp / float(n_pos)
    fprs = fp / float(n_neg)

    if "auprc" in out:
        precision = tp / np.maximum(tp + fp, 1.0)
        recall_prev = np.r_[0.0, recalls[:-1]]
        out["auprc"] = float(np.sum((recalls - recall_prev) * precision))
    if "auroc" in out:
        out["auroc"] = float(roc_auc_score(y, s))
    if "recall_at_1pct_fpr" in out or "recall_at_5pct_fpr" in out:
        neg = s[y == 0]
        pos = s[y == 1]
        for metric, fpr in [("recall_at_1pct_fpr", 0.01), ("recall_at_5pct_fpr", 0.05)]:
            if metric not in out:
                continue
            threshold = float(np.quantile(neg, 1.0 - fpr))
            out[metric] = float(np.mean(pos >= threshold)) if len(pos) else np.nan
    if "recall_at_1pct_fpr_strict" in out or "recall_at_5pct_fpr_strict" in out:
        no_alert_recall = np.r_[0.0, recalls]
        no_alert_fpr = np.r_[0.0, fprs]
        for metric, fpr in [("recall_at_1pct_fpr_strict", 0.01), ("recall_at_5pct_fpr_strict", 0.05)]:
            if metric not in out:
                continue
            valid = no_alert_fpr <= fpr + 1e-12
            if np.any(valid):
                out[metric] = float(np.max(no_alert_recall[valid]))
            else:
                out[metric] = 0.0
    _ = desc_scores
    return out


def endpoint_frame(df: pd.DataFrame, mode: str = "known_failure") -> pd.DataFrame:
    if mode == "known_failure":
        out = df[df["failure_taxonomy"] != "unknown_failure"].copy()
        out["_y"] = (out["failure_taxonomy"] == "known_failure").astype(int)
        return out
    if mode == "all_failures_positive":
        out = df.copy()
        out["_y"] = out["failure_taxonomy"].isin(["known_failure", "unknown_failure"]).astype(int)
        return out
    if mode == "unknown_as_negative":
        out = df.copy()
        out["_y"] = (out["failure_taxonomy"] == "known_failure").astype(int)
        return out
    raise ValueError(f"unknown endpoint mode: {mode}")


def evaluate_external_scores(
    merged: pd.DataFrame,
    score_columns: Sequence[str] | None = None,
    endpoint: str = "known_failure",
) -> list[dict[str, Any]]:
    frame = endpoint_frame(merged, endpoint)
    cols = list(score_columns or default_score_columns(frame))
    rows: list[dict[str, Any]] = []
    y = frame["_y"].to_numpy(int)
    for score in cols:
        if score not in frame.columns:
            continue
        metrics = binary_metric_values(y, pd.to_numeric(frame[score], errors="coerce").to_numpy(float))
        rows.append({"endpoint": endpoint, "score": score, "primary_endpoint": endpoint == "known_failure", **metrics})
    return rows


def evaluate_external_scores_strict(
    merged: pd.DataFrame,
    score_columns: Sequence[str] | None = None,
    endpoint: str = "known_failure",
) -> list[dict[str, Any]]:
    frame = endpoint_frame(merged, endpoint)
    cols = list(score_columns or default_score_columns(frame))
    rows: list[dict[str, Any]] = []
    y = frame["_y"].to_numpy(int)
    for score in cols:
        if score not in frame.columns:
            continue
        metrics = binary_metric_values_strict(y, pd.to_numeric(frame[score], errors="coerce").to_numpy(float))
        rows.append(
            {
                "endpoint": endpoint,
                "score": score,
                "primary_endpoint": endpoint == "known_failure",
                "fpr_metric_definition": "strict_tied_threshold_actual_fpr_lte_target",
                **metrics,
            }
        )
    return rows


def failure_taxonomy_rows(labels_or_merged: pd.DataFrame) -> list[dict[str, Any]]:
    tax = classify_failure_taxonomy(labels_or_merged)
    counts = Counter(tax.astype(str))
    total = max(len(tax), 1)
    return [{"failure_taxonomy": k, "count": int(v), "fraction": float(v / total)} for k, v in sorted(counts.items())]


def stratum_metrics(merged: pd.DataFrame, score_columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cols = list(score_columns or default_score_columns(merged))
    strata = [c for c in ["neutral_stratum", "distance_stratum", "ttc_stratum", "speed_stratum", "source_hint"] if c in merged.columns]
    if not strata:
        return rows
    for stratum_col in strata:
        for value, group in merged.groupby(stratum_col, dropna=False):
            frame = endpoint_frame(group, "known_failure")
            y = frame["_y"].to_numpy(int)
            for score in cols:
                if score not in frame.columns:
                    continue
                m = binary_metric_values(y, pd.to_numeric(frame[score], errors="coerce").to_numpy(float))
                rows.append({"stratum_column": stratum_col, "stratum": str(value), "score": score, **m})
    return rows


def unknown_failure_sensitivity(merged: pd.DataFrame, score_columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint in ["known_failure", "all_failures_positive", "unknown_as_negative"]:
        for row in evaluate_external_scores(merged, score_columns, endpoint=endpoint):
            row["sensitivity_mode"] = endpoint
            rows.append(row)
    return rows


def _metric_value(y: np.ndarray, score: np.ndarray, metric: str) -> float:
    vals = binary_metric_values(y, score)
    key = {
        "auprc": "AUPRC",
        "auroc": "AUROC",
        "recall_at_1pct_fpr": "Recall@1%FPR",
        "recall_at_5pct_fpr": "Recall@5%FPR",
    }[metric]
    return _fmt_float(vals.get(key))


def _metric_value_strict(y: np.ndarray, score: np.ndarray, metric: str) -> float:
    return _fmt_float(_ranked_metric_values(y, score, [metric]).get(metric))


def parse_comparisons(value: str | Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    if isinstance(value, str):
        pairs = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"invalid comparison '{item}', expected enhanced:baseline")
            a, b = [x.strip() for x in item.split(":", 1)]
            pairs.append((a, b))
        return pairs
    return list(value)


def scenario_bootstrap_deltas(
    merged: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    metrics: Sequence[str] = ("auprc", "recall_at_5pct_fpr", "auroc", "recall_at_1pct_fpr"),
    n_bootstrap: int = 1000,
    seed: int = 42,
    endpoint: str = "known_failure",
) -> list[dict[str, Any]]:
    frame = endpoint_frame(merged, endpoint)
    if frame.empty:
        return []
    groups = frame["scenario_id"].fillna(frame["sample_id"]).astype(str).to_numpy()
    uniq = np.unique(groups)
    group_indices = [np.where(groups == g)[0] for g in uniq]
    y = frame["_y"].to_numpy(int)
    score_arrays = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy(float) for pair in comparisons for c in pair if c in frame.columns}
    rng = np.random.default_rng(int(seed))
    boot: dict[tuple[str, str, str], list[float]] = {(a, b, m): [] for a, b in comparisons for m in metrics}
    for _ in range(int(n_bootstrap)):
        if not group_indices:
            continue
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        for enhanced, baseline in comparisons:
            if enhanced not in score_arrays or baseline not in score_arrays:
                continue
            for metric in metrics:
                ev = _metric_value(yy, score_arrays[enhanced][idx], metric)
                bv = _metric_value(yy, score_arrays[baseline][idx], metric)
                if np.isfinite(ev) and np.isfinite(bv):
                    boot[(enhanced, baseline, metric)].append(float(ev - bv))
    rows: list[dict[str, Any]] = []
    for enhanced, baseline in comparisons:
        if enhanced not in score_arrays or baseline not in score_arrays:
            continue
        for metric in metrics:
            ev = _metric_value(y, score_arrays[enhanced], metric)
            bv = _metric_value(y, score_arrays[baseline], metric)
            delta = ev - bv if np.isfinite(ev) and np.isfinite(bv) else np.nan
            arr = np.asarray(boot[(enhanced, baseline, metric)], dtype=float)
            rows.append(
                {
                    "endpoint": endpoint,
                    "bootstrap_unit": "scenario_id",
                    "enhanced_score": enhanced,
                    "baseline_score": baseline,
                    "metric": metric,
                    "enhanced_point": ev,
                    "baseline_point": bv,
                    "delta": delta,
                    "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                    "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                    "P_delta_gt_0": float(np.mean(arr > 0.0)) if len(arr) else np.nan,
                    "n_bootstrap_valid": int(len(arr)),
                    "n_samples": int(len(frame)),
                    "n_scenarios": int(len(uniq)),
                    "positive_count": int(y.sum()),
                    "positive_rate": float(y.mean()) if len(y) else np.nan,
                }
            )
    return rows


def scenario_bootstrap_deltas_strict(
    merged: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    metrics: Sequence[str] = ("auprc", "recall_at_5pct_fpr_strict"),
    n_bootstrap: int = 1000,
    seed: int = 42,
    endpoint: str = "known_failure",
) -> list[dict[str, Any]]:
    frame = endpoint_frame(merged, endpoint)
    if frame.empty:
        return []
    groups = frame["scenario_id"].fillna(frame["sample_id"]).astype(str).to_numpy()
    uniq = np.unique(groups)
    group_indices = [np.where(groups == g)[0] for g in uniq]
    y = frame["_y"].to_numpy(int)
    score_arrays = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy(float) for pair in comparisons for c in pair if c in frame.columns}
    finite_masks = {c: np.isfinite(arr) for c, arr in score_arrays.items()}
    mask_ids: dict[bytes, int] = {}
    pair_masks: dict[tuple[str, str], tuple[int, np.ndarray]] = {}
    for enhanced, baseline in comparisons:
        if enhanced not in finite_masks or baseline not in finite_masks:
            continue
        mask = finite_masks[enhanced] & finite_masks[baseline]
        key = mask.tobytes()
        if key not in mask_ids:
            mask_ids[key] = len(mask_ids)
        pair_masks[(enhanced, baseline)] = (mask_ids[key], mask)
    rng = np.random.default_rng(int(seed))
    boot: dict[tuple[str, str, str], list[float]] = {(a, b, m): [] for a, b in comparisons for m in metrics}
    for _ in range(int(n_bootstrap)):
        if not group_indices:
            continue
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        cache: dict[tuple[str, int], dict[str, float]] = {}
        for enhanced, baseline in comparisons:
            if enhanced not in score_arrays or baseline not in score_arrays:
                continue
            mask_id, pair_mask = pair_masks[(enhanced, baseline)]
            pair_idx = idx[pair_mask[idx]]
            yy = y[pair_idx]
            if len(yy) == 0 or len(np.unique(yy)) < 2:
                continue
            enhanced_key = (enhanced, mask_id)
            baseline_key = (baseline, mask_id)
            if enhanced_key not in cache:
                cache[enhanced_key] = _ranked_metric_values(yy, score_arrays[enhanced][pair_idx], metrics)
            if baseline_key not in cache:
                cache[baseline_key] = _ranked_metric_values(yy, score_arrays[baseline][pair_idx], metrics)
            for metric in metrics:
                ev = _fmt_float(cache[enhanced_key].get(metric))
                bv = _fmt_float(cache[baseline_key].get(metric))
                if np.isfinite(ev) and np.isfinite(bv):
                    boot[(enhanced, baseline, metric)].append(float(ev - bv))
    rows: list[dict[str, Any]] = []
    for enhanced, baseline in comparisons:
        if enhanced not in score_arrays or baseline not in score_arrays:
            continue
        _, pair_ok = pair_masks[(enhanced, baseline)]
        yy = y[pair_ok]
        ev_all = _ranked_metric_values(yy, score_arrays[enhanced][pair_ok], metrics)
        bv_all = _ranked_metric_values(yy, score_arrays[baseline][pair_ok], metrics)
        pairwise_scenarios = int(pd.Series(groups[pair_ok]).nunique()) if len(yy) else 0
        for metric in metrics:
            ev = _fmt_float(ev_all.get(metric)) if len(np.unique(yy)) >= 2 else np.nan
            bv = _fmt_float(bv_all.get(metric)) if len(np.unique(yy)) >= 2 else np.nan
            delta = ev - bv if np.isfinite(ev) and np.isfinite(bv) else np.nan
            arr = np.asarray(boot[(enhanced, baseline, metric)], dtype=float)
            rows.append(
                {
                    "endpoint": endpoint,
                    "bootstrap_unit": "scenario_id",
                    "fpr_metric_definition": "strict_tied_threshold_actual_fpr_lte_target",
                    "enhanced_score": enhanced,
                    "baseline_score": baseline,
                    "metric": metric,
                    "enhanced_point": ev,
                    "baseline_point": bv,
                    "delta": delta,
                    "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                    "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                    "P_delta_gt_0": float(np.mean(arr > 0.0)) if len(arr) else np.nan,
                    "n_bootstrap_valid": int(len(arr)),
                    "n_samples": int(len(frame)),
                    "n_scenarios": int(len(uniq)),
                    "positive_count": int(y.sum()),
                    "positive_rate": float(y.mean()) if len(y) else np.nan,
                    "pairwise_n": int(len(yy)),
                    "pairwise_positive_count": int(yy.sum()) if len(yy) else 0,
                    "pairwise_scenario_count": pairwise_scenarios,
                }
            )
    return rows
