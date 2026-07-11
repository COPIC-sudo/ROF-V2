from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.progress import ProgressReporter


REQUIRED_COLUMNS = ["sample_id", "task", "model", "y_true", "score"]


def _parse_csv_arg(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _parse_comparisons(value: str) -> list[tuple[str, str]]:
    pairs = []
    for item in _parse_csv_arg(value):
        if ":" not in item:
            raise SystemExit(f"invalid comparison '{item}'; expected baseline:enhanced")
        base, enhanced = [x.strip() for x in item.split(":", 1)]
        if not base or not enhanced:
            raise SystemExit(f"invalid comparison '{item}'; expected baseline:enhanced")
        pairs.append((base, enhanced))
    return pairs


def _metric_value(y_true: np.ndarray, score: np.ndarray, metric: str) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    if metric == "auroc":
        return float(roc_auc_score(y, s))
    if metric == "auprc":
        return float(average_precision_score(y, s))
    if metric == "recall_at_1pct_fpr":
        return _recall_at_fpr(y, s, 0.01)
    if metric == "recall_at_5pct_fpr":
        return _recall_at_fpr(y, s, 0.05)
    raise ValueError(f"unknown metric={metric}")


def _recall_at_fpr(y_true: np.ndarray, score: np.ndarray, fpr_level: float) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    neg = s[y == 0]
    pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    threshold = float(np.quantile(neg, 1.0 - float(fpr_level)))
    return float(np.mean(pos >= threshold))


def _validate_predictions(df: pd.DataFrame, feature_column: str) -> None:
    required = REQUIRED_COLUMNS + [str(feature_column)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            "predictions CSV missing required columns: "
            f"{missing}; actual columns={list(df.columns)}"
        )


def _prepare_comparison_frame(
    df: pd.DataFrame,
    task: str,
    model: str,
    baseline_feature_set: str,
    enhanced_feature_set: str,
) -> pd.DataFrame:
    base = df[(df["task"] == task) & (df["model"] == model) & (df["feature_set"] == baseline_feature_set)].copy()
    enh = df[(df["task"] == task) & (df["model"] == model) & (df["feature_set"] == enhanced_feature_set)].copy()
    if base.empty:
        raise ValueError(f"no predictions for task={task}, model={model}, feature_set={baseline_feature_set}")
    if enh.empty:
        raise ValueError(f"no predictions for task={task}, model={model}, feature_set={enhanced_feature_set}")
    base_cols = ["sample_id", "scenario_id", "y_true", "score"]
    enh_cols = ["sample_id", "y_true", "score"]
    merged = base[base_cols].merge(
        enh[enh_cols],
        on="sample_id",
        how="inner",
        suffixes=("_baseline", "_enhanced"),
    )
    if merged.empty:
        raise ValueError(
            "empty merge for "
            f"task={task}, model={model}, comparison={baseline_feature_set}:{enhanced_feature_set}"
        )
    mismatch = merged["y_true_baseline"].astype(int) != merged["y_true_enhanced"].astype(int)
    if bool(mismatch.any()):
        raise ValueError(
            "y_true mismatch after merging predictions for "
            f"task={task}, model={model}, comparison={baseline_feature_set}:{enhanced_feature_set}"
        )
    merged = merged.rename(columns={"y_true_baseline": "y_true", "score_baseline": "score_baseline", "score_enhanced": "score_enhanced"})
    merged["scenario_id"] = merged["scenario_id"].fillna(merged["sample_id"]).astype(str)
    return merged[["sample_id", "scenario_id", "y_true", "score_baseline", "score_enhanced"]].copy()


def _bootstrap_one(
    task: str,
    model: str,
    baseline_feature_set: str,
    enhanced_feature_set: str,
    metric: str,
    y_true: np.ndarray,
    score_baseline: np.ndarray,
    score_enhanced: np.ndarray,
    scenario_id: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    point_base = _metric_value(y_true, score_baseline, metric)
    point_enh = _metric_value(y_true, score_enhanced, metric)
    groups = np.asarray(scenario_id).astype(str)
    uniq = np.unique(groups)
    group_indices = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(int(seed))
    deltas = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        base_v = _metric_value(yb, score_baseline[idx], metric)
        enh_v = _metric_value(yb, score_enhanced[idx], metric)
        if np.isfinite(base_v) and np.isfinite(enh_v):
            deltas.append(float(enh_v - base_v))
    delta_arr = np.asarray(deltas, dtype=float)
    if len(delta_arr):
        ci_low = float(np.percentile(delta_arr, 2.5))
        ci_high = float(np.percentile(delta_arr, 97.5))
        prob_gt0 = float(np.mean(delta_arr > 0.0))
        p_two = float(min(1.0, 2.0 * min(np.mean(delta_arr <= 0.0), np.mean(delta_arr >= 0.0))))
    else:
        ci_low = np.nan
        ci_high = np.nan
        prob_gt0 = np.nan
        p_two = np.nan
    return {
        "task": task,
        "model": model,
        "comparison": f"{baseline_feature_set}:{enhanced_feature_set}",
        "baseline_feature_set": baseline_feature_set,
        "enhanced_feature_set": enhanced_feature_set,
        "metric": metric,
        "baseline_point": point_base,
        "enhanced_point": point_enh,
        "delta_point": float(point_enh - point_base) if np.isfinite(point_base) and np.isfinite(point_enh) else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_prob_delta_gt_0": prob_gt0,
        "p_value_two_sided": p_two,
        "n_bootstrap_valid": int(len(delta_arr)),
        "n_scenarios": int(len(uniq)),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
    }


def _seed_for_combo(seed: int, task: str, model: str, comparison: tuple[str, str], metric: str) -> int:
    text = f"{seed}|{task}|{model}|{comparison[0]}|{comparison[1]}|{metric}"
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _run_parallel(jobs, n_jobs: int, progress: ProgressReporter) -> list[dict]:
    rows = []
    total = len(jobs)
    progress.update("bootstrapping", step=0, total=total, message=f"starting {total} bootstrap jobs")
    try:
        gen = Parallel(n_jobs=int(n_jobs), return_as="generator_unordered")(jobs)
        for i, row in enumerate(gen, start=1):
            rows.append(row)
            progress.update(
                "bootstrapping",
                step=i,
                total=total,
                message=f"finished {row['task']} {row['model']} {row['comparison']} {row['metric']}",
            )
    except TypeError:
        rows = Parallel(n_jobs=int(n_jobs))(jobs)
        for i, row in enumerate(rows, start=1):
            progress.update(
                "bootstrapping",
                step=i,
                total=total,
                message=f"finished {row['task']} {row['model']} {row['comparison']} {row['metric']}",
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Scenario-level bootstrap deltas from existing actionability predictions.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default="actionability_critical,actionability_infeasible")
    ap.add_argument("--models", default="rf")
    ap.add_argument("--comparisons", default="strong_baseline:strong_baseline_actionability,strong_baseline_cv:strong_baseline_cv_actionability")
    ap.add_argument("--metrics", default="auroc,auprc,recall_at_1pct_fpr,recall_at_5pct_fpr")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-task", default="actionability_bootstrap")
    ap.add_argument("--feature-column", default="feature_set", help="column containing baseline/enhanced names")
    ap.add_argument("--out-name", default="actionability_bootstrap_deltas.csv")
    args = ap.parse_args()

    started = time.perf_counter()
    progress: ProgressReporter | None = None
    try:
        cfg = load_config(args.config)
        work = Path(cfg["project"]["work_dir"])
        progress = ProgressReporter(work, args.progress_task)
        out_dir = ensure_dir(Path(args.out_dir))
        tasks = _parse_csv_arg(args.tasks)
        models = _parse_csv_arg(args.models)
        comparisons = _parse_comparisons(args.comparisons)
        metrics = _parse_csv_arg(args.metrics)
        valid_metrics = {"auroc", "auprc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"}
        unknown_metrics = [m for m in metrics if m not in valid_metrics]
        if unknown_metrics:
            raise SystemExit(f"unknown metrics: {unknown_metrics}; valid={sorted(valid_metrics)}")

        progress.update("loading", message=str(args.predictions_csv))
        pred = pd.read_csv(args.predictions_csv)
        _validate_predictions(pred, args.feature_column)
        pred = pred.copy()
        pred["sample_id"] = pred["sample_id"].astype(str)
        if "scenario_id" not in pred.columns:
            pred["scenario_id"] = pred["sample_id"]
        pred["scenario_id"] = pred["scenario_id"].fillna(pred["sample_id"]).astype(str)
        pred["task"] = pred["task"].astype(str)
        pred["model"] = pred["model"].astype(str)
        pred["feature_set"] = pred[str(args.feature_column)].astype(str)
        pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce").astype(int)
        pred["score"] = pd.to_numeric(pred["score"], errors="coerce")

        progress.update("preparing", message="aligning baseline/enhanced predictions")
        data_by_combo: dict[tuple[str, str, str, str], pd.DataFrame] = {}
        for task in tasks:
            for model in models:
                for base, enhanced in comparisons:
                    data_by_combo[(task, model, base, enhanced)] = _prepare_comparison_frame(pred, task, model, base, enhanced)

        jobs = []
        for task in tasks:
            for model in models:
                for comparison in comparisons:
                    base, enhanced = comparison
                    df = data_by_combo[(task, model, base, enhanced)]
                    y = df["y_true"].to_numpy(int)
                    sb = df["score_baseline"].to_numpy(float)
                    se = df["score_enhanced"].to_numpy(float)
                    groups = df["scenario_id"].astype(str).to_numpy()
                    for metric in metrics:
                        combo_seed = _seed_for_combo(int(args.seed), task, model, comparison, metric)
                        jobs.append(
                            delayed(_bootstrap_one)(
                                task,
                                model,
                                base,
                                enhanced,
                                metric,
                                y,
                                sb,
                                se,
                                groups,
                                int(args.n_bootstrap),
                                combo_seed,
                            )
                        )
        rows = _run_parallel(jobs, int(args.n_jobs), progress)

        progress.update("writing_outputs", message=str(out_dir))
        out_path = out_dir / str(args.out_name)
        pd.DataFrame(rows).sort_values(["task", "model", "comparison", "metric"]).to_csv(out_path, index=False)
        elapsed = time.perf_counter() - started
        progress.complete(f"wrote {out_path}; elapsed={elapsed:.1f}s")
        print(f"[actionability-bootstrap] wrote {out_path}")
        print(f"[actionability-bootstrap] elapsed={elapsed:.1f}s, rows={len(rows)}, n_bootstrap={int(args.n_bootstrap)}")
    except Exception as exc:
        if progress is not None:
            progress.fail(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
