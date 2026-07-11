#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from _utils import bool_str, load_yaml, output_dir, resolve_path, write_csv


METRICS = [
    "auprc",
    "auroc",
    "recall_at_5pct_fpr",
    "precision_at_5pct_fpr",
    "achieved_fpr_at_5pct_fpr",
]


def seed_for(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def metric_bundle(y: np.ndarray, score: np.ndarray, alert: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    alert = np.asarray(alert, dtype=bool)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    alert = alert[ok]
    pos = y == 1
    neg = y == 0
    out = {
        "auprc": np.nan,
        "auroc": np.nan,
        "recall_at_5pct_fpr": float(np.mean(alert[pos])) if np.any(pos) else np.nan,
        "precision_at_5pct_fpr": float(np.sum(alert & pos) / max(np.sum(alert), 1)) if np.any(alert) else np.nan,
        "achieved_fpr_at_5pct_fpr": float(np.mean(alert[neg])) if np.any(neg) else np.nan,
    }
    if len(y) and len(np.unique(y)) == 2:
        out["auprc"] = float(average_precision_score(y, score))
        out["auroc"] = float(roc_auc_score(y, score))
    return out


def group_boot_indices(groups: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
    groups = np.asarray(groups).astype(str)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if np.all(counts == 1):
        singleton_lookup = np.empty(len(uniq), dtype=int)
        for i, gidx in enumerate(inv):
            singleton_lookup[gidx] = i
        return [], singleton_lookup
    return [np.where(inv == i)[0] for i in range(len(uniq))], None


def bootstrap_comparison(job: dict[str, Any]) -> list[dict[str, Any]]:
    y = job.pop("y")
    b_score = job.pop("baseline_score")
    e_score = job.pop("enhanced_score")
    b_alert = job.pop("baseline_alert")
    e_alert = job.pop("enhanced_alert")
    groups = job.pop("scenario_id")
    n_bootstrap = int(job.pop("n_bootstrap"))
    seed = int(job.pop("seed_value"))

    group_indices, singleton_lookup = group_boot_indices(groups)
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {m: [] for m in METRICS}
    n_groups = len(singleton_lookup) if singleton_lookup is not None else len(group_indices)
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, size=n_groups)
        if singleton_lookup is not None:
            idx = singleton_lookup[sampled]
        else:
            idx = np.concatenate([group_indices[i] for i in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        b = metric_bundle(yy, b_score[idx], b_alert[idx])
        e = metric_bundle(yy, e_score[idx], e_alert[idx])
        for metric in METRICS:
            bv = b[metric]
            ev = e[metric]
            if np.isfinite(bv) and np.isfinite(ev):
                deltas[metric].append(float(ev - bv))

    b_point = metric_bundle(y, b_score, b_alert)
    e_point = metric_bundle(y, e_score, e_alert)
    rows = []
    for metric in METRICS:
        arr = np.asarray(deltas[metric], dtype=float)
        point_delta = float(e_point[metric] - b_point[metric]) if np.isfinite(e_point[metric]) and np.isfinite(b_point[metric]) else np.nan
        p_gt0 = float(np.mean(arr > 0)) if len(arr) else np.nan
        p_lt0 = float(np.mean(arr < 0)) if len(arr) else np.nan
        p_two = float(min(1.0, 2.0 * min(p_gt0, p_lt0))) if len(arr) else np.nan
        row = dict(job)
        row.update(
            {
                "metric": metric,
                "baseline_point": b_point[metric],
                "enhanced_point": e_point[metric],
                "delta": point_delta,
                "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                "bootstrap_prob_delta_gt_0": p_gt0,
                "p_value_two_sided": p_two,
                "n_bootstrap_requested": n_bootstrap,
                "n_bootstrap_valid": int(len(arr)),
                "bootstrap_unit": "scenario_id",
                "ci_type": "percentile",
                "threshold_operator": ">=",
                "threshold_source": "v090 fold-calibrated thresholds; alerts are resampled, not recalibrated inside bootstrap",
            }
        )
        rows.append(row)
    return rows


def feature_metrics(sub: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (endpoint, model, seed, feature_set), g in sub.groupby(["endpoint", "model", "seed", "feature_set"], dropna=False):
        mb = metric_bundle(g["y_true"].to_numpy(int), g["score"].to_numpy(float), g["alert"].to_numpy(bool))
        for metric, value in mb.items():
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": model,
                    "seed": int(seed),
                    "feature_set": feature_set,
                    "metric": metric,
                    "value": value,
                    "n_samples": int(len(g)),
                    "n_scenarios": int(g["scenario_id"].astype(str).nunique()),
                    "positive_count": int(pd.to_numeric(g["y_true"], errors="coerce").fillna(0).sum()),
                    "positive_rate": float(pd.to_numeric(g["y_true"], errors="coerce").fillna(0).mean()),
                    "threshold_median": float(pd.to_numeric(g["threshold"], errors="coerce").median()) if "threshold" in g.columns else np.nan,
                    "threshold_min": float(pd.to_numeric(g["threshold"], errors="coerce").min()) if "threshold" in g.columns else np.nan,
                    "threshold_max": float(pd.to_numeric(g["threshold"], errors="coerce").max()) if "threshold" in g.columns else np.nan,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v095/nc_v095_p0_extension.yaml")
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--n-bootstrap-manuscript-critical", type=int, default=None)
    parser.add_argument("--n-bootstrap-exploratory", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    pred_path = resolve_path(args.predictions_csv or cfg["inputs"]["waymo_oof_predictions_csv"])
    endpoints = set(cfg["secondary_context"]["endpoints"])
    models = set(cfg["secondary_context"].get("models", ["rf"]))
    baseline = cfg["secondary_context"]["baseline_feature_set"]
    enhanced_sets = list(cfg["secondary_context"]["enhanced_feature_sets"])
    all_feature_sets = set([baseline] + enhanced_sets)

    usecols = ["sample_id", "scenario_id", "endpoint", "model", "seed", "feature_set", "y_true", "score", "threshold", "alert_at_calibrated_5pct_fpr"]
    pred = pd.read_csv(pred_path, usecols=usecols)
    pred = pred[pred["endpoint"].isin(endpoints) & pred["model"].isin(models) & pred["feature_set"].isin(all_feature_sets)].copy()
    pred["sample_id"] = pred["sample_id"].astype(str)
    pred["scenario_id"] = pred["scenario_id"].astype(str)
    pred["alert"] = pred["alert_at_calibrated_5pct_fpr"].map(bool_str)
    metrics_rows = feature_metrics(pred)
    write_csv(out_dir / "waymo_secondary_context_metrics_v095.csv", metrics_rows)
    if "envswitch" in out_dir.name.lower():
        write_csv(out_dir / "waymo_secondary_context_metrics_full.csv", metrics_rows)

    manuscript_endpoints = set(cfg["secondary_context"].get("manuscript_critical_endpoints", []))
    ncrit = int(args.n_bootstrap_manuscript_critical or cfg["evaluation"]["bootstrap_replicates_manuscript_critical"])
    nexp = int(args.n_bootstrap_exploratory or cfg["evaluation"]["bootstrap_replicates_exploratory"])
    jobs = []
    for (endpoint, model, seed), sub in pred.groupby(["endpoint", "model", "seed"], dropna=False):
        base = sub[sub["feature_set"] == baseline]
        if base.empty:
            continue
        base_cols = base[["sample_id", "scenario_id", "y_true", "score", "alert"]].rename(columns={"score": "baseline_score", "alert": "baseline_alert"})
        for enhanced in enhanced_sets:
            enh = sub[sub["feature_set"] == enhanced]
            if enh.empty:
                continue
            merged = base_cols.merge(
                enh[["sample_id", "score", "alert"]].rename(columns={"score": "enhanced_score", "alert": "enhanced_alert"}),
                on="sample_id",
                how="inner",
            )
            n_boot = ncrit if endpoint in manuscript_endpoints else nexp
            y = merged["y_true"].to_numpy(int)
            jobs.append(
                {
                    "endpoint": endpoint,
                    "model": model,
                    "seed": int(seed),
                    "baseline_feature_set": baseline,
                    "enhanced_feature_set": enhanced,
                    "comparison": f"{baseline}__vs__{enhanced}",
                    "n_samples": int(len(merged)),
                    "n_scenarios": int(merged["scenario_id"].nunique()),
                    "positive_count": int(y.sum()),
                    "positive_rate": float(np.mean(y)),
                    "manuscript_critical": bool(endpoint in manuscript_endpoints),
                    "n_bootstrap": n_boot,
                    "seed_value": seed_for(str(endpoint), str(model), str(seed), str(enhanced), "v095"),
                    "y": y,
                    "baseline_score": merged["baseline_score"].to_numpy(float),
                    "enhanced_score": merged["enhanced_score"].to_numpy(float),
                    "baseline_alert": merged["baseline_alert"].to_numpy(bool),
                    "enhanced_alert": merged["enhanced_alert"].to_numpy(bool),
                    "scenario_id": merged["scenario_id"].to_numpy(str),
                }
            )
    n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg["evaluation"].get("n_jobs", -1))
    nested = Parallel(n_jobs=n_jobs, verbose=5)(delayed(bootstrap_comparison)(job) for job in jobs)
    rows = [row for chunk in nested for row in chunk]
    rows = sorted(rows, key=lambda r: (r["endpoint"], r["model"], r["seed"], r["enhanced_feature_set"], r["metric"]))
    write_csv(out_dir / "waymo_secondary_context_bootstrap_v095.csv", rows)
    if "envswitch" in out_dir.name.lower():
        write_csv(out_dir / "waymo_secondary_context_bootstrap_full.csv", rows)

    gate_rows = []
    for row in rows:
        if row["metric"] != "auprc":
            continue
        status = "PASS" if np.isfinite(row["ci_low"]) and float(row["ci_low"]) > 0 else "FAIL_OR_INCONCLUSIVE"
        gate_rows.append(
            {
                "endpoint": row["endpoint"],
                "model": row["model"],
                "seed": row["seed"],
                "comparison": row["comparison"],
                "metric": row["metric"],
                "delta": row["delta"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "status": status,
            }
        )
    write_csv(out_dir / "waymo_secondary_context_claim_gate.csv", gate_rows)
    if "envswitch" in out_dir.name.lower():
        fig_rows = []
        for row in rows:
            if row["metric"] in {"auprc", "auroc", "recall_at_5pct_fpr"}:
                fig_rows.append(
                    {
                        **{k: row.get(k) for k in ["endpoint", "model", "seed", "comparison", "metric", "baseline_point", "enhanced_point", "delta", "ci_low", "ci_high", "bootstrap_prob_delta_gt_0", "n_bootstrap_valid"]},
                        "figure4_ready": bool(row["model"] == "rf" and row["metric"] == "auprc" and np.isfinite(row["ci_low"]) and float(row["ci_low"]) > 0),
                        "recommended_scope": "main_candidate" if row["endpoint"] == "map_critical_or_worse" else "supplement_or_exploratory",
                    }
                )
        write_csv(out_dir / "figure4_ready_table.csv", fig_rows)
    lines = [
        "# Waymo Secondary/Context Bootstrap Claim Gate",
        "",
        f"Jobs: {len(jobs)} comparisons; elapsed_s={time.perf_counter() - t0:.1f}",
        "",
        "Thresholds are the frozen v0.9 fold-calibrated 5% FPR thresholds. Bootstrap resamples scenario IDs and reuses alert decisions; thresholds are not recomputed inside bootstrap replicates.",
        "",
    ]
    for r in gate_rows:
        if r["status"] == "PASS":
            lines.append(f"- PASS `{r['endpoint']}` seed {r['seed']} `{r['comparison']}` AUPRC delta {float(r['delta']):.6f}, CI [{float(r['ci_low']):.6f}, {float(r['ci_high']):.6f}]")
    (out_dir / "waymo_secondary_context_claim_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[v095-secondary] wrote {len(rows)} bootstrap rows elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
