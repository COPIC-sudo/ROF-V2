#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


OUTPUT_DIR = Path("results/nc_v100_pr_curve_derivation")
PREDICTION_FILE = Path("results/nc_v090_scientific_audit/waymo_oof_predictions.csv")
LOCKED_METRICS_FILE = Path("results/nc_v090_scientific_audit/waymo_confirmatory_metrics.csv")
ENDPOINT = "map_critical_or_worse"
MODEL = "rf"
SEEDS = [41, 42, 43]
FEATURE_SETS = [
    "strong_baseline_cv",
    "strong_baseline_cv_plus_strict_temporal_dynamics",
]
DISPLAY_LABELS = {
    "strong_baseline_cv": "Strong baseline + CV",
    "strong_baseline_cv_plus_strict_temporal_dynamics": "Strong baseline + CV + strict temporal dynamics",
}
COLOR_KEYS = {
    "strong_baseline_cv": "baseline_cv",
    "strong_baseline_cv_plus_strict_temporal_dynamics": "strict_temporal",
}
SCORE_CANDIDATES = ["y_score", "score", "pred_score", "probability", "positive_score", "pred_prob", "prediction"]
LABEL_CANDIDATES = ["y_true", "label", "target", "binary_label", "positive"]
AUPRC_TOL = 1e-9


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def write_blockers(path: Path, blockers: list[dict[str, str]]) -> None:
    if not blockers:
        blockers = [{"status": "PASS", "blocker": "None", "detail": "No blocking condition detected."}]
    pd.DataFrame(blockers).to_csv(path, index=False)


def find_locked_auprc(metrics: pd.DataFrame, seed: int, feature_set: str) -> float:
    sub = metrics[
        metrics["level"].astype(str).eq("pooled_oof")
        & metrics["endpoint"].astype(str).eq(ENDPOINT)
        & metrics["model"].astype(str).eq(MODEL)
        & metrics["seed"].astype(int).eq(int(seed))
        & metrics["feature_set"].astype(str).eq(feature_set)
    ]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub.iloc[0]["auprc"], errors="coerce"))


def threshold_array(thresholds: np.ndarray, n_points: int) -> np.ndarray:
    out = np.full(n_points, np.nan, dtype=float)
    if len(thresholds):
        out[: len(thresholds)] = thresholds
    return out


def thin_curve(df: pd.DataFrame, max_points: int = 1500) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    precision = df["precision"].to_numpy(float)
    recall = df["recall"].to_numpy(float)
    keep = np.zeros(len(df), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stride = max(1, int(math.ceil(len(df) / max_points)))
    keep[::stride] = True
    keep |= np.r_[False, np.abs(np.diff(precision)) >= 0.01]
    keep |= np.r_[False, np.abs(np.diff(recall)) >= 0.01]
    return df.loc[keep].copy()


def mean_curve(seed_curves: list[pd.DataFrame], recall_step: float) -> pd.DataFrame:
    grid = np.round(np.arange(0.0, 1.0 + 1e-12, recall_step), 6)
    values = []
    for curve in seed_curves:
        c = curve[["recall", "precision"]].dropna().copy()
        c = c.sort_values("recall")
        c = c.groupby("recall", as_index=False)["precision"].max()
        recall = c["recall"].to_numpy(float)
        precision = c["precision"].to_numpy(float)
        if len(recall) == 0:
            values.append(np.full_like(grid, np.nan, dtype=float))
        else:
            values.append(np.interp(grid, recall, precision, left=precision[0], right=precision[-1]))
    arr = np.vstack(values)
    return pd.DataFrame({"recall": grid, "precision": np.nanmean(arr, axis=0)})


def copy_optional_outputs(out_dir: Path, copied: list[str]) -> None:
    targets = [
        Path("03_main_figure_source_data_v100_redesigned/Figure3_primary_waymo_oof"),
        Path("source_data/v100/03_main_figure_source_data_v100_redesigned/Figure3_primary_waymo_oof"),
    ]
    names = [
        "Figure3a_oof_pr_curve_plot_ready.csv",
        "Figure3a_oof_pr_curve_summary.csv",
        "Figure3a_oof_pr_curve_QA.md",
    ]
    for target in targets:
        if not target.exists():
            continue
        for name in names:
            shutil.copy2(out_dir / name, target / name)
            copied.append(str(target / name))


def make_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "nc_v100_pr_curve_derivation_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(out_dir))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", default=str(PREDICTION_FILE))
    parser.add_argument("--locked-metrics-file", default=str(LOCKED_METRICS_FILE))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--recall-step", type=float, default=0.002)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = Path(args.prediction_file)
    locked_file = Path(args.locked_metrics_file)
    blockers: list[dict[str, str]] = []
    copied: list[str] = []

    if not prediction_file.exists():
        blockers.append({"status": "BLOCKED", "blocker": "missing_prediction_file", "detail": str(prediction_file)})
        write_blockers(out_dir / "BLOCKERS_PR_CURVE.csv", blockers)
        return
    if not locked_file.exists():
        blockers.append({"status": "BLOCKED", "blocker": "missing_locked_metrics_file", "detail": str(locked_file)})
        write_blockers(out_dir / "BLOCKERS_PR_CURVE.csv", blockers)
        return

    pred = pd.read_csv(prediction_file)
    metrics = pd.read_csv(locked_file)
    score_col = detect_column(list(pred.columns), SCORE_CANDIDATES)
    label_col = detect_column(list(pred.columns), LABEL_CANDIDATES)
    required_filter_cols = ["endpoint", "model", "seed", "feature_set"]
    missing_filter = [c for c in required_filter_cols if c not in pred.columns]
    if score_col is None:
        blockers.append({"status": "BLOCKED", "blocker": "missing_score_column", "detail": f"columns={list(pred.columns)}"})
    if label_col is None:
        blockers.append({"status": "BLOCKED", "blocker": "missing_label_column", "detail": f"columns={list(pred.columns)}"})
    if missing_filter:
        blockers.append({"status": "BLOCKED", "blocker": "missing_filter_columns", "detail": ",".join(missing_filter)})
    if blockers:
        write_blockers(out_dir / "BLOCKERS_PR_CURVE.csv", blockers)
        return

    pred = pred[
        pred["endpoint"].astype(str).eq(ENDPOINT)
        & pred["model"].astype(str).eq(MODEL)
        & pred["feature_set"].astype(str).isin(FEATURE_SETS)
        & pred["seed"].astype(int).isin(SEEDS)
    ].copy()
    if pred.empty:
        blockers.append({"status": "BLOCKED", "blocker": "empty_filtered_predictions", "detail": "No rows after primary Figure 3a filter."})
        write_blockers(out_dir / "BLOCKERS_PR_CURVE.csv", blockers)
        return

    raw_rows = []
    plot_rows = []
    summary_rows = []
    seed_curves_by_feature: dict[str, list[pd.DataFrame]] = {fs: [] for fs in FEATURE_SETS}
    point_order = 0

    for feature_set in FEATURE_SETS:
        for seed in SEEDS:
            sub = pred[pred["feature_set"].astype(str).eq(feature_set) & pred["seed"].astype(int).eq(seed)].copy()
            if sub.empty:
                blockers.append({"status": "BLOCKED", "blocker": "missing_group", "detail": f"{ENDPOINT}/{MODEL}/{seed}/{feature_set}"})
                continue
            y_true = pd.to_numeric(sub[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
            y_score = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(float)
            valid = np.isfinite(y_score)
            y_true = y_true[valid]
            y_score = y_score[valid]
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            auprc = float(average_precision_score(y_true, y_score))
            prevalence = float(np.mean(y_true))
            positive_count = int(np.sum(y_true))
            n_samples = int(len(y_true))
            locked = find_locked_auprc(metrics, seed, feature_set)
            diff = abs(auprc - locked) if np.isfinite(locked) else float("nan")
            qa_status = "PASS" if np.isfinite(diff) and diff <= AUPRC_TOL else "BLOCKED"
            if qa_status != "PASS":
                blockers.append(
                    {
                        "status": "BLOCKED",
                        "blocker": "auprc_mismatch",
                        "detail": f"seed={seed} feature_set={feature_set} derived={auprc} locked={locked} abs_diff={diff}",
                    }
                )
            summary_rows.append(
                {
                    "endpoint": ENDPOINT,
                    "model": MODEL,
                    "seed": seed,
                    "feature_set": feature_set,
                    "display_label": DISPLAY_LABELS[feature_set],
                    "n_samples": n_samples,
                    "positive_count": positive_count,
                    "prevalence": prevalence,
                    "auprc_derived": auprc,
                    "auprc_locked": locked,
                    "auprc_abs_diff": diff,
                    "qa_status": qa_status,
                }
            )
            thresholds_full = threshold_array(thresholds, len(precision))
            curve = pd.DataFrame({"recall": recall, "precision": precision, "threshold": thresholds_full})
            seed_curves_by_feature[feature_set].append(curve)
            for i, row in curve.iterrows():
                raw_rows.append(
                    {
                        "figure": "Figure 3",
                        "panel": "3a",
                        "curve_type": "seed_curve",
                        "endpoint": ENDPOINT,
                        "model": MODEL,
                        "seed": seed,
                        "feature_set": feature_set,
                        "display_label": DISPLAY_LABELS[feature_set],
                        "recall": row["recall"],
                        "precision": row["precision"],
                        "threshold": row["threshold"],
                        "auprc": auprc,
                        "prevalence": prevalence,
                        "n_samples": n_samples,
                        "positive_count": positive_count,
                        "color_key": COLOR_KEYS[feature_set],
                        "line_weight": 0.7,
                        "line_alpha": 0.35,
                        "point_order": int(i),
                        "source_prediction_file": str(prediction_file),
                    }
                )
            thinned = thin_curve(curve)
            for _, row in thinned.iterrows():
                plot_rows.append(
                    {
                        "figure": "Figure 3",
                        "panel": "3a",
                        "curve_type": "seed_curve",
                        "endpoint": ENDPOINT,
                        "model": MODEL,
                        "seed": seed,
                        "feature_set": feature_set,
                        "display_label": DISPLAY_LABELS[feature_set],
                        "recall": row["recall"],
                        "precision": row["precision"],
                        "threshold": row["threshold"],
                        "auprc": auprc,
                        "prevalence": prevalence,
                        "n_samples": n_samples,
                        "positive_count": positive_count,
                        "color_key": COLOR_KEYS[feature_set],
                        "line_weight": 0.7,
                        "line_alpha": 0.35,
                        "point_order": point_order,
                        "source_prediction_file": str(prediction_file),
                    }
                )
                point_order += 1

    summary = pd.DataFrame(summary_rows)
    for feature_set, curves in seed_curves_by_feature.items():
        if not curves:
            continue
        mean = mean_curve(curves, args.recall_step)
        sub_summary = summary[summary["feature_set"].eq(feature_set)]
        mean_auprc = float(sub_summary["auprc_derived"].mean())
        prevalence = float(sub_summary["prevalence"].mean())
        n_samples = int(sub_summary["n_samples"].iloc[0])
        positive_count = int(sub_summary["positive_count"].iloc[0])
        for _, row in mean.iterrows():
            plot_rows.append(
                {
                    "figure": "Figure 3",
                    "panel": "3a",
                    "curve_type": "seed_mean_curve",
                    "endpoint": ENDPOINT,
                    "model": MODEL,
                    "seed": "mean_41_42_43",
                    "feature_set": feature_set,
                    "display_label": DISPLAY_LABELS[feature_set],
                    "recall": row["recall"],
                    "precision": row["precision"],
                    "threshold": np.nan,
                    "auprc": mean_auprc,
                    "prevalence": prevalence,
                    "n_samples": n_samples,
                    "positive_count": positive_count,
                    "color_key": COLOR_KEYS[feature_set],
                    "line_weight": 2.2,
                    "line_alpha": 1.0,
                    "point_order": point_order,
                    "source_prediction_file": str(prediction_file),
                }
            )
            point_order += 1

    prevalence = float(summary["prevalence"].mean()) if not summary.empty else float("nan")
    positive_count = int(summary["positive_count"].iloc[0]) if not summary.empty else 0
    n_samples = int(summary["n_samples"].iloc[0]) if not summary.empty else 0
    for i, recall in enumerate([0.0, 1.0]):
        plot_rows.append(
            {
                "figure": "Figure 3",
                "panel": "3a",
                "curve_type": "prevalence_baseline",
                "endpoint": ENDPOINT,
                "model": MODEL,
                "seed": "baseline",
                "feature_set": "prevalence",
                "display_label": "Prevalence baseline",
                "recall": recall,
                "precision": prevalence,
                "threshold": np.nan,
                "auprc": prevalence,
                "prevalence": prevalence,
                "n_samples": n_samples,
                "positive_count": positive_count,
                "color_key": "prevalence",
                "line_weight": 0.9,
                "line_alpha": 0.7,
                "point_order": point_order + i,
                "source_prediction_file": str(prediction_file),
            }
        )

    raw = pd.DataFrame(raw_rows)
    plot = pd.DataFrame(plot_rows)
    summary.to_csv(out_dir / "Figure3a_oof_pr_curve_summary.csv", index=False)
    raw.to_csv(out_dir / "Figure3a_oof_pr_curve_seed_raw.csv", index=False)
    plot.to_csv(out_dir / "Figure3a_oof_pr_curve_plot_ready.csv", index=False)
    write_blockers(out_dir / "BLOCKERS_PR_CURVE.csv", blockers)

    max_diff = float(summary["auprc_abs_diff"].max()) if not summary.empty else float("nan")
    qa_lines = [
        "# Figure 3a OOF PR Curve QA",
        "",
        f"Created at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Source prediction file: `{prediction_file}`",
        f"Locked metrics file: `{locked_file}`",
        f"Score column: `{score_col}`",
        f"Label column: `{label_col}`",
        f"Endpoint/model: `{ENDPOINT}` / `{MODEL}`",
        f"Feature sets: `{', '.join(FEATURE_SETS)}`",
        f"Seeds: `{', '.join(str(s) for s in SEEDS)}`",
        f"Raw curve rows: {len(raw)}",
        f"Plot-ready rows: {len(plot)}",
        f"Max absolute AUPRC difference vs locked metrics: {max_diff:.12g}",
        f"QA status: {'BLOCKED' if blockers else 'PASS'}",
        "",
        "## AUPRC Check",
        "",
        summary.to_markdown(index=False),
    ]
    if blockers:
        qa_lines.extend(["", "## Blockers", "", pd.DataFrame(blockers).to_markdown(index=False)])
    (out_dir / "Figure3a_oof_pr_curve_QA.md").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")

    copy_optional_outputs(out_dir, copied)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "NC v1.00 Figure 3a OOF PR curve derivation",
        "source_prediction_file": str(prediction_file),
        "locked_metrics_file": str(locked_file),
        "output_dir": str(out_dir),
        "endpoint": ENDPOINT,
        "model": MODEL,
        "seeds": SEEDS,
        "feature_sets": FEATURE_SETS,
        "score_column": score_col,
        "label_column": label_col,
        "raw_rows": int(len(raw)),
        "plot_ready_rows": int(len(plot)),
        "max_auprc_abs_diff": max_diff,
        "auprc_tolerance": AUPRC_TOL,
        "copied_to_optional_source_data": copied,
        "blocked": bool(blockers),
    }
    (out_dir / "Figure3a_oof_pr_curve_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = make_zip(out_dir)
    final = [
        "# CODEX Final Message: NC v1.00 Figure 3a PR Curve",
        "",
        f"Status: {'BLOCKED' if blockers else 'PASS'}",
        f"Input prediction file: `{prediction_file}`",
        f"Plot-ready rows: {len(plot)}",
        f"Raw curve rows: {len(raw)}",
        f"Maximum absolute AUPRC difference from locked metrics: {max_diff:.12g}",
        f"Optional source-data copies: {len(copied)}",
        f"Zip path: `{zip_path}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
    ]
    if copied:
        final.extend(["", "## Copied Files", "", *[f"- `{p}`" for p in copied]])
    if blockers:
        final.extend(["", "## Blockers", "", pd.DataFrame(blockers).to_markdown(index=False)])
    (out_dir / "CODEX_FINAL_MESSAGE.md").write_text("\n".join(final) + "\n", encoding="utf-8")
    print(f"[nc_v100_pr] status={'BLOCKED' if blockers else 'PASS'}")
    print(f"[nc_v100_pr] plot_ready_rows={len(plot)} raw_rows={len(raw)} max_auprc_abs_diff={max_diff:.12g}")
    print(f"[nc_v100_pr] zip={zip_path}")


if __name__ == "__main__":
    main()
