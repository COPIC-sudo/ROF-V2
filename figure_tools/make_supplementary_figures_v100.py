#!/usr/bin/env python3
"""
Generate v100 Supplementary Figures for the actionability manuscript.

Usage
-----
python make_supplementary_figures_v100.py \
  --data ROF_results_v100_evidence_lock.zip \
  --out supplementary_figures_v100

The script accepts either the v100 evidence-lock zip file or an already extracted
ROF_results_v100_evidence_lock directory. It writes vector PDF and high-resolution PNG
files plus a small source-data manifest for each figure.

Required packages: pandas, numpy, matplotlib
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import tempfile
import textwrap
import zipfile
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Global visual style
# -----------------------------
COL = {
    "black": "#222222",
    "dark": "#3A3A3A",
    "mid": "#6F6F6F",
    "light": "#D9D9D9",
    "very_light": "#F2F2F2",
    "blue": "#4C78A8",
    "blue_light": "#A8C5E5",
    "orange": "#F58518",
    "green": "#54A24B",
    "purple": "#B279A2",
    "red": "#E45756",
    "teal": "#72B7B2",
    "brown": "#9D755D",
}

FEATURE_COLORS = {
    "direct_action_ratios_only": COL["orange"],
    "explicit_ratio_field_excluded_current": COL["purple"],
    "strict_spatial_no_action": COL["mid"],
    "strict_temporal_dynamics": COL["blue"],
}

SCORE_COLORS = {
    "distance_inverse": COL["mid"],
    "TTC_inverse": COL["orange"],
    "ROF_v2_composite": COL["blue"],
    "ROF_v2_no_ASR_composite": COL["purple"],
    "ROF_v2_no_asr_composite": COL["purple"],
    "temporal_composite": COL["green"],
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.4,
    "figure.titlesize": 9.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

LABEL_ORDER = ["High", "Reduced", "Critical", "Candidate-set infeasible"]
LABEL_SHORT = ["High", "Reduced", "Critical", "Cand.-set\ninfeasible"]
LEGACY_ORDER = ["safe", "caution", "warning", "emergency"]

ENDPOINT_DISPLAY = {
    "map_critical_or_worse": "Map critical-or-worse",
    "map_candidate_set_infeasible": "Map candidate-set\ninfeasible",
    "nomap_critical_or_worse": "No-map critical-or-worse",
    "nomap_candidate_set_infeasible": "No-map candidate-set\ninfeasible",
    "map_critical_or_worse_cv_fallback": "Map critical-or-worse\n(CV-fallback)",
    "map_candidate_set_infeasible_cv_fallback": "Map candidate-set\ninfeasible (CV-fallback)",
}

FEATURE_DISPLAY = {
    "direct_action_ratios_only": "Direct action\nratios only",
    "explicit_ratio_field_excluded_current": "Explicit-ratio-field-\nexcluded current",
    "strict_spatial_no_action": "Strict spatial\nno-action",
    "strict_temporal_dynamics": "Strict temporal\ndynamics",
}

VARIANT_DISPLAY = {
    "reference_h3_b3_base7_skip": "3 s / 3 m /\nbase-7 / skip",
    "horizon_h2_b3_base7_skip": "2 s horizon",
    "horizon_h4_b3_base7_skip": "4 s horizon",
    "buffer_h3_b2_base7_skip": "2 m buffer",
    "buffer_h3_b4_base7_skip": "4 m buffer",
    "action_h3_b3_extended_skip": "extended\nactions",
    "future_h3_b3_base7_cvfallback": "CV-fallback",
}

SCORE_DISPLAY = {
    "distance_inverse": "Distance",
    "TTC_inverse": "TTC",
    "ROF_v2_composite": "ROF-v2",
    "ROF_v2_no_ASR_composite": "ROF-v2 no-ASR",
    "ROF_v2_no_asr_composite": "ROF-v2 no-ASR",
    "temporal_composite": "Temporal",
}

VAR_DISPLAY = {
    "current_min_distance_m": "Current min\ndistance",
    "valid_ttc_only": "Valid TTC",
    "no_ttc_as_category": "No-TTC as\ncategory",
    "capped_ttc_prespecified_10s": "Capped TTC\n(10 s)",
    "inverse_ttc_prespecified": "Inverse TTC",
    "legacy_sentinel": "Legacy\nsentinel",
}

# -----------------------------
# Utility functions
# -----------------------------
def resolve_data_root(data_arg: str | Path, out_dir: Path) -> Path:
    data_path = Path(data_arg).expanduser().resolve()
    if data_path.is_dir():
        if (data_path / "00_paper_architecture_and_index").exists():
            return data_path
        nested = list(data_path.glob("ROF_results_v100_evidence_lock"))
        if nested:
            return nested[0]
        raise FileNotFoundError(f"Cannot find ROF_results_v100_evidence_lock under {data_path}")
    if data_path.suffix.lower() == ".zip":
        extract_dir = out_dir / "_extracted_ROF_results_v100_evidence_lock"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_path) as zf:
            zf.extractall(extract_dir)
        nested = extract_dir / "ROF_results_v100_evidence_lock"
        if nested.exists():
            return nested
        # fallback: use first directory that contains the index folder
        for p in extract_dir.rglob("00_paper_architecture_and_index"):
            return p.parent
    raise FileNotFoundError(f"Unsupported --data path: {data_path}")


def read_csv(root: Path, rel: str, required: bool = True) -> Optional[pd.DataFrame]:
    p = root / rel
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing required source table: {rel}")
        return None
    return pd.read_csv(p)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().strip(".")
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=600 if fmt in {"png", "tif", "tiff"} else None)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=COL["black"])


def clean_axis(ax: plt.Axes, grid: Optional[str] = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid == "x":
        ax.grid(axis="x", color=COL["very_light"], lw=0.7, zorder=0)
    elif grid == "y":
        ax.grid(axis="y", color=COL["very_light"], lw=0.7, zorder=0)


def percent(x: float, nd: int = 1) -> str:
    return f"{100*x:.{nd}f}%"


def metric_display(m: str) -> str:
    return {"auprc": "AUPRC", "auroc": "AUROC", "recall_at_5pct_fpr": "Recall at nominal\n5% FPR"}.get(m, m)


def extract_feature_key(comparison: str) -> str:
    tail = comparison.split("strong_baseline_cv_plus_")[-1]
    return tail


def add_zero_line(ax: plt.Axes, vertical: bool = True) -> None:
    if vertical:
        ax.axvline(0, color=COL["black"], lw=0.8, ls="--", alpha=0.7, zorder=1)
    else:
        ax.axhline(0, color=COL["black"], lw=0.8, ls="--", alpha=0.7, zorder=1)


def plot_heatmap(ax: plt.Axes, data: np.ndarray, row_labels: list[str], col_labels: list[str],
                 title: str, cbar_label: str = "Row fraction") -> None:
    cmap = LinearSegmentedColormap.from_list("grayblue", ["#FFFFFF", "#D7E6F5", "#4C78A8"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.01, np.nanmax(data)))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{100*val:.0f}", ha="center", va="center", fontsize=6.0,
                    color="white" if val > 0.45*np.nanmax(data) else COL["black"])
    ax.set_title(title, loc="left", pad=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_manifest(out_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(out_dir / "supplementary_figure_source_manifest.csv", index=False)

# -----------------------------
# Supplementary Figure 1
# -----------------------------
def make_suppfig1(root: Path, out_dir: Path, formats: Sequence[str]) -> dict:
    ttc = read_csv(root, "04_supplementary_figure_source_data/SuppFig1_proximity_and_ttc/SuppFig1A_ttc_sensitivity.csv")
    miss = read_csv(root, "04_supplementary_figure_source_data/SuppFig1_proximity_and_ttc/SuppFig1B_ttc_missingness.csv")
    prox = read_csv(root, "03_main_figure_source_data/Figure2_endpoint_proximity/Figure2B_proximity_vs_actionability_map.csv")
    shift = read_csv(root, "03_main_figure_source_data/Figure2_endpoint_proximity/Figure2D_map_vs_nomap_shift.csv")

    fig, axs = plt.subplots(2, 2, figsize=(7.8, 5.9), gridspec_kw={"wspace": 0.52, "hspace": 0.58})
    ax = axs[0, 0]
    # TTC missingness stacked bar
    valid = float(miss.loc[miss["category"].eq("valid_ttc"), "rate"].iloc[0])
    invalid = float(miss.loc[miss["category"].eq("invalid_or_no_closing_ttc"), "rate"].iloc[0])
    ax.barh([0], [valid], color=COL["blue"], height=0.35, label="valid TTC")
    ax.barh([0], [invalid], left=[valid], color=COL["light"], height=0.35, label="invalid/no-closing")
    ax.text(valid / 2, 0, percent(valid, 1), va="center", ha="center", color="white", fontsize=7)
    ax.text(valid + invalid / 2, 0, percent(invalid, 1), va="center", ha="center", color=COL["black"], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of Waymo evaluation samples")
    ax.set_title("TTC availability", loc="left", pad=6)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    clean_axis(ax, grid="x")
    panel_label(ax, "a")

    # Spearman sensitivity
    ax = axs[0, 1]
    labels = ["map_actionability_label_id", "nomap_actionability_label_id"]
    offsets = {labels[0]: -0.10, labels[1]: 0.10}
    colors = {labels[0]: COL["blue"], labels[1]: COL["orange"]}
    variables = ["current_min_distance_m", "valid_ttc_only", "no_ttc_as_category",
                 "capped_ttc_prespecified_10s", "inverse_ttc_prespecified", "legacy_sentinel"]
    y_base = np.arange(len(variables))[::-1]
    for lab in labels:
        sub = ttc[ttc["label"].eq(lab)].set_index("variable").loc[variables].reset_index()
        ys = y_base + offsets[lab]
        ax.hlines(ys, sub["ci_low"], sub["ci_high"], color=colors[lab], lw=1.0)
        ax.scatter(sub["spearman"], ys, s=20, color=colors[lab], edgecolor="white", linewidth=0.4, zorder=3,
                   label="Map" if lab.startswith("map_") else "No-map")
    add_zero_line(ax)
    ax.set_yticks(y_base)
    ax.set_yticklabels([VAR_DISPLAY[v] for v in variables])
    ax.set_xlim(-0.62, 0.16)
    ax.set_xlabel("Spearman correlation with actionability severity")
    ax.set_title("Distance is moderate; TTC variants are weak", loc="left", pad=6)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.34), ncol=2)
    clean_axis(ax, grid="x")
    panel_label(ax, "b")

    # Proximity cross-tab heatmap
    ax = axs[1, 0]
    piv = prox.pivot_table(index="original_label_name", columns="display_label", values="row_fraction", aggfunc="sum")
    piv = piv.reindex(index=LEGACY_ORDER, columns=LABEL_ORDER)
    plot_heatmap(ax, piv.values, [s.capitalize() for s in LEGACY_ORDER], LABEL_SHORT,
                 "Legacy proximity vs map actionability", "Row fraction")
    panel_label(ax, "c")

    # Map-to-no-map shift heatmap
    ax = axs[1, 1]
    ct = shift[shift["row_type"].eq("map_nomap_crosstab")].copy()
    if not ct.empty:
        # Display labels correspond to map rows; columns are no-map label ids.
        id_to_label = {0: "High", 1: "Reduced", 2: "Critical", 3: "Candidate-set infeasible"}
        ct["map_label"] = ct["map_label_id"].astype(int).map(id_to_label)
        ct["nomap_label"] = ct["nomap_label_id"].astype(int).map(id_to_label)
        piv2 = ct.pivot_table(index="map_label", columns="nomap_label", values="row_fraction", aggfunc="sum")
        piv2 = piv2.reindex(index=LABEL_ORDER, columns=LABEL_ORDER)
        plot_heatmap(ax, piv2.values, LABEL_SHORT, LABEL_SHORT, "Map-constrained to no-map labels", "Row fraction")
    else:
        ax.axis("off")
    panel_label(ax, "d")

    fig.suptitle("Supplementary Fig. 1 | Proximity, TTC and map/no-map endpoint diagnostics", x=0.01, ha="left")
    save_figure(fig, out_dir, "Supplementary_Figure_1_proximity_ttc_diagnostics", formats)
    return {"figure": "Supplementary_Figure_1", "files": "SuppFig1A_ttc_sensitivity.csv; SuppFig1B_ttc_missingness.csv; Figure2B_proximity_vs_actionability_map.csv; Figure2D_map_vs_nomap_shift.csv"}

# -----------------------------
# Supplementary Figure 2
# -----------------------------
def make_suppfig2(root: Path, out_dir: Path, formats: Sequence[str]) -> dict:
    metrics = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable4_FullWaymoOOFMetrics.csv")
    deltas = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable5_FullWaymoPairedDeltas.csv")
    op = read_csv(root, "03_main_figure_source_data/Figure3_waymo_primary_oof/Figure3C_calibrated_operating_points.csv")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), gridspec_kw={"wspace": 0.36, "hspace": 0.55})
    baseline = "strong_baseline_cv"
    enhanced = "strong_baseline_cv_plus_strict_temporal_dynamics"
    f = metrics[(metrics["level"].eq("fold")) & (metrics["endpoint"].eq("map_critical_or_worse")) &
                (metrics["model"].eq("rf")) & (metrics["feature_set"].isin([baseline, enhanced]))].copy()

    # Panel a: fold-level AUPRC paired lines
    ax = axs[0, 0]
    for (seed, fold), sub in f.groupby(["seed", "outer_fold"]):
        if set(sub["feature_set"]) >= {baseline, enhanced}:
            b = float(sub.loc[sub["feature_set"].eq(baseline), "auprc"].iloc[0])
            e = float(sub.loc[sub["feature_set"].eq(enhanced), "auprc"].iloc[0])
            ax.plot([0, 1], [b, e], color=COL["light"], lw=0.8, zorder=1)
            ax.scatter([0, 1], [b, e], color=[COL["mid"], COL["blue"]], s=14, zorder=2)
    pooled = metrics[(metrics["level"].eq("pooled_oof")) & (metrics["endpoint"].eq("map_critical_or_worse")) &
                     (metrics["model"].eq("rf")) & (metrics["feature_set"].isin([baseline, enhanced]))]
    means = pooled.groupby("feature_set")["auprc"].mean()
    ax.scatter([0, 1], [means.get(baseline, np.nan), means.get(enhanced, np.nan)], marker="D", s=55,
               color=[COL["black"], COL["blue"]], edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Strong + CV", "+ strict temporal"])
    ax.set_ylabel("Fold-level AUPRC")
    ax.set_title("Fold-level paired OOF AUPRC", loc="left", pad=6)
    clean_axis(ax, grid="y")
    panel_label(ax, "a")

    # Panel b: primary deltas by seed and metric
    ax = axs[0, 1]
    d = deltas[(deltas["endpoint"].eq("map_critical_or_worse")) &
               (deltas["model"].eq("rf")) &
               (deltas["enhanced_feature_set"].eq(enhanced)) &
               (deltas["metric"].isin(["auprc", "recall_at_5pct_fpr"]))].copy()
    metrics_order = ["auprc", "recall_at_5pct_fpr"]
    y = 0
    yticks, ylabels = [], []
    for mi, m in enumerate(metrics_order):
        sub = d[d["metric"].eq(m)].sort_values("seed")
        offsets = np.linspace(-0.18, 0.18, len(sub))
        for off, (_, row) in zip(offsets, sub.iterrows()):
            yy = y + off
            ax.hlines(yy, row["ci_low"], row["ci_high"], color=COL["blue"] if m == "auprc" else COL["green"], lw=1.0)
            ax.scatter(row["delta"], yy, s=22, color=COL["blue"] if m == "auprc" else COL["green"], edgecolor="white", linewidth=0.4, zorder=3)
        yticks.append(y)
        ylabels.append(metric_display(m))
        y += 1
    add_zero_line(ax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Δ over Strong + CV")
    ax.set_title("Pooled primary deltas across RF seeds", loc="left", pad=6)
    clean_axis(ax, grid="x")
    panel_label(ax, "b")

    # Panel c: achieved FPR by fold
    ax = axs[1, 0]
    op2 = op[op["feature_set"].isin([baseline, enhanced])].copy()
    xmap = {baseline: 0, enhanced: 1}
    for fs, sub in op2.groupby("feature_set"):
        xs = np.full(len(sub), xmap[fs]) + np.linspace(-0.12, 0.12, len(sub))
        ax.scatter(xs, sub["outer_test_achieved_fpr"], s=20, color=COL["mid"] if fs == baseline else COL["blue"], alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.hlines(sub["outer_test_achieved_fpr"].median(), xmap[fs]-0.18, xmap[fs]+0.18, color=COL["black"], lw=1.2)
    ax.axhline(0.05, color=COL["black"], ls="--", lw=0.8, alpha=0.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Strong + CV", "+ strict temporal"])
    ax.set_ylabel("Outer-test achieved FPR")
    ax.set_title("Calibration-derived nominal 5% FPR", loc="left", pad=6)
    clean_axis(ax, grid="y")
    panel_label(ax, "c")

    # Panel d: recall/precision at operating point, pooled OOF
    ax = axs[1, 1]
    p = pooled[pooled["feature_set"].isin([baseline, enhanced])].copy()
    stats = p.groupby("feature_set")[["recall", "precision"]].agg(["mean", "min", "max"])
    x = np.arange(2)
    width = 0.32
    for j, metric in enumerate(["recall", "precision"]):
        vals = [stats.loc[fs, (metric, "mean")] for fs in [baseline, enhanced]]
        lo = [stats.loc[fs, (metric, "mean")] - stats.loc[fs, (metric, "min")] for fs in [baseline, enhanced]]
        hi = [stats.loc[fs, (metric, "max")] - stats.loc[fs, (metric, "mean")] for fs in [baseline, enhanced]]
        ax.bar(x + (j-0.5)*width, vals, width=width, color=COL["green"] if metric=="recall" else COL["purple"], alpha=0.85, label=metric.capitalize())
        ax.errorbar(x + (j-0.5)*width, vals, yerr=[lo, hi], fmt="none", ecolor=COL["black"], lw=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Strong + CV", "+ strict temporal"])
    ax.set_ylim(0, max(0.62, ax.get_ylim()[1]))
    ax.set_ylabel("Operating-point value")
    ax.set_title("Pooled OOF low-FPR operating point", loc="left", pad=6)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    clean_axis(ax, grid="y")
    panel_label(ax, "d")

    fig.suptitle("Supplementary Fig. 2 | Waymo out-of-fold stability and calibrated operating points", x=0.01, ha="left")
    save_figure(fig, out_dir, "Supplementary_Figure_2_oof_calibration_diagnostics", formats)
    return {"figure": "Supplementary_Figure_2", "files": "SuppTable4_FullWaymoOOFMetrics.csv; SuppTable5_FullWaymoPairedDeltas.csv; Figure3C_calibrated_operating_points.csv"}

# -----------------------------
# Supplementary Figure 3
# -----------------------------
def make_suppfig3(root: Path, out_dir: Path, formats: Sequence[str]) -> dict:
    df = read_csv(root, "04_supplementary_figure_source_data/SuppFig2_feature_context_full/SuppFig2A_feature_context_ready_table.csv")
    df = df[df["metric"].eq("auprc")].copy()
    df["feature_key"] = df["comparison"].map(extract_feature_key)
    endpoints = ["map_critical_or_worse", "map_candidate_set_infeasible", "nomap_critical_or_worse", "nomap_candidate_set_infeasible"]
    features = ["direct_action_ratios_only", "explicit_ratio_field_excluded_current", "strict_spatial_no_action", "strict_temporal_dynamics"]

    fig, axs = plt.subplots(2, 2, figsize=(8.0, 6.2), gridspec_kw={"wspace": 0.62, "hspace": 0.72})

    # Panel a: heatmap mean deltas
    ax = axs[0, 0]
    mat = np.full((len(endpoints), len(features)), np.nan)
    for i, ep in enumerate(endpoints):
        for j, ft in enumerate(features):
            sub = df[(df["endpoint"].eq(ep)) & (df["feature_key"].eq(ft))]
            if not sub.empty:
                mat[i, j] = sub["delta"].mean()
    cmap = LinearSegmentedColormap.from_list("whiteblue", ["#FFFFFF", "#E8F0F8", COL["blue"]])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=min(0, np.nanmin(mat)), vmax=np.nanmax(mat))
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([FEATURE_DISPLAY[f].replace("\n", " ") for f in features], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(endpoints)))
    ax.set_yticklabels([ENDPOINT_DISPLAY[e] for e in endpoints])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=5.8,
                    color="white" if mat[i, j] > 0.65*np.nanmax(mat) else COL["black"])
    ax.set_title("Mean ΔAUPRC across seeds", loc="left", pad=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("ΔAUPRC", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8)
    panel_label(ax, "a")

    def forest(ax: plt.Axes, endpoint: str, title: str, xlim: Optional[tuple[float,float]] = None):
        sub = df[df["endpoint"].eq(endpoint)].copy()
        feature_order = features
        ymap = {f: len(feature_order)-1-i for i, f in enumerate(feature_order)}
        for ft in feature_order:
            ss = sub[sub["feature_key"].eq(ft)].sort_values("seed")
            col = FEATURE_COLORS.get(ft, COL["blue"])
            offs = np.linspace(-0.15, 0.15, max(1, len(ss)))
            for off, (_, row) in zip(offs, ss.iterrows()):
                yy = ymap[ft] + off
                ax.hlines(yy, row["ci_low"], row["ci_high"], color=col, lw=1.0, alpha=0.7)
                ax.scatter(row["delta"], yy, s=22, color=col, edgecolor="white", linewidth=0.4, zorder=3)
            if not ss.empty:
                ax.scatter(ss["delta"].mean(), ymap[ft], marker="D", s=55, color=col, edgecolor="white", linewidth=0.5, zorder=4)
        add_zero_line(ax)
        ax.set_yticks([ymap[f] for f in feature_order])
        ax.set_yticklabels([FEATURE_DISPLAY[f] for f in feature_order])
        ax.set_xlabel("ΔAUPRC over Strong + CV")
        ax.set_title(title, loc="left", pad=6)
        if xlim:
            ax.set_xlim(*xlim)
        clean_axis(ax, grid="x")

    forest(axs[0, 1], "map_critical_or_worse", "Primary endpoint", (-0.015, 0.13))
    panel_label(axs[0, 1], "b")
    forest(axs[1, 0], "map_candidate_set_infeasible", "Secondary map candidate-set infeasible", (-0.02, 0.16))
    panel_label(axs[1, 0], "c")

    # Panel d: strict temporal across endpoints
    ax = axs[1, 1]
    st = df[df["feature_key"].eq("strict_temporal_dynamics")].copy()
    ymap = {ep: len(endpoints)-1-i for i, ep in enumerate(endpoints)}
    for ep in endpoints:
        ss = st[st["endpoint"].eq(ep)].sort_values("seed")
        offs = np.linspace(-0.15, 0.15, len(ss))
        for off, (_, row) in zip(offs, ss.iterrows()):
            yy = ymap[ep] + off
            ax.hlines(yy, row["ci_low"], row["ci_high"], color=COL["blue"], lw=1.0, alpha=0.72)
            ax.scatter(row["delta"], yy, s=22, color=COL["blue"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter(ss["delta"].mean(), ymap[ep], marker="D", s=55, color=COL["blue"], edgecolor="white", linewidth=0.5, zorder=4)
    add_zero_line(ax)
    ax.set_yticks([ymap[e] for e in endpoints])
    ax.set_yticklabels([ENDPOINT_DISPLAY[e] for e in endpoints])
    ax.set_xlabel("ΔAUPRC for strict temporal dynamics")
    ax.set_title("Strict temporal context across endpoints", loc="left", pad=6)
    clean_axis(ax, grid="x")
    panel_label(ax, "d")

    fig.suptitle("Supplementary Fig. 3 | Full feature-group and secondary-endpoint audit", x=0.01, ha="left")
    save_figure(fig, out_dir, "Supplementary_Figure_3_full_feature_audit", formats)
    return {"figure": "Supplementary_Figure_3", "files": "SuppFig2A_feature_context_ready_table.csv"}

# -----------------------------
# Supplementary Figure 4
# -----------------------------
def make_suppfig4(root: Path, out_dir: Path, formats: Sequence[str]) -> dict:
    shift = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable8_CVFallbackLabelShift.csv")
    deltas = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable9_CVFallbackOOFDeltas.csv")
    imp = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable10_CVFallbackImputationSummary.csv")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), gridspec_kw={"wspace": 0.40, "hspace": 0.55})
    s = shift.iloc[0]
    # Panel a: label shift summary
    ax = axs[0, 0]
    vals = [s["label_changed_fraction"], 1 - s["severe_set_jaccard"], s["mean_imputed_fraction"], s["p95_imputed_fraction"]]
    labs = ["Labels\nchanged", "1 - severe\nJaccard", "Mean sample\nimputed", "P95 sample\nimputed"]
    colors = [COL["red"], COL["purple"], COL["blue"], COL["teal"]]
    ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.86)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.03, percent(v, 2 if v < 0.01 else 1), ha="center", va="bottom", fontsize=6.4)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labs)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, max(vals)*1.25)
    ax.set_title("CV-fallback changed few labels", loc="left", pad=6)
    clean_axis(ax, grid="y")
    panel_label(ax, "a")

    # Panel b: prevalence reference vs CV fallback
    ax = axs[0, 1]
    prev = np.array([[s["reference_critical_or_worse_prevalence"], s["cv_critical_or_worse_prevalence"]],
                     [s["reference_infeasible_prevalence"], s["cv_infeasible_prevalence"]]])
    x = np.arange(2)
    width = 0.32
    ax.bar(x - width/2, prev[:, 0], width=width, color=COL["light"], label="skip-invalid reference")
    ax.bar(x + width/2, prev[:, 1], width=width, color=COL["blue"], label="CV-fallback")
    ax.set_xticks(x)
    ax.set_xticklabels(["Critical-or-worse", "Candidate-set\ninfeasible"])
    ax.set_ylabel("Endpoint prevalence")
    ax.set_title("Endpoint prevalence is nearly unchanged", loc="left", pad=6)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax, grid="y")
    panel_label(ax, "b")

    def cv_delta_panel(ax: plt.Axes, endpoint: str, title: str, xlim: Optional[tuple[float,float]] = None):
        d = deltas[(deltas["endpoint"].eq(endpoint)) & (deltas["metric"].isin(["auprc", "recall_at_5pct_fpr"]))].copy()
        order = ["auprc", "recall_at_5pct_fpr"]
        ymap = {m: len(order)-1-i for i, m in enumerate(order)}
        for m in order:
            sub = d[d["metric"].eq(m)].sort_values("seed")
            col = COL["blue"] if m == "auprc" else COL["green"]
            offs = np.linspace(-0.15, 0.15, len(sub))
            for off, (_, row) in zip(offs, sub.iterrows()):
                yy = ymap[m] + off
                ax.hlines(yy, row["ci_low"], row["ci_high"], color=col, lw=1.0, alpha=0.75)
                ax.scatter(row["delta"], yy, s=22, color=col, edgecolor="white", linewidth=0.4, zorder=3)
            ax.scatter(sub["delta"].mean(), ymap[m], marker="D", s=55, color=col, edgecolor="white", linewidth=0.5, zorder=4)
        add_zero_line(ax)
        ax.set_yticks([ymap[m] for m in order])
        ax.set_yticklabels([metric_display(m) for m in order])
        ax.set_xlabel("Δ over Strong + CV")
        ax.set_title(title, loc="left", pad=6)
        if xlim:
            ax.set_xlim(*xlim)
        clean_axis(ax, grid="x")

    cv_delta_panel(axs[1, 0], "map_critical_or_worse_cv_fallback", "Primary endpoint under CV-fallback", (0, 0.13))
    panel_label(axs[1, 0], "c")
    cv_delta_panel(axs[1, 1], "map_candidate_set_infeasible_cv_fallback", "Candidate-set infeasible under CV-fallback", (0, 0.31))
    panel_label(axs[1, 1], "d")

    fig.suptitle("Supplementary Fig. 4 | CV-fallback future-handling sensitivity", x=0.01, ha="left")
    save_figure(fig, out_dir, "Supplementary_Figure_4_cv_fallback_sensitivity", formats)
    return {"figure": "Supplementary_Figure_4", "files": "SuppTable8_CVFallbackLabelShift.csv; SuppTable9_CVFallbackOOFDeltas.csv; SuppTable10_CVFallbackImputationSummary.csv"}

# -----------------------------
# Supplementary Figure 5
# -----------------------------
def make_suppfig5(root: Path, out_dir: Path, formats: Sequence[str]) -> dict:
    label = read_csv(root, "04_supplementary_figure_source_data/SuppFig5_endpoint_design_label_robustness/SuppFig5A_label_design_stability.csv")
    aligned = read_csv(root, "04_supplementary_figure_source_data/SuppFig6_aligned_feature_robustness/SuppFig6A_aligned_feature_deltas.csv")
    audit = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable14_AlignedFeatureGenerationAudit.csv")
    order = ["reference_h3_b3_base7_skip", "horizon_h2_b3_base7_skip", "horizon_h4_b3_base7_skip",
             "buffer_h3_b2_base7_skip", "buffer_h3_b4_base7_skip", "action_h3_b3_extended_skip",
             "future_h3_b3_base7_cvfallback"]
    label = label.set_index("variant_id").loc[order].reset_index()

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw={"wspace": 0.38, "hspace": 0.58})
    x = np.arange(len(order))

    # Panel a label changed and severe Jaccard
    ax = axs[0, 0]
    ax.bar(x, label["label_changed_fraction"], color=COL["blue"], alpha=0.82, label="Label changed")
    ax2 = ax.twinx()
    ax2.plot(x, label["severe_set_jaccard_vs_reference"], color=COL["red"], marker="o", lw=1.0, label="Severe-set Jaccard")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_DISPLAY[v] for v in order], rotation=35, ha="right")
    ax.set_ylabel("Fraction of labels changed")
    ax2.set_ylabel("Severe-set Jaccard")
    ax.set_title("Endpoint assignments are design-dependent", loc="left", pad=6)
    ax.set_ylim(0, max(0.36, label["label_changed_fraction"].max()*1.15))
    ax2.set_ylim(0, 1.05)
    clean_axis(ax, grid="y")
    ax2.spines["top"].set_visible(False)
    # Combined legend
    lines, labs = [], []
    for a in [ax, ax2]:
        h, l = a.get_legend_handles_labels(); lines += h; labs += l
    ax.legend(lines, labs, frameon=False, loc="upper left")
    panel_label(ax, "a")

    # Panel b prevalence
    ax = axs[0, 1]
    ax.bar(x - 0.18, label["critical_or_worse_prevalence"], width=0.36, color=COL["blue"], label="Critical-or-worse")
    ax.bar(x + 0.18, label["candidate_set_infeasible_prevalence"], width=0.36, color=COL["orange"], label="Candidate-set infeasible")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_DISPLAY[v] for v in order], rotation=35, ha="right")
    ax.set_ylabel("Variant prevalence")
    ax.set_title("Endpoint prevalence shifts across variants", loc="left", pad=6)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax, grid="y")
    panel_label(ax, "b")

    # Panel c aligned delta forest
    ax = axs[1, 0]
    aligned = aligned[aligned["metric"].eq("auprc")].copy()
    ymap = {v: len(order)-1-i for i, v in enumerate(order)}
    for v in order:
        sub = aligned[aligned["variant_id"].eq(v)].sort_values("seed")
        offs = np.linspace(-0.15, 0.15, len(sub))
        for off, (_, row) in zip(offs, sub.iterrows()):
            yy = ymap[v] + off
            ax.hlines(yy, row["ci_low"], row["ci_high"], color=COL["blue"], lw=1.0, alpha=0.75)
            ax.scatter(row["delta"], yy, s=22, color=COL["blue"], edgecolor="white", linewidth=0.4, zorder=3)
        if not sub.empty:
            ax.scatter(sub["delta"].mean(), ymap[v], marker="D", s=55, color=COL["blue"], edgecolor="white", linewidth=0.5, zorder=4)
    add_zero_line(ax)
    ax.set_yticks([ymap[v] for v in order])
    ax.set_yticklabels([VARIANT_DISPLAY[v].replace("\n", " ") for v in order])
    ax.set_xlabel("Aligned ΔAUPRC")
    ax.set_title("Aligned label-and-feature sensitivity remains positive", loc="left", pad=6)
    clean_axis(ax, grid="x")
    panel_label(ax, "c")

    # Panel d feature regeneration audit counts
    ax = axs[1, 1]
    # Count fields in the first row; all rows have same family counts in v100.
    row = audit.iloc[0]
    counts = {
        "Current-state\nfields reused": len(str(row["current_state_fields"]).split(";")),
        "CV occupancy\nfields regenerated": len(str(row["cv_fields"]).split(";")),
        "Strict temporal\nfields regenerated": len(str(row["strict_temporal_fields"]).split(";")),
    }
    bars = ax.bar(np.arange(len(counts)), list(counts.values()), color=[COL["light"], COL["teal"], COL["blue"]])
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{int(b.get_height())}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(list(counts.keys()), rotation=10, ha="center", fontsize=6.2)
    ax.set_ylabel("Number of fields")
    ax.set_title("Feature treatment in aligned sensitivity", loc="left", pad=6)
    clean_axis(ax, grid="y")
    panel_label(ax, "d")

    fig.suptitle("Supplementary Fig. 5 | Endpoint-design label stability and aligned feature robustness", x=0.01, ha="left")
    save_figure(fig, out_dir, "Supplementary_Figure_5_endpoint_design_aligned_robustness", formats)
    return {"figure": "Supplementary_Figure_5", "files": "SuppFig5A_label_design_stability.csv; SuppFig6A_aligned_feature_deltas.csv; SuppTable14_AlignedFeatureGenerationAudit.csv"}

# -----------------------------
# Supplementary Figure 6
# -----------------------------
def make_suppfig6(root: Path, out_dir: Path, formats: Sequence[str]) -> dict:
    metrics = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable15_CommonRoadNeutralMetrics.csv")
    deltas = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable16_CommonRoadNeutralDeltas.csv")
    tax = read_csv(root, "02_supplementary_table_results/supplementary_tables_v100/SuppTable17_CommonRoadNeutralFailureTaxonomy.csv")

    fig, axs = plt.subplots(2, 2, figsize=(8.2, 5.9), gridspec_kw={"wspace": 0.62, "hspace": 0.58})
    # Panel a taxonomy
    ax = axs[0, 0]
    # Robustly identify columns.
    if {"failure_reason", "count"}.issubset(tax.columns):
        labels = tax["failure_reason"].astype(str).tolist()
        counts = tax["count"].to_numpy()
    else:
        # table may be two columns without clear names
        labels = tax.iloc[:, 0].astype(str).tolist()
        counts = tax.iloc[:, 1].to_numpy()
    pretty = []
    for l in labels:
        l2 = l.replace("no_failure", "No failure").replace("collision_and_lane", "Known collision/lane").replace("unknown", "Unknown")
        l2 = l2.replace("_", " ")
        pretty.append(l2)
    colors = [COL["light"] if "No" in p else COL["blue"] if "Known" in p else COL["orange"] for p in pretty]
    ax.bar(np.arange(len(pretty)), counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts)*0.02, f"{int(c)}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(pretty)))
    ax.set_xticklabels(pretty, rotation=20, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Neutral CommonRoad failure taxonomy", loc="left", pad=6)
    clean_axis(ax, grid="y")
    panel_label(ax, "a")

    # Panel b AUPRC known vs all
    ax = axs[0, 1]
    score_order = ["distance_inverse", "TTC_inverse", "ROF_v2_composite", "ROF_v2_no_asr_composite", "temporal_composite"]
    au = metrics[(metrics["metric"].eq("auprc")) & (metrics["score"].isin(score_order))].copy()
    x = np.arange(len(score_order))
    width = 0.36
    for j, task in enumerate(["planner_failure_known", "planner_failure_all"]):
        sub = au[au["task"].eq(task)].set_index("score").reindex(score_order)
        vals = sub["point"].to_numpy()
        lo = vals - sub["ci_low"].to_numpy()
        hi = sub["ci_high"].to_numpy() - vals
        xpos = x + (-0.5 if j == 0 else 0.5)*width
        ax.bar(xpos, vals, width=width, color=COL["blue"] if j==0 else COL["teal"], alpha=0.86,
               label="Known failures" if j==0 else "All failures")
        ax.errorbar(xpos, vals, yerr=[lo, hi], fmt="none", ecolor=COL["black"], lw=0.7, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_DISPLAY[s] for s in score_order], rotation=35, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_title("Known-failure primary task and all-failure sensitivity", loc="left", pad=6)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax, grid="y")
    panel_label(ax, "b")

    def delta_panel(ax: plt.Axes, task: str, title: str):
        d = deltas[(deltas["task"].eq(task)) & (deltas["metric"].eq("auprc")) &
                   (deltas["enhanced_score"].isin(["ROF_v2_composite", "ROF_v2_no_asr_composite", "temporal_composite"])) &
                   (deltas["baseline_score"].isin(["distance_inverse", "TTC_inverse"]))].copy()
        d["label"] = d["enhanced_score"].map(SCORE_DISPLAY) + " vs " + d["baseline_score"].map(SCORE_DISPLAY)
        # order for readability
        order_labels = []
        for base in ["distance_inverse", "TTC_inverse"]:
            for enh in ["ROF_v2_composite", "ROF_v2_no_asr_composite", "temporal_composite"]:
                lab = SCORE_DISPLAY[enh] + " vs " + SCORE_DISPLAY[base]
                order_labels.append(lab)
        d["order"] = d["label"].map({lab:i for i,lab in enumerate(order_labels)})
        d = d.sort_values("order")
        y = np.arange(len(d))[::-1]
        for yy, (_, row) in zip(y, d.iterrows()):
            col = SCORE_COLORS.get(row["enhanced_score"], COL["blue"])
            ax.hlines(yy, row["ci_low"], row["ci_high"], color=col, lw=1.0)
            # open marker if CI crosses zero
            face = "white" if row["ci_low"] <= 0 <= row["ci_high"] else col
            ax.scatter(row["delta"], yy, s=32, color=face, edgecolor=col, linewidth=0.9, zorder=3)
        add_zero_line(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(d["label"].tolist())
        ax.set_xlabel("ΔAUPRC")
        ax.set_title(title, loc="left", pad=6)
        clean_axis(ax, grid="x")

    delta_panel(axs[1, 0], "planner_failure_known", "Known-failure ΔAUPRC")
    panel_label(axs[1, 0], "c")
    delta_panel(axs[1, 1], "planner_failure_all", "All-failure sensitivity ΔAUPRC")
    panel_label(axs[1, 1], "d")

    fig.suptitle("Supplementary Fig. 6 | Neutral CommonRoad validation details", x=0.01, ha="left")
    save_figure(fig, out_dir, "Supplementary_Figure_6_commonroad_neutral_details", formats)
    return {"figure": "Supplementary_Figure_6", "files": "SuppTable15_CommonRoadNeutralMetrics.csv; SuppTable16_CommonRoadNeutralDeltas.csv; SuppTable17_CommonRoadNeutralFailureTaxonomy.csv"}

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Supplementary Figures S1-S5 retained in the v1.1 manuscript from the v100 evidence-lock source data.")
    parser.add_argument("--data", required=True, help="Path to ROF_results_v100_evidence_lock.zip or extracted ROF_results_v100_evidence_lock directory.")
    parser.add_argument("--out", default="supplementary_figures_v100", help="Output directory.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats, e.g. pdf png svg.")
    parser.add_argument("--keep-extracted", action="store_true", help="Keep extracted data directory if --data is a zip.")
    parser.add_argument("--include-legacy-commonroad", action="store_true", help="Also render the superseded v100 CommonRoad pilot figure for provenance only.")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = resolve_data_root(args.data, out_dir)

    manifest = []
    manifest.append(make_suppfig1(root, out_dir, args.formats))
    manifest.append(make_suppfig2(root, out_dir, args.formats))
    manifest.append(make_suppfig3(root, out_dir, args.formats))
    manifest.append(make_suppfig4(root, out_dir, args.formats))
    manifest.append(make_suppfig5(root, out_dir, args.formats))
    if args.include_legacy_commonroad:
        manifest.append(make_suppfig6(root, out_dir, args.formats))
    write_manifest(out_dir, manifest)

    # Write a small LaTeX snippet file with placeholders.
    snippet = out_dir / "supplementary_figure_include_snippets.tex"
    with snippet.open("w", encoding="utf-8") as f:
        stems = [
            "Supplementary_Figure_1_proximity_ttc_diagnostics",
            "Supplementary_Figure_2_oof_calibration_diagnostics",
            "Supplementary_Figure_3_full_feature_audit",
            "Supplementary_Figure_4_cv_fallback_sensitivity",
            "Supplementary_Figure_5_endpoint_design_aligned_robustness",
        ]
        if args.include_legacy_commonroad:
            stems.append("Supplementary_Figure_6_commonroad_neutral_details")
        for i, stem in enumerate(stems, start=1):
            f.write(textwrap.dedent(f"""
            % Supplementary Figure {i}: replace caption text with the finalized manuscript caption.
            \\begin{{figure}}[p]
              \\centering
              \\includegraphics[width=0.98\\linewidth]{{figures/supplementary_v100/{stem}.pdf}}
              \\caption{{\\textbf{{Supplementary Figure {i}.}} Caption placeholder. Source data are listed in supplementary_figure_source_manifest.csv.}}
              \\label{{fig:suppfig{i}}}
            \\end{{figure}}
            """))

    if (root.parent == out_dir / "_extracted_ROF_results_v100_evidence_lock") and not args.keep_extracted:
        # In practice root is out/_extracted/.../ROF_results..., so do not delete if user asked to keep.
        pass

    print(f"Generated supplementary figures in: {out_dir}")
    print(f"Source root: {root}")
    print("Files:")
    for p in sorted(out_dir.glob("Supplementary_Figure_*.*")):
        print("  ", p.name)

if __name__ == "__main__":
    main()
