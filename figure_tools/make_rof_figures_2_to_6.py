#!/usr/bin/env python3
"""
Generate Nature Communications-style main figures 2–6 from
ROF_results_v100_evidence_lock.zip / ROF_results_v100_evidence_lock/.

Usage
-----
python make_rof_figures_2_to_6.py \
    --input ROF_results_v100_evidence_lock.zip \
    --out figures_nc

Outputs PDF, SVG and PNG by default:
    Figure2_endpoint_proximity.{pdf,svg,png}
    Figure3_waymo_primary_oof.{pdf,svg,png}
    Figure4_feature_mechanism_audit.{pdf,svg,png}
    Figure5_endpoint_design_robustness.{pdf,svg,png}
    Figure6_commonroad_neutral_validation.{pdf,svg,png}

Dependencies: pandas, numpy, matplotlib. No seaborn required.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


# -----------------------------------------------------------------------------
# Global style: conservative, vector-friendly, colorblind-aware.
# -----------------------------------------------------------------------------

MM = 1 / 25.4

COL = {
    "black": "#222222",
    "dark": "#3A3A3A",
    "mid": "#737373",
    "light": "#C7C7C7",
    "very_light": "#EFEFEF",
    "grid": "#E1E1E1",
    "baseline": "#5F6368",
    "temporal": "#0072B2",  # Okabe-Ito blue
    "rof": "#0072B2",
    "ratio_excluded": "#009E73",  # Okabe-Ito green
    "direct_ratio": "#E69F00",  # Okabe-Ito orange
    "spatial": "#9E9E9E",
    "distance": "#4D4D4D",
    "ttc": "#7A7A7A",
    "known": "#0072B2",
    "unknown": "#D55E00",
    "nofail": "#D9D9D9",
    "accent": "#CC79A7",
}

ACTION_COLORS = {
    "high_actionability": "#56B4E9",
    "reduced_actionability": "#E69F00",
    "critical_actionability": "#D55E00",
    "candidate_set_infeasible": "#CC79A7",
    "High": "#56B4E9",
    "Reduced": "#E69F00",
    "Critical": "#D55E00",
    "Candidate-set infeasible": "#CC79A7",
}

SEED_MARKERS = {41: "o", 42: "s", 43: "^"}
SEED_OFFSETS = {41: -0.16, 42: 0.0, 43: 0.16}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def clean_ax(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["black"])
    ax.spines["bottom"].set_color(COL["black"])
    ax.tick_params(colors=COL["black"], labelcolor=COL["black"])
    if grid == "y":
        ax.grid(axis="y", color=COL["grid"], linewidth=0.45, zorder=0)
    elif grid == "x":
        ax.grid(axis="x", color=COL["grid"], linewidth=0.45, zorder=0)
    elif grid == "both":
        ax.grid(color=COL["grid"], linewidth=0.45, zorder=0)
    else:
        ax.grid(False)


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=COL["black"],
    )


def pct(x: float, digits: int = 0) -> str:
    if pd.isna(x):
        return ""
    return f"{100 * float(x):.{digits}f}%"


def fmt_delta(x: float, digits: int = 3) -> str:
    return f"{float(x):+.{digits}f}"


def pretty_score(score: str) -> str:
    mapping = {
        "strong_baseline_cv": "Baseline",
        "strong_baseline_cv_plus_strict_temporal_dynamics": "+ strict temporal",
        "strong_baseline_cv_plus_direct_action_ratios_only": "Direct action ratios",
        "strong_baseline_cv_plus_explicit_ratio_field_excluded_current": "Current context\n(no explicit ratios)",
        "strong_baseline_cv_plus_strict_spatial_no_action": "Spatial no-action",
        "distance_inverse": "Distance",
        "TTC_inverse": "TTC",
        "ROF_v2_composite": "ROF-v2",
        "ROF_v2_no_asr_composite": "ROF-v2 no-ASR",
        "temporal_composite": "Temporal",
    }
    return mapping.get(score, score.replace("_", " "))


def pretty_variant(variant: str) -> str:
    mapping = {
        "reference_h3_b3_base7_skip": "Reference\nH3/B3",
        "horizon_h2_b3_base7_skip": "Horizon\n2 s",
        "horizon_h4_b3_base7_skip": "Horizon\n4 s",
        "buffer_h3_b2_base7_skip": "Lane buffer\n2 m",
        "buffer_h3_b4_base7_skip": "Lane buffer\n4 m",
        "action_h3_b3_extended_skip": "Extended\naction set",
        "future_h3_b3_base7_cvfallback": "CV fallback",
    }
    return mapping.get(variant, variant.replace("_", "\n"))


def variant_order() -> list[str]:
    return [
        "reference_h3_b3_base7_skip",
        "horizon_h2_b3_base7_skip",
        "horizon_h4_b3_base7_skip",
        "buffer_h3_b2_base7_skip",
        "buffer_h3_b4_base7_skip",
        "action_h3_b3_extended_skip",
        "future_h3_b3_base7_cvfallback",
    ]


def feature_comparison_label(comparison: str) -> str:
    if "strict_temporal_dynamics" in comparison:
        return "Strict temporal\ndynamics"
    if "explicit_ratio_field_excluded_current" in comparison:
        return "Current context\n(no explicit ratios)"
    if "direct_action_ratios_only" in comparison:
        return "Direct action\nratios only"
    if "strict_spatial_no_action" in comparison:
        return "Spatial\nno-action"
    return comparison.replace("strong_baseline_cv__vs__", "").replace("_", "\n")


def feature_comparison_color(comparison: str) -> str:
    if "strict_temporal_dynamics" in comparison:
        return COL["temporal"]
    if "explicit_ratio_field_excluded_current" in comparison:
        return COL["ratio_excluded"]
    if "direct_action_ratios_only" in comparison:
        return COL["direct_ratio"]
    if "strict_spatial_no_action" in comparison:
        return COL["spatial"]
    return COL["mid"]


def read_csv(base: Path, rel: str) -> pd.DataFrame:
    path = base / rel
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required source table: {path}")
    return pd.read_csv(path)


def resolve_source_dir(input_path: Path, tmp: tempfile.TemporaryDirectory[str] | None) -> Path:
    """Return path to 03_main_figure_source_data."""
    p = input_path.expanduser().resolve()
    if p.is_file() and p.suffix.lower() == ".zip":
        if tmp is None:
            raise RuntimeError("Temporary directory required for zip input")
        with zipfile.ZipFile(p) as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
    elif p.is_dir():
        root = p
    else:
        raise FileNotFoundError(f"Input path does not exist or is not a zip/directory: {p}")

    candidates = []
    if (root / "03_main_figure_source_data").exists():
        candidates.append(root / "03_main_figure_source_data")
    for q in root.glob("**/03_main_figure_source_data"):
        candidates.append(q)
    if not candidates:
        raise FileNotFoundError("Could not locate 03_main_figure_source_data under input path")
    return candidates[0]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        ext = ext.lower().lstrip(".")
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


def lighten_cmap(color: str, name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", color])


# -----------------------------------------------------------------------------
# Figure 2: endpoint/proximity diagnostic.
# -----------------------------------------------------------------------------



def plot_stacked_label_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    df = df[df["include_in_main_plot"].astype(bool)].copy()
    df = df.sort_values(["label_variant", "display_order"])
    variants = ["map_constrained", "no_map"]
    y_positions = np.arange(len(variants))[::-1]
    left = np.zeros(len(variants))

    for label_name, sub in df.groupby("display_label", sort=False):
        vals = []
        counts = []
        colors = []
        for v in variants:
            row = sub[sub["label_variant"] == v]
            vals.append(float(row["fraction"].iloc[0]) if not row.empty else 0.0)
            counts.append(int(row["count"].iloc[0]) if not row.empty else 0)
            colors.append(ACTION_COLORS.get(str(row["color_group"].iloc[0]), COL["mid"]) if not row.empty else COL["mid"])
        color = colors[0]
        ax.barh(y_positions, vals, left=left, height=0.48, color=color, edgecolor="white", linewidth=0.7)
        for yi, val, lft, n in zip(y_positions, vals, left, counts):
            # Label the dominant and medium-sized segments; omit tiny terminal labels to keep the panel clean.
            if val >= 0.05:
                ax.text(lft + val / 2, yi, f"{100*val:.1f}%", ha="center", va="center", fontsize=6.0, color="white" if val < 0.12 else COL["black"])
            elif val >= 0.015:
                ax.text(min(lft + val + 0.006, 1.025), yi, f"{100*val:.1f}%", ha="left", va="center", fontsize=5.8, color=COL["black"])
        left += np.array(vals)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(["Map-constrained", "No-map"])
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Fraction of Waymo samples")
    ax.set_title("Actionability label distribution", loc="left", pad=4)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    handles = [
        Rectangle((0, 0), 1, 1, color=ACTION_COLORS[k])
        for k in ["High", "Reduced", "Critical", "Candidate-set infeasible"]
    ]
    ax.legend(
        handles,
        ["High", "Reduced", "Critical", "Infeasible"],
        ncol=2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.47),
        handlelength=1.0,
        columnspacing=1.0,
    )

def plot_actionability_heatmap(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    df = df[df["include_in_main_plot"].astype(bool)].copy()
    row_order = ["safe", "caution", "warning", "emergency"]
    col_order = ["High", "Reduced", "Critical", "Candidate-set infeasible"]
    pivot = (
        df.assign(display_label=df["display_label"].replace({"Candidate-set infeasible": "Candidate-set infeasible"}))
        .pivot_table(index="original_label_name", columns="display_label", values="row_fraction", aggfunc="sum")
        .reindex(index=row_order, columns=col_order)
    )
    im = ax.imshow(pivot.values, cmap=lighten_cmap(COL["temporal"], "blue_white"), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(["High", "Reduced", "Critical", "Infeasible"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([r.capitalize() for r in row_order])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v > 0.55 else COL["black"]
            ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=6.1, color=color)
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Actionability label")
    # Row labels make the proximity axis explicit; omit a vertical y-label to reduce crowding.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Row %", labelpad=2)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    cbar.outline.set_linewidth(0.4)


def plot_feasible_ratio(ax: plt.Axes, df: pd.DataFrame) -> None:
    df = df[(df["include_in_main_plot"].astype(bool)) & (df["label_variant"] == "map_constrained")].copy()
    labels = ["High", "Reduced", "Critical", "Candidate-set infeasible"]
    ratio_order = ["comfort_feasible_ratio", "emergency_feasible_ratio"]
    ratio_labels = {"comfort_feasible_ratio": "Comfort", "emergency_feasible_ratio": "Emergency"}
    ratio_colors = {"comfort_feasible_ratio": COL["temporal"], "emergency_feasible_ratio": COL["direct_ratio"]}
    x = np.arange(len(labels))
    offsets = {"comfort_feasible_ratio": -0.14, "emergency_feasible_ratio": 0.14}
    for ratio in ratio_order:
        sub = df[df["ratio_type"] == ratio].set_index("display_label").reindex(labels)
        med = sub["median"].values.astype(float)
        lo = sub["p25"].values.astype(float)
        hi = sub["p75"].values.astype(float)
        ax.errorbar(
            x + offsets[ratio],
            med,
            yerr=[med - lo, hi - med],
            fmt="o",
            markersize=3.8,
            color=ratio_colors[ratio],
            ecolor=ratio_colors[ratio],
            elinewidth=1.0,
            capsize=2.0,
            label=ratio_labels[ratio],
            zorder=3,
        )
        # faint mean markers
        ax.scatter(x + offsets[ratio], sub["mean"].values.astype(float), s=9, marker="_", color=ratio_colors[ratio], zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["High", "Reduced", "Critical", "Infeasible"], rotation=25, ha="right")
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Feasible-action ratio")
    ax.set_title("Feasible actions by label", loc="left", pad=4)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    clean_ax(ax, grid="y")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4)
    ax.text(0.02, 0.02, "dot = median, line = IQR; tick = mean", transform=ax.transAxes, fontsize=5.8, color=COL["mid"], ha="left", va="bottom")



def plot_ttc_sensitivity(ax: plt.Axes, df: pd.DataFrame) -> None:
    variable_order = [
        "current_min_distance_m",
        "valid_ttc_only",
        "no_ttc_as_category",
        "capped_ttc_prespecified_10s",
        "inverse_ttc_prespecified",
        "legacy_sentinel",
    ]
    var_labels = {
        "current_min_distance_m": "Distance",
        "valid_ttc_only": "TTC\nvalid",
        "no_ttc_as_category": "TTC\nmissing",
        "capped_ttc_prespecified_10s": "TTC\ncap 10 s",
        "inverse_ttc_prespecified": "1/TTC",
        "legacy_sentinel": "Legacy\nsentinel",
    }
    label_map = {
        "map_actionability_label_id": "Map-constrained",
        "nomap_actionability_label_id": "No-map",
    }
    colors = {"map_actionability_label_id": COL["temporal"], "nomap_actionability_label_id": COL["mid"]}
    x = np.arange(len(variable_order))
    offsets = {"map_actionability_label_id": -0.12, "nomap_actionability_label_id": 0.12}
    for label in ["map_actionability_label_id", "nomap_actionability_label_id"]:
        sub = df[df["label"] == label].set_index("variable").reindex(variable_order)
        point = np.abs(sub["spearman"].values.astype(float))
        lo = np.minimum(np.abs(sub["ci_low"].values.astype(float)), np.abs(sub["ci_high"].values.astype(float)))
        hi = np.maximum(np.abs(sub["ci_low"].values.astype(float)), np.abs(sub["ci_high"].values.astype(float)))
        ax.errorbar(
            x + offsets[label],
            point,
            yerr=[point - lo, hi - point],
            fmt="o",
            markersize=3.4,
            color=colors[label],
            ecolor=colors[label],
            elinewidth=0.9,
            capsize=1.8,
            label=label_map[label],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([var_labels[v] for v in variable_order], rotation=25, ha="right")
    ax.set_ylabel("|Spearman ρ| with actionability")
    ax.set_ylim(0, 0.62)
    ax.set_title("Proximity/TTC sensitivity", loc="left", pad=4)
    clean_ax(ax, grid="y")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4)

def make_figure2(base: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    fig = plt.figure(figsize=(180 * MM, 122 * MM))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.03, 1.05], height_ratios=[1, 1], wspace=0.34, hspace=0.58)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_stacked_label_distribution(ax_a, read_csv(base, "Figure2_endpoint_proximity/Figure2A_label_distribution.csv"))
    plot_actionability_heatmap(ax_b, read_csv(base, "Figure2_endpoint_proximity/Figure2B_proximity_vs_actionability_map.csv"), "Proximity vs map-constrained actionability")
    plot_feasible_ratio(ax_c, read_csv(base, "Figure2_endpoint_proximity/Figure2F_feasible_ratio_by_label.csv"))
    plot_ttc_sensitivity(ax_d, read_csv(base, "Figure2_endpoint_proximity/Figure2E_ttc_sensitivity.csv"))

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        panel_label(ax, lab)
    save_figure(fig, out_dir, "Figure2_endpoint_proximity", formats, dpi)


# -----------------------------------------------------------------------------
# Figure 3: primary Waymo OOF result.
# -----------------------------------------------------------------------------


def plot_fig3a(ax: plt.Axes, df: pd.DataFrame) -> None:
    df = df.copy()
    seeds = sorted(df["seed"].unique())
    feature_order = ["strong_baseline_cv", "strong_baseline_cv_plus_strict_temporal_dynamics"]
    colors = {feature_order[0]: COL["baseline"], feature_order[1]: COL["temporal"]}
    labels = {feature_order[0]: "Baseline", feature_order[1]: "+ strict temporal"}
    x = np.arange(len(seeds))
    offsets = {feature_order[0]: -0.16, feature_order[1]: 0.16}
    width = 0.26
    for seed_i, seed in enumerate(seeds):
        vals = []
        for fs in feature_order:
            row = df[(df["seed"] == seed) & (df["feature_set"] == fs)]
            vals.append(float(row["auprc"].iloc[0]))
        ax.plot([x[seed_i] + offsets[feature_order[0]], x[seed_i] + offsets[feature_order[1]]], vals, color=COL["light"], lw=0.9, zorder=1)
    for fs in feature_order:
        sub = df[df["feature_set"] == fs].set_index("seed").reindex(seeds)
        ax.bar(x + offsets[fs], sub["auprc"].values, width=width, color=colors[fs], alpha=0.92, edgecolor="white", linewidth=0.6, label=labels[fs], zorder=2)
        for xi, val in zip(x + offsets[fs], sub["auprc"].values):
            ax.text(xi, val + 0.010, f"{val:.3f}", ha="center", va="bottom", fontsize=5.8, rotation=0, color=COL["black"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, max(df["auprc"]) + 0.075)
    ax.set_title("Primary OOF AUPRC", loc="left", pad=4)
    clean_ax(ax, grid="y")
    ax.legend(frameon=False, loc="lower left", handlelength=1.0)


def plot_fig3b(ax: plt.Axes, df: pd.DataFrame) -> None:
    df = df[df["metric"] == "auprc"].copy().sort_values("seed")
    y = np.arange(len(df))[::-1]
    ax.axvline(0, color=COL["mid"], lw=0.8, zorder=1)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.errorbar(row["delta"], yi, xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]], fmt="o", color=COL["temporal"], ecolor=COL["temporal"], markersize=4.0, elinewidth=1.1, capsize=2.2, zorder=3)
        ax.text(row["ci_high"] + 0.004, yi, f"{row['delta']:+.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]", va="center", ha="left", fontsize=5.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Seed {int(s)}" for s in df["seed"]])
    ax.set_xlabel("ΔAUPRC vs baseline")
    ax.set_title("Paired bootstrap delta", loc="left", pad=4)
    ax.set_xlim(min(-0.005, df["ci_low"].min() - 0.008), df["ci_high"].max() + 0.05)
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.02, 0.08, "All 95% CIs > 0", transform=ax.transAxes, fontsize=6.4, color=COL["temporal"], ha="left", va="bottom")


def plot_fig3c(ax: plt.Axes, df_fold: pd.DataFrame) -> None:
    df = df_fold.copy()
    df["feature_label"] = df["feature_set"].map(pretty_score)
    feature_order = ["Baseline", "+ strict temporal"]
    metric_cols = ["outer_test_recall", "outer_test_achieved_fpr"]
    metric_labels = ["Recall", "Achieved FPR"]
    xbase = np.array([0, 1.1])
    offsets = {"Baseline": -0.14, "+ strict temporal": 0.14}
    colors = {"Baseline": COL["baseline"], "+ strict temporal": COL["temporal"]}

    rng = np.random.default_rng(123)
    for mi, metric in enumerate(metric_cols):
        for flab in feature_order:
            sub = df[df["feature_label"] == flab]
            vals = sub[metric].values.astype(float)
            # deterministic jitter for fold/seed points
            jitter = rng.normal(0, 0.018, size=len(vals))
            ax.scatter(np.full(len(vals), xbase[mi] + offsets[flab]) + jitter, vals, s=8, color=colors[flab], alpha=0.32, linewidth=0, zorder=2)
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.errorbar(xbase[mi] + offsets[flab], mean, yerr=1.96 * sem, fmt="o", color=colors[flab], ecolor=colors[flab], markersize=4.4, elinewidth=1.2, capsize=2.2, zorder=4)
            ax.text(xbase[mi] + offsets[flab], mean + (0.035 if metric == "outer_test_recall" else 0.025), f"{100*mean:.1f}%", ha="center", va="bottom", fontsize=5.8, color=COL["black"])
    ax.axhline(0.05, color=COL["mid"], lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(1.74, 0.052, "nominal 5% FPR", ha="right", va="bottom", fontsize=5.8, color=COL["mid"])
    ax.set_xticks(xbase)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Operating point")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.58)
    ax.set_title("Fold-calibrated 5% FPR operating points", loc="left", pad=4)
    clean_ax(ax, grid="y")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["baseline"], markersize=4.5), Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["temporal"], markersize=4.5)]
    ax.legend(handles, feature_order, frameon=False, loc="upper right", ncol=1, handletextpad=0.35, borderpad=0.2)


def make_figure3(base: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    fig = plt.figure(figsize=(180 * MM, 72 * MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.05, 1.28], wspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    plot_fig3a(ax_a, read_csv(base, "Figure3_waymo_primary_oof/Figure3A_primary_oof_metrics.csv"))
    plot_fig3b(ax_b, read_csv(base, "Figure3_waymo_primary_oof/Figure3B_primary_paired_deltas.csv"))
    plot_fig3c(ax_c, read_csv(base, "Figure3_waymo_primary_oof/Figure3C_calibrated_operating_points.csv"))
    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        panel_label(ax, lab)
    save_figure(fig, out_dir, "Figure3_waymo_primary_oof", formats, dpi)


# -----------------------------------------------------------------------------
# Figure 4: feature mechanism audit.
# -----------------------------------------------------------------------------



def plot_feature_lineage_counts(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Compact lineage audit: show what is primary-allowed and what carries explicit ratio fields.
    group_counts = df.groupby("feature_group_v090").agg(
        n=("feature_name", "count"),
        explicit=("uses_explicit_asr_field", "sum"),
        transformed=("uses_transformed_or_composite_asr", "sum"),
        primary=("allowed_in_primary_predictor", "sum"),
    ).reset_index()
    order = [
        "strong_baseline_cv",
        "strict_temporal_dynamics;explicit_ratio_field_excluded_current",
        "explicit_ratio_field_excluded_current",
        "direct_action_ratios_only",
        "strict_spatial_no_action",
    ]
    labels = {
        "strong_baseline_cv": "Baseline current",
        "strict_temporal_dynamics;explicit_ratio_field_excluded_current": "Strict temporal\n(primary)",
        "explicit_ratio_field_excluded_current": "Current context\n(no explicit ratios)",
        "direct_action_ratios_only": "Direct action\nratios",
        "strict_spatial_no_action": "Spatial\nno-action",
    }
    colors = {
        "strong_baseline_cv": COL["baseline"],
        "strict_temporal_dynamics;explicit_ratio_field_excluded_current": COL["temporal"],
        "explicit_ratio_field_excluded_current": COL["ratio_excluded"],
        "direct_action_ratios_only": COL["direct_ratio"],
        "strict_spatial_no_action": COL["spatial"],
    }
    sub = group_counts.set_index("feature_group_v090").reindex(order).reset_index()
    y = np.arange(len(order))[::-1]
    ax.barh(y, sub["n"], color=[colors[g] for g in order], height=0.58, edgecolor="white", linewidth=0.7, zorder=2)
    for yi, (_, row), g in zip(y, sub.iterrows(), order):
        ax.text(row["n"] + 0.35, yi, f"{int(row['n'])}", ha="left", va="center", fontsize=6.2, fontweight="bold")
        tag = None
        if row["explicit"] > 0:
            tag = "explicit ratio fields"
        elif row["primary"] > 0:
            tag = "primary-allowed"
        if tag and row["n"] >= 7:
            ax.text(max(row["n"] * 0.50, 0.8), yi, tag, ha="center", va="center", fontsize=5.6, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[g] for g in order])
    ax.set_xlabel("Number of features")
    ax.set_title("Prespecified feature groups and lineage", loc="left", pad=4)
    ax.set_xlim(0, max(sub["n"]) * 1.28)
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

def plot_feature_delta_forest(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Main-text mechanism: strict temporal is prespecified; direct ratios/context are shown as audit controls.
    keep = [
        "strong_baseline_cv__vs__strong_baseline_cv_plus_strict_temporal_dynamics",
        "strong_baseline_cv__vs__strong_baseline_cv_plus_explicit_ratio_field_excluded_current",
        "strong_baseline_cv__vs__strong_baseline_cv_plus_direct_action_ratios_only",
    ]
    sub = df[(df["metric"] == "auprc") & (df["comparison"].isin(keep))].copy()
    order = keep
    y_base = np.arange(len(order))[::-1]
    comp_to_y = {comp: y for comp, y in zip(order, y_base)}
    ax.axvline(0, color=COL["mid"], lw=0.8, zorder=1)
    for comp in order:
        rows = sub[sub["comparison"] == comp].sort_values("seed")
        c = feature_comparison_color(comp)
        for _, row in rows.iterrows():
            seed = int(row["seed"])
            y = comp_to_y[comp] + SEED_OFFSETS[seed] * 0.72
            ax.errorbar(
                row["delta"], y,
                xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]],
                fmt=SEED_MARKERS.get(seed, "o"),
                color=c,
                ecolor=c,
                markersize=3.7,
                elinewidth=1.0,
                capsize=1.8,
                zorder=3,
            )
        mean_delta = rows["delta"].mean()
        ax.text(rows["ci_high"].max() + 0.004, comp_to_y[comp], f"mean {mean_delta:+.3f}", va="center", ha="left", fontsize=5.9, color=COL["black"])
    ax.set_yticks(y_base)
    ax.set_yticklabels([feature_comparison_label(c) for c in order])
    ax.set_xlabel("ΔAUPRC vs strong baseline")
    ax.set_title("Primary endpoint mechanism audit", loc="left", pad=4)
    ax.set_xlim(-0.005, max(sub["ci_high"].max() + 0.04, 0.11))
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    seed_handles = [Line2D([0], [0], marker=SEED_MARKERS[s], color=COL["black"], linestyle="None", markersize=3.6, label=f"Seed {s}") for s in sorted(SEED_MARKERS)]
    ax.legend(handles=seed_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.54, -0.16), ncol=3, handletextpad=0.25, columnspacing=0.65)


def make_figure4(base: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    fig = plt.figure(figsize=(180 * MM, 76 * MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.25], wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    plot_feature_lineage_counts(ax_a, read_csv(base, "Figure4_feature_mechanism_audit/Figure4C_feature_lineage_flags.csv"))
    plot_feature_delta_forest(ax_b, read_csv(base, "Figure4_feature_mechanism_audit/Figure4B_primary_feature_audit_bootstrap.csv"))
    for ax, lab in zip([ax_a, ax_b], "ab"):
        panel_label(ax, lab)
    save_figure(fig, out_dir, "Figure4_feature_mechanism_audit", formats, dpi)


# -----------------------------------------------------------------------------
# Figure 5: endpoint design robustness.
# -----------------------------------------------------------------------------


def plot_design_stability_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = variant_order()
    sub = df.set_index("variant_id").reindex(order).reset_index()
    metrics = [
        ("critical_or_worse_prevalence", "Severe\nprevalence", "%"),
        ("label_changed_fraction", "Label\nchanged", "%"),
        ("severe_set_jaccard_vs_reference", "Severe-set\nJaccard", "unit"),
    ]
    vals = sub[[m[0] for m in metrics]].values.astype(float)
    # Normalize each column separately; annotate raw values to avoid scale ambiguity.
    norm = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        mn, mx = np.nanmin(col), np.nanmax(col)
        norm[:, j] = 0 if np.isclose(mx, mn) else (col - mn) / (mx - mn)
    cmap = LinearSegmentedColormap.from_list("grayblue", ["#F7F7F7", "#DCEAF7", COL["temporal"]])
    ax.imshow(norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([pretty_variant(v) for v in order])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    for i in range(vals.shape[0]):
        for j, (_, _, unit) in enumerate(metrics):
            v = vals[i, j]
            if unit == "%":
                txt = f"{100*v:.1f}%" if v >= 0.01 else f"{100*v:.3f}%"
            else:
                txt = f"{v:.3f}"
            color = "white" if norm[i, j] > 0.62 else COL["black"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.8, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Endpoint-design variants change labels", loc="left", pad=4)


def plot_variant_delta_points(ax: plt.Axes, df: pd.DataFrame, title: str, has_ci: bool, xlim: tuple[float, float] | None = None) -> None:
    order = variant_order()
    sub = df[df["metric"] == "auprc"].copy()
    sub["variant_id"] = pd.Categorical(sub["variant_id"], categories=order, ordered=True)
    sub = sub.sort_values(["variant_id", "seed"])
    y_positions = {v: i for i, v in enumerate(order[::-1])}
    colors = {41: COL["baseline"], 42: COL["temporal"], 43: COL["direct_ratio"]}
    ax.axvline(0, color=COL["mid"], lw=0.8, zorder=1)
    for variant in order:
        rows = sub[sub["variant_id"] == variant]
        if rows.empty:
            continue
        y0 = y_positions[variant]
        for _, row in rows.iterrows():
            seed = int(row["seed"])
            y = y0 + SEED_OFFSETS.get(seed, 0) * 0.72
            if has_ci and pd.notna(row.get("ci_low", np.nan)):
                ax.errorbar(
                    row["delta"], y,
                    xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]],
                    fmt=SEED_MARKERS.get(seed, "o"),
                    color=colors.get(seed, COL["black"]),
                    ecolor=colors.get(seed, COL["black"]),
                    markersize=3.4,
                    elinewidth=0.9,
                    capsize=1.6,
                    zorder=3,
                )
            else:
                ax.scatter(row["delta"], y, marker=SEED_MARKERS.get(seed, "o"), color=colors.get(seed, COL["black"]), s=18, zorder=3)
        mean_delta = rows["delta"].mean()
        ax.scatter(mean_delta, y0, marker="D", s=22, color=COL["black"], zorder=4)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([pretty_variant(v) for v in order[::-1]])
    ax.set_xlabel("ΔAUPRC vs baseline")
    ax.set_title(title, loc="left", pad=4)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        maxv = float(sub["delta"].max())
        if has_ci and "ci_high" in sub.columns:
            maxv = max(maxv, float(sub["ci_high"].max()))
        ax.set_xlim(-0.01, maxv + 0.06)
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    seed_handles = [Line2D([0], [0], marker=SEED_MARKERS[s], color="none", markerfacecolor=colors[s], markeredgecolor=colors[s], linestyle="None", markersize=3.7, label=f"{s}") for s in sorted(SEED_MARKERS)]
    mean_handle = Line2D([0], [0], marker="D", color="none", markerfacecolor=COL["black"], markeredgecolor=COL["black"], linestyle="None", markersize=3.8, label="mean")
    ax.legend(handles=seed_handles + [mean_handle], frameon=False, loc="lower right", ncol=4, handletextpad=0.25, columnspacing=0.55, title="Seed")


def plot_cv_fallback(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Text stats from label_shift/imputation_summary and preserved primary ΔAUPRC from primary_deltas.
    label = df[df["section"] == "label_shift"].iloc[0]
    imp = df[df["section"] == "imputation_summary"].iloc[0]
    deltas = df[(df["section"] == "primary_deltas") & (df["endpoint"] == "map_critical_or_worse_cv_fallback") & (df["metric"] == "auprc")].copy().sort_values("seed")
    ax.axis("off")
    ax.set_title("CV-fallback robustness", loc="left", pad=4)
    stats = [
        ("Labels changed", f"{int(label['label_changed_count']):,} / {int(label['n_common']):,}"),
        ("Changed fraction", f"{100*label['label_changed_fraction']:.3f}%"),
        ("Severe-set Jaccard", f"{label['severe_set_jaccard']:.4f}"),
        ("Mean imputed slots", f"{100*imp['mean_sample_imputed_fraction']:.1f}%"),
    ]
    y0 = 0.90
    for i, (k, v) in enumerate(stats):
        y = y0 - i * 0.135
        ax.text(0.02, y, k, transform=ax.transAxes, ha="left", va="center", fontsize=6.2, color=COL["mid"])
        ax.text(0.58, y, v, transform=ax.transAxes, ha="left", va="center", fontsize=7.2, fontweight="bold", color=COL["black"])

    # Mini forest inset for ΔAUPRC.
    inset = ax.inset_axes([0.05, 0.06, 0.90, 0.30])
    y = np.arange(len(deltas))[::-1]
    inset.axvline(0, color=COL["mid"], lw=0.7)
    for yi, (_, row) in zip(y, deltas.iterrows()):
        inset.errorbar(row["delta"], yi, xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]], fmt=SEED_MARKERS.get(int(row["seed"]), "o"), markersize=3.2, color=COL["temporal"], ecolor=COL["temporal"], elinewidth=0.8, capsize=1.5)
    inset.set_yticks(y)
    inset.set_yticklabels([f"{int(s)}" for s in deltas["seed"]])
    inset.set_xlabel("CV-fallback ΔAUPRC", labelpad=1)
    inset.set_xlim(0.04, max(0.09, deltas["ci_high"].max() + 0.006))
    clean_ax(inset, grid="x")
    inset.spines["left"].set_visible(False)
    inset.tick_params(axis="y", length=0)
    inset.tick_params(labelsize=5.8)


def make_figure5(base: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    fig = plt.figure(figsize=(180 * MM, 132 * MM))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.15], height_ratios=[1.05, 1.0], wspace=0.40, hspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    plot_design_stability_heatmap(ax_a, read_csv(base, "Figure5_endpoint_design_robustness/Figure5A_label_design_stability.csv"))
    plot_variant_delta_points(ax_b, read_csv(base, "Figure5_endpoint_design_robustness/Figure5B_reference_feature_label_variant_deltas.csv"), "Full relabeling with reference features", has_ci=False, xlim=(-0.005, 0.18))
    plot_variant_delta_points(ax_c, read_csv(base, "Figure5_endpoint_design_robustness/Figure5C_aligned_feature_deltas.csv"), "Aligned label + feature sensitivity", has_ci=True, xlim=(-0.01, 0.47))
    plot_cv_fallback(ax_d, read_csv(base, "Figure5_endpoint_design_robustness/Figure5D_cv_fallback_summary.csv"))
    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        panel_label(ax, lab)
    fig.text(0.5, 0.005, "Label assignments are design-dependent, but the strict-temporal gain direction persists across variants.", ha="center", va="bottom", fontsize=7.0, color=COL["dark"])
    save_figure(fig, out_dir, "Figure5_endpoint_design_robustness", formats, dpi)


# -----------------------------------------------------------------------------
# Figure 6: CommonRoad neutral validation.
# -----------------------------------------------------------------------------


def plot_cohort_funnel(ax: plt.Axes, df: pd.DataFrame) -> None:
    metrics = [
        ("neutral_samples", "Neutral\nsamples"),
        ("unique_commonroad_scenarios", "Unique\nscenarios"),
        ("all_planner_failures", "All planner\nfailures"),
        ("known_collision_lane_failures", "Known collision/\nlane failures"),
    ]
    val_map = dict(zip(df["metric"], df["value"]))
    vals = [int(val_map[m]) for m, _ in metrics]
    labels = [lab for _, lab in metrics]
    y = np.arange(len(vals))[::-1]
    colors = [COL["baseline"], COL["mid"], COL["unknown"], COL["temporal"]]
    ax.barh(y, vals, color=colors, height=0.52, edgecolor="white", linewidth=0.7, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, f"{v:,}", ha="left", va="center", fontsize=6.8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title("Neutral validation cohort", loc="left", pad=4)
    ax.set_xlim(0, max(vals) * 1.20)
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)



def plot_failure_taxonomy(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = ["no_failure", "unknown", "collision_and_lane"]
    labels = {"no_failure": "No failure", "unknown": "Unknown", "collision_and_lane": "Known\ncollision/lane"}
    colors = {"no_failure": COL["nofail"], "unknown": COL["unknown"], "collision_and_lane": COL["known"]}
    sub = df.set_index("failure_reason").reindex(order).reset_index()
    y = np.arange(len(order))[::-1]
    ax.barh(y, sub["count"].values, color=[colors[o] for o in order], height=0.52, edgecolor="white", linewidth=0.7, zorder=2)
    for yi, (_, row), reason in zip(y, sub.iterrows(), order):
        ax.text(row["count"] + 22, yi, f"{int(row['count'])} ({100*row['fraction']:.1f}%)", ha="left", va="center", fontsize=6.2, fontweight="bold" if reason != "no_failure" else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[o] for o in order])
    ax.set_xlabel("Count")
    ax.set_title("Failure taxonomy", loc="left", pad=4)
    ax.set_xlim(0, max(sub["count"]) * 1.28)
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

def plot_commonroad_known_auprc(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = ["distance_inverse", "TTC_inverse", "temporal_composite", "ROF_v2_composite"]
    sub = df[(df["metric"] == "auprc") & (df["score"].isin(order))].copy().set_index("score").reindex(order).reset_index()
    x = np.arange(len(order))
    colors = [COL["distance"], COL["ttc"], COL["temporal"], COL["rof"]]
    for xi, (_, row), c in zip(x, sub.iterrows(), colors):
        ax.errorbar(xi, row["point"], yerr=[[row["point"] - row["ci_low"]], [row["ci_high"] - row["point"]]], fmt="o", color=c, ecolor=c, markersize=4.2, elinewidth=1.2, capsize=2.2, zorder=3)
        ax.text(xi, row["ci_high"] + 0.018, f"{row['point']:.3f}", ha="center", va="bottom", fontsize=5.9)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_score(s) for s in order], rotation=25, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, max(sub["ci_high"]) + 0.12)
    ax.set_title("Known-failure performance (22/963)", loc="left", pad=4)
    clean_ax(ax, grid="y")


def plot_commonroad_known_deltas(ax: plt.Axes, df: pd.DataFrame) -> None:
    keep_enhanced = ["ROF_v2_composite", "temporal_composite"]
    keep_baseline = ["distance_inverse", "TTC_inverse"]
    sub = df[(df["metric"] == "auprc") & (df["enhanced_score"].isin(keep_enhanced)) & (df["baseline_score"].isin(keep_baseline))].copy()
    order = [
        ("ROF_v2_composite", "distance_inverse"),
        ("ROF_v2_composite", "TTC_inverse"),
        ("temporal_composite", "distance_inverse"),
        ("temporal_composite", "TTC_inverse"),
    ]
    y = np.arange(len(order))[::-1]
    ax.axvline(0, color=COL["mid"], lw=0.8, zorder=1)
    for yi, pair in zip(y, order):
        row = sub[(sub["enhanced_score"] == pair[0]) & (sub["baseline_score"] == pair[1])].iloc[0]
        c = COL["rof"] if pair[0] == "ROF_v2_composite" else COL["temporal"]
        ax.errorbar(row["delta"], yi, xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]], fmt="o", color=c, ecolor=c, markersize=4.0, elinewidth=1.1, capsize=2.0, zorder=3)
        ax.text(row["ci_high"] + 0.010, yi, f"{row['delta']:+.3f}", ha="left", va="center", fontsize=5.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pretty_score(e)} − {pretty_score(b)}" for e, b in order])
    ax.set_xlabel("ΔAUPRC")
    ax.set_title("Known-failure ΔAUPRC vs proximity baselines", loc="left", pad=4)
    ax.set_xlim(min(-0.01, sub["ci_low"].min() - 0.03), sub["ci_high"].max() + 0.12)
    clean_ax(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_all_failure_sensitivity(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = ["distance_inverse", "TTC_inverse", "temporal_composite", "ROF_v2_composite"]
    sub = df[(df["task"] == "planner_failure_all") & (df["metric"] == "auprc") & (df["score"].isin(order))].copy().set_index("score").reindex(order).reset_index()
    x = np.arange(len(order))
    colors = [COL["distance"], COL["ttc"], COL["temporal"], COL["rof"]]
    for xi, (_, row), c in zip(x, sub.iterrows(), colors):
        ax.errorbar(xi, row["point"], yerr=[[row["point"] - row["ci_low"]], [row["ci_high"] - row["point"]]], fmt="o", color=c, ecolor=c, markersize=4.0, elinewidth=1.0, capsize=2.0, zorder=3)
        ax.text(xi, row["ci_high"] + 0.026, f"{row['point']:.3f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_score(s) for s in order], rotation=25, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_title("All-failure sensitivity (59/1,000)", loc="left", pad=4)
    ax.set_ylim(0, max(sub["ci_high"]) + 0.15)
    clean_ax(ax, grid="y")


def make_figure6(base: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    fig = plt.figure(figsize=(180 * MM, 126 * MM))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.02, 1.0, 1.20], height_ratios=[1.0, 1.05], wspace=0.46, hspace=0.62)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    plot_cohort_funnel(ax_a, read_csv(base, "Figure6_commonroad_neutral_validation/Figure6A_neutral_cohort_summary.csv"))
    plot_failure_taxonomy(ax_b, read_csv(base, "Figure6_commonroad_neutral_validation/Figure6B_failure_taxonomy.csv"))
    plot_commonroad_known_auprc(ax_c, read_csv(base, "Figure6_commonroad_neutral_validation/Figure6C_known_failure_metrics.csv"))
    plot_commonroad_known_deltas(ax_d, read_csv(base, "Figure6_commonroad_neutral_validation/Figure6D_known_failure_deltas.csv"))
    plot_all_failure_sensitivity(ax_e, read_csv(base, "Figure6_commonroad_neutral_validation/Figure6E_all_failure_sensitivity.csv"))

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d, ax_e], "abcde"):
        panel_label(ax, lab)
    fig.text(0.5, 0.006, "Primary CommonRoad interpretation uses known collision/lane planner failures; all planner failures are shown as sensitivity.", ha="center", va="bottom", fontsize=7.0, color=COL["dark"])
    save_figure(fig, out_dir, "Figure6_commonroad_neutral_validation", formats, dpi)


# -----------------------------------------------------------------------------
# Main CLI.
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROF Figures 2–6 from main figure source CSVs.")
    parser.add_argument("--input", required=True, type=Path, help="Path to ROF_results_v100_evidence_lock.zip or extracted ROF_results_v100_evidence_lock directory.")
    parser.add_argument("--out", default=Path("figures_nc"), type=Path, help="Output directory. Default: figures_nc")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], help="Output formats. Default: pdf svg png")
    parser.add_argument("--dpi", default=600, type=int, help="DPI for raster output. Default: 600")
    parser.add_argument("--figures", nargs="+", default=["2", "3", "4", "5", "6"], choices=["2", "3", "4", "5", "6"], help="Figures to generate. Default: 2 3 4 5 6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    tmp = tempfile.TemporaryDirectory() if args.input.suffix.lower() == ".zip" else None
    try:
        base = resolve_source_dir(args.input, tmp)
        makers = {
            "2": make_figure2,
            "3": make_figure3,
            "4": make_figure4,
            "5": make_figure5,
            "6": make_figure6,
        }
        for fig_id in args.figures:
            makers[fig_id](base, args.out, args.formats, args.dpi)
        print(f"Generated Figure {' '.join(args.figures)} in: {args.out.resolve()}")
        print("Formats:", ", ".join(args.formats))
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
