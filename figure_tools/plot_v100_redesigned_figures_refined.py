#!/usr/bin/env python3
"""
Generate refined Nature Communications-style Figure 2–6 drafts from the
v100 redesigned main-figure source-data package.

Input package expected:
  03_main_figure_source_data_v100_redesigned_with_pr_curve.zip
or an extracted folder containing:
  03_main_figure_source_data_v100_redesigned/

The script does not run experiments and does not change source data. It only
reads plot-ready CSV files and writes figure files.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
MM_TO_IN = 1.0 / 25.4

COL = {
    "black": "#111111",
    "dark": "#333333",
    "mid": "#6F6F6F",
    "light": "#D9D9D9",
    "very_light": "#F3F3F3",
    "grid": "#E5E5E5",
    "baseline": "#6F6F6F",
    "temporal": "#0072B2",
    "temporal_dark": "#004B7A",
    "rof": "#009E73",
    "rof_no_asr": "#56B4E9",
    "direct": "#D55E00",
    "explicit_excluded": "#009E73",
    "spatial": "#CC79A7",
    "unknown": "#E69F00",
    "known": "#D55E00",
    "safe": "#D9F0D3",
    "reduced": "#A6DBA0",
    "critical": "#F4A582",
    "infeasible": "#D6604D",
    "purple": "#7E3F98",
    "blue_light": "#D7EBF7",
    "green_light": "#DFF2EA",
    "orange_light": "#FCE8D7",
    "purple_light": "#F1E0F0",
}

SEVERITY_COLORS = {
    "High": COL["safe"],
    "Reduced": COL["reduced"],
    "Critical": COL["critical"],
    "Candidate-set infeasible": COL["infeasible"],
}

FEATURE_COLORS = {
    "strong_baseline_cv": COL["baseline"],
    "direct_action_ratios_only": COL["direct"],
    "explicit_ratio_field_excluded_current": COL["explicit_excluded"],
    "strict_spatial_no_action": COL["spatial"],
    "strict_temporal_dynamics": COL["temporal"],
}

VARIANT_FAMILY_COLORS = {
    "reference": COL["dark"],
    "horizon": COL["temporal"],
    "lane_buffer": COL["rof"],
    "action_library": COL["direct"],
    "future_handling": COL["purple"],
}

SCORE_COLORS = {
    "Distance inverse": COL["baseline"],
    "TTC inverse": "#9A9A9A",
    "ROF-v2 composite": COL["rof"],
    "ROF-v2 no-ASR": COL["rof_no_asr"],
    "Temporal composite": COL["temporal"],
}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def clean_axis(ax, grid: str | None = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=COL["grid"], linewidth=0.6)
        ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.16, y: float = 1.10) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=COL["black"])


def wrap(s: str, width: int = 26) -> str:
    if not isinstance(s, str):
        return ""
    return "\n".join(textwrap.wrap(s.replace("\n", " "), width=width, break_long_words=False))


def fmt_pct(x, digits: int = 1) -> str:
    return f"{float(x) * 100:.{digits}f}%"


def fmt_float(x, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def save_figure(fig: mpl.figure.Figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt.lower() in {"png", "tif", "tiff", "jpg", "jpeg"}:
            fig.savefig(path, dpi=dpi)
        else:
            fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths


def load_csv(root: Path, rel: str) -> pd.DataFrame:
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def locate_root(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return directory containing 03_main_figure_source_data_v100_redesigned."""
    tmp = None
    p = input_path.resolve()
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="v100_figdata_")
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp.name)
        base = Path(tmp.name)
    elif p.is_dir():
        base = p
    else:
        raise FileNotFoundError(f"Input not found or unsupported: {input_path}")

    candidates = []
    if (base / "03_main_figure_source_data_v100_redesigned").is_dir():
        candidates.append(base / "03_main_figure_source_data_v100_redesigned")
    if base.name == "03_main_figure_source_data_v100_redesigned":
        candidates.append(base)
    candidates.extend(base.rglob("03_main_figure_source_data_v100_redesigned"))
    for c in candidates:
        if (c / "Figure2_endpoint_diagnostic").is_dir():
            return c, tmp
    raise FileNotFoundError("Could not locate 03_main_figure_source_data_v100_redesigned in input.")


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------

def plot_figure2(root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> list[str]:
    f2a = load_csv(root, "Figure2_endpoint_diagnostic/Figure2a_severity_ladder_distribution_plot_ready.csv")
    f2b = load_csv(root, "Figure2_endpoint_diagnostic/Figure2b_proximity_actionability_heatmap_plot_ready.csv")
    f2c = load_csv(root, "Figure2_endpoint_diagnostic/Figure2c_feasible_ratio_by_severity_plot_ready.csv")
    f2d = load_csv(root, "Figure2_endpoint_diagnostic/Figure2d_distance_ttc_spearman_plot_ready.csv")

    fig = plt.figure(figsize=(190 * MM_TO_IN, 148 * MM_TO_IN), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.15], height_ratios=[1.0, 1.0],
                           wspace=0.42, hspace=0.56)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # 2a severity ladder
    d = f2a.sort_values("display_order")
    y = np.arange(len(d))[::-1]
    colors = [d.iloc[i].get("severity_color_hint", SEVERITY_COLORS.get(d.iloc[i]["display_label"], COL["light"])) for i in range(len(d))]
    ax_a.barh(y, d["fraction_pct"], height=0.68, color=colors, edgecolor="white", linewidth=0.8)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(d["display_label"])
    ax_a.set_xlabel("Samples (%)")
    ax_a.set_xlim(0, max(72, d["fraction_pct"].max() * 1.20))
    clean_axis(ax_a, grid="x")
    for yy, row in zip(y, d.itertuples(index=False)):
        ax_a.text(float(row.fraction_pct) + 1.0, yy, f"{int(row.count):,}  ({row.fraction_pct:.1f}%)",
                  va="center", ha="left", fontsize=7, color=COL["dark"])
    crit_rows = d[d["primary_endpoint_member"] == True]
    if not crit_rows.empty:
        ys = [y[list(d["display_label"]).index(lbl)] for lbl in crit_rows["display_label"]]
        y0, y1 = min(ys) - 0.35, max(ys) + 0.35
        x = ax_a.get_xlim()[1] * 0.92
        ax_a.plot([x, x], [y0, y1], color=COL["black"], lw=0.9)
        ax_a.plot([x - 1.4, x], [y0, y0], color=COL["black"], lw=0.9)
        ax_a.plot([x - 1.4, x], [y1, y1], color=COL["black"], lw=0.9)
        cnt = int(d["primary_endpoint_count"].iloc[0])
        frac = float(d["primary_endpoint_fraction"].iloc[0]) * 100
        ax_a.text(x - 2.1, (y0 + y1) / 2, f"Primary endpoint\ncritical-or-worse\n{cnt:,} ({frac:.2f}%)",
                  va="center", ha="right", fontsize=6.8, color=COL["black"],
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85))
    ax_a.set_title("Actionability severity and primary endpoint", loc="left", pad=7)
    panel_label(ax_a, "a")

    # 2b heatmap
    hb = f2b.copy()
    row_order = hb[["original_display_label", "original_display_order"]].drop_duplicates().sort_values("original_display_order")
    col_order = hb[["actionability_display_label", "display_order"]].drop_duplicates().sort_values("display_order")
    mat = hb.pivot(index="original_display_label", columns="actionability_display_label", values="row_fraction")
    mat = mat.loc[row_order["original_display_label"], col_order["actionability_display_label"]]
    count_mat = hb.pivot(index="original_display_label", columns="actionability_display_label", values="count").loc[mat.index, mat.columns]
    cmap = LinearSegmentedColormap.from_list("v100_heat", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    im = ax_b.imshow(mat.values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax_b.set_xticks(np.arange(mat.shape[1]))
    ax_b.set_xticklabels([wrap(c, 12) for c in mat.columns], rotation=0, ha="center")
    ax_b.set_yticks(np.arange(mat.shape[0]))
    ax_b.set_yticklabels(mat.index)
    ax_b.set_xlabel("Actionability label")
    ax_b.set_ylabel("Proximity label")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            cnt = int(count_mat.values[i, j])
            color = "white" if val > 0.55 else COL["black"]
            ax_b.text(j, i, f"{val*100:.0f}%\n{cnt:,}", ha="center", va="center", fontsize=6, color=color)
    ax_b.set_title("Proximity labels do not map one-to-one", loc="left", pad=7)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.045, pad=0.02)
    cbar.set_label("Row fraction", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    panel_label(ax_b, "b")

    # 2c feasible ratios
    c = f2c.sort_values(["display_order", "ratio_type"])
    labels = c[["display_order", "display_label"]].drop_duplicates().sort_values("display_order")
    xbase = np.arange(len(labels))
    ratio_types = list(c["ratio_display_label"].drop_duplicates())
    offsets = np.linspace(-0.13, 0.13, len(ratio_types)) if len(ratio_types) > 1 else [0]
    ratio_cols = {ratio_types[0]: COL["temporal"], ratio_types[-1]: COL["direct"]} if ratio_types else {}
    for off, rlab in zip(offsets, ratio_types):
        sub = c[c["ratio_display_label"] == rlab].sort_values("display_order")
        xx = xbase + off
        med = sub["median"].to_numpy()
        lo = sub["p25"].to_numpy()
        hi = sub["p75"].to_numpy()
        col = ratio_cols.get(rlab, COL["temporal"])
        ax_c.errorbar(xx, med * 100, yerr=[(med - lo) * 100, (hi - med) * 100], fmt="o",
                      ms=4.5, color=col, ecolor=col, elinewidth=1.2, capsize=2.5, label=rlab)
        ax_c.plot(xx, med * 100, color=col, lw=0.7, alpha=0.22)
    ax_c.set_xticks(xbase)
    ax_c.set_xticklabels([wrap(x, 12) for x in labels["display_label"]])
    ax_c.set_ylim(-3, 104)
    ax_c.set_ylabel("Feasible-candidate ratio (%)")
    ax_c.set_title("Candidate feasibility collapses with severity", loc="left", pad=7)
    ax_c.legend(frameon=False, loc="upper right")
    clean_axis(ax_c, grid="y")
    panel_label(ax_c, "c")

    # 2d Spearman coefficient plot
    dd = f2d.sort_values("display_order")
    y = np.arange(len(dd))[::-1]
    colors = [COL["black"] if g == "Distance" else COL["mid"] for g in dd["variable_group"]]
    for yy, row, col in zip(y, dd.itertuples(index=False), colors):
        ax_d.plot([row.ci_low, row.ci_high], [yy, yy], color=col, lw=1.6)
        ax_d.scatter(row.spearman, yy, s=28, color=col, zorder=3)
        ax_d.text(row.ci_high + 0.015, yy, f"{row.spearman:+.2f}", va="center", fontsize=7, color=COL["dark"])
    ax_d.axvline(0, color=COL["light"], lw=0.9, ls="--")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(dd["variable_display_label"])
    ax_d.set_xlim(-0.56, 0.16)
    ax_d.set_xlabel("Spearman correlation with actionability label")
    ax_d.set_title("Distance is moderate; TTC variants are weak", loc="left", pad=7)
    clean_axis(ax_d, grid="x")
    panel_label(ax_d, "d")

    return save_figure(fig, out_dir, "Figure2_endpoint_diagnostic_refined", formats, dpi)


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------

# def plot_figure3(root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> list[str]:
#     pr = load_csv(root, "Figure3_primary_waymo_oof/Figure3a_oof_pr_curve_plot_ready.csv")
#     f3b = load_csv(root, "Figure3_primary_waymo_oof/Figure3b_primary_auprc_paired_plot_ready.csv")
#     f3c = load_csv(root, "Figure3_primary_waymo_oof/Figure3c_delta_auprc_forest_plot_ready.csv")
#     f3d = load_csv(root, "Figure3_primary_waymo_oof/Figure3d_low_fpr_recall_paired_plot_ready.csv")
#     f3d_sum = load_csv(root, "Figure3_primary_waymo_oof/Figure3d_achieved_fpr_precision_summary.csv")
#
#     # Figure with top strip and 2x2 panels
#     fig = plt.figure(figsize=(190 * MM_TO_IN, 152 * MM_TO_IN), constrained_layout=False)
#     gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.17, 1.0, 1.03], wspace=0.42, hspace=0.56)
#     ax_strip = fig.add_subplot(gs[0, :])
#     ax_a = fig.add_subplot(gs[1, 0])
#     ax_b = fig.add_subplot(gs[1, 1])
#     ax_c = fig.add_subplot(gs[2, 0])
#     ax_d = fig.add_subplot(gs[2, 1])
#
#     # Top info strip
#     ax_strip.axis("off")
#     n = int(f3b["n"].dropna().iloc[0])
#     pos = int(f3b["positive_count"].dropna().iloc[0])
#     prev = float(f3b["prevalence"].dropna().iloc[0]) * 100
#     ax_strip.add_patch(Rectangle((0.01, 0.12), 0.98, 0.76, transform=ax_strip.transAxes,
#                                  facecolor=COL["very_light"], edgecolor=COL["light"], linewidth=0.6))
#     ax_strip.text(0.03, 0.62, "Waymo OOF primary analysis", ha="left", va="center", fontsize=7.3,
#                   fontweight="bold", color=COL["black"], transform=ax_strip.transAxes)
#     info = f"N = {n:,}   |   positives = {pos:,} ({prev:.2f}%)   |   5-fold OOF   |   RF seeds 41–43   |   calibration-derived nominal 5% FPR"
#     ax_strip.text(0.03, 0.32, info, ha="left", va="center", fontsize=6.7,
#                   color=COL["dark"], transform=ax_strip.transAxes)
#
#     # 3a PR curves
#     p = pr.copy()
#     # thin seed curves
#     for (feature, seed), sub in p[p["curve_type"] == "seed_curve"].groupby(["feature_set", "seed"]):
#         col = COL["baseline"] if feature == "strong_baseline_cv" else COL["temporal"]
#         ax_a.plot(sub["recall"], sub["precision"], color=col, lw=0.55, alpha=0.23)
#     # thick mean curves
#     handles = []
#     labels = []
#     for feature, sub in p[p["curve_type"] == "seed_mean_curve"].groupby("feature_set"):
#         col = COL["baseline"] if feature == "strong_baseline_cv" else COL["temporal"]
#         label = "Strong + CV" if feature == "strong_baseline_cv" else "+ strict temporal"
#         h, = ax_a.plot(sub["recall"], sub["precision"], color=col, lw=2.0, label=label)
#         handles.append(h); labels.append(label)
#     prev_val = float(p["prevalence"].dropna().iloc[0])
#     ax_a.axhline(prev_val, color=COL["light"], lw=1.0, ls="--")
#     ax_a.text(0.02, prev_val + 0.018, f"Prevalence {prev_val*100:.1f}%", fontsize=6.5, color=COL["mid"])
#     ax_a.set_xlim(0, 1.0)
#     ax_a.set_ylim(0, max(0.62, min(1.0, p["precision"].quantile(0.995) * 1.10)))
#     ax_a.set_xlabel("Recall")
#     ax_a.set_ylabel("Precision")
#     ax_a.set_title("Out-of-fold precision–recall", loc="left", pad=7)
#     ax_a.legend(handles, labels, frameon=False, loc="upper right")
#     clean_axis(ax_a, grid="both")
#     panel_label(ax_a, "a")
#
#     # 3b AUPRC mean bars with seed overlays. Bars make the absolute model
#     # comparison clearer than near-overlapping paired lines while retaining
#     # seed-level transparency.
#     b = f3b.copy()
#     seed_rows = b[b["row_type"] == "seed"].copy()
#     mean_calc = seed_rows.groupby(["x_order", "feature_set_display", "model_role"], as_index=False).agg(
#         mean_auprc=("auprc", "mean"),
#         min_auprc=("auprc", "min"),
#         max_auprc=("auprc", "max"),
#     ).sort_values("x_order")
#     bar_width = 0.56
#     for _, row in mean_calc.iterrows():
#         x = float(row.x_order)
#         col = COL["baseline"] if row.model_role == "baseline" else COL["temporal"]
#         face = "#BEBEBE" if row.model_role == "baseline" else COL["blue_light"]
#         ax_b.bar(x, row.mean_auprc, width=bar_width, color=face, edgecolor=col,
#                  linewidth=1.2, alpha=0.95, zorder=2)
#         # Seed range whisker.
#         ax_b.errorbar([x], [row.mean_auprc],
#                       yerr=[[row.mean_auprc - row.min_auprc], [row.max_auprc - row.mean_auprc]],
#                       fmt="none", ecolor=col, elinewidth=1.2, capsize=3.0, capthick=1.0, zorder=4)
#         sub = seed_rows[seed_rows["x_order"] == row.x_order].sort_values("seed")
#         jitter = np.linspace(-0.115, 0.115, len(sub))
#         ax_b.scatter(np.full(len(sub), x) + jitter, sub["auprc"], s=22,
#                      facecolor="white", edgecolor=col, linewidth=0.85, alpha=0.98, zorder=5)
#         ax_b.text(x, row.mean_auprc + 0.010, f"mean {row.mean_auprc:.3f}", ha="center", va="bottom",
#                   fontsize=6.5, color=COL["dark"])
#     ax_b.set_xticks([0, 1])
#     ax_b.set_xticklabels(["Strong + CV", "+ strict\ntemporal"])
#     ax_b.set_xlim(-0.55, 1.55)
#     ax_b.set_ylabel("AUPRC")
#     ax_b.set_ylim(0, 0.455)
#     dvals = f3c[f3c["row_type"] == "seed"]["delta"]
#     ybr = 0.430
#     ax_b.plot([0, 0, 1, 1], [ybr-0.006, ybr, ybr, ybr-0.006], color=COL["temporal_dark"], lw=0.8)
#     ax_b.text(0.5, ybr + 0.012, f"ΔAUPRC = +{dvals.min():.3f} to +{dvals.max():.3f}",
#               ha="center", va="bottom", fontsize=7, color=COL["temporal_dark"])
#     ax_b.set_title("Primary AUPRC across RF seeds", loc="left", pad=7)
#     clean_axis(ax_b, grid="y")
#     panel_label(ax_b, "b")
#
#     # 3c forest delta AUPRC
#     c = f3c.sort_values(["row_type", "seed"]).copy()
#     rows = c[c["row_type"] == "seed"].copy()
#     rows["y"] = list(range(len(rows), 0, -1))
#     mean = c[c["row_type"] == "mean"]
#     for _, row in rows.iterrows():
#         ax_c.plot([row.ci_low, row.ci_high], [row.y, row.y], color=COL["temporal"], lw=1.6)
#         ax_c.scatter(row.delta, row.y, s=42, color=COL["temporal"], edgecolor="white", linewidth=0.7, zorder=3)
#         ax_c.text(row.ci_high + 0.006, row.y, f"{row.delta:+.3f}", va="center", fontsize=6.7, color=COL["dark"])
#     if not mean.empty:
#         m = mean.iloc[0]
#         ax_c.scatter(m.delta, 0.55, s=60, marker="D", color=COL["temporal_dark"], edgecolor="white", linewidth=0.7, zorder=4)
#         ax_c.text(m.delta + 0.004, 0.55, f"Mean {m.delta:+.3f}", va="center", fontsize=7, color=COL["dark"])
#         ytick_pos = list(rows["y"]) + [0.55]
#         ytick_lab = list(rows["display_label"]) + ["Mean"]
#     else:
#         ytick_pos = list(rows["y"]); ytick_lab = list(rows["display_label"])
#     ax_c.axvline(0, color=COL["light"], lw=0.9, ls="--")
#     ax_c.set_yticks(ytick_pos)
#     ax_c.set_yticklabels(ytick_lab)
#     ax_c.set_xlim(0, 0.112)
#     ax_c.set_ylim(0.1, len(rows) + 0.45)
#     ax_c.set_xlabel("ΔAUPRC")
#     ax_c.set_title("Paired bootstrap gain", loc="left", pad=7)
#     clean_axis(ax_c, grid="x")
#     panel_label(ax_c, "c")
#
#     # 3d low-FPR recall mean bars with seed overlays, matching panel b.
#     d = f3d.copy()
#     mean_rec = d.groupby(["x_order", "feature_set_display", "model_role"], as_index=False).agg(
#         mean_recall=("recall", "mean"),
#         min_recall=("recall", "min"),
#         max_recall=("recall", "max"),
#     ).sort_values("x_order")
#     bar_width = 0.56
#     for _, row in mean_rec.iterrows():
#         x = float(row.x_order)
#         col = COL["baseline"] if row.model_role == "baseline" else COL["temporal"]
#         face = "#BEBEBE" if row.model_role == "baseline" else COL["blue_light"]
#         ax_d.bar(x, row.mean_recall, width=bar_width, color=face, edgecolor=col,
#                  linewidth=1.2, alpha=0.95, zorder=2)
#         ax_d.errorbar([x], [row.mean_recall],
#                       yerr=[[row.mean_recall - row.min_recall], [row.max_recall - row.mean_recall]],
#                       fmt="none", ecolor=col, elinewidth=1.2, capsize=3.0, capthick=1.0, zorder=4)
#         sub = d[d["x_order"] == row.x_order].sort_values("seed")
#         jitter = np.linspace(-0.115, 0.115, len(sub))
#         ax_d.scatter(np.full(len(sub), x) + jitter, sub["recall"], s=22,
#                      facecolor="white", edgecolor=col, linewidth=0.85, alpha=0.98, zorder=5)
#         ax_d.text(x, row.mean_recall + 0.015, f"mean {row.mean_recall:.3f}", ha="center", va="bottom",
#                   fontsize=6.5, color=COL["dark"])
#     ax_d.set_xticks([0, 1])
#     ax_d.set_xticklabels(["Strong + CV", "+ strict\ntemporal"])
#     ax_d.set_xlim(-0.55, 1.55)
#     ax_d.set_ylabel("Recall")
#     ax_d.set_ylim(0, 0.565)
#     dd = d.drop_duplicates("seed")
#     ybr = 0.530
#     ax_d.plot([0, 0, 1, 1], [ybr-0.008, ybr, ybr, ybr-0.008], color=COL["temporal_dark"], lw=0.8)
#     ax_d.text(0.5, ybr + 0.015, f"ΔRecall = +{dd['delta'].min():.3f} to +{dd['delta'].max():.3f}",
#               ha="center", va="bottom", fontsize=7, color=COL["temporal_dark"])
#     # Achieved FPR and precision ranges are reported in the caption/source data.
#     ax_d.set_title("Fold-calibrated low-FPR operation", loc="left", pad=7)
#     clean_axis(ax_d, grid="y")
#     panel_label(ax_d, "d")
#
#     return save_figure(fig, out_dir, "Figure3_primary_waymo_oof_refined", formats, dpi)
def plot_figure3(root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> list[str]:
    pr = load_csv(root, "Figure3_primary_waymo_oof/Figure3a_oof_pr_curve_plot_ready.csv")
    f3b = load_csv(root, "Figure3_primary_waymo_oof/Figure3b_primary_auprc_paired_plot_ready.csv")
    f3c = load_csv(root, "Figure3_primary_waymo_oof/Figure3c_delta_auprc_forest_plot_ready.csv")
    f3d = load_csv(root, "Figure3_primary_waymo_oof/Figure3d_low_fpr_recall_paired_plot_ready.csv")
    f3d_sum = load_csv(root, "Figure3_primary_waymo_oof/Figure3d_achieved_fpr_precision_summary.csv")

    # Figure with top strip and 2x2 panels
    fig = plt.figure(figsize=(190 * MM_TO_IN, 152 * MM_TO_IN), constrained_layout=False)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.17, 1.0, 1.03], wspace=0.42, hspace=0.56)
    ax_strip = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    # Top info strip
    ax_strip.axis("off")
    n = int(f3b["n"].dropna().iloc[0])
    pos = int(f3b["positive_count"].dropna().iloc[0])
    prev = float(f3b["prevalence"].dropna().iloc[0]) * 100
    # ax_strip.add_patch(Rectangle((0.01, -1.00), 0.98, 1.46, transform=ax_strip.transAxes,
    #                              facecolor=COL["very_light"], edgecolor=COL["light"], linewidth=0.6))
    # ax_strip.text(0.03, 0.72, "Waymo OOF primary analysis", ha="left", va="center", fontsize=7.3,
    #               fontweight="bold", color=COL["black"], transform=ax_strip.transAxes)
    # info = f"N = {n:,}   |   positives = {pos:,} ({prev:.2f}%)   |   5-fold OOF   |   RF seeds 41–43   |   calibration-derived nominal 5% FPR"
    # ax_strip.text(0.03, 0.12, info, ha="left", va="center", fontsize=6.7,
    #               color=COL["dark"], transform=ax_strip.transAxes)

    strip_dy = -0.50

    ax_strip.add_patch(Rectangle((0.01, -0.25 + strip_dy), 0.98, -5.50, transform=ax_strip.transAxes,
                                 facecolor=COL["very_light"], edgecolor=COL["light"], linewidth=0.6))

    ax_strip.text(0.03, 0.82 + strip_dy, "Waymo OOF primary analysis",
                  ha="left", va="center", fontsize=7.3,
                  fontweight="bold", color=COL["black"], transform=ax_strip.transAxes)

    info = f"N = {n:,}   |   positives = {pos:,} ({prev:.2f}%)   |   5-fold OOF   |   RF seeds 41–43   |   calibration-derived nominal 5% FPR"

    ax_strip.text(0.03, 0.18 + strip_dy, info,
                  ha="left", va="center", fontsize=6.7,
                  color=COL["dark"], transform=ax_strip.transAxes)

    # 3a PR curves
    p = pr.copy()

    # Thin seed curves
    for (feature, seed), sub in p[p["curve_type"] == "seed_curve"].groupby(["feature_set", "seed"]):
        col = COL["baseline"] if feature == "strong_baseline_cv" else COL["temporal"]
        ax_a.plot(sub["recall"], sub["precision"], color=col, lw=0.55, alpha=0.23)

    # Thick mean curves
    handles = []
    labels = []
    for feature, sub in p[p["curve_type"] == "seed_mean_curve"].groupby("feature_set"):
        col = COL["baseline"] if feature == "strong_baseline_cv" else COL["temporal"]
        label = "Strong + CV" if feature == "strong_baseline_cv" else "+ strict temporal"
        h, = ax_a.plot(sub["recall"], sub["precision"], color=col, lw=2.0, label=label)
        handles.append(h)
        labels.append(label)

    prev_val = float(p["prevalence"].dropna().iloc[0])
    ax_a.axhline(prev_val, color=COL["light"], lw=1.0, ls="--")
    ax_a.text(0.02, prev_val + 0.018, f"Prevalence {prev_val*100:.1f}%", fontsize=6.5, color=COL["mid"])
    ax_a.set_xlim(0, 1.0)
    ax_a.set_ylim(0, max(0.62, min(1.0, p["precision"].quantile(0.995) * 1.10)))
    ax_a.set_xlabel("Recall")
    ax_a.set_ylabel("Precision")
    ax_a.set_title("Out-of-fold precision–recall", loc="left", pad=7)
    ax_a.legend(handles, labels, frameon=False, loc="upper right")
    clean_axis(ax_a, grid="both")
    panel_label(ax_a, "a")

    # ------------------------------------------------------------------
    # 3b Per-seed AUPRC gain: three bars, one bar per RF seed.
    # Bar height = (+ strict temporal) - (Strong + CV).
    # Use x_order rather than model_role to avoid dependency on label names.
    # ------------------------------------------------------------------
    b = f3b.copy()
    seed_rows = b[b["row_type"].astype(str).str.lower() == "seed"].copy()

    seed_rows["x_order"] = pd.to_numeric(seed_rows["x_order"], errors="coerce")
    seed_rows["seed"] = pd.to_numeric(seed_rows["seed"], errors="coerce")
    seed_rows["auprc"] = pd.to_numeric(seed_rows["auprc"], errors="coerce")
    seed_rows = seed_rows.dropna(subset=["seed", "x_order", "auprc"])

    x_orders = sorted(seed_rows["x_order"].unique())
    if len(x_orders) < 2:
        raise ValueError("Figure3b data must contain at least two x_order groups for baseline and temporal models.")

    baseline_x = x_orders[0]
    temporal_x = x_orders[-1]

    b_pivot = (
        seed_rows.pivot_table(
            index="seed",
            columns="x_order",
            values="auprc",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    if baseline_x not in b_pivot.columns or temporal_x not in b_pivot.columns:
        raise ValueError("Figure3b data cannot be pivoted into baseline and temporal AUPRC columns.")

    b_pivot["seed"] = b_pivot["seed"].astype(int)
    b_pivot = b_pivot.sort_values("seed").reset_index(drop=True)
    b_pivot["baseline"] = b_pivot[baseline_x].astype(float)
    b_pivot["temporal"] = b_pivot[temporal_x].astype(float)
    b_pivot["delta"] = b_pivot["temporal"] - b_pivot["baseline"]

    x = np.arange(len(b_pivot))

    # Low-saturation blue gradient, suitable for journal-style figures.
    seed_colors = ["#D8E8F3", "#8DB9D8", "#3E7FAE"]
    seed_edge = "#24506F"
    mean_line_col = "#1F5E83"

    ax_b.bar(
        x,
        b_pivot["delta"],
        width=0.62,
        color=seed_colors[:len(b_pivot)],
        edgecolor=seed_edge,
        linewidth=1.0,
        alpha=0.98,
        zorder=3,
    )

    ax_b.axhline(0, color=COL["light"], lw=0.9, ls="--", zorder=1)

    mean_delta_b = float(b_pivot["delta"].mean())
    ax_b.axhline(mean_delta_b, color=mean_line_col, lw=0.9, ls="-", alpha=0.70, zorder=2)
    ax_b.text(
        len(b_pivot) - 0.45,
        mean_delta_b,
        f"mean {mean_delta_b:+.3f}",
        ha="left",
        va="center",
        fontsize=6.4,
        color=mean_line_col,
    )

    span_b = max(abs(b_pivot["delta"]).max(), 1e-6)
    for i, row in b_pivot.iterrows():
        y = float(row["delta"])
        ax_b.text(
            i,
            y + 0.09 * span_b if y >= 0 else y - 0.11 * span_b,
            f"{y:+.3f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=6.8,
            color=COL["dark"],
            fontweight="bold",
            zorder=5,
        )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([
        f"Seed {int(row.seed)}\n{row.baseline:.3f} → {row.temporal:.3f}"
        for _, row in b_pivot.iterrows()
    ])

    ymin_b = min(0, float(b_pivot["delta"].min())) - 0.22 * span_b
    ymax_b = max(0, float(b_pivot["delta"].max())) + 0.35 * span_b
    ax_b.set_ylim(ymin_b, ymax_b)

    ax_b.set_ylabel("ΔAUPRC")
    ax_b.set_title("Per-seed AUPRC gain", loc="left", pad=7)
    ax_b.text(
        0.02,
        0.97,
        "(+ strict temporal) − (Strong + CV)",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=COL["mid"],
    )

    clean_axis(ax_b, grid="y")
    panel_label(ax_b, "b")

    # 3c forest delta AUPRC
    c = f3c.sort_values(["row_type", "seed"]).copy()
    rows = c[c["row_type"] == "seed"].copy()
    rows["y"] = list(range(len(rows), 0, -1))
    mean = c[c["row_type"] == "mean"]

    for _, row in rows.iterrows():
        ax_c.plot([row.ci_low, row.ci_high], [row.y, row.y], color=COL["temporal"], lw=1.6)
        ax_c.scatter(row.delta, row.y, s=42, color=COL["temporal"], edgecolor="white", linewidth=0.7, zorder=3)
        ax_c.text(row.ci_high + 0.006, row.y, f"{row.delta:+.3f}", va="center", fontsize=6.7, color=COL["dark"])

    if not mean.empty:
        m = mean.iloc[0]
        ax_c.scatter(m.delta, 0.55, s=60, marker="D", color=COL["temporal_dark"], edgecolor="white", linewidth=0.7, zorder=4)
        ax_c.text(m.delta + 0.004, 0.55, f"Mean {m.delta:+.3f}", va="center", fontsize=7, color=COL["dark"])
        ytick_pos = list(rows["y"]) + [0.55]
        ytick_lab = list(rows["display_label"]) + ["Mean"]
    else:
        ytick_pos = list(rows["y"])
        ytick_lab = list(rows["display_label"])

    ax_c.axvline(0, color=COL["light"], lw=0.9, ls="--")
    ax_c.set_yticks(ytick_pos)
    ax_c.set_yticklabels(ytick_lab)
    ax_c.set_xlim(0, 0.112)
    ax_c.set_ylim(0.1, len(rows) + 0.45)
    ax_c.set_xlabel("ΔAUPRC")
    ax_c.set_title("Paired bootstrap gain", loc="left", pad=7)
    clean_axis(ax_c, grid="x")
    panel_label(ax_c, "c")

    # ------------------------------------------------------------------
    # 3d Per-seed low-FPR recall gain: three bars, one bar per RF seed.
    # Bar height = (+ strict temporal) - (Strong + CV).
    # Use x_order rather than model_role to avoid dependency on label names.
    # ------------------------------------------------------------------
    d = f3d.copy()

    d["x_order"] = pd.to_numeric(d["x_order"], errors="coerce")
    d["seed"] = pd.to_numeric(d["seed"], errors="coerce")
    d["recall"] = pd.to_numeric(d["recall"], errors="coerce")
    d = d.dropna(subset=["seed", "x_order", "recall"])

    x_orders = sorted(d["x_order"].unique())
    if len(x_orders) < 2:
        raise ValueError("Figure3d data must contain at least two x_order groups for baseline and temporal models.")

    baseline_x = x_orders[0]
    temporal_x = x_orders[-1]

    d_pivot = (
        d.pivot_table(
            index="seed",
            columns="x_order",
            values="recall",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    if baseline_x not in d_pivot.columns or temporal_x not in d_pivot.columns:
        raise ValueError("Figure3d data cannot be pivoted into baseline and temporal recall columns.")

    d_pivot["seed"] = d_pivot["seed"].astype(int)
    d_pivot = d_pivot.sort_values("seed").reset_index(drop=True)
    d_pivot["baseline"] = d_pivot[baseline_x].astype(float)
    d_pivot["temporal"] = d_pivot[temporal_x].astype(float)
    d_pivot["delta"] = d_pivot["temporal"] - d_pivot["baseline"]

    x = np.arange(len(d_pivot))

    # Same color scheme as panel b.
    seed_colors = ["#D8E8F3", "#8DB9D8", "#3E7FAE"]
    seed_edge = "#24506F"
    mean_line_col = "#1F5E83"

    ax_d.bar(
        x,
        d_pivot["delta"],
        width=0.62,
        color=seed_colors[:len(d_pivot)],
        edgecolor=seed_edge,
        linewidth=1.0,
        alpha=0.98,
        zorder=3,
    )

    ax_d.axhline(0, color=COL["light"], lw=0.9, ls="--", zorder=1)

    mean_delta_d = float(d_pivot["delta"].mean())
    ax_d.axhline(mean_delta_d, color=mean_line_col, lw=0.9, ls="-", alpha=0.70, zorder=2)
    ax_d.text(
        len(d_pivot) - 0.45,
        mean_delta_d,
        f"mean {mean_delta_d:+.3f}",
        ha="left",
        va="center",
        fontsize=6.4,
        color=mean_line_col,
    )

    span_d = max(abs(d_pivot["delta"]).max(), 1e-6)
    for i, row in d_pivot.iterrows():
        y = float(row["delta"])
        ax_d.text(
            i,
            y + 0.09 * span_d if y >= 0 else y - 0.11 * span_d,
            f"{y:+.3f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=6.8,
            color=COL["dark"],
            fontweight="bold",
            zorder=5,
        )

    ax_d.set_xticks(x)
    ax_d.set_xticklabels([
        f"Seed {int(row.seed)}\n{row.baseline:.3f} → {row.temporal:.3f}"
        for _, row in d_pivot.iterrows()
    ])

    ymin_d = min(0, float(d_pivot["delta"].min())) - 0.22 * span_d
    ymax_d = max(0, float(d_pivot["delta"].max())) + 0.35 * span_d
    ax_d.set_ylim(ymin_d, ymax_d)

    ax_d.set_ylabel("ΔRecall")
    ax_d.set_title("Per-seed low-FPR recall gain", loc="left", pad=7)
    ax_d.text(
        0.02,
        0.97,
        "(+ strict temporal) − (Strong + CV) at nominal 5% FPR",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=COL["mid"],
    )

    # Achieved FPR and precision ranges are reported in the caption/source data.
    clean_axis(ax_d, grid="y")
    panel_label(ax_d, "d")

    return save_figure(fig, out_dir, "Figure3_primary_waymo_oof_refined", formats, dpi)

# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------

def plot_figure4(root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> list[str]:
    tax = load_csv(root, "Figure4_mechanism_audit/Figure4a_feature_family_taxonomy_cards_plot_ready.csv")
    forest = load_csv(root, "Figure4_mechanism_audit/Figure4b_feature_family_delta_auprc_forest_plot_ready.csv")
    access = load_csv(root, "Figure4_mechanism_audit/Figure4c_information_access_matrix_plot_ready.csv")

    fig = plt.figure(figsize=(190 * MM_TO_IN, 138 * MM_TO_IN), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.62, 1.38], width_ratios=[1.30, 1.0],
                           hspace=0.36, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # 4a taxonomy cards
    ax_a.axis("off")
    d = tax.sort_values("display_order")
    n = len(d)
    margin = 0.015
    gap = 0.012
    card_w = (1 - 2*margin - (n-1)*gap) / n
    for i, row in enumerate(d.itertuples(index=False)):
        x = margin + i * (card_w + gap)
        fam = row.feature_family
        role = row.family_role
        face = {
            "strong_baseline_cv": "#F7F7F7",
            "direct_action_ratios_only": COL["orange_light"],
            "explicit_ratio_field_excluded_current": COL["green_light"],
            "strict_spatial_no_action": COL["purple_light"],
            "strict_temporal_dynamics": COL["blue_light"],
        }.get(fam, "white")
        edge = FEATURE_COLORS.get(fam, COL["mid"])
        box = FancyBboxPatch((x, 0.18), card_w, 0.62, boxstyle="round,pad=0.012,rounding_size=0.02",
                             facecolor=face, edgecolor=edge, linewidth=0.9, transform=ax_a.transAxes)
        ax_a.add_patch(box)
        ax_a.text(x + 0.018, 0.75, wrap(row.feature_family_display, 18), transform=ax_a.transAxes,
                  ha="left", va="top", fontsize=6.1, fontweight="bold", color=COL["black"])
        ax_a.text(x + 0.018, 0.50, f"{int(row.feature_count)} fields", transform=ax_a.transAxes,
                  ha="left", va="top", fontsize=5.8, color=COL["dark"])
        flags = []
        if not row.uses_observed_future_any and not row.uses_label_file_any:
            flags.append("no future/label")
        if row.uses_explicit_asr_field_any:
            flags.append("explicit ASR")
        elif row.uses_transformed_or_composite_asr_any:
            flags.append("ASR-derived")
        if role == "primary mechanism":
            flags.append("primary")
        ax_a.text(x + 0.018, 0.33, wrap(" · ".join(flags), 28), transform=ax_a.transAxes,
                  ha="left", va="top", fontsize=5.6, color=COL["dark"])
        # Key feature lists are kept in source tables/Supplementary Methods; cards show only role and lineage flags.
    ax_a.text(0.0, 1.00, "(a)", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax_a.text(0.07, 1.00, "Feature-family taxonomy", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=9)

    # # 4b forest by feature family
    # f = forest[forest["metric"] == "auprc"].copy().sort_values("display_order")
    # mean_rows = f[f["row_type"] == "mean"].sort_values("display_order")
    # seed_rows = f[f["row_type"] == "seed"].sort_values(["display_order", "seed"])
    # families = list(mean_rows["feature_family"].drop_duplicates())
    # fam_to_y = {fam: len(families)-1-i for i, fam in enumerate(families)}
    # for fam, sub in seed_rows.groupby("feature_family"):
    #     yy = fam_to_y[fam]
    #     col = FEATURE_COLORS.get(fam, COL["temporal"])
    #     offsets = np.linspace(-0.16, 0.16, len(sub))
    #     for off, (_, row) in zip(offsets, sub.iterrows()):
    #         y = yy + off
    #         ax_b.plot([row.ci_low, row.ci_high], [y, y], color=col, lw=1.0, alpha=0.62)
    #         ax_b.scatter(row.delta, y, s=22, color=col, edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    # # Keep numeric labels in a dedicated right-hand column so they do not overlap the CI bars.
    # label_x = max(0.112, float(max(seed_rows["ci_high"].max(), mean_rows["delta"].max())) * 1.32)
    # for _, row in mean_rows.iterrows():
    #     yy = fam_to_y[row.feature_family]
    #     col = FEATURE_COLORS.get(row.feature_family, COL["temporal"])
    #     ax_b.scatter(row.delta, yy, s=62, marker="D", color=col, edgecolor="white", linewidth=0.7, zorder=4)
    #     ax_b.text(label_x, yy, f"{row.delta:+.3f}", va="center", ha="right", fontsize=6.8, color=COL["dark"])
    # ax_b.axvline(0, color=COL["light"], lw=0.9, ls="--")
    # ax_b.set_yticks([fam_to_y[fam] for fam in families])
    # label_map = dict(zip(mean_rows["feature_family"], mean_rows["feature_family_display"]))
    # ax_b.set_yticklabels([wrap(label_map[fam], 24) for fam in families])
    # ax_b.set_xlim(0, label_x * 1.06)
    # ax_b.text(label_x, max(fam_to_y.values()) + 0.43, "mean Δ", ha="right", va="bottom", fontsize=6.2, color=COL["mid"])
    # ax_b.set_xlabel("ΔAUPRC over Strong + CV")
    # ax_b.set_title("Incremental signal by feature family", loc="left", pad=7)
    # clean_axis(ax_b, grid="x")
    # panel_label(ax_b, "b")

    # 4b forest by feature family
    f = forest[forest["metric"] == "auprc"].copy().sort_values("display_order")
    mean_rows = f[f["row_type"] == "mean"].sort_values("display_order")
    seed_rows = f[f["row_type"] == "seed"].sort_values(["display_order", "seed"])
    families = list(mean_rows["feature_family"].drop_duplicates())
    fam_to_y = {fam: len(families)-1-i for i, fam in enumerate(families)}

    # Axis range: include a small negative region so CIs crossing zero are visible.
    # The v100 source data include strict_spatial_no_action seed 43 with a CI
    # below zero, so the x-axis must not start at 0.
    max_right = float(max(seed_rows["ci_high"].max(), mean_rows["delta"].max()))
    min_left = float(min(seed_rows["ci_low"].min(), mean_rows["delta"].min()))
    x_pad = 0.008
    x_left = min(-0.012, min_left - x_pad)
    label_x = max(0.112, max_right * 1.32)
    x_right = label_x * 1.06

    for fam, sub in seed_rows.groupby("feature_family"):
        yy = fam_to_y[fam]
        col = FEATURE_COLORS.get(fam, COL["temporal"])
        offsets = np.linspace(-0.16, 0.16, len(sub))
        for off, (_, row) in zip(offsets, sub.iterrows()):
            y = yy + off
            ax_b.plot(
                [row.ci_low, row.ci_high],
                [y, y],
                color=col,
                lw=1.0,
                alpha=0.62,
                zorder=2,
            )
            ax_b.scatter(
                row.delta,
                y,
                s=22,
                color=col,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )

    # Mean markers and numeric labels.
    for _, row in mean_rows.iterrows():
        yy = fam_to_y[row.feature_family]
        col = FEATURE_COLORS.get(row.feature_family, COL["temporal"])
        ax_b.scatter(
            row.delta,
            yy,
            s=62,
            marker="D",
            color=col,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax_b.text(
            label_x,
            yy,
            f"{row.delta:+.3f}",
            va="center",
            ha="right",
            fontsize=6.8,
            color=COL["dark"],
        )

    # Explicit zero line. This should be visible inside the plotting region.
    ax_b.axvline(0, color=COL["black"], lw=0.8, ls="--", alpha=0.75, zorder=1)

    ax_b.set_yticks([fam_to_y[fam] for fam in families])
    label_map = dict(zip(mean_rows["feature_family"], mean_rows["feature_family_display"]))
    ax_b.set_yticklabels([wrap(label_map[fam], 24) for fam in families])

    ax_b.set_xlim(x_left, x_right)
    ax_b.text(
        label_x,
        max(fam_to_y.values()) + 0.43,
        "mean Δ",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=COL["mid"],
    )
    ax_b.set_xlabel("ΔAUPRC over Strong + CV")
    ax_b.set_title("Incremental signal by feature family", loc="left", pad=7)
    clean_axis(ax_b, grid="x")
    panel_label(ax_b, "b")

    # 4c information access matrix
    mat = access.pivot(index="feature_family_display", columns="access_dimension_display", values="access_value")
    order_rows = access[["feature_family_display", "display_order"]].drop_duplicates().sort_values("display_order")["feature_family_display"]
    order_cols = ["Observed future", "Label file", "Explicit ASR ratios", "Transformed/composite ASR", "Primitive-safety dynamics"]
    mat = mat.loc[order_rows, order_cols].astype(bool)
    ax_c.set_xlim(-0.5, len(order_cols)-0.5)
    ax_c.set_ylim(-0.5, len(order_rows)-0.5)
    ax_c.invert_yaxis()
    for i, row_lab in enumerate(order_rows):
        fam = access[access["feature_family_display"] == row_lab]["feature_family"].iloc[0]
        for j, col_lab in enumerate(order_cols):
            used = bool(mat.loc[row_lab, col_lab])
            if used:
                face = FEATURE_COLORS.get(fam, COL["mid"])
                ax_c.scatter(j, i, s=65, facecolor=face, edgecolor="white", linewidth=0.7, zorder=3)
            else:
                ax_c.scatter(j, i, s=42, facecolor="white", edgecolor=COL["light"], linewidth=0.9, zorder=2)
    ax_c.set_xticks(np.arange(len(order_cols)))
    ax_c.set_xticklabels(["Future", "Label", "Explicit\nASR", "Composite\nASR", "Primitive\nsafety"], rotation=25, ha="right", rotation_mode="anchor")
    ax_c.set_yticks(np.arange(len(order_rows)))
    ax_c.set_yticklabels([wrap(x, 22) for x in order_rows])
    ax_c.set_title("Information-access lineage", loc="left", pad=7)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.tick_params(length=0)
    ax_c.grid(False)
    panel_label(ax_c, "c")
    ax_c.text(0.0, -0.24, "Filled circles indicate accessed or derived information.\nPrimary predictors do not read observed futures or labels.",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=6.3, color=COL["mid"])

    return save_figure(fig, out_dir, "Figure4_mechanism_audit_refined", formats, dpi)


# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------

def plot_figure5(root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> list[str]:
    lab = load_csv(root, "Figure5_endpoint_design_sensitivity/Figure5a_label_design_sensitivity_dot_columns_plot_ready.csv")
    delt = load_csv(root, "Figure5_endpoint_design_sensitivity/Figure5b_aligned_label_feature_delta_auprc_forest_plot_ready.csv")
    cards = load_csv(root, "Figure5_endpoint_design_sensitivity/Figure5c_cv_fallback_inset_cards_plot_ready.csv")
    effects = load_csv(root, "Figure5_endpoint_design_sensitivity/Figure5c_cv_fallback_primary_effects_plot_ready.csv")

    fig = plt.figure(figsize=(190 * MM_TO_IN, 155 * MM_TO_IN), constrained_layout=False)
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.62, 0.70], width_ratios=[1.0, 1.0, 1.0, 1.82],
                           hspace=0.50, wspace=0.32)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_ch = fig.add_subplot(gs[0, 1], sharey=ax_p)
    ax_j = fig.add_subplot(gs[0, 2], sharey=ax_p)
    ax_f = fig.add_subplot(gs[0, 3], sharey=ax_p)
    ax_cv = fig.add_subplot(gs[1, :])

    # Row ordering
    variant_order = lab[["variant_id", "variant_display_label", "display_order", "family"]].drop_duplicates().sort_values("display_order")
    variant_ids = list(variant_order["variant_id"])
    ymap = {vid: len(variant_ids)-1-i for i, vid in enumerate(variant_ids)}
    y_positions = [ymap[v] for v in variant_ids]
    def _variant_display_label(x):
        s = str(x).replace("\\n", "\n")
        s = s.replace("Reference\n3 s / 3 m / base7", "Reference\n3 s / 3 m / base7")
        return s
    y_labels = [_variant_display_label(x) for x in variant_order["variant_display_label"]]
    fams = dict(zip(variant_order["variant_id"], variant_order["family"]))
    colors = {vid: VARIANT_FAMILY_COLORS.get(fams[vid], COL["dark"]) for vid in variant_ids}

    def draw_dot_col(ax, metric, xlabel, xlim, pct=False):
        sub = lab[lab["metric"] == metric]
        for _, row in sub.iterrows():
            y = ymap[row.variant_id]
            val = row.value_pct if pct else row.value
            ax.scatter(val, y, s=42, color=colors[row.variant_id], edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        clean_axis(ax, grid="x")

    draw_dot_col(ax_p, "critical_or_worse_prevalence", "Prevalence (%)", (0, 24), pct=True)
    draw_dot_col(ax_ch, "label_changed_fraction", "Labels changed (%)", (0, 36), pct=True)
    draw_dot_col(ax_j, "severe_set_jaccard_vs_reference", "Severe-set\nJaccard", (0, 1.05), pct=False)
    ax_p.set_title("Endpoint labels shift under design perturbations", loc="left", pad=7)
    panel_label(ax_p, "a")
    plt.setp(ax_ch.get_yticklabels(), visible=False)
    plt.setp(ax_j.get_yticklabels(), visible=False)
    ax_ch.tick_params(axis="y", length=0)
    ax_j.tick_params(axis="y", length=0)

    # Background family bands across all top axes
    for ax in [ax_p, ax_ch, ax_j, ax_f]:
        for vid in variant_ids:
            y = ymap[vid]
            if y % 2 == 0:
                ax.axhspan(y-0.45, y+0.45, color="#FAFAFA", zorder=0)

    # aligned forest plot
    d = delt[delt["metric"] == "auprc"].copy()
    seed = d[d["row_type"] == "seed"]
    mean = d[d["row_type"] == "mean"]
    for vid, sub in seed.groupby("variant_id"):
        y0 = ymap[vid]
        col = colors.get(vid, COL["temporal"])
        offsets = np.linspace(-0.16, 0.16, len(sub))
        for off, (_, row) in zip(offsets, sub.iterrows()):
            y = y0 + off
            ax_f.plot([row.ci_low, row.ci_high], [y, y], color=col, lw=1.0, alpha=0.62)
            ax_f.scatter(row.delta, y, s=22, color=col, edgecolor="white", linewidth=0.5, zorder=3)
    for _, row in mean.iterrows():
        y0 = ymap[row.variant_id]
        col = colors.get(row.variant_id, COL["temporal"])
        ax_f.scatter(row.delta, y0, s=62, marker="D", color=col, edgecolor="white", linewidth=0.7, zorder=4)
        ax_f.text(row.delta + 0.020, y0, f"{row.delta:+.3f}", va="center", fontsize=6.5, color=COL["dark"])
    ax_f.axvline(0, color=COL["light"], lw=0.9, ls="--")
    ax_f.set_xlim(0, max(0.54, seed["ci_high"].max()*1.22))
    ax_f.set_xlabel("Aligned ΔAUPRC")
    ax_f.set_title("Strict-temporal gain remains positive", loc="left", pad=7)
    clean_axis(ax_f, grid="x")
    plt.setp(ax_f.get_yticklabels(), visible=False)
    ax_f.tick_params(axis="y", length=0)
    panel_label(ax_f, "b")

    # 5c CV-fallback focused inset
    ax_cv.axis("off")
    ax_cv.text(0.0, 1.05, "(c)", transform=ax_cv.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax_cv.text(0.07, 1.05, "CV-fallback future handling has minimal label impact", transform=ax_cv.transAxes,
               ha="left", va="bottom", fontsize=9)
    # compact text summary on the left
    cv_summary = {str(r.card_label): str(r.display_value) for r in cards.itertuples(index=False)}
    summary_lines = [
        ("Labels changed", cv_summary.get("Labels changed", "")),
        ("Severe Jaccard", cv_summary.get("Severe-set Jaccard", "")),
        ("ΔAUPRC", cv_summary.get("ΔAUPRC range", "")),
        ("ΔRecall", cv_summary.get("ΔRecall@5% FPR range", "")),
    ]
    y0 = 0.76
    for i, (labtxt, valtxt) in enumerate(summary_lines):
        y = y0 - i * 0.18
        ax_cv.text(0.03, y, labtxt, transform=ax_cv.transAxes, ha="left", va="center", fontsize=6.6, color=COL["mid"])
        ax_cv.text(0.185, y, valtxt, transform=ax_cv.transAxes, ha="left", va="center", fontsize=7.0,
                   fontweight="bold", color=COL["black"])
    # Smaller left summary box; leave a clear gutter before the right mini-forest.
    ax_cv.add_patch(FancyBboxPatch((0.015, 0.15), 0.42, 0.70, boxstyle="round,pad=0.010,rounding_size=0.02",
                                   facecolor="white", edgecolor=COL["light"], linewidth=0.7, transform=ax_cv.transAxes, zorder=-1))
    # compact effect intervals on the right
    inset = ax_cv.inset_axes([0.64, 0.20, 0.33, 0.62])
    eff = effects[(effects["section"] == "primary_deltas") & (effects["metric"].isin(["auprc", "recall_at_5pct_fpr"]))]
    # Compute per metric mean delta and min/max CI for a compact interval band
    ys = {"auprc": 1, "recall_at_5pct_fpr": 0}
    labels = {"auprc": "AUPRC", "recall_at_5pct_fpr": "Recall"}
    for metric, sub in eff.groupby("metric"):
        y = ys[metric]
        mean_delta = sub["delta"].mean()
        ci_low = sub["ci_low"].min()
        ci_high = sub["ci_high"].max()
        inset.plot([ci_low, ci_high], [y, y], color=COL["temporal"], lw=1.4)
        inset.scatter(mean_delta, y, marker="D", s=42, color=COL["temporal_dark"], edgecolor="white", linewidth=0.6)
        inset.text(ci_high + 0.004, y, f"{mean_delta:+.3f}", va="center", fontsize=6.6)
    inset.axvline(0, color=COL["light"], lw=0.9, ls="--")
    inset.set_yticks([ys["recall_at_5pct_fpr"], ys["auprc"]])
    inset.set_yticklabels([labels["recall_at_5pct_fpr"], labels["auprc"]])
    inset.set_xlabel("Effect under CV-fallback")
    inset.set_xlim(0, 0.13)
    clean_axis(inset, grid="x")

    return save_figure(fig, out_dir, "Figure5_endpoint_design_sensitivity_refined", formats, dpi)


# -----------------------------------------------------------------------------
# Figure 6
# -----------------------------------------------------------------------------

def plot_figure6(root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> list[str]:
    flow = load_csv(root, "Figure6_commonroad_neutral_validation/Figure6a_neutral_validation_flow_plot_ready.csv")
    tax = load_csv(root, "Figure6_commonroad_neutral_validation/Figure6b_planner_outcome_taxonomy_strip_plot_ready.csv")
    metrics = load_csv(root, "Figure6_commonroad_neutral_validation/Figure6c_known_failure_auprc_point_range_plot_ready.csv")
    delta = load_csv(root, "Figure6_commonroad_neutral_validation/Figure6d_known_failure_delta_auprc_forest_plot_ready.csv")

    fig = plt.figure(figsize=(190 * MM_TO_IN, 150 * MM_TO_IN), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.02, 1.26], height_ratios=[0.88, 1.12],
                           wspace=0.36, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # 6a neutral validation flow
    ax_a.axis("off")
    flow = flow.sort_values("node_order")
    ys = np.linspace(0.84, 0.16, len(flow))
    for i, (y, row) in enumerate(zip(ys, flow.itertuples(index=False))):
        color = COL["very_light"] if row.node_type in ["protocol", "cohort"] else (COL["orange_light"] if row.node_type == "outcome" else COL["blue_light"])
        edge = COL["mid"] if row.node_type != "primary_outcome" else COL["temporal"]
        box = FancyBboxPatch((0.06, y-0.055), 0.88, 0.095, boxstyle="round,pad=0.012,rounding_size=0.022",
                             facecolor=color, edgecolor=edge, linewidth=0.8, transform=ax_a.transAxes)
        ax_a.add_patch(box)
        value = "" if pd.isna(row.value) else f"{row.display_value}  "
        ax_a.text(0.50, y, value + row.node_label, transform=ax_a.transAxes, ha="center", va="center",
                  fontsize=6.6, color=COL["black"], fontweight="bold" if row.node_type == "primary_outcome" else "normal")
        if i < len(flow) - 1:
            ax_a.annotate("", xy=(0.50, ys[i+1]+0.065), xytext=(0.50, y-0.065), xycoords=ax_a.transAxes,
                          arrowprops=dict(arrowstyle="-|>", lw=0.8, color=COL["mid"]))
    ax_a.text(0.50, 0.02, "Outcome-blind sampling; no actionability/ROF/planner-label enrichment", transform=ax_a.transAxes,
              ha="center", va="bottom", fontsize=6.2, color=COL["mid"])
    ax_a.set_title("Neutral CommonRoad validation flow", loc="left", pad=10)
    panel_label(ax_a, "a", x=-0.14, y=1.18)

    # 6b taxonomy strip
    ax_b.set_title("Planner outcome taxonomy", loc="left", pad=7)
    tax = tax.sort_values("display_order")
    total = tax["count"].sum()
    left = 0.0
    color_map = {"No failure": COL["light"], "Unknown failure": COL["unknown"], "Known collision/lane": COL["known"]}
    for _, row in tax.iterrows():
        cnt = float(row["count"])
        width = cnt / total
        label = row["failure_display_label"]
        frac_pct = float(row["fraction_pct"])
        col = color_map.get(label, COL["mid"])
        ax_b.barh([0], [width], left=left, height=0.34, color=col, edgecolor="white", linewidth=0.8)
        # label: small groups outside with leader lines
        center = left + width / 2
        if width > 0.08:
            ax_b.text(center, 0, f"{label}\n{int(cnt):,} ({frac_pct:.1f}%)",
                      ha="center", va="center", fontsize=7, color=COL["black"])
        else:
            ytxt = 0.38 if label == "Unknown failure" else -0.38
            ax_b.plot([center, center], [0.18 if ytxt > 0 else -0.18, ytxt*0.75], color=COL["mid"], lw=0.6)
            ax_b.text(center, ytxt, f"{label}\n{int(cnt):,} ({frac_pct:.1f}%)",
                      ha="center", va="center", fontsize=6.7, color=COL["dark"])
        left += width
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.65, 0.65)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Fraction of neutral cohort")
    clean_axis(ax_b, grid="x")
    ax_b.spines["left"].set_visible(False)
    panel_label(ax_b, "b")

    # 6c known-failure AUPRC
    m = metrics.sort_values("display_order")
    y = np.arange(len(m))[::-1]
    for yy, row in zip(y, m.itertuples(index=False)):
        col = SCORE_COLORS.get(row.score_display_label, COL["mid"])
        ax_c.plot([row.ci_low, row.ci_high], [yy, yy], color=col, lw=1.5)
        ax_c.scatter(row.point, yy, s=42, color=col, edgecolor="white", linewidth=0.7, zorder=3)
        # numeric AUPRC values are reported in caption/source data; omit inline labels to keep panel compact
    prev = float(m["positive_rate"].iloc[0])
    ax_c.axvline(prev, color=COL["light"], ls="--", lw=0.9)
    ax_c.text(prev + 0.006, 0.04, "failure prevalence", transform=ax_c.get_xaxis_transform(), fontsize=6, color=COL["mid"], ha="left", va="bottom", rotation=90)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([wrap(x, 20) for x in m["score_display_label"]])
    ax_c.set_xlim(0, max(0.60, m["ci_high"].max() * 1.30))
    ax_c.set_xlabel("AUPRC for known planner failures")
    ax_c.set_title("Known-failure ranking performance", loc="left", pad=7)
    clean_axis(ax_c, grid="x")
    panel_label(ax_c, "c")

    # 6d delta forest by baseline group
    d = delta.sort_values(["baseline_order", "enhanced_order"]).copy()
    rows = []
    yvals = []
    labels = []
    y = len(d) + 1
    current_group = None
    group_positions = {}
    for _, row in d.iterrows():
        if row.comparison_group != current_group:
            current_group = row.comparison_group
            y -= 0.65
            group_positions[current_group] = y + 0.25
        rows.append(row)
        yvals.append(y)
        labels.append(wrap(row.enhanced_display_label, 18))
        y -= 1.0
    for row, yy in zip(rows, yvals):
        col = SCORE_COLORS.get(row.enhanced_display_label, COL["temporal"])
        if bool(row.borderline_ci_crosses_zero):
            face = "white"; edge = col
        else:
            face = col; edge = "white"
        ax_d.plot([row.ci_low, row.ci_high], [yy, yy], color=col, lw=1.4, alpha=0.85)
        ax_d.scatter(row.delta, yy, s=42, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=3)
        ax_d.text(row.ci_high + 0.020, yy, row.delta_label, va="center", fontsize=5.8, color=COL["dark"])
    for group, gy in group_positions.items():
        ax_d.text(0.002, gy, group, fontsize=6.4, color=COL["mid"], ha="left", va="bottom")
    ax_d.axvline(0, color=COL["light"], lw=0.9, ls="--")
    ax_d.set_yticks(yvals)
    ax_d.set_yticklabels(labels)
    ax_d.set_xlim(-0.03, max(0.56, d["ci_high"].max()*1.38))
    ax_d.set_xlabel("ΔAUPRC")
    ax_d.set_title("AUPRC gain over proximity baselines", loc="left", pad=7)
    clean_axis(ax_d, grid="x")
    panel_label(ax_d, "d")
    ax_d.text(0.0, -0.20, "Open marker denotes a borderline interval crossing zero.", transform=ax_d.transAxes,
              ha="left", va="top", fontsize=6.2, color=COL["mid"])

    return save_figure(fig, out_dir, "Figure6_commonroad_neutral_validation_refined", formats, dpi)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_figures(s: str) -> set[int]:
    if s.lower() in {"all", "2-6", "2,3,4,5,6"}:
        return {2, 3, 4, 5, 6}
    figs = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            figs.update(range(int(a), int(b)+1))
        elif part:
            figs.add(int(part))
    valid = {2, 3, 4, 5, 6}
    bad = figs - valid
    if bad:
        raise ValueError(f"Unsupported figure numbers: {sorted(bad)}")
    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refined v100 Figure 2–6 drafts.")
    parser.add_argument("--input", required=True, help="Input redesigned source-data zip or extracted folder.")
    parser.add_argument("--out", default="figures_v100_redesigned", help="Output directory.")
    parser.add_argument("--formats", default="pdf,png,svg", help="Comma-separated output formats, e.g. pdf,png,svg.")
    parser.add_argument("--figures", default="2-6", help="Figures to render, e.g. all, 2-6, 3, 2,3,6.")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for raster outputs.")
    args = parser.parse_args()

    setup_style()
    formats = [x.strip().lower() for x in args.formats.split(',') if x.strip()]
    figures = parse_figures(args.figures)
    out_dir = Path(args.out)
    root, tmp = locate_root(Path(args.input))
    try:
        outputs = {}
        if 2 in figures:
            outputs["Figure2"] = plot_figure2(root, out_dir, formats, args.dpi)
        if 3 in figures:
            outputs["Figure3"] = plot_figure3(root, out_dir, formats, args.dpi)
        if 4 in figures:
            outputs["Figure4"] = plot_figure4(root, out_dir, formats, args.dpi)
        if 5 in figures:
            outputs["Figure5"] = plot_figure5(root, out_dir, formats, args.dpi)
        if 6 in figures:
            outputs["Figure6"] = plot_figure6(root, out_dir, formats, args.dpi)
        manifest = {
            "input": str(Path(args.input).resolve()),
            "located_root": str(root),
            "figures": sorted(list(figures)),
            "formats": formats,
            "dpi": args.dpi,
            "outputs": outputs,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plotting_manifest_refined.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
