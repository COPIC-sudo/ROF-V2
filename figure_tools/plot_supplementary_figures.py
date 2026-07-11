#!/usr/bin/env python3
"""Generate ROF v1.1 Supplementary Figures S6–S9.

The script accepts either the integrated evidence-lock ZIP archive or its
extracted directory. It writes publication-ready PDF/SVG and high-resolution
PNG files together with panel-level derived data and provenance manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

# Palette aligned with the locked main Figures 4–5. It is discrete,
# colour-accessible and deliberately avoids rainbow scales.
BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
GREEN = "#009E73"
TEAL = "#2A9D8F"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#6A51A3"
GOLD = "#C49A00"
GRAY = "#8C8C8C"
DARK_GRAY = "#555555"
LIGHT_GRAY = "#D9D9D9"
VERY_LIGHT_GRAY = "#F3F3F3"
BLACK = "#111111"
RED = "#B2182B"

SCORE_LABELS = {
    "commonroad_crime_risk_score": "CriMe aggregate",
    "HW_inverse": "HW",
    "THW_inverse": "THW",
    "TTC_inverse_crime": "CriMe TTC",
    "ALongReq_mps2": "Required long. accel.",
    "ALatReq_mps2": "Required lat. accel.",
    "rss_danger_score": "RSS aggregate",
    "rss_longitudinal_margin_inverse": "RSS longitudinal margin",
    "rss_lateral_margin_inverse": "RSS lateral margin",
    "drivability_risk_score": "Drivability aggregate",
    "emergency_brake_infeasible_score": "Emergency-brake infeasible",
    "keep_lane_cv_infeasible_score": "Keep-lane CV infeasible",
    "min_collision_time_keep_lane_inverse": "Keep-lane min collision time",
    "min_road_margin_keep_lane_inverse": "Keep-lane road margin",
    "forecast_risk_score": "Forecast aggregate",
    "cv_forecast_collision_risk": "CV collision risk",
    "occupancy_overlap_integral_3s": "Occupancy overlap (3 s)",
    "minimum_predicted_separation_3s_inverse": "Min predicted separation (3 s)",
    "temporal_composite": "Temporal",
    "ROF_v2_no_asr_composite": "ROF-noASR",
    "ROF_v2_composite": "ROF-full",
    "REDI_actionability": "REDI",
    "distance_inverse": "Distance",
    "TTC_inverse": "TTC",
}

FAMILY_LABELS = {
    "reference": "Study/reference scores",
    "commonroad_crime_style": "CriMe-style",
    "rss_style": "RSS-style",
    "drivability": "Drivability",
    "forecast_risk": "Forecast risk",
}

FAMILY_COLORS = {
    "reference": DARK_GRAY,
    "commonroad_crime_style": GOLD,
    "rss_style": PURPLE,
    "drivability": ORANGE,
    "forecast_risk": TEAL,
}

SCORE_COLORS = {
    "temporal_composite": GREEN,
    "ROF_v2_no_asr_composite": BLUE,
    "ROF_v2_composite": LIGHT_BLUE,
    "REDI_actionability": PURPLE,
    "distance_inverse": GRAY,
    "TTC_inverse": "#A7A7A7",
}

FEATURE_SET_LABELS = {
    "strong_baseline_cv": "Base\n(9)",
    "strong_baseline_cv_plus_strict_non_action_current_cv": "+Non-action\n(20)",
    "strong_baseline_cv_plus_strict_temporal_dynamics": "+Temporal\n(26)",
    "strong_baseline_cv_plus_full_actionability": "+Full\n(51)",
}

FEATURE_SET_SHORT = {
    "strong_baseline_cv_plus_strict_non_action_current_cv": "+Non-action",
    "strong_baseline_cv_plus_strict_temporal_dynamics": "+Temporal",
    "strong_baseline_cv_plus_full_actionability": "+Full",
}

VARIANT_LABELS = {
    "h2_buffer3_base7": "H2/B3",
    "h3_buffer3_base7": "H3/B3",
    "h4_buffer3_base7": "H4/B3",
    "h3_buffer2_base7": "H3/B2",
    "h3_buffer4_base7": "H3/B4",
    "h3_buffer3_extended": "H3/Ext",
}

SPEED_LABELS = {"lt5mps": "<5", "5to15mps": "5–15", "gte15mps": "≥15"}
SPEED_ORDER = ["lt5mps", "5to15mps", "gte15mps"]

SUBTYPE_LABELS = {
    "known_failure:collision_and_kinematic": "Collision + kinematic",
    "known_failure:collision_road_boundary_and_kinematic": "Collision + road + kinematic",
    "known_failure:road_boundary_and_kinematic": "Road boundary + kinematic",
}
SUBTYPE_COLORS = {
    "known_failure:collision_and_kinematic": TEAL,
    "known_failure:collision_road_boundary_and_kinematic": "#5B7FAE",
    "known_failure:road_boundary_and_kinematic": GOLD,
}

FIGURE_SOURCE_DIRS = {
    # The evidence lock retains its historical internal numbering. These are
    # deliberately mapped to the manuscript's final S6–S9 numbering.
    "S6": "SuppFig7_commonroad_field_baselines_strict_fpr",
    "S7": "SuppFig8_lattice_extended_sensitivity",
    "S8": "SuppFig9_decoupling_audit",
    "S9": "SuppFig10_low_speed_boundary",
}

OUTPUT_BASENAMES = {
    "S6": "Supplementary_Figure_6_field_baselines",
    "S7": "Supplementary_Figure_7_lattice_extended",
    "S8": "Supplementary_Figure_8_decoupling",
    "S9": "Supplementary_Figure_9_boundary",
}


class PlotDataError(RuntimeError):
    """Raised when a locked input table is absent or structurally invalid."""


@dataclass
class EvidenceLock(AbstractContextManager):
    source: Path
    _tempdir: tempfile.TemporaryDirectory[str] | None = None
    root: Path | None = None

    def __enter__(self) -> "EvidenceLock":
        source = self.source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Evidence lock does not exist: {source}")
        if source.is_file():
            if source.suffix.lower() != ".zip":
                raise ValueError("Evidence-lock file must be a .zip archive")
            self._tempdir = tempfile.TemporaryDirectory(prefix="rof_evidence_lock_")
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(self._tempdir.name)
            search_root = Path(self._tempdir.name)
        else:
            search_root = source
        self.root = self._find_root(search_root)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
        self._tempdir = None
        self.root = None

    @staticmethod
    def _find_root(search_root: Path) -> Path:
        if (search_root / "00_paper_architecture_and_index").is_dir():
            return search_root
        candidates = [p.parent for p in search_root.rglob("00_paper_architecture_and_index") if p.is_dir()]
        if not candidates:
            raise PlotDataError(
                "Could not identify an integrated evidence-lock root containing "
                "00_paper_architecture_and_index"
            )
        # The archive intentionally retains a legacy v100 copy under 99_cleanup_QA.
        # Prefer the shallowest non-legacy candidate.
        candidates = sorted(
            candidates,
            key=lambda p: (
                "99_cleanup_QA" in p.parts or "legacy_v100_reference" in p.parts,
                len(p.relative_to(search_root).parts),
            ),
        )
        return candidates[0]

    def required(self, relative: str | Path) -> Path:
        if self.root is None:
            raise RuntimeError("EvidenceLock must be used as a context manager")
        path = self.root / relative
        if not path.exists():
            raise PlotDataError(f"Required locked artifact is missing: {path}")
        return path

    def supp_dir(self, figure_id: str) -> Path:
        return self.required(
            Path("04_supplementary_figure_source_data") / FIGURE_SOURCE_DIRS[figure_id]
        )

    def base_planner_labels(self) -> Path:
        return self.required(
            "05_reproducibility_and_manifests/v1_1_raw_input_exports/"
            "nc_v110_commonroad_scaleup/full_10k_fixed_taxonomy_lattice_base/planner_labels.csv"
        )

    def extended_planner_labels(self) -> Path:
        return self.required(
            "05_reproducibility_and_manifests/v1_1_raw_input_exports/"
            "nc_v110_commonroad_scaleup/full_10k_lattice_extended_fixed_taxonomy/planner_labels.csv"
        )

    def full_initial_overlap_sensitivity(self) -> Path:
        return self.required(
            "05_reproducibility_and_manifests/v1_1_raw_input_exports/"
            "nc_v110_commonroad_scaleup/full_10k_fixed_taxonomy_lattice_base/"
            "initial_overlap_sensitivity.csv"
        )


def choose_font(preferred: str) -> str:
    candidates = [preferred, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=candidate), fallback_to_default=False
            )
            return candidate
        except Exception:
            continue
    return "DejaVu Sans"


def apply_nature_style(font_family: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_heading(ax: plt.Axes, letter: str, title: str, x: float = -0.13) -> None:
    ax.text(
        x,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )
    ax.set_title(title, loc="left", pad=8.0, fontsize=9.0, fontweight="bold")


def light_x_grid(ax: plt.Axes) -> None:
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.65, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)


def light_y_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.65, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def score_label(score: str) -> str:
    return SCORE_LABELS.get(score, score.replace("_", " "))


def score_color(score: str, family: str | None = None) -> str:
    if score in SCORE_COLORS:
        return SCORE_COLORS[score]
    return FAMILY_COLORS.get(str(family), DARK_GRAY)


def require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise PlotDataError(f"{source} is missing required columns: {missing}")


def read_csv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if columns is not None:
        require_columns(df, columns, path)
    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    basename: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    outputs: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower()
        path = output_dir / f"{basename}.{fmt}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035, "facecolor": "white"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_derived(df: pd.DataFrame, path: Path, enabled: bool) -> None:
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def format_delta(value: float) -> str:
    return f"{value:+.3f}"


def plot_s6(
    lock: EvidenceLock,
    output_dir: Path,
    derived_dir: Path,
    formats: Sequence[str],
    dpi: int,
    write_data: bool,
) -> tuple[list[Path], list[Path]]:
    source_dir = lock.supp_dir("S6")
    metrics_path = source_dir / "field_baseline_metrics_strict_fpr.csv"
    bootstrap_path = source_dir / "field_baseline_bootstrap_deltas_strict_fpr.csv"
    warning_path = source_dir / "metric_warning_table.csv"
    metrics = read_csv(
        metrics_path,
        ["score", "AUPRC", "Recall@5%FPR_strict", "score_family", "n", "positive_count"],
    )
    bootstrap = read_csv(
        bootstrap_path,
        ["enhanced_score", "baseline_score", "metric", "pairwise_n", "pairwise_positive_count", "n_samples"],
    )
    warnings = read_csv(
        warning_path,
        ["score", "legacy_metric", "target_fpr", "legacy_actual_fpr", "legacy_recall"],
    )
    if len(metrics) != 24 or int(metrics["positive_count"].max()) != 610:
        raise PlotDataError("S6 lock check failed: expected 24 scores and 610 positives")

    metrics = metrics.copy()
    metrics["display"] = metrics["score"].map(score_label)
    metrics["colour"] = [score_color(s, f) for s, f in zip(metrics["score"], metrics["score_family"])]
    metrics = metrics.sort_values("AUPRC", ascending=False).reset_index(drop=True)
    metrics["rank"] = np.arange(len(metrics))

    pairwise = bootstrap[
        (bootstrap["metric"] == "auprc")
        & (bootstrap["enhanced_score"] == "temporal_composite")
    ].drop_duplicates("baseline_score")
    max_n = int(pairwise["n_samples"].max())
    incomplete = pairwise[pairwise["pairwise_n"] < max_n].copy()
    incomplete["display"] = incomplete["baseline_score"].map(score_label)
    incomplete = incomplete.sort_values("pairwise_n")

    warn5 = warnings[np.isclose(warnings["target_fpr"], 0.05)].copy()
    warn5["overshoot_ratio"] = warn5["legacy_actual_fpr"] / warn5["target_fpr"]
    warn5["display"] = warn5["score"].map(score_label)
    warn_top = warn5.nlargest(8, "overshoot_ratio").sort_values("overshoot_ratio")

    write_derived(metrics.drop(columns="colour"), derived_dir / "S6_panel_ab_score_ranking.csv", write_data)
    write_derived(incomplete, derived_dir / "S6_panel_c_pairwise_incomplete_scores.csv", write_data)
    write_derived(warn_top, derived_dir / "S6_panel_d_largest_legacy_fpr_overshoots.csv", write_data)

    fig = plt.figure(figsize=(7.18, 9.15), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.45, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.255,
        right=0.985,
        top=0.90,
        bottom=0.07,
        wspace=0.18,
        hspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    y = metrics["rank"].to_numpy()
    for ax, value_col, xlabel, letter, title in [
        (ax_a, "AUPRC", "AUPRC", "a", "Complete AUPRC ranking"),
        (ax_b, "Recall@5%FPR_strict", "Strict recall at 5% FPR", "b", "Strict fixed-FPR ranking"),
    ]:
        vals = metrics[value_col].to_numpy()
        ax.hlines(y, 0, vals, color="#E5E5E5", linewidth=0.65, zorder=1)
        ax.scatter(vals, y, c=metrics["colour"], s=25, edgecolor="white", linewidth=0.35, zorder=3)
        ax.set_ylim(-0.7, len(metrics) - 0.3)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
        light_x_grid(ax)
        panel_heading(ax, letter, title)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(metrics["display"], fontsize=5.7)
    ax_a.axvline(610 / 10000, color="#BDBDBD", linestyle="--", linewidth=0.75, zorder=0)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_b.tick_params(axis="y", length=0)

    family_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=FAMILY_COLORS[k], label=FAMILY_LABELS[k], markersize=4.5)
        for k in ["commonroad_crime_style", "rss_style", "drivability", "forecast_risk"]
    ]
    family_handles += [
        Line2D([0], [0], marker="o", linestyle="none", color=GREEN, label="Temporal", markersize=4.5),
        Line2D([0], [0], marker="o", linestyle="none", color=BLUE, label="ROF-noASR", markersize=4.5),
    ]
    fig.legend(
        handles=family_handles,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.987),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.3,
        fontsize=5.8,
    )

    panel_heading(ax_c, "c", "Pairwise common-row evaluation")
    if incomplete.empty:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "All scores use the complete cohort.", ha="center", va="center")
    else:
        yc = np.arange(len(incomplete))
        ax_c.barh(yc, incomplete["pairwise_n"], color=LIGHT_BLUE, edgecolor="none", height=0.55)
        ax_c.set_yticks(yc)
        ax_c.set_yticklabels(incomplete["display"], fontsize=6.2)
        ax_c.invert_yaxis()
        lower = max(0, int(incomplete["pairwise_n"].min()) - 250)
        ax_c.set_xlim(lower, max_n + 90)
        ax_c.axvline(max_n, color=DARK_GRAY, linestyle="--", linewidth=0.8)
        ax_c.set_xlabel("Pairwise samples")
        light_x_grid(ax_c)
        for yi, (_, row) in enumerate(incomplete.iterrows()):
            ax_c.text(
                row["pairwise_n"] + 20,
                yi,
                f"n={int(row['pairwise_n']):,}; positives={int(row['pairwise_positive_count'])}",
                va="center",
                ha="left",
                fontsize=6.0,
            )
        ax_c.text(
            0.02,
            -0.22,
            "All other comparisons: n=10,000; positives=610.",
            transform=ax_c.transAxes,
            fontsize=5.8,
            color=DARK_GRAY,
        )

    panel_heading(ax_d, "d", "Largest legacy FPR overshoots")
    yd = np.arange(len(warn_top))
    ratios = warn_top["overshoot_ratio"].to_numpy()
    severity_colors = [VERMILION if r >= 1.5 else ORANGE if r >= 1.1 else GRAY for r in ratios]
    ax_d.hlines(yd, 1, ratios, color="#D9D9D9", linewidth=1.0)
    ax_d.scatter(ratios, yd, c=severity_colors, s=27, edgecolor="white", linewidth=0.35, zorder=3)
    ax_d.axvline(1.0, color=BLACK, linewidth=0.85, linestyle="--")
    ax_d.set_xscale("log")
    ax_d.set_xlim(0.95, max(16.0, ratios.max() * 1.15))
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels(warn_top["display"], fontsize=5.8)
    ax_d.set_xlabel("Legacy actual FPR / target FPR (target = 5%)")
    light_x_grid(ax_d)
    for yi, (_, row) in enumerate(warn_top.iterrows()):
        ax_d.text(
            row["overshoot_ratio"] * 1.06,
            yi,
            f"{100 * row['legacy_actual_fpr']:.1f}%",
            va="center",
            ha="left",
            fontsize=5.7,
        )
    ax_d.text(
        0.02,
        -0.22,
        "Strict metrics restrict thresholds to empirical FPR ≤ target; legacy rows are retained only for audit.",
        transform=ax_d.transAxes,
        fontsize=5.6,
        color=DARK_GRAY,
    )

    outputs = save_figure(fig, output_dir, OUTPUT_BASENAMES["S6"], formats, dpi)
    return outputs, [metrics_path, bootstrap_path, warning_path]


def _metric_value(metrics: pd.DataFrame, name: str) -> float:
    row = metrics[metrics["score"] == name]
    if len(row) != 1:
        raise PlotDataError(f"Expected exactly one metric row for {name}")
    return float(row.iloc[0]["AUPRC"])


def plot_s7(
    lock: EvidenceLock,
    output_dir: Path,
    derived_dir: Path,
    formats: Sequence[str],
    dpi: int,
    write_data: bool,
) -> tuple[list[Path], list[Path]]:
    source_dir = lock.supp_dir("S7")
    metrics_path = source_dir / "external_metrics_strict_fpr.csv"
    boot_path = source_dir / "external_bootstrap_deltas_strict_fpr.csv"
    agreement_path = source_dir / "label_agreement_with_lattice_base.csv"
    field_boot_path = source_dir / "field_baseline_bootstrap_deltas_strict_fpr.csv"
    base_labels_path = lock.base_planner_labels()
    ext_labels_path = lock.extended_planner_labels()

    metrics = read_csv(metrics_path, ["score", "AUPRC", "Recall@5%FPR_strict", "n", "positive_count"])
    boot = read_csv(boot_path, ["enhanced_score", "baseline_score", "metric", "delta", "ci_low", "ci_high"])
    agreement = read_csv(agreement_path, ["metric", "value"])
    field_boot = read_csv(field_boot_path, ["enhanced_score", "baseline_score", "metric", "delta", "ci_low", "ci_high"])
    base_labels = read_csv(base_labels_path, ["sample_id", "known_failure", "candidate_count", "feasible_candidate_count", "feasible_candidate_ratio"])
    ext_labels = read_csv(ext_labels_path, ["sample_id", "known_failure", "candidate_count", "feasible_candidate_count", "feasible_candidate_ratio"])

    a = agreement.set_index("metric")["value"].to_dict()
    expected = {
        "matched_samples": 10000,
        "base_known_failure_count": 610,
        "extended_known_failure_count": 420,
        "base_only_positives": 190,
        "extended_only_positives": 0,
    }
    for key, value in expected.items():
        if not math.isclose(float(a.get(key, np.nan)), value, rel_tol=0, abs_tol=1e-8):
            raise PlotDataError(f"S7 lock check failed for {key}: {a.get(key)} != {value}")

    confusion = pd.DataFrame(
        [
            {"base": "No failure", "extended": "No failure", "count": int(a["both_not_positive"])},
            {"base": "No failure", "extended": "Known failure", "count": int(a["extended_only_positives"])},
            {"base": "Known failure", "extended": "No failure", "count": int(a["base_only_positives"])},
            {"base": "Known failure", "extended": "Known failure", "count": int(a["both_positive"])},
        ]
    )
    write_derived(confusion, derived_dir / "S7_panel_a_endpoint_confusion.csv", write_data)

    metric_order = [
        "ROF_v2_no_asr_composite",
        "temporal_composite",
        "ROF_v2_composite",
        "REDI_actionability",
        "TTC_inverse",
        "distance_inverse",
    ]
    point = metrics.set_index("score").loc[metric_order].reset_index()
    point["display"] = point["score"].map(score_label)
    write_derived(point, derived_dir / "S7_panel_c_extended_point_metrics.csv", write_data)

    diag_summary = pd.DataFrame(
        [
            {
                "library": "Base",
                "candidate_count": int(base_labels["candidate_count"].iloc[0]),
                "mean_feasible_count": base_labels["feasible_candidate_count"].mean(),
                "median_feasible_ratio": base_labels["feasible_candidate_ratio"].median(),
                "zero_feasible_fraction": (base_labels["feasible_candidate_count"] == 0).mean(),
            },
            {
                "library": "Extended",
                "candidate_count": int(ext_labels["candidate_count"].iloc[0]),
                "mean_feasible_count": ext_labels["feasible_candidate_count"].mean(),
                "median_feasible_ratio": ext_labels["feasible_candidate_ratio"].median(),
                "zero_feasible_fraction": (ext_labels["feasible_candidate_count"] == 0).mean(),
            },
        ]
    )
    write_derived(diag_summary, derived_dir / "S7_panel_d_candidate_diagnostics.csv", write_data)

    # Full AUPRC field-baseline delta matrix is written as derived data even
    # though panel c presents point metrics; it supports caption/table checks.
    field_auprc = field_boot[field_boot["metric"] == "auprc"].copy()
    write_derived(field_auprc, derived_dir / "S7_full_extended_field_baseline_auprc_deltas.csv", write_data)

    fig = plt.figure(figsize=(7.18, 7.65), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.985,
        top=0.965,
        bottom=0.075,
        wspace=0.34,
        hspace=0.38,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_heading(ax_a, "a", "Base versus extended endpoint")
    matrix = np.array(
        [
            [a["both_not_positive"], a["extended_only_positives"]],
            [a["base_only_positives"], a["both_positive"]],
        ],
        dtype=float,
    )
    im = ax_a.imshow(matrix, cmap="Blues", norm=Normalize(vmin=0, vmax=matrix.max()), aspect="auto")
    ax_a.set_xticks([0, 1], ["No failure", "Known failure"])
    ax_a.set_yticks([0, 1], ["No failure", "Known failure"])
    ax_a.set_xlabel("Extended endpoint")
    ax_a.set_ylabel("Base endpoint")
    row_sums = matrix.sum(axis=1)
    for i in range(2):
        for j in range(2):
            value = int(matrix[i, j])
            pct = 100 * matrix[i, j] / row_sums[i] if row_sums[i] else 0
            colour = "white" if matrix[i, j] > 0.45 * matrix.max() else BLACK
            ax_a.text(j, i, f"{value:,}\n{pct:.1f}% of row", ha="center", va="center", color=colour, fontsize=7.0)
    for spine in ax_a.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#BDBDBD")
    ax_a.text(
        0.5,
        -0.25,
        f"Overall taxonomy agreement: {100 * a['taxonomy_agreement_rate']:.1f}%",
        transform=ax_a.transAxes,
        ha="center",
        fontsize=6.2,
        color=DARK_GRAY,
    )

    panel_heading(ax_b, "b", "Failure-status transitions")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    left_box = FancyBboxPatch((0.04, 0.49), 0.34, 0.28, boxstyle="round,pad=0.02", fc="#EAF2F8", ec="#7AA6CE", lw=1.0)
    right_box = FancyBboxPatch((0.62, 0.49), 0.34, 0.28, boxstyle="round,pad=0.02", fc="#EAF6F2", ec="#7FB7A6", lw=1.0)
    ax_b.add_patch(left_box)
    ax_b.add_patch(right_box)
    ax_b.text(0.21, 0.64, "610", ha="center", va="center", fontsize=19, fontweight="bold")
    ax_b.text(0.21, 0.52, "base known failures", ha="center", va="center", fontsize=6.5)
    ax_b.text(0.79, 0.64, "420", ha="center", va="center", fontsize=19, fontweight="bold")
    ax_b.text(0.79, 0.52, "extended known failures", ha="center", va="center", fontsize=6.5)
    arrow = FancyArrowPatch((0.39, 0.63), (0.61, 0.63), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=DARK_GRAY)
    ax_b.add_patch(arrow)
    ax_b.text(0.50, 0.69, "190 rescued", ha="center", va="center", fontsize=6.5, color=DARK_GRAY)
    stats = [
        (0.08, "98.1%", "agreement"),
        (0.34, "+190", "base-only"),
        (0.62, "0", "extended-only"),
        (0.87, "0", "unknown"),
    ]
    for x, val, lab in stats:
        ax_b.text(x, 0.29, val, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax_b.text(x, 0.20, lab, ha="center", va="center", fontsize=6.2, color=DARK_GRAY)
    ax_b.text(0.5, 0.04, "Same 10,000-sample outcome-blind cohort", ha="center", fontsize=6.4, color=DARK_GRAY)

    panel_heading(ax_c, "c", "Extended-label point performance")
    yc = np.arange(len(point))
    ax_c.hlines(yc, 0, np.maximum(point["AUPRC"], point["Recall@5%FPR_strict"]), color="#E5E5E5", lw=0.7)
    ax_c.scatter(point["AUPRC"], yc - 0.10, color=[score_color(s) for s in point["score"]], marker="o", s=27, label="AUPRC", zorder=3)
    ax_c.scatter(point["Recall@5%FPR_strict"], yc + 0.10, facecolor="white", edgecolor=[score_color(s) for s in point["score"]], marker="s", s=25, linewidth=1.0, label="Strict recall at 5% FPR", zorder=3)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(point["display"], fontsize=6.4)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 0.52)
    ax_c.set_xlabel("Metric value")
    light_x_grid(ax_c)
    ax_c.legend(frameon=False, loc="lower right", handletextpad=0.4)

    panel_heading(ax_d, "d", "Candidate-library diagnostics")
    base_x, base_y = empirical_cdf(base_labels["feasible_candidate_ratio"].to_numpy(float))
    ext_x, ext_y = empirical_cdf(ext_labels["feasible_candidate_ratio"].to_numpy(float))
    ax_d.plot(base_x, base_y, color=BLUE, lw=1.5, label="Base: 35 candidates")
    ax_d.plot(ext_x, ext_y, color=GREEN, lw=1.5, label="Extended: 130 candidates")
    ax_d.set_xlabel("Feasible-candidate ratio")
    ax_d.set_ylabel("Empirical cumulative fraction")
    ax_d.set_xlim(0, 0.9)
    ax_d.set_ylim(0, 1)
    light_y_grid(ax_d)
    ax_d.legend(frameon=False, loc="lower right")
    base_med = float(base_labels["feasible_candidate_ratio"].median())
    ext_med = float(ext_labels["feasible_candidate_ratio"].median())
    ax_d.axvline(base_med, color=BLUE, linestyle="--", lw=0.8)
    ax_d.axvline(ext_med, color=GREEN, linestyle="--", lw=0.8)
    ax_d.text(
        0.02,
        0.96,
        f"Median feasible ratio: {base_med:.3f} → {ext_med:.3f}\n"
        f"Zero-feasible fraction: {100*(base_labels.feasible_candidate_count==0).mean():.1f}% → "
        f"{100*(ext_labels.feasible_candidate_count==0).mean():.1f}%",
        transform=ax_d.transAxes,
        va="top",
        fontsize=6.0,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#D0D0D0", lw=0.6),
    )

    outputs = save_figure(fig, output_dir, OUTPUT_BASENAMES["S7"], formats, dpi)
    inputs = [metrics_path, boot_path, agreement_path, field_boot_path, base_labels_path, ext_labels_path]
    return outputs, inputs


def plot_s8(
    lock: EvidenceLock,
    output_dir: Path,
    derived_dir: Path,
    formats: Sequence[str],
    dpi: int,
    write_data: bool,
) -> tuple[list[Path], list[Path]]:
    source_dir = lock.supp_dir("S8")
    point_path = source_dir / "non_action_feature_oof_metrics.csv"
    delta_path = source_dir / "non_action_feature_bootstrap_deltas.csv"
    mismatch_path = source_dir / "label_feature_mismatch_bootstrap.csv"
    lineage_path = source_dir / "feature_lineage_v111.csv"

    point = read_csv(point_path, ["feature_set", "n_features", "AUPRC", "Recall@5%FPR_strict", "model"])
    deltas = read_csv(delta_path, ["metric", "enhanced_feature_set", "delta", "ci_low", "ci_high", "model"])
    mismatch = read_csv(mismatch_path, ["metric", "delta", "ci_low", "ci_high", "label_variant", "feature_variant", "diagonal"])
    lineage = read_csv(lineage_path, ["feature_name", "reads_recorded_future", "reads_label", "uses_action_library", "uses_candidate_survival"])

    expected_sets = list(FEATURE_SET_LABELS)
    if set(point["feature_set"]) != set(expected_sets) or set(point["model"]) != {"rf"}:
        raise PlotDataError("S8 lock check failed: expected four RF feature sets")
    auprc_mismatch = mismatch[mismatch["metric"] == "auprc"].copy()
    if len(auprc_mismatch) != 36:
        raise PlotDataError("S8 lock check failed: expected 36 AUPRC mismatch cells")

    point = point.set_index("feature_set").loc[expected_sets].reset_index()
    point["display"] = point["feature_set"].map(FEATURE_SET_LABELS)
    write_derived(point, derived_dir / "S8_panel_a_feature_set_point_metrics.csv", write_data)

    delta_plot = deltas.copy()
    delta_plot["display"] = delta_plot["enhanced_feature_set"].map(FEATURE_SET_SHORT)
    write_derived(delta_plot, derived_dir / "S8_panel_b_feature_set_bootstrap_deltas.csv", write_data)

    distribution = auprc_mismatch.assign(group=np.where(auprc_mismatch["diagonal"], "Diagonal", "Off-diagonal"))
    write_derived(distribution, derived_dir / "S8_panel_c_mismatch_gain_distribution.csv", write_data)

    weak = auprc_mismatch[~auprc_mismatch["diagonal"]].nsmallest(8, "delta").copy()
    weak["comparison"] = [
        f"{VARIANT_LABELS.get(l,l)} → {VARIANT_LABELS.get(f,f)}"
        for l, f in zip(weak["label_variant"], weak["feature_variant"])
    ]
    weak = weak.sort_values("delta", ascending=True)
    write_derived(weak, derived_dir / "S8_panel_d_weakest_off_diagonal_transfers.csv", write_data)

    lineage_summary = pd.DataFrame(
        {
            "quantity": [
                "features",
                "reads_recorded_future_true",
                "reads_label_true",
                "uses_action_library_true",
                "uses_candidate_survival_true",
            ],
            "value": [
                len(lineage),
                int(lineage["reads_recorded_future"].sum()),
                int(lineage["reads_label"].sum()),
                int(lineage["uses_action_library"].sum()),
                int(lineage["uses_candidate_survival"].sum()),
            ],
        }
    )
    write_derived(lineage_summary, derived_dir / "S8_feature_lineage_summary.csv", write_data)

    fig = plt.figure(figsize=(7.18, 7.15), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.985,
        top=0.965,
        bottom=0.08,
        wspace=0.34,
        hspace=0.40,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_heading(ax_a, "a", "Nested feature-set performance")
    x = np.arange(len(point))
    colours = [GRAY, GRAY, GREEN, BLUE]
    ax_a.plot(x, point["AUPRC"], color="#8F8F8F", lw=1.0, zorder=1)
    ax_a.scatter(x, point["AUPRC"], c=colours, s=31, marker="o", zorder=3, label="AUPRC")
    ax_a.plot(x, point["Recall@5%FPR_strict"], color="#B5B5B5", lw=0.9, linestyle="--", zorder=1)
    ax_a.scatter(x, point["Recall@5%FPR_strict"], facecolor="white", edgecolor=colours, s=29, marker="s", linewidth=1.1, zorder=3, label="Strict recall at 5% FPR")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(point["display"], fontsize=6.1)
    ax_a.set_ylabel("Metric value")
    ax_a.set_ylim(0.24, 0.53)
    light_y_grid(ax_a)
    ax_a.legend(frameon=False, loc="lower right", handletextpad=0.4)
    for xi, val in enumerate(point["AUPRC"]):
        ax_a.text(xi, val + 0.009, f"{val:.3f}", ha="center", fontsize=5.8)

    panel_heading(ax_b, "b", "Paired-bootstrap feature gains")
    sets = [
        "strong_baseline_cv_plus_strict_non_action_current_cv",
        "strong_baseline_cv_plus_strict_temporal_dynamics",
        "strong_baseline_cv_plus_full_actionability",
    ]
    yb = np.arange(len(sets))
    for metric, marker, offset, colour, label in [
        ("auprc", "o", -0.10, BLUE, "ΔAUPRC"),
        ("recall_at_5pct_fpr_strict", "s", 0.10, GREEN, "Δ strict recall"),
    ]:
        sub = deltas[deltas["metric"] == metric].set_index("enhanced_feature_set").loc[sets]
        vals = sub["delta"].to_numpy()
        xerr = np.vstack([vals - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - vals])
        ax_b.errorbar(vals, yb + offset, xerr=xerr, fmt=marker, color=colour, ecolor=colour, capsize=2.2, lw=1.0, label=label)
    ax_b.axvline(0, color="#7A7A7A", lw=0.75)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([FEATURE_SET_SHORT[s] for s in sets], fontsize=6.4)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Gain over 9-field base set")
    ax_b.set_xlim(-0.005, 0.18)
    light_x_grid(ax_b)
    ax_b.legend(frameon=False, loc="lower left")

    panel_heading(ax_c, "c", "Diagonal versus off-diagonal transfer")
    rng = np.random.default_rng(20260710)
    groups = ["Diagonal", "Off-diagonal"]
    values = [distribution.loc[distribution["group"] == g, "delta"].to_numpy() for g in groups]
    bp = ax_c.boxplot(
        values,
        positions=[0, 1],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=BLACK, linewidth=1.2),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, colour in zip(bp["boxes"], [LIGHT_BLUE, "#F1C6B7"]):
        patch.set_facecolor(colour)
        patch.set_edgecolor(DARK_GRAY)
    for i, vals in enumerate(values):
        jitter = rng.uniform(-0.09, 0.09, size=len(vals))
        ax_c.scatter(np.full(len(vals), i) + jitter, vals, s=15, color=[BLUE, VERMILION][i], alpha=0.78, edgecolor="white", linewidth=0.25, zorder=3)
    ax_c.axhline(0, color="#7A7A7A", lw=0.75)
    ax_c.set_xticks([0, 1], groups)
    ax_c.set_ylabel("ΔAUPRC")
    light_y_grid(ax_c)
    diag_med = float(np.median(values[0]))
    off_med = float(np.median(values[1]))
    off_pos = float(np.mean(values[1] > 0))
    note_box = dict(boxstyle="round,pad=0.20", fc="white", ec="#D0D0D0", lw=0.5, alpha=0.92)
    ax_c.text(0.02, 0.98, f"Diagonal median {diag_med:+.3f}", transform=ax_c.transAxes, va="top", fontsize=5.8, bbox=note_box)
    ax_c.text(0.52, 0.98, f"Off-diagonal median {off_med:+.3f}\n{100*off_pos:.1f}% positive", transform=ax_c.transAxes, va="top", fontsize=5.8, bbox=note_box)

    panel_heading(ax_d, "d", "Weakest off-diagonal AUPRC transfers")
    yd = np.arange(len(weak))
    vals = weak["delta"].to_numpy()
    xerr = np.vstack([vals - weak["ci_low"].to_numpy(), weak["ci_high"].to_numpy() - vals])
    colours_d = [RED if hi < 0 else ORANGE if lo <= 0 <= hi else TEAL for lo, hi in zip(weak["ci_low"], weak["ci_high"])]
    for i in range(len(weak)):
        ax_d.errorbar(vals[i], yd[i], xerr=np.array([[vals[i]-weak.iloc[i]["ci_low"]], [weak.iloc[i]["ci_high"]-vals[i]]]), fmt="o", color=colours_d[i], ecolor=colours_d[i], capsize=2.0, lw=1.0)
    ax_d.axvline(0, color="#7A7A7A", lw=0.8)
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels(weak["comparison"], fontsize=5.7)
    ax_d.set_xlabel("ΔAUPRC")
    light_x_grid(ax_d)
    ax_d.text(
        0.02,
        -0.24,
        "Label → feature variant. Segment/family leave-out was not available in the locked metadata.",
        transform=ax_d.transAxes,
        fontsize=5.5,
        color=DARK_GRAY,
    )

    outputs = save_figure(fig, output_dir, OUTPUT_BASENAMES["S8"], formats, dpi)
    return outputs, [point_path, delta_path, mismatch_path, lineage_path]


def _speed_stratum_from_mps(value: float) -> str:
    if value < 5:
        return "lt5mps"
    if value < 15:
        return "5to15mps"
    return "gte15mps"


def plot_s9(
    lock: EvidenceLock,
    output_dir: Path,
    derived_dir: Path,
    formats: Sequence[str],
    dpi: int,
    write_data: bool,
) -> tuple[list[Path], list[Path]]:
    source_dir = lock.supp_dir("S9")
    speed_path = source_dir / "speed_stratum_metrics_bootstrap.csv"
    overlap_path = source_dir / "initial_overlap_by_stratum.csv"
    subtype_path = source_dir / "low_speed_failure_subtype_summary.csv"
    negative_path = source_dir / "neutral_stratum_negative_delta_audit.csv"
    labels_path = lock.base_planner_labels()
    full_overlap_path = lock.full_initial_overlap_sensitivity()

    speed = read_csv(speed_path, ["row_type", "stratum", "enhanced_score", "baseline_score", "metric", "delta", "ci_low", "ci_high"])
    overlap = read_csv(overlap_path, ["stratum_column", "stratum", "enhanced_score", "baseline_score", "AUPRC_delta_all", "AUPRC_delta_exclude_initial_overlap"])
    low_subtypes = read_csv(subtype_path, ["failure_subtype", "count", "speed_stratum"])
    negative = read_csv(negative_path, ["stratum", "negative_delta", "clearly_negative_delta", "low_speed_fraction", "collision_positive_fraction"])
    labels = read_csv(labels_path, ["known_failure", "failure_subtype", "ego_speed_mps", "initial_overlap_count", "collision_flag"])
    full_overlap = read_csv(full_overlap_path, ["score", "AUPRC", "sensitivity_mode"])

    bootstrap = speed[(speed["row_type"] == "bootstrap_delta") & (speed["metric"] == "auprc")].copy()
    if set(bootstrap["stratum"].dropna()) != set(SPEED_ORDER):
        raise PlotDataError("S9 lock check failed: expected three speed strata")

    labels = labels.copy()
    labels["speed_stratum"] = labels["ego_speed_mps"].map(_speed_stratum_from_mps)
    failures = labels[labels["known_failure"] == 1].copy()
    subtype_counts = (
        failures.groupby(["speed_stratum", "failure_subtype"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    subtype_counts = subtype_counts[subtype_counts["failure_subtype"].isin(SUBTYPE_LABELS)].copy()
    write_derived(bootstrap, derived_dir / "S9_panel_a_speed_stratum_auprc_deltas.csv", write_data)
    write_derived(subtype_counts, derived_dir / "S9_panel_b_failure_subtypes_by_speed.csv", write_data)

    speed_diag_rows = []
    for stratum in SPEED_ORDER:
        g = labels[labels["speed_stratum"] == stratum]
        gp = g[g["known_failure"] == 1]
        speed_diag_rows.append(
            {
                "speed_stratum": stratum,
                "n": len(g),
                "positive_count": len(gp),
                "initial_overlap_fraction_all": float((g["initial_overlap_count"] > 0).mean()),
                "collision_positive_fraction": float(gp["collision_flag"].mean()) if len(gp) else np.nan,
            }
        )
    speed_diag = pd.DataFrame(speed_diag_rows)
    write_derived(speed_diag, derived_dir / "S9_panel_c_overlap_collision_by_speed.csv", write_data)

    low_rows = overlap[(overlap["stratum_column"] == "speed_stratum") & (overlap["stratum"] == "lt5mps")].copy()
    low_rows["scope"] = "Low speed"
    low_rows["all"] = low_rows["AUPRC_delta_all"]
    low_rows["exclude_overlap"] = low_rows["AUPRC_delta_exclude_initial_overlap"]

    full_pivot = full_overlap.pivot_table(index="score", columns="sensitivity_mode", values="AUPRC", aggfunc="first")
    full_rows = []
    for enhanced in ["temporal_composite", "ROF_v2_no_asr_composite"]:
        for baseline in ["distance_inverse", "TTC_inverse"]:
            full_rows.append(
                {
                    "scope": "Full cohort",
                    "enhanced_score": enhanced,
                    "baseline_score": baseline,
                    "all": float(full_pivot.loc[enhanced, "all_primary"] - full_pivot.loc[baseline, "all_primary"]),
                    "exclude_overlap": float(
                        full_pivot.loc[enhanced, "exclude_initial_overlap_count_gt0"]
                        - full_pivot.loc[baseline, "exclude_initial_overlap_count_gt0"]
                    ),
                }
            )
    sensitivity = pd.concat(
        [
            low_rows[["scope", "enhanced_score", "baseline_score", "all", "exclude_overlap"]],
            pd.DataFrame(full_rows),
        ],
        ignore_index=True,
    )
    sensitivity["comparison"] = [
        f"{'T' if e == 'temporal_composite' else 'R'} · {'Distance' if b == 'distance_inverse' else 'TTC'}"
        for e, b in zip(sensitivity["enhanced_score"], sensitivity["baseline_score"])
    ]
    write_derived(sensitivity, derived_dir / "S9_panel_d_initial_overlap_sensitivity.csv", write_data)

    fig = plt.figure(figsize=(7.18, 7.60), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.985,
        top=0.965,
        bottom=0.085,
        wspace=0.35,
        hspace=0.56,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c_parent = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_heading(ax_a, "a", "Speed-stratum AUPRC gains")
    series = [
        ("temporal_composite", "distance_inverse", GREEN, "o", "T · Distance"),
        ("temporal_composite", "TTC_inverse", GREEN, "s", "T · TTC"),
        ("ROF_v2_no_asr_composite", "distance_inverse", BLUE, "o", "R · Distance"),
        ("ROF_v2_no_asr_composite", "TTC_inverse", BLUE, "s", "R · TTC"),
    ]
    x = np.arange(3)
    offsets = [-0.21, -0.07, 0.07, 0.21]
    for (enh, base, colour, marker, label), offset in zip(series, offsets):
        sub = bootstrap[(bootstrap["enhanced_score"] == enh) & (bootstrap["baseline_score"] == base)].set_index("stratum").loc[SPEED_ORDER]
        vals = sub["delta"].to_numpy()
        yerr = np.vstack([vals - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - vals])
        ax_a.errorbar(x + offset, vals, yerr=yerr, fmt=marker, color=colour, ecolor=colour, capsize=2.0, lw=0.95, label=label)
    ax_a.axhline(0, color="#6F6F6F", lw=0.8)
    ax_a.set_xticks(x, [SPEED_LABELS[s] for s in SPEED_ORDER])
    ax_a.set_xlabel("Ego speed stratum (m s$^{-1}$)")
    ax_a.set_ylabel("ΔAUPRC")
    light_y_grid(ax_a)
    ax_a.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.8, handletextpad=0.35)

    panel_heading(ax_b, "b", "Failure-subtype composition")
    pivot = subtype_counts.pivot_table(index="speed_stratum", columns="failure_subtype", values="count", fill_value=0, aggfunc="sum").reindex(SPEED_ORDER)
    totals = pivot.sum(axis=1)
    left = np.zeros(len(pivot))
    for subtype in SUBTYPE_LABELS:
        vals = pivot.get(subtype, pd.Series(0, index=pivot.index)).to_numpy(float)
        frac = np.divide(vals, totals.to_numpy(float), out=np.zeros_like(vals), where=totals.to_numpy(float) > 0)
        ax_b.barh(np.arange(3), frac, left=left, color=SUBTYPE_COLORS[subtype], height=0.55, label=SUBTYPE_LABELS[subtype])
        left += frac
    ax_b.set_yticks(np.arange(3), [SPEED_LABELS[s] for s in SPEED_ORDER])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("Fraction of known failures")
    ax_b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    light_x_grid(ax_b)
    for yi, total in enumerate(totals):
        ax_b.text(1.015, yi, f"n={int(total)}", va="center", fontsize=5.8)
    ax_b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.50, -0.27), ncol=1, fontsize=5.6)

    ax_c_parent.axis("off")
    panel_heading(ax_c_parent, "c", "Overlap and collision composition", x=-0.13)
    ax_c1 = ax_c_parent.inset_axes([0.00, 0.02, 0.46, 0.82])
    ax_c2 = ax_c_parent.inset_axes([0.55, 0.02, 0.45, 0.82])
    yy = np.arange(3)
    ax_c1.barh(yy, 100 * speed_diag.set_index("speed_stratum").loc[SPEED_ORDER, "initial_overlap_fraction_all"], color=ORANGE, height=0.50)
    ax_c1.set_yticks(yy, [SPEED_LABELS[s] for s in SPEED_ORDER])
    ax_c1.invert_yaxis()
    ax_c1.set_xlabel("Initial overlap\n(% of all samples)", fontsize=6.2)
    light_x_grid(ax_c1)
    for yi, val in enumerate(100 * speed_diag.set_index("speed_stratum").loc[SPEED_ORDER, "initial_overlap_fraction_all"]):
        ax_c1.text(val + 0.05, yi, f"{val:.2f}%", va="center", fontsize=5.6)
    coll = 100 * speed_diag.set_index("speed_stratum").loc[SPEED_ORDER, "collision_positive_fraction"]
    ax_c2.barh(yy, coll, color=TEAL, height=0.50)
    ax_c2.set_yticks(yy)
    ax_c2.set_yticklabels([])
    ax_c2.invert_yaxis()
    ax_c2.set_xlim(0, 105)
    ax_c2.set_xlabel("Collision flag\n(% of positives)", fontsize=6.2)
    light_x_grid(ax_c2)
    for yi, val in enumerate(coll):
        ax_c2.text(val + 1.0, yi, f"{val:.1f}%", va="center", fontsize=5.6)

    panel_heading(ax_d, "d", "Initial-overlap exclusion")
    rows = []
    for scope in ["Low speed", "Full cohort"]:
        sub = sensitivity[sensitivity["scope"] == scope].copy()
        order = ["T · Distance", "T · TTC", "R · Distance", "R · TTC"]
        sub = sub.set_index("comparison").loc[order].reset_index()
        for _, row in sub.iterrows():
            rows.append(row)
    sens_plot = pd.DataFrame(rows).reset_index(drop=True)
    yd = np.arange(len(sens_plot))
    for yi, row in sens_plot.iterrows():
        ax_d.plot([row["all"], row["exclude_overlap"]], [yi, yi], color="#BDBDBD", lw=0.9, zorder=1)
    ax_d.scatter(sens_plot["all"], yd, color=ORANGE, s=24, marker="o", label="All samples", zorder=3)
    ax_d.scatter(sens_plot["exclude_overlap"], yd, color=DARK_GRAY, s=24, marker="s", label="Exclude initial overlap", zorder=3)
    ax_d.axvline(0, color="#6F6F6F", lw=0.8)
    labels_y = []
    for i, row in sens_plot.iterrows():
        prefix = "Low" if row["scope"] == "Low speed" else "Full"
        labels_y.append(f"{prefix}: {row['comparison']}")
    ax_d.set_yticks(yd, labels_y, fontsize=5.7)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("ΔAUPRC")
    light_x_grid(ax_d)
    ax_d.axhline(3.5, color="#D0D0D0", lw=0.8)
    ax_d.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=5.7)
    ax_d.text(
        0.01,
        -0.23,
        "Low-speed boundary attenuates after overlap exclusion; full-cohort gains remain positive.",
        transform=ax_d.transAxes,
        fontsize=5.5,
        color=DARK_GRAY,
    )

    outputs = save_figure(fig, output_dir, OUTPUT_BASENAMES["S9"], formats, dpi)
    inputs = [speed_path, overlap_path, subtype_path, negative_path, labels_path, full_overlap_path]
    return outputs, inputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready ROF Supplementary Figures S6–S9 from the v1.1 integrated evidence lock."
    )
    parser.add_argument("--evidence-lock", required=True, type=Path, help="Path to the integrated evidence-lock ZIP or extracted directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for figures and manifests")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600)")
    parser.add_argument("--only", nargs="+", choices=["S6", "S7", "S8", "S9"], default=["S6", "S7", "S8", "S9"])
    parser.add_argument("--font-family", default="Arial", help="Preferred sans-serif font")
    parser.add_argument("--no-derived-data", action="store_true", help="Do not write panel-level derived CSVs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = output_dir / "derived_source_data"
    if not args.no_derived_data:
        derived_dir.mkdir(parents=True, exist_ok=True)

    selected_font = choose_font(args.font_family)
    apply_nature_style(selected_font)

    functions = {"S6": plot_s6, "S7": plot_s7, "S8": plot_s8, "S9": plot_s9}
    all_outputs: list[Path] = []
    all_inputs: list[Path] = []

    with EvidenceLock(args.evidence_lock) as lock:
        assert lock.root is not None
        for figure_id in args.only:
            print(f"[ROF] Generating {figure_id} ...", flush=True)
            outputs, inputs = functions[figure_id](
                lock,
                output_dir,
                derived_dir,
                args.formats,
                args.dpi,
                not args.no_derived_data,
            )
            all_outputs.extend(outputs)
            all_inputs.extend(inputs)
            print(f"[ROF] {figure_id}: " + ", ".join(p.name for p in outputs), flush=True)

        input_rows = []
        for path in sorted(set(p.resolve() for p in all_inputs)):
            try:
                relative = str(path.relative_to(lock.root))
            except ValueError:
                relative = str(path)
            input_rows.append(
                {
                    "kind": "input",
                    "relative_or_absolute_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

        output_rows = []
        for path in sorted(all_outputs):
            output_rows.append(
                {
                    "kind": "output",
                    "relative_or_absolute_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if not args.no_derived_data:
            for path in sorted(derived_dir.glob("*.csv")):
                output_rows.append(
                    {
                        "kind": "derived_source_data",
                        "relative_or_absolute_path": str(path.relative_to(output_dir)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

        manifest_df = pd.DataFrame(input_rows + output_rows)
        manifest_path = output_dir / "figure_file_sha256_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        run_manifest = {
            "script_version": SCRIPT_VERSION,
            "evidence_lock_argument": str(args.evidence_lock),
            "evidence_lock_root_name": lock.root.name,
            "figure_number_mapping": FIGURE_SOURCE_DIRS,
            "generated_figures": args.only,
            "formats": args.formats,
            "png_dpi": args.dpi,
            "preferred_font": args.font_family,
            "selected_font": selected_font,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "output_files": [p.name for p in all_outputs],
            "derived_data_written": not args.no_derived_data,
        }
        with (output_dir / "figure_run_manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(run_manifest, fh, indent=2, ensure_ascii=False)

    print(f"[ROF] Done. Outputs: {output_dir}")
    print(f"[ROF] Font selected: {selected_font}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
