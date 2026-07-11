from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "black": "#222222",
    "dark_gray": "#555555",
    "light_gray": "#D9D9D9",
    "navy": "#1F4E79",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "red": "#C1121F",
}


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def set_nature_data_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 0.85,
            "patch.linewidth": 0.8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["black"],
            "xtick.color": PALETTE["black"],
            "ytick.color": PALETTE["black"],
            "axes.labelcolor": PALETTE["black"],
            "text.color": PALETTE["black"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["black"],
    )


def finish_axis(ax, y_grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color=PALETTE["light_gray"], alpha=0.15, linewidth=0.7)
    else:
        ax.grid(False)


def save_figure(fig, out_base: str | Path, dpi: int = 300) -> tuple[Path, Path, Path]:
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf, svg
