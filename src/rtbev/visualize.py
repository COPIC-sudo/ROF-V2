from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from shapely.affinity import rotate, translate

from .geometry import oriented_box_polygon
from .tube.rt_library import TubeLibrary, get_record_slice_at_time
from .utils import nearest_value


def _draw_geom(ax, geom, color: str, lw: float = 1.5, alpha: float = 1.0, fill: bool = False):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        if fill:
            ax.fill(x, y, color=color, alpha=alpha, linewidth=0)
        else:
            ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
        return
    try:
        for g in geom.geoms:
            _draw_geom(ax, g, color=color, lw=lw, alpha=alpha, fill=fill)
    except Exception:
        return


def _transform_local_poly(poly, x: float, y: float, heading: float):
    if poly is None:
        return None
    g = rotate(poly, float(heading), origin=(0.0, 0.0), use_radians=True)
    g = translate(g, xoff=float(x), yoff=float(y))
    return g


def _select_mu(is_ego: bool, cfg: dict) -> float:
    tcfg = cfg["tube"]
    policy = str(tcfg.get("friction_policy", "nominal")).lower()
    if policy == "asymmetric" or policy == "fixed":
        return float(tcfg.get("mu_ego" if is_ego else "mu_other", tcfg.get("assumed_mu", 0.5)))
    return float(tcfg.get("assumed_mu", 0.5))


def render_sample_overlay(sample: dict, lib: TubeLibrary, cfg: dict, out_png: Path, device: str = "cpu") -> None:
    bev = cfg["bev"]
    preview_times = [float(x) for x in bev.get("preview_times_s", [0.5, 1.0, 1.5, 2.0])]
    mu_bins = [float(x) for x in cfg["tube"]["mu_bins"]]
    speed_bins = [int(x) for x in cfg["tube"]["speed_bins_kph"]]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    title = f"{sample['sample_id']} | label={sample.get('label', {}).get('label_name', 'NA')}"
    ax.set_title(title)

    for line in sample.get("map_lane_centerlines", []):
        arr = np.asarray(line)
        if len(arr) >= 2:
            ax.plot(arr[:, 0], arr[:, 1], color="0.82", linewidth=1.0)

    for i in range(sample["agent_count"]):
        cx, cy = sample["current_xy"][i]
        L, W = sample["current_size_lw"][i]
        hd = sample["current_heading"][i]
        poly = oriented_box_polygon(float(cx), float(cy), float(L), float(W), float(hd))
        color = "tab:blue" if i == sample["ego_index"] else "tab:orange"
        _draw_geom(ax, poly, color=color, lw=2.0)
        ax.text(float(cx), float(cy), f"{int(sample['agent_ids'][i])}", fontsize=7, color=color)

    colors = {0.5: "#2ca02c", 1.0: "#1f77b4", 1.5: "#ff7f0e", 2.0: "#d62728"}
    for i in range(sample["agent_count"]):
        speed_kph = float(np.linalg.norm(sample["current_vel_xy"][i])) * 3.6
        rec = lib.select(_select_mu(i == sample["ego_index"], cfg), speed_kph, mu_bins, speed_bins)
        x, y = sample["current_xy"][i]
        hd = sample["current_heading"][i]
        for pt in preview_times:
            poly = get_record_slice_at_time(rec, pt)
            g = _transform_local_poly(poly, x, y, hd)
            _draw_geom(ax, g, color=colors.get(round(pt, 1), "k"), lw=1.1, alpha=0.75)

    ax.set_xlim(float(bev["x_min"]), float(bev["x_max"]))
    ax.set_ylim(float(bev["y_min"]), float(bev["y_max"]))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render_tensor_maps(npz_path: Path, cfg: dict, out_png: Path, title: str = "") -> None:
    data = np.load(npz_path)
    bev = cfg["bev"]
    extent = [float(bev["x_min"]), float(bev["x_max"]), float(bev["y_min"]), float(bev["y_max"])]
    maps = [
        ("overlap", data["overlap"].max(axis=0)),
        ("ego_conflict", data["ego_conflict"].max(axis=0)),
        ("free_drv", data["free_drv"].max(axis=0)),
        ("overlap_count_max", data["overlap_count"].max(axis=0)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    if title:
        fig.suptitle(title)
    for ax, (name, arr) in zip(axes.ravel(), maps):
        ax.imshow(arr, origin="lower", extent=extent, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
