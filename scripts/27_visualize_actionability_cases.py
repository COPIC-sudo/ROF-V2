from __future__ import annotations

import argparse
import gzip
import html
import math
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir

try:
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    LineString = None
    Point = None
    unary_union = None


PRIORITY_CATEGORIES = [
    "recovered_positive_at_5fpr",
    "recovered_positive_at_1fpr",
    "original_safe_but_actionability_critical",
    "original_warning_emergency_but_high_actionability",
    "infeasible_true_positive",
    "infeasible_missed",
    "enhanced_false_positive_at_5fpr",
    "baseline_false_positive_fixed",
]

ACTION_SPECS = [
    {"name": "keep", "accel": 0.0, "lat_m": 0.0},
    {"name": "mild_brake", "accel": -2.0, "lat_m": 0.0},
    {"name": "hard_brake", "accel": -5.0, "lat_m": 0.0},
    {"name": "left", "accel": 0.0, "lat_m": 3.0},
    {"name": "right", "accel": 0.0, "lat_m": -3.0},
    {"name": "brake_left", "accel": -4.0, "lat_m": 3.0},
    {"name": "brake_right", "accel": -4.0, "lat_m": -3.0},
]

TEXT_COLS = [
    "sample_id",
    "scenario_id",
    "task",
    "case_category",
    "original_label_id",
    "original_label_name",
    "actionability_label_id",
    "actionability_label_name",
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "agent_count",
    "baseline_score",
    "enhanced_score",
    "score_delta",
    "redi_actionability",
    "ttad_s",
    "early_blocking_ratio",
    "collapse_rate_max_per_s",
    "asr_cum_final",
    "asr_slice_final",
    "comfort_asr",
    "emergency_asr",
]


def _parse_csv_arg(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _normalize_task(task: str) -> str:
    task = str(task).strip()
    aliases = {
        "critical": "actionability_critical",
        "infeasible": "actionability_infeasible",
        "degraded": "actionability_degraded",
    }
    return aliases.get(task, task)


def _safe_slug(value: str, max_len: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value[:max_len] or "case"


def _fmt(value, ndigits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        f = float(value)
        if not np.isfinite(f):
            return ""
        return f"{f:.{ndigits}f}"
    except Exception:
        return str(value)


def _load_sample(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _rect_corners(cx: float, cy: float, heading: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(float(heading)), math.sin(float(heading))
    hl, hw = float(length) * 0.5, float(width) * 0.5
    local = np.asarray([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=float)
    rot = np.asarray([[c, -s], [s, c]], dtype=float)
    return local @ rot.T + np.asarray([float(cx), float(cy)], dtype=float)


def _quad_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.asarray([-edge[1], edge[0]], dtype=float)
            norm = np.linalg.norm(axis)
            if norm <= 1e-12:
                continue
            axis /= norm
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _smoothstep(u: float) -> tuple[float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    return 3.0 * u * u - 2.0 * u * u * u, 6.0 * u * (1.0 - u)


def _rollout_action(sample: dict, action: dict, horizon_s: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    ego = int(sample["ego_index"])
    xy0 = np.asarray(sample["current_xy"][ego], dtype=float)
    vxy = np.asarray(sample["current_vel_xy"][ego], dtype=float)
    heading0 = float(sample["current_heading"][ego])
    speed0 = float(np.linalg.norm(vxy))
    if speed0 > 0.2:
        heading0 = float(math.atan2(vxy[1], vxy[0]))
    c, s = math.cos(heading0), math.sin(heading0)
    forward = np.asarray([c, s], dtype=float)
    left = np.asarray([-s, c], dtype=float)

    times = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    poses = []
    headings = []
    for t in times:
        accel = float(action["accel"])
        if abs(accel) < 1e-12:
            long_s = speed0 * t
            long_v = speed0
        else:
            t_stop = max(speed0 / max(-accel, 1e-9), 0.0) if accel < 0 else np.inf
            tt = min(t, t_stop)
            long_s = max(speed0 * tt + 0.5 * accel * tt * tt, 0.0)
            long_v = max(speed0 + accel * tt, 0.0)
        u, du = _smoothstep(t / max(horizon_s, 1e-9))
        lat = float(action["lat_m"]) * u
        lat_v = float(action["lat_m"]) * du / max(horizon_s, 1e-9)
        xy = xy0 + forward * long_s + left * lat
        hd = heading0 + math.atan2(lat_v, max(long_v, 0.2))
        poses.append(xy)
        headings.append(hd)
    return np.asarray(poses, dtype=float), np.asarray(headings, dtype=float)


def _build_drivable(sample: dict, lane_buffer_m: float):
    if LineString is None or unary_union is None:
        return None
    lanes = sample.get("map_lane_centerlines", []) or []
    polys = []
    for line in lanes:
        arr = np.asarray(line, dtype=float)
        if len(arr) < 2:
            continue
        try:
            polys.append(LineString(arr).buffer(float(lane_buffer_m), cap_style=2, join_style=2))
        except Exception:
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def _map_ok(drivable, xy: np.ndarray) -> bool:
    if drivable is None or Point is None:
        return True
    try:
        return bool(drivable.intersects(Point(float(xy[0]), float(xy[1]))))
    except Exception:
        return True


def _obstacle_pose(sample: dict, agent_i: int, step_i: int, t: float, obstacle_mode: str) -> tuple[np.ndarray, float, bool]:
    if obstacle_mode == "oracle_future":
        fut_valid = np.asarray(sample["future_valid"])
        fut_xy = np.asarray(sample["future_xy"])
        fut_heading = np.asarray(sample["future_heading"])
        k = min(max(int(step_i), 0), fut_xy.shape[1] - 1)
        valid = bool(fut_valid[agent_i, k])
        return np.asarray(fut_xy[agent_i, k], dtype=float), float(fut_heading[agent_i, k]), valid
    xy = np.asarray(sample["current_xy"][agent_i], dtype=float) + np.asarray(sample["current_vel_xy"][agent_i], dtype=float) * float(t)
    v = np.asarray(sample["current_vel_xy"][agent_i], dtype=float)
    heading = float(sample["current_heading"][agent_i])
    if np.linalg.norm(v) > 0.2:
        heading = float(math.atan2(v[1], v[0]))
    return xy, heading, True


def _action_failure_time(sample: dict, action: dict, horizon_s: float, dt_s: float, obstacle_mode: str, drivable) -> float | None:
    ego = int(sample["ego_index"])
    ego_l, ego_w = np.asarray(sample["current_size_lw"][ego], dtype=float)
    sizes = np.asarray(sample["current_size_lw"], dtype=float)
    poses, headings = _rollout_action(sample, action, horizon_s, dt_s)
    times = np.arange(dt_s, horizon_s + 1e-9, dt_s, dtype=float)
    for step_i, (t, xy, hd) in enumerate(zip(times, poses, headings), start=1):
        if not _map_ok(drivable, xy):
            return float(t)
        ego_quad = _rect_corners(float(xy[0]), float(xy[1]), float(hd), float(ego_l), float(ego_w))
        for j in range(int(sample["agent_count"])):
            if j == ego:
                continue
            obs_xy, obs_hd, valid = _obstacle_pose(sample, j, step_i, float(t), obstacle_mode)
            if not valid or not np.all(np.isfinite(obs_xy)):
                continue
            length, width = sizes[j]
            obs_quad = _rect_corners(float(obs_xy[0]), float(obs_xy[1]), float(obs_hd), float(length), float(width))
            if _quad_overlap(ego_quad, obs_quad):
                return float(t)
    return None


def _roi_from_config(cfg: dict) -> tuple[float, float, float, float]:
    roi = cfg.get("roi", {}) or cfg.get("bev", {})
    return (
        float(roi.get("x_min", -20.0)),
        float(roi.get("x_max", 80.0)),
        float(roi.get("y_min", -30.0)),
        float(roi.get("y_max", 30.0)),
    )


def _sort_cases(sub: pd.DataFrame, category: str) -> pd.DataFrame:
    out = sub.copy()
    if category.startswith("recovered_positive"):
        return out.sort_values("score_delta", ascending=False)
    if category == "original_safe_but_actionability_critical":
        return out.sort_values("enhanced_score", ascending=False)
    if category == "original_warning_emergency_but_high_actionability":
        return out.sort_values("baseline_score", ascending=False)
    if category == "infeasible_true_positive":
        return out.sort_values("enhanced_score", ascending=False)
    if category == "infeasible_missed":
        return out.sort_values("enhanced_score", ascending=True)
    if category == "enhanced_false_positive_at_5fpr":
        return out.sort_values("enhanced_score", ascending=False)
    if category == "baseline_false_positive_fixed":
        out["baseline_minus_enhanced_score"] = out["baseline_score"] - out["enhanced_score"]
        return out.sort_values("baseline_minus_enhanced_score", ascending=False)
    return out.sort_values("score_delta", ascending=False)


def _select_visualization_rows(
    cases: pd.DataFrame,
    categories: list[str],
    tasks: list[str],
    top_per_category: int,
    max_total: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = cases.copy()
    cases["sample_id"] = cases["sample_id"].astype(str)
    cases["task"] = cases["task"].astype(str).map(_normalize_task)
    cases["case_category"] = cases["case_category"].astype(str)
    selected_chunks = []
    for category in categories:
        for task in tasks:
            sub = cases[cases["case_category"].eq(category) & cases["task"].eq(task)].copy()
            if sub.empty:
                continue
            selected_chunks.append(_sort_cases(sub, category).head(int(top_per_category)))
    memberships = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame(columns=cases.columns)
    if memberships.empty:
        return pd.DataFrame(columns=cases.columns), memberships

    selected_ids = []
    seen = set()
    for sample_id in memberships["sample_id"].astype(str).tolist():
        if sample_id in seen:
            continue
        selected_ids.append(sample_id)
        seen.add(sample_id)
        if len(selected_ids) >= int(max_total):
            break
    memberships = memberships[memberships["sample_id"].isin(selected_ids)].copy()
    primary_rows = []
    for sample_id in selected_ids:
        first = memberships[memberships["sample_id"].eq(sample_id)].iloc[0].copy()
        primary_rows.append(first)
    primary = pd.DataFrame(primary_rows).reset_index(drop=True)
    return primary, memberships


def _draw_vehicle(ax, xy: np.ndarray, heading: float, length: float, width: float, face: str, edge: str, alpha: float, zorder: int):
    quad = _rect_corners(float(xy[0]), float(xy[1]), float(heading), float(length), float(width))
    patch = Polygon(quad, closed=True, facecolor=face, edgecolor=edge, linewidth=0.8, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    nose = np.asarray([math.cos(float(heading)), math.sin(float(heading))], dtype=float) * (float(length) * 0.35)
    ax.plot([xy[0], xy[0] + nose[0]], [xy[1], xy[1] + nose[1]], color=edge, linewidth=1.0, zorder=zorder + 1)


def _draw_map(ax, sample: dict):
    lanes = sample.get("map_lane_centerlines", []) or []
    for line in lanes:
        arr = np.asarray(line, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
            ax.plot(arr[:, 0], arr[:, 1], color="#9aa0a6", linewidth=0.45, alpha=0.65, zorder=0)


def _draw_future(ax, sample: dict):
    ego = int(sample["ego_index"])
    fut_xy = np.asarray(sample.get("future_xy", []), dtype=float)
    fut_valid = np.asarray(sample.get("future_valid", []))
    if fut_xy.ndim != 3 or fut_valid.ndim != 2:
        return
    n_agents = min(int(sample.get("agent_count", fut_xy.shape[0])), fut_xy.shape[0])
    for i in range(n_agents):
        valid = fut_valid[i].astype(bool)
        if not valid.any():
            continue
        arr = fut_xy[i, valid]
        if arr.ndim != 2 or len(arr) < 2:
            continue
        color = "#0b5fff" if i == ego else "#8c5a2b"
        lw = 1.5 if i == ego else 0.8
        ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=lw, alpha=0.55, zorder=2)


def _draw_current_scene(ax, sample: dict, draw_map: bool, draw_future: bool):
    if draw_map:
        _draw_map(ax, sample)
    if draw_future:
        _draw_future(ax, sample)

    ego = int(sample["ego_index"])
    current_xy = np.asarray(sample["current_xy"], dtype=float)
    current_heading = np.asarray(sample["current_heading"], dtype=float)
    sizes = np.asarray(sample["current_size_lw"], dtype=float)
    n_agents = min(int(sample["agent_count"]), len(current_xy))
    for i in range(n_agents):
        xy = current_xy[i]
        if not np.all(np.isfinite(xy)):
            continue
        length, width = sizes[i]
        if i == ego:
            _draw_vehicle(ax, xy, current_heading[i], length, width, "#1a73e8", "#08306b", 0.85, 6)
            ax.text(xy[0], xy[1], "ego", color="white", fontsize=6, ha="center", va="center", zorder=8)
        else:
            _draw_vehicle(ax, xy, current_heading[i], length, width, "#f28e2b", "#7a3b00", 0.45, 4)


def _draw_action_rollouts(ax, sample: dict, cfg: dict, horizon_s: float, dt_s: float, obstacle_mode: str, draw_map: bool) -> tuple[bool, str]:
    lane_buffer = max(float(cfg.get("bev", {}).get("lane_buffer_m", 2.0)), 3.0)
    drivable = _build_drivable(sample, lane_buffer) if draw_map else None
    rendered_feasibility = True
    message = "rendered_with_map" if draw_map else "rendered_no_map"
    for action in ACTION_SPECS:
        poses, _ = _rollout_action(sample, action, horizon_s, dt_s)
        fail_t = _action_failure_time(sample, action, horizon_s, dt_s, obstacle_mode, drivable)
        safe = fail_t is None
        color = "#1b9e77" if safe else "#d62728"
        label = action["name"] if safe else f"{action['name']} unsafe@{fail_t:.1f}s"
        ax.plot(poses[:, 0], poses[:, 1], color=color, linewidth=1.2, alpha=0.9, zorder=5)
        ax.scatter(poses[-1, 0], poses[-1, 1], s=8, color=color, zorder=6)
        ax.text(poses[-1, 0], poses[-1, 1], label, fontsize=5.5, color=color, zorder=7)
    return rendered_feasibility, message


def _case_text(row: pd.Series, categories: str, tasks: str, rollout_status: str) -> str:
    lines = [
        f"sample_id: {row.get('sample_id', '')}",
        f"scenario_id: {row.get('scenario_id', '')}",
        f"tasks: {tasks}",
        f"categories: {categories}",
        "",
    ]
    for col in TEXT_COLS:
        if col in {"sample_id", "scenario_id", "task", "case_category"}:
            continue
        value = row.get(col, "")
        lines.append(f"{col}: {_fmt(value)}")
    lines.append("")
    lines.append(f"action_rollouts: {rollout_status}")
    return "\n".join(lines)


def _render_case(
    out_path: Path,
    row: pd.Series,
    memberships: pd.DataFrame,
    sample: dict,
    cfg: dict,
    dpi: int,
    draw_future: bool,
    draw_map: bool,
    draw_action_rollouts: bool,
    rollout_use_map: bool,
    horizon_s: float,
    obstacle_mode: str,
) -> tuple[bool, str]:
    dt_s = float(cfg.get("labels", {}).get("dt_s", 0.1))
    categories = ";".join(sorted(memberships["case_category"].astype(str).unique()))
    tasks = ";".join(sorted(memberships["task"].astype(str).unique()))
    rollout_rendered = False
    rollout_status = "disabled"

    fig = plt.figure(figsize=(13.5, 7.2), dpi=int(dpi))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.9], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    text_ax = fig.add_subplot(grid[0, 1])
    try:
        _draw_current_scene(ax, sample, bool(draw_map), bool(draw_future))
        if draw_action_rollouts:
            try:
                rollout_rendered, rollout_status = _draw_action_rollouts(ax, sample, cfg, horizon_s, dt_s, obstacle_mode, bool(rollout_use_map))
            except Exception as exc:  # pragma: no cover - kept per-case so other panels still render
                rollout_rendered = False
                rollout_status = f"action feasibility not rendered: {type(exc).__name__}: {exc}"
                for action in ACTION_SPECS:
                    poses, _ = _rollout_action(sample, action, horizon_s, dt_s)
                    ax.plot(poses[:, 0], poses[:, 1], color="#6f4cc3", linewidth=1.0, alpha=0.75, zorder=5)
        x_min, x_max, y_min, y_max = _roi_from_config(cfg)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#d0d7de", linewidth=0.4, alpha=0.7)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{row.get('sample_id', '')} | {row.get('case_category', '')}", fontsize=10)

        text_ax.axis("off")
        text_ax.text(
            0.0,
            1.0,
            _case_text(row, categories, tasks, rollout_status),
            va="top",
            ha="left",
            fontsize=7.4,
            family="monospace",
            color="#202124",
            linespacing=1.25,
        )
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return rollout_rendered, rollout_status


def _write_index(out_dir: Path, visualized: pd.DataFrame):
    grouped = visualized.copy()
    grouped["categories_list"] = grouped["case_categories"].fillna("").astype(str).str.split(";")
    rows = []
    for _, row in grouped.iterrows():
        for category in row["categories_list"]:
            if category:
                rows.append((category, row))
    by_category: dict[str, list[pd.Series]] = {}
    for category, row in rows:
        by_category.setdefault(category, []).append(row)

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Actionability Case Panels</title>",
        "<style>body{font-family:Arial,sans-serif;margin:20px;color:#202124} "
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px} "
        ".card{border:1px solid #dadce0;border-radius:6px;padding:10px;background:#fff} "
        "img{width:100%;height:auto;border:1px solid #eee} .meta{font-size:12px;line-height:1.35}</style>",
        "</head><body>",
        "<h1>Actionability Case Panels</h1>",
    ]
    for category in PRIORITY_CATEGORIES:
        items = by_category.get(category, [])
        if not items:
            continue
        parts.append(f"<h2>{html.escape(category)} ({len(items)})</h2><div class='grid'>")
        for row in items:
            rel = Path(row["image_path"]).relative_to(out_dir).as_posix()
            meta = (
                f"{row['sample_id']}<br>"
                f"tasks: {html.escape(str(row['tasks']))}<br>"
                f"labels: original={_fmt(row.get('original_label_id'))}, actionability={_fmt(row.get('actionability_label_id'))}<br>"
                f"scores: base={_fmt(row.get('baseline_score'))}, enhanced={_fmt(row.get('enhanced_score'))}, delta={_fmt(row.get('score_delta'))}"
            )
            parts.append(
                "<div class='card'>"
                f"<a href='{html.escape(rel)}'><img src='{html.escape(rel)}'></a>"
                f"<div class='meta'>{meta}</div>"
                "</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _summary_from_memberships(memberships: pd.DataFrame, visualized_ids: set[str]) -> pd.DataFrame:
    if memberships.empty:
        return pd.DataFrame(columns=["case_category", "task", "n_visualized"])
    sub = memberships[memberships["sample_id"].astype(str).isin(visualized_ids)].copy()
    return (
        sub.groupby(["case_category", "task"], dropna=False)["sample_id"]
        .nunique()
        .reset_index(name="n_visualized")
        .sort_values(["case_category", "task"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render BEV review panels for selected actionability cases.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cases-csv", required=True)
    ap.add_argument("--out-name", default="actionability_case_panels_v1")
    ap.add_argument("--categories", default=None)
    ap.add_argument("--tasks", default="critical,infeasible")
    ap.add_argument("--top-per-category", type=int, default=6)
    ap.add_argument("--max-total", type=int, default=80)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--draw-future", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--draw-map", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--draw-action-rollouts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rollout-use-map", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--horizon-s", type=float, default=3.0)
    ap.add_argument("--obstacle-mode", choices=["oracle_future", "cv_current"], default="oracle_future")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out_dir = ensure_dir(work / "results" / "nc_actionability_cases" / args.out_name)
    image_dir = ensure_dir(out_dir / "images")

    cases = pd.read_csv(args.cases_csv)
    required = ["sample_id", "case_category", "task"]
    missing = [c for c in required if c not in cases.columns]
    if missing:
        raise ValueError(f"cases CSV missing required columns {missing}; columns={list(cases.columns)}")

    categories = _parse_csv_arg(args.categories, PRIORITY_CATEGORIES)
    tasks = [_normalize_task(t) for t in _parse_csv_arg(args.tasks, ["critical", "infeasible"])]
    rollout_use_map = bool(args.draw_map) if args.rollout_use_map is None else bool(args.rollout_use_map)
    primary, memberships = _select_visualization_rows(
        cases,
        categories,
        tasks,
        int(args.top_per_category),
        int(args.max_total),
    )
    if primary.empty:
        raise ValueError("no cases selected for visualization after category/task filtering")

    visualized_rows = []
    samples_dir = work / "samples"
    for idx, row in primary.iterrows():
        sample_id = str(row["sample_id"])
        sample_path = samples_dir / f"{sample_id}.pkl.gz"
        if not sample_path.exists():
            raise FileNotFoundError(f"sample pkl not found: {sample_path}")
        sample = _load_sample(sample_path)
        sample_memberships = memberships[memberships["sample_id"].astype(str).eq(sample_id)].copy()
        filename = f"{idx + 1:03d}_{_safe_slug(sample_id)}.png"
        image_path = image_dir / filename
        rollout_rendered, rollout_status = _render_case(
            image_path,
            row,
            sample_memberships,
            sample,
            cfg,
            int(args.dpi),
            bool(args.draw_future),
            bool(args.draw_map),
            bool(args.draw_action_rollouts),
            rollout_use_map,
            float(args.horizon_s),
            args.obstacle_mode,
        )
        visualized = row.to_dict()
        visualized["case_categories"] = ";".join(sorted(sample_memberships["case_category"].astype(str).unique()))
        visualized["tasks"] = ";".join(sorted(sample_memberships["task"].astype(str).unique()))
        visualized["image_path"] = str(image_path)
        visualized["action_rollouts_rendered"] = bool(rollout_rendered)
        visualized["action_rollout_status"] = rollout_status
        visualized["rollout_use_map"] = bool(rollout_use_map)
        visualized_rows.append(visualized)

    visualized_df = pd.DataFrame(visualized_rows)
    visualized_df.to_csv(out_dir / "selected_visualized_cases.csv", index=False)
    summary_df = _summary_from_memberships(memberships, set(visualized_df["sample_id"].astype(str)))
    summary_df.to_csv(out_dir / "visualization_summary.csv", index=False)
    _write_index(out_dir, visualized_df)

    print(f"[case-panels] wrote {len(visualized_df)} PNG files")
    print(f"[case-panels] output_dir={out_dir}")
    print(f"[case-panels] index={out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
