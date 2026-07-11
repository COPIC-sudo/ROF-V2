from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from shapely.affinity import rotate, translate
from shapely import wkt as shapely_wkt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from .geometry import oriented_box_polygon, line_to_buffer_polygon, union_polygons
from .rasterizer import make_grid, geometry_to_mask, oriented_box_mask_np
from .tube.rt_library import TubeLibrary, PrimitiveLibrary, get_record_slice_at_time, get_primitive_aligned


# ---------------------------------------------------------------------------
# CPU compatibility helpers from the original implementation
# ---------------------------------------------------------------------------

def _transform_local_poly(poly, x: float, y: float, heading: float):
    if poly is None:
        return None
    g = rotate(poly, float(heading), origin=(0.0, 0.0), use_radians=True)
    g = translate(g, xoff=float(x), yoff=float(y))
    return g


def _transform_local_point(px: float, py: float, x0: float, y0: float, heading: float):
    c, s = math.cos(float(heading)), math.sin(float(heading))
    return (float(x0) + c * float(px) - s * float(py), float(y0) + s * float(px) + c * float(py))


def _make_lane_mask(sample: dict, cfg: dict, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Lane/drivable mask. This remains a CPU Shapely path because map centerlines
    are arbitrary polylines and are evaluated only once per sample.
    """
    buffer_m = float(cfg["bev"].get("lane_buffer_m", 2.0))
    polys = []
    for line in sample.get("map_lane_centerlines", []):
        poly = line_to_buffer_polygon(np.asarray(line, dtype=np.float64), buffer_m)
        if poly is not None:
            polys.append(poly)
    u = union_polygons(polys)
    return geometry_to_mask(u, xx, yy).astype(np.uint8)


def _make_current_boxes_mask(sample: dict, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    polys = []
    for i in range(sample["agent_count"]):
        cx, cy = sample["current_xy"][i]
        L, W = sample["current_size_lw"][i]
        hd = sample["current_heading"][i]
        polys.append(oriented_box_polygon(float(cx), float(cy), float(L), float(W), float(hd)))
    u = union_polygons(polys)
    return geometry_to_mask(u, xx, yy).astype(np.uint8)


def _select_mu(is_ego: bool, cfg: dict) -> float:
    tcfg = cfg["tube"]
    policy = str(tcfg.get("friction_policy", "nominal")).lower()
    if policy == "asymmetric":
        return float(tcfg.get("mu_ego" if is_ego else "mu_other", tcfg.get("assumed_mu", 0.5)))
    if policy == "fixed":
        return float(tcfg.get("mu_ego" if is_ego else "mu_other", tcfg.get("assumed_mu", 0.5)))
    return float(tcfg.get("assumed_mu", 0.5))


def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a) / (float(b) + eps)


def _empty_actionability_features() -> dict:
    """Return a full feature dict when primitive/actionability metrics are unavailable."""
    out = {
        "msr_available": 0,
        "stable_primitive_count": 0,
        "msr": np.nan,
        "c_maneuver": np.nan,
        "asr_available": 0,
        # Backward-compatible aliases.  In v1.1 these refer to cumulative ASR.
        "asr": np.nan,
        "asr_final": np.nan,
        "asr_min": np.nan,
        "asr_initial": np.nan,
        "c_actionability": np.nan,
        # v1.1 explicit actionability curves.
        "asr_cum_initial": np.nan,
        "asr_cum_final": np.nan,
        "asr_cum_min": np.nan,
        "asr_slice_initial": np.nan,
        "asr_slice_final": np.nan,
        "asr_slice_min": np.nan,
        "comfort_asr": np.nan,
        "emergency_asr": np.nan,
        "comfort_to_emergency_gap": np.nan,
        "ttad_s": np.nan,
        "time_to_first_conflict_s": np.nan,
        "collapse_rate_per_s": np.nan,
        "collapse_rate_mean_per_s": np.nan,
        "collapse_rate_max_per_s": np.nan,
        "early_blocking_ratio": np.nan,
        "min_safe_action_cost": np.nan,
    }
    for fam in ["keep", "accelerate", "brake", "hard_brake", "left", "right", "brake_left", "brake_right", "unknown"]:
        out[f"survival_{fam}"] = -1.0
        out[f"slice_survival_{fam}"] = -1.0
    return out


def _asr_time_key(t: float, kind: str = "cum") -> str:
    return f"asr_{kind}_t{str(round(float(t), 1)).replace('.', 'p')}s"


def _weighted_bool_curve(mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
        return np.asarray([], dtype=float)
    if w.shape[0] != m.shape[0]:
        w = np.ones(m.shape[0], dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    return (m.astype(float).T @ w) / max(float(np.sum(w)), 1e-9)


def _summarize_actionability(
    safe_slice_matrix: np.ndarray,
    safe_cum_matrix: np.ndarray,
    weights: np.ndarray,
    times: np.ndarray,
    prim: dict,
    cfg: dict,
) -> dict:
    """Summarize primitive safety into actionability metrics.

    v1.0 only exposed cumulative survival, so ASR often collapsed to MSR.  v1.1
    exports both instantaneous/slice safety and cumulative safety.  The old MSR
    remains the final cumulative ASR, but ASR-slice, TTAD, collapse rate, comfort
    ASR and family survival provide genuinely different actionability evidence.
    """
    safe_slice = np.asarray(safe_slice_matrix, dtype=bool)
    safe_cum = np.asarray(safe_cum_matrix, dtype=bool)
    if safe_slice.ndim != 2 or safe_slice.shape[0] == 0 or safe_slice.shape[1] == 0:
        return _empty_actionability_features()
    if safe_cum.shape != safe_slice.shape:
        safe_cum = np.minimum.accumulate(safe_slice.astype(np.uint8), axis=1).astype(bool)

    M, K = safe_slice.shape
    times = np.asarray(times, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != M:
        w = np.ones(M, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    denom = max(float(np.sum(w)), 1e-9)

    cum_curve = _weighted_bool_curve(safe_cum, w)
    slice_curve = _weighted_bool_curve(safe_slice, w)
    cum_final = float(cum_curve[-1])
    cum_min = float(np.min(cum_curve))
    cum_initial = float(cum_curve[0])
    slice_final = float(slice_curve[-1])
    slice_min = float(np.min(slice_curve))
    slice_initial = float(slice_curve[0])

    acfg = cfg.get("actionability", {})
    depletion_thr = float(acfg.get("depletion_asr_threshold", 0.05))
    idx_dep = np.where(cum_curve <= depletion_thr)[0]
    if len(idx_dep):
        ttad = float(times[int(idx_dep[0])])
    else:
        dt = float(np.median(np.diff(times))) if len(times) >= 2 else float(cfg.get("tube", {}).get("query_dt_s", 0.1))
        ttad = float(times[-1] + dt) if len(times) else float(cfg.get("tube", {}).get("horizon_s", 2.0))

    # First time the instantaneous action set is no longer fully available.
    conflict_thr = float(acfg.get("first_conflict_asr_threshold", 0.999))
    idx_conf = np.where(slice_curve < conflict_thr)[0]
    time_to_first_conflict = float(times[int(idx_conf[0])]) if len(idx_conf) else float(ttad)

    if len(times) >= 2:
        dt_arr = np.maximum(np.diff(times), 1e-9)
        drops = np.maximum(-(np.diff(cum_curve) / dt_arr), 0.0)
        collapse_max = float(np.max(drops)) if len(drops) else 0.0
        collapse_mean = float(max(cum_curve[0] - cum_curve[-1], 0.0) / max(float(times[-1] - times[0]), 1e-9))
    else:
        collapse_max = 0.0
        collapse_mean = 0.0

    early_window_s = float(acfg.get("early_window_s", 0.5))
    early_mask = times <= early_window_s
    if np.any(early_mask):
        early_blocking = float(1.0 - np.mean(cum_curve[early_mask]))
    else:
        early_blocking = float(1.0 - cum_initial)

    costs = np.asarray(prim.get("action_cost", np.ones(M)), dtype=float).reshape(-1)
    if costs.shape[0] != M:
        costs = np.ones(M, dtype=float)
    costs = np.where(np.isfinite(costs), costs, 1.0)
    families = np.asarray(prim.get("action_family", np.asarray(["unknown"] * M))).astype(str).reshape(-1)
    if families.shape[0] != M:
        families = np.asarray(["unknown"] * M)

    final_cum = safe_cum[:, -1]
    final_slice = safe_slice[:, -1]
    min_cost = float(np.nanmin(costs[final_cum])) if np.any(final_cum) else np.inf

    comfort_thr = float(acfg.get("comfort_action_cost_threshold", 0.45))
    emergency_thr = float(acfg.get("emergency_action_cost_threshold", 1.01))
    comfort_mask = costs <= comfort_thr
    emergency_mask = costs <= emergency_thr

    def _asr_for(mask: np.ndarray, alive: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return -1.0
        return float(np.sum(w[mask] * alive[mask].astype(float)) / max(float(np.sum(w[mask])), 1e-9))

    comfort_asr = _asr_for(comfort_mask, final_cum)
    emergency_asr = _asr_for(emergency_mask, final_cum)
    gap = float(emergency_asr - comfort_asr) if np.isfinite(comfort_asr) and np.isfinite(emergency_asr) and comfort_asr >= 0 and emergency_asr >= 0 else np.nan

    out = {
        "msr_available": 1,
        "stable_primitive_count": int(M),
        "msr": cum_final,
        "c_maneuver": 1.0 - cum_final,
        "asr_available": 1,
        # Backward-compatible aliases.
        "asr": cum_final,
        "asr_final": cum_final,
        "asr_min": cum_min,
        "asr_initial": cum_initial,
        "c_actionability": 1.0 - cum_final,
        # Explicit v1.1 metrics.
        "asr_cum_initial": cum_initial,
        "asr_cum_final": cum_final,
        "asr_cum_min": cum_min,
        "asr_slice_initial": slice_initial,
        "asr_slice_final": slice_final,
        "asr_slice_min": slice_min,
        "comfort_asr": comfort_asr,
        "emergency_asr": emergency_asr,
        "comfort_to_emergency_gap": gap,
        "ttad_s": ttad,
        "time_to_first_conflict_s": time_to_first_conflict,
        "collapse_rate_per_s": collapse_max,
        "collapse_rate_mean_per_s": collapse_mean,
        "collapse_rate_max_per_s": collapse_max,
        "early_blocking_ratio": early_blocking,
        "min_safe_action_cost": min_cost,
    }

    # Export compact time curves. These are useful for debugging lead-time.
    for k, t in enumerate(times):
        out[_asr_time_key(t, "cum")] = float(cum_curve[k])
        out[_asr_time_key(t, "slice")] = float(slice_curve[k])
    # Backward compatibility with v1.0 asr_t*.  It now refers to cumulative ASR.
    for k, t in enumerate(times):
        out[f"asr_t{str(round(float(t), 1)).replace('.', 'p')}s"] = float(cum_curve[k])

    known_fams = ["keep", "accelerate", "brake", "hard_brake", "left", "right", "brake_left", "brake_right", "unknown"]
    for fam in known_fams:
        m = families == fam
        out[f"survival_{fam}"] = _asr_for(m, final_cum) if np.any(m) else -1.0
        out[f"slice_survival_{fam}"] = _asr_for(m, final_slice) if np.any(m) else -1.0
    return out


def _tfrc_from_area(
    times: np.ndarray,
    conflict_area: np.ndarray,
    ego_area: np.ndarray,
    cfg: dict,
) -> tuple[float, float]:
    """
    Return:
        tfrc_s:
            Time-to-first reachable conflict. If no reachable conflict is
            triggered within the horizon, use horizon + dt instead of -1.
        c_time:
            Time urgency term. No conflict -> 0.
    """
    mc = cfg.get("metrics", {})
    min_area = float(mc.get("tfrc_min_area_m2", 1.0))
    min_ratio = float(mc.get("tfrc_min_ratio", 0.01))

    ratio = conflict_area / np.maximum(ego_area, 1e-9)
    ok = (conflict_area >= min_area) | (ratio >= min_ratio)
    idx = np.where(ok)[0]

    tau = float(mc.get("tau_time_s", 1.0))

    if len(idx) == 0:
        if len(times) >= 2:
            dt = float(np.median(np.diff(times)))
            no_conflict_time = float(times[-1] + dt)
        elif len(times) == 1:
            dt = float(cfg.get("tube", {}).get("query_dt_s", 0.1))
            no_conflict_time = float(times[-1] + dt)
        else:
            horizon = float(cfg.get("tube", {}).get("horizon_s", 2.0))
            dt = float(cfg.get("tube", {}).get("query_dt_s", 0.1))
            no_conflict_time = horizon + dt
        return no_conflict_time, 0.0

    t = float(times[int(idx[0])])
    return t, float(np.exp(-t / max(tau, 1e-6)))


def _weighted_areas(mask: np.ndarray, weights: np.ndarray, cell_area: float) -> tuple[np.ndarray, float]:
    per_t = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(float) * cell_area
    return per_t, float(np.sum(weights * per_t))


def _overlap_entropy(overlap_count: np.ndarray, cfg: dict) -> tuple[float, float, int, float]:
    vals = overlap_count.ravel()
    if bool(cfg.get("metrics", {}).get("oce_ignore_zero", True)):
        vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0, 0.0, 0, 0.0
    vals = vals.astype(int)
    max_c = int(vals.max())
    hist = np.bincount(vals, minlength=max_c + 1).astype(float)
    if bool(cfg.get("metrics", {}).get("oce_ignore_zero", True)) and len(hist) > 0:
        hist[0] = 0.0
    p = hist[hist > 0] / max(hist.sum(), 1e-9)
    oce = float(-np.sum(p * np.log(p + 1e-12)))
    oce_norm = float(oce / max(np.log(max_c + 1), 1e-9)) if max_c > 0 else 0.0
    return oce, oce_norm, max_c, float(vals.mean())


def _compute_current_distance_ttc(sample: dict) -> dict:
    ego_idx = int(sample["ego_index"])
    xy = np.asarray(sample["current_xy"], dtype=np.float64)
    vel = np.asarray(sample["current_vel_xy"], dtype=np.float64)
    hd = np.asarray(sample["current_heading"], dtype=np.float64)
    size = np.asarray(sample["current_size_lw"], dtype=np.float64)
    ego_poly = oriented_box_polygon(float(xy[ego_idx, 0]), float(xy[ego_idx, 1]), float(size[ego_idx, 0]), float(size[ego_idx, 1]), float(hd[ego_idx]))
    dmin = np.inf
    nearest_j = -1
    collision = False
    ttc_min = np.inf
    closing_speed_at_ttc = 0.0
    nearby10 = 0
    nearby20 = 0
    for j in range(sample["agent_count"]):
        if j == ego_idx:
            continue
        poly = oriented_box_polygon(float(xy[j, 0]), float(xy[j, 1]), float(size[j, 0]), float(size[j, 1]), float(hd[j]))
        if ego_poly.intersects(poly):
            collision = True
            d = 0.0
        else:
            d = float(ego_poly.distance(poly))
        if d < dmin:
            dmin = d
            nearest_j = j
        if d <= 10.0:
            nearby10 += 1
        if d <= 20.0:
            nearby20 += 1
        p = xy[j] - xy[ego_idx]
        rv = vel[j] - vel[ego_idx]
        denom = float(np.dot(rv, rv))
        if denom > 1e-6 and float(np.dot(p, rv)) < 0.0:
            ttc = -float(np.dot(p, rv)) / denom
            if ttc >= 0:
                dist_center = max(float(np.linalg.norm(p)), 1e-6)
                closing = max(0.0, -float(np.dot(p, rv)) / dist_center)
                if ttc < ttc_min:
                    ttc_min = ttc
                    closing_speed_at_ttc = closing
    if nearest_j >= 0:
        rel_v = vel[nearest_j] - vel[ego_idx]
        nearest_rel_speed = float(np.linalg.norm(rel_v))
        p_near = xy[nearest_j] - xy[ego_idx]
        dist_center = max(float(np.linalg.norm(p_near)), 1e-6)
        nearest_closing = max(0.0, -float(np.dot(p_near, rel_v)) / dist_center)
    else:
        nearest_rel_speed = 0.0
        nearest_closing = 0.0
    return {
        "current_min_distance_m": float(dmin if np.isfinite(dmin) else 9999.0),
        "current_collision": bool(collision),
        "current_ttc_s": float(ttc_min if np.isfinite(ttc_min) else -1.0),
        "nearest_agent_index": int(nearest_j),
        "nearest_agent_rel_speed_mps": nearest_rel_speed,
        "nearest_agent_closing_speed_mps": nearest_closing,
        "ttc_closing_speed_mps": closing_speed_at_ttc,
        "nearby_agent_count_10m": int(nearby10),
        "nearby_agent_count_20m": int(nearby20),
    }


def _cv_heading(current_heading: float, velocity_xy: np.ndarray, cfg: dict) -> float:
    cvcfg = cfg.get("cv_baseline", {})
    source = str(cvcfg.get("heading_source", "velocity_or_current")).lower()
    v = np.asarray(velocity_xy, dtype=float)
    speed = float(np.linalg.norm(v))
    thr = float(cvcfg.get("heading_speed_threshold_mps", 0.5))
    if source == "current":
        return float(current_heading)
    if speed >= thr and source in {"velocity", "velocity_or_current"}:
        return float(np.arctan2(v[1], v[0]))
    return float(current_heading)


def _cv_interval_polygon(
    p0: np.ndarray,
    p1: np.ndarray,
    length: float,
    width: float,
    heading: float,
    cfg: dict,
):
    cvcfg = cfg.get("cv_baseline", {})
    inflate = float(cvcfg.get("footprint_inflation_m", 0.5))
    use_swept = bool(cvcfg.get("use_swept_footprint", True))
    L = max(float(length) + 2.0 * inflate, 0.1)
    W = max(float(width) + 2.0 * inflate, 0.1)
    b1 = oriented_box_polygon(float(p1[0]), float(p1[1]), L, W, float(heading))
    if not use_swept:
        return b1
    min_motion = float(cvcfg.get("min_motion_for_swept_m", 0.05))
    if float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))) < min_motion:
        return b1
    b0 = oriented_box_polygon(float(p0[0]), float(p0[1]), L, W, float(heading))
    try:
        return union_polygons([b0, b1]).convex_hull
    except Exception:
        return b1


# ---------------------------------------------------------------------------
# GPU geometry/rasterization path
# ---------------------------------------------------------------------------

_TORCH_GRID_CACHE: dict[tuple, tuple] = {}
_TUBE_RING_CACHE: dict[tuple, list] = {}


def _resolve_torch_device(device: str | None):
    dev = str(device or "cpu").strip()
    if not dev or dev.lower() == "cpu":
        return None
    if dev.lower().startswith("cuda"):
        # Return None rather than raising so the script can fall back to CPU unless
        # the user explicitly passes --require-cuda.
        if torch is None:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            out = torch.device(dev)
            idx = out.index if out.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(idx)
            return out
        except Exception:
            return None
    # Non-CUDA torch devices are intentionally not used here; the accelerated
    # kernels below are written for CUDA throughput.
    return None


def _torch_dtype(cfg: dict):
    if torch is None:
        return None
    name = str(cfg.get("runtime", {}).get("gpu_dtype", "float64")).lower()
    if name in {"float64", "double", "fp64"}:
        return torch.float64
    return torch.float32


def _get_torch_grid(cfg: dict, xx: np.ndarray, yy: np.ndarray, device, dtype):
    H, W = xx.shape
    bev = cfg["bev"]
    key = (
        str(device),
        str(dtype),
        H,
        W,
        float(bev["x_min"]),
        float(bev["x_max"]),
        float(bev["y_min"]),
        float(bev["y_max"]),
        float(bev["resolution_m"]),
    )
    cached = _TORCH_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    grid_x = torch.as_tensor(xx.reshape(-1), device=device, dtype=dtype)
    grid_y = torch.as_tensor(yy.reshape(-1), device=device, dtype=dtype)
    _TORCH_GRID_CACHE[key] = (grid_x, grid_y, H, W)
    return grid_x, grid_y, H, W


def _transform_ring_np(ring_xy: np.ndarray, x: float, y: float, heading: float) -> np.ndarray:
    ring = np.asarray(ring_xy, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    c, s = math.cos(float(heading)), math.sin(float(heading))
    out = np.empty_like(ring, dtype=np.float64)
    out[:, 0] = float(x) + c * ring[:, 0] - s * ring[:, 1]
    out[:, 1] = float(y) + s * ring[:, 0] + c * ring[:, 1]
    return out


def _extract_polygon_rings_np(geom) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Return [(exterior, holes), ...] as local-coordinate numpy rings."""
    if geom is None:
        return []
    try:
        if geom.is_empty:
            return []
    except Exception:
        return []

    if isinstance(geom, Polygon):
        ext = np.asarray(geom.exterior.coords, dtype=np.float64)
        if ext.shape[0] < 4:
            return []
        holes = []
        for ring in geom.interiors:
            h = np.asarray(ring.coords, dtype=np.float64)
            if h.shape[0] >= 4:
                holes.append(h)
        return [(ext, holes)]

    if isinstance(geom, MultiPolygon):
        out = []
        for poly in geom.geoms:
            out.extend(_extract_polygon_rings_np(poly))
        return out

    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_extract_polygon_rings_np(g))
        return out

    return []


def _get_tube_rings_at_times(rec, times: np.ndarray) -> list[list[tuple[np.ndarray, list[np.ndarray]]]]:
    """Cache WKT -> polygon rings by TubeUnionRecord and physical query times.

    The original implementation parsed WKT and applied Shapely affine transforms
    inside the per-agent/per-time loop. This cache removes repeated WKT parsing;
    the GPU path then applies the rigid transform numerically.
    """
    tkey = tuple(round(float(t), 6) for t in np.asarray(times, dtype=np.float64))
    key = (id(rec), tkey)
    cached = _TUBE_RING_CACHE.get(key)
    if cached is not None:
        return cached

    if not getattr(rec, "time_s", None):
        out = [[] for _ in tkey]
        _TUBE_RING_CACHE[key] = out
        return out

    src_times = np.asarray(rec.time_s, dtype=np.float64)
    out = []
    for t in tkey:
        idx = int(np.argmin(np.abs(src_times - float(t))))
        try:
            w = rec.slice_wkts[idx]
        except Exception:
            w = None
        if w is None or w == "":
            out.append([])
            continue
        try:
            geom = shapely_wkt.loads(w)
            out.append(_extract_polygon_rings_np(geom))
        except Exception:
            out.append([])
    _TUBE_RING_CACHE[key] = out
    return out


def _point_in_ring_flat_gpu(
    ring_xy: np.ndarray,
    grid_x,
    grid_y,
    dtype,
    edge_chunk: int = 256,
):
    """Even-odd point-in-ring test on CUDA for flattened BEV grid points."""
    N = int(grid_x.numel())
    ring = np.asarray(ring_xy, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] != 2:
        return torch.zeros((N,), device=grid_x.device, dtype=torch.bool)
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 3:
        return torch.zeros((N,), device=grid_x.device, dtype=torch.bool)

    verts = torch.as_tensor(ring, device=grid_x.device, dtype=dtype)
    x1_all = verts[:, 0]
    y1_all = verts[:, 1]
    x2_all = torch.roll(x1_all, shifts=-1, dims=0)
    y2_all = torch.roll(y1_all, shifts=-1, dims=0)

    inside = torch.zeros((N,), device=grid_x.device, dtype=torch.bool)
    edge_chunk = max(16, int(edge_chunk))
    eps = torch.finfo(dtype).eps

    # Accumulate parity in edge chunks to cap temporary memory when a tube slice
    # has many alpha-shape vertices.
    for start in range(0, int(verts.shape[0]), edge_chunk):
        end = min(start + edge_chunk, int(verts.shape[0]))
        xi = x1_all[start:end].view(1, -1)
        yi = y1_all[start:end].view(1, -1)
        xj = x2_all[start:end].view(1, -1)
        yj = y2_all[start:end].view(1, -1)

        py = grid_y.view(-1, 1)
        px = grid_x.view(-1, 1)
        crosses_y = (yi > py) != (yj > py)
        denom = yj - yi
        denom = torch.where(torch.abs(denom) > eps, denom, torch.full_like(denom, eps))
        x_intersect = (xj - xi) * (py - yi) / denom + xi
        crosses = crosses_y & (px < x_intersect)
        parity = (torch.count_nonzero(crosses, dim=1) & 1).to(torch.bool)
        inside ^= parity
    return inside


def _rings_to_mask_flat_gpu(
    rings: list[tuple[np.ndarray, list[np.ndarray]]],
    x: float,
    y: float,
    heading: float,
    grid_x,
    grid_y,
    dtype,
    edge_chunk: int = 256,
):
    N = int(grid_x.numel())
    if not rings:
        return torch.zeros((N,), device=grid_x.device, dtype=torch.bool)
    out = torch.zeros((N,), device=grid_x.device, dtype=torch.bool)
    for ext, holes in rings:
        ext_g = _transform_ring_np(ext, x, y, heading)
        if ext_g.shape[0] < 3:
            continue
        poly_mask = _point_in_ring_flat_gpu(ext_g, grid_x, grid_y, dtype, edge_chunk=edge_chunk)
        for h in holes:
            h_g = _transform_ring_np(h, x, y, heading)
            if h_g.shape[0] >= 3:
                poly_mask &= ~_point_in_ring_flat_gpu(h_g, grid_x, grid_y, dtype, edge_chunk=edge_chunk)
        out |= poly_mask
    return out


def _oriented_box_masks_flat_gpu(cx, cy, length, width, heading, grid_x, grid_y, dtype):
    """Return [B, H*W] CUDA bool masks for oriented boxes."""
    device = grid_x.device

    def _as_vec(v):
        if torch.is_tensor(v):
            t = v.to(device=device, dtype=dtype)
        else:
            t = torch.as_tensor(v, device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.view(1)
        return t

    cx_t = _as_vec(cx)
    cy_t = _as_vec(cy)
    hd_t = _as_vec(heading)
    B = int(max(cx_t.numel(), cy_t.numel(), hd_t.numel()))

    if cx_t.numel() == 1 and B > 1:
        cx_t = cx_t.expand(B)
    if cy_t.numel() == 1 and B > 1:
        cy_t = cy_t.expand(B)
    if hd_t.numel() == 1 and B > 1:
        hd_t = hd_t.expand(B)

    L_t = _as_vec(length)
    W_t = _as_vec(width)
    if L_t.numel() == 1 and B > 1:
        L_t = L_t.expand(B)
    if W_t.numel() == 1 and B > 1:
        W_t = W_t.expand(B)

    c = torch.cos(hd_t).view(B, 1)
    s = torch.sin(hd_t).view(B, 1)
    dx = grid_x.view(1, -1) - cx_t.view(B, 1)
    dy = grid_y.view(1, -1) - cy_t.view(B, 1)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return (torch.abs(local_x) <= (L_t.view(B, 1) * 0.5)) & (torch.abs(local_y) <= (W_t.view(B, 1) * 0.5))


def _make_current_boxes_mask_gpu(sample: dict, H: int, W: int, grid_x, grid_y, dtype, cfg: dict) -> np.ndarray:
    xy = np.asarray(sample["current_xy"], dtype=np.float64)
    size = np.asarray(sample["current_size_lw"], dtype=np.float64)
    hd = np.asarray(sample["current_heading"], dtype=np.float64)
    n = int(sample["agent_count"])
    batch = int(cfg.get("runtime", {}).get("gpu_box_batch_size", 256))
    out = torch.zeros((H * W,), device=grid_x.device, dtype=torch.bool)
    for start in range(0, n, max(1, batch)):
        end = min(start + max(1, batch), n)
        masks = _oriented_box_masks_flat_gpu(
            xy[start:end, 0], xy[start:end, 1], size[start:end, 0], size[start:end, 1], hd[start:end], grid_x, grid_y, dtype
        )
        out |= masks.any(dim=0)
    return out.view(H, W).to(torch.uint8).cpu().numpy()


def _box_corners_np(cx: float, cy: float, length: float, width: float, heading: float) -> np.ndarray:
    hl = float(length) * 0.5
    hw = float(width) * 0.5
    pts = np.asarray([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)
    c, s = math.cos(float(heading)), math.sin(float(heading))
    out = np.empty_like(pts)
    out[:, 0] = float(cx) + c * pts[:, 0] - s * pts[:, 1]
    out[:, 1] = float(cy) + s * pts[:, 0] + c * pts[:, 1]
    return out


def _convex_hull_np(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.unique(pts, axis=0)
    if pts.shape[0] <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


def _cv_interval_mask_flat_gpu(
    p0: np.ndarray,
    p1: np.ndarray,
    length: float,
    width: float,
    heading: float,
    cfg: dict,
    grid_x,
    grid_y,
    dtype,
    edge_chunk: int = 256,
):
    cvcfg = cfg.get("cv_baseline", {})
    inflate = float(cvcfg.get("footprint_inflation_m", 0.5))
    use_swept = bool(cvcfg.get("use_swept_footprint", True))
    L = max(float(length) + 2.0 * inflate, 0.1)
    W = max(float(width) + 2.0 * inflate, 0.1)
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    min_motion = float(cvcfg.get("min_motion_for_swept_m", 0.05))
    if (not use_swept) or float(np.linalg.norm(p1 - p0)) < min_motion:
        return _oriented_box_masks_flat_gpu([p1[0]], [p1[1]], [L], [W], [heading], grid_x, grid_y, dtype)[0]

    # Exact counterpart of union_polygons([box0, box1]).convex_hull for two
    # equal-orientation rectangles, using only the 8 rectangle corners.
    corners = np.vstack([
        _box_corners_np(float(p0[0]), float(p0[1]), L, W, float(heading)),
        _box_corners_np(float(p1[0]), float(p1[1]), L, W, float(heading)),
    ])
    hull = _convex_hull_np(corners)
    if hull.shape[0] < 3:
        return _oriented_box_masks_flat_gpu([p1[0]], [p1[1]], [L], [W], [heading], grid_x, grid_y, dtype)[0]
    return _point_in_ring_flat_gpu(hull, grid_x, grid_y, dtype, edge_chunk=edge_chunk)


def _rt_masks_gpu(
    sample: dict,
    lib: TubeLibrary,
    cfg: dict,
    times: np.ndarray,
    current_xy: np.ndarray,
    current_vel: np.ndarray,
    current_heading: np.ndarray,
    mu_bins: list[float],
    speed_bins: list[int],
    H: int,
    W: int,
    grid_x,
    grid_y,
    dtype,
):
    K = int(len(times))
    ego_idx = int(sample["ego_index"])
    ego_rt_t = torch.zeros((K, H * W), device=grid_x.device, dtype=torch.bool)
    others_rt_t = torch.zeros((K, H * W), device=grid_x.device, dtype=torch.bool)
    count_t = torch.zeros((K, H * W), device=grid_x.device, dtype=torch.int32)

    ego_speed_kph = float(np.linalg.norm(current_vel[ego_idx])) * 3.6
    ego_speed_bin = None

    edge_chunk = int(cfg.get("runtime", {}).get("gpu_polygon_edge_chunk", 256))

    with torch.no_grad():
        for i in range(int(sample["agent_count"])):
            speed_kph = float(np.linalg.norm(current_vel[i])) * 3.6
            is_ego = i == ego_idx
            rec = lib.select(_select_mu(is_ego, cfg), speed_kph, mu_bins, speed_bins)
            if is_ego:
                ego_speed_bin = rec.v0_kph
            rings_by_time = _get_tube_rings_at_times(rec, times)
            x, y = current_xy[i]
            hd = float(current_heading[i])
            for k, rings in enumerate(rings_by_time):
                mask = _rings_to_mask_flat_gpu(rings, float(x), float(y), hd, grid_x, grid_y, dtype, edge_chunk=edge_chunk)
                count_t[k].add_(mask.to(torch.int32))
                if is_ego:
                    ego_rt_t[k] |= mask
                else:
                    others_rt_t[k] |= mask

    ego_rt = ego_rt_t.view(K, H, W).to(torch.uint8).cpu().numpy()
    others_rt = others_rt_t.view(K, H, W).to(torch.uint8).cpu().numpy()
    overlap_count = count_t.view(K, H, W).cpu().numpy().astype(np.uint16)
    return ego_rt, others_rt, overlap_count, ego_speed_kph, ego_speed_bin


def _cv_masks_gpu(sample: dict, times: np.ndarray, H: int, W: int, grid_x, grid_y, dtype, cfg: dict):
    K = int(len(times))
    ego_idx = int(sample["ego_index"])
    xy = np.asarray(sample["current_xy"], dtype=np.float64)
    vel = np.asarray(sample["current_vel_xy"], dtype=np.float64)
    hd = np.asarray(sample["current_heading"], dtype=np.float64)
    size = np.asarray(sample["current_size_lw"], dtype=np.float64)

    ego_t = torch.zeros((K, H * W), device=grid_x.device, dtype=torch.bool)
    oth_t = torch.zeros((K, H * W), device=grid_x.device, dtype=torch.bool)
    cnt_t = torch.zeros((K, H * W), device=grid_x.device, dtype=torch.int32)
    edge_chunk = int(cfg.get("runtime", {}).get("gpu_polygon_edge_chunk", 256))

    with torch.no_grad():
        for i in range(int(sample["agent_count"])):
            heading = _cv_heading(float(hd[i]), vel[i], cfg)
            for k, t in enumerate(times):
                t1 = float(t)
                t0 = 0.0 if k == 0 else float(times[k - 1])
                p0 = xy[i] + vel[i] * t0
                p1 = xy[i] + vel[i] * t1
                mask = _cv_interval_mask_flat_gpu(p0, p1, float(size[i, 0]), float(size[i, 1]), heading, cfg, grid_x, grid_y, dtype, edge_chunk=edge_chunk)
                cnt_t[k].add_(mask.to(torch.int32))
                if i == ego_idx:
                    ego_t[k] |= mask
                else:
                    oth_t[k] |= mask

    ego = ego_t.view(K, H, W).to(torch.uint8).cpu().numpy()
    oth = oth_t.view(K, H, W).to(torch.uint8).cpu().numpy()
    cnt = cnt_t.view(K, H, W).cpu().numpy().astype(np.uint16)
    return ego, oth, cnt


def _subset_primitive_dict(prim: dict, idx: np.ndarray) -> dict:
    out = {}
    for key, val in prim.items():
        try:
            arr = np.asarray(val)
            if arr.shape and arr.shape[0] == len(idx) or (hasattr(val, "shape") and len(getattr(val, "shape", ())) > 0 and getattr(val, "shape", (None,))[0] >= int(np.max(idx)) + 1):
                out[key] = val[idx]
            else:
                out[key] = val
        except Exception:
            out[key] = val
    return out


def _compute_msr_gpu(
    sample: dict,
    primitive_lib: PrimitiveLibrary | None,
    others_rt: np.ndarray,
    lane_mask: np.ndarray,
    times: np.ndarray,
    grid_x,
    grid_y,
    dtype,
    cfg: dict,
) -> dict:
    if primitive_lib is None or not primitive_lib.available:
        return _empty_actionability_features()

    ego_idx = int(sample["ego_index"])
    xy = np.asarray(sample["current_xy"], dtype=float)
    vel = np.asarray(sample["current_vel_xy"], dtype=float)
    hd = np.asarray(sample["current_heading"], dtype=float)
    size = np.asarray(sample["current_size_lw"], dtype=float)

    speed_kph = float(np.linalg.norm(vel[ego_idx])) * 3.6
    mu_bins = [float(x) for x in cfg["tube"]["mu_bins"]]
    speed_bins = [int(x) for x in cfg["tube"]["speed_bins_kph"]]
    rec = primitive_lib.select(_select_mu(True, cfg), speed_kph, mu_bins, speed_bins)
    if rec is None or rec.count == 0:
        return _empty_actionability_features()

    prim = get_primitive_aligned(rec, times)
    M = int(prim["x"].shape[0])
    max_m = int(cfg.get("metrics", {}).get("max_msr_primitives", 0))
    if max_m > 0 and M > max_m:
        idx = np.linspace(0, M - 1, max_m).round().astype(int)
        for key in ["x", "y", "heading", "weight", "primitive_id", "action_family", "action_cost"]:
            if key in prim:
                prim[key] = prim[key][idx] if getattr(prim[key], "ndim", 1) > 1 else prim[key][idx]
        M = max_m

    require_drv = bool(cfg.get("metrics", {}).get("msr_require_drivable", True))
    min_drv_frac = float(cfg.get("metrics", {}).get("msr_min_drivable_fraction", 0.8))
    runtime_cfg = cfg.get("runtime", {})
    batch = int(runtime_cfg.get("msr_gpu_chunk_size", runtime_cfg.get("gpu_box_batch_size", 256)))
    batch = max(1, batch)

    K = int(len(times))
    N = int(grid_x.numel())
    device = grid_x.device

    with torch.no_grad():
        others_t = torch.as_tensor((others_rt.reshape(K, -1) > 0), device=device, dtype=torch.bool)
        lane_t = torch.as_tensor((lane_mask.reshape(-1) > 0), device=device, dtype=torch.bool)
        lane_any = bool(lane_t.any().item())

        px_t = torch.as_tensor(prim["x"], device=device, dtype=dtype)
        py_t = torch.as_tensor(prim["y"], device=device, dtype=dtype)
        ph_t = torch.as_tensor(prim["heading"], device=device, dtype=dtype)
        survive = torch.ones((M,), device=device, dtype=torch.bool)
        safe_slice_time = torch.ones((M, K), device=device, dtype=torch.bool)
        safe_cum_time = torch.ones((M, K), device=device, dtype=torch.bool)

        L = float(size[ego_idx, 0])
        W = float(size[ego_idx, 1])
        x0, y0, h0 = float(xy[ego_idx, 0]), float(xy[ego_idx, 1]), float(hd[ego_idx])
        c0, s0 = math.cos(h0), math.sin(h0)
        all_idx = torch.arange(M, device=device)

        for k in range(K):
            other_k = others_t[k].view(1, N)
            for start in range(0, M, batch):
                end = min(start + batch, M)
                idxs = all_idx[start:end]
                ax = px_t[idxs, k]
                ay = py_t[idxs, k]
                gx = float(x0) + float(c0) * ax - float(s0) * ay
                gy = float(y0) + float(s0) * ax + float(c0) * ay
                gh = float(h0) + ph_t[idxs, k]
                masks = _oriented_box_masks_flat_gpu(gx, gy, L, W, gh, grid_x, grid_y, dtype)
                bad = (masks & other_k).any(dim=1)
                if require_drv and lane_any:
                    denom = masks.sum(dim=1).clamp_min(1).to(dtype)
                    drv = (masks & lane_t.view(1, N)).sum(dim=1).to(dtype) / denom
                    bad |= drv < float(min_drv_frac)
                safe_slice_time[idxs[bad], k] = False
            survive &= safe_slice_time[:, k]
            safe_cum_time[:, k] = survive

        safe_slice_np = safe_slice_time.cpu().numpy().astype(bool)
        safe_cum_np = safe_cum_time.cpu().numpy().astype(bool)

    weights = np.asarray(prim["weight"], dtype=float)
    return _summarize_actionability(safe_slice_np, safe_cum_np, weights, times, prim, cfg)


# ---------------------------------------------------------------------------
# Original CPU rasterization fallbacks
# ---------------------------------------------------------------------------

def _cv_masks_cpu(sample: dict, times: np.ndarray, xx: np.ndarray, yy: np.ndarray, cfg: dict):
    K, H, W = len(times), *xx.shape
    ego_idx = int(sample["ego_index"])
    xy = np.asarray(sample["current_xy"], dtype=np.float64)
    vel = np.asarray(sample["current_vel_xy"], dtype=np.float64)
    hd = np.asarray(sample["current_heading"], dtype=np.float64)
    size = np.asarray(sample["current_size_lw"], dtype=np.float64)
    ego = np.zeros((K, H, W), dtype=np.uint8)
    oth = np.zeros((K, H, W), dtype=np.uint8)
    cnt = np.zeros((K, H, W), dtype=np.uint16)
    for i in range(sample["agent_count"]):
        heading = _cv_heading(float(hd[i]), vel[i], cfg)
        for k, t in enumerate(times):
            t1 = float(t)
            t0 = 0.0 if k == 0 else float(times[k - 1])
            p0 = xy[i] + vel[i] * t0
            p1 = xy[i] + vel[i] * t1
            poly = _cv_interval_polygon(p0, p1, float(size[i, 0]), float(size[i, 1]), heading, cfg)
            m = geometry_to_mask(poly, xx, yy).astype(np.uint8)
            cnt[k] += m.astype(np.uint16)
            if i == ego_idx:
                ego[k] = np.maximum(ego[k], m)
            else:
                oth[k] = np.maximum(oth[k], m)
    return ego, oth, cnt


def _compute_msr_cpu(sample: dict, primitive_lib: PrimitiveLibrary | None, others_rt: np.ndarray, lane_mask: np.ndarray, times: np.ndarray, xx: np.ndarray, yy: np.ndarray, cfg: dict) -> dict:
    if primitive_lib is None or not primitive_lib.available:
        return _empty_actionability_features()
    ego_idx = int(sample["ego_index"])
    xy = np.asarray(sample["current_xy"], dtype=float)
    vel = np.asarray(sample["current_vel_xy"], dtype=float)
    hd = np.asarray(sample["current_heading"], dtype=float)
    size = np.asarray(sample["current_size_lw"], dtype=float)
    speed_kph = float(np.linalg.norm(vel[ego_idx])) * 3.6
    mu_bins = [float(x) for x in cfg["tube"]["mu_bins"]]
    speed_bins = [int(x) for x in cfg["tube"]["speed_bins_kph"]]
    rec = primitive_lib.select(_select_mu(True, cfg), speed_kph, mu_bins, speed_bins)
    if rec is None or rec.count == 0:
        return _empty_actionability_features()
    prim = get_primitive_aligned(rec, times)
    M = prim["x"].shape[0]
    max_m = int(cfg.get("metrics", {}).get("max_msr_primitives", 0))
    if max_m > 0 and M > max_m:
        idx = np.linspace(0, M - 1, max_m).round().astype(int)
        for key in ["x", "y", "heading", "weight", "primitive_id", "action_family", "action_cost"]:
            if key in prim:
                prim[key] = prim[key][idx] if getattr(prim[key], "ndim", 1) > 1 else prim[key][idx]
        M = max_m
    require_drv = bool(cfg.get("metrics", {}).get("msr_require_drivable", True))
    min_drv_frac = float(cfg.get("metrics", {}).get("msr_min_drivable_fraction", 0.8))
    lane_bool = lane_mask.astype(bool)
    safe_slice_time = np.ones((M, len(times)), dtype=bool)
    safe_cum_time = np.ones((M, len(times)), dtype=bool)
    survive = np.ones(M, dtype=bool)
    weights = np.asarray(prim["weight"], dtype=float)
    L, W = float(size[ego_idx, 0]), float(size[ego_idx, 1])
    x0, y0, h0 = float(xy[ego_idx, 0]), float(xy[ego_idx, 1]), float(hd[ego_idx])
    for k in range(len(times)):
        for m in range(M):
            gx, gy = _transform_local_point(float(prim["x"][m, k]), float(prim["y"][m, k]), x0, y0, h0)
            gh = h0 + float(prim["heading"][m, k])
            mask = oriented_box_mask_np(gx, gy, L, W, gh, xx, yy)
            bad = False
            if np.any(mask & (others_rt[k] > 0)):
                bad = True
            elif require_drv and lane_bool.any():
                denom = max(int(mask.sum()), 1)
                drv_frac = float((mask & lane_bool).sum()) / denom
                if drv_frac < min_drv_frac:
                    bad = True
            if bad:
                safe_slice_time[m, k] = False
        survive &= safe_slice_time[:, k]
        safe_cum_time[:, k] = survive
    return _summarize_actionability(safe_slice_time, safe_cum_time, weights, times, prim, cfg)


# ---------------------------------------------------------------------------
# Metrics and public pipeline entrypoint
# ---------------------------------------------------------------------------

def _basic_reachability_metrics(prefix: str, ego: np.ndarray, oth: np.ndarray, cnt: np.ndarray, lane_mask: np.ndarray, times: np.ndarray, weights: np.ndarray, cell_area: float, cfg: dict) -> tuple[dict, dict]:
    overlap = (cnt >= 2).astype(np.uint8)
    conflict = ((ego > 0) & (oth > 0)).astype(np.uint8)
    free = ((ego > 0) & (oth == 0)).astype(np.uint8)
    lane_bool = lane_mask.astype(bool)[None, :, :]
    ego_drv = ((ego > 0) & lane_bool).astype(np.uint8)
    free_drv = ((free > 0) & lane_bool).astype(np.uint8)

    overlap_area, gtoa = _weighted_areas(overlap, weights, cell_area)
    conflict_area, weighted_conf = _weighted_areas(conflict, weights, cell_area)
    ego_area, weighted_ego = _weighted_areas(ego, weights, cell_area)
    free_area, weighted_free = _weighted_areas(free, weights, cell_area)
    ego_drv_area, weighted_ego_drv = _weighted_areas(ego_drv, weights, cell_area)
    free_drv_area, weighted_free_drv = _weighted_areas(free_drv, weights, cell_area)
    union_mask = (cnt > 0).astype(np.uint8)
    union_area, weighted_union = _weighted_areas(union_mask, weights, cell_area)

    rcr = _safe_div(weighted_conf, weighted_ego)
    rfr = _safe_div(weighted_free, weighted_ego)
    rfr_drv = _safe_div(weighted_free_drv, weighted_ego_drv)
    tfrc, c_time = _tfrc_from_area(times, conflict_area, ego_area, cfg)
    oce, oce_norm, max_c, mean_c = _overlap_entropy(cnt, cfg)
    gtoa_norm_roi = _safe_div(gtoa, (cnt.shape[1] * cnt.shape[2] * cell_area) * float(np.sum(weights)))
    gtoa_norm_union = _safe_div(gtoa, weighted_union)

    lam_g = float(cfg.get("redi", {}).get("density_lambda_gtoa", 0.7))
    lam_o = float(cfg.get("redi", {}).get("density_lambda_oce", 0.3))
    c_density = float(np.clip(lam_g * gtoa_norm_union + lam_o * oce_norm, 0.0, 1.0))

    feats = {
        f"{prefix}weighted_overlap_area_m2": gtoa,
        f"{prefix}weighted_union_area_m2": weighted_union,
        f"{prefix}gtoa_norm_roi": gtoa_norm_roi,
        f"{prefix}gtoa_norm_union": gtoa_norm_union,
        f"{prefix}weighted_ego_area_m2": weighted_ego,
        f"{prefix}weighted_ego_conflict_area_m2": weighted_conf,
        f"{prefix}weighted_free_area_m2": weighted_free,
        f"{prefix}rcr": rcr,
        f"{prefix}rfr": rfr,
        f"{prefix}c_space": 1.0 - rfr,
        f"{prefix}weighted_ego_drv_area_m2": weighted_ego_drv,
        f"{prefix}weighted_free_drv_area_m2": weighted_free_drv,
        f"{prefix}rfr_drv": rfr_drv,
        f"{prefix}c_space_drv": 1.0 - rfr_drv,
        f"{prefix}tfrc_s": tfrc,
        f"{prefix}c_time": c_time,
        f"{prefix}oce": oce,
        f"{prefix}oce_norm": oce_norm,
        f"{prefix}max_overlap_count": max_c,
        f"{prefix}mean_overlap_count_nonzero": mean_c,
        f"{prefix}c_density": c_density,
    }
    maps = {"overlap": overlap, "ego_conflict": conflict, "free_space": free, "ego_drv": ego_drv, "free_drv": free_drv}
    return feats, maps


def _redi(feats: dict, use_msr: bool, cfg: dict, prefix: str = "") -> float:
    r = cfg.get("redi", {})
    a_s = float(r.get("alpha_space", 0.4))
    a_t = float(r.get("alpha_time", 0.2))
    a_m = float(r.get("alpha_maneuver", 0.3))
    a_d = float(r.get("alpha_density", 0.1))
    c_space = float(feats.get(f"{prefix}c_space_drv", feats.get(f"{prefix}c_space", 0.0)))
    c_time = float(feats.get(f"{prefix}c_time", 0.0))
    c_density = float(feats.get(f"{prefix}c_density", 0.0))
    if use_msr and np.isfinite(feats.get("c_maneuver", np.nan)):
        val = a_s * c_space + a_t * c_time + a_m * float(feats["c_maneuver"]) + a_d * c_density
        return float(np.clip(val, 0.0, 1.0))
    denom = a_s + a_t + a_d
    val = (a_s * c_space + a_t * c_time + a_d * c_density) / max(denom, 1e-9)
    return float(np.clip(val, 0.0, 1.0))


def _rt_masks_cpu(
    sample: dict,
    lib: TubeLibrary,
    cfg: dict,
    times: np.ndarray,
    current_xy: np.ndarray,
    current_vel: np.ndarray,
    current_heading: np.ndarray,
    mu_bins: list[float],
    speed_bins: list[int],
    H: int,
    W: int,
    xx: np.ndarray,
    yy: np.ndarray,
):
    ego_idx = int(sample["ego_index"])
    ego_rt = np.zeros((len(times), H, W), dtype=np.uint8)
    others_rt = np.zeros((len(times), H, W), dtype=np.uint8)
    overlap_count = np.zeros((len(times), H, W), dtype=np.uint16)
    ego_speed_kph = float(np.linalg.norm(current_vel[ego_idx])) * 3.6
    ego_speed_bin = None

    for i in range(sample["agent_count"]):
        speed_kph = float(np.linalg.norm(current_vel[i])) * 3.6
        is_ego = (i == ego_idx)
        rec = lib.select(_select_mu(is_ego, cfg), speed_kph, mu_bins, speed_bins)
        if is_ego:
            ego_speed_bin = rec.v0_kph
        x, y = current_xy[i]
        hd = current_heading[i]
        for k, t in enumerate(times):
            poly = get_record_slice_at_time(rec, float(t))
            g = _transform_local_poly(poly, x, y, hd)
            m = geometry_to_mask(g, xx, yy).astype(np.uint8)
            overlap_count[k] += m.astype(np.uint16)
            if is_ego:
                ego_rt[k] = np.maximum(ego_rt[k], m)
            else:
                others_rt[k] = np.maximum(others_rt[k], m)
    return ego_rt, others_rt, overlap_count, ego_speed_kph, ego_speed_bin


def sample_to_bev_tensor_and_features(sample: dict, lib: TubeLibrary, cfg: dict, device: str = "cpu", primitive_lib: PrimitiveLibrary | None = None, return_tensors: bool = True):
    xs, ys, xx, yy = make_grid(cfg)
    H, W = xx.shape
    K = int(cfg["dataset"]["max_future_steps"])
    cell_area = float(cfg["bev"]["resolution_m"]) ** 2
    mu_bins = [float(x) for x in cfg["tube"]["mu_bins"]]
    speed_bins = [int(x) for x in cfg["tube"]["speed_bins_kph"]]

    ego_idx = int(sample["ego_index"])
    current_xy = np.asarray(sample["current_xy"], dtype=np.float64)
    current_vel = np.asarray(sample["current_vel_xy"], dtype=np.float64)
    current_heading = np.asarray(sample["current_heading"], dtype=np.float64)

    raw_times = np.asarray(sample["times_s"], dtype=np.float64)
    times = raw_times[1 : K + 1]
    if len(times) < K:
        times = np.arange(1, K + 1, dtype=float) * float(cfg["labels"].get("dt_s", 0.1))
    weights = np.exp(-times / float(cfg.get("metrics", {}).get("tau_time_s", 1.0)))

    allow_gpu = bool(cfg.get("runtime", {}).get("gpu_rasterization", True))
    torch_device = _resolve_torch_device(device) if allow_gpu else None
    use_gpu = torch_device is not None

    if use_gpu:
        dtype = _torch_dtype(cfg)
        grid_x, grid_y, H, W = _get_torch_grid(cfg, xx, yy, torch_device, dtype)
        ego_rt, others_rt, overlap_count, ego_speed_kph, ego_speed_bin = _rt_masks_gpu(
            sample, lib, cfg, times, current_xy, current_vel, current_heading, mu_bins, speed_bins, H, W, grid_x, grid_y, dtype
        )
    else:
        ego_rt, others_rt, overlap_count, ego_speed_kph, ego_speed_bin = _rt_masks_cpu(
            sample, lib, cfg, times, current_xy, current_vel, current_heading, mu_bins, speed_bins, H, W, xx, yy
        )

    # Map lane mask is intentionally kept on the robust CPU Shapely path because
    # it is a low-frequency arbitrary-polyline operation.
    lane_mask = _make_lane_mask(sample, cfg, xx, yy)

    rt_feats, rt_maps = _basic_reachability_metrics("", ego_rt, others_rt, overlap_count, lane_mask, times, weights, cell_area, cfg)

    if use_gpu:
        msr_feats = _compute_msr_gpu(sample, primitive_lib, others_rt, lane_mask, times, grid_x, grid_y, dtype, cfg)
    else:
        msr_feats = _compute_msr_cpu(sample, primitive_lib, others_rt, lane_mask, times, xx, yy, cfg)

    feats = {**rt_feats, **msr_feats}
    feats["redi_no_msr"] = _redi(feats, use_msr=False, cfg=cfg)
    feats["redi_full"] = _redi(feats, use_msr=True, cfg=cfg)
    # v1.1 actionability-specific REDI: deliberately not an alias of redi_full.
    acfg = cfg.get("actionability", {})
    aw = acfg.get("redi_weights", {})
    w_space = float(aw.get("space", 0.20))
    w_time = float(aw.get("time", 0.15))
    w_cum = float(aw.get("cum_action_loss", 0.25))
    w_comfort = float(aw.get("comfort_loss", 0.15))
    w_collapse = float(aw.get("collapse", 0.15))
    w_density = float(aw.get("density", 0.10))
    horizon = float(cfg.get("tube", {}).get("horizon_s", 3.0))
    dt = float(cfg.get("tube", {}).get("query_dt_s", cfg.get("labels", {}).get("dt_s", 0.1)))
    collapse_scale = float(acfg.get("collapse_rate_scale", 1.0))
    c_space = float(feats.get("c_space_drv", feats.get("c_space", 0.0)))
    c_time = float(feats.get("c_time", 0.0))
    c_cum = float(np.clip(1.0 - float(feats.get("asr_cum_final", feats.get("asr", 1.0))), 0.0, 1.0))
    comfort = float(feats.get("comfort_asr", np.nan))
    c_comfort = float(np.clip(1.0 - comfort, 0.0, 1.0)) if np.isfinite(comfort) and comfort >= 0 else c_cum
    c_collapse = float(np.clip(float(feats.get("collapse_rate_max_per_s", feats.get("collapse_rate_per_s", 0.0))) / max(collapse_scale, 1e-6), 0.0, 1.0))
    ttad = float(feats.get("ttad_s", horizon + dt))
    c_tta = float(np.clip(1.0 - ttad / max(horizon + dt, 1e-6), 0.0, 1.0))
    # Let time-to-depletion modulate cumulative action loss rather than duplicating c_time.
    c_action = float(np.clip(0.5 * c_cum + 0.5 * max(c_cum, c_tta), 0.0, 1.0))
    c_density = float(feats.get("c_density", 0.0))
    denom = max(w_space + w_time + w_cum + w_comfort + w_collapse + w_density, 1e-9)
    redi_actionability = (
        w_space * c_space + w_time * c_time + w_cum * c_action +
        w_comfort * c_comfort + w_collapse * c_collapse + w_density * c_density
    ) / denom
    feats["redi_actionability"] = float(np.clip(redi_actionability, 0.0, 1.0))
    feats["redi_actionability_delta"] = float(feats["redi_actionability"] - feats["redi_full"])

    if use_gpu:
        cv_ego, cv_oth, cv_count = _cv_masks_gpu(sample, times, H, W, grid_x, grid_y, dtype, cfg)
    else:
        cv_ego, cv_oth, cv_count = _cv_masks_cpu(sample, times, xx, yy, cfg)

    cv_feats, _ = _basic_reachability_metrics("cv_", cv_ego, cv_oth, cv_count, lane_mask, times, weights, cell_area, cfg)
    feats.update(cv_feats)
    feats.update(_compute_current_distance_ttc(sample))

    # Compatibility aliases used by the paper-analysis scripts.
    if "msr" in feats and "msr_w" not in feats:
        feats["msr_w"] = feats["msr"]
    if "redi_full" in feats and "redi_fixed" not in feats:
        feats["redi_fixed"] = feats["redi_full"]
    if "oce_norm" in feats and "overlap_count_entropy_norm" not in feats:
        feats["overlap_count_entropy_norm"] = feats["oce_norm"]
    if "weighted_overlap_area_m2" in feats and "gtoa_m2" not in feats:
        feats["gtoa_m2"] = feats["weighted_overlap_area_m2"]

    feats.update({
        "agent_count": int(sample["agent_count"]),
        "ego_speed_kph": ego_speed_kph,
        "ego_speed_bin_kph": float(ego_speed_bin if ego_speed_bin is not None else -1),
        "ego_speed_mismatch_abs_kph": abs(ego_speed_kph - float(ego_speed_bin if ego_speed_bin is not None else 0)),
        "ego_speed_bin_abs_error_kph": abs(ego_speed_kph - float(ego_speed_bin if ego_speed_bin is not None else 0)),
        "mu_ego": _select_mu(True, cfg),
        "mu_other": _select_mu(False, cfg),
    })

    # Round for compact CSV; keep NaN intact.
    feats = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) and np.isfinite(v) else v) for k, v in feats.items()}

    tensors = {}
    if return_tensors:
        if use_gpu:
            current_boxes = _make_current_boxes_mask_gpu(sample, H, W, grid_x, grid_y, dtype, cfg)
        else:
            current_boxes = _make_current_boxes_mask(sample, xx, yy)
        tensors = {
            "ego_rt": ego_rt.astype(np.uint8),
            "others_rt": others_rt.astype(np.uint8),
            "overlap": rt_maps["overlap"].astype(np.uint8),
            "ego_conflict": rt_maps["ego_conflict"].astype(np.uint8),
            "free_space": rt_maps["free_space"].astype(np.uint8),
            "ego_drv": rt_maps["ego_drv"].astype(np.uint8),
            "free_drv": rt_maps["free_drv"].astype(np.uint8),
            "current_boxes": current_boxes.astype(np.uint8),
            "lane_mask": lane_mask.astype(np.uint8),
            "overlap_count": overlap_count.astype(np.uint16),
            "times_s": times.astype(np.float32),
            "label": np.asarray([int(sample["label"]["label_id"])], dtype=np.int64),
            "risk_score": np.asarray([float(sample["label"]["risk_score"])], dtype=np.float32),
        }
    return tensors, feats
