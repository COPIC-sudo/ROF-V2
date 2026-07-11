from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from matplotlib.path import Path as MplPath
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

try:  # Torch is optional for the old CPU-only pipeline, required for CUDA acceleration.
    import torch
except Exception:  # pragma: no cover
    torch = None


_GRID_CACHE: dict[tuple[float, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
_CPU_PTS_CACHE: dict[tuple[int, int, float, float, float, float], np.ndarray] = {}


def make_grid(cfg: dict):
    """Create and cache the BEV grid.

    The original implementation rebuilt the same xs/ys/xx/yy arrays for every
    sample.  They are immutable in normal use, so caching is safe and removes a
    small but repeated allocation cost.
    """
    bev = cfg["bev"]
    res = float(bev["resolution_m"])
    key = (float(bev["x_min"]), float(bev["x_max"]), float(bev["y_min"]), float(bev["y_max"]), res)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    xs = np.arange(float(bev["x_min"]) + 0.5 * res, float(bev["x_max"]), res, dtype=np.float64)
    ys = np.arange(float(bev["y_min"]) + 0.5 * res, float(bev["y_max"]), res, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    _GRID_CACHE[key] = (xs, ys, xx, yy)
    return xs, ys, xx, yy


def _points_from_grid(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    H, W = xx.shape
    key = (H, W, float(xx[0, 0]), float(xx[-1, -1]), float(yy[0, 0]), float(yy[-1, -1]))
    pts = _CPU_PTS_CACHE.get(key)
    if pts is None:
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        _CPU_PTS_CACHE[key] = pts
    return pts


def _poly_to_mask_single(poly: Polygon, pts: np.ndarray, H: int, W: int) -> np.ndarray:
    ext = np.asarray(poly.exterior.coords, dtype=np.float64)
    mask = MplPath(ext).contains_points(pts)
    for ring in poly.interiors:
        hole = np.asarray(ring.coords, dtype=np.float64)
        mask &= ~MplPath(hole).contains_points(pts)
    return mask.reshape(H, W)


def geometry_to_mask(geom, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """CPU Shapely/Matplotlib rasterization kept for backward compatibility."""
    H, W = xx.shape
    pts = _points_from_grid(xx, yy)
    mask = np.zeros((H, W), dtype=bool)
    if geom is None or geom.is_empty:
        return mask
    if isinstance(geom, Polygon):
        return _poly_to_mask_single(geom, pts, H, W)
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            mask |= _poly_to_mask_single(poly, pts, H, W)
        return mask
    try:
        for poly in geom.geoms:
            if isinstance(poly, Polygon):
                mask |= _poly_to_mask_single(poly, pts, H, W)
        return mask
    except Exception:
        return mask


@dataclass(frozen=True)
class TorchRasterContext:
    xx: Any
    yy: Any
    x_flat: Any
    y_flat: Any
    shape: tuple[int, int]
    device: Any
    dtype: Any


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Torch is not installed. Install a CUDA-enabled PyTorch build or use --device cpu.")


def torch_dtype_from_string(dtype: str | None):
    _require_torch()
    name = str(dtype or "float64").lower()
    if name in {"float64", "double", "fp64"}:
        return torch.float64
    if name in {"float32", "single", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {dtype!r}; use float32 or float64")


def make_torch_raster_context(xx: np.ndarray, yy: np.ndarray, device: str | Any, dtype: str | Any = "float64") -> TorchRasterContext:
    _require_torch()
    dev = torch.device(device)
    dt = torch_dtype_from_string(dtype) if isinstance(dtype, str) or dtype is None else dtype
    xx_t = torch.as_tensor(xx, device=dev, dtype=dt)
    yy_t = torch.as_tensor(yy, device=dev, dtype=dt)
    return TorchRasterContext(
        xx=xx_t,
        yy=yy_t,
        x_flat=xx_t.reshape(-1),
        y_flat=yy_t.reshape(-1),
        shape=tuple(xx.shape),
        device=dev,
        dtype=dt,
    )


def _iter_polygon_parts(geom) -> Iterable[Polygon]:
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            if isinstance(poly, Polygon) and not poly.is_empty:
                yield poly
        return
    if isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from _iter_polygon_parts(g)
        return
    try:
        for g in geom.geoms:
            if isinstance(g, Polygon):
                yield g
    except Exception:
        return


def _closed_ring_xy(coords) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        return None
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[0]])
    if arr.shape[0] < 4:
        return None
    return arr[:, 0].copy(), arr[:, 1].copy()


def _grid_points_in_local_frame(ctx: TorchRasterContext, x: float, y: float, heading: float):
    # Inverse of local->global transform: R(-heading) * ([X,Y] - [x,y]).
    if abs(float(x)) < 1e-15 and abs(float(y)) < 1e-15 and abs(float(heading)) < 1e-15:
        return ctx.x_flat, ctx.y_flat
    c = torch.cos(torch.as_tensor(float(heading), device=ctx.device, dtype=ctx.dtype))
    s = torch.sin(torch.as_tensor(float(heading), device=ctx.device, dtype=ctx.dtype))
    dx = ctx.x_flat - float(x)
    dy = ctx.y_flat - float(y)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return local_x, local_y


def _ring_contains_points_torch(px, py, xs_np: np.ndarray, ys_np: np.ndarray, ctx: TorchRasterContext, edge_chunk: int = 256):
    """Even-odd point-in-ring test on CUDA/torch.

    px/py are flattened grid coordinates in the same coordinate frame as the ring.
    The implementation mirrors matplotlib.path.contains_points semantics for grid
    centre inclusion; boundary cells may differ only at exact floating-point ties.
    """
    _require_torch()
    if xs_np.shape[0] < 4:
        return torch.zeros_like(px, dtype=torch.bool)
    xs = torch.as_tensor(xs_np, device=ctx.device, dtype=ctx.dtype)
    ys = torch.as_tensor(ys_np, device=ctx.device, dtype=ctx.dtype)
    x0 = xs[:-1]
    y0 = ys[:-1]
    x1 = xs[1:]
    y1 = ys[1:]
    inside = torch.zeros(px.shape, device=ctx.device, dtype=torch.bool)
    px_b = px.unsqueeze(0)
    py_b = py.unsqueeze(0)
    n_edges = int(x0.numel())
    edge_chunk = max(8, int(edge_chunk))
    for start in range(0, n_edges, edge_chunk):
        end = min(start + edge_chunk, n_edges)
        x0c = x0[start:end].unsqueeze(1)
        y0c = y0[start:end].unsqueeze(1)
        x1c = x1[start:end].unsqueeze(1)
        y1c = y1[start:end].unsqueeze(1)
        cond = (y0c > py_b) != (y1c > py_b)
        denom = y1c - y0c
        safe_denom = torch.where(denom.abs() > 1e-30, denom, torch.ones_like(denom))
        x_intersect = (x1c - x0c) * (py_b - y0c) / safe_denom + x0c
        crossings = cond & (px_b < x_intersect)
        odd = (crossings.sum(dim=0) & 1).to(torch.bool)
        inside ^= odd
    return inside


def _polygon_contains_points_torch(poly: Polygon, px, py, ctx: TorchRasterContext, edge_chunk: int = 256):
    ext = _closed_ring_xy(poly.exterior.coords)
    if ext is None:
        return torch.zeros(px.shape, device=ctx.device, dtype=torch.bool)
    xs, ys = ext
    inside = _ring_contains_points_torch(px, py, xs, ys, ctx, edge_chunk=edge_chunk)
    if not bool(inside.any().item()):
        return inside
    for ring in poly.interiors:
        hole = _closed_ring_xy(ring.coords)
        if hole is None:
            continue
        hx, hy = hole
        inside &= ~_ring_contains_points_torch(px, py, hx, hy, ctx, edge_chunk=edge_chunk)
    return inside


def geometry_to_mask_torch(
    geom,
    ctx: TorchRasterContext,
    *,
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    edge_chunk: int = 256,
):
    """Rasterize a Shapely geometry onto the BEV grid using torch/CUDA.

    When x/y/heading are supplied, ``geom`` is interpreted in the local frame and
    rasterized after the same local->global transform used by the original CPU
    Shapely affine path.  This avoids constructing a transformed Shapely object
    for every agent/time slice.
    """
    _require_torch()
    H, W = ctx.shape
    out = torch.zeros((H * W,), device=ctx.device, dtype=torch.bool)
    parts = list(_iter_polygon_parts(geom))
    if not parts:
        return out.view(H, W)
    px, py = _grid_points_in_local_frame(ctx, x, y, heading)
    for poly in parts:
        out |= _polygon_contains_points_torch(poly, px, py, ctx, edge_chunk=edge_chunk)
    return out.view(H, W)


def oriented_box_mask_np(cx: float, cy: float, length: float, width: float, heading: float, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Analytic CPU mask for a rotated rectangle, matching grid-centre rasterization."""
    c = np.cos(float(heading))
    s = np.sin(float(heading))
    dx = xx - float(cx)
    dy = yy - float(cy)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return (np.abs(local_x) <= 0.5 * float(length)) & (np.abs(local_y) <= 0.5 * float(width))


def oriented_box_mask_torch(cx: float, cy: float, length: float, width: float, heading: float, ctx: TorchRasterContext):
    _require_torch()
    c = torch.cos(torch.as_tensor(float(heading), device=ctx.device, dtype=ctx.dtype))
    s = torch.sin(torch.as_tensor(float(heading), device=ctx.device, dtype=ctx.dtype))
    dx = ctx.xx - float(cx)
    dy = ctx.yy - float(cy)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return (local_x.abs() <= 0.5 * float(length)) & (local_y.abs() <= 0.5 * float(width))


def oriented_box_union_mask_torch(
    centers_xy: np.ndarray,
    sizes_lw: np.ndarray,
    headings: np.ndarray,
    ctx: TorchRasterContext,
    chunk_size: int = 64,
):
    """Union mask for many oriented rectangles on CUDA/torch."""
    _require_torch()
    centers_xy = np.asarray(centers_xy, dtype=np.float64)
    sizes_lw = np.asarray(sizes_lw, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    H, W = ctx.shape
    out = torch.zeros((H, W), device=ctx.device, dtype=torch.bool)
    n = int(len(centers_xy))
    if n == 0:
        return out
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        cx = torch.as_tensor(centers_xy[start:end, 0], device=ctx.device, dtype=ctx.dtype).view(-1, 1, 1)
        cy = torch.as_tensor(centers_xy[start:end, 1], device=ctx.device, dtype=ctx.dtype).view(-1, 1, 1)
        L = torch.as_tensor(sizes_lw[start:end, 0], device=ctx.device, dtype=ctx.dtype).view(-1, 1, 1)
        Wd = torch.as_tensor(sizes_lw[start:end, 1], device=ctx.device, dtype=ctx.dtype).view(-1, 1, 1)
        hd = torch.as_tensor(headings[start:end], device=ctx.device, dtype=ctx.dtype).view(-1, 1, 1)
        c = torch.cos(hd)
        s = torch.sin(hd)
        dx = ctx.xx.unsqueeze(0) - cx
        dy = ctx.yy.unsqueeze(0) - cy
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        masks = (local_x.abs() <= 0.5 * L) & (local_y.abs() <= 0.5 * Wd)
        out |= masks.any(dim=0)
    return out


def rasterize_oriented_boxes_flat_torch(
    cx,
    cy,
    length,
    width,
    heading,
    ctx: TorchRasterContext,
):
    """Return [B, N] masks for B oriented boxes over flattened BEV cells.

    Inputs are torch tensors on ctx.device with shape [B].  This routine is used
    by the batched CUDA MSR implementation.
    """
    _require_torch()
    cx = cx.reshape(-1, 1)
    cy = cy.reshape(-1, 1)
    length = length.reshape(-1, 1)
    width = width.reshape(-1, 1)
    heading = heading.reshape(-1, 1)
    c = torch.cos(heading)
    s = torch.sin(heading)
    dx = ctx.x_flat.unsqueeze(0) - cx
    dy = ctx.y_flat.unsqueeze(0) - cy
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return (local_x.abs() <= 0.5 * length) & (local_y.abs() <= 0.5 * width)
