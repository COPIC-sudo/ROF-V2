from __future__ import annotations

import numpy as np
from shapely.geometry import Point, LineString, MultiPoint

try:
    import alphashape as _alphashape  # type: ignore
    _HAS_ALPHASHAPE = True
except Exception:
    _alphashape = None
    _HAS_ALPHASHAPE = False


def _nn_median_distance(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]
        nn = nn[np.isfinite(nn)]
        return float(np.median(nn)) if len(nn) else 0.0
    except Exception:
        dmin = []
        for i in range(len(pts)):
            di = np.sqrt(((pts[i] - pts) ** 2).sum(axis=1))
            di[i] = np.inf
            dmin.append(np.min(di))
        return float(np.median(dmin)) if dmin else 0.0


def _alpha_shape_fallback(pts: np.ndarray, alpha: float):
    from scipy.spatial import Delaunay
    from shapely.geometry import MultiLineString
    from shapely.ops import polygonize, unary_union

    if alpha <= 0.0:
        return MultiPoint(pts).convex_hull
    tri = Delaunay(pts)
    edge_count = {}

    def add_edge(i, j):
        a, b = (i, j) if i < j else (j, i)
        edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

    for s in tri.simplices:
        ia, ib, ic = int(s[0]), int(s[1]), int(s[2])
        A, B, C = pts[ia], pts[ib], pts[ic]
        a = float(np.linalg.norm(B - C))
        b = float(np.linalg.norm(A - C))
        c = float(np.linalg.norm(A - B))
        area2 = float(abs(np.cross(B - A, C - A)))
        if area2 <= 1e-12:
            continue
        area = area2 / 2.0
        r = (a * b * c) / (4.0 * area)
        if r < (1.0 / alpha):
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return MultiPoint(pts).convex_hull
    lines = [(tuple(pts[i]), tuple(pts[j])) for i, j in boundary_edges]
    mls = MultiLineString(lines)
    polys = list(polygonize(mls))
    if not polys:
        return MultiPoint(pts).convex_hull
    return unary_union(polys)


def alpha_shape_polygon(points_xy, alpha_cfg):
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return None
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return None
    pts = np.unique(pts, axis=0)

    buf = float(alpha_cfg.get("buffer_eps_m", 0.05))
    minA = float(alpha_cfg.get("min_area_m2", 1e-3))
    thr = int(alpha_cfg.get("min_pts_alpha", 8))
    aval = alpha_cfg.get("value", "auto")

    if len(pts) == 1:
        return Point(pts[0]).buffer(buf)
    if len(pts) == 2:
        return LineString(pts).buffer(buf)
    if np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        return MultiPoint(pts).convex_hull.buffer(buf)

    try:
        if len(pts) >= thr:
            if aval is None or str(aval).lower() == "auto":
                if _HAS_ALPHASHAPE:
                    try:
                        aval = _alphashape.optimizealpha(pts)
                    except Exception:
                        aval = 0.0
                else:
                    d_med = _nn_median_distance(pts)
                    aval = 1.0 / max(1.5 * d_med, 1e-3)

            if _HAS_ALPHASHAPE:
                shp = _alphashape.alphashape(pts, float(aval))
            else:
                shp = _alpha_shape_fallback(pts, float(aval))
        else:
            shp = MultiPoint(pts).convex_hull
    except Exception:
        shp = MultiPoint(pts).convex_hull

    if shp.is_empty:
        return None
    if shp.geom_type not in ("Polygon", "MultiPolygon"):
        shp = shp.buffer(buf)
    elif getattr(shp, "area", 0.0) < minA:
        shp = shp.buffer(buf)
    return shp
