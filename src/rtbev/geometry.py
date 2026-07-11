from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate, translate
from shapely.ops import unary_union


def oriented_box_polygon(cx: float, cy: float, length: float, width: float, heading: float) -> Polygon:
    hl = float(length) / 2.0
    hw = float(width) / 2.0
    poly = Polygon([(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)])
    poly = rotate(poly, float(heading), origin=(0.0, 0.0), use_radians=True)
    poly = translate(poly, xoff=float(cx), yoff=float(cy))
    return poly


def line_to_buffer_polygon(points_xy: np.ndarray, width_m: float):
    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < 2:
        return None
    ls = LineString(pts)
    if ls.is_empty:
        return None
    return ls.buffer(width_m, cap_style=2, join_style=2)


def union_polygons(polys):
    polys = [p for p in polys if p is not None and (not p.is_empty)]
    if not polys:
        return None
    u = unary_union(polys)
    if u.is_empty:
        return None
    return u
