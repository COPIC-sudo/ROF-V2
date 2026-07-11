from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points_global_to_local(points_xy: np.ndarray, origin_xy: np.ndarray, heading: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    o = np.asarray(origin_xy, dtype=np.float64).reshape(1, 2)
    R = rotation_matrix(-heading)
    return (pts - o) @ R.T


def transform_vectors_global_to_local(vectors_xy: np.ndarray, heading: float) -> np.ndarray:
    vec = np.asarray(vectors_xy, dtype=np.float64)
    R = rotation_matrix(-heading)
    return vec @ R.T


def nearest_value(x: float, candidates: Iterable[float]) -> float:
    arr = np.asarray(list(candidates), dtype=np.float64)
    return float(arr[np.argmin(np.abs(arr - float(x)))])
