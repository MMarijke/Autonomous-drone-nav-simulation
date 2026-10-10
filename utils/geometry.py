"""Small vector / polyline helpers shared by planners and guidance."""
from __future__ import annotations

import numpy as np


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return np.asarray(v, dtype=float) / n


def polyline_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def cumulative_length(path: np.ndarray) -> np.ndarray:
    """Arc length at every vertex; always returns an array of len(path)."""
    if len(path) < 2:
        return np.zeros(len(path))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def ensure_segment(path: np.ndarray) -> np.ndarray:
    """A polyline needs >= 2 vertices for projection; duplicate a lone point."""
    path = np.asarray(path, dtype=float)
    if len(path) == 1:
        return np.vstack([path, path])
    return path


def project_on_polyline(path: np.ndarray, cum: np.ndarray, p: np.ndarray):
    """Closest point on a polyline.

    Returns ``(distance, arclength_of_closest_point, closest_point)``.
    """
    a, b = path[:-1], path[1:]
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((p - a) * ab).sum(axis=1) / ab2, 0.0, 1.0)
    q = a + ab * t[:, None]
    d = np.linalg.norm(p - q, axis=1)
    i = int(np.argmin(d))
    return float(d[i]), float(cum[i] + t[i] * np.sqrt(ab2[i])), q[i]


def point_at(path: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    """Point at arc length ``s`` (clamped to the polyline)."""
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, side="right")) - 1
    i = min(max(i, 0), len(path) - 2)
    seg = max(cum[i + 1] - cum[i], 1e-9)
    t = (s - cum[i]) / seg
    return path[i] + t * (path[i + 1] - path[i])
