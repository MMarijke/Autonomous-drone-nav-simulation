"""Uniform voxel grid used by the planners (indexing + box rasterisation)."""
from __future__ import annotations

import numpy as np


class VoxelGrid:
    """Axis-aligned voxel grid covering ``[0, size]`` with cubic cells of edge ``res``."""

    def __init__(self, size, res: float):
        self.size = np.asarray(size, dtype=float)
        self.res = float(res)
        self.shape = tuple(int(np.ceil(s / self.res - 1e-9)) for s in self.size)

    # ---- indexing --------------------------------------------------------
    def index(self, p) -> tuple[int, int, int]:
        idx = np.floor(np.asarray(p, dtype=float) / self.res).astype(int)
        idx = np.clip(idx, 0, np.asarray(self.shape) - 1)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def index_batch(self, pts: np.ndarray) -> np.ndarray:
        idx = np.floor(np.asarray(pts, dtype=float) / self.res).astype(int)
        return np.clip(idx, 0, np.asarray(self.shape) - 1)

    def center(self, idx) -> np.ndarray:
        return (np.asarray(idx, dtype=float) + 0.5) * self.res

    def inside(self, p) -> bool:
        p = np.asarray(p)
        return bool(np.all(p >= 0) and np.all(p < self.size))

    # ---- rasterisation ----------------------------------------------------
    def rasterize_boxes(self, lo: np.ndarray, hi: np.ndarray, inflation: float = 0.0) -> np.ndarray:
        """Boolean occupancy: a cell is occupied if its centre lies inside a box
        grown by ``inflation`` metres on every side."""
        occ = np.zeros(self.shape, dtype=bool)
        shape = np.asarray(self.shape)
        for l, h in zip(np.atleast_2d(lo), np.atleast_2d(hi)):
            i0 =np.ceil((l - inflation) / self.res - 0.5).astype(int)
            i1 = np.floor((h + inflation) / self.res - 0.5).astype(int)
            i0 = np.maximum(i0, 0)
            i1 = np.minimum(i1, shape - 1)
            if np.any(i1 < i0):
                continue
            occ[i0[0]:i1[0] + 1, i0[1]:i1[1] + 1, i0[2]:i1[2] + 1] = True
        return occ
