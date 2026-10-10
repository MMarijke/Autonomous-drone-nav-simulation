"""3-D weighted A* on a voxel grid (26-connected) with any-angle path smoothing."""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from utils.voxel import VoxelGrid

SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)


@dataclass
class PlanResult:
    success: bool
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # smoothed, start..goal
    raw_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # cell centres
    expansions: int = 0
    time_ms: float = 0.0
    cost: float = float("inf")


class AStar3D:
    """Weighted A* (``weight`` = 1 gives optimal 26-connected paths).

    The occupancy array is padded with a blocked border so neighbour lookups
    need no bounds checks, and the search loop works on flat integer indices
    and a ``bytes`` object - ~10x faster than naive tuple/array indexing in
    pure Python.
    """

    def __init__(self, grid: VoxelGrid, weight: float = 1.2):
        self.grid = grid
        self.weight = float(weight)
        nx, ny, nz = grid.shape
        self.px, self.py, self.pz = nx + 2, ny + 2, nz + 2
        sx, sy = self.py * self.pz, self.pz
        offs, costs = [], []
        for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
            if dx == dy == dz == 0:
                continue
            offs.append(dx * sx + dy * sy + dz)
            costs.append(math.sqrt(dx * dx + dy * dy + dz * dz) * grid.res)
        self.offs, self.costs = offs, costs
        n = self.px * self.py * self.pz
        idx = np.arange(n)
        self._ci = (idx // sx).tolist()
        self._cj = ((idx // sy) % self.py).tolist()
        self._ck = (idx % self.pz).tolist()
        self._sx, self._sy = sx, sy
        self._n = n

    # -- helpers -----------------------------------------------------------
    def _flat(self, i: int, j: int, k: int) -> int:
        return (i + 1) * self._sx + (j + 1) * self._sy + (k + 1)

    def plan(self, blocked: np.ndarray, start, goal, smooth: bool = True) -> PlanResult:
        t0 = time.perf_counter()
        grid = self.grid
        si, gi = grid.index(start), grid.index(goal)

        free = blocked.copy()
        free[si] = False          # the vehicle is physically here, so its own cell is free
        free[gi] = False
        padded = np.ones((self.px, self.py, self.pz), dtype=bool)
        padded[1:-1, 1:-1, 1:-1] = free
        blk = padded.tobytes()

        s, g = self._flat(*si), self._flat(*gi)
        gx, gy, gz = gi[0] + 1, gi[1] + 1, gi[2] + 1
        ci, cj, ck = self._ci, self._cj, self._ck
        res, w = grid.res, self.weight
        offs, costs = self.offs, self.costs
        c1, c2 = (SQ3 - SQ2), (SQ2 - 1.0)

        def heur(n: int) -> float:
            a, b, c = abs(ci[n] - gx), abs(cj[n] - gy), abs(ck[n] - gz)
            if a > b:
                a, b = b, a
            if b > c:
                b, c = c, b
            if a > b:
                a, b = b, a
            return (c1 * a + c2 * b + c) * res

        g_score = {s: 0.0}
        parent: dict[int, int] = {}
        closed = bytearray(self._n)
        open_heap = [(w * heur(s), 0.0, s)]
        expansions = 0
        found = False
        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            if closed[cur]:
                continue
            closed[cur] = 1
            expansions += 1
            if cur == g:
                found = True
                break
            for off, cost in zip(offs, costs):
                nb = cur + off
                if blk[nb] or closed[nb]:
                    continue
                ng = g_cur + cost
                if ng < g_score.get(nb, 1e18):
                    g_score[nb] = ng
                    parent[nb] = cur
                    heapq.heappush(open_heap, (ng + w * heur(nb), ng, nb))

        if not found:
            return PlanResult(False, expansions=expansions,
                              time_ms=(time.perf_counter() - t0) * 1e3)

        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        centres = np.array([[ci[c] - 1, cj[c] - 1, ck[c] - 1] for c in cells], dtype=float)
        centres = (centres + 0.5) * res
        raw = np.vstack([np.asarray(start, float), centres[1:-1], np.asarray(goal, float)]) \
            if len(centres) > 2 else np.vstack([np.asarray(start, float), np.asarray(goal, float)])
        path = self._smooth(raw, free) if smooth else raw
        cost = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        return PlanResult(True, path, raw, expansions, (time.perf_counter() - t0) * 1e3, cost)

    # -- any-angle smoothing ------------------------------------------------
    def line_free(self, a: np.ndarray, b: np.ndarray, free_blocked: np.ndarray) -> bool:
        d = float(np.linalg.norm(b - a))
        n = max(int(d / (0.25 * self.grid.res)), 1) + 1
        pts = a[None, :] + (b - a)[None, :] * np.linspace(0.0, 1.0, n)[:, None]
        idx = self.grid.index_batch(pts)
        return not free_blocked[idx[:, 0], idx[:, 1], idx[:, 2]].any()

    def _smooth(self, path: np.ndarray, blocked: np.ndarray) -> np.ndarray:
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        n = len(path)
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not self.line_free(path[i], path[j], blocked):
                j -= 1
            out.append(path[j])
            i = j
        return np.asarray(out)
