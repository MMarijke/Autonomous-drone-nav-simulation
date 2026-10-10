"""Discrete action set shared by the RL policy, the range sensors and the viewer.

Action 0 is "hover"; actions 1..18 are unit velocity directions along the three
axes (6) and the twelve in-plane / vertical diagonals (12). The range finder
casts one ray along each of the 18 directions, so ``ranges[k-1]`` is literally
"how far can I fly if I pick action k".
"""
from __future__ import annotations

import itertools

import numpy as np


def _build() -> tuple[np.ndarray, list[str]]:
    vecs = [(0, 0, 0)]
    names = ["hover"]
    axis_names = "xyz"
    for axis in range(3):
        for sign in (1, -1):
            v = [0, 0, 0]
            v[axis] = sign
            vecs.append(tuple(v))
            names.append(f"{'+' if sign > 0 else '-'}{axis_names[axis]}")
    for a, b in itertools.combinations(range(3), 2):
        for sa in (1, -1):
            for sb in (1, -1):
                v = [0, 0, 0]
                v[a], v[b] = sa, sb
                vecs.append(tuple(v))
                names.append(f"{'+' if sa > 0 else '-'}{axis_names[a]}{'+' if sb > 0 else '-'}{axis_names[b]}")
    arr = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32), names


ACTION_DIRS, ACTION_NAMES = _build()
NUM_ACTIONS = len(ACTION_DIRS)            # 19
RANGE_DIRS = ACTION_DIRS[1:].copy()       # 18 ray directions
NUM_RANGES = len(RANGE_DIRS)
