"""Global guidance handed to the RL policy.

* :class:`GoalGuidance`  - "the goal is over there" (pure end-to-end RL).
* :class:`PathGuidance`  - an A* path through the static map; the policy is shown
  a way-point a few metres ahead on that path (hybrid planner + learned local
  controller). Dynamic obstacles are *not* in the planner's map, the policy must
  dodge them using its sensors.

Both expose ``remaining(pos)``, the distance-to-go used for potential-based
reward shaping.
"""
from __future__ import annotations

import numpy as np

from agent.astar import AStar3D
from agent.observation import EpisodeInfo
from utils.geometry import (cumulative_length, ensure_segment, point_at,
                            project_on_polyline)
from utils.voxel import VoxelGrid


class GoalGuidance:
    kind = "goal"

    def __init__(self, cfg):
        self.goal = np.zeros(3)
        self.path: np.ndarray | None = None
        self.replans = 0

    def reset(self, info: EpisodeInfo) -> None:
        self.goal = np.asarray(info.goal, float)
        self.replans = 0

    def update(self, pos: np.ndarray) -> None:
        pass

    def lookahead(self, pos: np.ndarray) -> np.ndarray:
        return self.goal

    def remaining(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal - pos))


class PathGuidance(GoalGuidance):
    kind = "astar"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.grid = VoxelGrid(cfg.world.size, cfg.world.resolution)
        self.planner = AStar3D(self.grid, cfg.planner.heuristic_weight)
        self.inflation = float(cfg.guidance.inflation)
        self.fallback_inflation = float(cfg.planner.inflation)
        self.look = float(cfg.guidance.lookahead)
        self.dev = float(cfg.guidance.deviation_replan)
        self.blocked = np.zeros(self.grid.shape, bool)
        self.blocked_fallback = np.zeros(self.grid.shape, bool)
        self.cum = np.zeros(2)
        self.s = 0.0

    def reset(self, info: EpisodeInfo) -> None:
        super().reset(info)
        self.blocked = self.grid.rasterize_boxes(info.static_lo, info.static_hi, self.inflation)
        self.blocked_fallback = self.grid.rasterize_boxes(info.static_lo, info.static_hi,
                                                          self.fallback_inflation)
        self._plan(np.asarray(info.start, float))

    def _plan(self, pos: np.ndarray) -> None:
        res = self.planner.plan(self.blocked, pos, self.goal)
        if not res.success:                         # too tight for the wide margin: use the planner's
            res = self.planner.plan(self.blocked_fallback, pos, self.goal)
        path = res.path if res.success else np.vstack([pos, self.goal])
        self.path = ensure_segment(path)
        self.cum = cumulative_length(self.path)
        self.replans += 1

    def update(self, pos: np.ndarray) -> None:
        d, s, _ = project_on_polyline(self.path, self.cum, pos)
        if d > self.dev:
            self._plan(pos)
            d, s, _ = project_on_polyline(self.path, self.cum, pos)
        self.s = s

    def lookahead(self, pos: np.ndarray) -> np.ndarray:
        _, s, _ = project_on_polyline(self.path, self.cum, pos)
        return point_at(self.path, self.cum, s + self.look)

    def remaining(self, pos: np.ndarray) -> float:
        d, s, _ = project_on_polyline(self.path, self.cum, pos)
        return float(d + (self.cum[-1] - s))


def make_guidance(kind: str, cfg) -> GoalGuidance:
    if kind in ("goal", "rl"):
        return GoalGuidance(cfg)
    if kind in ("astar", "hybrid"):
        return PathGuidance(cfg)
    raise ValueError(f"unknown guidance kind {kind!r}")
