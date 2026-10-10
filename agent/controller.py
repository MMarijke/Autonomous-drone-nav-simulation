"""Path following (pure pursuit) and a reactive safety layer."""
from __future__ import annotations

import numpy as np

from agent.actions import RANGE_DIRS
from utils.geometry import (cumulative_length, ensure_segment, point_at,
                            project_on_polyline, unit)


class PathFollower:
    """Pure-pursuit controller: aim at a point ``lookahead`` metres further along
    the path, slowing down on the final approach."""

    def __init__(self, lookahead: float, cruise_speed: float):
        self.lookahead = float(lookahead)
        self.cruise = float(cruise_speed)
        self.path = np.zeros((2, 3))
        self.cum = np.zeros(2)

    def set_path(self, path: np.ndarray) -> None:
        self.path = ensure_segment(path)
        self.cum = cumulative_length(self.path)

    def command(self, pos: np.ndarray, goal: np.ndarray):
        """Returns ``(velocity_command, lookahead_point, distance_to_path)``."""
        dist, s, _ = project_on_polyline(self.path, self.cum, pos)
        target = point_at(self.path, self.cum, s + self.lookahead)
        goal_dist = float(np.linalg.norm(goal - pos))
        speed = min(self.cruise, 0.8 * goal_dist + 0.5)
        return unit(target - pos) * speed, target, dist

    def path_ahead(self, pos: np.ndarray, horizon: float, step: float = 0.5) -> np.ndarray:
        _, s, _ = project_on_polyline(self.path, self.cum, pos)
        ss = np.arange(0.0, horizon + 1e-6, step) + s
        return np.array([point_at(self.path, self.cum, v) for v in ss])


class SafetyLayer:
    """Last line of defence for the A* baseline.

    * limits speed so the drone can still stop (``v <= sqrt(2 a d)``) in the free
      distance measured by the range finder in the direction of travel;
    * adds a small repulsion away from anything that is uncomfortably close.
    """

    def __init__(self, cfg, drone_radius: float):
        self.enabled = bool(cfg.enabled)
        self.margin = float(cfg.margin) + float(drone_radius)
        self.decel = float(cfg.brake_decel)
        self.rep_range = float(cfg.repulse_range)
        self.rep_gain = float(cfg.repulse_gain)

    def apply(self, v_cmd: np.ndarray, ranges: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns ``(safe_command, speed_scale)`` where scale<1 means braking."""
        if not self.enabled:
            return v_cmd, 1.0
        speed = float(np.linalg.norm(v_cmd))
        out = v_cmd.copy()
        scale = 1.0
        if speed > 1e-6:
            u = v_cmd / speed
            cone = RANGE_DIRS @ u > 0.7
            fwd = float(ranges[cone].min()) if cone.any() else float(ranges.max())
            allowed = np.sqrt(max(0.0, 2.0 * self.decel * (fwd - self.margin)))
            if allowed < speed:
                scale = allowed / speed
                out = u * allowed
        close = ranges < self.rep_range
        if close.any():
            push = ((self.rep_range - ranges[close]) / self.rep_range)[:, None] * RANGE_DIRS[close]
            out = out - self.rep_gain * push.sum(axis=0)
        return out, scale
