"""A* baseline agent: global planning on a prior map + sensed dynamic obstacles."""
from __future__ import annotations

import numpy as np

from agent.astar import AStar3D
from agent.controller import PathFollower, SafetyLayer
from agent.mapping import OccupancyMapper
from agent.observation import EpisodeInfo, Observation
from agent.sensors import DepthCamera
from agent.vision import VisionProcessor


class AStarAgent:
    """Plan -> follow -> sense -> replan.

    * Initial path: A* on the inflated static map (any-angle smoothed).
    * Each step the depth image and range finder are back-projected into the
      occupancy map to reveal moving obstacles (see :mod:`agent.mapping`).
    * The path is re-planned when (a) something blocks the next few metres,
      (b) the drone has drifted off the path, or (c) periodically.
    * A reactive safety layer limits speed to the stopping distance.
    """

    name = "astar"

    def __init__(self, cfg):
        self.cfg = cfg
        self.pc = cfg.planner
        self.dt = float(cfg.drone.dt)
        self.max_range = float(cfg.sensors.camera.max_range)
        camera_model = DepthCamera(cfg.sensors.camera, np.random.default_rng(0))
        self.vision = VisionProcessor(cfg.vision, camera_model)
        self.safety = SafetyLayer(self.pc.safety, cfg.drone.radius)
        self.follower = PathFollower(self.pc.lookahead, self.pc.cruise_speed)
        self.planner: AStar3D | None = None
        self.mapper: OccupancyMapper | None = None
        self.info: dict = {}
        self.replans = 0

    # ------------------------------------------------------------------
    def reset(self, ep: EpisodeInfo) -> None:
        self.ep = ep
        self.mapper = OccupancyMapper(self.pc, ep, self.max_range)
        if self.planner is None or self.planner.grid.shape != self.mapper.grid.shape:
            self.planner = AStar3D(self.mapper.grid, self.pc.heuristic_weight)
        self.replans = 0
        self.steps_since_plan = 0
        self.plan_info = {"expansions": 0, "plan_ms": 0.0, "reason": "initial"}
        res = self.planner.plan(self.mapper.static_blocked, ep.start, ep.goal)
        path = res.path if res.success else np.vstack([ep.start, ep.goal])
        self.follower.set_path(path)
        self.plan_info = {"expansions": res.expansions, "plan_ms": res.time_ms, "reason": "initial"}
        self.replans += 1
        self.info = {"path": path, "mode": "initial plan"}

    def _plan(self, obs: Observation, blocked: np.ndarray, reason: str) -> bool:
        res = self.planner.plan(blocked, obs.pos, self.ep.goal)
        self.plan_info = {"expansions": res.expansions, "plan_ms": res.time_ms, "reason": reason}
        self.replans += 1
        self.steps_since_plan = 0
        if res.success:
            self.follower.set_path(res.path)
            return True
        return False

    # ------------------------------------------------------------------
    def act(self, obs: Observation) -> np.ndarray:
        vis = self.vision.process(obs.depth)
        self.mapper.update(obs)
        blocked = self.mapper.blocked(obs.t)
        self.steps_since_plan += 1

        cmd, target, dev = self.follower.command(obs.pos, self.ep.goal)
        ahead = self.follower.path_ahead(obs.pos, horizon=6.0)
        idx = self.mapper.grid.index_batch(ahead[2:])      # skip the first metre (own cell)
        path_blocked = bool(blocked[idx[:, 0], idx[:, 1], idx[:, 2]].any())

        reason = None
        if path_blocked and self.steps_since_plan >= int(self.pc.replan_interval):
            reason = "path blocked"
        elif dev > float(self.pc.deviation_replan):
            reason = "off path"
        elif self.steps_since_plan >= 4 * int(self.pc.replan_interval):
            reason = "periodic"

        replanned, mode = False, "follow A* path"
        if reason is not None:
            ok = self._plan(obs, blocked, reason)
            if not ok:
                ok = self._plan(obs, self.mapper.static_blocked, reason + " (static fallback)")
                mode = "no dynamic-free path: static fallback" if ok else "planning failed"
            else:
                mode = f"replanned: {reason}"
            replanned = True
            cmd, target, dev = self.follower.command(obs.pos, self.ep.goal)

        cmd, scale = self.safety.apply(cmd, obs.ranges)
        if scale < 0.95:
            mode += " | braking"
        self.info = {
            "mode": mode, "path": self.follower.path, "lookahead": target, "replanned": replanned,
            "plan": dict(self.plan_info), "safety_scale": scale, "vision": vis, "cmd": cmd.copy(),
            "dyn_cells": int(self.mapper.dynamic_blocked(obs.t).sum()),
            "dyn_hits": self.mapper.last_hits,
        }
        return cmd
