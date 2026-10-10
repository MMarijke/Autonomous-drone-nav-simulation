"""Gym-style wrapper: discrete actions, feature vectors and shaped reward."""
from __future__ import annotations

import numpy as np

from agent.actions import ACTION_DIRS, NUM_ACTIONS
from agent.features import FeaturePipeline
from env.drone_env import DroneNavEnv


class DroneRLEnv:
    """Training environment.

    Reward (before the global ``scale``)::

        r = progress * (phi(s') - phi(s))            phi = -distance-to-go (potential shaping)
            - step_cost
            - proximity * max(0, (safe_dist - gap) / safe_dist)^2
            + goal_bonus / - collision_penalty / - timeout_penalty

    ``distance-to-go`` is the straight-line distance for end-to-end RL, and the
    remaining length along the A* path for the hybrid agent.
    """

    def __init__(self, cfg, guidance_kind: str = "goal", seed: int = 0):
        self.cfg = cfg
        self.env = DroneNavEnv(cfg, seed)
        self.pipeline = FeaturePipeline(cfg, guidance_kind)
        self.guidance_kind = guidance_kind
        self.obs_dim = self.pipeline.dim
        self.n_actions = NUM_ACTIONS
        self.rw = cfg.rl.reward
        self.vmax = float(cfg.drone.max_speed)
        self.radius = float(cfg.drone.radius)
        self._last_obs = None

    def reset(self, seed: int | None = None, scenario: str | None = None,
              difficulty: float | None = None) -> np.ndarray:
        obs = self.env.reset(seed=seed, scenario=scenario, difficulty=difficulty)
        self.pipeline.reset(self.env.episode_info)
        x, _ = self.pipeline.encode(obs)
        self._last_obs = obs
        return x

    def step(self, action: int):
        prev_pos = self._last_obs.pos
        obs, info = self.env.step(ACTION_DIRS[action].astype(float) * self.vmax)
        x, _ = self.pipeline.encode(obs)           # also lets guidance re-plan if we drifted
        g = self.pipeline.guidance
        progress = g.remaining(prev_pos) - g.remaining(obs.pos)

        gap = float(obs.ranges.min()) - self.radius
        prox = max(0.0, (self.rw.safe_dist - gap) / self.rw.safe_dist) ** 2
        r = self.rw.progress * progress - self.rw.step - self.rw.proximity * prox
        event = info["event"]
        if event == "goal":
            r += self.rw.goal
        elif event == "collision":
            r -= self.rw.collision
        elif event == "timeout":
            r -= self.rw.timeout
        self._last_obs = obs
        return x, float(r * self.rw.scale), info["terminated"], info["truncated"], info
