"""Physics-level environment: world + drone + sensors, no reward shaping.

``DroneNavEnv`` is the common simulator used by every agent (A*, RL, hybrid).
Reward-shaped, discrete-action training lives in :mod:`env.rl_env`.
"""
from __future__ import annotations

import numpy as np

from agent.drone import Drone
from agent.observation import EpisodeInfo, Observation
from agent.sensors import SensorSuite
from env.generator import WorldGenerator
from env.world import World


class DroneNavEnv:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.base_seed = int(seed)
        self.generator = WorldGenerator(cfg)
        self.drone = Drone(cfg.drone)
        self.dt = self.drone.dt
        self.max_steps = int(cfg.task.max_steps)
        self.goal_radius = float(cfg.task.goal_radius)
        self.world: World | None = None
        self.sensors: SensorSuite | None = None
        self.episode_info: EpisodeInfo | None = None
        self.t = 0.0
        self.steps = 0
        self.min_gap = np.inf
        self.last_obs: Observation | None = None
        self._counter = 0
        self.scenario_name = "custom"
        self.seed = seed

    # ------------------------------------------------------------------
    def counts_for(self, scenario: str | None, difficulty: float | None,
                   n_static: int | None, n_dynamic: int | None) -> tuple[int, int]:
        if scenario is not None:
            sc = self.cfg.scenarios[scenario]
            ns, nd = int(sc["static"]), int(sc["dynamic"])
        elif difficulty is not None:
            c = self.cfg.curriculum
            d = float(np.clip(difficulty, 0.0, 1.0))
            ns = int(round(c.static[0] + d * (c.static[1] - c.static[0])))
            nd = int(round(c.dynamic[0] + d * (c.dynamic[1] - c.dynamic[0])))
        else:
            sc = self.cfg.scenarios["medium"]
            ns, nd = int(sc["static"]), int(sc["dynamic"])
        return (ns if n_static is None else int(n_static)), (nd if n_dynamic is None else int(n_dynamic))

    def reset(self, seed: int | None = None, scenario: str | None = None,
              difficulty: float | None = None, n_static: int | None = None,
              n_dynamic: int | None = None) -> Observation:
        if seed is None:
            seed = self.base_seed + self._counter
        self._counter += 1
        self.seed = int(seed)
        self.scenario_name = scenario or ("curriculum" if difficulty is not None else "custom")
        ns, nd = self.counts_for(scenario, difficulty, n_static, n_dynamic)

        rng = np.random.default_rng(self.seed)
        self.world = self.generator.generate(rng, ns, nd)
        self.sensors = SensorSuite(self.cfg.sensors, np.random.default_rng(self.seed + 1_000_003))
        self.drone.reset(self.world.start, self.world.goal)
        self.t, self.steps, self.min_gap = 0.0, 0, np.inf
        self.episode_info = EpisodeInfo(
            start=self.world.start.copy(), goal=self.world.goal.copy(),
            world_size=self.world.size.copy(), resolution=float(self.cfg.world.resolution),
            static_lo=self.world.box_lo.copy(), static_hi=self.world.box_hi.copy())
        self.last_obs = self._observe()
        return self.last_obs

    # ------------------------------------------------------------------
    def _observe(self) -> Observation:
        frame, ranges, prev = self.sensors.sense(self.world, self.drone)
        return Observation(
            t=self.t, step=self.steps, pos=self.drone.pos.copy(), vel=self.drone.vel.copy(),
            heading=self.drone.heading.copy(), goal=self.world.goal.copy(),
            depth=frame.depth, cam_dirs=frame.dirs, ranges=ranges, prev_ranges=prev,
            privileged=self.world.privileged_state())

    def step(self, v_cmd: np.ndarray) -> tuple[Observation, dict]:
        """Advance one control step with a desired-velocity command (m/s)."""
        self.drone.step(v_cmd)
        self.world.step(self.dt)
        self.t += self.dt
        self.steps += 1

        clearance = self.world.clearance(self.drone.pos)
        gap = clearance - self.drone.radius
        self.min_gap = min(self.min_gap, gap)
        dist_goal = float(np.linalg.norm(self.world.goal - self.drone.pos))

        event = None
        if gap < 0.0:
            event = "collision"
        elif dist_goal <= self.goal_radius:
            event = "goal"
        elif self.steps >= self.max_steps:
            event = "timeout"

        self.last_obs = self._observe()
        info = {"event": event, "terminated": event in ("collision", "goal"),
                "truncated": event == "timeout", "gap": float(gap), "dist_goal": dist_goal,
                "clearance": float(clearance)}
        return self.last_obs, info
