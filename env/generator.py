"""Random environment generator (buildings, floating blocks, moving spheres).

Every generated world is guaranteed *solvable on its static map*: a collision-free
path from start to goal exists once obstacles are inflated by the safety margin.
Dynamic obstacles may of course still block the way at run time.
"""
from __future__ import annotations

import numpy as np

from agent.astar import AStar3D
from env.world import World
from utils.voxel import VoxelGrid


class WorldGenerator:
    def __init__(self, cfg):
        self.cfg = cfg                                    # full config
        w = cfg.world
        self.size = np.asarray(w.size, dtype=float)
        self.grid = VoxelGrid(self.size, w.resolution)
        self.planner = AStar3D(self.grid, weight=1.0)
        self.inflation = float(cfg.planner.inflation)

    # ------------------------------------------------------------------
    def _sample_boxes(self, rng: np.random.Generator, n: int):
        s = self.cfg.world.static
        W, D, H = self.size
        los, his = [], []
        for _ in range(n):
            if rng.random() < s.building_prob:
                fx, fy = rng.uniform(*s.building_footprint, 2)
                h = rng.uniform(*s.building_height)
                z0 = 0.0
            else:
                fx, fy = rng.uniform(*s.block_footprint, 2)
                h = rng.uniform(*s.block_height)
                z0 = rng.uniform(2.0, max(H - h - 1.0, 2.5))
            cx = rng.uniform(4.0, W - 4.0)
            cy = rng.uniform(2.0, D - 2.0)
            los.append([cx - fx / 2, cy - fy / 2, z0])
            his.append([cx + fx / 2, cy + fy / 2, min(z0 + h, H - 0.5)])
        if not los:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.asarray(los), np.asarray(his)

    def _sample_endpoints(self, rng, world: World):
        sg = self.cfg.world.start_goal
        W, D, H = self.size
        for _ in range(200):
            start = np.array([rng.uniform(1.5, sg.zone_depth), rng.uniform(2.0, D - 2.0),
                              rng.uniform(2.0, H - 2.0)])
            goal = np.array([rng.uniform(W - sg.zone_depth, W - 1.5), rng.uniform(2.0, D - 2.0),
                             rng.uniform(2.0, H - 2.0)])
            if np.linalg.norm(goal - start) < sg.min_distance:
                continue
            if world.static_clearance(start) < sg.clearance or world.static_clearance(goal) < sg.clearance:
                continue
            return start, goal
        return None

    def _sample_dynamic(self, rng, world: World, n: int):
        """Moving spheres. A fraction spawn around the start-goal line and fly
        *across* it (crossing traffic); the rest roam randomly."""
        if n <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        d = self.cfg.world.dynamic
        sg = self.cfg.world.start_goal
        axis = world.goal - world.start
        ahat = axis / np.linalg.norm(axis)
        perp1 = np.cross(ahat, [0.0, 0.0, 1.0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(ahat, perp1)
        pos, vel, rad = [], [], []
        tries = 0
        while len(pos) < n and tries < 1000:
            tries += 1
            r = rng.uniform(*d.radius)
            corridor = rng.random() < d.corridor_bias
            if corridor:
                lat = rng.normal(0.0, d.corridor_sigma, 2)
                p = world.start + axis * rng.uniform(0.2, 0.85) + perp1 * lat[0] + perp2 * lat[1] * 0.6
                direction = (perp1 * rng.choice([-1.0, 1.0]) + 0.3 * ahat * rng.normal()
                             + 0.2 * perp2 * rng.normal())
            else:
                p = rng.uniform(r + 0.5, self.size - r - 0.5)
                direction = rng.normal(size=3)
                direction[2] *= 0.4                           # mostly horizontal flyers
            if np.any(p < r + 0.5) or np.any(p > self.size - r - 0.5):
                continue
            if min(np.linalg.norm(p - world.start), np.linalg.norm(p - world.goal)) < sg.dynamic_keepout:
                continue
            if world.static_clearance(p) < r + 0.3:
                continue
            pos.append(p)
            vel.append(direction / np.linalg.norm(direction) * rng.uniform(*d.speed))
            rad.append(r)
        return np.asarray(pos).reshape(-1, 3), np.asarray(vel).reshape(-1, 3), np.asarray(rad)

    # ------------------------------------------------------------------
    def static_blocked(self, world: World) -> np.ndarray:
        return self.grid.rasterize_boxes(world.box_lo, world.box_hi, self.inflation)

    def generate(self, rng: np.random.Generator, n_static: int, n_dynamic: int) -> World:
        max_attempts = int(self.cfg.world.start_goal.max_attempts)
        n = n_static
        for attempt in range(max_attempts):
            lo, hi = self._sample_boxes(rng, n)
            world = World(self.size.copy(), lo, hi, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0),
                          wander=float(self.cfg.world.dynamic.wander),
                          rng=np.random.default_rng(int(rng.integers(2**31))))
            ends = self._sample_endpoints(rng, world)
            if ends is None:
                continue
            world.start, world.goal = ends
            plan = self.planner.plan(self.static_blocked(world), world.start, world.goal)
            if not plan.success:
                if attempt > max_attempts // 2:
                    n = max(n - 1, 0)                         # progressively thin the field
                continue
            world.ref_path = plan.path
            world.dyn_pos, world.dyn_vel, world.dyn_rad = self._sample_dynamic(rng, world, n_dynamic)
            return world
        raise RuntimeError("could not generate a solvable world; reduce obstacle counts")
