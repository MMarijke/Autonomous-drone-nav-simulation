"""Online occupancy map for the A* baseline.

The planner is given a *prior map of the static scene* (a building survey) but
no knowledge of moving obstacles. Dynamic obstacles are recovered from the
sensors: depth-camera and range-finder returns are back-projected into world
coordinates; any return that does not lie on a known static surface must belong
to something moving, and the voxels around it are blocked for ``dynamic_ttl``
seconds. ``perception="oracle"`` bypasses sensing and reads true obstacle
positions (with a short constant-velocity prediction) - an upper bound used for
ablation.
"""
from __future__ import annotations

import numpy as np

from agent.actions import RANGE_DIRS
from agent.observation import EpisodeInfo, Observation
from utils.voxel import VoxelGrid


class OccupancyMapper:
    def __init__(self, planner_cfg, info: EpisodeInfo, max_range: float):
        self.cfg = planner_cfg
        self.max_range = float(max_range)
        self.grid = VoxelGrid(info.world_size, info.resolution)
        self.size = np.asarray(info.world_size, float)
        self.static_blocked = self.grid.rasterize_boxes(info.static_lo, info.static_hi,
                                                        planner_cfg.inflation)
        self.static_near = self.grid.rasterize_boxes(info.static_lo, info.static_hi,
                                                     planner_cfg.static_filter_margin)
        self.expiry = np.full(self.grid.shape, -1.0)
        self.margin = float(planner_cfg.static_filter_margin)
        self.perception = planner_cfg.perception
        r = float(planner_cfg.dynamic_inflate)
        k = int(np.ceil(r / self.grid.res))
        rng = np.arange(-k, k + 1)
        off = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
        self.offsets = off[np.linalg.norm(off, axis=1) * self.grid.res <= r + 1e-9]
        self.last_hits = np.zeros((0, 3))
        self._centres: np.ndarray | None = None      # lazily built, oracle mode only

    # ------------------------------------------------------------------
    def update(self, obs: Observation) -> None:
        if self.perception == "oracle":
            self._update_oracle(obs)
        else:
            self._update_sensed(obs)

    def _update_sensed(self, obs: Observation) -> None:
        d = obs.depth.ravel()
        cam_hit = d < self.max_range - 0.05
        pts = obs.pos + obs.cam_dirs[cam_hit] * d[cam_hit, None]
        rng_hit = obs.ranges < self.max_range - 0.05
        pts_r = obs.pos + RANGE_DIRS[rng_hit] * obs.ranges[rng_hit, None]
        pts = np.vstack([pts, pts_r])
        dirs = np.vstack([obs.cam_dirs[cam_hit], RANGE_DIRS[rng_hit]])
        if not len(pts):
            self.last_hits = pts
            return
        near_wall = (pts < self.margin).any(axis=1) | (pts > self.size - self.margin).any(axis=1)
        idx = self.grid.index_batch(pts)
        on_static = self.static_near[idx[:, 0], idx[:, 1], idx[:, 2]]
        dyn = ~(near_wall | on_static)
        self.last_hits = pts[dyn]
        if not dyn.any():
            return
        centres = pts[dyn] + dirs[dyn] * float(self.cfg.dynamic_depth)
        cidx = np.unique(self.grid.index_batch(centres), axis=0)
        self._mark(cidx, obs.t + float(self.cfg.dynamic_ttl))

    def _mark(self, cell_idx: np.ndarray, expire_at: float) -> None:
        nb = (cell_idx[:, None, :] + self.offsets[None, :, :]).reshape(-1, 3)
        shape = np.asarray(self.grid.shape)
        ok = ((nb >= 0) & (nb < shape)).all(axis=1)
        nb = nb[ok]
        self.expiry[nb[:, 0], nb[:, 1], nb[:, 2]] = expire_at

    def _update_oracle(self, obs: Observation) -> None:
        pos = obs.privileged["dyn_pos"]
        vel = obs.privileged["dyn_vel"]
        rad = obs.privileged["dyn_rad"]
        margin = float(self.cfg.inflation)
        if self._centres is None:
            self._centres = self.grid.center(np.indices(self.grid.shape).reshape(3, -1).T)
        centres = self._centres
        blocked = np.zeros(len(centres), bool)
        for k in (0.0, 0.5, 1.0):                       # constant-velocity look-ahead (s)
            for p, v, r in zip(pos, vel, rad):
                c = p + v * k
                blocked |= np.linalg.norm(centres - c, axis=1) <= r + margin
        flat = blocked.reshape(self.grid.shape)
        self.expiry[flat] = obs.t + 0.2

    # ------------------------------------------------------------------
    def dynamic_blocked(self, t: float) -> np.ndarray:
        return self.expiry > t

    def blocked(self, t: float) -> np.ndarray:
        return self.static_blocked | (self.expiry > t)
