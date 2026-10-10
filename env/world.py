"""Ground-truth 3-D world: static boxes + moving spheres inside a closed room."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.raycast import cast_rays


@dataclass
class World:
    size: np.ndarray                          # (3,) room extent
    box_lo: np.ndarray                        # (N, 3) static obstacles
    box_hi: np.ndarray
    dyn_pos: np.ndarray                       # (M, 3) moving spheres
    dyn_vel: np.ndarray
    dyn_rad: np.ndarray
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_path: np.ndarray | None = None        # smoothed A* path on the static map
    wander: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # ------------------------------------------------------------------ sim
    @property
    def n_static(self) -> int:
        return len(self.box_lo)

    @property
    def n_dynamic(self) -> int:
        return len(self.dyn_pos)

    def step(self, dt: float) -> None:
        """Advance moving obstacles: constant velocity + small random wander,
        specular reflection off the room walls and off static boxes."""
        if not self.n_dynamic:
            return
        speed = np.linalg.norm(self.dyn_vel, axis=1, keepdims=True)
        if self.wander > 0:
            self.dyn_vel = self.dyn_vel + self.rng.normal(0, self.wander * speed * dt ** 0.5,
                                                          self.dyn_vel.shape)
            new_speed = np.maximum(np.linalg.norm(self.dyn_vel, axis=1, keepdims=True), 1e-6)
            self.dyn_vel = self.dyn_vel / new_speed * speed
        self.dyn_pos = self.dyn_pos + self.dyn_vel * dt

        r = self.dyn_rad[:, None]
        lo, hi = r + 0.2, self.size[None, :] - r - 0.2
        below, above = self.dyn_pos < lo, self.dyn_pos > hi
        self.dyn_pos = np.clip(self.dyn_pos, lo, hi)
        flip = below | above
        self.dyn_vel = np.where(flip, -self.dyn_vel, self.dyn_vel)

        if self.n_static:
            closest = np.clip(self.dyn_pos[:, None, :], self.box_lo[None], self.box_hi[None])
            diff = self.dyn_pos[:, None, :] - closest                # (M, N, 3)
            dist = np.linalg.norm(diff, axis=2)                       # (M, N)
            pen = self.dyn_rad[:, None] - dist
            j = pen.argmax(axis=1)
            for m in np.nonzero(pen[np.arange(len(j)), j] > 0)[0]:
                n = diff[m, j[m]]
                d = np.linalg.norm(n)
                if d < 1e-6:
                    n = -self.dyn_vel[m] / max(np.linalg.norm(self.dyn_vel[m]), 1e-6)
                else:
                    n = n / d
                self.dyn_pos[m] = closest[m, j[m]] + n * (self.dyn_rad[m] + 1e-3)
                self.dyn_vel[m] = self.dyn_vel[m] - 2.0 * (self.dyn_vel[m] @ n) * n

    # -------------------------------------------------------------- queries
    def raycast(self, origin: np.ndarray, dirs: np.ndarray, max_range: float) -> np.ndarray:
        return cast_rays(np.asarray(origin, float), dirs, self.box_lo, self.box_hi,
                         self.dyn_pos, self.dyn_rad, np.zeros(3), self.size, max_range)

    def clearance(self, p: np.ndarray) -> float:
        """Distance from point ``p`` to the nearest surface (negative = inside)."""
        p = np.asarray(p, float)
        best = float(min((p).min(), (self.size - p).min()))        # room walls
        if self.n_static:
            outside = np.linalg.norm(np.maximum(np.maximum(self.box_lo - p, p - self.box_hi), 0.0), axis=1)
            inside = np.all((p > self.box_lo) & (p < self.box_hi), axis=1)
            pen = np.minimum(p - self.box_lo, self.box_hi - p).min(axis=1)
            d = np.where(inside, -pen, outside)
            best = min(best, float(d.min()))
        if self.n_dynamic:
            best = min(best, float((np.linalg.norm(self.dyn_pos - p, axis=1) - self.dyn_rad).min()))
        return best

    def static_clearance(self, p: np.ndarray) -> float:
        """Like :meth:`clearance` but ignoring dynamic obstacles (used by the generator)."""
        p = np.asarray(p, float)
        best = float(min(p.min(), (self.size - p).min()))
        if self.n_static:
            outside = np.linalg.norm(np.maximum(np.maximum(self.box_lo - p, p - self.box_hi), 0.0), axis=1)
            inside = np.all((p > self.box_lo) & (p < self.box_hi), axis=1)
            d = np.where(inside, -1.0, outside)
            best = min(best, float(d.min()))
        return best

    def privileged_state(self) -> dict:
        return {"dyn_pos": self.dyn_pos.copy(), "dyn_vel": self.dyn_vel.copy(),
                "dyn_rad": self.dyn_rad.copy()}
