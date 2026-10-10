"""Simulated sensors: forward depth camera and an 18-ray range finder.

Sensors only need a ``world`` exposing ``raycast(origin, dirs, max_range)``;
they never touch the ground-truth obstacle lists directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.actions import NUM_RANGES, RANGE_DIRS


@dataclass
class CameraFrame:
    depth: np.ndarray        # (H, W) range image (metres), noisy
    dirs: np.ndarray         # (H*W, 3) world-frame unit rays
    basis: np.ndarray        # (3, 3) rows = right, up, forward


def camera_basis(heading: np.ndarray) -> np.ndarray:
    """Right/up/forward orthonormal basis for a camera looking along ``heading``."""
    f = heading / max(np.linalg.norm(heading), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    if abs(f @ up) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])


class DepthCamera:
    """Pinhole range camera. Pixel values are Euclidean distance along the ray."""

    def __init__(self, cfg, rng: np.random.Generator):
        self.w, self.h = int(cfg.width), int(cfg.height)
        self.max_range = float(cfg.max_range)
        self.noise_std = float(cfg.noise_std)
        self.dropout = float(cfg.dropout)
        self.rng = rng
        tan_h = np.tan(np.radians(cfg.hfov_deg) / 2)
        tan_v = np.tan(np.radians(cfg.vfov_deg) / 2)
        xs = (2 * (np.arange(self.w) + 0.5) / self.w - 1) * tan_h
        ys = (1 - 2 * (np.arange(self.h) + 0.5) / self.h) * tan_v
        X, Y = np.meshgrid(xs, ys)
        self.cam_rays = np.stack([X, Y, np.ones_like(X)], axis=-1).reshape(-1, 3)
        self.tan_h, self.tan_v = tan_h, tan_v

    def pixel_angles(self, col: float, row: float) -> tuple[float, float]:
        """(azimuth, elevation) in radians of a pixel relative to the optical axis."""
        x = (2 * (col + 0.5) / self.w - 1) * self.tan_h
        y = (1 - 2 * (row + 0.5) / self.h) * self.tan_v
        return float(np.arctan(x)), float(np.arctan(y))

    def render(self, world, pos: np.ndarray, heading: np.ndarray) -> CameraFrame:
        basis = camera_basis(heading)
        dirs = self.cam_rays @ basis
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        rng = world.raycast(pos, dirs, self.max_range)
        hit = rng < self.max_range - 1e-6
        noisy = rng.copy()
        if self.noise_std > 0:
            noisy[hit] += self.rng.normal(0.0, self.noise_std, hit.sum()) * (1 + 0.05 * rng[hit])
        if self.dropout > 0:
            noisy[self.rng.random(rng.shape) < self.dropout] = self.max_range
        noisy = np.clip(noisy, 0.1, self.max_range).astype(np.float32)
        return CameraFrame(noisy.reshape(self.h, self.w), dirs.astype(np.float32), basis)


class RangeFinder:
    """18 distance sensors, one along each non-hover action direction."""

    def __init__(self, cfg, rng: np.random.Generator):
        self.max_range = float(cfg.max_range)
        self.noise_std = float(cfg.noise_std)
        self.rng = rng

    def read(self, world, pos: np.ndarray) -> np.ndarray:
        r = world.raycast(pos, RANGE_DIRS.astype(np.float64), self.max_range)
        if self.noise_std > 0:
            hit = r < self.max_range - 1e-6
            r = r.copy()
            r[hit] += self.rng.normal(0.0, self.noise_std, hit.sum())
        return np.clip(r, 0.05, self.max_range).astype(np.float32)


class SensorSuite:
    """Camera + range finder, with the previous range reading kept for deltas."""

    def __init__(self, sensor_cfg, rng: np.random.Generator):
        self.camera = DepthCamera(sensor_cfg.camera, rng)
        self.rangefinder = RangeFinder(sensor_cfg.rangefinder, rng)
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the previous reading (first delta of an episode is zero)."""
        self._prev = None

    def sense(self, world, drone) -> tuple[CameraFrame, np.ndarray, np.ndarray]:
        frame = self.camera.render(world, drone.pos, drone.heading)
        ranges = self.rangefinder.read(world, drone.pos)
        prev = self._prev if self._prev is not None else ranges
        self._prev = ranges
        return frame, ranges, prev
