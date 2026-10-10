"""Vectorised ray casting against boxes, spheres and the world bounding box."""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def cast_rays(origin: np.ndarray, dirs: np.ndarray,
              box_lo: np.ndarray, box_hi: np.ndarray,
              sph_c: np.ndarray, sph_r: np.ndarray,
              bounds_lo: np.ndarray, bounds_hi: np.ndarray,
              max_range: float) -> np.ndarray:
    """Distance from ``origin`` to the first surface along each unit ray.

    * boxes   - slab test, shapes ``(N, 3)``
    * spheres - quadratic test, centres ``(M, 3)``, radii ``(M,)``
    * bounds  - the world volume itself acts as a closed room (floor, ceiling, walls)

    Rays that hit nothing within ``max_range`` return ``max_range``.

    The slab test is written per axis on ``(R, N)`` arrays; reducing over a
    trailing axis of length 3 (``.max(axis=2)``) is several times slower in numpy.
    """
    dirs = np.where(np.abs(dirs) < _EPS, _EPS, dirs)
    inv = 1.0 / dirs                                            # (R, 3)

    # World bounds: exit distance of the ray from the room.
    t1 = (bounds_lo - origin) * inv
    t2 = (bounds_hi - origin) * inv
    t_best = np.maximum(t1, t2).min(axis=1)                     # (R,)

    if len(box_lo):
        t_near = t_far = None
        for ax in range(3):
            a = (box_lo[:, ax][None, :] - origin[ax]) * inv[:, ax:ax + 1]   # (R, N)
            b = (box_hi[:, ax][None, :] - origin[ax]) * inv[:, ax:ax + 1]
            lo_t, hi_t = np.minimum(a, b), np.maximum(a, b)
            if t_near is None:
                t_near, t_far = lo_t, hi_t
            else:
                np.maximum(t_near, lo_t, out=t_near)
                np.minimum(t_far, hi_t, out=t_far)
        np.maximum(t_near, 0.0, out=t_near)
        t_box = np.where(t_far >= t_near, t_near, np.inf).min(axis=1)
        t_best = np.minimum(t_best, t_box)

    if len(sph_c):
        oc = origin - sph_c                                      # (M, 3)
        b = dirs @ oc.T                                          # (R, M)
        c = (oc * oc).sum(axis=1) - sph_r ** 2                   # (M,)
        disc = b * b - c[None, :]
        t_s = -b - np.sqrt(np.maximum(disc, 0.0))
        valid = (disc >= 0.0) & (t_s >= 0.0)
        t_sph = np.where(valid, t_s, np.inf).min(axis=1)
        t_best = np.minimum(t_best, t_sph)

    return np.minimum(t_best, max_range)
