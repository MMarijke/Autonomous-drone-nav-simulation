"""Point-mass quadrotor kinematics with a velocity-command interface."""
from __future__ import annotations

import numpy as np

from utils.geometry import unit


class Drone:
    """Commanded-velocity vehicle with bounded speed and acceleration.

    The flight controller is abstracted: the agent supplies a desired velocity
    vector and the vehicle slews towards it subject to ``max_accel``. This keeps
    inertia (the drone cannot stop instantly, which is what makes dynamic
    obstacles hard) without simulating rotor dynamics.
    """

    def __init__(self, cfg):
        self.radius = float(cfg.radius)
        self.max_speed = float(cfg.max_speed)
        self.max_accel = float(cfg.max_accel)
        self.dt = float(cfg.dt)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.heading = np.array([1.0, 0.0, 0.0])
        self.path_length = 0.0

    def reset(self, pos, goal) -> None:
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel = np.zeros(3)
        h = unit(np.asarray(goal, dtype=float) - self.pos)
        self.heading = h if np.linalg.norm(h) > 0 else np.array([1.0, 0.0, 0.0])
        self.path_length = 0.0

    def step(self, v_cmd) -> None:
        v_cmd = np.asarray(v_cmd, dtype=float)
        speed = np.linalg.norm(v_cmd)
        if speed > self.max_speed:
            v_cmd = v_cmd * (self.max_speed / speed)
        dv = v_cmd - self.vel
        dv_max = self.max_accel * self.dt
        n = np.linalg.norm(dv)
        if n > dv_max:
            dv = dv * (dv_max / n)
        new_vel = self.vel + dv
        displacement = 0.5 * (self.vel + new_vel) * self.dt
        self.pos = self.pos + displacement
        self.vel = new_vel
        self.path_length += float(np.linalg.norm(displacement))
        # The camera follows the direction of travel (low-pass filtered).
        if np.linalg.norm(self.vel) > 0.4:
            h = 0.7 * self.heading + 0.3 * unit(self.vel)
            self.heading = unit(h)
