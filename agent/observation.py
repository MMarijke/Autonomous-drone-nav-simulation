"""Plain data containers exchanged between environment and agents."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeInfo:
    """What an agent is told once, at the start of an episode (the 'mission')."""
    start: np.ndarray
    goal: np.ndarray
    world_size: np.ndarray
    resolution: float
    # Prior map of the *static* environment (building survey). Dynamic obstacles
    # are never part of it; they have to be sensed.
    static_lo: np.ndarray
    static_hi: np.ndarray


@dataclass
class Observation:
    """Everything the drone senses at one time step."""
    t: float
    step: int
    pos: np.ndarray                   # state estimate (assumed exact)
    vel: np.ndarray
    heading: np.ndarray               # camera optical axis (unit vector)
    goal: np.ndarray
    depth: np.ndarray                 # (H, W) range image in metres (noisy)
    cam_dirs: np.ndarray              # (H*W, 3) world-frame unit ray directions
    ranges: np.ndarray                # (18,) range finder, metres (noisy)
    prev_ranges: np.ndarray           # (18,) readings from the previous step
    # Ground truth that only the "oracle" planner is allowed to read.
    privileged: dict = field(default_factory=dict)
