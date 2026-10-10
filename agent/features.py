"""Observation encoding shared by the RL environment and the deployed RL agent."""
from __future__ import annotations

import numpy as np

from agent.actions import NUM_RANGES
from agent.guidance import make_guidance
from agent.observation import EpisodeInfo, Observation
from agent.sensors import DepthCamera
from agent.vision import VisionOutput, VisionProcessor
from utils.geometry import unit


class ObservationEncoder:
    """Packs sensors + guidance into one float32 vector.

    ========  =====  ==============================================
    block     dim    meaning
    ========  =====  ==============================================
    ranges    18     range finder / max range              (0..1)
    d_ranges  18     per-step change of the 18 ranges      (-1..1)
    camera    15     min-pooled depth grid from OpenCV     (0..1)
    target    3      unit vector to guidance way-point
    |target|  1      distance to way-point / 10 m
    |goal|    1      distance to final goal / 40 m
    velocity  3      v / v_max
    altitude  1      z / H
    ========  =====  ==============================================
    """

    def __init__(self, cfg, vision_dim: int):
        self.max_range = float(cfg.sensors.rangefinder.max_range)
        self.vmax = float(cfg.drone.max_speed)
        self.height = float(cfg.world.size[2])
        # A closing speed of 5 m/s at dt = 0.1 s changes a range by 0.5 m per step.
        self.delta_scale = 0.5
        self.dim = NUM_RANGES * 2 + vision_dim + 3 + 1 + 1 + 3 + 1

    def encode(self, obs: Observation, vis: VisionOutput, target: np.ndarray) -> np.ndarray:
        to_target = target - obs.pos
        to_goal = float(np.linalg.norm(obs.goal - obs.pos))
        x = np.concatenate([
            obs.ranges / self.max_range,
            np.clip((obs.ranges - obs.prev_ranges) / self.delta_scale, -1.0, 1.0),
            vis.features,
            unit(to_target),
            [min(np.linalg.norm(to_target) / 10.0, 1.0)],
            [min(to_goal / 40.0, 1.0)],
            obs.vel / self.vmax,
            [obs.pos[2] / self.height],
        ])
        return x.astype(np.float32)


class FeaturePipeline:
    """Vision + guidance + encoder: turns an :class:`Observation` into a policy input."""

    def __init__(self, cfg, guidance_kind: str):
        camera_model = DepthCamera(cfg.sensors.camera, np.random.default_rng(0))
        self.vision = VisionProcessor(cfg.vision, camera_model)
        self.guidance = make_guidance(guidance_kind, cfg)
        self.encoder = ObservationEncoder(cfg, self.vision.feature_dim)
        self.dim = self.encoder.dim

    def reset(self, info: EpisodeInfo) -> None:
        self.guidance.reset(info)

    def encode(self, obs: Observation) -> tuple[np.ndarray, VisionOutput]:
        vis = self.vision.process(obs.depth)
        self.guidance.update(obs.pos)
        target = self.guidance.lookahead(obs.pos)
        return self.encoder.encode(obs, vis, target), vis
