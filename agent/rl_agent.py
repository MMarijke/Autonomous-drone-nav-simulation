"""Deployed reinforcement-learning agent (greedy Q-policy).

``guidance="goal"``  -> end-to-end RL (policy only knows where the goal is)
``guidance="astar"`` -> hybrid: A* way-points on the static map + learned local avoidance
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agent.actions import ACTION_DIRS, ACTION_NAMES
from agent.features import FeaturePipeline
from agent.observation import EpisodeInfo, Observation
from models.dqn import load_checkpoint
from models.networks import DuelingQNetwork

torch.set_num_threads(1)


class RLAgent:
    def __init__(self, cfg, net: DuelingQNetwork, guidance_kind: str, name: str | None = None):
        self.cfg = cfg
        self.net = net.eval()
        self.guidance_kind = guidance_kind
        self.name = name or ("hybrid" if guidance_kind == "astar" else "rl")
        self.pipeline = FeaturePipeline(cfg, guidance_kind)
        if self.pipeline.dim != net.obs_dim:
            raise ValueError(f"feature dim {self.pipeline.dim} != network input {net.obs_dim}; "
                             "was the checkpoint trained with a different sensor config?")
        self.vmax = float(cfg.drone.max_speed)
        self.info: dict = {}

    @classmethod
    def from_checkpoint(cls, cfg, path: str | Path, name: str | None = None) -> "RLAgent":
        net, guidance, _, meta = load_checkpoint(path)
        agent = cls(cfg, net, guidance, name)
        agent.meta = meta
        return agent

    @property
    def replans(self) -> int:
        return self.pipeline.guidance.replans

    def reset(self, ep: EpisodeInfo) -> None:
        self.pipeline.reset(ep)
        self.info = {}

    @torch.no_grad()
    def act(self, obs: Observation) -> np.ndarray:
        x, vis = self.pipeline.encode(obs)
        q = self.net(torch.from_numpy(x).unsqueeze(0))[0].numpy()
        a = int(np.argmax(q))
        cmd = ACTION_DIRS[a].astype(float) * self.vmax
        g = self.pipeline.guidance
        self.info = {
            "mode": f"{'A* waypoint + ' if g.kind == 'astar' else ''}DQN policy",
            "action": a, "action_name": ACTION_NAMES[a], "q": q, "vision": vis,
            "path": g.path, "lookahead": g.lookahead(obs.pos), "cmd": cmd,
        }
        return cmd
