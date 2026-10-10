"""Q-network architecture."""
from __future__ import annotations

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """MLP trunk with separate state-value and advantage heads (Wang et al., 2016)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256), head: int = 128):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Sequential(nn.Linear(last, head), nn.ReLU(), nn.Linear(head, 1))
        self.advantage = nn.Sequential(nn.Linear(last, head), nn.ReLU(), nn.Linear(head, n_actions))
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.hidden = tuple(hidden)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        adv = self.advantage(z)
        return self.value(z) + adv - adv.mean(dim=1, keepdim=True)
