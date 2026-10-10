"""Replay memory and n-step return accumulation."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


class NStepAccumulator:
    """Turns a stream of 1-step transitions into n-step transitions.

    Emits ``(s, a, R_n, s_n, terminal, gamma^k)`` where ``k <= n`` is the number
    of rewards actually summed (shorter at episode ends). Time-limit
    truncation does **not** set ``terminal`` so the learner still bootstraps.
    """

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = int(n), float(gamma)
        self.q: deque = deque()

    def reset(self) -> None:
        self.q.clear()

    def _make(self, terminal: bool):
        s0, a0 = self.q[0][0], self.q[0][1]
        ret, g = 0.0, 1.0
        for _, _, r, _ in self.q:
            ret += g * r
            g *= self.gamma
        return s0, a0, ret, self.q[-1][3], float(terminal), g

    def push(self, s, a, r, s2, terminal: bool, truncated: bool) -> list:
        self.q.append((s, a, r, s2))
        out = []
        if terminal or truncated:
            while self.q:
                out.append(self._make(terminal))
                self.q.popleft()
        elif len(self.q) == self.n:
            out.append(self._make(False))
            self.q.popleft()
        return out


class ReplayBuffer:
    """Uniform replay stored in preallocated numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.gk = np.zeros(capacity, np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, s2, done, gk) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i], self.gk[i] = s, a, r, s2, done, gk
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        idx = self.rng.integers(0, self.size, batch)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.done[idx]), torch.from_numpy(self.gk[idx]))
