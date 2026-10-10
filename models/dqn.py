"""Double / dueling / n-step DQN learner and checkpoint I/O."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.networks import DuelingQNetwork


class DQNLearner:
    def __init__(self, obs_dim: int, n_actions: int, rl_cfg, device: str = "cpu"):
        self.cfg = rl_cfg
        self.device = torch.device(device)
        self.n_actions = n_actions
        hidden = tuple(rl_cfg.hidden)
        self.online = DuelingQNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = DuelingQNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=float(rl_cfg.lr))
        self.tau = float(rl_cfg.target_tau)
        self.clip = float(rl_cfg.grad_clip)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    # ------------------------------------------------------------------
    @torch.no_grad()
    def q_values(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0).to(self.device)
        return self.online(t)[0].cpu().numpy()

    def act(self, x: np.ndarray, eps: float, rng: np.random.Generator) -> int:
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(x)))

    # ------------------------------------------------------------------
    def update(self, batch) -> tuple[float, float]:
        s, a, r, s2, done, gk = (b.to(self.device) for b in batch)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a2 = self.online(s2).argmax(dim=1, keepdim=True)            # Double DQN
            q2 = self.target(s2).gather(1, a2).squeeze(1)
            y = r + (1.0 - done) * gk * q2
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.clip)
        self.opt.step()
        with torch.no_grad():                                           # Polyak averaging
            for pt, po in zip(self.target.parameters(), self.online.parameters()):
                pt.mul_(1.0 - self.tau).add_(po, alpha=self.tau)
        return float(loss.item()), float(q.mean().item())


def save_checkpoint(path: str | Path, net: DuelingQNetwork, cfg, guidance_kind: str, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "obs_dim": net.obs_dim, "n_actions": net.n_actions, "hidden": list(net.hidden),
        "guidance": guidance_kind, "cfg": cfg.to_dict(), "meta": meta or {},
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Returns ``(network, guidance_kind, cfg_dict, meta)``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = DuelingQNetwork(ckpt["obs_dim"], ckpt["n_actions"], tuple(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["guidance"], ckpt["cfg"], ckpt.get("meta", {})
