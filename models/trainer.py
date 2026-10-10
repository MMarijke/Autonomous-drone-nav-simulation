"""DQN training loop with curriculum, periodic greedy evaluation and checkpointing."""
from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from agent.rl_agent import RLAgent
from env.drone_env import DroneNavEnv
from env.rl_env import DroneRLEnv
from models.dqn import DQNLearner, save_checkpoint
from models.replay import NStepAccumulator, ReplayBuffer
from utils.metrics import RollingStats, summarize
from utils.runner import run_episode

EVAL_SEED_BASE = 10_000          # evaluation worlds are disjoint from training worlds (< 10_000)


def evaluate_net(cfg, net, guidance_kind: str, env: DroneNavEnv, episodes: int,
                 scenario: str, seed0: int = EVAL_SEED_BASE) -> dict:
    agent = RLAgent(cfg, net, guidance_kind)
    results = [run_episode(env, agent, seed0 + i, scenario) for i in range(episodes)]
    return summarize(results, cfg.task.near_miss)


def train(cfg, mode: str, run_dir: str | Path, logger, total_steps: int | None = None,
          seed: int | None = None) -> dict:
    """Train a DQN policy.

    ``mode``: ``"rl"`` (goal-conditioned end-to-end) or ``"hybrid"`` (A* way-points).
    Returns a dict with the best evaluation summary and checkpoint paths.
    """
    rl = cfg.rl
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    seed = int(cfg.seed if seed is None else seed)
    total = int(total_steps or rl.total_steps)
    guidance_kind = "astar" if mode == "hybrid" else "goal"

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)

    env = DroneRLEnv(cfg, guidance_kind, seed)
    eval_env = DroneNavEnv(cfg, seed)
    learner = DQNLearner(env.obs_dim, env.n_actions, rl)
    buffer = ReplayBuffer(rl.buffer_size, env.obs_dim, seed)
    nstep = NStepAccumulator(rl.n_step, rl.gamma)
    logger.info("training %s | obs_dim=%d actions=%d steps=%d params=%d", mode, env.obs_dim,
                env.n_actions, total, sum(p.numel() for p in learner.online.parameters()))

    eps_cfg = rl.eps
    eps_steps = max(int(eps_cfg.decay_frac * total), 1)
    curriculum_steps = max(int(rl.curriculum_frac * total), 1)

    stats = {k: RollingStats(100) for k in ("success", "collision", "timeout", "ret", "len")}
    log_rows: list[dict] = []
    eval_rows: list[dict] = []
    best = {"success_rate": -1.0, "collision_rate": 1.0}
    loss_ma, q_ma = 0.0, 0.0
    episodes = 0
    ep_ret, ep_len = 0.0, 0
    t_start = time.perf_counter()

    difficulty = 0.0
    x = env.reset(seed=int(rng.integers(0, EVAL_SEED_BASE)), difficulty=difficulty)

    for step in range(1, total + 1):
        if step % 1000 == 1:                                     # cosine learning-rate schedule
            frac = step / total
            learner.set_lr(float(rl.lr_end) + 0.5 * (float(rl.lr) - float(rl.lr_end)) * (1 + np.cos(np.pi * frac)))
        eps = max(eps_cfg.end, eps_cfg.start + (eps_cfg.end - eps_cfg.start) * step / eps_steps)
        a = learner.act(x, eps, rng)
        x2, r, terminated, truncated, info = env.step(a)
        for tr in nstep.push(x, a, r, x2, terminated, truncated):
            buffer.add(*tr)
        x = x2
        ep_ret += r
        ep_len += 1

        if step >= rl.learning_starts and step % rl.train_every == 0 and len(buffer) >= rl.batch_size:
            loss, q = learner.update(buffer.sample(rl.batch_size))
            loss_ma = 0.99 * loss_ma + 0.01 * loss
            q_ma = 0.99 * q_ma + 0.01 * q

        if terminated or truncated:
            episodes += 1
            ev = info["event"]
            stats["success"].add(ev == "goal")
            stats["collision"].add(ev == "collision")
            stats["timeout"].add(ev == "timeout")
            stats["ret"].add(ep_ret / cfg.rl.reward.scale)
            stats["len"].add(ep_len)
            nstep.reset()
            ep_ret, ep_len = 0.0, 0
            difficulty = min(1.0, step / curriculum_steps)
            x = env.reset(seed=int(rng.integers(0, EVAL_SEED_BASE)), difficulty=difficulty)

            if episodes % int(rl.log_every_episodes) == 0:
                sps = step / (time.perf_counter() - t_start)
                row = {"step": step, "episodes": episodes, "success": stats["success"].mean,
                       "collision": stats["collision"].mean, "timeout": stats["timeout"].mean,
                       "return": stats["ret"].mean, "ep_len": stats["len"].mean, "eps": eps,
                       "difficulty": difficulty, "loss": loss_ma, "q_mean": q_ma, "sps": sps}
                log_rows.append(row)
                logger.info("step %7d | ep %5d | succ %.2f coll %.2f tout %.2f | ret %6.1f | "
                            "eps %.2f diff %.2f | loss %.4f q %.2f | %.0f sps",
                            step, episodes, row["success"], row["collision"], row["timeout"],
                            row["return"], eps, difficulty, loss_ma, q_ma, sps)

        if step % int(rl.eval_every) == 0 or step == total:
            net = copy.deepcopy(learner.online).eval()
            with torch.no_grad():
                summ = evaluate_net(cfg, net, guidance_kind, eval_env, int(rl.eval_episodes),
                                    rl.eval_scenario)
            summ["step"] = step
            eval_rows.append(summ)
            logger.info("EVAL @%d (%s x%d): success %.2f collision %.2f timeout %.2f eff %.2f",
                        step, rl.eval_scenario, rl.eval_episodes, summ["success_rate"],
                        summ["collision_rate"], summ["timeout_rate"], summ["path_efficiency"])
            save_checkpoint(run_dir / "checkpoints" / "last.pt", net, cfg, guidance_kind,
                            {"step": step, "mode": mode, "eval": summ})
            better = (summ["success_rate"], -summ["collision_rate"]) > (best["success_rate"], -best["collision_rate"])
            if better:
                best = {"success_rate": summ["success_rate"], "collision_rate": summ["collision_rate"],
                        "step": step}
                save_checkpoint(run_dir / "checkpoints" / "best.pt", net, cfg, guidance_kind,
                                {"step": step, "mode": mode, "eval": summ})
                logger.info("  new best checkpoint (step %d)", step)
            _dump(run_dir, log_rows, eval_rows)

    _dump(run_dir, log_rows, eval_rows)
    logger.info("done in %.1f min | best eval success %.2f @ step %s",
                (time.perf_counter() - t_start) / 60, best["success_rate"], best.get("step"))
    return {"best": best, "run_dir": str(run_dir), "log": log_rows, "eval": eval_rows}


def _dump(run_dir: Path, log_rows: list[dict], eval_rows: list[dict]) -> None:
    if log_rows:
        with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    with open(run_dir / "eval_log.json", "w", encoding="utf-8") as fh:
        json.dump(eval_rows, fh, indent=2, default=float)
