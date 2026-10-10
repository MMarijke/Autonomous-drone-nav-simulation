#!/usr/bin/env python3
"""Run the simulation with the live OpenCV visualisation.

    python run_sim.py --agent astar --scenario hard
    python run_sim.py --agent hybrid --checkpoint runs/train_hybrid/checkpoints/best.pt
    python run_sim.py --agent rl     --checkpoint runs/train_rl/checkpoints/best.pt --episodes 5

Headless (no window) rendering to a video / PNG:
    python run_sim.py --agent astar --no-display --video out.mp4 --snapshot frame.png

Keys in the window: q quit | space pause | n next episode | j/l i/k orbit |
z/x zoom | o auto-orbit | r rays | +/- playback speed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent.planner_agent import AStarAgent
from agent.rl_agent import RLAgent
from env.drone_env import DroneNavEnv
from utils.config import load_config
from utils.logger import setup_logger
from utils.metrics import summarize
from utils.runner import run_episode
from utils.seeding import seed_everything
from utils.visualization import Visualizer


def build_agent(kind: str, cfg, checkpoint: str | None):
    if kind == "astar":
        return AStarAgent(cfg)
    if checkpoint is None:
        default = {"rl": "runs/train_rl/checkpoints/best.pt", "hybrid": "runs/train_hybrid/checkpoints/best.pt"}[kind]
        checkpoint = default
    if not Path(checkpoint).exists():
        raise SystemExit(f"checkpoint {checkpoint!r} not found - train first: python train.py --mode {kind}")
    return RLAgent.from_checkpoint(cfg, checkpoint, name=kind)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=["astar", "rl", "hybrid"], default="astar")
    ap.add_argument("--checkpoint", default=None, help="DQN checkpoint (rl / hybrid)")
    ap.add_argument("--scenario", default="medium", help="easy | medium | hard (see configs/default.yaml)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None, help="first world seed (default: config seed)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--no-display", action="store_true", help="do not open a window")
    ap.add_argument("--video", default=None, help="write an mp4 of the run")
    ap.add_argument("--snapshot", default=None, help="save the final frame of every episode as PNG "
                                                     "(a numeric suffix is added for several episodes)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed0 = cfg.seed if args.seed is None else args.seed
    seed_everything(seed0)
    logger = setup_logger("run_sim")
    env = DroneNavEnv(cfg, seed0)
    agent = build_agent(args.agent, cfg, args.checkpoint)
    viz = Visualizer(cfg, display=not args.no_display, video_path=args.video, speed=args.speed)

    results = []
    try:
        for ep in range(args.episodes):
            viz.new_episode()
            res = run_episode(env, agent, seed0 + ep, args.scenario, callback=viz)
            results.append(res)
            logger.info("episode %d | seed %d | %s | %.1fs | path %.1f m (ref %.1f) | min gap %.2f m | %.1f ms/step",
                        ep + 1, res.seed, res.outcome.upper(), res.sim_time, res.path_length,
                        res.ref_length, res.min_clearance, res.decision_ms)
            if args.snapshot:
                p = Path(args.snapshot)
                viz.save_png(str(p if args.episodes == 1 else p.with_name(f"{p.stem}_{ep + 1}{p.suffix}")))
            if viz.quit:
                break
    finally:
        viz.close()
    s = summarize(results, cfg.task.near_miss)
    logger.info("summary: success %.0f%% | collisions %.0f%% | timeouts %.0f%% | path efficiency %.2f",
                100 * s["success_rate"], 100 * s["collision_rate"], 100 * s["timeout_rate"],
                s["path_efficiency"])


if __name__ == "__main__":
    main()
