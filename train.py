#!/usr/bin/env python3
"""Train the DQN navigation policy.

    python train.py --mode rl      # end-to-end: the policy only knows the goal direction
    python train.py --mode hybrid  # policy follows A* way-points and handles local avoidance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.trainer import train
from utils.config import load_config
from utils.logger import make_run_dir, setup_logger
from utils.seeding import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["rl", "hybrid"], required=True)
    ap.add_argument("--config", default=None, help="YAML file merged over configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="config overrides")
    ap.add_argument("--steps", type=int, default=None, help="override rl.total_steps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="runs", help="base directory for run folders")
    ap.add_argument("--name", default=None, help="run folder name (default: train_<mode>)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = cfg.seed if args.seed is None else args.seed
    seed_everything(seed)
    run_dir = make_run_dir(args.out, args.name or f"train_{args.mode}", stamp=args.name is None)
    logger = setup_logger(f"train_{args.mode}", run_dir)
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    logger.info("run directory: %s", run_dir)
    result = train(cfg, args.mode, run_dir, logger, total_steps=args.steps, seed=seed)
    logger.info("best checkpoint: %s", Path(run_dir) / "checkpoints" / "best.pt")
    logger.info("best eval: %s", result["best"])


if __name__ == "__main__":
    main()
