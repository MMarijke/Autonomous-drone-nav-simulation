#!/usr/bin/env python3
"""Benchmark agents on identical worlds and write tables + figures.

    python evaluate.py --agents astar rl hybrid --scenarios easy medium hard --episodes 100

Every agent sees exactly the same worlds (same seeds), so differences are due to
the agent, not the draw. Outputs go to ``runs/eval`` (CSV per episode, summary
JSON, markdown table, PNG figure).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from agent.planner_agent import AStarAgent
from agent.rl_agent import RLAgent
from env.drone_env import DroneNavEnv
from utils.config import load_config
from utils.logger import setup_logger
from utils.metrics import save_json, save_results, summarize, wilson_interval
from utils.plots import LABELS, plot_benchmark
from utils.runner import run_episode
from utils.seeding import seed_everything

DEFAULT_CKPT = {"rl": "runs/train_rl/checkpoints/best.pt", "hybrid": "runs/train_hybrid/checkpoints/best.pt"}


def make_agent(name: str, cfg, ckpts: dict):
    if name == "astar":
        return AStarAgent(cfg)
    if name == "astar_oracle":
        c = copy.deepcopy(cfg)
        c.planner.perception = "oracle"
        agent = AStarAgent(c)
        agent.name = "astar_oracle"
        return agent
    return RLAgent.from_checkpoint(cfg, ckpts.get(name) or DEFAULT_CKPT[name], name=name)


def markdown_table(summary: dict, scenarios, agents) -> str:
    head = ("| scenario | agent | n | success | collision | timeout | path eff. | time-to-goal (s) | "
            "min clearance (m) | near-miss | ms/step |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for sc in scenarios:
        for ag in agents:
            s = summary[ag][sc]
            lo, hi = s["success_ci"]
            rows.append(f"| {sc} | {LABELS[ag]} | {s['n']} | {100 * s['success_rate']:.0f}% "
                        f"[{100 * lo:.0f}-{100 * hi:.0f}] | {100 * s['collision_rate']:.0f}% | "
                        f"{100 * s['timeout_rate']:.0f}% | {s['path_efficiency']:.2f} | "
                        f"{s['time_to_goal_s']:.1f} | {s['min_clearance_m']:.2f} | "
                        f"{100 * s['near_miss_rate']:.0f}% | {s['decision_ms']:.2f} |")
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", default=["astar", "rl", "hybrid"],
                    choices=["astar", "astar_oracle", "rl", "hybrid"])
    ap.add_argument("--scenarios", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--rl-checkpoint", default=None)
    ap.add_argument("--hybrid-checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed_everything(cfg.seed)
    n = args.episodes or int(cfg.eval.episodes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("evaluate", out)
    ckpts = {"rl": args.rl_checkpoint, "hybrid": args.hybrid_checkpoint}

    env = DroneNavEnv(cfg)
    all_results, summary = [], {}
    for name in args.agents:
        agent = make_agent(name, cfg, ckpts)
        summary[name] = {}
        for sc in args.scenarios:
            res = [run_episode(env, agent, int(cfg.eval.seed_offset) + i, sc) for i in range(n)]
            for r in res:
                r.agent = name
            s = summarize(res, cfg.task.near_miss)
            k = sum(r.outcome == "success" for r in res)
            s["success_ci"] = wilson_interval(k, len(res))
            summary[name][sc] = s
            all_results += res
            logger.info("%-13s %-6s n=%d | success %.0f%% (CI %.0f-%.0f) | collision %.0f%% | timeout %.0f%% | "
                        "eff %.2f | %.2f ms/step", name, sc, n, 100 * s["success_rate"],
                        100 * s["success_ci"][0], 100 * s["success_ci"][1], 100 * s["collision_rate"],
                        100 * s["timeout_rate"], s["path_efficiency"], s["decision_ms"])
    save_results(all_results, out / "episodes.csv")
    save_json(summary, out / "summary.json")
    table = markdown_table(summary, args.scenarios, args.agents)
    (out / "summary.md").write_text(table, encoding="utf-8")
    plot_benchmark(summary, args.scenarios, args.agents, out / "benchmark.png")
    logger.info("\n%s", table)
    logger.info("wrote results to %s", out)


if __name__ == "__main__":
    main()
