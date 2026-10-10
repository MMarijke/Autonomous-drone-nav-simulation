"""Matplotlib figures for the README (training curves, benchmark bars).

Colours: first three slots of a validated categorical palette (safe for
colour-vision deficiency, all pairs); a legend and direct value labels
make identity independent of colour.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"astar": "#2a78d6", "rl": "#eb6834", "hybrid": "#1baf7a", "astar_oracle": "#6b7a8f"}
LABELS = {"astar": "A* (sensed)", "rl": "DQN end-to-end", "hybrid": "A* + DQN hybrid",
          "astar_oracle": "A* (oracle perception)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def plot_training(runs: dict[str, Path], out: str | Path) -> None:
    """``runs``: {"rl": run_dir, "hybrid": run_dir}. Left: rolling training outcomes,
    right: greedy evaluation success on the 'medium' scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), facecolor="#fcfcfb")
    for kind, d in runs.items():
        d = Path(d)
        log = _read_csv(d / "train_log.csv")
        ev = json.load(open(d / "eval_log.json"))
        c = COLORS[kind]
        axes[0].plot(log["step"] / 1e3, log["success"], color=c, lw=2, label=LABELS[kind])
        axes[1].plot(log["step"] / 1e3, log["collision"], color=c, lw=2, label=LABELS[kind])
        axes[2].plot([e["step"] / 1e3 for e in ev], [e["success_rate"] for e in ev], color=c, lw=2,
                     marker="o", ms=5, label=LABELS[kind])
    titles = ["Training success (rolling 100 episodes)", "Training collisions (rolling 100 episodes)",
              "Greedy evaluation success (medium scenario)"]
    for ax, t in zip(axes, titles):
        _style(ax)
        ax.set_title(t, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("environment steps (thousands)", color=MUTED, fontsize=9)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction of episodes", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_benchmark(summary: dict, scenarios: list[str], agents: list[str], out: str | Path) -> None:
    """Grouped bars: success, collision rate, path efficiency, decision time."""
    metrics = [("success_rate", "Success rate", True), ("collision_rate", "Collision rate", True),
               ("path_efficiency", "Path efficiency (successes)", False),
               ("decision_ms", "Decision time (ms / step)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), facecolor="#fcfcfb")
    width = 0.8 / len(agents)
    for ax, (key, title, pct) in zip(axes, metrics):
        _style(ax)
        for i, ag in enumerate(agents):
            vals = [summary[ag][sc].get(key, np.nan) for sc in scenarios]
            xs = np.arange(len(scenarios)) + (i - (len(agents) - 1) / 2) * width
            ax.bar(xs, vals, width * 0.88, color=COLORS[ag], label=LABELS[ag])
            for x, v in zip(xs, vals):
                if np.isfinite(v):
                    txt = f"{100 * v:.0f}" if pct else (f"{v:.2f}" if key == "path_efficiency" else f"{v:.1f}")
                    ax.text(x, v, txt, ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels(scenarios)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        if pct:
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, .25, .5, .75, 1.0])
            ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        elif key == "path_efficiency":
            ax.set_ylim(0, 1.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(agents), frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
