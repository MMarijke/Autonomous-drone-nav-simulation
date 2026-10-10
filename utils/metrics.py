"""Episode-level metrics and aggregation (collision rate, path efficiency, ...)."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

OUTCOMES = ("success", "collision", "timeout")


@dataclass
class EpisodeResult:
    agent: str
    scenario: str
    seed: int
    outcome: str                      # success | collision | timeout
    steps: int
    sim_time: float                   # seconds of simulated flight
    path_length: float                # metres actually flown
    ref_length: float                 # shortest static-map path (smoothed A*), metres
    straight_distance: float          # |start - goal|
    min_clearance: float              # smallest gap between drone hull and any surface
    mean_speed: float
    decision_ms: float                # mean wall-clock time of agent.act()
    replans: int = 0
    n_static: int = 0
    n_dynamic: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def path_efficiency(self) -> float:
        """Reference shortest path / flown path, clipped to [0, 1]."""
        if self.path_length <= 1e-6:
            return 0.0
        return float(min(1.0, self.ref_length / self.path_length))


def summarize(results: Iterable[EpisodeResult], near_miss: float = 0.5) -> dict:
    """Aggregate a list of episodes into the headline numbers used in the README."""
    results = list(results)
    n = len(results)
    if n == 0:
        return {"n": 0}
    out = np.array([r.outcome for r in results])
    succ = [r for r in results if r.outcome == "success"]

    def mean(vals):
        return float(np.mean(vals)) if len(vals) else float("nan")

    return {
        "n": n,
        "success_rate": float(np.mean(out == "success")),
        "collision_rate": float(np.mean(out == "collision")),
        "timeout_rate": float(np.mean(out == "timeout")),
        "path_efficiency": mean([r.path_efficiency for r in succ]),
        "time_to_goal_s": mean([r.sim_time for r in succ]),
        "path_length_m": mean([r.path_length for r in succ]),
        "mean_speed": mean([r.mean_speed for r in results]),
        "min_clearance_m": mean([r.min_clearance for r in results]),
        "near_miss_rate": float(np.mean([r.min_clearance < near_miss for r in results])),
        "decision_ms": mean([r.decision_ms for r in results]),
        "replans_per_episode": mean([r.replans for r in results]),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def save_results(results: list[EpisodeResult], path: str | Path) -> None:
    """Write per-episode results as CSV (one row per episode)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        d = asdict(r)
        d.pop("extra", None)
        d["path_efficiency"] = r.path_efficiency
        rows.append(d)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)


class RollingStats:
    """Fixed-window rolling mean, used for training logs."""

    def __init__(self, window: int = 100):
        self.window = window
        self.buf: list[float] = []

    def add(self, x: float) -> None:
        self.buf.append(float(x))
        if len(self.buf) > self.window:
            self.buf.pop(0)

    @property
    def mean(self) -> float:
        return float(np.mean(self.buf)) if self.buf else float("nan")
