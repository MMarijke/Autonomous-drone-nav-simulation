"""Run one episode of any agent in the simulator and collect its metrics."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from utils.geometry import polyline_length
from utils.metrics import EpisodeResult

OUTCOME = {"goal": "success", "collision": "collision", "timeout": "timeout"}


def run_episode(env, agent, seed: int, scenario: str = "medium",
                callback: Callable | None = None, **reset_kwargs) -> EpisodeResult:
    """Roll out ``agent`` in ``env``.

    ``callback(env, agent, obs, cmd, info)`` is invoked once after reset
    (``cmd=info=None``) and after every step; returning ``False`` aborts the
    episode (used by the live viewer when the user presses *q*).
    """
    obs = env.reset(seed=seed, scenario=scenario, **reset_kwargs)
    agent.reset(env.episode_info)
    if callback is not None and callback(env, agent, obs, None, None) is False:
        return _result(env, agent, "timeout", [], [])

    decision_ms: list[float] = []
    speeds: list[float] = []
    event = None
    while event is None:
        t0 = time.perf_counter()
        cmd = agent.act(obs)
        decision_ms.append((time.perf_counter() - t0) * 1e3)
        obs, info = env.step(cmd)
        speeds.append(float(np.linalg.norm(obs.vel)))
        event = info["event"]
        if callback is not None and callback(env, agent, obs, cmd, info) is False:
            break
    return _result(env, agent, OUTCOME.get(event, "timeout"), decision_ms, speeds)


def _result(env, agent, outcome: str, decision_ms: list[float], speeds: list[float]) -> EpisodeResult:
    w = env.world
    ref = polyline_length(w.ref_path) if w.ref_path is not None else float("nan")
    return EpisodeResult(
        agent=getattr(agent, "name", type(agent).__name__), scenario=env.scenario_name, seed=env.seed,
        outcome=outcome, steps=env.steps, sim_time=env.t, path_length=env.drone.path_length,
        ref_length=ref, straight_distance=float(np.linalg.norm(w.goal - w.start)),
        min_clearance=float(env.min_gap), mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        decision_ms=float(np.mean(decision_ms)) if decision_ms else 0.0,
        replans=int(getattr(agent, "replans", 0)), n_static=w.n_static, n_dynamic=w.n_dynamic)
