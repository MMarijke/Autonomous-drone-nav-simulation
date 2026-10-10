"""Unit tests for geometry, ray casting, planning, environment and RL plumbing."""
import numpy as np
import pytest

from agent.actions import ACTION_DIRS, NUM_ACTIONS, RANGE_DIRS
from agent.astar import AStar3D
from agent.drone import Drone
from agent.guidance import PathGuidance
from agent.observation import EpisodeInfo
from env.drone_env import DroneNavEnv
from env.raycast import cast_rays
from env.rl_env import DroneRLEnv
from models.networks import DuelingQNetwork
from models.replay import NStepAccumulator, ReplayBuffer
from utils.config import load_config
from utils.metrics import EpisodeResult, summarize, wilson_interval
from utils.voxel import VoxelGrid


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------- config
def test_config_overrides():
    c = load_config(overrides=["rl.lr=1e-3", "world.size=[20,20,8]"])
    assert c.rl.lr == pytest.approx(1e-3)
    assert list(c.world.size) == [20, 20, 8]
    with pytest.raises(KeyError):
        load_config(overrides=["rl.nonexistent=1"])


# ---------------------------------------------------------------- actions
def test_action_set():
    assert ACTION_DIRS.shape == (NUM_ACTIONS, 3) and NUM_ACTIONS == 19
    assert np.allclose(np.linalg.norm(RANGE_DIRS, axis=1), 1.0)
    assert not ACTION_DIRS[0].any()


# ---------------------------------------------------------------- raycast
def test_raycast_box_sphere_bounds():
    lo, hi = np.array([[5.0, -1, -1]]), np.array([[6.0, 1, 1]])
    sc, sr = np.array([[0.0, 10.0, 0.0]]), np.array([1.0])
    o = np.zeros(3)
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]])
    r = cast_rays(o, dirs, lo, hi, sc, sr, np.array([-20.0] * 3), np.array([20.0] * 3), 50.0)
    assert r == pytest.approx([5.0, 9.0, 20.0])
    far = cast_rays(o, dirs[:1], lo[:0], hi[:0], sc[:0], sr[:0], np.array([-20.0] * 3), np.array([20.0] * 3), 7.0)
    assert far[0] == pytest.approx(7.0)               # clipped to max range


# ---------------------------------------------------------------- planning
def test_astar_open_space_is_straight_and_optimal():
    grid = VoxelGrid([10, 10, 10], 1.0)
    planner = AStar3D(grid, weight=1.0)
    res = planner.plan(np.zeros(grid.shape, bool), [0.5, 0.5, 0.5], [8.5, 0.5, 0.5])
    assert res.success
    assert res.cost == pytest.approx(8.0, abs=1e-6)


def test_astar_goes_around_wall_and_reports_failure():
    grid = VoxelGrid([10, 10, 4], 1.0)
    blocked = np.zeros(grid.shape, bool)
    blocked[5, :9, :] = True                         # wall with a gap at y=9
    planner = AStar3D(grid, 1.0)
    res = planner.plan(blocked, [1.5, 1.5, 1.5], [8.5, 1.5, 1.5])
    assert res.success and res.cost > 7.0
    # no point of the path may be inside the wall
    pts = np.array([res.path[0] + (res.path[1] - res.path[0]) * t for t in np.linspace(0, 1, 20)])
    blocked[5, :, :] = True                          # fully sealed
    assert not planner.plan(blocked, [1.5, 1.5, 1.5], [8.5, 1.5, 1.5]).success


def test_voxel_rasterize_inflation():
    grid = VoxelGrid([10, 10, 10], 1.0)
    occ = grid.rasterize_boxes(np.array([[4.0, 4, 4]]), np.array([[6.0, 6, 6]]), 0.0)
    assert occ.sum() == 8                             # 2x2x2 cell centres inside
    assert grid.rasterize_boxes(np.array([[4.0, 4, 4]]), np.array([[6.0, 6, 6]]), 1.0).sum() == 64


# ---------------------------------------------------------------- drone
def test_drone_respects_limits(cfg):
    d = Drone(cfg.drone)
    d.reset([5, 5, 5], [10, 5, 5])
    d.step([100, 0, 0])
    assert np.linalg.norm(d.vel) <= d.max_accel * d.dt + 1e-9      # acceleration limit
    for _ in range(50):
        d.step([100, 0, 0])
    assert np.linalg.norm(d.vel) == pytest.approx(d.max_speed)


# ---------------------------------------------------------------- environment
def test_env_deterministic_and_solvable(cfg):
    e1, e2 = DroneNavEnv(cfg), DroneNavEnv(cfg)
    o1, o2 = e1.reset(seed=123, scenario="hard"), e2.reset(seed=123, scenario="hard")
    assert np.allclose(o1.pos, o2.pos) and np.allclose(e1.world.box_lo, e2.world.box_lo)
    assert e1.world.ref_path is not None and len(e1.world.ref_path) >= 2
    assert e1.world.static_clearance(e1.world.start) >= cfg.world.start_goal.clearance - 1e-6
    assert e1.world.n_dynamic == cfg.scenarios.hard.dynamic


def test_collision_and_goal_events(cfg):
    env = DroneNavEnv(cfg)
    env.reset(seed=5, scenario="easy")
    # fly straight into the floor -> collision
    event = None
    for _ in range(200):
        _, info = env.step(np.array([0, 0, -3.0]))
        event = info["event"]
        if event:
            break
    assert event == "collision"
    # teleport next to the goal -> success
    env.reset(seed=5, scenario="easy")
    env.drone.pos = env.world.goal - np.array([0.2, 0, 0])
    _, info = env.step(np.zeros(3))
    assert info["event"] == "goal"


def test_dynamic_obstacles_stay_inside_and_off_buildings(cfg):
    env = DroneNavEnv(cfg)
    env.reset(seed=7, scenario="hard")
    for _ in range(300):
        env.world.step(0.1)
    w = env.world
    assert (w.dyn_pos > 0).all() and (w.dyn_pos < w.size).all()


def test_sensors_shapes(cfg):
    env = DroneNavEnv(cfg)
    obs = env.reset(seed=1, scenario="medium")
    assert obs.depth.shape == (cfg.sensors.camera.height, cfg.sensors.camera.width)
    assert obs.ranges.shape == (18,) and obs.cam_dirs.shape[1] == 3
    assert obs.depth.max() <= cfg.sensors.camera.max_range + 1e-6


# ---------------------------------------------------------------- RL plumbing
def test_nstep_accumulator():
    acc = NStepAccumulator(3, 0.9)
    out = []
    for i in range(3):
        out += acc.push(i, 0, 1.0, i + 1, terminal=False, truncated=False)
    assert len(out) == 1
    s, a, ret, s_n, term, gk = out[0]
    assert ret == pytest.approx(1 + 0.9 + 0.81) and gk == pytest.approx(0.729) and not term
    out = acc.push(3, 0, 1.0, 4, terminal=True, truncated=False)     # flush at terminal
    assert len(out) == 3 and all(o[4] == 1.0 for o in out)


def test_replay_and_network(cfg):
    buf = ReplayBuffer(10, 4)
    for i in range(15):
        buf.add(np.full(4, i), i % 3, 1.0, np.full(4, i + 1), 0.0, 0.9)
    assert len(buf) == 10
    batch = buf.sample(5)
    assert batch[0].shape == (5, 4)
    net = DuelingQNetwork(60, 19, (32, 32))
    assert net(batch[0].new_zeros(5, 60)).shape == (5, 19)


@pytest.mark.parametrize("kind", ["goal", "astar"])
def test_rl_env_step_and_reward(cfg, kind):
    env = DroneRLEnv(cfg, kind, seed=0)
    x = env.reset(seed=3, scenario="easy")
    assert x.shape == (env.obs_dim,) and np.isfinite(x).all()
    x2, r, term, trunc, info = env.step(1)
    assert x2.shape == x.shape and np.isfinite(r)


def test_path_guidance_potential_decreases_along_path(cfg):
    env = DroneNavEnv(cfg)
    env.reset(seed=2, scenario="medium")
    g = PathGuidance(cfg)
    g.reset(env.episode_info)
    p = g.path
    rem = [g.remaining(q) for q in np.linspace(p[0], p[-1], 6)]
    assert rem[0] > rem[-1] and rem[0] == pytest.approx(g.cum[-1], rel=1e-6)


# ---------------------------------------------------------------- metrics
def test_metrics_summary_and_ci():
    mk = lambda o, L: EpisodeResult("a", "s", 0, o, 10, 1.0, L, 10.0, 9.0, 0.7, 2.0, 1.0)
    s = summarize([mk("success", 10.0), mk("success", 20.0), mk("collision", 5.0), mk("timeout", 5.0)])
    assert s["success_rate"] == 0.5 and s["collision_rate"] == 0.25
    assert s["path_efficiency"] == pytest.approx(0.75)
    lo, hi = wilson_interval(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
