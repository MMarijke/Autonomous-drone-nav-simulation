"""Real-time OpenCV visualiser.

Panels (logical canvas 1360 x 760, rescaled to ``viz.width x viz.height``):

* 3-D orbit view - buildings, floating blocks, moving obstacles, trail, A* path,
  range-finder rays, camera frustum, sensed dynamic obstacles.
* top-down map  - same scene from above with velocity arrows and camera cone.
* depth camera  - the range image as seen by the drone, with OpenCV detections.
* vision features - the min-pooled depth grid fed to the policy.
* decision panel - Q-values of the 19 actions (RL / hybrid) or planner state (A*).
* range finder, telemetry, speed / clearance history.

Use as the ``callback`` of :func:`utils.runner.run_episode`.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from agent.actions import ACTION_NAMES, NUM_ACTIONS, RANGE_DIRS
from agent.sensors import camera_basis
from utils.geometry import polyline_length

LW, LH = 1360, 760                                   # logical canvas size

# BGR palette
BG = (24, 26, 30)
PANEL = (34, 37, 43)
EDGE = (62, 66, 76)
GRID = (52, 56, 64)
TEXT = (228, 228, 228)
DIM = (150, 156, 168)
BUILDING = (168, 140, 104)
FLOATING = (96, 160, 190)
DYNAMIC = (60, 120, 255)
DRONE = (255, 245, 130)
PATH = (255, 200, 20)
TRAIL = (110, 235, 255)
GOAL = (90, 225, 110)
WARN = (60, 200, 255)
BAD = (70, 70, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _i(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def put(img, text, org, scale=0.45, color=TEXT, thick=1):
    cv2.putText(img, text, _i(org), FONT, scale, color, thick, cv2.LINE_AA)


def shade(color, f):
    return tuple(int(np.clip(c * f, 0, 255)) for c in color)


def range_color(r: float, rmax: float) -> tuple[int, int, int]:
    """Red (near) -> amber -> green (far), BGR."""
    t = float(np.clip(r / rmax, 0, 1))
    near, mid, far = np.array([70, 70, 235.0]), np.array([60, 200, 250.0]), np.array([110, 215, 100.0])
    col = near + (mid - near) * (t / 0.4) if t < 0.4 else mid + (far - mid) * ((t - 0.4) / 0.6)
    return int(col[0]), int(col[1]), int(col[2])


class OrbitCamera:
    """Perspective camera orbiting a target point (for the 3-D panel)."""

    def __init__(self, target, dist, az_deg, el_deg, fov_deg, w, h):
        self.target = np.asarray(target, float)
        self.dist, self.az, self.el = float(dist), float(az_deg), float(el_deg)
        self.fov, self.w, self.h = float(fov_deg), w, h
        self.offset = np.zeros(2)
        self.zoom = 1.0
        self._update()

    def _update(self) -> None:
        az, el = np.radians(self.az), np.radians(np.clip(self.el, 5, 85))
        self.pos = self.target + self.dist * np.array([np.cos(el) * np.cos(az),
                                                       np.cos(el) * np.sin(az), np.sin(el)])
        f = self.target - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.basis = np.stack([r, u, f])
        self.focal = (self.h / 2) / np.tan(np.radians(self.fov) / 2)

    def set(self, az=None, el=None, zoom=None) -> None:
        if az is not None:
            self.az = az
        if el is not None:
            self.el = float(np.clip(el, 5, 85))
        if zoom is not None:
            self.zoom = float(np.clip(zoom, 0.5, 2.5))
        self._update()

    def fit(self, corners: np.ndarray, margin: float = 0.94) -> None:
        """Choose distance + image offset so ``corners`` fill the viewport (re-run
        whenever the orbit angle changes, so the room stays centred while rotating)."""
        for _ in range(3):
            self.offset = np.zeros(2)
            self._update()
            xy, _ = self.project(corners)
            bw, bh = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
            scale = min(self.w * margin / max(bw, 1e-6), self.h * margin / max(bh, 1e-6)) * self.zoom
            self.dist = float(np.clip(self.dist / scale, 20, 200))
        self._update()
        xy, _ = self.project(corners)
        centre = np.array([(xy[:, 0].min() + xy[:, 0].max()) / 2, (xy[:, 1].min() + xy[:, 1].max()) / 2])
        self.offset = np.array([self.w / 2, self.h / 2]) - centre

    def project(self, pts: np.ndarray):
        pts = np.atleast_2d(pts)
        rel = (pts - self.pos) @ self.basis.T
        z = np.maximum(rel[:, 2], 1e-3)
        sx = self.w / 2 + self.offset[0] + self.focal * rel[:, 0] / z
        sy = self.h / 2 + self.offset[1] - self.focal * rel[:, 1] / z
        return np.stack([sx, sy], axis=1), rel[:, 2]


# --------------------------------------------------------------------------
class Visualizer:
    PANELS = {
        "view3d": (0, 36, 800, 520),
        "top": (800, 36, 280, 280),
        "depth": (1080, 36, 280, 168),
        "feat": (1080, 208, 280, 108),
        "decision": (800, 320, 560, 236),
        "ranges": (0, 560, 560, 200),
        "telemetry": (560, 560, 400, 200),
        "history": (960, 560, 400, 200),
    }

    def __init__(self, cfg, display: bool = True, video_path: str | None = None,
                 video_fps: float | None = None, speed: float = 1.0, hold_seconds: float = 1.2):
        self.cfg = cfg
        vc = cfg.viz
        self.out_w, self.out_h = int(vc.width), int(vc.height)
        self.display = display
        self.speed = float(speed)
        self.dt = float(cfg.drone.dt)
        self.trail = deque(maxlen=int(vc.trail_length))
        self.history = deque(maxlen=300)
        self.show_rays = bool(vc.show_rays)
        self.auto_orbit = float(vc.orbit_speed_deg_per_s) != 0.0
        self.orbit_speed = float(vc.orbit_speed_deg_per_s)
        size = np.asarray(cfg.world.size, float)
        self.size = size
        self.cam3d = OrbitCamera(size * [0.5, 0.5, 0.4], 64.0, vc.orbit_azimuth_deg,
                                 vc.orbit_elevation_deg, 38.0, 800, 520)
        self.room = np.array([[x, y, z] for z in (0, size[2]) for y in (0, size[1]) for x in (0, size[0])], float)
        self.cam3d.fit(self.room)
        self.quit = False
        self.skip = False
        self.paused = False
        self.hold = float(hold_seconds)
        self.frame_idx = 0
        self.last_frame: np.ndarray | None = None
        self.replan_marks: deque = deque(maxlen=60)
        self.label = ""
        self.window = "Drone navigation - A* vs RL"
        self._last_t = time.perf_counter()
        self.writer = None
        if video_path:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            fps = video_fps or 1.0 / self.dt
            self.writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                          (self.out_w, self.out_h))
            if not self.writer.isOpened():
                raise RuntimeError(f"could not open video writer for {video_path}")
        if display:
            cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)

    # ------------------------------------------------------------------ api
    def __call__(self, env, agent, obs, cmd, info) -> bool:
        if cmd is None:                               # first call of an episode
            self.trail.clear()
            self.history.clear()
            self.replan_marks.clear()
        self.trail.append(obs.pos.copy())
        gap = float(obs.ranges.min()) - env.drone.radius
        self.history.append((float(np.linalg.norm(obs.vel)), gap))
        if getattr(agent, "info", None) and agent.info.get("replanned"):
            self.replan_marks.append(env.steps)
        frame = self.render(env, agent, obs, cmd, info)
        self.last_frame = frame
        self._emit(frame)
        event = info["event"] if info else None
        if event:
            n_hold = int(self.hold / self.dt) if self.writer else 0
            for _ in range(n_hold):
                self._emit(frame, wait=False)
            if self.display:
                cv2.waitKey(int(self.hold * 1000))
        self.frame_idx += 1
        if self.auto_orbit:
            self.cam3d.set(az=self.cam3d.az + self.orbit_speed * self.dt)
        self.cam3d.fit(self.room)
        return not (self.quit or self.skip)

    def _emit(self, frame: np.ndarray, wait: bool = True) -> None:
        if self.writer is not None:
            self.writer.write(frame)
        if self.display and wait:
            cv2.imshow(self.window, frame)
            target = self.dt / max(self.speed, 1e-3)
            spent = time.perf_counter() - self._last_t
            key = cv2.waitKey(max(1, int((target - spent) * 1000))) & 0xFF
            self._last_t = time.perf_counter()
            self._handle_key(key)
            while self.paused and not self.quit:
                self._handle_key(cv2.waitKey(50) & 0xFF)

    def _handle_key(self, key: int) -> None:
        if key in (255, 0):
            return
        c = chr(key) if 32 <= key < 127 else ""
        if key == 27 or c == "q":
            self.quit = True
        elif c == " ":
            self.paused = not self.paused
        elif c == "n":
            self.skip = True
        elif c == "o":
            self.auto_orbit = not self.auto_orbit
        elif c == "j":
            self.cam3d.set(az=self.cam3d.az - 6)
        elif c == "l":
            self.cam3d.set(az=self.cam3d.az + 6)
        elif c == "i":
            self.cam3d.set(el=self.cam3d.el + 4)
        elif c == "k":
            self.cam3d.set(el=self.cam3d.el - 4)
        elif c == "z":
            self.cam3d.set(zoom=self.cam3d.zoom * 1.1)
        elif c == "x":
            self.cam3d.set(zoom=self.cam3d.zoom / 1.1)
        elif c == "r":
            self.show_rays = not self.show_rays
        elif c in "+=":
            self.speed = min(self.speed * 1.5, 16)
        elif c in "-_":
            self.speed = max(self.speed / 1.5, 0.1)

    def new_episode(self) -> None:
        self.skip = False

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.display:
            cv2.destroyAllWindows()

    def save_png(self, path: str) -> None:
        if self.last_frame is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), self.last_frame)

    # --------------------------------------------------------------- render
    def render(self, env, agent, obs, cmd, info) -> np.ndarray:
        canvas = np.full((LH, LW, 3), BG, np.uint8)
        for name, (x, y, w, h) in self.PANELS.items():
            cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), PANEL, -1)
            cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), EDGE, 1)
        ainfo = getattr(agent, "info", {}) or {}
        self._header(canvas, env, agent, obs, info)
        self._view3d(canvas, env, obs, ainfo)
        self._topdown(canvas, env, obs, ainfo)
        self._depth(canvas, env, obs, ainfo)
        self._decision(canvas, env, agent, obs, ainfo)
        self._ranges(canvas, env, obs, ainfo)
        self._telemetry(canvas, env, agent, obs, ainfo, cmd, info)
        self._history(canvas)
        if info and info.get("event"):
            self._banner(canvas, info["event"])
        if (self.out_w, self.out_h) != (LW, LH):
            canvas = cv2.resize(canvas, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return canvas

    # ---- header -------------------------------------------------------
    def _header(self, c, env, agent, obs, info):
        cv2.rectangle(c, (0, 0), (LW, 35), (40, 44, 52), -1)
        name = {"astar": "A* planner (baseline)", "rl": "DQN policy (end-to-end RL)",
                "hybrid": "A* waypoints + DQN local policy"}.get(agent.name, agent.name)
        put(c, name, (12, 24), 0.62, (255, 255, 255), 1)
        put(c, f"scenario: {env.scenario_name}  |  seed {env.seed}  |  {env.world.n_static} buildings/blocks, "
               f"{env.world.n_dynamic} moving", (430, 24), 0.48, DIM)
        put(c, f"t = {env.t:5.1f}s   step {env.steps:3d}   x{self.speed:.1f}", (1130, 24), 0.5, TEXT)

    # ---- 3-D view -----------------------------------------------------
    def _view3d(self, c, env, obs, ainfo):
        x0, y0, w, h = self.PANELS["view3d"]
        view = c[y0:y0 + h, x0:x0 + w]
        W, D, H = self.size
        cam = self.cam3d

        def L(a, b, color, thick=1):
            xy, z = cam.project(np.vstack([a, b]))
            if (z > 0.5).all():
                cv2.line(view, _i(xy[0]), _i(xy[1]), color, thick, cv2.LINE_AA)

        # floor grid + room wireframe
        for gx in np.arange(0, W + 0.1, 4.0):
            L([gx, 0, 0], [gx, D, 0], GRID)
        for gy in np.arange(0, D + 0.1, 4.0):
            L([0, gy, 0], [W, gy, 0], GRID)
        corners = np.array([[x, y, z] for z in (0, H) for y in (0, D) for x in (0, W)], float)
        for i, j in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
            if i < 4 and j < 4 and (i, j) in [(0, 1), (2, 3), (0, 2), (1, 3)]:
                continue                                    # floor drawn as grid
            L(corners[i], corners[j], (70, 74, 84))

        w_ = env.world
        items = []                                          # (depth, kind, payload)
        for lo, hi in zip(w_.box_lo, w_.box_hi):
            ground = lo[2] < 0.05
            base = BUILDING if ground else FLOATING
            # ground shadow
            sh = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
            if not ground:
                xy, _ = cam.project(sh)
                cv2.fillPoly(view, [xy.astype(np.int32)], (28, 30, 34))
            ctr = (lo + hi) / 2
            half = (hi - lo) / 2
            for ax, shade_f in ((2, 1.0), (0, 0.76), (1, 0.56)):
                for sgn in (1, -1):
                    n = np.zeros(3)
                    n[ax] = sgn
                    fc = ctr + n * half
                    if n @ (cam.pos - fc) <= 0:
                        continue
                    o = [a for a in range(3) if a != ax]
                    pts = []
                    for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                        p = fc.copy()
                        p[o[0]] += sa * half[o[0]]
                        p[o[1]] += sb * half[o[1]]
                        pts.append(p)
                    xy, z = cam.project(np.array(pts))
                    if (z > 0.5).all():
                        items.append((float(np.linalg.norm(cam.pos - fc)), "poly",
                                      (xy.astype(np.int32), shade(base, shade_f * (0.8 + 0.2 * min(fc[2] / H + 0.5, 1))))))
        for pos, rad in zip(w_.dyn_pos, w_.dyn_rad):
            items.append((float(np.linalg.norm(cam.pos - pos)), "sphere", (pos, rad)))
            # shadow on floor
            xy, _ = cam.project(np.array([[pos[0], pos[1], 0.0]]))
            cv2.circle(view, _i(xy[0]), max(int(rad * cam.focal / cam.dist * 0.9), 2), (30, 32, 36), -1)
            L(pos, [pos[0], pos[1], 0], (60, 64, 74))
        items.append((float(np.linalg.norm(cam.pos - obs.pos)), "drone", obs.pos))
        # goal pole
        g = w_.goal
        L(g, [g[0], g[1], 0], (60, 130, 70))
        items.sort(key=lambda t: -t[0])

        for _, kind, pl in items:
            if kind == "poly":
                cv2.fillPoly(view, [pl[0]], pl[1])
                cv2.polylines(view, [pl[0]], True, shade(pl[1], 0.55), 1, cv2.LINE_AA)
            elif kind == "sphere":
                pos, rad = pl
                xy, z = cam.project(pos[None])
                rp = max(int(rad * cam.focal / z[0]), 3)
                cv2.circle(view, _i(xy[0]), rp, shade(DYNAMIC, 0.75), -1, cv2.LINE_AA)
                cv2.circle(view, _i(xy[0] + [-rp * 0.25, -rp * 0.25]), max(rp // 2, 1), DYNAMIC, -1, cv2.LINE_AA)
                cv2.circle(view, _i(xy[0]), rp, shade(DYNAMIC, 0.5), 1, cv2.LINE_AA)
            else:
                self._draw_drone3d(view, cam, pl, obs.heading)

        # goal & start markers
        xy, z = cam.project(np.vstack([g, w_.start]))
        cv2.circle(view, _i(xy[0]), 9, GOAL, 2, cv2.LINE_AA)
        cv2.circle(view, _i(xy[0]), 3, GOAL, -1, cv2.LINE_AA)
        put(view, "GOAL", xy[0] + [12, 4], 0.4, GOAL)
        cv2.drawMarker(view, _i(xy[1]), (170, 170, 170), cv2.MARKER_DIAMOND, 8, 1, cv2.LINE_AA)

        # planned path
        path = ainfo.get("path")
        if path is not None and len(path) >= 2:
            xy, z = cam.project(path)
            if (z > 0.5).all():
                cv2.polylines(view, [xy.astype(np.int32)], False, PATH, 2, cv2.LINE_AA)
                for p in xy:
                    cv2.circle(view, _i(p), 3, PATH, -1, cv2.LINE_AA)
        la = ainfo.get("lookahead")
        if la is not None:
            xy, _ = cam.project(np.asarray(la)[None])
            cv2.drawMarker(view, _i(xy[0]), (255, 255, 255), cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)

        # trail (fading)
        if len(self.trail) > 1:
            tr = np.array(self.trail)
            xy, z = cam.project(tr)
            n = len(tr)
            for k in range(1, n):
                f = 0.25 + 0.75 * k / n
                cv2.line(view, _i(xy[k - 1]), _i(xy[k]), shade(TRAIL, f), 2, cv2.LINE_AA)

        # sensed dynamic obstacle points
        hits = ainfo.get("dyn_hits")
        if hits is not None and len(hits):
            xy, z = cam.project(hits[::max(len(hits) // 150, 1)])
            for p in xy:
                cv2.circle(view, _i(p), 2, (40, 150, 255), -1)

        # range rays + camera frustum
        if self.show_rays:
            rmax = float(self.cfg.sensors.rangefinder.max_range)
            for d, r in zip(RANGE_DIRS, obs.ranges):
                if r < rmax - 0.05:
                    L(obs.pos, obs.pos + d * r, range_color(r, rmax), 1)
            b = camera_basis(obs.heading)
            tan_h = np.tan(np.radians(self.cfg.sensors.camera.hfov_deg) / 2)
            tan_v = np.tan(np.radians(self.cfg.sensors.camera.vfov_deg) / 2)
            cs = [obs.pos + 5.0 * (b[2] + sx * tan_h * b[0] + sy * tan_v * b[1]) for sx, sy in
                  ((-1, 1), (1, 1), (1, -1), (-1, -1))]
            for k in range(4):
                L(obs.pos, cs[k], (90, 160, 190), 1)
                L(cs[k], cs[(k + 1) % 4], (90, 160, 190), 1)

        # legend
        lx, ly = 12, 18
        for col, txt in ((BUILDING, "building"), (FLOATING, "floating block"), (DYNAMIC, "moving obstacle"),
                         (PATH, "planned path"), (TRAIL, "flown trail")):
            cv2.rectangle(view, (lx, ly - 8), (lx + 10, ly + 2), col, -1)
            put(view, txt, (lx + 15, ly + 2), 0.38, DIM)
            ly += 15
        put(view, "keys: q quit | space pause | n next | j/l i/k orbit | z/x zoom | o auto-orbit | r rays | +/- speed",
            (12, h - 10), 0.36, (110, 114, 124))
        put(c, "3-D VIEW", (x0 + w - 78, y0 + 16), 0.4, DIM)

    def _draw_drone3d(self, view, cam, pos, heading):
        xy, z = cam.project(pos[None])
        s = cam.focal * 0.45 / z[0]
        ctr = xy[0]
        arms = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
        for sx, sy in arms:
            p = ctr + np.array([sx * s, sy * s * 0.55])
            cv2.line(view, _i(ctr), _i(p), (200, 200, 200), 2, cv2.LINE_AA)
            cv2.circle(view, _i(p), max(int(s * 0.45), 2), DRONE, -1, cv2.LINE_AA)
            cv2.circle(view, _i(p), max(int(s * 0.45), 2), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.circle(view, _i(ctr), max(int(s * 0.4), 3), (255, 255, 255), -1, cv2.LINE_AA)
        tip, _ = cam.project((pos + heading * 1.4)[None])
        cv2.line(view, _i(ctr), _i(tip[0]), (80, 80, 255), 2, cv2.LINE_AA)

    # ---- top-down -----------------------------------------------------
    def _topdown(self, c, env, obs, ainfo):
        x0, y0, w, h = self.PANELS["top"]
        v = c[y0:y0 + h, x0:x0 + w]                    # clipped sub-image: nothing leaks out
        W, D, H = self.size
        sc = (w - 10) / W
        ox, oy = 5, 5

        def P(p):
            return (ox + p[0] * sc, oy + (D - p[1]) * sc)

        cv2.rectangle(v, _i(P([0, D])), _i(P([W, 0])), (28, 30, 35), -1)
        for g in np.arange(0, W + 0.1, 4.0):
            cv2.line(v, _i(P([g, 0])), _i(P([g, D])), GRID, 1)
            cv2.line(v, _i(P([0, g])), _i(P([W, g])), GRID, 1)
        w_ = env.world
        for lo, hi in sorted(zip(w_.box_lo, w_.box_hi), key=lambda t: t[1][2]):
            col = shade(BUILDING if lo[2] < 0.05 else FLOATING, 0.6 + 0.4 * hi[2] / H)
            cv2.rectangle(v, _i(P([lo[0], hi[1]])), _i(P([hi[0], lo[1]])), col, -1)
            cv2.rectangle(v, _i(P([lo[0], hi[1]])), _i(P([hi[0], lo[1]])), shade(col, 0.55), 1)
        # camera cone (translucent, clipped to the world square)
        az = np.arctan2(obs.heading[1], obs.heading[0])
        half = np.radians(self.cfg.sensors.camera.hfov_deg) / 2
        rr = float(self.cfg.sensors.camera.max_range)
        cone = np.array([P(obs.pos)] + [P(obs.pos + rr * np.array([np.cos(az + a), np.sin(az + a), 0]))
                                        for a in np.linspace(-half, half, 12)], np.int32)
        ov = v.copy()
        cv2.fillPoly(ov, [cone], (150, 120, 70))
        cv2.addWeighted(ov, 0.30, v, 0.70, 0, v)
        path = ainfo.get("path")
        if path is not None and len(path) >= 2:
            cv2.polylines(v, [np.array([_i(P(p)) for p in path], np.int32)], False, PATH, 1, cv2.LINE_AA)
        if len(self.trail) > 1:
            cv2.polylines(v, [np.array([_i(P(p)) for p in self.trail], np.int32)], False, TRAIL, 1, cv2.LINE_AA)
        hits = ainfo.get("dyn_hits")
        if hits is not None and len(hits):
            for p in hits[::max(len(hits) // 120, 1)]:
                cv2.circle(v, _i(P(p)), 1, (40, 150, 255), -1)
        for pos, vel, rad in zip(w_.dyn_pos, w_.dyn_vel, w_.dyn_rad):
            cv2.circle(v, _i(P(pos)), max(int(rad * sc), 2), DYNAMIC, -1, cv2.LINE_AA)
            cv2.arrowedLine(v, _i(P(pos)), _i(P(pos + vel * 1.0)), (120, 190, 255), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(v, _i(P(w_.goal)), int(env.goal_radius * sc) + 2, GOAL, 1, cv2.LINE_AA)
        cv2.circle(v, _i(P(w_.goal)), 2, GOAL, -1)
        la = ainfo.get("lookahead")
        if la is not None:
            cv2.drawMarker(v, _i(P(la)), (255, 255, 255), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        cv2.circle(v, _i(P(obs.pos)), 4, DRONE, -1, cv2.LINE_AA)
        cv2.arrowedLine(v, _i(P(obs.pos)), _i(P(obs.pos + obs.heading * 2.2)), (80, 80, 255), 2, cv2.LINE_AA, tipLength=0.35)
        put(v, "TOP-DOWN", (w - 74, 16), 0.4, DIM)
        put(v, f"z = {obs.pos[2]:.1f} m", (8, h - 8), 0.4, TEXT)

    # ---- depth camera ---------------------------------------------------
    def _depth(self, c, env, obs, ainfo):
        x0, y0, w, h = self.PANELS["depth"]
        vis = ainfo.get("vision")
        cam = env.sensors.camera
        scale = w // cam.w
        img = cv2.applyColorMap(((1 - np.clip(obs.depth / cam.max_range, 0, 1)) * 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
        img = cv2.resize(img, (cam.w * scale, cam.h * scale), interpolation=cv2.INTER_NEAREST)
        if vis is not None:
            for k, d in enumerate(vis.detections[:5]):
                bx, by, bw, bh = d.bbox
                col = (255, 255, 255) if k == 0 else (210, 210, 210)
                cv2.rectangle(img, (bx * scale, by * scale), ((bx + bw) * scale, (by + bh) * scale), col, 1)
                put(img, f"{d.min_depth:.1f}m", (bx * scale + 2, max(by * scale - 3, 10)), 0.38, col)
        cv2.drawMarker(img, (img.shape[1] // 2, img.shape[0] // 2), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        c[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        put(c, "DEPTH CAMERA (near=red)", (x0 + 6, y0 + 14), 0.4, (255, 255, 255))
        # pooled features
        fx, fy, fw, fh = self.PANELS["feat"]
        put(c, "VISION FEATURES  (min-pooled depth, m)", (fx + 6, fy + 14), 0.4, DIM)
        if vis is not None:
            rows, cols = 3, 5
            cw, ch = (fw - 12) // cols, (fh - 28) // rows
            vals = vis.features.reshape(rows, cols)
            for r in range(rows):
                for q in range(cols):
                    v = float(vals[r, q])
                    col = cv2.applyColorMap(np.uint8([[int((1 - v) * 255)]]), cv2.COLORMAP_TURBO)[0, 0]
                    p0 = (fx + 6 + q * cw, fy + 22 + r * ch)
                    cv2.rectangle(c, p0, (p0[0] + cw - 2, p0[1] + ch - 2), tuple(int(t) for t in col), -1)
                    put(c, f"{v * cam.max_range:.1f}", (p0[0] + 8, p0[1] + ch // 2 + 4), 0.4, (20, 20, 20), 1)

    # ---- decision panel -----------------------------------------------------
    def _decision(self, c, env, agent, obs, ainfo):
        x0, y0, w, h = self.PANELS["decision"]
        put(c, "DECISION MAKING", (x0 + 10, y0 + 18), 0.5, (255, 255, 255))
        put(c, ainfo.get("mode", "-"), (x0 + 170, y0 + 18), 0.45, WARN)
        if "q" in ainfo:
            q = ainfo["q"]
            a = ainfo["action"]
            lo, hi = float(q.min()), float(q.max())
            top2 = np.sort(q)[-2:]
            row = (h - 52) / NUM_ACTIONS
            for k in range(NUM_ACTIONS):
                yy = y0 + 32 + k * row
                frac = (q[k] - lo) / max(hi - lo, 1e-6)
                col = (60, 230, 120) if k == a else (120, 130, 150)
                put(c, ACTION_NAMES[k], (x0 + 10, yy + row - 3), 0.36, TEXT if k == a else DIM)
                cv2.rectangle(c, (x0 + 62, int(yy + 1)), (x0 + 62 + int(240 * frac), int(yy + row - 2)), col, -1)
                put(c, f"{q[k]:+.2f}", (x0 + 312, yy + row - 3), 0.34, TEXT if k == a else DIM)
            put(c, "Q-values (bar = min-max scaled)", (x0 + 62, y0 + h - 8), 0.36, DIM)
            gx = x0 + 370
            put(c, f"chosen: {ainfo['action_name']}", (gx, y0 + 50), 0.5, (60, 230, 120))
            put(c, f"max Q: {top2[1]:+.2f}", (gx, y0 + 76), 0.42, TEXT)
            put(c, f"margin over 2nd: {top2[1] - top2[0]:.3f}", (gx, y0 + 96), 0.42, TEXT)
            la = ainfo.get("lookahead")
            if la is not None:
                d = np.asarray(la) - obs.pos
                put(c, f"guidance: {'A* waypoint' if agent.name == 'hybrid' else 'goal'}", (gx, y0 + 126), 0.42, DIM)
                put(c, f"  -> {np.linalg.norm(d):.1f} m away", (gx, y0 + 144), 0.42, TEXT)
                put(c, f"  dz = {d[2]:+.1f} m", (gx, y0 + 162), 0.42, TEXT)
            if agent.name == "hybrid":
                put(c, f"path replans: {agent.replans}", (gx, y0 + 186), 0.42, DIM)
        else:                                          # A* planner
            plan = ainfo.get("plan", {})
            path = ainfo.get("path")
            lines = [
                ("last plan", plan.get("reason", "-")),
                ("expansions", f"{plan.get('expansions', 0)}"),
                ("plan time", f"{plan.get('plan_ms', 0.0):.1f} ms"),
                ("path waypoints", f"{0 if path is None else len(path)}"),
                ("path length", f"{0.0 if path is None else float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()):.1f} m"),
                ("replans so far", f"{agent.replans}"),
                ("sensed dyn. voxels", f"{ainfo.get('dyn_cells', 0)}"),
            ]
            for k, (a, b) in enumerate(lines):
                put(c, a, (x0 + 14, y0 + 48 + 21 * k), 0.45, DIM)
                put(c, b, (x0 + 200, y0 + 48 + 21 * k), 0.45, TEXT)
            s = float(ainfo.get("safety_scale", 1.0))
            put(c, "safety layer: speed scale", (x0 + 14, y0 + h - 30), 0.42, DIM)
            cv2.rectangle(c, (x0 + 14, y0 + h - 22), (x0 + 254, y0 + h - 10), (60, 64, 74), -1)
            cv2.rectangle(c, (x0 + 14, y0 + h - 22), (x0 + 14 + int(240 * s), y0 + h - 10),
                          (60, 230, 120) if s > 0.95 else WARN, -1)
            put(c, f"{s:.2f}", (x0 + 262, y0 + h - 11), 0.4, TEXT)
            # replan timeline
            tx, ty = x0 + 310, y0 + 60
            put(c, "replan timeline (last 60 steps)", (tx, ty - 14), 0.4, DIM)
            cv2.rectangle(c, (tx, ty), (tx + 230, ty + 26), (28, 30, 35), -1)
            for stp in self.replan_marks:
                rel = (env.steps - stp)
                if 0 <= rel < 60:
                    xx = tx + 230 - int(rel * 230 / 60)
                    cv2.line(c, (xx, ty + 2), (xx, ty + 24), PATH, 2)
            put(c, "A* re-plans when the path ahead is blocked,", (tx, ty + 56), 0.38, DIM)
            put(c, "the drone leaves it, or every 20 steps.", (tx, ty + 72), 0.38, DIM)

    # ---- range finder ----------------------------------------------------------
    def _ranges(self, c, env, obs, ainfo):
        x0, y0, w, h = self.PANELS["ranges"]
        put(c, "RANGE FINDER  (one ray per action direction)", (x0 + 10, y0 + 18), 0.5, (255, 255, 255))
        rmax = float(self.cfg.sensors.rangefinder.max_range)
        n = len(obs.ranges)
        bw = (w - 20) / n
        base = y0 + h - 34
        top = y0 + 34
        chosen = ainfo.get("action")
        for k, r in enumerate(obs.ranges):
            x = x0 + 10 + k * bw
            hh = (base - top) * (r / rmax)
            col = range_color(r, rmax)
            cv2.rectangle(c, (int(x + 2), int(base - hh)), (int(x + bw - 3), int(base)), col, -1)
            if chosen is not None and chosen - 1 == k:
                cv2.rectangle(c, (int(x), int(top - 4)), (int(x + bw - 1), int(base + 3)), (255, 255, 255), 1)
            put(c, ACTION_NAMES[k + 1], (x + 1, base + 14), 0.27, DIM)
            put(c, f"{r:.0f}", (x + 6, base - hh - 3), 0.3, TEXT)
        put(c, "metres (white box = chosen direction)", (x0 + 10, y0 + h - 6), 0.34, DIM)

    # ---- telemetry --------------------------------------------------------------
    def _telemetry(self, c, env, agent, obs, ainfo, cmd, info):
        x0, y0, w, h = self.PANELS["telemetry"]
        put(c, "TELEMETRY", (x0 + 10, y0 + 18), 0.5, (255, 255, 255))
        gap = float(obs.ranges.min()) - env.drone.radius
        true_gap = info["gap"] if info else gap
        dgoal = float(np.linalg.norm(obs.goal - obs.pos))
        ref_len = polyline_length(env.world.ref_path)
        eff = ref_len / max(env.drone.path_length, 1e-6)
        rows = [
            ("position", f"({obs.pos[0]:5.1f}, {obs.pos[1]:5.1f}, {obs.pos[2]:4.1f}) m"),
            ("speed", f"{np.linalg.norm(obs.vel):.2f} m/s   (max {env.drone.max_speed:.1f})"),
            ("distance to goal", f"{dgoal:.1f} m"),
            ("flown / reference", f"{env.drone.path_length:.1f} / {ref_len:.1f} m"),
            ("path efficiency so far", f"{min(eff, 1.0):.2f}"),
            ("clearance (true / sensed)", f"{true_gap:.2f} / {gap:.2f} m"),
            ("min clearance this run", f"{env.min_gap:.2f} m"),
            ("velocity command", "-" if cmd is None else f"({cmd[0]:+.1f}, {cmd[1]:+.1f}, {cmd[2]:+.1f})"),
        ]
        for k, (a, b) in enumerate(rows):
            put(c, a, (x0 + 12, y0 + 42 + 17 * k), 0.42, DIM)
            col = BAD if (a.startswith("clearance") and true_gap < 0.4) else TEXT
            put(c, b, (x0 + 190, y0 + 42 + 17 * k), 0.42, col)
        # goal-distance progress bar
        total = float(np.linalg.norm(env.world.goal - env.world.start))
        frac = float(np.clip(1.0 - dgoal / max(total, 1e-6), 0, 1))
        cv2.rectangle(c, (x0 + 12, y0 + h - 22), (x0 + w - 12, y0 + h - 10), (60, 64, 74), -1)
        cv2.rectangle(c, (x0 + 12, y0 + h - 22), (x0 + 12 + int((w - 24) * frac), y0 + h - 10), GOAL, -1)

    # ---- history ---------------------------------------------------------------
    def _history(self, c):
        x0, y0, w, h = self.PANELS["history"]
        put(c, "SPEED (blue) / CLEARANCE (orange)", (x0 + 10, y0 + 18), 0.45, (255, 255, 255))
        px0, py0, pw, ph = x0 + 12, y0 + 34, w - 24, h - 70
        cv2.rectangle(c, (px0, py0), (px0 + pw, py0 + ph), (28, 30, 35), -1)
        if len(self.history) > 1:
            hs = np.array(self.history)
            n = len(hs)
            xs = px0 + np.arange(n) * pw / max(self.history.maxlen - 1, 1)
            sp = py0 + ph - np.clip(hs[:, 0] / 3.5, 0, 1) * ph
            gp = py0 + ph - np.clip(hs[:, 1] / 4.0, 0, 1) * ph
            cv2.polylines(c, [np.stack([xs, sp], 1).astype(np.int32)], False, (255, 160, 60), 1, cv2.LINE_AA)
            cv2.polylines(c, [np.stack([xs, gp], 1).astype(np.int32)], False, (60, 160, 255), 1, cv2.LINE_AA)
            danger = py0 + ph - (0.5 / 4.0) * ph
            cv2.line(c, (px0, int(danger)), (px0 + pw, int(danger)), (70, 70, 200), 1)
        put(c, "red line = 0.5 m near-miss", (px0, y0 + h - 14), 0.36, DIM)

    def _banner(self, c, event):
        text, col = {"goal": ("GOAL REACHED", GOAL), "collision": ("COLLISION", BAD),
                     "timeout": ("TIME OUT", WARN)}[event]
        x0, y0, w, h = self.PANELS["view3d"]
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.3, 3)
        bx, by = x0 + w // 2 - tw // 2, y0 + h // 2
        ov = c.copy()
        cv2.rectangle(ov, (bx - 20, by - th - 14), (bx + tw + 20, by + 16), (15, 15, 18), -1)
        cv2.addWeighted(ov, 0.7, c, 0.3, 0, c)
        cv2.putText(c, text, (bx, by), FONT, 1.3, col, 3, cv2.LINE_AA)
