"""OpenCV perception on the depth image.

Pipeline
--------
1. ``cv2.medianBlur`` removes dropout / impulse noise from the range image.
2. Threshold at ``danger_range`` -> binary obstacle mask.
3. Morphological opening removes speckle; ``connectedComponentsWithStats``
   splits the mask into obstacle blobs.
4. Every blob becomes a :class:`Detection` (bounding box, nearest depth,
   bearing relative to the optical axis).
5. A min-pooled, normalised depth grid is returned as the compact feature
   vector consumed by the RL policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]      # x, y, w, h in pixels
    centroid: tuple[float, float]        # (col, row)
    min_depth: float
    mean_depth: float
    azimuth: float                       # rad, + to the right of the optical axis
    elevation: float                     # rad, + above the optical axis
    area: int


@dataclass
class VisionOutput:
    filtered: np.ndarray                 # median-filtered depth (H, W)
    mask: np.ndarray                     # uint8 obstacle mask (H, W)
    detections: list[Detection] = field(default_factory=list)
    features: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    nearest: float = float("inf")


class VisionProcessor:
    def __init__(self, cfg, camera):
        self.max_range = float(camera.max_range)
        self.camera = camera
        self.danger = float(cfg.danger_range)
        self.min_area = int(cfg.min_blob_area)
        self.ksize = int(cfg.median_ksize)
        self.pool_rows, self.pool_cols = (int(v) for v in cfg.pool_grid)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        h, w = camera.h, camera.w
        if h % self.pool_rows or w % self.pool_cols:
            raise ValueError(f"camera {w}x{h} not divisible by pool grid "
                             f"{self.pool_cols}x{self.pool_rows}")

    @property
    def feature_dim(self) -> int:
        return self.pool_rows * self.pool_cols

    def process(self, depth: np.ndarray) -> VisionOutput:
        filtered = cv2.medianBlur(np.ascontiguousarray(depth, dtype=np.float32), self.ksize)

        mask = (filtered < self.danger).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detections: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.min_area:
                continue
            vals = filtered[labels == i]
            cx, cy = float(cents[i][0]), float(cents[i][1])
            az, el = self.camera.pixel_angles(cx, cy)
            detections.append(Detection((x, y, w, h), (cx, cy), float(vals.min()),
                                        float(vals.mean()), az, el, area))
        detections.sort(key=lambda d: d.min_depth)

        # Min-pooling keeps the *closest* obstacle in each cell (conservative).
        gh, gw = self.camera.h // self.pool_rows, self.camera.w // self.pool_cols
        pooled = filtered.reshape(self.pool_rows, gh, self.pool_cols, gw).min(axis=(1, 3))
        features = (pooled / self.max_range).astype(np.float32).ravel()

        return VisionOutput(filtered, mask, detections, features,
                            float(filtered.min()))

    def colorize(self, depth: np.ndarray, scale: int = 1) -> np.ndarray:
        """Depth image as a TURBO colour map (near = hot) for display."""
        norm = np.clip(depth / self.max_range, 0.0, 1.0)
        img = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return img
