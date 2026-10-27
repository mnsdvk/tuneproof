"""The workload CoolPilot optimises: a real OpenCV 5 anomaly-detection pipeline.

Each stage is timed separately so the agent can see WHERE the time goes.
Every knob trades speed for fidelity, which is exactly why a quality guard is needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

DEFAULT_THREADS = cv2.getNumThreads()


@dataclass(frozen=True)
class Cfg:
    scale: float = 1.0        # work at a fraction of the input resolution
    bg: str = "median"        # background model: median | median_ds2 | median_ds4 | box
    ksize: int = 41           # background window (at full resolution)
    threads: int = DEFAULT_THREADS

    def label(self) -> str:
        return f"scale={self.scale} bg={self.bg} k={self.ksize} thr={self.threads}"

    def with_(self, **kw) -> "Cfg":
        return replace(self, **kw)


BASELINE = Cfg()
STAGES = ("resize", "background", "residual", "morph", "components")


def _odd(n: int) -> int:
    return int(max(5, n)) | 1


def run(img: np.ndarray, cfg: Cfg):
    """Returns (detections as [x,y,w,h] in input coordinates, per-stage seconds)."""
    cv2.setNumThreads(cfg.threads)
    T: dict[str, float] = {}
    t = time.perf_counter()
    g = img if cfg.scale == 1.0 else cv2.resize(img, None, fx=cfg.scale, fy=cfg.scale, interpolation=cv2.INTER_AREA)
    T["resize"] = time.perf_counter() - t

    t = time.perf_counter()
    k = _odd(round(cfg.ksize * cfg.scale))
    if cfg.bg == "median":
        bg = cv2.medianBlur(g, k)
    elif cfg.bg in ("median_ds2", "median_ds4"):
        f = 0.5 if cfg.bg == "median_ds2" else 0.25
        small = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        m = cv2.medianBlur(small, _odd(round(k * f)))
        bg = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    elif cfg.bg == "box":
        bg = cv2.blur(g, (k, k))
    else:
        raise ValueError(cfg.bg)
    T["background"] = time.perf_counter() - t

    t = time.perf_counter()
    resid = bg.astype(np.float32) - g.astype(np.float32)  # defects are darker than surroundings
    resid = cv2.GaussianBlur(resid, (0, 0), max(0.6, cfg.scale))
    med = float(np.median(resid))
    sigma = max(1.4826 * float(np.median(np.abs(resid - med))), 0.75)
    z = (resid - med) / sigma
    T["residual"] = time.perf_counter() - t

    t = time.perf_counter()
    mask = (z > 3.0).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(7 * cfg.scale) | 1,) * 2))
    T["morph"] = time.perf_counter() - t

    t = time.perf_counter()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    inv = 1.0 / cfg.scale
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 * cfg.scale * cfg.scale:
            continue
        if float(z[y:y + h, x:x + w].max()) < 4.5:
            continue
        dets.append([int(x * inv), int(y * inv), int(w * inv), int(h * inv)])
    T["components"] = time.perf_counter() - t
    return dets, T
