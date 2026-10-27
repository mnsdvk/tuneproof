"""Synthetic inspection dataset with exact ground truth.

Why synthetic: judges need a reproducible evaluation that anyone can re-run with
one command, and we need ground-truth defect locations to compute precision /
recall honestly. Real-image support is provided through the web endpoint.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_surface(rng: np.random.Generator, size: int = 512) -> np.ndarray:
    base = rng.normal(0, 1, (size, size)).astype(np.float32)
    fine = cv2.GaussianBlur(base, (0, 0), 1.2)
    coarse = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 25) * 6
    img = 128 + 14 * fine / (fine.std() + 1e-6) + coarse * 4
    return np.clip(img, 0, 255).astype(np.float32)


def add_defects(img: np.ndarray, rng: np.random.Generator, n: int, hard_frac: float = 0.3):
    """Dark scratches and dents. hard_frac of them are low-contrast (ambiguous)."""
    size = img.shape[0]
    out = img.copy()
    truth = []
    for _ in range(n):
        hard = rng.random() < hard_frac
        depth = rng.uniform(22, 34) if hard else rng.uniform(45, 70)
        layer = np.zeros_like(out)
        if rng.random() < 0.5:  # scratch
            x0, y0 = rng.integers(40, size - 40, 2)
            ang = rng.uniform(0, np.pi)
            ln = int(rng.integers(30, 60))
            x1, y1 = int(x0 + ln * np.cos(ang)), int(y0 + ln * np.sin(ang))
            cv2.line(layer, (int(x0), int(y0)), (x1, y1), 1.0, 2)
            kind = "scratch"
        else:  # dent
            x0, y0 = rng.integers(40, size - 40, 2)
            r = int(rng.integers(5, 9))
            cv2.circle(layer, (int(x0), int(y0)), r, 1.0, -1)
            kind = "dent"
        layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
        layer /= layer.max() + 1e-6
        out -= depth * layer
        ys, xs = np.where(layer > 0.3)
        truth.append({"bbox": [int(xs.min()), int(ys.min()),
                               int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                      "kind": kind, "hard": bool(hard)})
    return np.clip(out, 0, 255).astype(np.uint8), truth


def degrade(img: np.ndarray, rng: np.random.Generator, mode: str):
    out = img.astype(np.float32)
    if mode in ("blur", "mixed"):
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(1.8, 2.8))
    if mode in ("dark", "mixed"):
        out = 255 * (out / 255.0) ** rng.uniform(2.2, 3.0) * rng.uniform(0.75, 1.0)
    if mode in ("noise", "mixed"):
        out = out + rng.normal(0, rng.uniform(6, 11), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_dataset(n: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    modes = ["clean", "blur", "dark", "noise", "mixed"]
    data = []
    for i in range(n):
        surf = make_surface(rng)
        img, truth = add_defects(surf, rng, int(rng.integers(2, 5)))
        mode = modes[i % len(modes)]
        if mode != "clean":
            img = degrade(img, rng, mode)
        data.append({"id": i, "mode": mode, "image": img, "truth": truth})
    return data
