"""OpenCV 5 vision tools exposed to the VisionPilot agent.

Every tool is a pure function: image in -> (image out | None, evidence dict).
The evidence dict is what the agent reasons over, so each number here can change
a later decision. Tools are also registered in TOOLS so they can be exposed over
MCP / an LLM tool-calling interface without changes.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# Perception tools
# --------------------------------------------------------------------------
def assess_quality(img: np.ndarray):
    """Blur, exposure and noise estimates for the whole frame.

    sharpness: variance of Laplacian on a mildly denoised copy (higher = sharper)
    mean / p5 / p95: exposure statistics
    noise_sigma: robust noise estimate from the finest wavelet-like residual
    """
    g = _gray(img)
    den = cv2.GaussianBlur(g, (3, 3), 0)
    sharp = float(cv2.Laplacian(den, cv2.CV_32F).var())
    p5, p95 = np.percentile(g, [5, 95])
    resid = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    ev = {
        "sharpness": round(sharp, 2),
        "mean": round(float(g.mean()), 1),
        "p5": float(p5),
        "p95": float(p95),
        "dynamic_range": float(p95 - p5),
        "noise_sigma": round(noise, 2),
    }
    return None, ev


def detect_defects(img: np.ndarray, sensitivity: float = 1.0, scale: float = 1.0):
    """Classical anomaly detector (background model + robust threshold).

    Builds a smooth background with a large Gaussian, takes the residual,
    normalises by MAD sigma, thresholds, cleans with morphology, and scores each
    connected component by peak z-score and area. Works at any `scale` so the
    agent can re-run it on an upscaled zoom tile.
    """
    g = _gray(img)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # defect-scale background: a median filter ignores blobs smaller than half its
    # window, and unlike a big Gaussian it does not let slow illumination drift
    # leak into the residual (which inflated z-scores on blurred / denoised frames).
    k = int(max(15, round(41 * scale)) | 1)
    bg = cv2.medianBlur(g, min(k, 255)).astype(np.float32)
    resid = bg - g.astype(np.float32)  # defects are darker than the surrounding texture
    resid = cv2.GaussianBlur(resid, (0, 0), 1.0 * scale)
    med = np.median(resid)
    sigma = max(1.4826 * np.median(np.abs(resid - med)), 0.75)  # floor: 8-bit quantisation
    z = (resid - med) / sigma
    thr = 3.0 / max(sensitivity, 0.1)
    mask = (z > thr).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=2)
    # merge fragments of one scratch: bridge gaps up to ~6 px before labelling
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(7 * scale) | 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    min_area = 12 * scale * scale
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        peak = float(z[lab == i].max())
        cands.append({
            "bbox": [int(x / scale), int(y / scale), int(w / scale), int(h / scale)],
            "area": float(area / (scale * scale)),
            "peak_z": round(peak, 2),
            "score": round(peak * np.log1p(area / (scale * scale)), 2),
        })
    cands.sort(key=lambda c: -c["score"])
    ev = {"n_candidates": len(cands), "candidates": cands, "scale": scale,
          "threshold_z": round(thr, 2)}
    return mask, ev


# --------------------------------------------------------------------------
# Action tools (change the image the agent looks at next)
# --------------------------------------------------------------------------
def sharpen(img: np.ndarray, amount: float = 1.5, sigma: float = 2.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    return out, {"op": "unsharp", "amount": amount, "sigma": sigma}


def fix_exposure(img: np.ndarray):
    """Gamma lift for under-exposed frames (maps the current mean to mid-grey).

    Plain gamma beat CLAHE in our ablation: CLAHE amplifies the fine texture the
    detector normalises against and erased most true defects (see docs/REPORT.md).
    """
    g = _gray(img)
    mean = float(g.mean())
    gamma = float(np.clip(np.log(0.5) / np.log(max(mean, 1) / 255.0), 0.25, 1.0)) if mean < 110 else 1.0
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(g, lut), {"op": "gamma", "gamma": round(gamma, 2)}


def denoise(img: np.ndarray, ksize: int = 5):
    """Median filter: removes impulsive/grain noise without flattening texture the
    way NL-means did (which collapsed the noise estimate and caused false alarms)."""
    return cv2.medianBlur(_gray(img), ksize), {"op": "median", "ksize": ksize}


def zoom_tile(img: np.ndarray, bbox, pad: int = 24, factor: float = 3.0):
    x, y, w, h = bbox
    H, W = _gray(img).shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    tile = _gray(img)[y0:y1, x0:x1]
    return tile, {"op": "zoom_tile", "origin": [x0, y0], "factor": factor,
                  "tile_shape": list(tile.shape)}


TOOLS = {
    "assess_quality": assess_quality,
    "detect_defects": detect_defects,
    "sharpen": sharpen,
    "fix_exposure": fix_exposure,
    "denoise": denoise,
    "zoom_tile": zoom_tile,
}
