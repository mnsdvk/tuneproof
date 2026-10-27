"""Shared inspection service used by the web endpoint and the MCP server."""
from __future__ import annotations

import base64
import dataclasses
import platform
import time

import cv2
import numpy as np

from . import synth
from .agent import run_agent


def build_info() -> dict:
    info = cv2.getBuildInformation()
    flags = [k for k in ("KleidiCV", "NEON", "SVE", "AVX2", "OpenMP") if k.lower() in info.lower()]
    return {"opencv": cv2.__version__, "machine": platform.machine(), "build_flags_seen": flags}


def _b64png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _overlay(img, defects, review):
    vis = cv2.cvtColor(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    for d in defects:
        x, y, w, h = d["bbox"]
        cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (60, 200, 60), 2)
    for d in review:
        x, y, w, h = d["bbox"]
        cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 150, 255), 1)
    return vis


def inspect(img: np.ndarray, truth=None) -> dict:
    t0 = time.perf_counter()
    res = run_agent(img)
    infer_ms = round((time.perf_counter() - t0) * 1000, 1)
    out = {
        "inspect_ms": infer_ms,
        "defects": res.defects,
        "needs_review": res.needs_review,
        "trace": [dataclasses.asdict(s) for s in res.trace],
        "actions_kept": [s.evidence.get("action") for s in res.trace if s.tool == "keep_repair"],
        "actions_rolled_back": [s.evidence.get("action") for s in res.trace if s.tool == "rollback_repair"],
        "input_png": _b64png(img),
        "overlay_png": _b64png(_overlay(res.final_image, res.defects, res.needs_review)),
        "runtime": build_info(),
    }
    if truth is not None:
        out["ground_truth_boxes"] = [t["bbox"] for t in truth]
    return out


def inspect_bytes(data: bytes) -> dict:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    if max(img.shape) > 2048:  # keep latency bounded for large uploads
        s = 2048 / max(img.shape)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return inspect(img)


def inspect_sample(mode: str = "mixed", seed: int = 0) -> dict:
    ds = synth.make_dataset(50, 1000 + seed)
    pick = [d for d in ds if d["mode"] == mode] or ds
    d = pick[seed % len(pick)]
    return inspect(d["image"], d["truth"])
