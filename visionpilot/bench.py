"""COOL-vs-baseline benchmark harness. Run the same file on both builds.

    # On a Graviton instance with the COOL AMI/container:
    python -m visionpilot.bench --label cool-graviton4 --out results/bench_cool.json
    # On the same instance type with a stock pip OpenCV in a clean venv:
    python -m visionpilot.bench --label pip-graviton4 --out results/bench_pip.json
    # Compare:
    python -m visionpilot.bench --compare results/bench_pip.json results/bench_cool.json

Reports median ms over N runs, plus the end-to-end VisionPilot pipeline, so the
claim is about the *claimed core workload*, not a micro-benchmark alone.
No numbers are shipped in this repo: they must be produced on the target hardware.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import cv2
import numpy as np

from . import synth
from .agent import run_agent


def _time(fn, runs):
    fn()  # warm-up
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(ts), 3), "p95_ms": round(sorted(ts)[int(0.95 * (len(ts) - 1))], 3)}


def run(label: str, runs: int = 30, size: int = 1536):
    rng = np.random.default_rng(0)
    big = (rng.random((size, size)) * 255).astype(np.uint8)
    big3 = cv2.merge([big, big[::-1], big[:, ::-1]])
    M = cv2.getRotationMatrix2D((size / 2, size / 2), 7, 1.0)
    frame = synth.make_dataset(5, 3)[3]["image"]
    ops = {
        "GaussianBlur_5x5": lambda: cv2.GaussianBlur(big, (5, 5), 0),
        "medianBlur_5": lambda: cv2.medianBlur(big, 5),
        "medianBlur_41": lambda: cv2.medianBlur(big, 41),
        "resize_0.5_linear": lambda: cv2.resize(big3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR),
        "cvtColor_BGR2GRAY": lambda: cv2.cvtColor(big3, cv2.COLOR_BGR2GRAY),
        "warpAffine": lambda: cv2.warpAffine(big3, M, (size, size)),
        "Canny": lambda: cv2.Canny(big, 50, 150),
        "morph_close_7x7": lambda: cv2.morphologyEx(big, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)),
        "VisionPilot_pipeline(512px)": lambda: run_agent(frame),
    }
    out = {"label": label, "opencv": cv2.__version__, "machine": platform.machine(),
           "python": platform.python_version(), "threads": cv2.getNumThreads(), "runs": runs,
           "image": f"{size}x{size}", "ops": {k: _time(f, runs) for k, f in ops.items()}}
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"baseline: {a['label']} (OpenCV {a['opencv']})   candidate: {b['label']} (OpenCV {b['opencv']})")
    print(f"{'op':32}{'base ms':>10}{'cand ms':>10}{'speedup':>9}")
    for k in a["ops"]:
        x, y = a["ops"][k]["median_ms"], b["ops"][k]["median_ms"]
        print(f"{k:32}{x:10.2f}{y:10.2f}{x / y:8.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="local")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="results/bench.json")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        import os
        r = run(a.label, a.runs)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=2)
        for k, v in r["ops"].items():
            print(f"{k:32}{v['median_ms']:10.2f} ms")
        print(f"saved {a.out}  [{r['machine']}, OpenCV {r['opencv']}]")
