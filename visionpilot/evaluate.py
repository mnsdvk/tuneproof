"""Reproducible evaluation: VisionPilot agent vs fixed pipeline on ground-truth data.

    python -m visionpilot.evaluate --n 100 --seed 7 --out results/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from . import synth
from .agent import run_agent, run_baseline


def _match(pred_boxes, truth, tol=10):
    used, tp = set(), 0
    for b in pred_boxes:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        for i, t in enumerate(truth):
            if i in used:
                continue
            x, y, w, h = t["bbox"]
            if x - tol <= cx <= x + w + tol and y - tol <= cy <= y + h + tol:
                used.add(i)
                tp += 1
                break
    fp = len(pred_boxes) - tp
    fn = len(truth) - tp
    return tp, fp, fn


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def evaluate(n=100, seed=7):
    data = synth.make_dataset(n, seed)
    agg = {"baseline": defaultdict(lambda: [0, 0, 0]), "agent": defaultdict(lambda: [0, 0, 0])}
    review_total = review_true = review_fp = 0
    step_counts, failures = [], []
    for d in data:
        truth = d["truth"]
        b = run_baseline(d["image"])
        a = run_agent(d["image"])
        for name, r in (("baseline", b), ("agent", a)):
            tp, fp, fn = _match([x["bbox"] for x in r.defects], truth)
            for key in (d["mode"], "ALL"):
                for j, v in enumerate((tp, fp, fn)):
                    agg[name][key][j] += v
            if name == "agent" and (fp or fn):
                failures.append({"id": d["id"], "mode": d["mode"], "tp": tp, "fp": fp, "fn": fn})
        # human-review queue quality: what fraction of flagged items are real defects?
        tp_r, fp_r, _ = _match([x["bbox"] for x in a.needs_review], truth)
        review_total += len(a.needs_review)
        review_true += tp_r
        review_fp += fp_r
        step_counts.append(len(a.trace))
    out = {"n_images": n, "seed": seed, "modes": {}}
    for key in sorted(set(agg["agent"]) | set(agg["baseline"])):
        out["modes"][key] = {nm: dict(zip(("precision", "recall", "f1"), _prf(*agg[nm][key])))
                             for nm in ("baseline", "agent")}
    out["human_review"] = {"flagged": review_total, "real_defects": review_true,
                           "false_alarms": review_fp}
    out["avg_trace_steps"] = round(sum(step_counts) / len(step_counts), 1)
    out["agent_failure_cases"] = failures[:15]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    res = evaluate(a.n, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"{'mode':8} {'base P/R/F1':>22} {'agent P/R/F1':>22}")
    for k, v in res["modes"].items():
        b, g = v["baseline"], v["agent"]
        print(f"{k:8} {b['precision']:.2f}/{b['recall']:.2f}/{b['f1']:.2f}".ljust(32),
              f"{g['precision']:.2f}/{g['recall']:.2f}/{g['f1']:.2f}")
    print("human review:", res["human_review"], "| avg trace steps:", res["avg_trace_steps"])
