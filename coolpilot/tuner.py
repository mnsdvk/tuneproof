"""CoolPilot agent: measure -> pick the next experiment from the evidence -> verify quality -> accept/reject.

Decision rule (inspectable, no LLM needed; a Bedrock planner can replace `choose_next`):
  1. Profile the current best config on the corpus -> stage time shares.
  2. Attack the dominant stage: pick an un-exhausted knob that affects it, step one rung.
  3. Measure. Accept ONLY if quality guard passes AND it is measurably faster.
  4. Otherwise reject, mark that knob exhausted, and re-plan from the (unchanged) best config.
After each acceptance the profile changes, so the next target changes: the measurement drives the plan.
"""
from __future__ import annotations

import itertools
import random
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from visionpilot import synth
from visionpilot.evaluate import _match, _prf

from .workload import BASELINE, DEFAULT_THREADS, STAGES, Cfg, run

LADDERS = {
    "scale": [1.0, 0.75, 0.5, 0.35, 0.25],
    "bg": ["median", "median_ds2", "median_ds4", "box"],
    "ksize": [41, 31, 21, 15, 9],
    "threads": [DEFAULT_THREADS] + [t for t in (4, 2, 1) if t < DEFAULT_THREADS],
}
STAGE_KNOBS = {
    "background": ["bg", "ksize"],
    "resize": ["scale"],
    "residual": ["scale", "threads"],
    "morph": ["scale"],
    "components": ["scale"],
}
# Quality guard (declared up front): the tuned output must agree with the baseline output.
GUARD = {"min_ref_f1": 0.95, "max_gt_f1_drop": 0.03}


def make_corpus(n=8, size=1024, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        surf = synth.make_surface(rng, size)
        img, truth = synth.add_defects(surf, rng, int(rng.integers(4, 7)))
        out.append({"id": i, "image": img, "truth": truth})
    return out


@dataclass
class Measure:
    cfg: Cfg
    ms_per_image: float
    stage_ms: dict
    ref_f1: float
    gt_f1: float

    def valid(self, base_gt_f1: float) -> tuple[bool, str]:
        if self.ref_f1 < GUARD["min_ref_f1"]:
            return False, f"agreement with baseline output {self.ref_f1:.3f} < {GUARD['min_ref_f1']}"
        if base_gt_f1 - self.gt_f1 > GUARD["max_gt_f1_drop"]:
            return False, f"ground-truth F1 fell {base_gt_f1:.3f}->{self.gt_f1:.3f} (> {GUARD['max_gt_f1_drop']})"
        return True, "within guard"


class Bench:
    def __init__(self, corpus, repeats=3):
        self.corpus, self.repeats, self.n_measured = corpus, repeats, 0
        self.ref = {d["id"]: run(d["image"], BASELINE)[0] for d in corpus}

    def measure(self, cfg: Cfg) -> Measure:
        self.n_measured += 1
        per_img, stage = [], {s: [] for s in STAGES}
        tp = fp = fn = rtp = rfp = rfn = 0
        for d in self.corpus:
            run(d["image"], cfg)  # warm-up
            tot = []
            st = {s: [] for s in STAGES}
            for _ in range(self.repeats):
                t0 = time.perf_counter()
                dets, T = run(d["image"], cfg)
                tot.append(time.perf_counter() - t0)
                for s in STAGES:
                    st[s].append(T[s])
            per_img.append(statistics.median(tot) * 1000)
            for s in STAGES:
                stage[s].append(statistics.median(st[s]) * 1000)
            a, b, c = _match(dets, d["truth"])
            tp, fp, fn = tp + a, fp + b, fn + c
            ref = [{"bbox": r} for r in self.ref[d["id"]]]
            a, b, c = _match(dets, ref)
            rtp, rfp, rfn = rtp + a, rfp + b, rfn + c
        return Measure(cfg, round(statistics.mean(per_img), 2),
                       {s: round(statistics.mean(v), 2) for s, v in stage.items()},
                       _prf(rtp, rfp, rfn)[2], _prf(tp, fp, fn)[2])


def _next_rung(cfg: Cfg, knob: str):
    lad = LADDERS[knob]
    cur = getattr(cfg, knob)
    i = lad.index(cur) if cur in lad else 0
    return lad[i + 1] if i + 1 < len(lad) else None


def choose_next(best: Measure, exhausted: set):
    """Pick (knob, value, reason) from evidence, or None."""
    total = sum(best.stage_ms.values())
    for stage, ms in sorted(best.stage_ms.items(), key=lambda kv: -kv[1]):
        for knob in STAGE_KNOBS[stage]:
            if knob in exhausted:
                continue
            nxt = _next_rung(best.cfg, knob)
            if nxt is not None:
                return knob, nxt, f"'{stage}' is {100 * ms / total:.0f}% of runtime ({ms:.1f} ms) -> step {knob} to {nxt}"
    return None


def tune(bench: Bench, max_experiments=40):
    base = bench.measure(BASELINE)
    best, exhausted = base, set()
    trace = [{"n": 0, "action": "baseline", "cfg": asdict(BASELINE), "ms": base.ms_per_image,
              "stage_ms": base.stage_ms, "ref_f1": base.ref_f1, "gt_f1": base.gt_f1,
              "decision": "start", "reason": "profile baseline"}]
    for n in range(1, max_experiments + 1):
        pick = choose_next(best, exhausted)
        if pick is None:
            break
        knob, val, why = pick
        cand = bench.measure(best.cfg.with_(**{knob: val}))
        ok, vwhy = cand.valid(base.gt_f1)
        faster = cand.ms_per_image < best.ms_per_image * 0.98
        if ok and faster:
            decision, reason = "ACCEPT", f"{why}; {best.ms_per_image:.1f}->{cand.ms_per_image:.1f} ms, {vwhy}"
            best = cand
        elif not ok:
            decision, reason = "REJECT(quality)", f"{why}; {vwhy}"
            exhausted.add(knob)
        else:
            decision, reason = "REJECT(not faster)", f"{why}; {cand.ms_per_image:.1f} ms vs {best.ms_per_image:.1f} ms"
            exhausted.add(knob)
        trace.append({"n": n, "action": f"{knob}={val}", "cfg": asdict(cand.cfg), "ms": cand.ms_per_image,
                      "stage_ms": cand.stage_ms, "ref_f1": cand.ref_f1, "gt_f1": cand.gt_f1,
                      "decision": decision, "reason": reason})
    return base, best, trace


def all_configs():
    return [Cfg(s, b, k, t) for s, b, k, t in itertools.product(
        LADDERS["scale"], LADDERS["bg"], LADDERS["ksize"], LADDERS["threads"])]


def best_valid(measures, base):
    ok = [m for m in measures if m.valid(base.gt_f1)[0]]
    return min(ok, key=lambda m: m.ms_per_image) if ok else base


def random_search(bench: Bench, base: Measure, budget: int, seeds=5):
    cfgs = all_configs()
    res = []
    for s in range(seeds):
        rng = random.Random(s)
        picks = rng.sample(cfgs, budget)
        res.append(base.ms_per_image / best_valid([bench.measure(c) for c in picks], base).ms_per_image)
    return round(statistics.mean(res), 2), round(min(res), 2), round(max(res), 2)
