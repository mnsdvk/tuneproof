"""VisionPilot agent: a perception -> decision -> action loop over OpenCV 5 tools.

The policy is deliberately inspectable: every decision records the OpenCV
evidence that triggered it, so the trace *proves* that visual output changed
what the system did next (the Agentic Vision criterion).

`policy="rules"` is a deterministic planner (default, reproducible, no API key).
`policy="llm"` delegates the *choice of next tool* to an LLM (e.g. Amazon Bedrock)
given the same evidence JSON; see `llm_policy.py`. Tools are identical.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import tools

CFG = {
    "min_sharpness": 25.0,
    "min_mean": 90.0,
    "max_noise": 4.5,
    "hi": 22.0,   # score >= hi  -> confirmed defect
    "lo": 12.0,   # lo <= score < hi -> ambiguous -> zoom / human review
    "zoom_confirm": 18.0,  # zoomed-tile score (peak_z * log area) needed to confirm; calibrated on seed 101
    "max_steps": 8,
    "explode_factor": 3.0,  # candidate count multiplying by this => distrust the repair
}


@dataclass
class Step:
    n: int
    kind: str          # perceive | decide | act
    tool: str
    evidence: dict
    reason: str = ""
    ms: float = 0.0


@dataclass
class Result:
    defects: list = field(default_factory=list)       # confirmed
    needs_review: list = field(default_factory=list)  # human approval requested
    trace: list = field(default_factory=list)
    final_image: np.ndarray | None = None


def _log(res, kind, tool, ev, reason="", t0=None):
    res.trace.append(Step(len(res.trace) + 1, kind, tool, ev, reason,
                          round((time.perf_counter() - t0) * 1000, 2) if t0 else 0.0))


def _fix_plan(q, cfg, tried):
    """Rule policy: pick the next corrective action from quality evidence."""
    if q["mean"] < cfg["min_mean"] and "fix_exposure" not in tried:
        return "fix_exposure", f"mean {q['mean']} < {cfg['min_mean']} (under-exposed)"
    if q["noise_sigma"] > cfg["max_noise"] and "denoise" not in tried:
        return "denoise", f"noise_sigma {q['noise_sigma']} > {cfg['max_noise']}"
    if q["sharpness"] < cfg["min_sharpness"] and "sharpen" not in tried:
        return "sharpen", f"sharpness {q['sharpness']} < {cfg['min_sharpness']} (blurry)"
    return None, "quality acceptable or all repairs tried"


def _confident(ev, cfg):
    return sum(1 for c in ev["candidates"] if c["score"] >= cfg["hi"])


def _verify(before, after, cfg):
    """Closed-loop check on the *visual evidence* (no ground truth needed):
    a repair is kept only if it does not lose confident detections and does not
    make the candidate count explode (a sign it manufactured structure)."""
    cb, ca = _confident(before, cfg), _confident(after, cfg)
    nb, na = before["n_candidates"], after["n_candidates"]
    if ca < cb:
        return False, f"confident detections fell {cb}->{ca}"
    if na > max(cfg["explode_factor"] * max(nb, 1), 12):
        return False, f"candidate count exploded {nb}->{na}"
    return True, f"confident {cb}->{ca}, candidates {nb}->{na}"


def run_agent(img: np.ndarray, cfg: dict | None = None, planner=None) -> Result:
    cfg = {**CFG, **(cfg or {})}
    res = Result()
    cur = img if img.ndim == 2 else tools._gray(img)
    tried: set[str] = set()

    # ---- 1. perceive -> decide -> act -> verify repair loop --------------
    _, base_ev = tools.detect_defects(cur)
    for _ in range(cfg["max_steps"]):
        t0 = time.perf_counter()
        _, q = tools.assess_quality(cur)
        _log(res, "perceive", "assess_quality", q, t0=t0)
        action, why = (planner or _fix_plan)(q, cfg, tried)
        if action is None:
            _log(res, "decide", "stop_repair", {}, why)
            break
        tried.add(action)
        _log(res, "decide", action, {}, why)
        t0 = time.perf_counter()
        cand, ev = tools.TOOLS[action](cur)
        _, new_ev = tools.detect_defects(cand)
        _log(res, "act", action, ev, t0=t0)
        ok, vwhy = _verify(base_ev, new_ev, cfg)
        if ok:
            cur, base_ev = cand, new_ev
            _log(res, "decide", "keep_repair", {"action": action}, vwhy)
        else:
            _log(res, "decide", "rollback_repair", {"action": action}, vwhy)

    # ---- 2. detect ------------------------------------------------------
    t0 = time.perf_counter()
    ev = base_ev
    _log(res, "perceive", "detect_defects", {k: v for k, v in ev.items() if k != "candidates"}
         | {"top": ev["candidates"][:5]}, t0=t0)

    # ---- 3. triage each candidate by its own evidence --------------------
    for c in ev["candidates"]:
        if c["score"] >= cfg["hi"]:
            res.defects.append({**c, "decision": "confirmed", "via": "detector"})
            _log(res, "decide", "confirm", {"bbox": c["bbox"], "score": c["score"]},
                 f"score {c['score']} >= {cfg['hi']}")
        elif c["score"] >= cfg["lo"]:
            _log(res, "decide", "zoom_tile", {"bbox": c["bbox"], "score": c["score"]},
                 f"ambiguous score {c['score']} in [{cfg['lo']},{cfg['hi']}) -> look closer")
            t0 = time.perf_counter()
            tile, tev = tools.zoom_tile(cur, c["bbox"])
            _log(res, "act", "zoom_tile", tev, t0=t0)
            t0 = time.perf_counter()
            _, zev = tools.detect_defects(tile, scale=3.0)
            zpeak = max([z["score"] for z in zev["candidates"]], default=0.0)
            _log(res, "perceive", "detect_defects(zoom)", {"zoom_score": zpeak,
                 "n_candidates": zev["n_candidates"]}, t0=t0)
            if zpeak >= cfg["zoom_confirm"]:
                res.defects.append({**c, "decision": "confirmed", "via": "zoom", "zoom_score": zpeak})
                _log(res, "decide", "confirm", {"bbox": c["bbox"]},
                     f"zoom score {zpeak} >= {cfg['zoom_confirm']}")
            else:
                res.needs_review.append({**c, "decision": "human_review", "zoom_score": zpeak})
                _log(res, "decide", "request_human_approval", {"bbox": c["bbox"]},
                     f"zoom score {zpeak} < {cfg['zoom_confirm']}: evidence inconclusive")
    res.final_image = cur
    return res


def run_baseline(img: np.ndarray, cfg: dict | None = None) -> Result:
    """Fixed pipeline: one detection pass on the raw frame, no decisions."""
    cfg = {**CFG, **(cfg or {})}
    res = Result()
    _, ev = tools.detect_defects(img)
    for c in ev["candidates"]:
        if c["score"] >= cfg["hi"]:
            res.defects.append({**c, "decision": "confirmed", "via": "fixed"})
    return res
