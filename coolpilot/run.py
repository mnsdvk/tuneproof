"""Run CoolPilot end to end and write an evidence file.

    python -m coolpilot.run --label cool-graviton5 --out results/coolpilot_cool.json
    # same command in a clean venv with stock pip OpenCV on the SAME instance type:
    python -m coolpilot.run --label pip-graviton5 --out results/coolpilot_pip.json
    python -m coolpilot.run --compare results/coolpilot_pip.json results/coolpilot_cool.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform

import cv2

from . import tuner
from .workload import BASELINE, DEFAULT_THREADS


def build_evidence() -> dict:
    info = cv2.getBuildInformation()
    keys = ("KleidiCV", "NEON", "SVE", "ARMPL", "AVX2", "OpenMP", "pthreads")
    return {"opencv": cv2.__version__, "file": getattr(cv2, "__file__", ""), "machine": platform.machine(),
            "python": platform.python_version(), "default_threads": DEFAULT_THREADS,
            "build_flags_mentioned": [k for k in keys if k.lower() in info.lower()]}


def execute(label, n_images, size, repeats, grid=False):
    corpus = tuner.make_corpus(n_images, size)
    bench = tuner.Bench(corpus, repeats)
    base, best, trace = tuner.tune(bench)
    n_exp = len(trace) - 1
    rnd = tuner.random_search(bench, base, budget=max(n_exp, 1))
    out = {"label": label, "env": build_evidence(), "corpus": {"images": n_images, "size": size, "repeats": repeats},
           "guard": tuner.GUARD, "baseline": {"cfg": base.cfg.label(), "ms": base.ms_per_image, "stage_ms": base.stage_ms,
                                              "gt_f1": base.gt_f1},
           "tuned": {"cfg": best.cfg.label(), "ms": best.ms_per_image, "stage_ms": best.stage_ms,
                     "ref_f1": best.ref_f1, "gt_f1": best.gt_f1},
           "speedup": round(base.ms_per_image / best.ms_per_image, 2),
           "agent_experiments": n_exp,
           "random_search_same_budget_speedup": {"mean": rnd[0], "min": rnd[1], "max": rnd[2]},
           "trace": trace}
    if grid:
        ms = [bench.measure(c) for c in tuner.all_configs()]
        g = tuner.best_valid(ms, base)
        out["exhaustive_grid"] = {"configs": len(ms), "best_cfg": g.cfg.label(),
                                  "speedup": round(base.ms_per_image / g.ms_per_image, 2)}
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"A = {a['label']} (OpenCV {a['env']['opencv']}, {a['env']['machine']})   "
          f"B = {b['label']} (OpenCV {b['env']['opencv']}, {b['env']['machine']})")
    print(f"{'':28}{'A ms':>9}{'B ms':>9}{'B vs A':>9}")
    rows = [("baseline config", a["baseline"]["ms"], b["baseline"]["ms"]),
            ("tuned config", a["tuned"]["ms"], b["tuned"]["ms"])]
    for s in a["baseline"]["stage_ms"]:
        rows.append((f"  baseline stage: {s}", a["baseline"]["stage_ms"][s], b["baseline"]["stage_ms"][s]))
    for name, x, y in rows:
        print(f"{name:28}{x:9.2f}{y:9.2f}{x / y:8.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="local")
    ap.add_argument("--out", default="results/coolpilot.json")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--grid", action="store_true", help="also run the exhaustive grid (slow)")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        r = execute(a.label, a.images, a.size, a.repeats, a.grid)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=2)
        print(f"[{r['label']}] OpenCV {r['env']['opencv']} on {r['env']['machine']}, threads={r['env']['default_threads']}")
        for t in r["trace"]:
            print(f"{t['n']:2} {t['decision']:20} {t['ms']:8.2f} ms  ref_f1={t['ref_f1']:.3f} gt_f1={t['gt_f1']:.3f}  {t['action']}")
        print(f"baseline {r['baseline']['ms']} ms -> tuned {r['tuned']['ms']} ms  ({r['speedup']}x)  [{r['tuned']['cfg']}]")
        print(f"experiments used: {r['agent_experiments']} | random search with same budget: {r['random_search_same_budget_speedup']}")
        if "exhaustive_grid" in r:
            print("exhaustive grid:", r["exhaustive_grid"])
        print("saved", a.out)
