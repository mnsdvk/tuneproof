# Devpost submission text (fill [BRACKETS])

**Project name:** TuneProof
**Tagline:** Every speedup is a claim. Prove it.
**Path to select:** Agentic Vision (do NOT select Best Use of COOL: COOL was not used)

## Inspiration
To cut cloud vision costs, teams shrink images, swap algorithms or reduce kernels, then ship. Nobody checks what silently broke, and the first casualties are the small, faint features the pipeline exists to find. We wanted an agent that makes vision pipelines faster and is not allowed to make them wrong.

## What it does
TuneProof profiles an OpenCV 5 pipeline stage by stage, picks the next experiment from that evidence (attack the dominant stage), and accepts a change only if the output stays within a declared quality guard (agreement with the baseline >= 0.95 F1, ground-truth F1 drop <= 0.03) and is measurably faster. Rejected experiments are logged with reasons. It outputs the best valid configuration, a full evidence trace and a reproducible command.

## How we built it
Python, OpenCV 5.0 and NumPy only, with pinned dependencies and 13 unit tests. The workload is a real anomaly-detection pipeline (large-window median background, residual z-score, morphology, connected components). The tuner is a measure -> choose -> verify -> accept/reject loop; `choose_next` is the hook where an LLM planner (e.g. Amazon Bedrock) can be substituted. We ran the full install, tests and benchmark on AWS EC2 (t3.micro, Ubuntu 24.04).

## Results (reproducible: `python -m coolpilot.run --images 6 --repeats 3 --grid`)
On AWS EC2 (x86_64, 2 vCPU, stock OpenCV 5.0.0): baseline 62.5 ms/image -> 43.9 ms (1.43x) in 5 experiments, with no loss of detection quality (agreement 0.974; ground-truth F1 unchanged at 0.676). The exhaustive 200-configuration grid found the same best configuration; random search with the same 5-experiment budget averaged only 1.07x. The guard rejected three tempting speedups (scale 0.75, a 4x-downsampled background, a smaller window) that would have silently changed which defects are found.

## Challenges
Calibrating a guard that blocks real damage without blocking everything; an early image-repair prototype made dark images darker because of an inverted gamma exponent, which we found by tracing per-action effects; getting honest timings (warm-up, medians) and reporting one accepted step that turned out to be noise-level.

## Limitations (stated up front)
Synthetic data; the baseline detector is modest (F1 0.68 against known defects); the guard measures fidelity to a baseline, so it is conservative; the planner is rule-based, not an LLM; single workload; the residual stage (64% of runtime after tuning) was not reduced; COOL and Graviton were not evaluated.

## What's next
LLM planner over the same tools via MCP, real image corpora, COOL/Graviton comparison, multi-workload guards.

## Built with
OpenCV 5, Python, NumPy, AWS EC2

---

# 5-minute video script
**0:00-0:30 Hook + team.** "Every speedup is a claim. Here is what happens when nobody checks it." Show the three rejected speedups from the trace.
**0:30-1:15 Problem.** Cheaper vision pipelines silently lose small features.
**1:15-2:45 Live run.** Run `python -m coolpilot.run --images 6 --repeats 3` (or play the recording). Narrate: profile -> "background is 41% of runtime" -> try median_ds2 -> quality OK, faster -> ACCEPT -> next bottleneck -> REJECT with reason.
**2:45-3:30 Why it is agentic.** Diagram: measure -> choose -> verify -> accept/reject; the measured profile changes the next tool call; the guard is the human-set limit on autonomy.
**3:30-4:20 AWS + evidence.** Show the EC2 console (instance, region), the terminal output, `results/coolpilot_aws.json`, and the results table.
**4:20-5:00 Limits + close.** Be explicit: synthetic data, rule-based planner, one noise-level step, no COOL. Repo link.
