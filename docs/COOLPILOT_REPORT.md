# TuneProof — every speedup is a claim. Prove it.

*A measurement-driven agent that speeds up OpenCV 5 vision pipelines and refuses any speedup that quietly breaks the output. (The Python package is still named `coolpilot`.)*

## 1. The problem
To cut cloud vision costs, teams shrink images, swap algorithms or reduce kernels, then ship. Nobody re-checks what broke, and the first things lost are the small, faint features the pipeline exists to find. Performance tuning is done blind: faster is the only goal and "still correct" is never declared.

## 2. What it does
1. **Profile** the real pipeline per stage (resize, background model, residual/z-score, morphology, components).
2. **Choose the next experiment from that evidence**: attack the dominant stage with a knob that affects it.
3. **Verify against a declared quality guard**: agreement with the baseline output F1 >= 0.95, and ground-truth F1 drop <= 0.03.
4. **Accept only if valid and measurably faster** (>= 2% gain); otherwise reject, mark the knob exhausted, re-plan from the unchanged best config.
5. **Emit an evidence trace**: every experiment, timing, quality score and reason (`results/coolpilot_aws.json`).

**Agentic Vision criteria.** OpenCV 5 measurements and output quality change the next tool call and configuration, ending in a running image workload (substantive OpenCV + agent integration); each decision is logged with its evidence (observability); failed experiments are rejected and logged (failure handling); the guard is a human-declared limit on autonomy (human control). *Honest note:* the shipped planner is rule-based, not an LLM; `choose_next` is the hook for an LLM planner. There is no interactive human-approval step.

## 3. OpenCV 5 implementation
Workload: large-window `medianBlur` background model, `resize` (INTER_AREA/LINEAR), `GaussianBlur`, `morphologyEx`, `connectedComponentsWithStats`, `setNumThreads`. Pinned dependencies: opencv-python-headless 5.0.0.93, NumPy 2.5.3. 13 unit tests (including: a different stage profile yields a different next action; every decision is logged with a reason; the guard rejects a destructive setting).

## 4. Results — measured on AWS EC2 (Ubuntu 24.04, x86_64, 2 vCPU, Python 3.12.3, stock pip OpenCV 5.0.0)
Corpus: 6 synthetic 1024x1024 textured surfaces, 4-6 scratches/dents each (30% low-contrast), ground truth known; 3 repeats, medians. Reproduce: `python -m coolpilot.run --images 6 --repeats 3 --grid`.

| | value |
|---|---|
| Baseline | 62.5 ms/image (`bg=median k=41`, full resolution) |
| Tuned by the agent | **43.9 ms/image, 1.43x**, `bg=median_ds2` |
| Quality of tuned config | agreement with baseline F1 0.974; ground-truth F1 0.676 -> 0.676 (no loss) |
| Experiments used | 5 |
| Exhaustive grid (200 configs) | best valid config is the same; 1.41x in its own measurement |
| Random search, same 5-experiment budget | mean 1.07x (range 1.00-1.36x), 5 seeds |

Per-stage (baseline -> tuned, ms): background 25.0 -> 7.4; residual 28.5 -> 28.0; morph 3.2 -> 2.2; components 5.9 -> 6.2.

**What the guard blocked:** `scale=0.75` (agreement 0.845), `bg=median_ds4` (0.944), `ksize=31` (0.949). Each would have been faster (up to ~40% less time) while silently changing which defects are found.

A separate run in a 2-vCPU x86 sandbox gave 55.6 -> 36.3 ms (1.53x) with the same chosen background setting (`results/coolpilot_local.json`).

## 5. Limitations (observed, not hypothetical)
- **One accepted step is noise-level:** `threads=1` was accepted for a 2.3% gain (62.5 -> 61.1 ms), barely above the 2% acceptance margin; the sandbox run rejected the same change as not faster. The real gain is `bg=median_ds2` (background 25.0 -> 7.4 ms). The final 1.43x vs the grid's 1.41x is the same configuration measured twice, i.e. measurement noise.
- **Residual stage is now ~64% of runtime** and no knob in the search space reduces it without failing the guard.
- Ground-truth F1 of the *baseline* detector is 0.68: this is a tuning study of a modest detector on synthetic data, not a state-of-the-art inspection result.
- The guard measures fidelity to a baseline, so it is conservative (e.g. `scale=0.75` raised ground-truth F1 to 0.73 yet was rejected).
- Single workload, single corpus, small burstable instance (t3.micro, 2 vCPU, 1 GiB): absolute timings are noisy and will differ elsewhere.
- **COOL was not used or evaluated.** No claim is made about COOL or Graviton performance. The scripts (`deploy/run_coolpilot.sh`, `--compare`) can compare stock OpenCV with COOL on a Graviton instance; that was not run.

## 6. AWS delivery
The full pipeline (install from pinned `requirements.txt`, 13 tests, tuner run with exhaustive-grid comparison) was executed on an AWS EC2 instance t3.micro; commands in `deploy/run_aws.sh`. Results JSON is the evidence file. No credentials in code; SSH restricted to the operator's IP; instance terminated after the run.

## 7. Responsible use
Synthetic data; results are a tuning study, not a production accuracy claim. The quality guard is a fidelity check against a chosen baseline and does not prove semantic correctness for other workloads: declare a new guard for each real pipeline.
