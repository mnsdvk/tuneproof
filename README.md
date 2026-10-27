# TuneProof — every speedup is a claim. Prove it.

An agent that speeds up OpenCV 5 vision pipelines and **refuses any speedup that quietly breaks the output**. It profiles each stage, attacks the real bottleneck, verifies every change against a declared quality guard, and logs every accept/reject with its reason. (Python package name: `coolpilot`.)

```bash
pip install -r requirements.txt
python -m unittest discover -s tests                      # 13 tests
python -m coolpilot.run --images 6 --repeats 3 --grid     # agent vs exhaustive grid vs random search
```
Read `docs/COOLPILOT_REPORT.md` first (problem, method, measured AWS results, limits). Measured on AWS EC2 with stock OpenCV 5: 62.5 -> 43.9 ms (1.43x) in 5 experiments, no quality loss. **COOL/Graviton was not evaluated.** `deploy/run_aws.sh` lists the exact commands used.

The sections below describe VisionPilot, the earlier inspection-agent prototype whose detector is reused as CoolPilot's workload.

---

# VisionPilot — an agentic OpenCV 5 inspection loop for AWS Graviton

> What the camera sees decides what the system does next.

VisionPilot inspects a surface image, **measures its own image quality with OpenCV 5**, picks a repair, **verifies the repair against new visual evidence (and rolls it back if it made things worse)**, detects defects, **zooms into ambiguous ones**, and **asks a human** when evidence stays inconclusive. Every decision in the trace cites the OpenCV numbers that caused it.

Targets: OpenCV AI Competition 2026 — **Agentic Vision** path and **Best Use of COOL** path.

```
 image ─► assess_quality ─► PLAN (rules / any MCP agent) ─► repair tool ─► detect_defects
              ▲                                                │                │
              └──────── verify: keep or ROLL BACK ◄────────────┘                ▼
                                                       score ≥ hi ─► CONFIRM
                                                       lo ≤ score < hi ─► zoom_tile ─► re-detect
                                                                              ├─ strong ─► CONFIRM
                                                                              └─ weak ───► HUMAN APPROVAL queue
```

Runtime: stdlib web server + OpenCV 5 + NumPy only → one small arm64 container on Graviton, running **COOL** for the core workload.

## Quickstart
```bash
pip install -r requirements.txt
python -m unittest discover -s tests          # 7 tests
python -m visionpilot.evaluate --n 100 --seed 7   # reproducible agent-vs-baseline table
python -m visionpilot.server --port 8080      # open http://localhost:8080
python -m visionpilot.mcp_server              # MCP tools over stdio
python -m visionpilot.bench --label local     # COOL vs pip benchmark harness
```

## Layout
| Path | What |
|---|---|
| `visionpilot/tools.py` | OpenCV 5 tools: `assess_quality`, `detect_defects`, `fix_exposure`, `denoise`, `sharpen`, `zoom_tile` |
| `visionpilot/agent.py` | Perceive → decide → act → verify loop, triage, human escalation; fixed-pipeline baseline |
| `visionpilot/synth.py` | Ground-truth synthetic inspection data (clean / blur / dark / noise / mixed) |
| `visionpilot/evaluate.py` | Precision/recall/F1, agent vs baseline, failure cases |
| `visionpilot/server.py`, `static/` | Web endpoint + UI with trace viewer and approval queue |
| `visionpilot/mcp_server.py` | MCP (JSON-RPC/stdio) so any agent can drive the same tools |
| `visionpilot/bench.py`, `deploy/` | COOL vs baseline benchmark, Graviton run script, Dockerfile, deploy notes |
| `results/` | `eval.json` (held-out), `calib.json`, `sample_trace_dark.json`, `bench_local.json` (x86 sanity run) |
| `docs/` | `REPORT.md`, `PAST_WINNERS.md`, `VIDEO_SCRIPT.md`, `SUBMISSION_CHECKLIST.md` |

## Status — be honest with the judges
- Evaluated on **synthetic** data with exact ground truth (reproducible by anyone). Not validated on real industrial parts.
- COOL timings are **not** included: they must be produced on Graviton with `deploy/run_on_graviton.sh`.
