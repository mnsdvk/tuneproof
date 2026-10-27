import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionpilot import mcp_server, service, synth, tools  # noqa: E402
from visionpilot.agent import run_agent, run_baseline  # noqa: E402


def _frame(mode):
    return [d for d in synth.make_dataset(30, 5) if d["mode"] == mode][0]


class AgentTests(unittest.TestCase):
    def test_deterministic(self):
        img = _frame("mixed")["image"]
        a, b = run_agent(img), run_agent(img)
        self.assertEqual([d["bbox"] for d in a.defects], [d["bbox"] for d in b.defects])

    def test_visual_evidence_changes_action(self):
        """Core Agentic Vision requirement: OpenCV output changes the next tool call."""
        dark = run_agent(_frame("dark")["image"])
        clean = run_agent(_frame("clean")["image"])
        acts_dark = [s.tool for s in dark.trace if s.kind == "act"]
        acts_clean = [s.tool for s in clean.trace if s.kind == "act"]
        self.assertIn("fix_exposure", acts_dark)
        self.assertNotIn("fix_exposure", acts_clean)

    def test_agent_surfaces_findings_baseline_misses_on_dark(self):
        # On under-exposed frames the fixed pipeline is blind; the agent repairs the
        # exposure and surfaces findings (confirmed or escalated to a human).
        d = _frame("dark")
        a = run_agent(d["image"])
        self.assertGreater(len(a.defects) + len(a.needs_review), len(run_baseline(d["image"]).defects))

    def test_bad_repair_is_rolled_back(self):
        # at least one frame across the set must trigger a rollback decision
        seen = any(s.tool == "rollback_repair" for d in synth.make_dataset(30, 5)
                   for s in run_agent(d["image"]).trace)
        self.assertTrue(seen)

    def test_trace_is_json_serialisable(self):
        out = service.inspect(_frame("blur")["image"])
        json.dumps({k: v for k, v in out.items() if k not in ("input_png", "overlay_png")})

    def test_mcp_roundtrip(self):
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertGreaterEqual(len(r["result"]["tools"]), 5)
        mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                           "params": {"name": "load_sample", "arguments": {"mode": "dark"}}})
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                               "params": {"name": "assess_quality", "arguments": {}}})
        self.assertLess(json.loads(r["result"]["content"][0]["text"])["mean"], 90)

    def test_tools_registry(self):
        self.assertEqual(set(tools.TOOLS), {"assess_quality", "detect_defects", "sharpen",
                                            "fix_exposure", "denoise", "zoom_tile"})


if __name__ == "__main__":
    unittest.main()
