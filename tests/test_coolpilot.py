import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coolpilot import tuner  # noqa: E402
from coolpilot.workload import BASELINE, Cfg, run  # noqa: E402


class CoolPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = tuner.Bench(tuner.make_corpus(3, 512), repeats=1)

    def test_baseline_agrees_with_itself(self):
        m = self.bench.measure(BASELINE)
        self.assertEqual(m.ref_f1, 1.0)

    def test_guard_rejects_destructive_setting(self):
        base = self.bench.measure(BASELINE)
        bad = self.bench.measure(Cfg(scale=0.25))
        self.assertFalse(bad.valid(base.gt_f1)[0])

    def test_tuned_config_always_satisfies_guard(self):
        base, best, trace = tuner.tune(self.bench, max_experiments=12)
        self.assertTrue(best.valid(base.gt_f1)[0])
        self.assertLessEqual(best.ms_per_image, base.ms_per_image)

    def test_measurement_changes_next_action(self):
        """Core Agentic Vision requirement for a developer agent: a different profile yields a different plan."""
        base = self.bench.measure(BASELINE)
        fake = type(base)(base.cfg, 10.0, {"resize": 0.1, "background": 9.6, "residual": 0.1, "morph": 0.1, "components": 0.1},
                          1.0, base.gt_f1)
        # real baseline profile is residual-dominated on this corpus -> 'scale'; background-dominated -> 'bg'
        prof = dict(base.stage_ms); prof_top = max(prof, key=prof.get)
        a = tuner.choose_next(base, set())
        b = tuner.choose_next(fake, set())
        self.assertEqual(b[0], 'bg')
        if prof_top != 'background':
            self.assertNotEqual(a[0], b[0])

    def test_every_decision_is_logged_with_reason(self):
        _, _, trace = tuner.tune(self.bench, max_experiments=6)
        self.assertTrue(all(t["reason"] and t["decision"] for t in trace))

    def test_detections_in_input_coordinates(self):
        d = tuner.make_corpus(1, 512)[0]
        dets, _ = run(d["image"], BASELINE)
        for x, y, w, h in dets:
            self.assertTrue(0 <= x < 512 and 0 <= y < 512)


if __name__ == "__main__":
    unittest.main()
