import unittest

from scripts.audit_v273_semitone_disappearance_fold import trajectory_summary
from scripts.summarize_v273_semitone_disappearance import select_rule, accounting


class SemitoneDisappearanceTests(unittest.TestCase):
    def test_trajectory_detects_first_k2_and_reentry(self):
        out = trajectory_summary([3, 3, 2, 2, 3, 2])
        self.assertEqual(out["first_non3_step"], 2)
        self.assertEqual(out["first_below3_step"], 2)
        self.assertEqual(out["first_k2_step"], 2)
        self.assertTrue(out["reentered_k3"])
        self.assertFalse(out["monotone_nonincreasing"])

    def test_censored_if_no_k2(self):
        out = trajectory_summary([3] * 25)
        self.assertEqual(out["first_k2_step"], 25)
        self.assertEqual(out["first_non3_step"], 25)
        self.assertFalse(out["reentered_k3"])
        self.assertTrue(out["monotone_nonincreasing"])

    def test_fit_rule_can_select_early_disappearance(self):
        rows = []
        for value in (2, 3, 4, 5):
            rows.append({"true_k": 2, "first_k2_step": value})
        for value in (15, 18, 22, 25, 25):
            rows.append({"true_k": 3, "first_k2_step": value})
        selected, _ = select_rule(rows)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["direction"], "le")
        self.assertGreater(selected["global_net"], 0)

    def test_fit_rule_abstains_when_no_positive_net(self):
        rows = [
            {"true_k": 2, "first_k2_step": 5},
            {"true_k": 3, "first_k2_step": 5},
            {"true_k": 3, "first_k2_step": 5},
        ]
        selected, _ = select_rule(rows)
        self.assertIsNone(selected)

    def test_accounting(self):
        rows = [
            {"true_k": 2, "first_k2_step": 2},
            {"true_k": 3, "first_k2_step": 10},
            {"true_k": 2, "first_k2_step": 20},
        ]
        q = accounting(rows, "le", 5)
        self.assertEqual(q["corrections"], 1)
        self.assertEqual(q["regressions"], 0)
        self.assertEqual(q["global_net"], 1)


if __name__ == "__main__":
    unittest.main()
