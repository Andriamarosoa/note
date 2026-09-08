import unittest

import numpy as np

from scripts import audit_v270_class_conditional_fusion as v


class FusionRuleTests(unittest.TestCase):
    def test_null_veto_only_changes_one_to_zero(self):
        anchor = np.arange(7, dtype=np.int32)
        specialist = np.zeros(7, dtype=np.int32)
        actual = v.fuse_counts(anchor, specialist, "null_veto")
        np.testing.assert_array_equal(actual, np.array([0, 0, 2, 3, 4, 5, 6]))
        changed = actual != anchor
        self.assertTrue(np.all(anchor[changed] == 1))
        self.assertTrue(np.all(specialist[changed] == 0))

    def test_low_k_fusion_only_changes_low_anchor(self):
        anchor = np.arange(7, dtype=np.int32)
        specialist = np.array([2, 0, 6, 6, 6, 6, 6], dtype=np.int32)
        actual = v.fuse_counts(anchor, specialist, "low_k_fusion")
        np.testing.assert_array_equal(actual, np.array([2, 0, 2, 3, 4, 5, 6]))
        self.assertTrue(np.all(actual[anchor >= 2] == anchor[anchor >= 2]))

    def test_both_rules_preserve_every_correct_poly_prediction(self):
        truth = np.tile(np.arange(7, dtype=np.int32), 20)
        anchor = np.roll(truth, 3)
        anchor[truth >= 2] = truth[truth >= 2]
        specialist = np.arange(len(truth), dtype=np.int32) % 7
        before = (truth >= 2) & (anchor == truth)
        for arm in ("null_veto", "low_k_fusion"):
            after = v.fuse_counts(anchor, specialist, arm)
            self.assertTrue(np.all(after[before] == truth[before]))

    def test_invalid_inputs_and_arm_are_rejected(self):
        with self.assertRaises(v.V270Error):
            v.fuse_counts([0, 1], [0], "null_veto")
        with self.assertRaises(v.V270Error):
            v.fuse_counts([0, 7], [0, 1], "null_veto")
        with self.assertRaises(ValueError):
            v.fuse_counts([0], [0], "typo")


class ReportTests(unittest.TestCase):
    def test_cardinality_report_is_exact_and_conserves_rows(self):
        truth = np.array([0, 0, 1, 2, 2, 6], dtype=np.int32)
        pred = np.array([0, 1, 0, 2, 1, 5], dtype=np.int32)
        report = v.cardinality_report(truth, pred)
        self.assertEqual(report["rows"], 6)
        self.assertEqual(report["correct"], 2)
        self.assertAlmostEqual(report["exact"], 2 / 6)
        self.assertEqual(report["poly_rows"], 3)
        self.assertEqual(report["poly_correct"], 1)
        self.assertEqual(report["k0_false_birth_rows"], 1)
        self.assertEqual(report["k1_omission_rows"], 1)
        self.assertEqual(np.sum(report["confusion_true_by_predicted"]), 6)

    def test_transition_separates_corrections_and_regressions(self):
        truth = np.array([0, 1, 2, 3], dtype=np.int32)
        anchor = np.array([1, 1, 1, 3], dtype=np.int32)
        treatment = np.array([0, 0, 2, 3], dtype=np.int32)
        report = v.transition_report(truth, anchor, treatment)
        self.assertEqual(report["changed_rows"], 3)
        self.assertEqual(report["corrected_rows"], 2)
        self.assertEqual(report["regressed_rows"], 1)
        self.assertEqual(report["net_correct_rows"], 1)


if __name__ == "__main__":
    unittest.main()
