import unittest

import numpy as np

from scripts import audit_v271_poly_rescue as v


class PolyRescueRuleTests(unittest.TestCase):
    @staticmethod
    def probs(rows):
        return np.asarray(rows, dtype=np.float32)

    def test_rule_only_lifts_remaining_low_k_rows_with_confidence(self):
        anchor = np.array([0, 1, 1, 2, 0], dtype=np.int32)
        low = np.array([0, 1, 2, 2, 1], dtype=np.int32)
        weighted = self.probs([
            [.02, .03, .90, .05, 0, 0, 0],
            [.05, .10, .15, .60, .05, .03, .02],
            [.02, .03, .90, .05, 0, 0, 0],
            [.02, .03, .90, .05, 0, 0, 0],
            [.25, .20, .15, .10, .10, .10, .10],
        ])
        actual, active, pred, confidence = v.apply_poly_rescue(anchor, low, weighted, .70)
        np.testing.assert_array_equal(pred, np.array([2, 3, 2, 2, 0]))
        np.testing.assert_array_equal(active, np.array([True, False, False, False, False]))
        np.testing.assert_array_equal(actual, np.array([2, 1, 2, 2, 1]))
        self.assertAlmostEqual(confidence[0], .90, places=5)

    def test_correct_polyphonic_low_k_predictions_are_immutable(self):
        truth = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        anchor = truth.copy()
        low = truth.copy()
        weighted = np.zeros((5, 7), dtype=np.float32)
        weighted[:, 6] = 1.0
        actual, active, _, _ = v.apply_poly_rescue(anchor, low, weighted, 0.0)
        np.testing.assert_array_equal(actual, truth)
        self.assertFalse(active.any())

    def test_low_k_polyphonic_correctness_cannot_regress(self):
        truth = np.array([0, 1, 2, 2, 3, 4], dtype=np.int32)
        anchor = np.array([0, 1, 1, 2, 1, 4], dtype=np.int32)
        low = np.array([0, 1, 1, 2, 1, 4], dtype=np.int32)
        weighted = np.zeros((6, 7), dtype=np.float32)
        weighted[:, 2] = 1.0
        actual, _, _, _ = v.apply_poly_rescue(anchor, low, weighted, 0.5)
        before = (truth >= 2) & (low == truth)
        self.assertTrue(np.all(actual[before] == truth[before]))
        self.assertGreaterEqual(
            v.v270.cardinality_report(truth, actual)["poly_correct"],
            v.v270.cardinality_report(truth, low)["poly_correct"],
        )

    def test_invalid_probability_is_rejected(self):
        anchor = np.array([0], dtype=np.int32)
        low = np.array([0], dtype=np.int32)
        with self.assertRaises(v.V271Error):
            v.apply_poly_rescue(anchor, low, np.ones((1, 6), dtype=np.float32) / 6, .5)
        with self.assertRaises(v.V271Error):
            v.apply_poly_rescue(anchor, low, np.ones((1, 7), dtype=np.float32), .5)


class ThresholdSelectionTests(unittest.TestCase):
    def test_selection_prioritizes_poly_then_global_then_conservatism(self):
        truth = np.array([0, 2, 2, 1], dtype=np.int32)
        anchor = np.array([0, 1, 1, 1], dtype=np.int32)
        low = anchor.copy()
        weighted = np.asarray([
            [.02, .03, .90, .05, 0, 0, 0],
            [.05, .05, .80, .05, .02, .02, .01],
            [.20, .15, .10, .40, .05, .05, .05],
            [.20, .20, .30, .10, .10, .05, .05],
        ], dtype=np.float32)
        selected = v.select_threshold(truth, anchor, low, weighted)
        self.assertAlmostEqual(selected["threshold"], .80, places=5)
        self.assertEqual(selected["selected"]["poly_correct"], 1)
        self.assertEqual(selected["activated_rows"], 2)
        self.assertEqual(selected["transition_from_low_k_fusion"]["net_correct_rows"], 0)

    def test_no_poly_gain_prefers_no_rescue_when_global_would_regress(self):
        truth = np.array([0, 1, 2], dtype=np.int32)
        anchor = np.array([0, 1, 1], dtype=np.int32)
        low = anchor.copy()
        weighted = np.asarray([
            [.02, .03, .90, .05, 0, 0, 0],
            [.05, .05, .10, .70, .05, .03, .02],
            [.05, .05, .10, .70, .05, .03, .02],
        ], dtype=np.float32)
        selected = v.select_threshold(truth, anchor, low, weighted)
        self.assertGreater(selected["threshold"], 1.0)
        self.assertEqual(selected["activated_rows"], 0)
        self.assertEqual(
            selected["selected"]["poly_correct"], selected["baseline"]["poly_correct"]
        )


if __name__ == "__main__":
    unittest.main()
