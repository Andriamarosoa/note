import unittest

import numpy as np

from scripts import train_v273_selective_transition_corrector as v


class TransitionRuleTests(unittest.TestCase):
    def test_only_predeclared_adjacent_transitions_can_change(self):
        base = np.array([0, 1, 2, 3, 3, 4, 5, 6], dtype=np.int32)
        proposal = np.array([3, 2, 3, 2, 4, 3, 4, 5], dtype=np.int32)
        probability = np.zeros((len(base), 5), dtype=np.float32)
        probability[np.arange(len(base)), proposal - 2] = 0.8
        probability[np.arange(len(base)), np.clip(base - 2, 0, 4)] += 0.2
        thresholds = {v.transition_name(*pair): 0.5 for pair in v.TRANSITIONS}
        corrected, active, _, _, _ = v.apply_transition_corrector(
            base, probability, thresholds
        )
        np.testing.assert_array_equal(corrected, np.array([0, 1, 3, 2, 4, 3, 5, 6]))
        np.testing.assert_array_equal(
            active, np.array([False, False, True, True, True, True, False, False])
        )

    def test_margin_is_proposal_probability_minus_base_probability(self):
        base = np.array([2, 3], dtype=np.int32)
        probability = np.array(
            [[0.20, 0.65, 0.10, 0.04, 0.01], [0.55, 0.30, 0.10, 0.04, 0.01]],
            dtype=np.float32,
        )
        proposal, margin = v.conditional_proposal(base, probability)
        np.testing.assert_array_equal(proposal, np.array([3, 2]))
        np.testing.assert_allclose(margin, np.array([0.45, 0.25]), atol=1e-6)

    def test_nonpolyphonic_reference_rows_are_immutable(self):
        base = np.array([0, 1], dtype=np.int32)
        probability = np.array(
            [[0.0, 1.0, 0, 0, 0], [1.0, 0, 0, 0, 0]], dtype=np.float32
        )
        thresholds = {v.transition_name(*pair): 0.0 for pair in v.TRANSITIONS}
        corrected, active, _, _, _ = v.apply_transition_corrector(
            base, probability, thresholds
        )
        np.testing.assert_array_equal(corrected, base)
        self.assertFalse(active.any())

    def test_invalid_threshold_set_is_rejected(self):
        probability = np.array([[1.0, 0, 0, 0, 0]], dtype=np.float32)
        with self.assertRaises(v.V273Error):
            v.apply_transition_corrector(np.array([2]), probability, {})


class TransitionSelectionTests(unittest.TestCase):
    @staticmethod
    def probability(base, proposal, proposal_probability):
        result = np.zeros((len(base), 5), dtype=np.float32)
        for row, (source, target, confidence) in enumerate(
            zip(base, proposal, proposal_probability)
        ):
            result[row, source - 2] = 1.0 - confidence
            result[row, target - 2] += confidence
        return result

    def test_selection_is_independent_per_direction(self):
        base = np.array([2, 2, 2, 3, 3, 3], dtype=np.int32)
        proposal = np.array([3, 3, 3, 2, 2, 2], dtype=np.int32)
        truth = np.array([3, 4, 3, 2, 3, 3], dtype=np.int32)
        probability = self.probability(base, proposal, [0.9, 0.8, 0.7, 0.9, 0.8, 0.7])
        selected = v.select_transition_thresholds(truth, base, probability)
        self.assertAlmostEqual(selected["thresholds"]["2_to_3"], 0.4, places=5)
        self.assertAlmostEqual(selected["thresholds"]["3_to_2"], 0.8, places=5)
        self.assertEqual(selected["directions"]["2_to_3"]["net_correct_rows"], 2)
        self.assertEqual(selected["directions"]["3_to_2"]["net_correct_rows"], 1)
        self.assertGreater(selected["thresholds"]["3_to_4"], 1.0)
        self.assertGreater(selected["thresholds"]["4_to_3"], 1.0)

    def test_no_positive_net_prefers_no_override(self):
        base = np.array([2, 2, 2], dtype=np.int32)
        proposal = np.array([3, 3, 3], dtype=np.int32)
        truth = np.array([2, 2, 4], dtype=np.int32)
        probability = self.probability(base, proposal, [0.9, 0.8, 0.7])
        selected = v.select_transition_thresholds(truth, base, probability)
        self.assertGreater(selected["thresholds"]["2_to_3"], 1.0)
        self.assertEqual(selected["directions"]["2_to_3"]["activated_rows"], 0)

    def test_correct_nonpolyphonic_rows_cannot_regress(self):
        truth = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        base = truth.copy()
        probability = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        thresholds = {v.transition_name(*pair): 0.0 for pair in v.TRANSITIONS}
        corrected, _, _, _, _ = v.apply_transition_corrector(base, probability, thresholds)
        self.assertTrue(np.all(corrected[truth < 2] == truth[truth < 2]))


if __name__ == "__main__":
    unittest.main()
