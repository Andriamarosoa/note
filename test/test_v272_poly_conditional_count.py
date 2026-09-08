import unittest

import numpy as np

from scripts import train_v272_poly_conditional_count as v


class ConditionalPolyRuleTests(unittest.TestCase):
    def test_reclassification_only_changes_base_poly_rows(self):
        base = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        probability = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
        ], dtype=np.float32)
        out, active, specialist = v.apply_poly_reclassification(base, probability)
        np.testing.assert_array_equal(active, np.array([False, False, True, True, True]))
        np.testing.assert_array_equal(specialist, np.array([2, 3, 3, 4, 6]))
        np.testing.assert_array_equal(out, np.array([0, 1, 3, 4, 6]))

    def test_correct_nonpoly_rows_cannot_be_changed(self):
        truth = np.array([0, 1, 0, 1, 2, 3], dtype=np.int32)
        base = np.array([0, 1, 2, 3, 2, 2], dtype=np.int32)
        probability = np.zeros((6, 5), dtype=np.float32)
        probability[:, 4] = 1.0
        out, _, _ = v.apply_poly_reclassification(base, probability)
        correct_nonpoly = (truth < 2) & (base == truth)
        np.testing.assert_array_equal(out[correct_nonpoly], base[correct_nonpoly])
        self.assertEqual(
            int(np.sum((truth < 2) & (out == truth))),
            int(np.sum((truth < 2) & (base == truth))),
        )

    def test_invalid_probability_is_rejected(self):
        base = np.array([2], dtype=np.int32)
        with self.assertRaises(v.V272Error):
            v.apply_poly_reclassification(base, np.ones((1, 7), dtype=np.float32) / 7)
        with self.assertRaises(v.V272Error):
            v.apply_poly_reclassification(base, np.ones((1, 5), dtype=np.float32))


class PolyWeightTests(unittest.TestCase):
    def test_uniform_weights_are_one(self):
        k = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        np.testing.assert_array_equal(v.poly_class_weights(k, "uniform"), np.ones(5, dtype=np.float32))

    def test_weighted_table_is_normalized_on_fit_rows(self):
        k = np.array([2, 2, 2, 3, 3, 4, 5, 6], dtype=np.int32)
        table = v.poly_class_weights(k, "weighted")
        self.assertEqual(table.shape, (5,))
        self.assertTrue(np.isfinite(table).all())
        self.assertAlmostEqual(float(np.mean(table[k - 2])), 1.0, places=6)
        self.assertGreater(table[4], table[0])

    def test_missing_poly_class_is_rejected_for_weighted_arm(self):
        with self.assertRaises(v.V272Error):
            v.poly_class_weights(np.array([2, 3, 4, 5], dtype=np.int32), "weighted")


class ArmSelectionTests(unittest.TestCase):
    def test_exact_count_has_priority_over_nll(self):
        inner = {
            "uniform": {"correct": 10, "selected_val_loss": 0.9},
            "weighted": {"correct": 11, "selected_val_loss": 1.2},
        }
        self.assertEqual(v.select_arm(inner), "weighted")

    def test_lower_nll_breaks_exact_tie(self):
        inner = {
            "uniform": {"correct": 10, "selected_val_loss": 0.8},
            "weighted": {"correct": 10, "selected_val_loss": 0.7},
        }
        self.assertEqual(v.select_arm(inner), "weighted")

    def test_uniform_breaks_full_tie(self):
        inner = {
            "uniform": {"correct": 10, "selected_val_loss": 0.8},
            "weighted": {"correct": 10, "selected_val_loss": 0.8},
        }
        self.assertEqual(v.select_arm(inner), "uniform")


if __name__ == "__main__":
    unittest.main()
