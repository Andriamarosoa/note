import unittest
import numpy as np

from scripts.audit_v273_semitone_trajectory_shape import (
    feature_row, longest_run, entropy
)


class SemitoneTrajectoryShapeTests(unittest.TestCase):
    def test_longest_run(self):
        self.assertEqual(longest_run([2,2,3,2,2,2,3], 2), 3)

    def test_entropy_constant(self):
        self.assertAlmostEqual(entropy([3,3,3]), 0.0)

    def test_feature_row_shape(self):
        trajectory = [3,3,2,2,3,2,2,2] + [3]*17
        vec, named = feature_row(
            trajectory,
            first_k2=2,
            first_non3=2,
            first_below3=2,
            reentered=True,
            monotone=False,
        )
        self.assertEqual(vec.shape, (17,))
        self.assertGreater(named["fraction_k2"], 0.0)
        self.assertGreaterEqual(named["transitions_3_to_2"], 1.0)
        self.assertEqual(named["longest_k2_run"], 3.0)
        self.assertEqual(named["reentered_k3"], 1.0)

    def test_requires_initial_k3(self):
        with self.assertRaises(RuntimeError):
            feature_row(
                [2] + [3]*24,
                first_k2=0,
                first_non3=0,
                first_below3=0,
                reentered=False,
                monotone=False,
            )


if __name__ == "__main__":
    unittest.main()
