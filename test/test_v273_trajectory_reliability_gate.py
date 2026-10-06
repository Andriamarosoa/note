import unittest
import numpy as np

from scripts.audit_v273_trajectory_reliability_gate import (
    GATE_FEATURES_A, GATE_FEATURES_B, gate_matrix, accounting
)


class TrajectoryReliabilityGateTests(unittest.TestCase):
    def _row(self, true_k=2):
        return {
            "true_k": true_k,
            "traj": np.arange(17, dtype=np.float64),
            "acoustic": np.arange(6, dtype=np.float64),
        }

    def test_gate_dimensions(self):
        rows = [self._row(2), self._row(3)]
        p = np.array([.8, .7])
        a = gate_matrix(rows, p, "trajectory_only")
        b = gate_matrix(rows, p, "trajectory_plus_acoustic")
        self.assertEqual(a.shape, (2, len(GATE_FEATURES_A)))
        self.assertEqual(b.shape, (2, len(GATE_FEATURES_B)))

    def test_accounting(self):
        rows = [self._row(2), self._row(3), self._row(2)]
        q = accounting(rows, [True, True, False])
        self.assertEqual(q["corrections"], 1)
        self.assertEqual(q["regressions"], 1)
        self.assertEqual(q["net"], 0)

    def test_blocking_regression_improves_net(self):
        rows = [self._row(2), self._row(3)]
        raw = accounting(rows, [True, True])
        gated = accounting(rows, [True, False])
        self.assertGreater(gated["net"], raw["net"])


if __name__ == "__main__":
    unittest.main()
