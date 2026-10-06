import unittest
import numpy as np

from scripts.audit_v273_b_low_subclusters import (
    fit_projection, select_k, policy_from_fit, evaluate_policy
)


class BLowSubclusterTests(unittest.TestCase):
    def test_select_k_finds_clear_multi_cluster_structure(self):
        rng = np.random.default_rng(273)
        a = rng.normal(loc=-3.0, scale=.25, size=(80, 37))
        b = rng.normal(loc=0.0, scale=.25, size=(80, 37))
        c = rng.normal(loc=3.0, scale=.25, size=(80, 37))
        X = np.vstack([a, b, c])
        _, _, P, _ = fit_projection(X)
        k, _, labels, candidates = select_k(P, 42000)
        self.assertGreaterEqual(k, 2)
        self.assertLessEqual(k, 6)
        self.assertEqual(len(labels), len(X))
        self.assertTrue(any(q["k"] == k and q["admissible"] for q in candidates))

    def test_policy_activates_only_fit_positive_clusters(self):
        y = np.array([2,2,2,3, 3,3,3,2, 4,4])
        labels = np.array([0,0,0,0, 1,1,1,1, 2,2])
        active, table = policy_from_fit(y, labels, 3)
        self.assertIn(0, active)
        self.assertNotIn(1, active)
        self.assertNotIn(2, active)
        self.assertEqual(table["0"]["fit_or_val_net_if_3to2"], 2)

    def test_val_oracle_is_at_least_selected_policy_net(self):
        y = np.array([2,2,3,3, 2,3,3,3])
        labels = np.array([0,0,0,0, 1,1,1,1])
        out = evaluate_policy(y, labels, [0], 2)
        self.assertGreaterEqual(out["oracle_val_cluster_net"], out["global_net"])
        self.assertEqual(out["corrections"], 2)
        self.assertEqual(out["regressions"], 2)

    def test_other_k_actions_do_not_change_exact_net(self):
        y = np.array([2,3,4,5])
        labels = np.zeros(4, dtype=int)
        out = evaluate_policy(y, labels, [0], 1)
        self.assertEqual(out["global_net"], 0)
        self.assertEqual(out["other_k_actions"], 2)


if __name__ == "__main__":
    unittest.main()
