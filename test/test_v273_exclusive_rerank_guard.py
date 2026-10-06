import unittest
import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_decay_guard import extraction_tables
from scripts.audit_v273_hann_shape_guard import best_fast
from scripts.audit_v273_exclusive_rerank_guard import (
    all_costs, combo_min_unique, rerank, TOP_N,
)


class ExclusiveRerankTests(unittest.TestCase):
    def test_all_costs_argmin_matches_best_fast(self):
        rng = np.random.default_rng(273)
        D = np.abs(rng.normal(size=(32, 7)))
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
        x = np.abs(rng.normal(size=32))
        for k in (2, 3):
            costs, combos, x2 = all_costs(D, x, k)
            best, bx2 = best_fast(D, x, k)
            self.assertAlmostEqual(x2, bx2, places=10)
            self.assertAlmostEqual(float(costs.min()), float(best[0]), places=9)
            winner = int(np.argmin(costs))
            self.assertEqual(tuple(combos[winner]), tuple(best[1]))

    def test_combo_unique_fraction_is_low_for_harmonic_overlap(self):
        freq = np.arange(50.0, 1201.0, 1.0)
        x = np.ones(len(freq), dtype=float)
        overlapping, _ = combo_min_unique(freq, x, [100.0, 200.0, 400.0])
        separated, _ = combo_min_unique(freq, x, [110.0, 173.0, 281.0])
        self.assertLess(overlapping, separated)

    def test_rerank_can_prefer_more_exclusive_plausible_combo(self):
        freq = np.arange(50.0, 1201.0, 1.0)
        x = np.ones(len(freq), dtype=float)
        # Find grid locations close to frequencies that create one highly
        # overlapping pair and one more separated pair.
        targets = [100.0, 200.0, 173.0]
        pool_ids = np.array([int(np.argmin(np.abs(h.F0_GRID - f))) for f in targets])
        combos = np.array([[0, 1], [0, 2]], dtype=int)
        costs = np.array([0.10, 0.11])
        result = rerank(freq, x, pool_ids, costs, combos, x2=1.0, top_n=2)
        self.assertEqual(tuple(result["combo"]), (0, 2))
        self.assertGreater(result["min_unique"], 0.0)

    def test_top_n_is_fixed(self):
        self.assertEqual(TOP_N, 64)


if __name__ == "__main__":
    unittest.main()
