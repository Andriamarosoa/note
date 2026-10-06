import unittest
import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_novel_marginal_gain import fit_small_nnls, marginal_feature, template_t2


class NovelMarginalGainTests(unittest.TestCase):
    def test_small_nnls_recovers_nonnegative_exact_mix(self):
        rng = np.random.default_rng(273)
        D = np.abs(rng.normal(size=(64, 3)))
        coefficient = np.array([0.7, 0.0, 1.2])
        x = D @ coefficient
        cost, got, active = fit_small_nnls(D, x)
        self.assertLess(cost, 1e-16)
        np.testing.assert_allclose(D @ got, x, atol=1e-8)
        self.assertTrue(set(active) <= {0, 1, 2})

    def test_marginal_feature_is_nonnegative_and_bounded_by_gain(self):
        freq = np.linspace(h.MIN_HZ, h.MAX_ANALYSIS_HZ, 1800)
        f0 = np.array([101.0, 173.0, 283.0])
        D = np.column_stack([template_t2(freq, f) for f in f0])
        x = D @ np.array([1.0, 0.8, 0.6])
        out = marginal_feature(freq, x, f0)
        self.assertGreaterEqual(out["novel_marginal_gain"], 0.0)
        self.assertGreaterEqual(out["delta_unique_fraction"], 0.0)
        self.assertLessEqual(out["delta_unique_fraction"], 1.0 + 1e-9)
        self.assertLessEqual(
            out["novel_marginal_gain"], out["marginal_gain"] + 1e-9
        )

    def test_marginal_component_is_not_in_best_internal_pair(self):
        freq = np.linspace(h.MIN_HZ, h.MAX_ANALYSIS_HZ, 1800)
        f0 = np.array([110.0, 196.0, 311.0])
        D = np.column_stack([template_t2(freq, f) for f in f0])
        x = D @ np.array([1.0, 0.9, 0.2])
        out = marginal_feature(freq, x, f0)
        self.assertNotIn(out["marginal_component"], out["best_internal_pair"])
        self.assertEqual(len(out["best_internal_pair"]), 2)


if __name__ == "__main__":
    unittest.main()
