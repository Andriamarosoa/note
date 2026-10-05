import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import nnls
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_templates import (
    harmonic_basis, rigid_dictionary, gram_nnls,
)
from scripts.audit_v273_harmonic_decay_guard import best_batch, extract_variants, extraction_tables, select_on_fit, NAMES


class HarmonicTemplateTests(unittest.TestCase):
    def test_hann_power_matches_phase_averaged_real_sinusoid_fft(self):
        f0 = 171.27
        t = np.arange(h.WINDOW) / 44100
        w = np.hanning(h.WINDOW)
        cosine = np.abs(np.fft.rfft(w * np.cos(2 * np.pi * f0 * t), n=h.FFT_SIZE)) ** 2
        sine = np.abs(np.fft.rfft(w * np.sin(2 * np.pi * f0 * t), n=h.FFT_SIZE)) ** 2
        freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / 44100)
        keep = (freq >= 65) & (freq <= 6000)
        basis = harmonic_basis(freq[keep], f0, 'hann_power')[:, 0]
        np.testing.assert_allclose(basis, 2 * (cosine + sine)[keep] / w.sum() ** 2,
                                   rtol=1e-10, atol=1e-12)

    def test_historical_dictionary_is_exact_control(self):
        freq = np.linspace(65, 6000, 1000)
        pool = [81.3, 165.9, 249.1]
        D = rigid_dictionary([harmonic_basis(freq, f, 'gaussian') for f in pool], .5)
        np.testing.assert_allclose(D, np.column_stack([h.template(freq, f) for f in pool]),
                                   rtol=1e-14, atol=1e-14)

    def test_gram_solver_matches_rectangular_nnls_with_redundant_columns(self):
        rng = np.random.default_rng(174)
        D = rng.random((80, 12))
        D[:, 11] = D[:, 2]
        x = D[:, [1, 2, 5]] @ np.array([.2, .7, 1.1]) + .01 * rng.random(80)
        direct, residual = nnls(D, x)
        cost, a = gram_nnls(D.T @ D, D.T @ x, float(x @ x))
        self.assertTrue(np.all(a >= 0))
        self.assertAlmostEqual(cost, residual ** 2, places=10)
        np.testing.assert_allclose(D @ a, D @ direct, rtol=1e-8, atol=1e-8)

    def test_batched_exhaustive_solver_matches_reference_with_inactive_columns(self):
        rng = np.random.default_rng(853)
        D = rng.random((70, 8))
        D /= np.linalg.norm(D, axis=0)
        for x in (D[:, 2] + .01 * rng.random(70), rng.random(70)):
            for k in (2, 3):
                batched, _ = best_batch(D, x, k)
                direct, _, _, _ = h.fit_best(D, x, k)
                self.assertAlmostEqual(batched[0], direct[0], places=10)
                np.testing.assert_allclose(D[:, batched[1]] @ batched[2],
                                           D[:, direct[1]] @ direct[2], rtol=1e-7, atol=1e-7)

    def test_vectorized_historical_arm_matches_existing_extraction(self):
        freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / 44100)
        freq = freq[(freq >= h.MIN_HZ) & (freq <= h.MAX_ANALYSIS_HZ)]
        x = h.template(freq, 171.1) + .4 * h.template(freq, 327.7) + .1 * h.template(freq, 438.1)
        x /= x.sum()
        reference = h.extract_one(freq, x)
        vectorized = extract_variants(freq, x, extraction_tables(freq))[0]
        np.testing.assert_allclose(vectorized[:2], [reference['best_pair_residual_ratio'],
                                                    reference['best_triplet_residual_ratio']], atol=1e-10, rtol=0)

    def test_fit_selection_excludes_heldout_rows_and_requires_positive_net(self):
        y = np.tile([2, 3, 4], 6)
        folds = np.repeat([0, 1, 2], 6)
        X = np.zeros((len(y), len(NAMES), 3))
        X[:, :, 0] = np.arange(len(y))[:, None]
        X[:, 0, 1] = (y == 2).astype(float)
        def prediction(train, yt, valid):
            train_ids, valid_ids = train[:, 0].astype(int), valid[:, 0].astype(int)
            self.assertFalse(set(folds[train_ids]) & set(folds[valid_ids]))
            np.testing.assert_array_equal(yt, y[train_ids])
            return valid[:, 1]
        with patch('scripts.audit_v273_harmonic_decay_guard.probability', side_effect=prediction):
            chosen, scores, _ = select_on_fit(X, y, folds)
        self.assertEqual(chosen, 0)
        self.assertEqual(scores[0]['global_net'], 6)
        with patch('scripts.audit_v273_harmonic_decay_guard.probability', side_effect=lambda tr, yt, va: np.ones(len(va))):
            chosen, _, _ = select_on_fit(X, y, folds)
        self.assertIsNone(chosen)


if __name__ == '__main__':
    unittest.main()
