import unittest
from unittest.mock import patch

import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_decay_guard import best_batch
from scripts.audit_v273_temporal_stability import (
    costs_for_combinations, post_spectra, select_on_fit,
)
from scripts.audit_v273_temporal_persistence_guard import (
    FEATURE_NAMES, persistence_features, score, train_predict,
)


class TemporalStabilityTests(unittest.TestCase):
    def test_first_window_exact_and_pre_is_shared(self):
        t = np.arange(6 * h.WINDOW) / 44100
        audio = np.sin(2 * np.pi * 220 * t)
        audio[:2 * h.WINDOW] = 0
        freq, x, mass, _, padding = post_spectra(audio, 2 * h.WINDOW)
        ref_freq, ref_x = h.transition_spectrum(audio, 2 * h.WINDOW)
        np.testing.assert_array_equal(freq, ref_freq)
        np.testing.assert_array_equal(x[0], ref_x)
        self.assertGreater(mass[1] / mass[0], .99)
        self.assertEqual(int(np.sum(padding)), 0)
        # A successive delta would cancel a sustained note instead.
        _, wrong = h.transition_spectrum(audio, 3 * h.WINDOW)
        self.assertGreater(np.max(np.abs(x[1] - wrong)), .01)

    def test_joint_repeated_window_and_reference_solver(self):
        rng = np.random.default_rng(60104)
        D = rng.random((70, 9)); D /= np.linalg.norm(D, axis=0)
        x = rng.random(70)
        for k in (2, 3):
            costs, combos = costs_for_combinations(D, x, k)
            best, x2 = best_batch(D, x, k)
            self.assertEqual(tuple(combos[costs.argmin()]), best[1])
            self.assertAlmostEqual(float(costs.min()), best[0] / (x2 + 1e-12), places=12)
            repeat = np.mean([costs, costs], axis=0)
            np.testing.assert_array_equal(repeat, costs)

    def test_synthetic_variable_amplitudes_and_transient(self):
        D = np.eye(5)
        for xs in (
            [np.array([1., .8, .6, 0, 0]), np.array([.4, .9, .7, 0, 0])],
            [np.array([1., .8, .6, .7, 0]), np.array([1., .8, .6, 0, 0])],
        ):
            cost = [costs_for_combinations(D, x, 3)[0] for x in xs]
            _, combos = costs_for_combinations(D, xs[0], 3)
            winner = tuple(combos[np.mean(cost, axis=0).argmin()])
            self.assertEqual(winner, (0, 1, 2))

    def test_fit_selection_never_sees_heldout_and_abstains(self):
        y = np.tile([2, 3, 4], 6)
        folds = np.repeat([0, 1, 4], 6)
        X = np.zeros((len(y), 2, 3))
        X[:, :, 0] = np.arange(len(y))[:, None]
        X[:, 1, 1] = y == 2
        def predict(train, yt, val, states):
            fit_ids, val_ids = train[:, 0].astype(int), val[:, 0].astype(int)
            self.assertFalse(set(folds[fit_ids]) & set(folds[val_ids]))
            return val[:, 1]
        with patch('scripts.audit_v273_temporal_stability.probability', side_effect=predict):
            selected, _, _ = select_on_fit(X, y, folds)
        self.assertEqual(selected, 1)
        with patch('scripts.audit_v273_temporal_stability.probability', return_value=np.ones(6)):
            selected, _, _ = select_on_fit(X, y, folds)
        self.assertIsNone(selected)
        with self.assertRaisesRegex(RuntimeError, 'excluded'):
            select_on_fit(X, y, np.full(len(y), 3))

    def test_persistence_uses_fixed_triplets_and_replays_three_features(self):
        first = np.array([[.4, .3, 200], [.6, .1, 300]])
        triplets = np.array([[[100, 200, 300], [101, 203, 500], [100, 200, 500]],
                             [[100, 200, 300], [800, 900, 1000], [100, 200, 300]]], float)
        X = persistence_features(first, triplets)
        np.testing.assert_array_equal(X[:, :2], first[:, :2])
        np.testing.assert_allclose(X[:, 2], [2 / 3, 0])
        rng = np.random.default_rng(904)
        Xf, Xv = rng.random((30, 3)), rng.random((9, 3))
        yf = np.tile([2, 3, 4], 10)
        for names in FEATURE_NAMES:
            dim = len(names)
            p, state = train_predict(Xf[:, :dim], yf, Xv[:, :dim], names)
            np.testing.assert_allclose(score(Xv[:, :dim], state, names), p, atol=1e-15, rtol=0)
            self.assertEqual(state['scale_n_samples_seen'], 20)


if __name__ == '__main__':
    unittest.main()
