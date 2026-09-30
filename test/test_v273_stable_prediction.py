import unittest
import numpy as np

from scripts.probe_v273_coherent_decay import LONG_HISTORY, POST_SAMPLES, make_wave
from scripts.v273_stable_prediction import (
    fit_burg, forecast_stable, predict_polynomial, REFLECTION_MARGIN)


class StablePredictionTest(unittest.TestCase):
    def test_silence_and_invalid_input(self):
        p, info = forecast_stable(np.zeros(LONG_HISTORY), POST_SAMPLES)
        np.testing.assert_array_equal(p, 0)
        self.assertEqual(info['effective_order'], 0)
        for x in (np.zeros(20), np.full(LONG_HISTORY, np.nan)):
            with self.assertRaises(ValueError):
                forecast_stable(x, POST_SAMPLES)
        for length in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                forecast_stable(np.ones(LONG_HISTORY), length)

    def test_block_prediction_matches_scalar_recurrence(self):
        x = np.random.default_rng(22).normal(size=512)
        polynomial, _ = fit_burg(x, order=8, lag=7)
        actual = predict_polynomial(x, 127, polynomial, lag=7)
        oracle = list(x)
        for _ in range(127):
            oracle.append(-sum(polynomial[j] * oracle[-j*7]
                               for j in range(1, len(polynomial))))
        np.testing.assert_allclose(actual, oracle[len(x):], rtol=1e-12, atol=1e-12)

    def test_fitting_does_not_join_interleavings(self):
        x = np.random.default_rng(23).normal(size=1000)
        # Independently calculate first-order pooled within-stream products.
        num = den = 0.
        for offset in range(7):
            stream = x[offset::7]
            num += float(stream[1:] @ stream[:-1])
            den += float(stream[1:] @ stream[1:] + stream[:-1] @ stream[:-1])
        _, reflection = fit_burg(x, order=1, lag=7)
        self.assertAlmostEqual(reflection[0], -2*num/den, places=13)

    def test_past_only_and_scale_equivariance(self):
        a = make_wave([220, 233.08188], 1.7, noise_db=20, event='none', seed=11)
        b = make_wave([220, 233.08188], 1.7, noise_db=20, event='same_pitch', seed=11)
        np.testing.assert_array_equal(a[:LONG_HISTORY], b[:LONG_HISTORY])
        p, info = forecast_stable(a[:LONG_HISTORY], POST_SAMPLES)
        q, other = forecast_stable(b[:LONG_HISTORY], POST_SAMPLES)
        np.testing.assert_array_equal(p, q)
        self.assertEqual(info, other)
        scaled, _ = forecast_stable(a[:LONG_HISTORY].astype(float)*3.75, POST_SAMPLES)
        np.testing.assert_allclose(scaled, p*3.75, rtol=1e-8, atol=1e-8)

    def test_continuation_retains_phase_information(self):
        t = np.arange(LONG_HISTORY+POST_SAMPLES)/44100.
        x = .3*np.sin(2*np.pi*220*t+.7)
        p, _ = forecast_stable(x[:LONG_HISTORY], POST_SAMPLES)
        rms_error = np.sqrt(np.mean((p-x[LONG_HISTORY:])**2))
        self.assertLess(rms_error, .03*np.sqrt(np.mean(x[LONG_HISTORY:]**2)))

    def test_growing_oscillator_has_no_growing_fitted_poles(self):
        t = np.arange(LONG_HISTORY)/44100.
        x = np.exp(20*(t-t[-1]))*np.sin(2*np.pi*173.2*t+.3)
        polynomial, reflection = fit_burg(x)
        self.assertLessEqual(np.max(np.abs(reflection)), 1-REFLECTION_MARGIN)
        self.assertLessEqual(np.max(np.abs(np.roots(polynomial))), 1+1e-7)
        p, _ = forecast_stable(x, 10*POST_SAMPLES)
        self.assertTrue(np.isfinite(p).all())
        # Long-run mode stability is distinct from a future/pre RMS threshold.

    def test_random_and_transient_histories_have_stable_roots(self):
        rng = np.random.default_rng(24)
        for i in range(12):
            x = rng.normal(size=LONG_HISTORY)
            if i % 3 == 0:
                x *= np.exp(np.linspace(-8, 0, len(x)))
            if i % 3 == 1:
                x[5000:] = 0
            _, info = forecast_stable(x, POST_SAMPLES)
            self.assertLessEqual(info['max_pole_radius_per_lag'], 1+1e-7)


if __name__ == '__main__':
    unittest.main()
