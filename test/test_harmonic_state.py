import math
import unittest

import numpy as np

from causal_note.guitarset import SAMPLE_RATE
from causal_note.harmonic_state import harmonic_change


def _pcm(values):
    clipped = np.clip(np.asarray(values), -0.999, 0.999)
    return np.asarray(np.round(clipped * 32767.0), dtype=np.int16)


class HarmonicStateTests(unittest.TestCase):
    def test_existing_harmonic_amplitude_change_is_more_explained_than_new_family(self):
        length = 4096
        split = 2048
        time = np.arange(length, dtype=np.float64) / SAMPLE_RATE
        old = np.sin(2.0 * math.pi * 110.0 * time)

        transform = 0.12 * old
        transform[split:] = 0.72 * old[split:]

        novel = 0.12 * old
        new_family = np.sin(2.0 * math.pi * 146.832 * time)
        novel[split:] += 0.60 * new_family[split:]

        existing_change = harmonic_change(
            np, _pcm(transform), split, (110.0,), window_samples=1024, fft_size=4096
        )
        novel_change = harmonic_change(
            np, _pcm(novel), split, (110.0,), window_samples=1024, fft_size=4096
        )

        self.assertGreater(existing_change.positive_flux_over_pre_energy, 0.0)
        self.assertGreater(novel_change.positive_flux_over_pre_energy, 0.0)
        self.assertGreater(
            existing_change.active_harmonic_flux_fraction,
            novel_change.active_harmonic_flux_fraction,
        )
        self.assertGreater(existing_change.active_harmonic_flux_fraction, 0.50)

    def test_feature_validation_rejects_bad_requests(self):
        samples = np.zeros(2048, dtype=np.int16)
        with self.assertRaises(ValueError):
            harmonic_change(np, samples, -1, (110.0,))
        with self.assertRaises(ValueError):
            harmonic_change(np, samples, 100, (110.0,), window_samples=0)


if __name__ == "__main__":
    unittest.main()
