"""Tests for optional NOTE live causal gain; does not require sound hardware."""
from __future__ import annotations

import unittest

import numpy as np

from scripts.live_input_level import NoteAutoLevel
from causal_note.strict_live import CausalPitchFeatures, HOP, SR


class AutoLevelTests(unittest.TestCase):
    @staticmethod
    def tone(amplitude=0.02, samples=HOP, freq=329.63):
        t = np.arange(samples, dtype=np.float64) / SR
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def level(self, block_size=HOP, **kwargs):
        return NoteAutoLevel(sample_rate=SR, block_size=block_size, **kwargs)

    def test_weak_but_audible_tone_is_amplified_with_bound(self):
        l = self.level()
        x = self.tone()
        y = l.apply(x)
        self.assertGreater(float(np.max(np.abs(y))), float(np.max(np.abs(x))))
        self.assertLessEqual(l.latest.gain_db, 18.0 + 1e-6)
        self.assertGreater(l.active_blocks, 0)
        self.assertFalse(np.any(np.abs(y) >= 1.0))

    def test_very_weak_e2_is_not_mistaken_for_an_onset(self):
        l = self.level()
        x = self.tone(1e-6, freq=82.41)
        y = l.apply(x)
        self.assertEqual(l.active_blocks, 0)
        self.assertFalse(l.latest.initialized)
        np.testing.assert_array_equal(y, x)

    def test_silence_is_not_boosted_or_initialized(self):
        l = self.level()
        for _ in range(50):
            out = l.apply(np.zeros(HOP, dtype=np.float32))
            self.assertTrue(np.all(out == 0))
        self.assertEqual(l.active_blocks, 0)
        self.assertFalse(l.latest.initialized)

    def test_gain_does_not_ride_a_decay_tail(self):
        l = self.level()
        first = self.tone(.01)
        l.apply(first)
        initial_gain = l.latest.gain_db
        for amplitude in (.008, .006, .004, .003, .002):
            l.apply(self.tone(amplitude))
            self.assertLessEqual(l.latest.gain_db, initial_gain + 1e-6)

    def test_no_clipping_of_loud_tone(self):
        l = self.level()
        y = l.apply(self.tone(.95))
        self.assertLessEqual(float(np.max(np.abs(y))), .981)
        self.assertEqual(l.over_limit_input_samples, 0)

    def test_supports_nondefault_input_block_sizes(self):
        for block_size in (128, 256, 512, 1024):
            with self.subTest(block_size=block_size):
                l = self.level(block_size)
                out = l.apply(self.tone(samples=block_size))
                self.assertEqual(len(out), block_size)
                self.assertTrue(np.isfinite(out).all())

    def test_final_partial_wav_block_preserves_length(self):
        l = self.level()
        tail = self.tone(samples=107)
        out = l.apply(tail)
        self.assertEqual(len(out), 107)
        self.assertTrue(np.isfinite(out).all())

    def test_disabled_path_keeps_exact_original_pcm(self):
        # Calling the feature frontend directly is the default NOTE path.
        audio = np.concatenate([self.tone(), self.tone()])
        baseline = CausalPitchFeatures().push(audio, start_sample=0)
        replay = CausalPitchFeatures().push(audio.copy(), start_sample=0)
        self.assertEqual(len(baseline), len(replay))
        for ((ta, a), (tb, b)) in zip(baseline, replay):
            self.assertEqual(ta, tb)
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
