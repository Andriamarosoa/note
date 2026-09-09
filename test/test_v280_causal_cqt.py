from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_note.v280_causal_cqt import (
    CausalCQTConfig,
    V280FeatureError,
    causal_cqt,
    cluster_feature_map,
    harmonic_ratios,
    harmonic_shift_stack,
    harmonic_validity_mask,
    midi_to_hz,
    read_track_cache,
    write_track_cache,
)


def _small_config() -> CausalCQTConfig:
    return CausalCQTConfig(
        sample_rate=4000,
        hop_length=64,
        bins_per_semitone=3,
        input_min_midi=60.0,
        output_max_midi=72.0,
        max_harmonic_order=3,
        cluster_post_samples=192,
        cluster_frames=8,
        block_frames=7,
    )


class CausalCQTConfigurationTests(unittest.TestCase):
    def test_default_grid_matches_v28_protocol(self):
        config = CausalCQTConfig()
        frequencies = config.center_frequencies_hz
        self.assertEqual(config.bins_per_octave, 36)
        self.assertEqual(config.output_bin_count, 130)
        self.assertAlmostEqual(float(frequencies[0]), float(midi_to_hz(40.0)), places=6)
        self.assertAlmostEqual(float(frequencies[1] / frequencies[0]), 2.0 ** (1.0 / 36.0), places=12)
        self.assertLessEqual(float(frequencies[-1]), config.input_max_hz * (1.0 + 1e-12))
        self.assertGreater(config.maximum_window_seconds, 0.60)
        self.assertLess(config.maximum_window_seconds, 0.65)

    def test_harmonic_contract_has_fourteen_noncentral_offsets(self):
        ratios = harmonic_ratios(8)
        self.assertEqual(len(ratios), 14)
        self.assertNotIn(1.0, ratios)
        self.assertEqual(ratios[:2], (0.5, 1.0 / 3.0))
        self.assertEqual(ratios[-2:], (7.0, 8.0))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(V280FeatureError):
            CausalCQTConfig(hop_length=0)
        with self.assertRaises(V280FeatureError):
            CausalCQTConfig(output_max_midi=83.1)
        with self.assertRaises(V280FeatureError):
            CausalCQTConfig(sample_rate=1000)


class CausalCQTSignalTests(unittest.TestCase):
    def test_pitch_grid_localizes_a_steady_sinusoid(self):
        config = _small_config()
        time_axis = np.arange(config.sample_rate, dtype=np.float64) / config.sample_rate
        audio = np.sin(2.0 * math.pi * 440.0 * time_axis).astype(np.float32)
        track = causal_cqt(audio, config)
        peak = int(np.argmax(track.magnitude[-1]))
        estimated_midi = config.input_min_midi + peak / config.bins_per_semitone
        self.assertLessEqual(abs(estimated_midi - 69.0), 1.0 / config.bins_per_semitone)
        self.assertEqual(track.frame_ends[0], 0)
        self.assertTrue(np.all(track.magnitude >= 0.0))
        self.assertTrue(np.isfinite(track.magnitude).all())

    def test_pcm16_and_normalized_float_inputs_agree(self):
        config = _small_config()
        rng = np.random.default_rng(280)
        pcm = rng.integers(-30000, 30001, 1024, dtype=np.int16)
        from_pcm = causal_cqt(pcm, config)
        from_float = causal_cqt(pcm.astype(np.float32) / 32768.0, config)
        np.testing.assert_allclose(from_pcm.magnitude, from_float.magnitude, rtol=0.0, atol=0.0)

    def test_mutating_future_samples_cannot_change_past_frames(self):
        config = _small_config()
        rng = np.random.default_rng(281)
        audio = rng.normal(0.0, 0.1, 2304).astype(np.float32)
        cutoff = 1536
        changed = audio.copy()
        changed[cutoff:] = rng.normal(0.0, 0.9, len(changed) - cutoff)
        original_track = causal_cqt(audio, config)
        changed_track = causal_cqt(changed, config)
        past = original_track.frame_ends <= cutoff
        np.testing.assert_array_equal(original_track.frame_ends, changed_track.frame_ends)
        np.testing.assert_allclose(
            original_track.magnitude[past], changed_track.magnitude[past], rtol=0.0, atol=0.0
        )

    def test_prefix_chunks_emit_the_same_frames_as_full_track(self):
        config = _small_config()
        rng = np.random.default_rng(282)
        audio = rng.normal(0.0, 0.1, 2304).astype(np.float32)
        full = causal_cqt(audio, config)
        emitted_ends = []
        emitted_magnitude = []
        last_endpoint = -1
        for stop in (517, 1301, len(audio)):
            prefix = causal_cqt(audio[:stop], config)
            new = prefix.frame_ends > last_endpoint
            emitted_ends.append(prefix.frame_ends[new])
            emitted_magnitude.append(prefix.magnitude[new])
            last_endpoint = int(prefix.frame_ends[-1])
        np.testing.assert_array_equal(np.concatenate(emitted_ends), full.frame_ends)
        np.testing.assert_allclose(
            np.concatenate(emitted_magnitude), full.magnitude, rtol=0.0, atol=0.0
        )

    def test_cluster_map_is_fixed_shape_and_ignores_post_decision_audio(self):
        config = _small_config()
        rng = np.random.default_rng(283)
        audio = rng.normal(0.0, 0.1, 2048).astype(np.float32)
        cluster_start = 960
        decision_end = 1152
        changed = audio.copy()
        changed[decision_end:] *= -7.0
        before = cluster_feature_map(
            causal_cqt(audio, config), cluster_start, config, decision_end=decision_end
        )
        after = cluster_feature_map(
            causal_cqt(changed, config), cluster_start, config, decision_end=decision_end
        )
        self.assertEqual(before.shape, (config.cluster_frames, len(config.center_frequencies_hz), 3))
        self.assertEqual(before.dtype, np.float32)
        self.assertTrue(np.all(before >= 0.0))
        np.testing.assert_allclose(before, after, rtol=0.0, atol=0.0)

    def test_cluster_contract_rejects_unavailable_future(self):
        config = _small_config()
        track = causal_cqt(np.zeros(1024, dtype=np.float32), config)
        with self.assertRaisesRegex(V280FeatureError, "available causal audio"):
            cluster_feature_map(track, 900, config, decision_end=1100)


class HarmonicShiftTests(unittest.TestCase):
    def test_fractional_gather_direction_and_padding(self):
        source = np.arange(30, dtype=np.float32).reshape(1, 30, 1)
        shifted = harmonic_shift_stack(source, bins_per_octave=12, max_harmonic_order=2)
        self.assertEqual(shifted.shape, (1, 30, 2))
        np.testing.assert_allclose(shifted[0, 15], np.array([3.0, 27.0]))
        np.testing.assert_allclose(shifted[0, 0], np.array([0.0, 12.0]))

    def test_validity_mask_exposes_out_of_range_shifts(self):
        mask = harmonic_validity_mask(30, bins_per_octave=12, max_harmonic_order=2)
        np.testing.assert_allclose(mask[15], np.ones(2))
        np.testing.assert_allclose(mask[0], np.array([0.0, 1.0]))


class CausalCQTCacheTests(unittest.TestCase):
    def test_float16_track_cache_round_trip_and_refuses_overwrite(self):
        config = _small_config()
        rng = np.random.default_rng(284)
        track = causal_cqt(rng.normal(0.0, 0.1, 1024).astype(np.float32), config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.npz"
            write_track_cache(path, track, config)
            loaded = read_track_cache(path, config)
            np.testing.assert_array_equal(loaded.frame_ends, track.frame_ends)
            np.testing.assert_array_equal(loaded.window_lengths, track.window_lengths)
            np.testing.assert_allclose(loaded.frequencies_hz, track.frequencies_hz, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                loaded.magnitude,
                track.magnitude.astype(np.float16).astype(np.float32),
                rtol=0.0,
                atol=0.0,
            )
            with self.assertRaises(FileExistsError):
                write_track_cache(path, track, config)

    def test_cache_configuration_mismatch_is_rejected(self):
        config = _small_config()
        track = causal_cqt(np.zeros(1024, dtype=np.float32), config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.npz"
            write_track_cache(path, track, config)
            changed = CausalCQTConfig(**{**config.__dict__, "log_gain": 50.0})
            with self.assertRaisesRegex(V280FeatureError, "configuration mismatch"):
                read_track_cache(path, changed)


if __name__ == "__main__":
    unittest.main()
