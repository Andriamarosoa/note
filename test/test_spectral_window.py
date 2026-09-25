"""Regressions for fixed-horizon coverage and historical input compatibility."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from scripts.spectral_window import LEGACY, COVERED, cache_window, window_metadata, time_coordinates
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts.train_v86_state_transition_proposals import MAX_HORIZON
from scripts.train_v90_structured_cluster_cardinality import CLUSTER_WINDOW_SAMPLES
from scripts.train_v92_string_factorized_cardinality import LOCAL_RADIUS_SAMPLES
from test.test_candidate_timing import fixture


class SpectralWindowTests(unittest.TestCase):
    def test_protocol_bound_and_late_attack_visibility(self):
        latest = CLUSTER_WINDOW_SAMPLES + LOCAL_RADIUS_SAMPLES
        self.assertEqual(COVERED.post_samples, CLUSTER_WINDOW_SAMPLES + MAX_HORIZON)
        self.assertLessEqual(COVERED.centers[0], -LOCAL_RADIUS_SAMPLES)
        self.assertGreaterEqual(COVERED.centers[-1], latest)
        segment = np.zeros(COVERED.segment_samples, np.float32)
        segment[COVERED.pre_samples + latest] = 1
        new = v100._spectral_map_from_segment(segment, window=COVERED)
        old = v100._spectral_map_from_segment(segment[:LEGACY.segment_samples])
        np.testing.assert_array_equal(new[:LEGACY.time_frames], old)
        self.assertTrue(np.any(new[LEGACY.time_frames:] > 0))

    def test_prefix_and_no_dependency_after_horizon(self):
        audio = np.random.default_rng(21).normal(size=10000).astype(np.float32)
        start = 2000
        def get(a):
            return v100._spectral_map_from_segment(v100._pcm_window(
                a, start - COVERED.pre_samples, COVERED.segment_samples), window=COVERED)
        before = get(audio)
        old = v100._spectral_map_from_segment(v100._pcm_window(
            audio, start - LEGACY.pre_samples, LEGACY.segment_samples))
        np.testing.assert_array_equal(before[:23], old)
        audio[start + COVERED.post_samples:] = 1e6
        np.testing.assert_array_equal(before, get(audio))
        np.testing.assert_array_equal(v100._pcm_window(np.ones(4), -2, 9), [0, 0, 1, 1, 1, 1, 0, 0, 0])

    def test_versioned_round_trip_and_reject_mixed_or_unlabelled_windows(self):
        cache, _, _ = fixture([[1000, 1200]])
        spectral = np.zeros((1, 31, 64, 3), np.float16)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'v100-spectral-shard-00.npz'
            v100._save_spectral_cache(path, cache, spectral, np.zeros((1, 6)), window=COVERED)
            loaded = v100._load_spectral_caches(root)
            self.assertEqual(cache_window(loaded), COVERED)
            for key in ('sequence', 'mask', 'candidate_samples', 'full_candidate_samples'):
                np.testing.assert_array_equal(loaded[key], cache[key])
            v100._save_spectral_cache(root / 'v100-spectral-shard-01.npz', cache, spectral[:, :23], np.zeros((1, 6)))
            with self.assertRaisesRegex(v100.V100Error, 'mix spectral'):
                v100._load_spectral_caches(root)
        for data, version in ((dict(target=[0], spectral=spectral), 2),
                              (dict(target=[0], spectral=spectral), 3),
                              (dict(target=[0], spectral=spectral, **window_metadata(COVERED)), 2)):
            with self.subTest(version=version), self.assertRaises(ValueError):
                cache_window(data, version=version)

    def test_supervision_extends_time_axis_without_changing_assignment(self):
        cache, _, _ = fixture([[1000, 2764]])
        new_cache = {**cache, **window_metadata(COVERED)}
        track = SimpleNamespace(player_id='00', annotation_member='00_test_comp.jams')
        with patch.object(v102, 'index_guitarset', return_value=[track]), \
             patch.object(v102.v101, '_pitch_events', return_value=[(0, 3646, 60)]):
            old = v102._derive_supervision(cache['members'], [], Path('.'), assignment_cache=cache)
            new = v102._derive_supervision(cache['members'], [], Path('.'), assignment_cache=new_cache)
            with self.assertRaisesRegex(v102.V102Error, 'window/cache mismatch'):
                v102._derive_supervision(cache['members'], [], Path('.'), assignment_cache=new_cache, window=LEGACY)
        for key in (0, 1, 3):
            np.testing.assert_array_equal(old[key], new[key])
        self.assertEqual(new[2].shape, (1, 6, 31))
        self.assertAlmostEqual(float(new[2][0, 0].sum()), 1., places=6)
        self.assertEqual(old[4]['outside_frame_center_range'], 1)
        self.assertEqual(new[4]['outside_frame_center_range'], 0)

    def test_runtime_and_cache_keep_original_start_after_truncation(self):
        cache, clusters, records = fixture([list(range(1000, 2000, 20))], [.01, .02] + [.9] * 48)
        track = SimpleNamespace(player_id='00', annotation_member='00_test_comp.jams',
                                audio_zip=Path('audio.zip'), audio_member='audio.wav')
        audio = SimpleNamespace(samples=(np.sin(np.arange(6000) / 11) * 15000).astype(np.int16))
        with patch.object(v100, 'index_guitarset', return_value=[track]), \
             patch.object(v100, 'decode_pcm16_mono_wav', return_value=audio):
            cached, _ = v100._spectral_maps_for_cache(cache, Path('.'), window=COVERED)
            runtime = v100._spectral_maps_for_runtime([track], clusters, records, window=COVERED)
        np.testing.assert_array_equal(cached, runtime.astype(np.float16))

    def test_coordinates_preserve_legacy_prefix(self):
        old = np.linspace(-1, 1, 23, dtype=np.float32)
        np.testing.assert_array_equal(time_coordinates(23), old)
        np.testing.assert_array_equal(time_coordinates(31)[:23], old)

    def test_transitive_mass_aware_builder_preserves_window_argument(self):
        import inspect
        from scripts import train_v260_count_weighting
        from scripts import train_v272_poly_conditional_count
        from scripts import train_v240_categorical_k_candidate_subset as v240
        self.assertIn('time_frames', inspect.signature(v240.v102._build_model).parameters)
        with self.assertRaisesRegex(v240.V240Error, 'count_only'):
            v240._build_model({}, time_frames=31)


if __name__ == '__main__':
    unittest.main()
