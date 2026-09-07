"""V24 target feasibility and runtime-count regression tests (no training)."""
from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import train_v240_categorical_k_candidate_subset as v240


class InjectiveTimeMatchTests(unittest.TestCase):
    def test_simultaneous_events_get_distinct_candidates(self):
        ids = v240._injective_time_match([100, 100], [100, 110], 50)
        self.assertEqual(set(ids), {0, 1})

    def test_maximum_cardinality_beats_greedy_nearest(self):
        # Greedy nearest gives 6 to event 4 and strands event 9.
        ids = v240._injective_time_match([4, 9], [0, 6], 4)
        np.testing.assert_array_equal(ids, [0, 1])

    def test_minimum_error_and_original_candidate_indices(self):
        ids = v240._injective_time_match([21, 9], [100, 20, 10], 50)
        np.testing.assert_array_equal(ids, [1, 2])

    def test_empty_nonfinite_and_tolerance_boundary(self):
        np.testing.assert_array_equal(v240._injective_time_match([0, 100], [], 50), [-1, -1])
        np.testing.assert_array_equal(v240._injective_time_match([np.nan, 0], [np.nan, 50], 50), [-1, 1])
        np.testing.assert_array_equal(v240._injective_time_match([0], [50.1], 50), [-1])

    def test_matches_exhaustive_optimum_on_small_random_inputs(self):
        rng = np.random.default_rng(240)
        for _ in range(60):
            n, m = rng.integers(1, 5, size=2)
            truth, cand = rng.integers(0, 20, size=n), rng.integers(0, 20, size=m)
            tolerance = 5
            best = (0, 0)
            for choice in itertools.product(range(-1, m), repeat=n):
                used = [j for j in choice if j >= 0]
                if len(set(used)) != len(used):
                    continue
                distances = [abs(truth[i] - cand[j]) for i, j in enumerate(choice) if j >= 0]
                if any(d > tolerance for d in distances):
                    continue
                best = min(best, (-len(used), sum(distances)))
            ids = v240._injective_time_match(truth, cand, tolerance)
            got = (-int(np.sum(ids >= 0)), sum(abs(truth[i] - cand[j]) for i, j in enumerate(ids) if j >= 0))
            self.assertEqual(got, best)


class CandidateSupervisionTests(unittest.TestCase):
    def weights(self, cache, mask, k, present, valid):
        # Unchanged string-head class balancing requires a full mixed dataset.
        # Isolate that inherited calculation while exercising V24's actual mask
        # and categorical class weighting on the small regression fixtures.
        with patch.object(v240.v171, '_sample_weights', return_value={}):
            return v240._sample_weights(cache, mask, k, present, valid)

    def fixture(self, candidate_times, event_times, count=None):
        n = 1
        cache = {"sequence": np.zeros((n, 48, 2), dtype=np.float32),
                 "mask": np.zeros((n, 48), dtype=np.float32),
                 "slot_targets": np.zeros((n, 6), dtype=np.float32)}
        cache["sequence"][0, :len(candidate_times), -2] = np.asarray(candidate_times) / v240.v130.CLUSTER_WINDOW_SAMPLES
        cache["mask"][0, :len(candidate_times)] = 1
        mask = np.zeros((n, 6), dtype=np.float32)
        samples = np.zeros((n, 6), dtype=np.float32)
        mask[0, :len(event_times)] = 1
        samples[0, :len(event_times)] = event_times
        times = np.zeros((n, 6, v240.v130.TIME_FRAMES), dtype=np.float32)
        times[:, :, 0] = 1
        k = np.array([len(event_times) if count is None else count])
        return cache, mask, times, samples, k

    def test_production_supervision_targets_and_weights_are_consistent(self):
        args = self.fixture([100, 110], [100, 100])
        cache, mask, times, _, k = args
        present, event_time, candidate, valid, _, diag = v240._ordered_event_supervision(*args)
        self.assertEqual(int(valid.sum()), 2)
        self.assertEqual(int(candidate.sum(axis=1).max()), 1)
        self.assertEqual(diag['fully_matched_positive_rows'], 1)
        with patch.object(v240.v190, '_birth_center_targets', return_value=(np.ones((1, 2)), np.ones(1), {})):
            targets = v240._targets(cache, np.zeros((1, 6)), times, k, present, event_time, candidate)
        self.assertEqual(float(targets['candidate_subset'].sum()), 2)
        weights = self.weights(cache, mask, k, present, valid)
        self.assertEqual(float(weights['candidate_subset'][0]), 1)

    def test_partial_match_masks_subset_loss_but_keeps_cardinality(self):
        args = self.fixture([100], [100, 100])
        present, _, _, valid, _, diag = v240._ordered_event_supervision(*args)
        self.assertEqual(int(valid.sum()), 1)
        self.assertEqual(diag['unmatched_present_events'], 1)
        weights = self.weights(args[0], args[1], args[-1], present, valid)
        self.assertEqual(float(weights['candidate_subset'][0]), 0)
        self.assertGreater(float(weights['cardinality'][0]), 0)

    def test_far_candidate_is_not_a_positive_target(self):
        args = self.fixture([0], [3000])  # beyond 50 ms at 44.1 kHz
        result = v240._ordered_event_supervision(*args)
        self.assertEqual(int(result[3].sum()), 0)
        self.assertEqual(result[-1]['incomplete_positive_rows'], 1)

    def test_missing_event_time_cannot_make_partial_subset_complete(self):
        args = self.fixture([100, 110], [100], count=2)
        result = v240._ordered_event_supervision(*args)
        weights = self.weights(args[0], args[1], args[-1], result[0], result[3])
        self.assertEqual(float(weights['candidate_subset'][0]), 0)

    def test_v24_installs_and_restores_its_supervision_even_on_failure(self):
        original = v240.v130._ordered_event_supervision
        def fail(args):
            self.assertIs(v240.v130._ordered_event_supervision, v240._ordered_event_supervision)
            raise RuntimeError('stop before training')
        args = SimpleNamespace(seed=v240.DEFAULT_SEED, arm=v240.BASE_ARM)
        with patch.object(v240.v172, '_fold_context', return_value={'meta_spec': {}, 'final_spec': {}}), patch.object(v240.v130, 'train_fold', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'stop before training'):
                v240.train_fold(args)
        self.assertIs(v240.v130._ordered_event_supervision, original)


class RuntimeCountDiagnosticsTests(unittest.TestCase):
    def diagnose(self, true_k, predicted_k, valid_count, selected):
        capture = {'cardinality': np.eye(7)[[predicted_k]],
                   'candidate_valid_count': np.array([valid_count]),
                   'candidate_selected_ids': np.array([selected + [-1] * (6 - len(selected))])}
        target = np.zeros((1, 48)); target[0, :true_k] = 1
        return v240._candidate_subset_diag(target, capture, np.array([true_k]))

    def test_wrong_prediction_can_still_be_realized_exactly(self):
        r = self.diagnose(3, 2, 8, [0, 1])
        self.assertEqual(r['runtime_realized_exact_categorical_k_rate'], 1)
        self.assertEqual(r['runtime_realized_exact_true_k_rate'], 0)
        self.assertEqual(r['rows_candidate_count_below_predicted_k'], 0)

    def test_candidate_shortage_is_separate_from_truth_accuracy(self):
        r = self.diagnose(2, 3, 2, [0, 1])
        self.assertEqual(r['runtime_realized_exact_categorical_k_rate'], 0)
        self.assertEqual(r['runtime_realized_exact_true_k_rate'], 1)
        self.assertEqual(r['rows_candidate_count_below_predicted_k'], 1)

    def test_emission_failure_is_not_candidate_shortage(self):
        r = self.diagnose(3, 3, 8, [0, 1])
        self.assertEqual(r['rows_candidate_count_below_predicted_k'], 0)
        self.assertEqual(r['rows_emitted_count_below_predicted_k'], 1)

    def test_zero_count_with_no_candidates(self):
        r = self.diagnose(0, 0, 0, [])
        self.assertEqual(r['runtime_realized_exact_categorical_k_rate'], 1)
        self.assertEqual(r['rows_candidate_count_below_predicted_k'], 0)


if __name__ == '__main__':
    unittest.main()
