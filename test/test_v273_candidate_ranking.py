import unittest
from unittest.mock import patch

import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_candidate_ranking import trace_nms, trace_notes
from scripts.audit_v273_candidate_capacity_guard import extract, select_on_fit, NAMES
from scripts.audit_v273_harmonic_decay_guard import best_batch, extraction_tables, extract_variants
from scripts.audit_v273_log_frequency_guard import (
    extract as log_extract, measure_roots, select_on_fit as select_log_on_fit,
)
from scripts.audit_v273_reconstruction_criterion import (
    best_restricted, distance_to_integer_relation, synthetic_control,
    valid_owned_combinations,
)
from scripts.audit_v273_hann_shape_guard import (
    best_fast, extract_hann, hann_dictionary,
)


class CandidateRankingTests(unittest.TestCase):
    def test_nms_trace_identifies_a_blocker_outside_annotation_tolerance(self):
        grid = np.array([100., 102., 160.])
        score = np.array([2., 1., 0.])
        order, selected, blocked = trace_nms(score, grid)
        self.assertEqual(selected.tolist(), [0, 2])
        self.assertEqual(blocked, {1: 0})
        note = trace_notes([104.], score, order, selected, blocked, grid)[0]
        self.assertEqual(note['status'], 'suppressed_by_nms')
        self.assertEqual(note['first_raw_rank'], 2)
        self.assertIsNone(note['first_nms_rank'])

    def test_capacity_and_outside_grid_are_distinct_and_ties_stable(self):
        grid = np.geomspace(100, 1600, 10)
        score = np.ones(10)
        order, selected, blocked = trace_nms(score, grid)
        self.assertEqual(order.tolist(), list(range(10)))
        notes = trace_notes([grid[0], grid[-1], 4000], score, order, selected, blocked, grid)
        self.assertEqual([n['status'] for n in notes], ['represented_in_8', 'after_capacity_8', 'outside_grid'])
        self.assertEqual(notes[1]['first_nms_rank'], 10)
        self.assertEqual((notes[1]['first_raw_tie_rank_min'], notes[1]['first_raw_tie_rank_max']), (1, 10))

    def test_expansion_preserves_controls_and_cannot_worsen_reconstruction(self):
        freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / 44100)
        freq = freq[(freq >= h.MIN_HZ) & (freq <= h.MAX_ANALYSIS_HZ)]
        x = h.template(freq, 99.1) + .3 * h.template(freq, 270.3) + .07 * h.template(freq, 673.2)
        x /= x.sum()
        tables = extraction_tables(freq)
        values, pairs, trips = extract(freq, x, tables)
        control = extract_variants(freq, x, tables)
        np.testing.assert_allclose(values[:2, :2], control[[0, 2], :2], atol=1e-12, rtol=0)
        self.assertTrue(np.all(values[2:, :2] <= values[:2, :2] + 1e-12))
        self.assertTrue(np.all(values[:, 1] <= values[:, 0] + 1e-12))
        self.assertEqual(pairs.shape, (4, 2))
        self.assertEqual(trips.shape, (4, 3))

    def test_selection_has_no_heldout_fold_in_fit_and_can_abstain(self):
        y = np.tile([2, 3, 4], 6)
        folds = np.repeat([0, 1, 4], 6)
        X = np.zeros((len(y), len(NAMES), 3))
        X[:, :, 0] = np.arange(len(y))[:, None]
        X[:, 3, 1] = (y == 2).astype(float)
        def predict(train, yt, val):
            fit_ids, val_ids = train[:, 0].astype(int), val[:, 0].astype(int)
            self.assertFalse(set(folds[fit_ids]) & set(folds[val_ids]))
            return val[:, 1]
        with patch('scripts.audit_v273_candidate_capacity_guard.probability', side_effect=predict):
            chosen, _, _ = select_on_fit(X, y, folds)
        self.assertEqual(chosen, 3)
        with patch('scripts.audit_v273_candidate_capacity_guard.probability', side_effect=lambda tr, yt, va: np.ones(len(va))):
            chosen, _, _ = select_on_fit(X, y, folds)
        self.assertIsNone(chosen)

    def test_log_measure_is_fixed_and_linear_control_is_exact(self):
        freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / 44100)
        freq = freq[(freq >= h.MIN_HZ) & (freq <= h.MAX_ANALYSIS_HZ)]
        linear, logarithmic = measure_roots(freq)
        np.testing.assert_array_equal(linear, 1)
        np.testing.assert_allclose(logarithmic ** 2, h.MIN_HZ / freq, rtol=1e-15, atol=0)
        self.assertTrue(np.all(np.diff(logarithmic) < 0))
        x = h.template(freq, 96.4) + .5 * h.template(freq, 263.8) + .1 * h.template(freq, 701.2)
        x /= x.sum()
        tables = extraction_tables(freq)
        reference = extract(freq, x, tables)
        values, pairs, triplets = log_extract(freq, x, tables)
        np.testing.assert_allclose(values[0], reference[0][3], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(pairs[0], reference[1][3])
        np.testing.assert_array_equal(triplets[0], reference[2][3])

    def test_log_selection_excludes_heldout_fold_and_abstains(self):
        y = np.tile([2, 3, 4], 6)
        folds = np.repeat([0, 2, 4], 6)
        X = np.zeros((len(y), 2, 3))
        X[:, :, 0] = np.arange(len(y))[:, None]
        X[:, 1, 1] = (y == 2).astype(float)
        def predict(train, yt, val):
            fit_ids, val_ids = train[:, 0].astype(int), val[:, 0].astype(int)
            self.assertFalse(set(folds[fit_ids]) & set(folds[val_ids]))
            return val[:, 1]
        with patch('scripts.audit_v273_log_frequency_guard.probability', side_effect=predict):
            chosen, _, _ = select_log_on_fit(X, y, folds)
        self.assertEqual(chosen, 1)
        with patch('scripts.audit_v273_log_frequency_guard.probability', side_effect=lambda tr, yt, va: np.ones(len(va))):
            chosen, _, _ = select_log_on_fit(X, y, folds)
        self.assertIsNone(chosen)

    def test_owned_combinations_are_one_to_one_and_restricted_cost_is_bounded(self):
        pool = np.array([100., 102., 200., 400.])
        expected = np.array([101., 200.])
        combos = valid_owned_combinations(expected, pool)
        self.assertEqual(combos, [(0, 2), (1, 2)])
        D = np.eye(4)
        x = np.array([.8, .2, 1., .1])
        restricted, _, x2 = best_restricted(D, x, combos)
        global_fit, _ = best_batch(D, x, 2)
        self.assertGreaterEqual(restricted[0] + 1e-12, global_fit[0])
        self.assertGreater(x2, 0)

    def test_integer_relations_and_synthetic_oracle_control(self):
        self.assertAlmostEqual(distance_to_integer_relation(300., 100., 'harmonic')[0], 0)
        self.assertEqual(distance_to_integer_relation(300., 100., 'harmonic')[1], 3)
        self.assertAlmostEqual(distance_to_integer_relation(50., 100., 'subharmonic')[0], 0)
        D = np.eye(4)
        pool = np.array([100., 200., 300., 400.])
        complete, residual = synthetic_control(D, (0, 2), np.array([100., 300.]), pool)
        self.assertTrue(complete)
        self.assertLess(residual, 1e-15)

    def test_fast_solver_matches_reference_and_hann_extracts(self):
        rng = np.random.default_rng(812)
        D = rng.random((90, 12)); D /= np.linalg.norm(D, axis=0)
        x0 = rng.random(90)
        for k in (2, 3):
            fast, x2 = best_fast(D, x0, k)
            reference, ref_x2 = best_batch(D, x0, k)
            self.assertAlmostEqual(x2, ref_x2, places=14)
            self.assertAlmostEqual(fast[0], reference[0], places=10)
            self.assertEqual(fast[1], reference[1])
        freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / 44100)
        freq = freq[(freq >= h.MIN_HZ) & (freq <= h.MAX_ANALYSIS_HZ)]
        x = h.template(freq, 121.4) + .4 * h.template(freq, 322.1) + .08 * h.template(freq, 754.8)
        x /= x.sum()
        tables = extraction_tables(freq)
        tables[2]['hann_t2'] = hann_dictionary(freq)
        values, pairs, triplets = extract_hann(freq, x, tables)
        self.assertEqual(values.shape, (3,))
        self.assertEqual(pairs.shape, (2,))
        self.assertEqual(triplets.shape, (3,))


if __name__ == '__main__':
    unittest.main()
