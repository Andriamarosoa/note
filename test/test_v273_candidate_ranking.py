import unittest
from unittest.mock import patch

import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_candidate_ranking import trace_nms, trace_notes
from scripts.audit_v273_candidate_capacity_guard import extract, select_on_fit, NAMES
from scripts.audit_v273_harmonic_decay_guard import extraction_tables, extract_variants


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


if __name__ == '__main__':
    unittest.main()
