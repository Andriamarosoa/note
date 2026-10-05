import unittest
from unittest.mock import patch
import numpy as np
from causal_note.guitarset_acoustics import RichNote
from scripts.audit_v273_internal_residual_acoustics import (
    ownership, match_frequencies, timing_metrics, select_controls,
)
from scripts import audit_v273_internal_b_low_harmonic_strata as h


class ResidualAcousticTests(unittest.TestCase):
    def test_equal_salience_keeps_lower_grid_frequency_before_nms(self):
        grid=np.array([100.,101.,200.,300.,400.,500.,600.,700.,800.])
        with patch.object(h,'F0_GRID',grid), patch.object(h,'harmonic_salience',return_value=1.0):
            pool=h.f0_pool(np.array([1.]),np.array([1.]))
        self.assertEqual(pool,[100.,200.,300.,400.,500.,600.,700.,800.])

    def test_ownership_uses_all_candidates_ties_and_exact_radius(self):
        groups = [np.array([1000, 2000]), np.array([3000])]
        assigned, exact = ownership(groups, [118, 117, 2500, 3000, 3882, 3883])
        np.testing.assert_array_equal(assigned, [0, -1, 0, 1, 1, -1])
        np.testing.assert_array_equal(exact, [2, 2])

    def test_unique_frequency_capacity_and_tolerance(self):
        self.assertEqual(match_frequencies([100, 100], [100])[0], 1)
        self.assertEqual(match_frequencies([100, 200, 300], [300, 100, 200])[0], 3)
        self.assertEqual(match_frequencies([100], [100 * 2 ** (54 / 1200)])[0], 1)
        self.assertEqual(match_frequencies([100], [100 * 2 ** (56 / 1200)])[0], 0)
        self.assertEqual(match_frequencies([100], [])[0], 0)

    def test_timing_separates_owned_late_notes_from_foreign_resonance(self):
        notes = [RichNote(0, 900, 4000, 48), RichNote(1, 2000, 5000, 52),
                 RichNote(2, 3200, 6000, 55), RichNote(3, 0, 1500, 43)]
        metrics, owned, events, _, _ = timing_metrics(notes, np.array([0, 0, 0, -1]), 0, 1000)
        self.assertEqual(metrics['owned_before_start'], 1)
        self.assertEqual(metrics['owned_onset_after_post'], 1)
        self.assertEqual(metrics['owned_not_overlapping_post'], 1)
        self.assertEqual(metrics['foreign_post_notes'], 1)
        self.assertEqual(metrics['foreign_post_births'], 0)
        self.assertEqual(metrics['min_post_cycles'], 0)
        self.assertEqual(owned[2]['post_taper_energy_coverage'], 0)
        self.assertEqual(len(events), 3)

    def test_controls_match_recording_and_pitch_before_time(self):
        def row(i, group, member, start, pitches):
            return dict(row_id=i, group=group, recording_id=member, start_sample=start,
                        owned_notes=[dict(midi=p) for p in pitches])
        rows = [row(1, 'K3_regressed', 'a', 1000, [48, 52, 55]),
                row(2, 'K3_preserved', 'b', 1000, [48, 52, 55]),
                row(3, 'K3_preserved', 'a', 1000, [60, 64, 67]),
                row(4, 'K3_preserved', 'a', 9000, [48, 52, 55]),
                row(5, 'K3_regressed', 'c', 1000, [48, 52, 55])]
        pairs = select_controls(rows)
        self.assertEqual(pairs[0]['control_row_id'], 4)
        self.assertIsNone(pairs[1]['control_row_id'])


if __name__ == '__main__':
    unittest.main()
