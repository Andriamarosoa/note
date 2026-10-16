"""Definition tests for TRUE annotation overlap, not predicted Exact-K."""
import unittest

from scripts.s624_audit_real_overlap import analyze_source, summarize


SPEC=dict(name="fixture",split="train",corpus="GuitarSet",
          composition="fixture_group",seconds=1.0)


def note(pitch,time,duration,track="guitar"):
    return dict(pitch=pitch,time=time,duration=duration,track=track)


class OverlapAuditTests(unittest.TestCase):
    def test_two_staggered_octaves_not_confused_with_same_onset(self):
        rows=[note(40,.10,.60),note(52,.12,.35)]
        d=analyze_source(SPEC,rows)
        self.assertEqual(d["notes_in_pitch_range"],2)
        self.assertEqual(d["onset_groups_30ms"],{"2":1})
        self.assertAlmostEqual(d["duration_active_poly_K_ge2_s"],.35)
        self.assertEqual(d["overlapping_distinct_pitch_note_pairs_at_least_10ms"],1)
        self.assertEqual(d["overlapping_pair_categories"]["octave_12"],1)
        self.assertEqual(d["overlapping_pair_categories"]["harmonic_2x_3x_4x_intervals"],1)
        self.assertEqual(d["note_attacks_while_a_different_note_still_sustains"],1)
        self.assertEqual(d["onset_pairs_while_first_still_active_under_ms"]["30"],1)

    def test_separate_simultaneous_K_and_onset_K(self):
        rows=[note(40,.1,.6),note(52,.4,.4),note(53,.49,.3)]
        d=analyze_source(SPEC,rows)
        self.assertEqual(d["onset_groups_30ms"],{"1":3})
        self.assertEqual(d["peak_simultaneously_active_distinct_pitches"],3)
        self.assertGreater(d["duration_active_poly_K_ge2_s"],.0)
        self.assertEqual(d["overlapping_pair_categories"]["close_1_or_2_semitones"],1)

    def test_identical_pitch_is_not_two_distinct_notes_active(self):
        rows=[note(40,.1,.5),note(40,.2,.3)]
        d=analyze_source(SPEC,rows)
        self.assertEqual(d["peak_simultaneously_active_distinct_pitches"],1)
        self.assertEqual(d["overlapping_distinct_pitch_note_pairs_at_least_10ms"],0)
        self.assertEqual(d["same_pitch_overlap_pairs"],1)

    def test_outside_supported_pitch_and_duration_reported(self):
        d=analyze_source(SPEC,[note(39,.1,.2),note(40,1.5,.1),note(40,.2,.5)])
        self.assertEqual(d["original_annotated"],3)
        self.assertEqual(d["excluded_outside_midi_range"],1)
        self.assertEqual(d["excluded_invalid_or_outside_duration"],1)
        self.assertEqual(d["notes_in_pitch_range"],1)

    def test_aggregation_is_pair_count_not_chord_count(self):
        d=analyze_source(SPEC,[note(40,.10,.4),note(52,.10,.4),note(64,.10,.4)])
        self.assertEqual(d["overlapping_distinct_pitch_note_pairs_at_least_10ms"],3)
        self.assertEqual(d["onset_groups_30ms"],{"3":1})
        agg=summarize([d,d])
        self.assertEqual(agg["raw"]["overlapping_distinct_pitch_note_pairs_at_least_10ms"],6)
        self.assertEqual(agg["onset_groups_30ms_by_unique_pitch_K"]["3"],2)


if __name__=="__main__":
    unittest.main()
