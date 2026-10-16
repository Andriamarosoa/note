"""S625 exact NOTE MIDI-pitch/chord-union census, not just semitone gaps."""
import unittest

from scripts.s625_audit_exact_note_combinations import (
    summarize_source, aggregate, pitch_name,
)


def make(pitch,start,duration):
    return dict(pitch=pitch,time=start,duration=duration,track="guitar")


SPEC=dict(name="fixture",id="abc123",split="train",corpus="GuitarSet",
          composition="test_song",seconds=2.)


class ExactCombinationsTests(unittest.TestCase):
    def test_exact_e2_e3_is_one_distinct_pitch_pair_not_generic_octave(self):
        r=summarize_source(SPEC,[make(40,.1,1.),make(52,.12,.5)])
        self.assertEqual(r["pair_counts_e2_e3"],1)
        self.assertEqual(r["exact_pair_overlap_at_least_10ms"]["40,52"],1)
        self.assertEqual(r["onset_exact_combinations"]["40,52"],1)
        self.assertAlmostEqual(r["focus_activity_seconds"]["E2_plus_E3"],.5)
        self.assertEqual(r["focus_active_segments"]["E2_plus_E3"],1)
        self.assertEqual(r["onset_K2plus_by_K"]["2"],1)
        self.assertEqual(r["near_onset_pair_30ms"]["40,52"],1)

    def test_octave_at_other_notes_not_mistaken_for_E2_E3(self):
        r=summarize_source(SPEC,[make(41,.1,.7),make(53,.1,.6)])
        self.assertEqual(r["pair_counts_e2_e3"],0)
        self.assertEqual(r["exact_pair_overlap_at_least_10ms"]["41,53"],1)

    def test_staggered_e2_e3_counts_active_but_not_same_onset(self):
        r=summarize_source(SPEC,[make(40,.1,1),make(52,.3,.4)])
        self.assertEqual(r["pair_counts_e2_e3"],1)
        self.assertEqual(r["focus_onset_clusters"]["E2_plus_E3"],0)
        self.assertEqual(r["near_onset_pair_30ms"].get("40,52",0),0)
        self.assertAlmostEqual(r["focus_activity_seconds"]["E2_plus_E3"],.4)

    def test_major_F_chord_overlaid_minor_E_chord_vs_single_F(self):
        # F3-A3-C4 (53,57,60), E3-G3-B3 (52,55,59)
        full=[make(p,.2,.6) for p in (53,57,60,52,55,59)]
        r=summarize_source(SPEC,full)
        self.assertAlmostEqual(
            r["focus_activity_seconds"]["F_major_chord_plus_E_minor_chord"],.6)
        self.assertEqual(
            r["focus_onset_clusters"]["F_major_chord_plus_E_minor_chord"],1)
        self.assertEqual(r["overlapping_pair_total"],15)
        self.assertEqual(r["onset_K2plus_by_K"]["6"],1)
        only_F_note=[make(p,.2,.6) for p in (53,52,55,59)]
        q=summarize_source(SPEC,only_F_note)
        self.assertEqual(q["focus_activity_seconds"]["F_major_chord_plus_E_minor_chord"],0)
        self.assertAlmostEqual(q["focus_activity_seconds"]["F_note_plus_Em_chord"],.6)
        self.assertEqual(q["onset_K2plus_by_K"]["4"],1)

    def test_pitch_class_union_inversions_aggregate_but_exact_sets_differ(self):
        a=summarize_source(SPEC,[make(p,.2,.5) for p in (53,57,60,52,55,59)])
        other={**SPEC,"id":"zz","name":"other",
               "composition":"unrelated","split":"dev"}
        b=summarize_source(other,[make(p,.3,.4) for p in (65,69,72,64,67,71)])
        # Same pitch classes F A C E G B, but one octave apart.
        self.assertEqual(len(aggregate([a,b])["most_frequent_exact_multi_note_onset_sets"]),2)
        r=aggregate([a,b])
        self.assertEqual(
            r["most_dominant_active_pitch_class_sets_by_seconds"][0]["combination"],
            "C+E+F+G+A+B")
        self.assertAlmostEqual(
            r["focal_cases"]["F_major_chord_plus_E_minor_chord"]["active_simultaneously_seconds"],
            .9)

    def test_real_note_name_convention(self):
        self.assertEqual(pitch_name(40),"E2")
        self.assertEqual(pitch_name(52),"E3")
        self.assertEqual(pitch_name(53),"F3")
        self.assertEqual(pitch_name(60),"C4")


if __name__=="__main__":
    unittest.main()
