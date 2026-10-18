"""S728 overlap matching contracts on actual emitted notes, no fabricated successes."""
import unittest
from scripts.s728_real_octave_overlap_audit import (
    has_earlier_parent,overlap_results)


class S728OctaveAuditTests(unittest.TestCase):
    def test_E3_over_E2_counts_independent_true_and_spurious(self):
        gt=[dict(pitch=40,time=0.,duration=1.),
            dict(pitch=52,time=.24,duration=.26)]
        ev=[dict(pitch=40,time=0.,type="note_on"),
            dict(pitch=52,time=.25,type="note_on"),
            dict(pitch=52,time=.55,type="note_on")]
        s=overlap_results(ev,gt,12)
        self.assertEqual(s["observed_real_targets"],1)
        self.assertEqual(s["correct_real_onsets"],1)
        self.assertEqual(s["missed_real_onsets"],0)
        self.assertEqual(s["spurious_harmonic_child_onsets"],1)

    def test_wrong_pitch_does_not_match_real_octave(self):
        gt=[dict(pitch=40,time=0.,duration=1.),
            dict(pitch=52,time=.25,duration=.20)]
        ev=[dict(pitch=53,time=.25,type="note_on")]
        s=overlap_results(ev,gt,12)
        self.assertEqual(s["missed_real_onsets"],1)
        self.assertEqual(s["correct_real_onsets"],0)
        self.assertEqual(s["spurious_harmonic_child_onsets"],0)

    def test_no_preexisting_parent_then_no_target(self):
        gt=[dict(pitch=40,time=.20,duration=.10),
            dict(pitch=52,time=.20,duration=.30)]
        self.assertFalse(has_earlier_parent(gt,52,.20,12))
        self.assertEqual(overlap_results([],gt,12)["observed_real_targets"],0)

    def test_fifth_19_interval_is_distinct_from_octave_12(self):
        gt=[dict(pitch=40,time=0.,duration=1.),
            dict(pitch=59,time=.30,duration=.2)]
        self.assertEqual(overlap_results([],gt,19)["observed_real_targets"],1)
        self.assertEqual(overlap_results([],gt,12)["observed_real_targets"],0)


if __name__=="__main__":
    unittest.main()
