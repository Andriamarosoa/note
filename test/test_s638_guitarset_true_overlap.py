"""S638 overlap score correctness on known annotated polyphonic chords."""
import unittest

from scripts.s638_guitarset_true_overlap import overlap_groups,coverage,truth_active_pitch_k


def note(pitch,start,duration):
    return dict(pitch=pitch,time=start,duration=duration)


class TrueOverlapTests(unittest.TestCase):
    def test_sustained_overlap_is_not_necessarily_same_chord_onset(self):
        truth=[note(52,0.,1.),note(59,.4,.4),note(65,1.5,.5),
               note(69,1.518,.4),note(71,3.,.2)]
        got=overlap_groups(truth)
        self.assertEqual(got["any_overlap"],set([0,1,2,3]))
        self.assertEqual(got["never_overlap"],set([4]))
        self.assertEqual(got["onset_overlap"],set([1,2,3]))
        self.assertEqual(got["onset_single"],set([0,4]))
        self.assertEqual(got["onset_K2"],set([1,2,3]))
        self.assertFalse(got["onset_K3plus"])

    def test_three_note_chord_and_same_pitch_unison(self):
        truth=[note(60,0,1),note(60,.004,.9),note(64,.010,.5),note(67,.015,.5)]
        got=overlap_groups(truth)
        self.assertEqual(got["onset_K3plus"],set(range(4)))
        self.assertEqual(got["onset_overlap"],set(range(4)))

    def test_new_and_lost_matched_notes_for_overlapping_stratum(self):
        truth=[note(52,0.,1.),note(59,.4,.4),note(65,1.5,.5),
               note(69,1.518,.4),note(71,3.,.2)]
        group=overlap_groups(truth)
        # Baseline matched note0 and note2. Rescuer loses note0,
        # picks up note1 and note3. Note4 is missed throughout.
        events=[dict(type="note_on",pitch=59,time=.4),
                dict(type="note_on",pitch=65,time=1.5),
                dict(type="note_on",pitch=69,time=1.518),
                dict(type="note_on",pitch=52,time=1.6)] # unmatched false positive
        matches=[(0,1),(1,2),(2,3)]
        baseline_pairs=[(0,0),(1,2)]
        g,fp=coverage(truth,events,matches,group,baseline_pairs)
        self.assertEqual(g["onset_overlap"]["true_notes"],3)
        self.assertEqual(g["onset_overlap"]["recovered"],3)
        self.assertEqual(g["onset_overlap"]["additional_correct_vs_S614"],2)
        self.assertEqual(g["onset_overlap"]["lost_correct_vs_S614"],0)
        self.assertEqual(g["onset_single"]["lost_correct_vs_S614"],1)
        self.assertEqual(fp["false_positives"],1)
        self.assertEqual(fp["FP_in_true_polyphonic_passage"],1)

    def test_unmatched_predicted_context_is_groundtruth_only(self):
        truth=[note(60,1.,.5),note(64,1.03,.3)]
        self.assertEqual(truth_active_pitch_k(truth,1.07),2)
        self.assertEqual(truth_active_pitch_k(truth,0.5),0)
        with self.assertRaises(ValueError):
            overlap_groups(truth,min_shared_s=-1)


if __name__=="__main__":
    unittest.main()
