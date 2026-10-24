"""S842 GT extraction/unit geometry; synthetic fixtures never become metrics."""
import unittest
import numpy as np
from scripts.s842_extract_real_octave_pairs256 import (
    SR,HOP,active_pitches,is_exactly,candidate_blocks_for_note,
    choose_windows_for_probe,split_of,GROUPS)
from scripts.s842_compare_neural_e2_e3_probe256 import safe_f1,pick_threshold


class S842GTTests(unittest.TestCase):
    def test_no_future_and_real_pcm256_grid(self):
        t=.1;off=.2
        windows=list(candidate_blocks_for_note(t,off))
        self.assertTrue(windows)
        for start,end in windows:
            self.assertEqual(end-start,256)
            self.assertGreaterEqual(start/SR,t)
            self.assertLessEqual(end/SR,off)

    def test_real_overlap_from_GT_durations_not_near_onsets(self):
        notes=[(.10,.30,40),(.14,.25,52)]
        pcm=np.zeros(int(SR*.5),np.int16)
        probe=choose_windows_for_probe(notes,pcm,max_each=50)
        joint=[x for x in probe if x["typ"]=="E2_E3_SUSTAIN"]
        self.assertTrue(joint)
        for j in joint:
            self.assertEqual(j["pcm"].shape,(256,))
            self.assertEqual(j["note_mask"][0],1.)
            self.assertEqual(j["note_mask"][12],1.)
        self.assertFalse(is_exactly(notes,.15,.155,{40}))
        self.assertEqual(active_pitches(notes,.20,.21),{40,52})

    def test_no_false_overlap_when_E3_begins_after_E2_ends(self):
        notes=[(.10,.125,40),(.16,.22,52)]
        pcm=np.zeros(int(SR*.5),np.int16)
        probe=choose_windows_for_probe(notes,pcm,max_each=50)
        self.assertFalse(any(x["typ"]=="E2_E3_SUSTAIN" for x in probe))

    def test_source_cal_outer_disjoint_by_composition(self):
        self.assertEqual(split_of("Jazz1"),"train")
        self.assertEqual(split_of("Rock3"),"cal")
        self.assertEqual(split_of("SS1"),"outer")
        self.assertFalse(set(GROUPS["train"])&set(GROUPS["cal"]))
        self.assertFalse(set(GROUPS["outer"])&set(GROUPS["train"]))

    def test_threshold_from_cal_only(self):
        prob=np.array([.1,.2,.75,.95])
        truth=np.array([False,False,True,True])
        cut=pick_threshold(prob,truth)
        self.assertGreater(cut,.2)
        result=safe_f1(prob>=cut,truth)
        self.assertEqual(result["tp"],2)
        self.assertEqual(result["fp"],0)

if __name__=="__main__":
    unittest.main()
