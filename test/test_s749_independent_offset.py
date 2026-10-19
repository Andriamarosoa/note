"""S749 independently adjustable causal extra-NoteOff without S708 edits."""
import unittest
import copy
import numpy as np

from causal_note.strict_live import SR,HOP
from causal_note.s707_causal_release import render_release
from scripts.s749_independent_offset_fit_cal_loops import OFFSETS, TRIALS

class OffsetSpecialistTests(unittest.TestCase):
    def fixture(self):
        ends=np.arange(1,45,dtype=np.int64)*HOP
        sample=int(ends[-1])
        on=dict(type="note_on",pitch=43,decision_sample=int(ends[3]),
                available_sample=int(ends[3]),time=float(ends[3]/SR),
                velocity=96,reason="s743_multifundamental")
        # Each specialist receives exactly the same fixed causal hazard
        # probabilities, never labels or a mutated S708 note.
        plan=dict(notes=[dict(pitch=43,start=3,next_start=len(ends),
                              native_end=None)],
            onsets=[on],frames=[np.array([8,12,16,20],np.int64)],
            probabilities=[np.array([.49,.65,.68,.84],np.float32)])
        return ends,sample,plan

    def test_three_release_experts_emit_causal_ends(self):
        ends,samples,plan=self.fixture()
        pristine=copy.deepcopy(plan)
        results={}
        for name,config in OFFSETS.items():
            midi,skip=render_release(plan,ends,samples,**config)
            self.assertEqual(skip,0)
            self.assertEqual([e["type"] for e in midi],
                             ["note_on","note_off"])
            self.assertEqual(midi[0]["pitch"],43)
            self.assertGreaterEqual(midi[1]["available_sample"],
                                    midi[1]["decision_sample"])
            self.assertEqual(midi[1]["time"],
                             midi[1]["available_sample"]/SR)
            self.assertGreater(midi[1]["available_sample"],
                               midi[0]["available_sample"])
            results[name]=midi[1]["decision_sample"]
        self.assertLess(results["sensitive"],results["persistent"])
        self.assertLess(results["persistent"],results["standard"])
        self.assertEqual(plan["onsets"],pristine["onsets"])
        np.testing.assert_array_equal(plan["probabilities"][0],
                                      pristine["probabilities"][0])

    def test_grid_and_source_release_remain_explicit(self):
        self.assertEqual(len(TRIALS),12)
        self.assertEqual(set(OFFSETS),{"sensitive","standard","persistent"})
        self.assertEqual(OFFSETS["standard"],dict(
            gate=.8,persistence=1,grace=24))
        for name,config in TRIALS.items():
            self.assertEqual(len(config),3)
            self.assertTrue(0 <= config[1].risk <= 1)
            self.assertIn(config[2],OFFSETS)

if __name__=="__main__":
    unittest.main()
