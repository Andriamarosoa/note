"""S690 test native original MIDI preservation and causal new-note filtering."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from scripts.s690_extra_note_event_quality import (
    new_intervals,accepted_merge,proposal_features,
    aggregate_guard,THRESHOLDS
)
def e(kind,pitch,frame):
    return dict(type=kind,pitch=pitch,decision_sample=frame*HOP,
                time=frame*HOP/SR)

class NativeAndAddedNoteTests(unittest.TestCase):
    def test_proposals_do_not_duplicate_original_pitch(self):
        original=[e("note_on",40,20),e("note_off",40,60)]
        proposed=original+[e("note_on",52,22),e("note_off",52,55)]
        out=new_intervals(original,proposed,np.arange(1,90)*HOP)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0][0]["pitch"],52)
        result=accepted_merge(original,out,np.array([.9]),.5)
        self.assertEqual(len(result),4)
        self.assertEqual([x for x in result if x["pitch"]==40],original)

    def test_same_pitch_overlap_guard(self):
        original=[e("note_on",40,10),e("note_off",40,45)]
        boosted=original+[e("note_on",40,22),e("note_off",40,33)]
        self.assertFalse(new_intervals(original,boosted,np.arange(1,100)*HOP))

    def test_native_chord_same_frame_order_untouched(self):
        original=[e("note_on",52,10),e("note_on",40,10),
                  e("note_off",52,31),e("note_off",40,32)]
        extras=[(e("note_on",64,12),e("note_off",64,25))]
        mix=accepted_merge(original,extras,np.array([.99]),.60)
        self.assertEqual([x for x in mix if x["pitch"] in (40,52)],original)
        self.assertEqual(sum(x["type"]=="note_on" for x in mix),3)

    def test_rejected_rescue_exact_original(self):
        original=[e("note_on",40,10),e("note_off",40,45)]
        extras=[(e("note_on",52,22),e("note_off",52,35))]
        self.assertEqual(accepted_merge(original,extras,np.array([.1]),.6),original)

    def test_features_never_see_future(self):
        rng=np.random.default_rng(690)
        n=100
        item=dict(ends=np.arange(1,n+1)*HOP,
                  feat=rng.random((n,49,126),dtype=np.float32),
                  p=rng.random((n,49,3),dtype=np.float32),
                  k=rng.random((n,7),dtype=np.float32))
        extra=[(e("note_on",52,45),e("note_off",52,50))]
        x=proposal_features(item,extra)
        item["feat"][46:]=4;item["p"][46:]=0;item["k"][46:]=0
        y=proposal_features(item,extra)
        np.testing.assert_array_equal(x,y)
        self.assertEqual(x.shape,(1,75))

    def test_precommitted_gates(self):
        self.assertEqual(THRESHOLDS[0],.10)
        self.assertIn(.65,THRESHOLDS)

if __name__=="__main__": unittest.main()
