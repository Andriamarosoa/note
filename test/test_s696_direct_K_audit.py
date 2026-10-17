"""S696 preserve frozen NOTE K and source MIDI while measuring raw K."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from scripts.s696_direct_K_causal_audit import (
    predictions,causal_vote,posterior_ema,
    native_count_timeline,gt_count_timeline
)
class S696NativeKTests(unittest.TestCase):
    def test_future_k_cannot_change_current_decision(self):
        k=np.zeros((110,7),np.float32)
        k[:,1]=1.
        midi=np.ones(110,dtype=np.int32)
        before=predictions(k,midi)
        k[80:]=0;k[80:,3]=1.
        after=predictions(k,midi)
        for name in before:
            np.testing.assert_array_equal(before[name][:79],after[name][:79])
    def test_last_received_frame_only(self):
        k=np.zeros((12,7),np.float32)
        k[:,1]=1;k[5:,1]=0;k[5:,2]=1
        self.assertEqual(causal_vote(k,3)[4],1)
        self.assertEqual(causal_vote(k,3)[7],2)
        self.assertEqual(posterior_ema(k,.5).argmax(axis=1)[4],1)
    def test_count_boundaries_are_causal(self):
        end=np.arange(1,12)*HOP
        events=[
            dict(type="note_on",pitch=40,decision_sample=2*HOP),
            dict(type="note_on",pitch=52,decision_sample=4*HOP),
            dict(type="note_off",pitch=52,decision_sample=6*HOP),
            dict(type="note_off",pitch=40,decision_sample=7*HOP)]
        expected=np.array([0,1,1,2,2,1,0,0,0,0,0])
        np.testing.assert_array_equal(native_count_timeline(events,end),expected)
    def test_real_gt_independent_of_midi(self):
        end=np.arange(1,12)*HOP
        truth=[dict(pitch=40,time=2*HOP/SR,duration=4*HOP/SR)]
        expected=np.array([0,1,1,1,1,0,0,0,0,0,0])
        np.testing.assert_array_equal(gt_count_timeline(truth,end),expected)
if __name__=="__main__":unittest.main()
