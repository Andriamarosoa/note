"""S685 event-veto keeps native pitch/K and preserves accepted NoteOff pairs."""
import unittest
import numpy as np
from causal_note.strict_live import HOP
from scripts.s685_causal_onset_event_veto import (
    filter_native_events,candidate_rows,guarded_choice,GATES
)

class S685EventVetoTests(unittest.TestCase):
    def test_suppress_false_note_clears_corresponding_release(self):
        events=[
            dict(type="note_on",pitch=40,decision_sample=HOP,time=HOP/44100),
            dict(type="note_on",pitch=52,decision_sample=2*HOP,time=2*HOP/44100),
            dict(type="note_off",pitch=52,decision_sample=8*HOP,time=8*HOP/44100),
            dict(type="note_off",pitch=40,decision_sample=9*HOP,time=9*HOP/44100),
        ]
        kept=filter_native_events(events,np.array([True,False]))
        self.assertEqual([e["pitch"] for e in kept],[40,40])
        self.assertEqual([e["type"] for e in kept],["note_on","note_off"])
        self.assertIs(events[0],kept[0])
    def test_accept_everything_is_exact_identity(self):
        events=[
            dict(type="note_on",pitch=40,time=.01),
            dict(type="note_off",pitch=40,time=.05),
            dict(type="note_on",pitch=52,time=.06),
            dict(type="note_off",pitch=52,time=.08),
        ]
        result=filter_native_events(events,np.ones(2,bool))
        self.assertEqual(result,events)
    def test_ranker_receives_only_current_and_past(self):
        n=95; rng=np.random.default_rng(685)
        feat=rng.random((n,49,126),dtype=np.float32)
        p=rng.random((n,49,3),dtype=np.float32)
        k=np.zeros((n,7),np.float32);k[:,1]=1
        end=np.arange(1,n+1)*HOP
        events=[dict(type="note_on",pitch=40,decision_sample=25*HOP),
                dict(type="note_on",pitch=52,decision_sample=50*HOP)]
        x=candidate_rows(feat,p,k,events,end)
        feat[51:]=100;p[51:]=100;k[51:]=100
        y=candidate_rows(feat,p,k,events,end)
        np.testing.assert_array_equal(x,y)
        self.assertEqual(x.shape,(2,75))
    def test_gates_precommitted(self):
        self.assertEqual(GATES[0],.08)
        self.assertIn(.42,GATES)
if __name__=="__main__":unittest.main()
