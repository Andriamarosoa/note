"""S668 causality and decision boundary unit tests, no corpus required."""
import unittest
import numpy as np
from scripts.s668_causal_live_noteoff_hazard import (
    frame_matrix, note_instances, choose_offset,score,
)
from causal_note.strict_live import HOP,SR

class S668HazardTests(unittest.TestCase):
    def data(self):
        n=120
        rng=np.random.default_rng(668)
        features=rng.random((n,49,126),dtype=np.float32)
        p=np.zeros((n,49,3),np.float32)
        p[:,:,1]=.2;p[:,:,2]=.1
        k=np.zeros((n,7),np.float32);k[:,1]=1.
        ends=np.arange(1,n+1)*HOP
        return features,p,k,ends

    def test_frame_features_strict_no_future(self):
        x,p,k,ends=self.data()
        a=frame_matrix(x,p,k,ends,40,10,np.array([15,20]))
        x[30:]*=300
        p[30:]=1
        k[30:]=1
        b=frame_matrix(x,p,k,ends,40,10,np.array([15,20]))
        self.assertEqual(a.shape[0],2)
        np.testing.assert_array_equal(a,b)

    def test_native_noteon_and_noteoff_chain_preserved(self):
        _,_,_,ends=self.data()
        events=[
            dict(type="note_on",pitch=40,decision_sample=10*HOP),
            dict(type="note_off",pitch=40,decision_sample=26*HOP,
                 reason="causal_release"),
            dict(type="note_on",pitch=40,decision_sample=38*HOP),
        ]
        out=note_instances(events,ends)
        self.assertEqual(len(out),2)
        self.assertEqual(out[0]["native_end"],25)
        self.assertEqual(out[0]["next_start"],37)

    def test_batch4_emission_not_backdated(self):
        _,_,_,ends=self.data()
        note=dict(pitch=40,start=10,next_start=90,native_end=50)
        item=dict(notes=[note],ends=ends)
        frames=np.arange(20,88)
        probs=np.zeros(len(frames))
        probs[7:]=.99
        for batch in (1,4,8):
            t=choose_offset(item,0,frames,probs,.85,"hazard",batch_hops=batch)
            self.assertGreaterEqual(t,ends[27]/SR)
            self.assertGreaterEqual(t,0)

    def test_native_mode_requires_no_hazard(self):
        _,_,_,ends=self.data()
        note=dict(pitch=40,start=10,next_start=90,native_end=50)
        item=dict(notes=[note],ends=ends,
                  pairs=[(0,0)],
                  truth=[dict(time=10*HOP/SR,duration=40*HOP/SR)])
        frames=np.arange(20,90)
        prob=np.ones(len(frames),np.float32)
        native=choose_offset(item,0,frames,prob,0.3,"native",1)
        self.assertAlmostEqual(native,51*HOP/SR)
        measure=score(item,([frames],[prob]),.3,"native")
        self.assertEqual(measure["eligible"],1)

if __name__=="__main__":
    unittest.main()
