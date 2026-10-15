from __future__ import annotations
import unittest
import numpy as np
from causal_note.s619_event_gate import noteon_event_features,relay_original_events,DIM
from causal_note.strict_live import HOP


class EventLevelGateTests(unittest.TestCase):
    def test_paired_noteon_noteoff_after_veto(self):
        x=[
          dict(type="note_on",pitch=48,decision_sample=HOP,time=.1),
          dict(type="note_on",pitch=60,decision_sample=HOP,time=.1),
          dict(type="note_off",pitch=60,decision_sample=2*HOP,time=.2),
          dict(type="note_on",pitch=60,decision_sample=2*HOP,time=.2),
          dict(type="note_off",pitch=48,decision_sample=3*HOP,time=.3),
          dict(type="note_off",pitch=60,decision_sample=3*HOP,time=.3)
        ]
        y=relay_original_events(x,[True,False,True])
        self.assertEqual([r["type"] for r in y],
                         ["note_on","note_on","note_off","note_off"])
        self.assertEqual([r["pitch"] for r in y],[48,60,48,60])
        self.assertEqual(relay_original_events(x,[True,True,True]),x)
        with self.assertRaises(ValueError):
            relay_original_events(x,[True])

    def test_event_feature_is_causal(self):
        rng=np.random.default_rng(619)
        x=rng.random((10,49,126)).astype(np.float32)
        p=rng.random((10,49,3)).astype(np.float32)
        k=np.broadcast_to(np.eye(7)[3],(10,7)).copy().astype(np.float32)
        events=[dict(type="note_on",pitch=52,decision_sample=5*HOP,time=.03),
                dict(type="note_off",pitch=52,decision_sample=6*HOP,time=.04)]
        mat,_=noteon_event_features(events,np.arange(1,11)*HOP,x,p,k)
        self.assertEqual(mat.shape,(1,DIM))
        z=x.copy();p2=p.copy()
        z[5:]*=17;p2[5:]=0
        mat2,_=noteon_event_features(events,np.arange(1,11)*HOP,z,p2,k)
        np.testing.assert_array_equal(mat,mat2)

if __name__=="__main__":unittest.main()
