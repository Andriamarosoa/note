"""S790 causal 49-pitch head and robust F1 geometry tests."""
import unittest
import numpy as np
from scripts.s790_independent_framepitch_expert import (
    causal_features,occupancy_from_emitted,counts,summarize,
    sample_training,evaluate_policy,DIM,CUTS,HARM_PENALTIES)

class S790FramePitchTests(unittest.TestCase):
    def fixture(self,n=6):
        ends=np.arange(1,n+1)*1024
        item=dict(x=np.zeros((n,49,288),np.float32),
                  decision_ends=ends)
        post=np.zeros((n,49,3),np.float32)
        k=np.ones((n,7),np.float32)/7
        return item,post,k

    def test_event_occupancy_matches_s704_frame_convention(self):
        ends=np.array([1024,2048,3072,4096],np.int64)
        ev=[dict(type="note_on",pitch=40,available_sample=1024),
            dict(type="note_off",pitch=40,available_sample=3072)]
        actual=occupancy_from_emitted(ev,ends)
        self.assertEqual(actual[:,0].tolist(),[True,True,False,False])
        self.assertFalse(actual[:,1:].any())

    def test_feature_shape_pitches_and_harmonic_masking(self):
        item,p,k=self.fixture()
        p[2,0,1]=.8;p[2,12,1]=.7
        x,source,harm,sust=causal_features(item,p,k,[])
        self.assertEqual(x.shape,(6,49,DIM))
        self.assertEqual(source.shape,(6,49))
        self.assertGreater(harm[2,12],.79)
        self.assertEqual(harm[2,0],0)
        self.assertTrue(np.isfinite(x).all())

    def test_no_future_leakage(self):
        item,p,k=self.fixture()
        left,h1,_,_=causal_features(item,p,k,[])
        item2=dict(item,x=item["x"].copy())
        item2["x"][3:,:,:]=5
        p2=p.copy();p2[3:,:,:]=.8
        k2=k.copy();k2[3:,:]=.1
        right,h2,_,_=causal_features(item2,p2,k2,[])
        np.testing.assert_array_equal(left[:3],right[:3])
        np.testing.assert_array_equal(h1[:3],h2[:3])

    def test_dynamic_gate_varies_by_pitch_and_fragment(self):
        prob=np.array([[.55,.55],[.55,.55]])
        base=np.zeros((2,2),dtype=bool)
        harm=np.array([[0,.9],[.9,0]],dtype=float)
        sustain=np.ones((2,2))
        out=evaluate_policy(prob,base,harm,sustain,.50,.10)
        self.assertEqual(out.tolist(),[[True,False],[False,True]])

    def test_sampling_is_finite_and_reproducible(self):
        X=np.zeros((80,49,DIM),dtype=np.float32)
        y=np.zeros((80,49),bool);y[:30,:3]=True
        p=np.zeros((80,49,3),dtype=np.float32)
        out1=sample_training(X,y,p,23)
        out2=sample_training(X,y,p,23)
        self.assertEqual(out1[0].shape[1],DIM)
        self.assertEqual(out1[0].shape[0],MAX_SAMPLE)
        np.testing.assert_array_equal(out1[0],out2[0])
        np.testing.assert_array_equal(out1[1],out2[1])

    def test_counts_and_f1_from_matching_grid(self):
        truth=np.array([[True,False],[True,True]])
        trial=np.array([[True,True],[False,True]])
        c=counts(trial,truth)
        self.assertEqual(c,dict(tp=2,fp=1,fn=1))
        score=summarize([dict(stats=c)],"stats")
        self.assertAlmostEqual(score["f1"],2/3)

MAX_SAMPLE=1600

if __name__=="__main__":
    unittest.main()
