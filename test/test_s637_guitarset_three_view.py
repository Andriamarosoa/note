"""S637 controls preserve baseline, branch-K and strictly causal candidate veto."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP,Selection,CausalSelection,feature_track,load_heads
from scripts.s637_guitarset_three_view_groundtruth import (
    POLICIES,corrected,acoustic_batches,
)
from scripts.live_strict_notes import DEFAULT_MODEL


class GuitarSetThreeViewTests(unittest.TestCase):
    def test_control_native_probability_object_unchanged(self):
        cfg=Selection(onset=.6)
        state=CausalSelection(cfg)
        n=np.zeros((49,3),np.float32)
        other=np.full((49,3),.99,np.float32)
        k=np.zeros(7,np.float32);k[1]=1.
        out,num=corrected(n,other,other,k,state,POLICIES[0],True)
        self.assertIs(out,n)
        self.assertEqual(num,0)
        self.assertFalse(np.any(n))

    def test_bass_high_and_octave_veto_only_change_eligible_pitch(self):
        cfg=Selection(onset=.6)
        state=CausalSelection(cfg)
        n=np.zeros((49,3),np.float32)
        k=np.zeros(7,np.float32);k[2]=1.
        bass=np.zeros((49,3),np.float32);high=np.zeros_like(bass)
        bass[52-40,:2]=[.85,.86];high[80-40,:2]=[.86,.87]
        policy=("both",.8,.8,False,0.)
        out,count=corrected(n,bass,high,k,state,policy,True)
        self.assertEqual(count,2)
        self.assertAlmostEqual(float(out[52-40,0]),.6,places=6)
        self.assertAlmostEqual(float(out[80-40,0]),.6,places=6)
        self.assertEqual(np.count_nonzero(out[:,0]),2)
        np.testing.assert_array_equal(n,np.zeros_like(n))
        n[68-40,0]=.90 # octave below high MIDI 80
        veto=corrected(n,bass,high,k,state,("veto",.8,.8,True,0.),True)
        self.assertEqual(veto[1],1)
        self.assertAlmostEqual(float(veto[0][80-40,0]),0.,places=6)

    def test_native_k_veto_can_prevent_false_additional_note(self):
        cfg=Selection(onset=.6)
        state=CausalSelection(cfg)
        n=np.zeros((49,3),np.float32)
        bass=np.zeros((49,3),np.float32);bass[52-40,:2]=[.85,.85]
        low=np.zeros(7,np.float32);low[0]=.9;low[1]=.1
        high=np.zeros(7,np.float32);high[0]=.1;high[1]=.9
        policy=("K",.8,None,False,.5)
        self.assertEqual(corrected(n,bass,n,low,state,policy,True)[1],0)
        self.assertEqual(corrected(n,bass,n,high,state,policy,True)[1],1)
        self.assertEqual(corrected(n,bass,n,high,state,policy,False)[1],0)

    def test_real_frozen_guitarset_runner_reuses_S614_features(self):
        model=load_heads(DEFAULT_MODEL)
        t=np.arange(256*74)/SR
        pcm=np.asarray(.15*np.sin(2*np.pi*196*t),np.float32)
        ends,features=feature_track(pcm)
        p,k=model.predict(features)
        rows=list(acoustic_batches(pcm,model,frames_per_batch=16))
        self.assertEqual(len(rows),len(ends))
        for i in (0,31,63,73):
            self.assertEqual(rows[i][0],int(ends[i]))
            np.testing.assert_allclose(rows[i][1],p[i],atol=5e-6,rtol=1e-5)
            np.testing.assert_allclose(rows[i][2],k[i],atol=2e-6,rtol=1e-5)


if __name__=="__main__":
    unittest.main()
