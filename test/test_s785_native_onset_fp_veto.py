"""S785 causal FP-veto event semantics and FIT-only score selection."""
import unittest
import numpy as np
from scripts.s785_native_onset_fp_veto import (
    event_features,select_events,robust_selection,policy_key,CS,CUTS,DIM)

class S785Tests(unittest.TestCase):
    def fixture(self):
        n=7
        item=dict(x=np.zeros((n,49,288),np.float32),
                  decision_ends=np.arange(1,n+1,dtype=np.int64)*1024)
        post=np.zeros((n,49,3),np.float32)
        count=np.eye(7,dtype=np.float32)[[1]*n]
        source=[
            dict(type="note_on",pitch=52,available_sample=2048,velocity=80,time=2048/44100),
            dict(type="note_off",pitch=52,available_sample=4096,time=4096/44100),
            dict(type="note_on",pitch=40,available_sample=5120,velocity=100,time=5120/44100)]
        post[1,12,0]=.7;item["x"][1,12,192:216]=1.2
        return item,post,count,source

    def test_features_20_and_no_future_lookahead(self):
        item,p,k,source=self.fixture()
        x=event_features(item,p,k,source)
        self.assertEqual(x.shape,(2,DIM))
        modified=dict(item,x=item["x"].copy())
        modified["x"][3:]=999.
        z=p.copy();z[3:]=999.
        k2=k.copy();k2[3:]=999.
        b=event_features(modified,z,k2,source)
        np.testing.assert_array_equal(x[0],b[0])
        self.assertEqual(x[0,18],12/48.)
        self.assertEqual(x[1,18],0.)

    def test_veto_shadow_keeps_noteoff_and_respects_event_order(self):
        _,_,_,source=self.fixture()
        shadow=select_events(source,np.array([.1,.9]),.5)
        self.assertEqual([(e["type"],e["pitch"]) for e in shadow],
                         [("note_off",52),("note_on",40)])
        self.assertEqual(len(source),3)

    def test_noop_allowed_if_veto_regresses_fit(self):
        rows=[dict(composition=c,score=dict(TP=12,FP=4,FN=2))
              for c in ("A","B")]
        worse=[dict(composition=c,score=dict(TP=8,FP=1,FN=6))
              for c in ("A","B")]
        chosen,*_=robust_selection(rows,{"bad":worse},["A","B"])
        self.assertIsNone(chosen)
        self.assertEqual(len(CS)*len(CUTS),15)
        self.assertNotEqual(policy_key(.1,.08),policy_key(1.,.08))

if __name__=="__main__":
    unittest.main()
