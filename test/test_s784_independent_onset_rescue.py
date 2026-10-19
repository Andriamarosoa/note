"""S784 synthetic causal spectral rescue, FIT-only risk and safety."""
import unittest
import numpy as np
from scripts.s784_independent_spectral_onset_rescue import (
    acoustic_grid,propose,policy_grid,policy_key,choose_policies)

class S784Tests(unittest.TestCase):
    def fixture(self):
        n=7
        item=dict(x=np.zeros((n,49,288),np.float32),
                  decision_ends=np.arange(1,n+1,dtype=np.int64)*1024)
        post=np.zeros((n,49,3),np.float32)
        post[1,12,0]=.8;item["x"][1,12,192:216]=1
        post[5,12,0]=.9;item["x"][5,12,192:216]=2
        return item,post

    def test_48_policies(self):
        policies=policy_grid()
        self.assertEqual(len(policies),48)
        self.assertEqual(len({policy_key(x) for x in policies}),48)

    def test_one_past_aware_acoustic_proposal_and_future_isolation(self):
        item,post=self.fixture()
        policy=(.5,.3,0.,1.)
        a=propose(acoustic_grid(item,post),item["decision_ends"],[],policy)
        self.assertEqual([(r["pitch"],r["decision_sample"]) for r in a],
                         [(52,2048),(52,6144)])
        copy=dict(item,x=item["x"].copy())
        cp=post.copy();copy["x"][5:,:,:]=99;cp[5:,:,:]=0
        b=propose(acoustic_grid(copy,cp),item["decision_ends"],[],policy)
        self.assertEqual(a[0],b[0])

    def test_past_source_event_cooldown(self):
        item,post=self.fixture()
        source=[dict(type="note_on",pitch=52,available_sample=2048)]
        a=propose(acoustic_grid(item,post),item["decision_ends"],source,
                  (.5,.3,0.,1.))
        self.assertNotIn(2048,[x["available_sample"] for x in a])

    def test_harmonic_contrast_veto_without_using_future(self):
        item,post=self.fixture()
        item["x"][1,0,192:216]=3.
        signal=acoustic_grid(item,post)
        a=propose(signal,item["decision_ends"],[],(.5,.3,0.,1.))
        b=propose(signal,item["decision_ends"],[],(.5,.3,.4,1.))
        self.assertEqual(a[0]["decision_sample"],2048)
        self.assertNotIn(2048,[x["decision_sample"] for x in b])

    def test_fit_policy_selection_refuses_regression(self):
        groups=("A","B")
        baseline=[dict(composition=g,score=dict(TP=10,FP=2,FN=3)) for g in groups]
        bad=[dict(composition=g,score=dict(TP=10,FP=8,FN=3)) for g in groups]
        best,*_=choose_policies(groups,baseline,{"candidate":bad})
        self.assertIsNone(best)

if __name__=="__main__":
    unittest.main()
