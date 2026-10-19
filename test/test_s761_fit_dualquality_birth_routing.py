"""S761 tests: FIT labels are distinct from strictly causal inference."""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP
from scripts.s761_fit_dualquality_birth_routing import (
    THRESHOLDS,GRID,candidate_quality_features,
    onset_labels,train_head,positive_probability
)
BLOCK=4*HOP

def event(kind,pitch,sample):
    return dict(type=kind,pitch=pitch,decision_sample=sample,
                available_sample=sample,time=sample/SR,
                reason="synthetic",velocity=90 if kind=="note_on" else 0)

class S761QualityTests(unittest.TestCase):
    def case(self):
        ends=np.arange(1,7,dtype=np.int64)*BLOCK
        item=dict(x=np.zeros((len(ends),49,288),np.float32),
                  native_post=np.zeros((len(ends),49,3),np.float32),
                  decision_ends=ends)
        post=np.zeros((len(ends),49,3),np.float32)
        k=np.ones((len(ends),7),np.float32)/7
        adjusted=k.copy()
        on=event("note_on",41,int(ends[2]))
        off=event("note_off",41,int(ends[5]))
        return item,post,k,adjusted,[(on,off)],np.array([.72],np.float32),[(2,1)]

    def test_12_expert_threshold_variants(self):
        self.assertEqual(THRESHOLDS,(.50,.70,.85))
        self.assertEqual(len(GRID),12)

    def test_feature_shape_and_routing(self):
        i,p,k,a,pairs,scores,indices=self.case()
        X=candidate_quality_features(i,p,k,a,pairs,scores,indices,3)
        self.assertEqual(X.shape,(1,37))
        self.assertAlmostEqual(float(X[0,-1]),1)
        self.assertEqual(float(X[0,-4]),0)

    def test_mutating_all_future_pcm_and_candidate_closure_keeps_birth(self):
        i,p,k,a,pairs,scores,indices=self.case()
        first=candidate_quality_features(i,p,k,a,pairs,scores,indices,0)
        i["x"][3:]=1000
        p[3:]=1000
        k[3:]=1000
        a[3:]=1000
        other=[(pairs[0][0],event("note_off",41,int(i["decision_ends"][4])))]
        second=candidate_quality_features(i,p,k,a,other,scores,indices,0)
        np.testing.assert_array_equal(first,second)

    def test_no_backdated_candidate(self):
        i,p,k,a,pairs,scores,indices=self.case()
        changed=[(dict(pairs[0][0],available_sample=pairs[0][0]["available_sample"]-1),
                  pairs[0][1])]
        with self.assertRaises(AssertionError):
            candidate_quality_features(i,p,k,a,changed,scores,indices,0)

    def test_fit_only_onset_supervision_excludes_signed_matched_truth(self):
        i,p,k,a,pairs,scores,indices=self.case()
        t=pairs[0][0]["time"]
        truth=[dict(pitch=41,time=t,duration=.2)]
        np.testing.assert_array_equal(onset_labels(pairs,truth,[]),[1])
        np.testing.assert_array_equal(onset_labels(pairs,truth,[0]),[0])

    def test_single_class_fallback_and_binary_probability(self):
        X=np.vstack([np.zeros(37),np.ones(37),np.full(37,2),np.full(37,3)])
        empty=positive_probability(None,X)
        np.testing.assert_array_equal(empty,np.zeros(4))
        only_negative=train_head(X,np.zeros(4,dtype=np.int32))
        np.testing.assert_array_equal(positive_probability(only_negative,X),
                                      np.zeros(4))
        mixed=train_head(X,np.array([0,0,1,1],np.int32))
        probabilities=positive_probability(mixed,X)
        self.assertEqual(probabilities.shape,(4,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.all((probabilities>=0)&(probabilities<=1)))

    def test_empty_candidates(self):
        i,p,k,a,pairs,scores,indices=self.case()
        X=candidate_quality_features(
            i,p,k,a,[],np.empty(0,np.float32),[],0)
        self.assertEqual(X.shape,(0,37))
        self.assertEqual(len(positive_probability(None,X)),0)

if __name__=="__main__":
    unittest.main()
