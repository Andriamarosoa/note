"""S744 independent FIT learner and causal acoustic selector tests."""
import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from causal_note.strict_live import HOP
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s744_birth_specialist import (
    frozen_source_activity,birth_features,select_fit_examples,
    trained_head_proposals,boost_fit_weights)
from scripts.s744_train_fit_birth_loops import CHOICES,MODELS


class S744Tests(unittest.TestCase):
    def setUp(self):
        self.n=5
        self.ends=np.arange(1,6)*4*HOP
        self.p=np.zeros((self.n,49,3),np.float32)
        self.native=self.p.copy()
        self.ctx=np.zeros((self.n,49,16),np.float32)
        self.raw=np.zeros((self.n,49,288),np.float32)
        self.k=np.zeros((self.n,7),np.float32)
        self.k[:,0]=.45
        self.k[:,1]=.40
        self.k[:,2]=.15
        self.policy=GuidedPolicy(count_confidence=1.1)
        self.h=np.repeat(self.p[:,:,0,None],6,axis=-1)

    def test_features_from_current_and_past_only(self):
        active=frozen_source_activity(
            self.p,self.k,self.ends,self.policy)
        x=birth_features(self.p,self.native,self.ctx,self.raw,self.k,active)
        self.assertEqual(x.shape[:2],(self.n,49))
        self.assertGreater(x.shape[-1],50)
        altered=self.raw.copy()
        altered[-1,:,:]=100
        later=birth_features(self.p,self.native,self.ctx,altered,self.k,active)
        np.testing.assert_array_equal(later[:-1],x[:-1])

    def test_sampling_true_onsets_only_in_fit_and_keeps_negative_control(self):
        state=np.zeros((self.n,49),bool)
        feat=birth_features(self.p,self.native,self.ctx,self.raw,self.k,state)
        desired=np.zeros((self.n,49),bool)
        desired[1,0]=True;desired[2,1]=True
        features,y,counts=select_fit_examples(
            feat,desired,state,self.ctx,self.h,seed=744,limit=100)
        self.assertEqual(counts["all_true_onsets"],2)
        self.assertEqual(counts["true_positive_candidates"],2)
        self.assertEqual(int(y.sum()),2)
        self.assertTrue(len(y)>int(y.sum()))
        self.assertEqual(len(y),len(features))
        self.assertGreater(boost_fit_weights(y,5.).max(),1.)

    def test_original_already_active_note_cannot_be_added_again(self):
        source=np.zeros((self.n,49),bool)
        source[0,0]=True
        feat=birth_features(self.p,self.native,self.ctx,self.raw,self.k,source)
        wanted=np.zeros((self.n,49),bool)
        wanted[0,0]=True
        _,y,counts=select_fit_examples(
            feat,wanted,source,self.ctx,self.h,seed=1,limit=100)
        self.assertEqual(int(y.sum()),0)
        self.assertEqual(counts["true_positive_candidates"],0)

    def test_fit_classifier_has_a_distinct_probability_head_while_h0_is_frozen(self):
        state=np.zeros((self.n,49),bool)
        self.raw[:,:,:]+=np.arange(self.n)[:,None,None]*.01
        feat=birth_features(self.p,self.native,self.ctx,self.raw,self.k,state)
        x=feat.reshape(-1,feat.shape[-1])
        y=np.zeros(len(x),dtype=np.int32)
        y[::3]=1
        model=HistGradientBoostingClassifier(
            max_iter=4,learning_rate=.1,random_state=744,
            min_samples_leaf=5,early_stopping=False)
        model.fit(x,y,sample_weight=boost_fit_weights(y,3.))
        updated=trained_head_proposals(model,self.h,feat)
        self.assertEqual(updated.shape,self.h.shape)
        np.testing.assert_array_equal(updated[:,:,0],self.h[:,:,0])
        self.assertTrue(np.isfinite(updated).all())
        self.assertTrue((updated[:,:,5]>=self.h[:,:,0]).all())

    def test_only_8_real_prespecified_loops_no_student(self):
        self.assertEqual(len(MODELS),2)
        self.assertEqual(len(CHOICES),8)
        self.assertTrue(all(mode.mask_strategy=="conservative"
                            for _,mode in CHOICES.values()))


if __name__=="__main__":
    unittest.main()
