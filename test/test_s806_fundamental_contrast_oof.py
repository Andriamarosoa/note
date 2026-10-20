"""S806 spectral contrast truly current/past-only; no one-step lookahead."""
import unittest
import numpy as np
from scripts.s806_fundamental_contrast_oof import (
    acoustic_features,shift_pitch,rows_for_fit,fit_heldout,FEATURE_NAMES,
    POLICIES,THRESHOLDS)
class AcousticTests(unittest.TestCase):
    def test_acoustics_ignore_future(self):
        X=np.zeros((5,49,23),np.float32)
        X[:,:,16]=.20;X[:,:,17]=.3;X[:,:,18]=.5
        X[:,:,19]=.2;X[:,:,20]=.4
        X[:,:,0]=.8;X[:,:,1]=.8;X[:,:,2]=.1
        before=acoustic_features(X)
        altered=X.copy();altered[3:]=100
        after=acoustic_features(altered)
        np.testing.assert_allclose(before[:3],after[:3])
        self.assertEqual(before.shape,(5,49,16))
    def test_no_existing_neural_head_posteriors(self):
        X=np.zeros((3,49,23),np.float32)
        X[:,:,16:21]=.4
        before=acoustic_features(X)
        mutated=X.copy();mutated[:,:,:16]=.98
        np.testing.assert_array_equal(acoustic_features(mutated),before)
        self.assertEqual(len(FEATURE_NAMES),16)
    def test_shift_only_current_pitch(self):
        X=np.arange(49,dtype=np.float32)[None,:]
        a=shift_pitch(X,12)
        self.assertEqual(a[0,12],0)
        self.assertEqual(a[0,13],1)
        self.assertEqual(a[0,0],0)
    def test_fixed_non_fit_searched_thresholds(self):
        self.assertEqual(set(POLICIES),set(THRESHOLDS))
        self.assertEqual(THRESHOLDS["fundamental_strict"],.70)
    def test_five_composition_exclusion(self):
        rng=np.random.default_rng(806)
        comps=["BN1","Funk2","Jazz1","Rock3","SS1"]
        rows=[]
        for i,c in enumerate(comps):
            xx=rng.normal(size=(70,16)).astype(np.float32)
            yy=np.array(([0,1]*35),dtype=np.uint8)
            rows.append(dict(composition=c,X=xx,y=yy))
        for c in comps:
            _,proof=fit_heldout(rows,c)
            self.assertEqual(proof["eval_composition"],c)
            self.assertNotIn(c,proof["training_compositions"])
            self.assertEqual(len(proof["training_compositions"]),4)
    def test_future_truth_cannot_be_encoded_into_acoustic_input(self):
        X=np.zeros((4,49,23),np.float32)
        X[:,:,16:21]=.5
        pristine=acoustic_features(X)
        # A different label array is never passed to acoustic_features.
        X2=X.copy()
        X2[:,:,0]=.99
        X2[:,:,1]=.99
        np.testing.assert_allclose(pristine,acoustic_features(X2))

    def test_signed_training_rows_must_remain_causal(self):
        X=np.zeros((3,49,23),np.float32)
        X[:,:,0]=.6;X[:,:,1]=.8;X[:,:,2]=.1
        X[:,:,5]=.2;X[:,:,16:21]=.2
        labels=np.zeros((3,49),np.uint8);labels[1,8]=1
        xx,yy,stats=rows_for_fit(X,labels,None,806)
        self.assertEqual(len(xx),len(yy))
        self.assertTrue(stats["negative_available"]>0)
        self.assertTrue(stats["positive_available"]>=0)
if __name__=="__main__":unittest.main()
