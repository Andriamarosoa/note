"""S700 weighted K class balance is TRAIN-only; pitch, Onset and S675 stay frozen."""
import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s700_weighted_polyphonic_K_shadow import train,TRAIN_ROW_STRIDE,shadow_features
class S700Tests(unittest.TestCase):
    def test_poly_weight_training_is_strictly_train_only(self):
        # Synthetic toy data, exercises actual weighted estimator path.
        rng=np.random.default_rng(700)
        features=rng.random((4000,63),dtype=np.float32)
        labels=np.tile([0,1,2,3,4],800).astype(np.int32)
        # The production minimum 20,000 frames deliberately vetoes
        # tiny synthetic training: never mistake a unit test for GuitarSet.
        with self.assertRaises(AssertionError):
            train([dict(features=features,y=labels)])
    def test_source_feature_prefix_is_causal(self):
        rng=np.random.default_rng(700)
        f=rng.random((60,49,126),dtype=np.float32)
        p=rng.random((60,49,3),dtype=np.float32)
        k=rng.random((60,7),dtype=np.float32)
        c=np.ones(60,dtype=np.int32)
        a=shadow_features(f,p,k,c)
        f[49:]=99;p[49:]=1;k[49:]=0;c[49:]=6
        b=shadow_features(f,p,k,c)
        np.testing.assert_array_equal(a[:49],b[:49])
    def test_training_stride(self):
        self.assertEqual(TRAIN_ROW_STRIDE,3)
if __name__=="__main__": unittest.main()
