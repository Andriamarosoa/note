"""S783 causal temporal features, composition-maximin, K-prior guard."""
import unittest
import numpy as np
from scripts.s783_temporal_maximin_tail_fit80 import (
    causal_features,predicted,merge_counts,robust_score,DIM,K6_BIASES,WEIGHTS)

class S783TailTests(unittest.TestCase):
    def fixture(self,n=7):
        item=dict(x=np.zeros((n,49,288),np.float32),
                  decision_ends=(np.arange(n)+1)*1024)
        p=np.zeros((n,49,3),np.float32)
        k=np.ones((n,7),np.float32)/7
        return item,p,k

    def test_56d_features_and_no_future_leak(self):
        item,post,k=self.fixture()
        a=causal_features(item,post,k,[])
        self.assertEqual(a.shape,(7,DIM))
        self.assertTrue(np.isfinite(a).all())
        modified=dict(item,x=item["x"].copy())
        modified["x"][5:,:,:]=999.
        later=post.copy();later[5:,:,:]=1.
        counts=k.copy();counts[5:,:]=1.
        b=causal_features(modified,later,counts,[])
        np.testing.assert_array_equal(a[:5],b[:5])

    def test_temporal_history_uses_previous_not_future(self):
        item,p,k=self.fixture()
        k[1,6]=.99;k[2,6]=0.
        feature=causal_features(item,p,k,[])
        self.assertAlmostEqual(float(feature[2,31+6]),.99,places=5)
        self.assertAlmostEqual(float(feature[1,31+7+6]),.99-1/7,places=5)

    def test_tail_bias_causal_and_low_k_guard(self):
        class Fake:
            classes_=np.arange(7)
            def predict_proba(self,X):
                return np.array([[.8,.02,.02,.02,.02,.1,.02],
                                 [.01,.01,.01,.01,.01,.91,.04],
                                 [.01,.01,.01,.01,.01,.51,.44]],np.float32)
        k=np.eye(7,dtype=np.float32)[[0,4,5]]
        x=np.zeros((3,DIM),np.float32)
        a=predicted(Fake(),x,k,0.)
        b=predicted(Fake(),x,k,1.5)
        self.assertEqual(a.tolist(),[0,5,5])
        self.assertEqual(b[0],0)
        self.assertEqual(b[2],6)

    def test_composition_maximin_prefers_worst_k6(self):
        def metrics(k5,k6,acc):
            return dict(exact_accuracy=acc,
                per_k=[dict(accuracy=(k5 if k==5 else k6 if k==6 else .8))
                       for k in range(7)])
        a={"BN1":metrics(.9,.2,.86),"Rock3":metrics(.9,.96,.87)}
        b={"BN1":metrics(.7,.6,.84),"Rock3":metrics(.8,.7,.85)}
        self.assertGreater(robust_score(b,metrics(.7,.6,.84)),
                           robust_score(a,metrics(.9,.8,.87)))
        self.assertEqual(len(K6_BIASES),4)
        self.assertGreaterEqual(len(WEIGHTS),3)

if __name__=="__main__":
    unittest.main()
