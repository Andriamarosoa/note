import unittest
import numpy as np
from scripts.s779_fit80_k_tail_specialist import causal_features,predicted,merge_counts,score,DIM,WEIGHTS
class S779KtailTests(unittest.TestCase):
    def test_31d_causal_features(self):
        n=6;post=np.zeros((n,49,3),np.float32)
        k=np.ones((n,7),np.float32)/7
        item=dict(x=np.zeros((n,49,288),np.float32),
                  decision_ends=np.arange(1,n+1)*1024)
        events=[]
        x=causal_features(item,post,k,events)
        self.assertEqual(x.shape,(n,DIM))
        self.assertTrue(np.isfinite(x).all())
        item2=dict(item);item2["x"]=item["x"].copy()
        item2["x"][4:]=1
        p2=post.copy();p2[4:]=1
        x2=causal_features(item2,p2,k,events)
        np.testing.assert_array_equal(x[:4],x2[:4])
    def test_no_k6_promotion_from_k0(self):
        class Fake:
            def predict(self,X):
                return np.array([6,6,5])
        k=np.eye(7,dtype=np.float32)[[0,2,4]]
        result=predicted(Fake(),np.zeros((3,DIM),np.float32),k)
        self.assertEqual(result.tolist(),[0,2,5])
    def test_metrics(self):
        records=[dict(gt=np.array([5,6,0]),trial=np.array([5,4,0]))]
        m=merge_counts(records,"trial")
        self.assertEqual(m["per_k"][5]["correct"],1)
        self.assertEqual(m["per_k"][6]["under"],1)
        self.assertGreaterEqual(len(WEIGHTS),3)
        self.assertEqual(len(score(m)),3)
if __name__=="__main__":
    unittest.main()
