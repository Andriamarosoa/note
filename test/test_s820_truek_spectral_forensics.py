"""S820 read-only 7x7x7 GT-conditioned transitions: synthetic contract."""
import unittest
import numpy as np
from scripts.s820_truek_spectral_forensics import (
    SCHEMA, THRESHOLD, sharp_proxies, record_counts, true_k_analysis
)

class S820TruthTransitionTest(unittest.TestCase):
    def test_exact_correctness_conservation_and_true_k_count(self):
        truth=np.array([0,1,2,3,4,5,6,2,3,6],dtype=np.uint8)
        base=np.array( [0,1,1,3,4,3,5,2,4,6],dtype=np.uint8)
        new=np.array(  [4,4,4,4,4,5,6,6,6,6],dtype=np.uint8)
        # only increasing corrections, never decrease; test includes K0.
        high=np.maximum(new,4).astype(np.uint8)
        X=np.ones((len(truth),49,222),dtype=np.float16)
        matrix,counters,per_k,proxies=record_counts(
            truth,base,new,np.full(len(truth),.9,dtype=np.float32),high,X)
        self.assertEqual(matrix.shape,(7,7,7))
        self.assertEqual(matrix.sum(),len(truth))
        self.assertEqual(sum(x["total"] for x in per_k.values()),len(truth))
        self.assertEqual(counters["became_correct"]-counters["became_incorrect"],
            int(np.count_nonzero(new==truth)-np.count_nonzero(base==truth)))
        self.assertEqual(counters["changed_all"],int(np.count_nonzero(new!=base)))
        s=true_k_analysis(matrix)
        self.assertEqual(s["0"]["n"],1)
        self.assertEqual(s["6"]["n"],2)
        self.assertEqual(s["0"]["changed"],1)
        self.assertEqual(sum(x["n"] for x in s.values()),len(truth))
        self.assertEqual(proxies["changed_all"]["n"],counters["changed_all"])

    def test_decreasing_prediction_fails_closed(self):
        X=np.zeros((2,49,222),dtype=np.float16)
        with self.assertRaisesRegex(AssertionError,"subtract"):
            record_counts(np.array([1,2]),np.array([4,3]),np.array([3,5]),
                          np.array([1.,1.]),np.array([4,5]),X)

    def test_acoustic_proxies_are_descriptors_not_pitch_id(self):
        X=np.zeros((3,49,222),dtype=np.float16)
        X[:,:,126]=1.
        X[:,:,134]=3.
        X[:,:,142]=5.
        X[:,:,127]=.5
        X[:,:,135]=1.
        X[:,:,143]=1.5
        a=sharp_proxies(X,np.array([True,True,False]))
        self.assertAlmostEqual(a["fundamental_proxy_mean"],3.)
        self.assertAlmostEqual(a["harmonic_2_proxy_mean"],1.)
        self.assertAlmostEqual(a["fund_minus_h2_mean"],2.)
        self.assertEqual(a["n"],2)
        self.assertIsNone(sharp_proxies(X,np.array([False]*3))["fundamental_proxy_mean"])
        self.assertEqual(THRESHOLD,.70)
        self.assertEqual(SCHEMA,"S820_TRUE_K_TRANSITION_SPECTRAL_PROXY_FIT80_RESEARCH_V1")

    def test_future_independence_of_descriptor(self):
        X=np.ones((11,49,222),dtype=np.float16)
        before=sharp_proxies(X[:5],np.ones(5,bool))
        X[5:]=100.
        after=sharp_proxies(X[:5],np.ones(5,bool))
        self.assertEqual(before,after)

if __name__=="__main__":
    unittest.main()
