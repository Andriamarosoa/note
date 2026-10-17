"""S700 causal K delays never read later audio; oracle is separate."""
import unittest
import numpy as np
from scripts.s700_causal_K_temporal_alignment import (
    causal_delay,oracle_future,DELAYS,FUTURE_ORACLE
)
class KTimeTests(unittest.TestCase):
    def test_delays_never_anticipate_future(self):
        v=np.array([0,1,2,3,4,5])
        np.testing.assert_array_equal(causal_delay(v,2),[0,0,0,1,2,3])
        np.testing.assert_array_equal(causal_delay(v,0),v)
        with self.assertRaises(ValueError):causal_delay(v,-2)
    def test_synthetic_noncausal_upper_bound(self):
        v=np.array([0,1,2,3,4,5])
        np.testing.assert_array_equal(oracle_future(v,2),[2,3,4,5,5,5])
        self.assertEqual(DELAYS[0],0)
        self.assertEqual(FUTURE_ORACLE[0],0)
    def test_prefix_invariance(self):
        v=np.arange(100)
        before=causal_delay(v,16)
        v[80:]=999
        after=causal_delay(v,16)
        np.testing.assert_array_equal(before[:80],after[:80])
if __name__=="__main__":unittest.main()
