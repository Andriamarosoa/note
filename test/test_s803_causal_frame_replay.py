"""S803 gates have no future access and cannot modify frozen probability heads."""
import unittest
import numpy as np
from scripts.s803_causal_frame_replay import gating,POLICIES
from scripts.s797_fragment_loop_fixture import example

class TestS803(unittest.TestCase):
    def test_policy_always_current_frame_and_no_mutation(self):
        d,X,P,orig=example()
        for name in POLICIES:
            snap={k:v.copy() for k,v in P.items()}
            out,blocked=gating(X,P,name)
            self.assertGreaterEqual(blocked,0)
            self.assertEqual(set(out),set(P))
            for h in P:
                np.testing.assert_array_equal(P[h],snap[h])
            # Future samples cannot influence an earlier emitted probability.
            if len(d)>1:
                XT=X.copy()
                XT[-1,:,:]=0
                PT={h:v.copy() for h,v in P.items()}
                for h in PT:PT[h][-1,:]=0
                now,_=gating(XT,PT,name)
                for h in P:
                    np.testing.assert_array_equal(out[h][:-1],now[h][:-1])
    def test_unknown_policy_fails(self):
        d,X,P,orig=example()
        with self.assertRaises(ValueError):gating(X,P,"oracle")
if __name__=="__main__":unittest.main()
