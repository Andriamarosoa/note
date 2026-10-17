"""S660: multiscale causal inference and confidence selection control."""
import unittest
import numpy as np
from scripts.s660_multiscale_fundamental_probe import (
    multiscale_contrasts, multiscale_permit, train_choice, POLICIES,
)

class CausalMultiscaleTests(unittest.TestCase):
    def test_right_aligned_views_no_future_access(self):
        n=8192
        x=np.sin(2*np.pi*82.406889*np.arange(n)/44100).astype("float32")
        old=multiscale_contrasts(x,n,40)
        new=multiscale_contrasts(np.concatenate(
            (x,np.ones(500,dtype="float32"))),n,40)
        self.assertEqual(old,new)
        self.assertEqual(set(old),{"short","long","recent"})

    def test_multiscale_policies_have_independent_behavior(self):
        x={"long":3., "short":1.5,"recent":3.}
        self.assertTrue(multiscale_permit("probe_low_2",x))
        self.assertFalse(multiscale_permit("probe_short_2",x))
        self.assertFalse(multiscale_permit("probe_both_2",x))
        self.assertTrue(multiscale_permit("probe_any_2",x))
        self.assertTrue(multiscale_permit("probe_recent_2",x))

    def test_single_high_with_and_without_physical_lower(self):
        n=8192
        t=np.arange(n)/44100
        high=np.sin(2*np.pi*164.813778*t).astype("float32")
        both=high + .4*np.sin(2*np.pi*82.406889*t).astype("float32")
        a=multiscale_contrasts(high,n,40)
        b=multiscale_contrasts(both,n,40)
        self.assertGreater(b["long"],a["long"]*5)
        self.assertTrue(multiscale_permit("probe_low_2",b))

    def test_selection_requires_calibration_and_held_improvement(self):
        base={"TP":10,"FP":3,"FN":4,"F1":20/27,"latency_ms":[]}
        row={"composition":"A","stats":{
            p:{"score":base,"new_true":0,"lost_true":0,
               "overlap_TP":0,"K3plus_TP":0,"E2_TP":0,
               "attempts":0,"accepted":0} for p in POLICIES}}
        winner,_=train_choice([row],[row])
        self.assertEqual(winner,"S614_original")

if __name__=="__main__":
    unittest.main()
