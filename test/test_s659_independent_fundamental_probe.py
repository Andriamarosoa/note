"""S659 causal/no-original-mutation and physically distinct octave tests."""
import unittest
import numpy as np
from scripts.s659_independent_fundamental_probe import (
    fundamental_contrasts, permit, train_choice, POLICIES,
)

class S659ProbeTests(unittest.TestCase):
    def test_subfundamental_distinguishes_two_sources_in_controlled_sines(self):
        sr = 44100
        t = np.arange(8192)/sr
        high = np.sin(2*np.pi*164.813778*t).astype("float32")
        low = .4*np.sin(2*np.pi*82.406889*t).astype("float32")
        only = fundamental_contrasts(high, len(high), 40)
        two = fundamental_contrasts(high+low, len(high), 40)
        self.assertGreater(two[0], only[0]*5)
        self.assertFalse(permit("probe_low_3", only[0], only[1]))
        self.assertTrue(permit("probe_low_3", two[0], two[1]))

    def test_strict_causality_future_pcm_cannot_change_decision(self):
        base=np.sin(2*np.pi*82.406889*np.arange(8192)/44100).astype("float32")
        a=fundamental_contrasts(base,len(base),40)
        extended=np.concatenate((base,np.ones(2048,dtype="float32")))
        b=fundamental_contrasts(extended,len(base),40)
        self.assertEqual(a,b)

    def test_reject_inadequate_three_partial(self):
        self.assertFalse(permit("probe_low_3_harm3",3.5,1.2))
        self.assertTrue(permit("probe_low_3_harm3",3.5,1.6))
        self.assertEqual(len(POLICIES),6)

    def test_never_select_without_calibration_gain(self):
        base={"TP":10,"FP":3,"FN":4,"F1":20/27,"latency_ms":[]}
        row={"composition":"A","stats":{p:{"score":base,"new_true":0,"lost_true":0,
          "overlap_TP":0,"K3plus_TP":0,"E2_TP":0,"attempts":0,"accepted":0}
          for p in POLICIES}}
        chosen,proof=train_choice([row],[row])
        self.assertEqual(chosen,"S614_original")
        self.assertEqual(proof["selected"],"S614_original")

if __name__=="__main__":
    unittest.main()
