"""S678 predicted-K experts preserve frozen original K distribution."""
import unittest
import numpy as np
from scripts.s678_predicted_k_noteoff_experts import PredictedKExperts,K2_CUTOFF
class Fixed:
    def __init__(self,p):self.p=p
    def predict_proba(self,X):
        a=np.full(len(X),self.p)
        return np.stack((1.-a,a),axis=1)
class PredictedKTests(unittest.TestCase):
    def test_selector_does_not_see_labelled_true_K(self):
        x=np.zeros((3,75),np.float32)
        x[0,56]=1. # predicted K1
        x[1,57]=.8 # predicted K2
        x[2,57]=.15;x[2,58]=.4 # split support K2plus
        m=PredictedKExperts(Fixed(.2),Fixed(.35),Fixed(.85))
        np.testing.assert_allclose(m.predict_proba(x)[:,1],[.35,.85,.85])
        self.assertEqual(K2_CUTOFF,.4)
    def test_other_acoustics_can_change_without_changing_expert_routing(self):
        x=np.zeros((2,75),np.float32)
        x[0,56]=.9;x[1,57]=.8
        m=PredictedKExperts(Fixed(.2),Fixed(.3),Fixed(.9))
        a=m.predict_proba(x)
        x[:,63]=9
        b=m.predict_proba(x)
        np.testing.assert_allclose(a,b)
if __name__=="__main__":unittest.main()
