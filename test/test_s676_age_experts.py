"""S676 note-age specialist routing is causal and never sees true duration."""
import unittest
import numpy as np
from scripts.s676_causal_noteoff_age_experts import AgeExperts,AGE_EDGES,frame_matrix

class Fixed:
    def __init__(self,value):self.value=value
    def predict_proba(self,X):
        v=np.full(len(X),self.value,np.float64)
        return np.column_stack((1-v,v))

class S676Tests(unittest.TestCase):
    def test_age_uses_native_original_feature_column_63(self):
        x=np.zeros((4,75),np.float32)
        x[:,63]=np.log1p([20,180,700,300])
        x[:,-2]=12 # Should NEVER be the source note age.
        h=AgeExperts(Fixed(.1),(Fixed(.2),Fixed(.4),Fixed(.8)))
        p=h.predict_proba(x)[:,1]
        np.testing.assert_allclose(p,[.2,.4,.8,.4])

    def test_same_prefix_same_prediction(self):
        x=np.zeros((2,75),np.float32)
        x[:,63]=np.log1p(300)
        h=AgeExperts(Fixed(.1),(Fixed(.2),Fixed(.4),Fixed(.8)))
        a=h.predict_proba(x)
        y=x.copy()
        y[:,-2]=99.
        b=h.predict_proba(y)
        np.testing.assert_allclose(a,b)
    def test_boundaries_fixed(self):
        self.assertEqual(AGE_EDGES[:3],(0.,160.,550.))

if __name__=="__main__":unittest.main()
