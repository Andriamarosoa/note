"""S679 near-boundary sample weights affect TRAIN, not causal inference."""
import unittest
import numpy as np
from scripts.s679_weighted_causal_noteoff_hazard import release_proximity_weights
class S679Tests(unittest.TestCase):
    def test_positive_close_release_weighted(self):
        positions=np.array([50,70,95,97,99,100,101,102,105,110,116])
        w=release_proximity_weights(positions,100)
        self.assertEqual(w[4],1.5)
        self.assertEqual(w[5],3.)
        self.assertEqual(w[8],3.)
        self.assertEqual(w[9],1.5)
        self.assertEqual(w[0],1.)
        self.assertEqual(w[-1],1.)
    def test_weights_have_train_only_shape(self):
        a=release_proximity_weights(np.arange(140),70)
        self.assertEqual(a.shape,(140,))
        self.assertTrue(np.isfinite(a).all())
if __name__=="__main__":unittest.main()
