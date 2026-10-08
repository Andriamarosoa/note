import unittest
import numpy as np
from scripts.audit_v273_risk_factor_swaps import combine


class FactorSwapTests(unittest.TestCase):
    def test_shared_risk_and_other_mass_remain_coherent(self):
        b=np.array([3,3,4]);p=np.array([[2,2,3],[2,2,3],[4,4,4]])
        out=combine(np.array([.1,.9,.2]),np.zeros((3,7)),p,b)
        np.testing.assert_allclose(out['class_probability'].sum(1)+out['other_probability'],1)
        np.testing.assert_array_equal(out['prediction'],[2,3,4])
        np.testing.assert_allclose(out['class_probability'][np.arange(3),b],[.1,.9,.2])
        self.assertAlmostEqual(out['other_probability'][2],.8)

    def test_unsupported_destination_is_not_invented_by_a_large_logit(self):
        logits=np.zeros((1,7));logits[0,6]=1e6
        out=combine(np.array([.1]),logits,np.array([[2,3]]),np.array([3]))
        self.assertEqual(out['class_probability'][0,6],0)
        self.assertEqual(out['prediction'][0],2)


if __name__=='__main__':unittest.main()
