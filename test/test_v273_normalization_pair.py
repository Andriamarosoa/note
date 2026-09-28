import unittest
import numpy as np
from scripts.v273_spectral_normalization import interpretation
from scripts.summarize_v273_training_budget import paired_change


class NormalizationVerdictTest(unittest.TestCase):
    def test_polyphonic_gain_cannot_hide_small_k_regression(self):
        delta={str(k):0 for k in range(7)}
        delta['1']=-1
        self.assertEqual(interpretation(20,delta),'polyphonic_gain_with_low_k_regressions')
        delta['1']=0
        self.assertEqual(interpretation(20,delta),'polyphonic_gain_without_low_k_regression_on_this_split')
        self.assertEqual(interpretation(0,delta),'no_polyphonic_validation_gain')

    def test_missing_class_rejected(self):
        with self.assertRaises(RuntimeError):
            interpretation(20,{'2':20})

    def test_fixed_scale_preserves_the_input_pair(self):
        x=np.array([[3.,1.,0.],[6.,2.,0.]])
        normalized=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+0.001)
        self.assertLess(np.linalg.norm(normalized[0]-normalized[1]),0.001)
        scaled=x/12.
        np.testing.assert_allclose(scaled*12.,x)
        self.assertGreater(np.linalg.norm(scaled[0]-scaled[1]),0.2)

    def test_paired_rows_cannot_be_reordered(self):
        a=dict(global_index=np.arange(3),member=np.array(['00_BN2-166-Ab_comp.jams']*3),
               k=np.array([1,2,3]),predicted=np.array([1,1,3]))
        b=dict(a,predicted=np.array([0,2,3]))
        r=paired_change(a,b)
        self.assertEqual(r['all']['delta'],0)
        self.assertEqual(r['poly']['delta'],1)
        self.assertEqual(r['by_k']['1']['delta'],-1)
        b['global_index']=np.array([1,0,2])
        with self.assertRaises(AssertionError):
            paired_change(a,b)


if __name__=='__main__':
    unittest.main()
