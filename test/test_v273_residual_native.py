import unittest
import numpy as np

from scripts.v273_residual_native import treatment_maps, inputs
from scripts.audit_v273_residual_native import count, bootstrap


class ResidualNativeTest(unittest.TestCase):
    def test_control_removes_all_residual_information(self):
        x = np.random.default_rng(93).uniform(0,13,(2,31,64,4)).astype(np.float16)
        changed = x.copy(); changed[:,:,:,[1,3]] += 50
        np.testing.assert_array_equal(treatment_maps(x,'observed_only'),
                                      treatment_maps(changed,'observed_only'))
        observed = treatment_maps(x,'observed_only')
        np.testing.assert_array_equal(observed[:,:,:,0],observed[:,:,:,1])
        np.testing.assert_array_equal(observed[:,:,:,2],observed[:,:,:,3])

    def test_residual_arm_retains_each_channel_without_clipping(self):
        x = np.full((1,31,64,4),13,np.float16)
        actual = treatment_maps(x,'with_residual')
        np.testing.assert_array_equal(actual,x.astype(np.float32))
        actual[:]=0
        np.testing.assert_array_equal(x,13)

    def test_rejects_missing_outer_maps(self):
        with self.assertRaises(RuntimeError):
            treatment_maps(np.full((1,31,64,4),np.nan),'with_residual')

    def test_other_inputs_identical_between_arms(self):
        rng = np.random.default_rng(94)
        cache = {'sequence':rng.random((3,48,10)), 'mask':np.ones((3,48)),
                 'stats':rng.random((3,12))}
        maps = rng.random((3,31,64,4)); geometry = rng.random((3,31,2))
        ids = np.array([2,0])
        a,b = (inputs(cache,maps,geometry,ids,arm) for arm in ('observed_only','with_residual'))
        for name in ('candidate_set','candidate_mask','cluster_stats','ownership_map'):
            np.testing.assert_array_equal(a[name],b[name])
        np.testing.assert_array_equal(a['spectral_map'][:,:,:,[0,2]],b['spectral_map'][:,:,:,[0,2]])

    def test_independent_exact_k_and_overcount_denominators(self):
        truth=np.array([0,1,1,2,2,3,4,5,6])
        prediction=np.array([0,0,2,1,3,3,4,5,6])
        result=count(truth,np.eye(7)[prediction])
        self.assertEqual((result['rows'],result['correct']),(9,5))
        self.assertEqual((result['poly_rows'],result['poly_correct']),(6,4))
        self.assertEqual((result['over_k_lt4'],result['under']),(2,2))
        self.assertEqual(result['by_true_k']['1']['exact'],0.)
        self.assertEqual(result['by_true_k']['3']['exact'],1.)

    def test_paired_identical_predictions_have_zero_bootstrap_difference(self):
        truth=np.array([0,1,2,3,4,5,6])
        member=np.array(['00_BN2-166-Ab_comp.jams']*3+['00_Jazz3-150-C_comp.jams']*4)
        prediction=np.array([0,0,2,3,3,5,5])
        result=bootstrap(truth,member,prediction,prediction,truth>=2)
        self.assertEqual(result['difference'],0.)
        self.assertEqual(result['percentile95'],[0.,0.])


if __name__ == '__main__':
    unittest.main()
