"""S721 guard tests: no probability/data leaks, separated head and K mixtures."""
import unittest
import numpy as np
from scripts.s704_joint_metrics import SCORES
from scripts.s721_shrink_poly_heads import blend_heads,trials,six_metric_guard


class S721ShrinkageTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(721)
        self.a=(rng.random((8,49,3),dtype=np.float32),
                rng.dirichlet(np.ones(7),size=8).astype(np.float32))
        self.b=(rng.random((8,49,3),dtype=np.float32),
                rng.dirichlet(np.ones(7),size=8).astype(np.float32))

    def test_grid_always_contains_unchanged_reference_and_head_specific_controls(self):
        opts=trials()
        self.assertEqual(len(opts),15)
        self.assertTrue(all(x==0 for x in opts["S708_frozen"].values()))
        self.assertEqual(opts["k_0.25"]["onset"],0.)
        self.assertEqual(opts["onset_0.50"]["sustain"],0.)

    def test_baseline_is_bitwise_identical_not_smoothed(self):
        p,k=blend_heads(self.a,self.b,trials()["S708_frozen"])
        np.testing.assert_array_equal(p,self.a[0])
        np.testing.assert_array_equal(k,self.a[1])

    def test_only_selected_heads_change_and_K_remains_probability_distribution(self):
        policy=trials()["k_0.25"]
        p,k=blend_heads(self.a,self.b,policy)
        np.testing.assert_array_equal(p,self.a[0])
        np.testing.assert_allclose(k,.75*self.a[1]+.25*self.b[1],atol=1e-7)
        np.testing.assert_allclose(k.sum(-1),np.ones(8),atol=1e-6)
        policy=trials()["onset_0.50"]
        p,k=blend_heads(self.a,self.b,policy)
        np.testing.assert_allclose(p[...,0],.5*self.a[0][...,0]+.5*self.b[0][...,0],atol=1e-7)
        np.testing.assert_array_equal(p[...,1],self.a[0][...,1])
        np.testing.assert_array_equal(p[...,2],self.a[0][...,2])
        np.testing.assert_array_equal(k,self.a[1])

    def test_no_regression_is_six_metrics_plus_offset_and_long_anchors(self):
        base=dict.fromkeys(SCORES,.5);base["tp"]=100
        good=dict.fromkeys(SCORES,.52);good["tp"]=102
        matched=dict(lost_long=0,lost_correct_offsets=0)
        self.assertTrue(six_metric_guard(base,good,matched))
        self.assertFalse(six_metric_guard(base,good,{**matched,"lost_correct_offsets":1}))
        for metric in SCORES:
            regression=good.copy();regression[metric]=.4999
            self.assertFalse(six_metric_guard(base,regression,matched),metric)

    def test_reject_invalid_abi_and_non_finite_outputs(self):
        with self.assertRaisesRegex(ValueError,"ABI"):
            blend_heads((self.a[0][:,:40],self.a[1]),self.b,trials()["S708_frozen"])
        bad=self.b[0].copy();bad[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            blend_heads(self.a,(bad,self.b[1]),trials()["all_0.25"])


if __name__=="__main__":
    unittest.main()
