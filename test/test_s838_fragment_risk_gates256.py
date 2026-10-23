"""S838: fragment-only gate input, zero wait, frozen six-axis shadow."""
from dataclasses import replace
import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from test.test_s837_six_axis_policy_loops256 import source
from scripts.s838_fragment_risk_gates256 import (
    AXES, fragment_features, expert_predictions, train_labels,
    gate_probs, select_predictions, frame_audit, THRESHOLDS,
)


class S838FragmentGateTests(unittest.TestCase):
    def test_private_features_have_fixed_30dim_current_pcm_only(self):
        a=source("BN1")
        x=fragment_features(a)
        self.assertEqual(x.shape,(len(a.snapshots),30))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x.dtype,np.float32)
        # Mutating *future* audio snapshots cannot affect any earlier features.
        posts=list(a.snapshots)
        arr=posts[-1].post.copy()
        arr[:]=1.
        from causal_note.s834_native256_decoders import Fragment
        posts[-1]=Fragment(posts[-1].sample,arr,posts[-1].k)
        b=replace(a,snapshots=tuple(posts))
        np.testing.assert_array_equal(x[:-1],fragment_features(b)[:-1])

    def test_reference_expert_matches_s834_for_both_k_axes(self):
        a=source("Rock3")
        for axis in AXES:
            names,pred,k,valid=expert_predictions(a,axis)
            self.assertEqual(names[0],"reference")
            self.assertEqual(pred.shape[1],len(k))
            self.assertGreater(pred.shape[0],1)
            self.assertEqual(pred.dtype,np.int16)
            self.assertTrue(valid.any())
            y=train_labels(pred,k,valid)
            self.assertEqual(y.shape,k.shape)
            self.assertTrue(np.all((y>=0)&(y<pred.shape[0])))

    def test_fragment_selector_rejects_uncertain_alternatives(self):
        experts=np.array([[0,1,2,3],[1,2,1,0]],np.int16)
        probs=np.array([[.1,.9],[.9,.1],[.5,.5],[0.,1.]],np.float32)
        out,indices=select_predictions(experts,probs,.80)
        np.testing.assert_array_equal(out,np.array([1,1,2,0]))
        np.testing.assert_array_equal(indices,np.array([1,0,0,1]))

    def test_train_labels_learns_only_when_reference_is_wrong(self):
        experts=np.array([[1,2,3,1],[2,2,1,4],[1,4,4,4]],np.int16)
        k=np.array([2,2,4,1],np.int16)
        valid=np.array([True,True,True,True])
        y=train_labels(experts,k,valid)
        np.testing.assert_array_equal(y,np.array([1,0,2,0]))

    def test_per_true_k_loss_and_recovery_have_identity(self):
        old=np.array([0,1,2,3,2,1],np.int16)
        new=np.array([0,2,2,2,1,1],np.int16)
        truth=np.array([0,2,1,3,2,1],np.int16)
        ends=np.arange(1,7)*256
        audit=frame_audit(old,new,truth,np.ones(6,bool),ends,comp="BN1")
        self.assertEqual(audit["recovered"],1)
        self.assertEqual(audit["lost"],2)
        self.assertEqual(audit["total_changed"],3)
        self.assertEqual(audit["per_true_k"]["2"]["frames"],2)

    def test_tree_fit_predict_current_only_and_prob_map(self):
        rng=np.random.default_rng(838)
        x=rng.normal(size=(800,30)).astype(np.float32)
        y=np.where(x[:,0]>.1,1,0)
        tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=10,random_state=838)
        tree.fit(x,y)
        p=gate_probs(tree,x,3)
        self.assertEqual(p.shape,(800,3))
        self.assertTrue(np.allclose(p.sum(axis=1),1))
        self.assertTrue(np.all(p[:,2]==0))
        self.assertTrue(all(t>=0 for t in THRESHOLDS))


if __name__=="__main__":
    unittest.main()
