"""Audit feature importance tests: change audit only, or event acoustic only."""
import unittest
import numpy as np
import tensorflow as tf

from scripts.audit_v273_selection_feature_influence import interventions
from scripts.learn_v273_transition_combo_risk import (
    compute_direct_options,attach_oof_subset_audits,TransitionCombinationArbiter
)


class AuditInfluenceContracts(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(27402)
        rng=np.random.default_rng(143)
        self.base=np.resize([2,3,4],18)
        self.true=np.resize([3,3,2,4,2,4],18)
        P=rng.uniform(.02,1,(18,6,5)).astype(np.float32)
        P/=P.sum(axis=-1,keepdims=True)
        fold=(np.arange(18)//3)%3
        direct,valid=compute_direct_options(P,self.base)
        train,held,log=attach_oof_subset_audits(
            direct,valid,self.base,self.true,fold,
            direct,valid,self.base)
        self.data={
            "subset_features":held,
            "subset_mask":valid,
            "baseline":np.eye(7,dtype=np.float32)[self.base],
            "context":rng.normal(size=(18,6)).astype(np.float32),
            "keep_heads":rng.uniform(size=(18,14)).astype(np.float32),
        }

    def test_interventions_touch_only_their_respective_features(self):
        d=self.data
        ab=interventions(d)
        self.assertEqual(set(ab),{"audit_flatten","audit_reverse",
                                   "acoustic_shuffle","direct_votes_flatten"})
        for name,v in ab.items():
            self.assertTrue(np.array_equal(v["baseline"],d["baseline"]))
            self.assertTrue(np.array_equal(v["subset_mask"],d["subset_mask"]))
            self.assertTrue(np.array_equal(v["keep_heads"],d["keep_heads"]))
            if name.startswith("audit"):
                self.assertTrue(np.array_equal(
                    v["subset_features"][...,:6],d["subset_features"][...,:6]))
                self.assertTrue(np.array_equal(v["context"],d["context"]))
            elif name=="acoustic_shuffle":
                self.assertTrue(np.array_equal(
                    v["subset_features"],d["subset_features"]))
                for k in (2,3,4):
                    old=d["context"][self.base==k]
                    new=v["context"][self.base==k]
                    self.assertTrue(np.allclose(old.sum(axis=0),new.sum(axis=0)))
            else:
                self.assertTrue(np.array_equal(
                    v["subset_features"][...,6:],d["subset_features"][...,6:]))

    def test_risk_and_selection_are_differentiably_connected_to_audit(self):
        model=TransitionCombinationArbiter(width=16)
        feat={k:tf.convert_to_tensor(v) for k,v in self.data.items()}
        with tf.GradientTape() as tape:
            tape.watch(feat["subset_features"])
            out=model(feat)
            # We test score sensitivity and selector ATTENTION.
            loss=tf.reduce_sum(out["action_logits"][:,:5])+tf.reduce_sum(
                out["subset_attention"][:,:,1])
        g=tape.gradient(loss,feat["subset_features"])
        self.assertIsNotNone(g)
        self.assertTrue(np.isfinite(g.numpy()).all())
        # Must be nonzero for the OOF audit channels, not just the
        # subset's direct-vote features.
        self.assertGreater(float(tf.reduce_sum(tf.abs(g[...,6:]))),1e-7)
        self.assertGreater(float(tf.reduce_sum(tf.abs(g[...,:6]))),1e-7)


if __name__=="__main__":unittest.main()
