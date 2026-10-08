"""Regression tests: a selection is NOT open for every hypothetical K.

Structural eligibility is based only on a head's action semantics and
baseline prediction, NEVER real true K. Soft weights are learned separately
for each K and may activate an alternate K despite a wrong baseline.
"""
from __future__ import annotations
import unittest

import numpy as np
import tensorflow as tf

from scripts.evaluate_v273_neural_history_mix import (
    build_candidate_logits, inputs, H
)
from scripts.learn_v273_neural_head_selector import (
    ClassConditionalAuditSelector, form_targets, train_loss
)


class ClassConditionalRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(27402)

    def test_correction_support_is_target_specific_not_truth_specific(self):
        base=np.array([2,3,4],int)
        P=np.full((3,6,5),.2,dtype="float32")
        logits,active,kind,support=build_candidate_logits(P,base)
        self.assertEqual(logits.shape,(3,H,8))
        self.assertEqual(support.shape,(3,H,7))
        # C23 is open only to K3 for a baseline K2 signal.
        self.assertTrue(active[0,6])
        self.assertTrue(support[0,6,3])
        self.assertFalse(support[0,6,2])
        self.assertFalse(support[0,6,4])
        self.assertFalse(active[1,6])
        self.assertFalse(support[1,6].any())
        # C32 and C34 both open for baseline K3, but DIFFERENT target K.
        self.assertTrue(support[1,7,2])
        self.assertTrue(support[1,8,4])
        self.assertFalse(support[1,7,4])
        self.assertFalse(support[1,8,2])
        # K4->K3 does not accidentally open to K2.
        self.assertTrue(support[2,9,3])
        self.assertFalse(support[2,9,2])
        # KEEP/fix heads must be able to influence abstention but
        # must never directly pretend they can identify all K.
        self.assertTrue(active[0,10])
        self.assertFalse(support[:,10:,:].any())
        self.assertTrue(np.all(support[:,0,:]))

    def test_separate_normalized_per_k_weights_and_finite_gradients(self):
        rng=np.random.default_rng(27402)
        base=np.array([2,3,4,2,3,4,2,3],int)
        P=rng.uniform(.1,1.,size=(len(base),6,5)).astype("float32")
        P/=P.sum(axis=2,keepdims=True)
        logits,mask,types,support=build_candidate_logits(P,base)
        audit=rng.uniform(0,.2,(len(base),H,21)).astype("float32")
        context=rng.normal(size=(len(base),5)).astype("float32")
        feat=inputs(logits,mask,types,support,audit,context,base)
        model=ClassConditionalAuditSelector(hidden=24)
        out=model(feat)
        w=out["class_selection_weights"].numpy()
        self.assertEqual(w.shape,(len(base),H,7))
        self.assertTrue(np.allclose(w.sum(axis=1),1.,atol=1e-6))
        self.assertTrue(np.all(w[~support]==0.))
        # K0 and K1 only have baseline as a class-compatible source.
        self.assertTrue(np.allclose(w[:,0,0],1.))
        self.assertTrue(np.allclose(w[:,0,1],1.))
        # C23 cannot be selected to *justify K2*, but it can help
        # classify candidate K3 on an event originally predicted K2.
        self.assertAlmostEqual(float(w[0,6,2]),0.,places=8)
        self.assertGreater(float(w[0,6,3]),0.)
        actions,risk=form_targets(
            np.array([3,3,4,2,2,4,3,4]),base,
            np.argmax(logits,axis=-1)
        )
        with tf.GradientTape() as tape:
            o=model(feat,training=True)
            loss=train_loss(o,actions,risk,mask,risk_weight=.2)
        grads=tape.gradient(loss,model.trainable_variables)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads))
        self.assertTrue(np.allclose(
            out["selection_weights"].numpy().sum(axis=1),1.,atol=1e-6
        ))

    def test_class_weight_depends_on_audio_not_true_k(self):
        n=4
        base=np.array([2,2,2,2])
        P=np.full((n,6,5),.2,np.float32)
        logits,active,kind,support=build_candidate_logits(P,base)
        # Deliberately different observable contexts, but no labels.
        context=np.array([[-3.,0.],[-1.,0.],[1.,0.],[3.,0.]],np.float32)
        sample=inputs(
            logits,active,kind,support,
            np.zeros((n,H,21),np.float32),
            context,base
        )
        nn=ClassConditionalAuditSelector(hidden=24)
        output=nn(sample)
        # Neural K-specific gates must have signal-context paths; their
        # weights need not be pre-specified by any hard audit coefficient.
        self.assertEqual(output["class_selection_weights"].shape,(n,H,7))
        self.assertTrue(np.isfinite(output["action_logits"].numpy()).all())
        # No true K property exists in the predictor contract.
        self.assertNotIn("true_k",sample)
        self.assertNotIn("target",sample)


if __name__=="__main__":
    unittest.main()
