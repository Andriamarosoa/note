"""Tests: identity, train-only contextual audits, learned connection and four modes."""
import unittest
import numpy as np
import tensorflow as tf
from scripts.learn_v273_transition_combo_risk import (
    compute_direct_options,COMBOS,TransitionCombinationArbiter)
from scripts.learn_v273_audit_aware_selection import (
    fit_acoustic_regimes,identity_descriptor,
    append_audit_aware_descriptors
)


class AuditAwareSelectorTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(27402)
        rng=np.random.default_rng(113)
        n=54
        self.base=np.resize([2,3,4],n)
        self.fold=(np.arange(n)//3)%3
        self.y=np.resize([3,4,2,3,0,4,5,2,4],n)
        P=rng.uniform(.1,1,(n,6,5)).astype(np.float32)
        P/=P.sum(axis=-1,keepdims=True)
        raw,avail=compute_direct_options(P,self.base)
        self.avail=avail
        # For structural test, the base ten fields are precomputed
        # with the production inner-fold auditor; labels excluded.
        from scripts.learn_v273_transition_combo_risk import attach_oof_subset_audits
        self.orig,self.held,_=attach_oof_subset_audits(
            raw,avail,self.base,self.y,self.fold,raw,avail,self.base)
        self.audio=rng.normal(size=(n,6)).astype(np.float32)
        self.b=np.eye(7,dtype=np.float32)[self.base]
        self.keep=rng.uniform(size=(n,14)).astype(np.float32)

    def test_composition_identifies_distinct_same_size_subsets(self):
        bits=identity_descriptor(self.orig,self.avail,self.base)
        self.assertEqual(bits.shape,(len(self.y),5,COMBOS,7))
        # Source K2 -> K3 j=1, subset id=12 uses H3+H4,
        # id=18 uses H2+H5; both contain H0 and have same size.
        i=0
        self.assertEqual(self.base[i],2)
        self.assertTrue(self.avail[i,1,12])
        self.assertTrue(self.avail[i,1,18])
        self.assertFalse(np.array_equal(bits[i,1,12],bits[i,1,18]))
        self.assertEqual(bits[i,1,12].sum(),bits[i,1,18].sum())
        # K2 source cannot use K3 -> K2 corrective head.
        self.assertFalse(np.any(bits[self.base==2,0,:,:]))

    def test_current_inner_fold_label_does_not_enter_contextual_audit(self):
        for mode,width in (("baseline",10),("identity",17),
                           ("regimes",14),("both",21)):
            tr,va,logs=append_audit_aware_descriptors(
                self.orig,self.held,self.avail,self.avail,
                self.base,self.base,self.y,self.fold,
                self.audio,self.audio,mode)
            self.assertEqual(tr.shape[-1],width)
            self.assertEqual(va.shape[-1],width)
            self.assertTrue(np.isfinite(tr).all())
            if mode in ("regimes","both"):
                changed=self.y.copy()
                changed[0]=5 if self.y[0]!=5 else 4
                tr2,va2,_=append_audit_aware_descriptors(
                    self.orig,self.held,self.avail,self.avail,
                    self.base,self.base,changed,self.fold,
                    self.audio,self.audio,mode)
                self.assertTrue(np.array_equal(
                    tr[self.fold==self.fold[0],:,:,10 if mode=="regimes" else 17:],
                    tr2[self.fold==self.fold[0],:,:,10 if mode=="regimes" else 17:]))
                self.assertTrue(len(logs)>0)

    def test_train_only_cluster_does_not_fit_on_heldout_audio(self):
        tr=self.audio.copy()
        va=self.audio[:15].copy()
        a,b,km=fit_acoustic_regimes(tr,va)
        c,d,km2=fit_acoustic_regimes(tr,va+200.)
        self.assertTrue(np.array_equal(a,c))
        self.assertTrue(np.allclose(km.cluster_centers_,km2.cluster_centers_))
        self.assertEqual(len(b),len(va))
        self.assertEqual(len(d),len(va))

    def test_both_channels_feed_learned_selection_gradients(self):
        tr,held,_=append_audit_aware_descriptors(
            self.orig,self.held,self.avail,self.avail,
            self.base,self.base,self.y,self.fold,
            self.audio,self.audio,"both")
        observed=tf.Variable(held)
        model=TransitionCombinationArbiter(width=16)
        with tf.GradientTape() as tape:
            feat=dict(subset_features=observed,subset_mask=self.avail,
                      baseline=self.b,context=self.audio,keep_heads=self.keep)
            output=model(feat)
            score=tf.reduce_sum(output["action_logits"][:,:5])+tf.reduce_sum(
                output["subset_attention"][:,:,11])
        grad=tape.gradient(score,observed)
        self.assertIsNotNone(grad)
        self.assertGreater(float(tf.reduce_sum(tf.abs(grad[...,10:17]))),1.e-7)
        self.assertGreater(float(tf.reduce_sum(tf.abs(grad[...,17:21]))),1.e-7)


if __name__=="__main__":unittest.main()
