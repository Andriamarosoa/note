"""Unit tests for a genuine trainable audited combination risk router."""
import unittest
import numpy as np
import tensorflow as tf
from scripts.learn_v273_transition_combo_risk import (
    compute_direct_options,attach_oof_subset_audits,
    COMBOS,FEAT_DIM,TransitionCombinationArbiter,targets,objective,
)


class TransitionCombinationRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(27402)

    def fixture(self,n=18):
        rng=np.random.default_rng(31)
        base=np.resize(np.array([2,3,4],int),n)
        truth=np.resize(np.array([3,3,2,0,1,4],int),n)
        P=rng.uniform(.1,1.,(n,6,5)).astype("float32")
        P/=P.sum(axis=-1,keepdims=True)
        folds=np.resize(np.array([0,1,2],int),n)
        return P,base,truth,folds

    def test_64_subsets_and_structural_masks(self):
        P,b,truth,fold=self.fixture()
        options,available=compute_direct_options(P,b)
        self.assertEqual(options.shape,(len(b),5,64,6))
        self.assertEqual(available.shape,(len(b),5,64))
        for i,base in enumerate(b):
            for j,k in enumerate((2,3,4,5,6)):
                expected=0 if base==k else (
                    64 if (base,k) in ((2,3),(3,2),(3,4),(4,3)) else 32)
                self.assertEqual(int(available[i,j].sum()),expected)
        self.assertTrue(np.isfinite(options).all())

    def test_oof_priors_exclude_current_fold_labels(self):
        P,b,y,fold=self.fixture()
        opts,mask=compute_direct_options(P,b)
        train,held,log=attach_oof_subset_audits(opts,mask,b,y,fold,opts,mask,b)
        self.assertEqual(train.shape,(len(b),5,64,FEAT_DIM))
        self.assertEqual(held.shape,train.shape)
        self.assertTrue(np.isfinite(train).all())
        self.assertTrue(len(log)>0)
        # Changing a K3 label on fold 0 must NOT change any of its
        # own fold-0 prior inputs. This tests direct label separation.
        changed=y.copy()
        changed[0]=4
        train2,held2,_=attach_oof_subset_audits(
            opts,mask,b,changed,fold,opts,mask,b)
        self.assertTrue(np.array_equal(train[fold==0,:, :,6:],
                                       train2[fold==0,:,:,6:]))
        self.assertFalse(np.array_equal(held[:,:,:,6:],held2[:,:,:,6:]))

    def get_batch(self):
        P,b,y,fold=self.fixture()
        opts,mask=compute_direct_options(P,b)
        xtr,xval,_=attach_oof_subset_audits(opts,mask,b,y,fold,opts,mask,b)
        audio=np.random.default_rng(11).normal(size=(len(b),6)).astype("float32")
        keep=np.zeros((len(b),14),dtype="float32")
        context=dict(subset_features=xtr,subset_mask=mask,
                     baseline=np.eye(7,dtype="float32")[b],
                     context=audio,keep_heads=keep)
        return context,y,b

    def test_risk_directly_controls_decision_and_training_gradients(self):
        xx,y,base=self.get_batch()
        model=TransitionCombinationArbiter(width=16)
        out=model(xx)
        self.assertEqual(out["action_logits"].shape,(len(y),6))
        self.assertEqual(out["subset_attention"].shape,(len(y),5,64))
        invalid=~np.any(xx["subset_mask"],axis=2)
        self.assertTrue(np.all(out["action_logits"].numpy()[:,:5][invalid]<-1e8))
        self.assertTrue(np.isfinite(out["candidate_correct"].numpy()).all())
        t,co,re,keep=targets(y,base)
        self.assertTrue(np.all(t[y<2]==5))
        with tf.GradientTape() as tape:
            p=model(xx,training=True)
            loss=objective(p,t,co,re,keep,np.ones(len(y),np.float32))
        grads=tape.gradient(loss,model.trainable_variables)
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads))
        with tf.GradientTape() as tape2:
            p=model(xx)
            score=tf.reduce_sum(p["action_logits"][:,1])
        rgrads=tape2.gradient(score,model.score_regress.trainable_variables)
        self.assertTrue(all(g is not None for g in rgrads))
        self.assertTrue(any(float(tf.reduce_sum(tf.abs(g)))>0 for g in rgrads))


if __name__=="__main__":
    unittest.main()
