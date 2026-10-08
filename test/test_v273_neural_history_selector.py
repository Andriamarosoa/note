"""Architecture tests: learned XOR interactions, dynamic weights and audit provenance."""
import unittest
import numpy as np
import tensorflow as tf

from scripts.learn_v273_neural_head_selector import (
    LearnedAuditSelector, HeadContract, form_targets,
    audited_head_descriptor, check_oof_contract, train_loss, KEEP
)

class LearnedHeadSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.random.set_seed(27402)

    def test_all_head_types_and_masked_selection(self):
        n,h=8,5
        rng=np.random.default_rng(27402)
        z=rng.normal(0,1,(n,h,8)).astype("float32")
        audit=np.zeros((n,h,21),"float32")
        kind=np.eye(4,dtype="float32")[[1,0,2,3,2]]
        kind=np.broadcast_to(kind,(n,h,4)).copy()
        mask=np.ones((n,h),bool);mask[:,4]=False
        x=dict(head_logits=z,audit=audit,head_type=kind,head_mask=mask,
               context=np.zeros((n,3),"float32"),
               baseline=np.eye(7,dtype="float32")[[2]*n])
        nn=LearnedAuditSelector(hidden=24)
        out=nn(x)
        weights=out["selection_weights"].numpy()
        self.assertEqual(out["action_logits"].shape,(n,8))
        self.assertEqual(out["head_risk"].shape,(n,h,2))
        self.assertTrue(np.allclose(weights[:,4],0.))
        self.assertTrue(np.allclose(weights.sum(axis=1),1.,atol=1e-5))
        actions,risk=form_targets(
            np.array([3,2,4,2,3,2,4,2]),
            np.array([2]*n),
            np.array([[3,2,2,2,KEEP]]*n))
        with tf.GradientTape() as tape:
            o=nn(x,training=True)
            loss=train_loss(o,actions,risk,mask)
        grads=tape.gradient(loss,nn.trainable_variables)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(np.all(np.isfinite(g)) for g in grads))

    def test_neural_interaction_not_static_average(self):
        # XOR: two independent corrective heads are individually 50% right.
        # A learned nonlinear combination can decide which joint pattern
        # means KEEP and which means K3. No true labels in input.
        bit_a=np.tile(np.array([0,0,1,1]),48)
        bit_b=np.tile(np.array([0,1,0,1]),48)
        y=np.where(bit_a!=bit_b,3,2)
        n=len(y);h=3
        base=np.full(n,2,int)
        props=np.stack([base,np.where(bit_a,3,KEEP),
                        np.where(bit_b,3,KEEP)],axis=1)
        logits=np.full((n,h,8),-5.,dtype="float32")
        for i in range(n):
            for j in range(h):logits[i,j,props[i,j]]=5.
        kind=np.broadcast_to(
            np.eye(4,dtype="float32")[[1,2,3]],(n,h,4)
        ).copy()
        inp=dict(head_logits=logits,audit=np.zeros((n,h,21),"float32"),
                 head_type=kind,head_mask=np.ones((n,h),bool),
                 context=np.zeros((n,2),"float32"),
                 baseline=np.eye(7,dtype="float32")[base])
        target,risk=form_targets(y,base,props)
        model=LearnedAuditSelector(hidden=32)
        _=model(inp)
        opt=tf.keras.optimizers.Adam(learning_rate=.012)
        for _ in range(100):
            with tf.GradientTape() as tape:
                out=model(inp,training=True)
                loss=train_loss(out,target,risk,inp["head_mask"],risk_weight=.10)
            grads=tape.gradient(loss,model.trainable_variables)
            opt.apply_gradients(zip(grads,model.trainable_variables))
        out=model(inp,training=False)
        accuracy=np.mean(np.argmax(out["action_logits"].numpy(),axis=1)==target)
        self.assertGreater(accuracy,.90,
                           "two weak specialists must become useful in combination")

    def test_audits_are_training_only_and_stratified(self):
        y=np.array([2,2,3,3,4,4])
        base=np.array([3,3,3,4,4,3])
        p=np.array([[2,3],[3,2],[2,4],[2,4],[4,2],[3,4]])
        folds=np.array([0,0,1,1,2,2])
        profile=audited_head_descriptor(y,base,p,folds,folds!=2)
        self.assertEqual(profile.shape,(2,7,3))
        self.assertEqual(profile[0,4,2],0)
        self.assertGreater(profile[0,2,0],0)
        self.assertTrue(check_oof_contract(
            np.array([0,1]),
            [(1,2,4),(0,2,4)],["model-A","model-B"]
        ))
        with self.assertRaises(ValueError):
            check_oof_contract(
                np.array([0,1]),[(0,2,4),(0,2,4)],["A","B"]
            )

    def test_historical_head_cannot_be_pretended_active(self):
        with self.assertRaises(ValueError):
            HeadContract("historic-id","fix","scripts/old_fix.py",25.,valid_oof=False).validate()
        HeadContract("new-id","correction","scripts/verified.py",25.,valid_oof=True).validate()

if __name__=="__main__":
    unittest.main()
