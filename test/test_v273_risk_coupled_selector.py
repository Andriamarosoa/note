"""Risk-coupled neural selector must use learned reliability FOR the decision."""
import unittest
import numpy as np
import tensorflow as tf
from scripts.learn_v273_risk_coupled_selector import (
    RiskCoupledClassSelector, risk_targets, risk_coupled_loss, fit_risk_selector,
)
from scripts.evaluate_v273_neural_history_mix import (
    H,build_candidate_logits, inputs, KEEP
)


class RiskGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(27402)

    def fixture(self):
        rng=np.random.default_rng(438)
        n=16
        base=np.array([2,2,2,2,3,3,3,3,4,4,4,4,2,3,4,2])
        P=rng.uniform(.1,1.,(n,6,5)).astype(np.float32)
        P/=P.sum(axis=-1,keepdims=True)
        logit,mask,types,support=build_candidate_logits(P,base)
        audit=rng.uniform(0,.2,(n,H,21)).astype(np.float32)
        context=rng.normal(size=(n,5)).astype(np.float32)
        feat=inputs(logit,mask,types,support,audit,context,base)
        true=np.array([2,3,0,1,3,2,4,1,4,2,3,0,5,3,4,2])
        return feat,true,base

    def test_risk_output_has_only_supported_final_actions(self):
        feat,y,base=self.fixture()
        nn=RiskCoupledClassSelector(hidden=24)
        out=nn(feat)
        scores=out["action_logits"].numpy()
        self.assertEqual(scores.shape,(len(y),8))
        self.assertTrue(np.all(scores[:,:2]<-1e8))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(out["class_selection_weights"].shape,(len(y),H,7))
        self.assertEqual(out["head_risk"].shape,(len(y),H,2))
        self.assertEqual(out["selected_class_risk"].shape,(len(y),7,2))
        self.assertTrue(np.all(np.isin(np.argmax(scores,axis=1),[2,3,4,5,6,KEEP])))

    def test_loss_reaches_risk_predictions_and_original_neural_backbone(self):
        feat,y,base=self.fixture()
        nn=RiskCoupledClassSelector(hidden=24)
        _=nn(feat)
        act,correct,keep,hr=risk_targets(
            y,base,np.argmax(feat["head_logits"],axis=-1))
        with tf.GradientTape() as tape:
            output=nn(feat,training=True)
            loss=risk_coupled_loss(output,act,correct,keep,hr,
                                  feat["head_mask"],y,base)
        gradients=tape.gradient(loss,nn.trainable_variables)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(g is not None for g in gradients),
                        "a trainable layer is disconnected from the loss")
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g)))
                            for g in gradients))
        with tf.GradientTape() as tape2:
            output=nn(feat,training=False)
            # Only ACTION loss, NO risk auxiliary losses.
            actiononly=tf.reduce_sum(output["action_logits"][:,3])
        grads=tape2.gradient(actiononly,nn.acoustic.risk.trainable_variables)
        self.assertTrue(all(g is not None for g in grads),
                        "head-risk branch must change the final K3 score")
        self.assertTrue(any(float(tf.reduce_sum(tf.abs(g)))>0 for g in grads))

    def test_nonpoly_truth_is_not_fabricated_as_action_head(self):
        feat,y,base=self.fixture()
        actions,correct,keep,hr=risk_targets(
            y,base,np.argmax(feat["head_logits"],axis=-1))
        self.assertTrue(np.all(actions[y<2]==KEEP))
        self.assertTrue(np.all(keep[y<2]==1))
        self.assertTrue(np.all(actions[(y>=2)&(y!=base)]==y[(y>=2)&(y!=base)]))
        self.assertEqual(correct.shape,(len(y),7))
        self.assertEqual(hr.shape,(len(y),H,2))

    def test_one_real_training_step_updates_risk_heads(self):
        feat,y,base=self.fixture()
        model=RiskCoupledClassSelector(hidden=24)
        action,class_true,keep,hr=risk_targets(
            y,base,np.argmax(feat["head_logits"],axis=-1))
        opt=tf.keras.optimizers.Adam(learning_rate=.002)
        out=model(feat)
        before=model.candidate_risk_out.kernel.numpy().copy()
        for i in range(3):
            with tf.GradientTape() as tape:
                o=model(feat,training=True)
                loss=risk_coupled_loss(o,action,class_true,keep,hr,
                                       feat["head_mask"],y,base)
            grads=tape.gradient(loss,model.trainable_variables)
            opt.apply_gradients(zip(grads,model.trainable_variables))
        self.assertFalse(np.allclose(before,model.candidate_risk_out.kernel.numpy()))


if __name__=="__main__":
    unittest.main()
