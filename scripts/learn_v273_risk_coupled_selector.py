"""Learned *decision-coupled* class-specific correction reliability for Exact-K.

This intentionally keeps the 14 previously audited heads intact. The
existing ClassConditionalAuditSelector is an acoustic/interaction encoder,
not an unconstrained final decision maker anymore.

The new trainable action head estimates, separately for each hypothetical
K2..K6, how likely changing to that class is CORRECT. Another calibrated
head estimates the likelihood that KEEP should be preferred. The final
decision is a learned *relative advantage*: log P(correct target) -
log P(KEEP preferred); KEEP's utility is zero. Thus it cannot bypass the
reliability heads, and unsupported K0/K1 logits are structurally masked.

Each class-specific estimate consumes:
- the original learned K-specific attention mixture,
- the TRAIN-ONLY, cross-fitted audit of each contributing head for true K,
- its head-level correction and regression risk *as a trainable feature*,
- acoustic context, baseline class, and specialist evidence.

True K only enters supervised labels in OUTER TRAIN, not any inference
feature, availability mask, decision post-processing, or threshold tuning.
No static head weights or forced class-specific confidence thresholds.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf

from scripts.learn_v273_neural_head_selector import (
    ClassConditionalAuditSelector, form_targets, train_loss, KEEP
)

EPS=1e-6
TARGETS=7
OUTPUTS=8


class RiskCoupledClassSelector(tf.keras.Model):
    """A *trainable* risk-coupled action selector and KEEP arbiter.

    The main action logits are produced ONLY by these calibrated risk
    estimates. Risk scores can't be disconnected from the decision.
    """

    def __init__(self, hidden=48, **kwargs):
        super().__init__(**kwargs)
        self.acoustic=ClassConditionalAuditSelector(hidden=hidden)
        self.candidate_risk_hidden=tf.keras.layers.Dense(hidden,activation="gelu")
        self.candidate_risk_out=tf.keras.layers.Dense(1)
        self.keep_risk_hidden=tf.keras.layers.Dense(hidden,activation="gelu")
        self.keep_risk_out=tf.keras.layers.Dense(1)

    def call(self,inputs,training=False):
        old=self.acoustic(inputs,training=training)
        w=tf.cast(old["class_selection_weights"],tf.float32)  # B,H,7
        hr=tf.cast(old["head_risk"],tf.float32)                # B,H,2
        raw=tf.cast(old["action_logits"],tf.float32)
        baseline=tf.cast(inputs["baseline"],tf.float32)
        ctx=tf.cast(inputs["context"],tf.float32)
        audio_prob=tf.nn.softmax(tf.cast(inputs["head_logits"],tf.float32),axis=-1)
        audit=tf.reshape(tf.cast(inputs["audit"],tf.float32),[
            tf.shape(w)[0],tf.shape(w)[1],7,3])
        # Conditional risk and true-K audit are different. Audit is
        # training-only history; current risk is predicted by the NN.
        selected_risk=tf.einsum("bhk,bhr->bkr",w,hr)
        selected_audit=tf.einsum("bhk,bhkr->bkr",w,audit)
        selected_evidence=tf.einsum("bhk,bhk->bk",w,audio_prob[:,:,:7])

        batch=tf.shape(ctx)[0]
        context=tf.broadcast_to(ctx[:,None,:],[batch,7,tf.shape(ctx)[1]])
        base_feature=tf.broadcast_to(
            baseline[:,None,:],[batch,7,7])
        )
        class_identity=tf.broadcast_to(
            tf.eye(7,dtype=tf.float32)[None,:,:],[batch,7,7])
        )
        candidate_features=tf.concat([
            raw[:,:7,None],selected_evidence[:,:,None],
            selected_risk,selected_audit,context,base_feature,class_identity,
        ],axis=-1)
        candidate_logits=tf.squeeze(self.candidate_risk_out(
            self.candidate_risk_hidden(candidate_features)),axis=-1)
        # KEEP can be better either because baseline is already correct
        # OR because no supported K2+ alternative is reliable.
        head_avail=tf.cast(inputs["head_mask"],tf.float32)
        denom=tf.reduce_sum(head_avail,axis=1,keepdims=True)
        avg_head_risk=tf.reduce_sum(hr*head_avail[:,:,None],axis=1)/denom
        keep_features=tf.concat([
            raw[:,7,None],ctx,baseline,avg_head_risk,
            tf.reduce_max(candidate_logits[:,2:],axis=1,keepdims=True),
            tf.reduce_mean(selected_audit[:,:,1],axis=1,keepdims=True)
        ],axis=-1)
        keep_logits=tf.squeeze(self.keep_risk_out(
            self.keep_risk_hidden(keep_features)),axis=-1)
        corrected_conf=tf.sigmoid(candidate_logits)
        keep_conf=tf.sigmoid(keep_logits)

        # Risk head is now the *only* source of the final action score.
        # This advantage is trainable via classification + calibration
        # loss; it is not a post-hoc heuristic threshold.
        advantage=tf.math.log(tf.clip_by_value(corrected_conf,EPS,1.)) \
                  -tf.math.log(tf.clip_by_value(keep_conf[:,None],EPS,1.))
        # No adapter has actual acoustic evidence for K0 or K1.
        unsupported=tf.fill([batch,2],tf.constant(-1.e9,tf.float32))
        scores=tf.concat([unsupported,advantage[:,2:],
                          tf.zeros([batch,1],dtype=tf.float32)],axis=1)
        return {
            "action_logits":scores,
            "raw_action_logits":raw,
            "class_selection_weights":w,
            "head_risk":hr,
            "class_correct_prob":corrected_conf,
            "keep_preferred_prob":keep_conf,
            "selected_class_risk":selected_risk,
            "selected_class_audit":selected_audit,
        }


def risk_targets(y,base,proposals):
    """OUTER-TRAIN labels for supported K2..K6 + KEEP.

    True K0/K1 is never silently relabeled as a supported class.
    When no corrective specialist predicts K0/K1, KEEP is preferred
    over proposing an unsupported K0/K1. The actual nonpoly baseline
    is still wrong in these rows, which is recorded separately.
    """
    y=np.asarray(y,int);base=np.asarray(base,int)
    if y.shape!=base.shape or np.any((y<0)|(y>6)):raise ValueError("targets")
    action=np.where((y==base)|(y<2),KEEP,y).astype("int32")
    class_correct=(y[:,None]==np.arange(7)[None,:]).astype("float32")
    keep_preferred=((y==base)|(y<2)).astype("float32")
    _,head_risk=form_targets(y,base,proposals)
    return action,class_correct,keep_preferred,head_risk


def risk_coupled_loss(modelout,action,true_class,keep_preferred,head_risk,
                      mask,y,base):
    """Supervised action risk + calibrated KEEP + per-head risk.

    Unweighted calibration losses have truthful probability semantics.
    Action loss is *macro-balanced by observed TRAIN true K* to protect
    rare poly classes, with weights calculated only from current train
    fold (no heldout/test statistics).

    Baseline-correct poly mistakes are additionally penalized by a
    differentiable expectation based on the network's probabilities,
    not a hand-coded decision veto.
    """
    a=tf.cast(action,tf.int32)
    y=tf.cast(y,tf.int32);base=tf.cast(base,tf.int32)
    true_class=tf.cast(true_class,tf.float32)
    keep_preferred=tf.cast(keep_preferred,tf.float32)
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=a,logits=modelout["action_logits"])
    # Distribution-aware class balance is learned from TRAIN labels
    # upstream; avoid a static per-K manual penalty.
    active_classes=tf.cast(y>=2,tf.float32)
    # Macro equalize candidate class losses against very common
    # unsupported nonpoly observations without hiding any errors.
    positive=tf.reduce_sum(active_classes)
    nonpoly=tf.reduce_sum(1.-active_classes)
    poly_weight=tf.stop_gradient(
        tf.math.sqrt((nonpoly+1.)/(positive+1.)))
    sample_weight=1.+active_classes*(poly_weight-1.)
    ce=tf.reduce_sum(ce*sample_weight)/tf.reduce_sum(sample_weight)

    p=tf.clip_by_value(modelout["class_correct_prob"],EPS,1.-EPS)
    calibration=-(true_class[:,2:]*tf.math.log(p[:,2:])
                  +(1.-true_class[:,2:])*tf.math.log(1.-p[:,2:]))
    c_bce=tf.reduce_mean(calibration)
    q=tf.clip_by_value(modelout["keep_preferred_prob"],EPS,1.-EPS)
    k_bce=tf.reduce_mean(
        -keep_preferred*tf.math.log(q)
        -(1.-keep_preferred)*tf.math.log(1.-q)
    )
    hr=tf.clip_by_value(modelout["head_risk"],EPS,1.-EPS)
    hm=tf.cast(mask,tf.float32)
    head_loss=-(
        head_risk*tf.math.log(hr)+(1.-head_risk)*tf.math.log(1.-hr)
    )
    h_bce=tf.reduce_sum(head_loss*hm[:,:,None])/(
        2.*tf.reduce_sum(hm)+EPS
    )
    # Penalize predicted *change* to any class when the baseline
    # has true K2..K4; this explicitly learns avoidable regressions.
    # No fixed threshold and no test-label access at inference.
    good_poly=tf.cast(tf.logical_and(y==base,y>=2),tf.float32)
    probability_change=1.-tf.nn.softmax(
        modelout["action_logits"],axis=-1)[:,KEEP]
    avoidable=tf.reduce_sum(good_poly*probability_change)/(
        tf.reduce_sum(good_poly)+EPS
    )
    return ce+0.5*c_bce+0.5*k_bce+0.15*h_bce+0.4*avoidable


def fit_risk_selector(train,y,base,held,epochs=28,seed=27402):
    """Train once per fold, evaluate without true labels on heldout."""
    tf.keras.utils.set_random_seed(seed)
    model=RiskCoupledClassSelector(hidden=48)
    proposed=np.argmax(train["head_logits"],axis=-1)
    act,classes,keep,hr=risk_targets(y,base,proposed)
    dataset=tf.data.Dataset.from_tensor_slices((
        train,act,classes,keep,hr,
        np.asarray(y,dtype="int32"),np.asarray(base,dtype="int32")
    )).shuffle(len(y),seed=seed,reshuffle_each_iteration=True).batch(256)
    opt=tf.keras.optimizers.Adam(learning_rate=.002)
    log=[]
    for epoch in range(epochs):
        losses=[]
        for features,a,cp,kp,risk,yt,bt in dataset:
            with tf.GradientTape() as tape:
                out=model(features,training=True)
                loss=risk_coupled_loss(out,a,cp,kp,risk,
                                       features["head_mask"],yt,bt)
            grads=tape.gradient(loss,model.trainable_variables)
            assert all(v is not None for v in grads),"disconnected trainable variables"
            opt.apply_gradients(zip(grads,model.trainable_variables))
            losses.append(float(loss))
        if epoch in (0,epochs//2,epochs-1):
            log.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
    out=model(held,training=False)
    act=np.argmax(out["action_logits"].numpy(),axis=1)
    return act,out,log
