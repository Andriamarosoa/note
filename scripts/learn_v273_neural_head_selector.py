"""Neural selection of historical predictors, features, corrections and fixes.

This is a trainable MULTI-HEAD SELECTOR, not a static audited rule table.
All head kinds are first represented by an adapter emitting the same
K0..K6 + KEEP action distribution. This selector learns:
  - selection logits per event/head (softmax);
  - nonlinear pairwise head interactions via self-attention;
  - action logits over K0..K6 and KEEP;
  - correction/regression risk per head (auxiliary supervised targets).

Audit features are train-only OOF statistics (by TRUE K for training and
by observed original -> proposed transition), not labels of the evaluated
event. They are *inputs* to trainable layers, never hand-computed weights.

A head, fix or correction is inactive until validated aligned OOF adapter
data exist; entries from the historical registry are NOT silently activated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import tensorflow as tf

ACTION_CLASSES = 8
KEEP = 7
ACTOR_TYPES = ("feature","predictor","correction","fix")
EPS=1e-7

@dataclass(frozen=True)
class HeadContract:
    uid: str
    type: str
    source_path: str
    native_latency_ms: float
    output_space: str = "K0-K6+KEEP"
    valid_oof: bool = False

    def validate(self):
        if self.type not in ACTOR_TYPES:raise ValueError(f"unrecognized head type: {self.type}")
        if self.output_space!="K0-K6+KEEP":raise ValueError("adapter must align to eight actions")
        if self.native_latency_ms<0:raise ValueError("invalid latency")
        if not self.valid_oof:raise ValueError(f"{self.uid}: missing aligned OOF adapter")
        return self

def form_targets(y,base,proposal):
    """Training-only supervision for head risk and optimal action.

    If freeze_local_combo already has the true K, KEEP is the desired
    behavior. Otherwise the corrective target is the true class.
    All values y are training targets, never input features.
    """
    y=np.asarray(y,int);base=np.asarray(base,int)
    proposal=np.asarray(proposal,int)
    if y.shape!=base.shape or proposal.shape[0]!=len(y):
        raise ValueError("target shapes mismatch")
    if np.any((y<0)|(y>6)|(base<0)|(base>6)):raise ValueError("bad K")
    if np.any((proposal<0)|(proposal>KEEP)):raise ValueError("bad action class")
    actions=np.where(y==base,KEEP,y).astype("int32")
    effective=np.where(proposal==KEEP,base[:,None],proposal)
    changed=effective!=base[:,None]
    corrected=(changed&(effective==y[:,None])).astype("float32")
    regressed=(changed&(y[:,None]==base[:,None])).astype("float32")
    return actions, np.stack([corrected,regressed],axis=-1)

def audited_head_descriptor(y,base,proposals,fold_ids,train_mask,alpha=16.):
    """OOF audit features to feed a neural selector, never fixed weights.

    Repeats the historical profile (7 true-K rows * 3 stats) for each head.
    For each held-out row, this function must be called with an explicit
    training-mask excluding that row's entire evaluation fold.
    """
    y=np.asarray(y,int);base=np.asarray(base,int)
    p=np.asarray(proposals,int)
    f=np.asarray(fold_ids)
    train_mask=np.asarray(train_mask,bool)
    if len(y)!=len(base) or p.shape[0]!=len(y) or len(f)!=len(y):
        raise ValueError("bad source dimensions")
    if len(train_mask)!=len(y) or not train_mask.any():
        raise ValueError("invalid audit training mask")
    n,h=p.shape
    # Convert KEEP to the original model's class for pairwise audit.
    actual=np.where(p==KEEP,base[:,None],p)
    changed=(actual!=base[:,None])
    out=np.zeros((h,7,3),np.float32)
    for k in range(7):
        rows=train_mask&(y==k)
        # Correction / regression / coverage with conservative Bayesian shrinkage.
        denom=float(rows.sum())+alpha
        for j in range(h):
            out[j,k,0]=np.sum(rows&changed[:,j]&(actual[:,j]==y))/denom
            out[j,k,1]=np.sum(rows&changed[:,j]&(base==y))/denom
            out[j,k,2]=np.sum(rows&changed[:,j])/denom
    return out

class LearnedAuditSelector(tf.keras.Model):
    """Shared trainable head encoder + interactions + soft selection + action.

    Inputs
      head_logits: [B,H,8], logits for 7 K proposals and KEEP, from OOF adapters.
      audit:       [B,H,21], training-only per-head audit profiles.
      head_type:   [B,H,4], one-hot feature/model/correction/fix.
      head_mask:   [B,H], one means adapter available and time-aligned.
      context:     [B,F], observed audio embeddings and original model confidence.
      baseline:    [B,7], original model K one-hot.
    Outputs
      action_logits: [B,8], choice K0..K6 or KEEP.
      selection_weights: [B,H], strictly zero on absent heads.
      head_risk: [B,H,2], learned likelihood of correction/regression.
    """

    def __init__(self, hidden=48, context_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.proj=tf.keras.layers.Dense(hidden,activation="gelu")
        self.context_projection=tf.keras.layers.Dense(hidden,activation="gelu")
        self.att=tf.keras.layers.MultiHeadAttention(
            num_heads=3,key_dim=hidden//3,dropout=0.0
        )
        self.fuse=tf.keras.layers.Dense(hidden,activation="gelu")
        self.gate=tf.keras.layers.Dense(1)
        self.risk=tf.keras.layers.Dense(2,activation="sigmoid")
        self.action_hidden=tf.keras.layers.Dense(hidden,activation="gelu")
        self.action_out=tf.keras.layers.Dense(ACTION_CLASSES)
        self.context_dim=context_dim

    def call(self,inputs,training=False):
        logits=tf.cast(inputs["head_logits"],tf.float32)
        audits=tf.cast(inputs["audit"],tf.float32)
        kind=tf.cast(inputs["head_type"],tf.float32)
        mask=tf.cast(inputs["head_mask"],tf.bool)
        ctx=tf.cast(inputs["context"],tf.float32)
        baseline=tf.cast(inputs["baseline"],tf.float32)
        tf.debugging.assert_equal(tf.shape(logits)[-1],ACTION_CLASSES)
        tf.debugging.assert_equal(tf.shape(audits)[-1],21)
        tf.debugging.assert_equal(tf.shape(kind)[-1],4)
        tf.debugging.assert_equal(tf.shape(baseline)[-1],7)
        tf.debugging.assert_greater(
            tf.reduce_min(tf.reduce_sum(tf.cast(mask,tf.int32),axis=1)),0,
            message="every event needs at least the baseline head"
        )
        x=self.proj(tf.concat([tf.nn.softmax(logits,axis=-1),audits,kind],-1))
        c=self.context_projection(tf.concat([ctx,baseline],axis=-1))
        # Pairwise learnable interactions: heads must be able to change
        # one another's evidence, not only contribute isolated votes.
        am=tf.logical_and(mask[:,:,None],mask[:,None,:])
        attended=self.att(x,x,attention_mask=am,training=training)
        state=self.fuse(x+attended+c[:,None,:])
        # Prevent masked heads from receiving any probability mass.
        scores=tf.squeeze(self.gate(state),axis=-1)
        scores=tf.where(mask,scores,tf.constant(-1.e9,tf.float32))
        weights=tf.nn.softmax(scores,axis=1)
        weights=tf.where(mask,weights,tf.zeros_like(weights))
        z=tf.reduce_sum(state*weights[:,:,None],axis=1)
        # A weighted logit ensemble alone cannot express head interactions.
        # The learned nonlinear decoder receives both the attended state
        # and class evidence from the selected heads.
        proposals=tf.reduce_sum(
            tf.nn.softmax(logits,axis=-1)*weights[:,:,None],axis=1
        )
        decision=self.action_out(self.action_hidden(
            tf.concat([z,proposals,ctx,baseline],axis=-1)
        ))
        risk=self.risk(state)
        return dict(action_logits=decision,
                    selection_weights=weights,head_risk=risk)



class ClassConditionalAuditSelector(LearnedAuditSelector):
    """Neural gate with one learned selection distribution FOR EACH possible K.

    A head's structural availability (adapter usable for the signal) is
    distinct from its possible target-K support. Both are input-observable.
    Audit features are cross-fitted on training folds, NEVER true-K at
    inference. Unsupported class/head pairs get *exactly zero* weight;
    compatible pairs remain free to learn context-dependent weights.

    head_k_mask: [B,H,7] structural eligibility for a head to inform each
                 hypothetical output K, independent of true-K labels.
    class_selection_weights: [B,H,7] learned conditional selection.
    selection_weights: [B,H] marginal over model-inferred possible K; a
                       backward-compatible descriptive summary, NOT gate.
    """

    def __init__(self, hidden=48, context_dim=16, **kwargs):
        super().__init__(hidden=hidden,context_dim=context_dim,**kwargs)
        self.class_gate=tf.keras.layers.Dense(7)
        self.class_hidden=tf.keras.layers.Dense(hidden,activation="gelu")
        self.class_out=tf.keras.layers.Dense(1)
        self.keep_hidden=tf.keras.layers.Dense(hidden,activation="gelu")
        self.keep_out=tf.keras.layers.Dense(1)

    def call(self,inputs,training=False):
        logits=tf.cast(inputs["head_logits"],tf.float32)
        audits=tf.cast(inputs["audit"],tf.float32)
        kind=tf.cast(inputs["head_type"],tf.float32)
        mask=tf.cast(inputs["head_mask"],tf.bool)
        ctx=tf.cast(inputs["context"],tf.float32)
        baseline=tf.cast(inputs["baseline"],tf.float32)

        tf.debugging.assert_equal(tf.shape(logits)[-1],ACTION_CLASSES)
        tf.debugging.assert_equal(tf.shape(audits)[-1],21)
        tf.debugging.assert_equal(tf.shape(kind)[-1],4)
        tf.debugging.assert_equal(tf.shape(baseline)[-1],7)
        tf.debugging.assert_greater(
            tf.reduce_min(tf.reduce_sum(tf.cast(mask,tf.int32),axis=1)),0,
            message="each event requires an active baseline/fallback"
        )

        if "head_k_mask" in inputs:
            class_mask=tf.cast(inputs["head_k_mask"],tf.bool)
            tf.debugging.assert_equal(tf.shape(class_mask)[:2],tf.shape(mask))
            tf.debugging.assert_equal(tf.shape(class_mask)[-1],7)
            class_mask=tf.logical_and(class_mask,mask[:,:,None])
        else:
            class_mask=tf.broadcast_to(mask[:,:,None],tf.concat([
                tf.shape(mask),[7]],axis=0))

        tf.debugging.assert_greater(
            tf.reduce_min(tf.reduce_sum(tf.cast(class_mask,tf.int32),axis=1)),0,
            message="every hypothetical K must retain an observable fallback"
        )

        features=tf.concat([tf.nn.softmax(logits,axis=-1),audits,kind],axis=-1)
        heads=self.proj(features)
        c=self.context_projection(tf.concat([ctx,baseline],axis=-1))
        attention_mask=tf.logical_and(mask[:,:,None],mask[:,None,:])
        interacted=self.att(heads,heads,attention_mask=attention_mask,training=training)
        state=self.fuse(heads+interacted+c[:,None,:])

        # Critically: softmax is OVER HEADS for EACH K, not over one global
        # head axis with weights reused for every K.
        class_scores=self.class_gate(state)           # [B,H,7]
        negative=tf.constant(-1.e9,tf.float32)
        class_scores=tf.where(class_mask,class_scores,negative)
        weights_by_k=tf.nn.softmax(class_scores,axis=1)
        weights_by_k=tf.where(class_mask,weights_by_k,tf.zeros_like(weights_by_k))
        weights_by_k/=tf.reduce_sum(weights_by_k,axis=1,keepdims=True)

        class_states=tf.einsum("bhk,bhd->bkd",weights_by_k,state)
        post=tf.nn.softmax(logits,axis=-1)
        class_evidence=tf.reduce_sum(weights_by_k*post[:,:,:7],axis=1)
        n=tf.shape(ctx)[0]
        context_by_k=tf.broadcast_to(c[:,None,:],tf.shape(class_states))
        identities=tf.broadcast_to(tf.eye(7,dtype=tf.float32)[None,:,:],
                                   tf.stack([n,7,7]))
        class_features=tf.concat([
            class_states,context_by_k,class_evidence[:,:,None],identities
        ],axis=-1)
        logits_by_k=tf.squeeze(
            self.class_out(self.class_hidden(class_features)),axis=-1
        )

        # KEEP is separate and can learn from fixes/guards which need
        # not represent any particular target K. Head masks still apply.
        keep_scores=tf.squeeze(self.gate(state),axis=-1)
        keep_scores=tf.where(mask,keep_scores,negative)
        keep_weights=tf.nn.softmax(keep_scores,axis=1)
        keep_state=tf.reduce_sum(state*keep_weights[:,:,None],axis=1)
        keep_prob=tf.reduce_sum(post[:,:,KEEP]*keep_weights,axis=1)
        keep_logits=self.keep_out(self.keep_hidden(tf.concat([
            keep_state,c,ctx,baseline,keep_prob[:,None]
        ],axis=-1)))
        decision=tf.concat([logits_by_k,keep_logits],axis=1)

        # Descriptive marginal weights use inferred classes, never true K.
        inferred_k=tf.nn.softmax(logits_by_k,axis=1)
        marginal=tf.einsum("bhk,bk->bh",weights_by_k,inferred_k)
        # Risk supervision remains per head from cross-fitted audits.
        risk=self.risk(state)
        return dict(
            action_logits=decision,
            selection_weights=marginal,
            class_selection_weights=weights_by_k,
            class_probability=inferred_k,
            keep_selection_weights=keep_weights,
            head_risk=risk
        )

def train_loss(outputs,actions,correct_regress,mask,risk_weight=.20):
    """Differentiable neural classification plus per-head risk supervision."""
    a=tf.convert_to_tensor(actions,tf.int32)
    labels=tf.cast(correct_regress,tf.float32)
    mask=tf.cast(mask,tf.float32)
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=a, logits=outputs["action_logits"]
    )
    p=tf.clip_by_value(outputs["head_risk"],EPS,1-EPS)
    bce=-(labels*tf.math.log(p)+(1-labels)*tf.math.log(1-p))
    risk=tf.reduce_sum(bce*mask[:,:,None])/(2.0*tf.reduce_sum(mask)+EPS)
    return tf.reduce_mean(ce)+risk_weight*risk

def check_oof_contract(train_fold,source_train_folds,source_model_ids):
    """Fail fast on reused in-sample / target-trained head predictions."""
    fold=np.asarray(train_fold)
    if len(source_train_folds)!=len(fold) or len(source_model_ids)!=len(fold):
        raise ValueError("provenance dimension mismatch")
    for i,(source,model_id) in enumerate(zip(source_train_folds,source_model_ids)):
        if model_id is None or fold[i] in set(source):
            raise ValueError(f"head output {i} leaks its validation fold")
    return True
