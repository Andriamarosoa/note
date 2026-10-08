"""Actual neural 64-subset source->target Exact-K correction risk arbiter.

Four adjacent corrective source/target transitions have seven structurally
available heads H0..H5 plus C23/C32/C34/C43 -> 64 H0-preserving
combinations. All others have H0..H5 -> 32 combinations.
The model learns per-event, source-K, target-K SUBSET ATTENTION and the
probability of a successful correction and of a destroyed correct baseline.

Every non-empty H0-preserving subset's own OOF correction/regression
audit (by TRUE K) is computed on OTHER inner folds, and exposed only as
features. Test labels are NEVER used in model selection or prediction.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from scripts.audit_v273_kconditional_127_subsets import votes
from scripts.evaluate_v273_neural_history_mix import (
    build_candidate_logits,HEADS,KEEP
)

K_RANGE=(2,3,4,5,6)
SRC_RANGE=(2,3,4)
CORRECTION={(2,3):6,(3,2):7,(3,4):8,(4,3):9}
COMBOS=64
FEAT_DIM=10
EPS=1e-6
SEED=27402
# Learned subsets contain H0, may also contain H1..H5, and the
# directional Cxy only when that adapter is actually available.
COMBO_BINARY=np.array([
    [1]+[int(bool(i&(1<<j))) for j in range(6)]
    for i in range(COMBOS)
],dtype=np.float32)


def _audits(mask,margin,y,src,dst,alpha=12.):
    """Independent per-subset historical correction/regression statistics.

    64 outputs, each with coverage, correction conditional on a proposal,
    regression conditional on a proposal, neutral conditional on proposal.
    'mask' is a binary prediction matrix (events, 64).
    """
    eligible=mask.sum(axis=0).astype(np.float64)
    corrections=((y==dst)[:,None]*mask).sum(axis=0)
    regressions=((y==src)[:,None]*mask).sum(axis=0)
    neutral=eligible-corrections-regressions
    den=eligible+alpha
    desc=np.stack([
        np.log1p(eligible),
        (corrections+alpha/3.)/den,
        (regressions+alpha/3.)/den,
        (neutral+alpha/3.)/den,
    ],axis=1).astype(np.float32)
    return desc, (eligible,corrections,regressions,neutral)


def compute_direct_options(P,b):
    """Generate OBSERVABLE K-conditional direct-vote subset indicators.

    Returns (N,5,64,6), (N,5,64) availability. All 14 eventual action
    heads are retained through KEEP metadata elsewhere. No true labels.
    """
    P=np.asarray(P,np.float32);b=np.asarray(b,int)
    n=len(b)
    if P.shape!=(n,6,5) or not np.isin(b,SRC_RANGE).all():
        raise ValueError("need cross-fitted 6x5 posteriors and K2/K3/K4 baseline")
    opt=np.zeros((n,5,COMBOS,6),np.float32)
    avail=np.zeros((n,5,COMBOS),bool)
    for src in SRC_RANGE:
        rows=np.flatnonzero(b==src)
        if not len(rows):continue
        for j,dst in enumerate(K_RANGE):
            if src==dst:continue
            correction=CORRECTION.get((src,dst))
            heads=(0,1,2,3,4,5) if correction is None else (
                0,1,2,3,4,5,correction)
            stay,replace=votes(P[rows],src,dst,heads)
            full=np.zeros((len(rows),7),np.float32)
            full[:,:len(heads)]=stay
            stay=full
            real=np.zeros_like(full)
            real[:,:len(heads)]=replace
            masks=COMBO_BINARY.copy()
            if correction is None:masks[:,6]=0
            counts=np.sum(masks,axis=1)
            # When C is not available, 32 duplicate masks are hidden
            # rather than double-counting equivalent subsets.
            good=np.ones(COMBOS,bool)
            if correction is None:good[32:]=False
            avg_keep=stay @ masks.T/counts[None,:]
            avg_target=real @ masks.T/counts[None,:]
            margin=avg_target-avg_keep
            ix=np.ix_(rows,[j],np.arange(COMBOS))
            # Assign through intermediate to avoid numpy advanced index pitfalls.
            local=np.zeros((len(rows),COMBOS,6),np.float32)
            local[:,:,0]=np.clip(margin,-1,1)
            local[:,:,1]=avg_keep
            local[:,:,2]=avg_target
            local[:,:,3]=(margin>0).astype(np.float32)
            local[:,:,4]=counts[None,:]/7.
            local[:,:,5]=float(correction is not None)
            opt[rows,j,:,:]=local
            avail[rows,j,:]=good[None,:]
    return opt,avail


def attach_oof_subset_audits(train_opt,train_available,train_b,
                             train_y,train_fold,held_opt,held_available,held_b):
    """Foldwise OOF auditor: per-row descriptors trained on OTHER inner folds.

    For heldout inference, descriptors come from the whole outer TRAIN.
    True labels are not placed in the decoder's forward interface.
    """
    y=np.asarray(train_y,int)
    bs=np.asarray(train_b,int)
    folds=np.asarray(train_fold,int)
    if len(np.unique(folds))<2:raise ValueError("not enough inner folds")
    full_train=np.zeros((*train_opt.shape[:3],FEAT_DIM),np.float32)
    full_held=np.zeros((*held_opt.shape[:3],FEAT_DIM),np.float32)
    full_train[...,:6]=train_opt
    full_held[...,:6]=held_opt
    records=[]
    for src in SRC_RANGE:
        for j,dst in enumerate(K_RANGE):
            if src==dst:continue
            source=bs==src
            vals={}
            for withheld in (list(np.unique(folds))+["heldout"]):
                ref=source if withheld=="heldout" else source&(folds!=withheld)
                if not np.any(ref):raise ValueError("missing training inner evidence")
                mask=train_opt[ref,j,:,3]>0
                audit,stats=_audits(mask,train_opt[ref,j,:,0],y[ref],src,dst)
                if withheld=="heldout":
                    receiver=np.flatnonzero(held_b==src)
                    if len(receiver):
                        full_held[receiver,j,:,6:]=audit[None,:,:]
                else:
                    receiver=np.flatnonzero((bs==src)&(folds==withheld))
                    if len(receiver):
                        full_train[receiver,j,:,6:]=audit[None,:,:]
                if withheld=="heldout":
                    for subset in range(COMBOS):
                        if not np.any(train_available[ref,j,subset]):continue
                        records.append(dict(
                            baseline_K=int(src),candidate_K=int(dst),
                            subset_id=int(subset),
                            heads="+".join(["H0"]+
                                ["H"+str(h) for h in range(1,6)
                                 if COMBO_BINARY[subset,h]]+
                                ([HEADS[CORRECTION[(src,dst)]]] if (src,dst) in CORRECTION
                                    and COMBO_BINARY[subset,6] else [])),
                            train_oof_reference_rows=int(ref.sum()),
                            train_oof_proposals=int(stats[0][subset]),
                            train_oof_corrections=int(stats[1][subset]),
                            train_oof_regressions=int(stats[2][subset]),
                            train_oof_neutral=int(stats[3][subset]),
                            audit_correction_rate=float(audit[subset,1]),
                            audit_regression_rate=float(audit[subset,2])
                        ))
    if not np.isfinite(full_train).all() or not np.isfinite(full_held).all():
        raise ValueError("bad prior")
    return full_train,full_held,records


def build_inputs(P,b,ctx,fold=None,truth=None,held=None):
    raise NotImplementedError("Use compute_direct_options and attach_oof_subset_audits")


class TransitionCombinationArbiter(tf.keras.Model):
    """Learn nonlinear combination routing conditioned on baseline and target.

    Outputs 5 candidate risks+1 KEEP. Every candidate scorer is tied to
    its source/target identities, subset-specific auditable priors and audio.
    """

    def __init__(self,width=32,**kw):
        super().__init__(**kw)
        self.subset_embed=tf.keras.layers.Dense(width,activation="gelu")
        self.subset_gate=tf.keras.layers.Dense(1)
        self.score_hidden=tf.keras.layers.Dense(width,activation="gelu")
        self.score_correct=tf.keras.layers.Dense(1)
        self.score_regress=tf.keras.layers.Dense(1)
        self.keep_hidden=tf.keras.layers.Dense(width,activation="gelu")
        self.keep_out=tf.keras.layers.Dense(1)

    def call(self,x,training=False):
        combo=tf.cast(x["subset_features"],tf.float32)
        valid=tf.cast(x["subset_mask"],tf.bool)
        base=tf.cast(x["baseline"],tf.float32)
        context=tf.cast(x["context"],tf.float32)
        keep_heads=tf.cast(x["keep_heads"],tf.float32)
        n=tf.shape(combo)[0]
        if combo.shape[-1]!=FEAT_DIM:raise ValueError("invalid feature width")
        if combo.shape[-2]!=COMBOS:raise ValueError("invalid subset axis")
        onehot_targets=tf.broadcast_to(tf.eye(5)[None,:,:],[n,5,5])
        base_exp=tf.broadcast_to(base[:,None,None,:],[n,5,COMBOS,7])
        tgt_exp=tf.broadcast_to(onehot_targets[:,:,None,:],[n,5,COMBOS,5])
        ctx_exp=tf.broadcast_to(context[:,None,None,:],
            [n,5,COMBOS,tf.shape(context)[-1]])
        xsub=tf.concat([combo,base_exp,tgt_exp,ctx_exp],axis=-1)
        state=self.subset_embed(xsub)
        score=tf.squeeze(self.subset_gate(state),axis=-1)
        neg=tf.constant(-1e9,tf.float32)
        safe=tf.where(valid,score,neg)
        # K identical to the frozen baseline is a no-op, not a candidate;
        # every remaining valid target has >=32 nonempty H0 subsets.
        selector=tf.nn.softmax(safe,axis=2)
        selector=tf.where(valid,selector,tf.zeros_like(selector))
        selector/=tf.maximum(tf.reduce_sum(selector,axis=2,keepdims=True),1e-9)
        pooled=tf.einsum("bks,bksd->bkd",selector,state)
        selected_feature=tf.einsum("bks,bksd->bkd",selector,combo)
        context_by_k=tf.broadcast_to(context[:,None,:],
                                  [n,5,tf.shape(context)[-1]])
        base_by_k=tf.broadcast_to(base[:,None,:],[n,5,7])
        candidate_feat=tf.concat([pooled,selected_feature,context_by_k,
                                   base_by_k,onehot_targets],axis=-1)
        hidden=self.score_hidden(candidate_feat)
        p_correct=tf.sigmoid(tf.squeeze(self.score_correct(hidden),-1))
        p_regress=tf.sigmoid(tf.squeeze(self.score_regress(hidden),-1))
        keep_feat=tf.concat([base,context,keep_heads,
                            tf.reduce_max(p_correct,axis=1,keepdims=True),
                            tf.reduce_mean(p_regress,axis=1,keepdims=True)],axis=-1)
        keep_good=tf.sigmoid(tf.squeeze(self.keep_out(self.keep_hidden(keep_feat)),-1))
        # Dedicated learned risk actively controls final action: no
        # independent decoder can circumvent this signal.
        # Mutual utility compares plausible correctness to risk of harm.
        candidate_logits=tf.math.log(tf.maximum(p_correct,EPS))-\
            tf.math.log(tf.maximum(p_regress,EPS))
        candidate_logits=tf.where(tf.reduce_any(valid,axis=-1),
                                  candidate_logits,neg)
        # KEEP confidence enters its own decision logit.
        keep_logit=tf.math.log(tf.maximum(keep_good,EPS))-\
            tf.math.log(tf.maximum(1.-keep_good,EPS))
        action_logits=tf.concat([candidate_logits,keep_logit[:,None]],axis=1)
        return dict(action_logits=action_logits,subset_attention=selector,
                    candidate_correct=p_correct,candidate_regress=p_regress,
                    keep_correct=keep_good,selected_subset_features=selected_feature)


def targets(y,base):
    y=np.asarray(y,int);b=np.asarray(base,int)
    if y.shape!=b.shape:raise ValueError("shape")
    # 0..4 for K2..6, 5 for KEEP. Unsupported true K0/K1 -> KEEP.
    action=np.where((y==b)|(y<2),5,y-2).astype("int32")
    cp=(y[:,None]==np.asarray(K_RANGE)[None,:]).astype("float32")
    rg=(y==b).astype("float32")
    correction=(y[:,None]==np.asarray(K_RANGE)[None,:]).astype("float32")
    regress=np.broadcast_to(rg[:,None],cp.shape).copy()
    keep=((y==b)|(y<2)).astype("float32")
    return action,correction,regress,keep


def objective(pred,target,correct_label,regress_label,keep_label,sample_weight):
    a=tf.cast(target,tf.int32)
    w=tf.cast(sample_weight,tf.float32)
    class_ce=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=a,logits=pred["action_logits"])
    classification=tf.reduce_sum(class_ce*w)/tf.reduce_sum(w)
    mask=tf.cast(pred["candidate_correct"]> -1.,tf.float32)
    # Scores for the baseline's own K are masked in final logits but the
    # corresponding risk predictions remain valid as auxiliary labels.
    cp=tf.clip_by_value(pred["candidate_correct"],EPS,1-EPS)
    rp=tf.clip_by_value(pred["candidate_regress"],EPS,1-EPS)
    kp=tf.clip_by_value(pred["keep_correct"],EPS,1-EPS)
    bce=lambda p,y:-(y*tf.math.log(p)+(1-y)*tf.math.log(1-p))
    return (classification+0.3*tf.reduce_mean(bce(cp,correct_label))
            +0.3*tf.reduce_mean(bce(rp,regress_label))
            +0.3*tf.reduce_mean(bce(kp,keep_label)))


def train_model(train,truth,base,held,seed=SEED,epochs=30):
    tf.keras.utils.set_random_seed(seed)
    model=TransitionCombinationArbiter(width=32)
    labels,cor,reg,keep=targets(truth,base)
    # Weights from outer TRAIN only. Balanced across observed TRUE K for
    # rare K3/K4 while avoiding manual class-specific lookup weights.
    hist=np.bincount(np.asarray(truth,dtype=int),minlength=7).astype(np.float32)
    scaled=np.sqrt(np.maximum(np.max(hist),1)/(hist+1.))
    scaled/=np.mean(scaled[truth])
    data=tf.data.Dataset.from_tensor_slices((
        train,labels,cor,reg,keep,scaled[truth].astype(np.float32)
    )).shuffle(len(labels),seed=seed,reshuffle_each_iteration=True).batch(192)
    opt=tf.keras.optimizers.Adam(learning_rate=.002)
    histlog=[]
    for epoch in range(epochs):
        loss_list=[]
        for feat,t,co,re,ke,sw in data:
            with tf.GradientTape() as tape:
                out=model(feat,training=True)
                loss=objective(out,t,co,re,ke,sw)
            grads=tape.gradient(loss,model.trainable_variables)
            if any(g is None for g in grads):raise ValueError("dead neural layer")
            opt.apply_gradients(zip(grads,model.trainable_variables))
            loss_list.append(float(loss))
        if epoch in (0,epochs//2,epochs-1):
            histlog.append(dict(epoch=epoch+1,loss=float(np.mean(loss_list))))
    out=model(held,training=False)
    classes=np.array([2,3,4,5,6,7],int)
    action=classes[np.argmax(out["action_logits"].numpy(),axis=1)]
    return action,out,histlog
