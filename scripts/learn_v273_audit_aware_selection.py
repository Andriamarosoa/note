"""Audit-aware per-K neural subset selection with *explicit head identity*.

Adds two independently testable channels to the established 64-subset
source-target neural router WITHOUT adding a head or changing the baseline:

  identity:  7 bits naming H0,H1,...,H5 and applicable C(source,target)
  context:   4 correction/regression/neutral/coverage OOF audit channels
             restricted to the current, TRAIN-ONLY unsupervised acoustic
             regime (not the old A/B cluster inferred from heldout labels).

Each contextual audit for a TRAIN row is computed from OTHER inner folds
within its outer-TRAIN partition. Outer-val priors are computed only from
the complete outer-TRAIN partition. A cluster assignment depends on
acoustic features ONLY and kmeans was fitted only on outer train.

No historical audit percentages (on exposed eval folds) are reused as
training features. 'Historical' documents serve as research hypotheses
unless re-derived on new train-only event-level data.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from scripts.learn_v273_transition_combo_risk import (
    COMBOS, COMBO_BINARY, CORRECTION, K_RANGE, SRC_RANGE
)

CONTEXT_FEATURES=4
IDENTITY_FEATURES=7
REGIMES=3


def identity_descriptor(original, available, base):
    """Explicit 7-bit composition for each legal candidate set.

    Bit 0=H0. Bits 1..5=H1..H5. Bit 6=only the structurally legal
    transition corrector. Inactive candidates have ZERO descriptors.
    """
    n=len(base)
    if original.shape[:3]!=(n,len(K_RANGE),COMBOS):
        raise ValueError("subset option shape")
    bitmask=np.broadcast_to(
        COMBO_BINARY[None,None,:,:],(n,len(K_RANGE),COMBOS,7)
    ).copy()
    bitmask*=available[:,:,:,None].astype(np.float32)
    for j,target in enumerate(K_RANGE):
        no_action=np.asarray(base)==target
        if np.any(no_action) and np.any(bitmask[no_action,j,:,:]):
            raise ValueError("same-K should not be a correction")
        for source in SRC_RANGE:
            if (source,target) not in CORRECTION:
                bitmask[base==source,j,:,6]=0.
    return bitmask.astype(np.float32)


def fit_acoustic_regimes(context_train,context_held,seed=27402):
    """Cluster *without labels* and without consulting outer-held features to fit."""
    x=np.asarray(context_train,dtype=np.float32)
    z=np.asarray(context_held,dtype=np.float32)
    if x.ndim!=2 or z.ndim!=2 or x.shape[1]!=z.shape[1]:
        raise ValueError("acoustic context shape")
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("non-finite audio")
    model=KMeans(n_clusters=REGIMES,n_init=10,random_state=seed)
    train_group=model.fit_predict(x)
    held_group=model.predict(z)
    return train_group.astype(int),held_group.astype(int),model


def _smoothed_subset_stats(proposal, labels, source, target, prior=None,
                           pseudo=12.):
    """4 features per subset. Region-specific correction/benefit vs risk.

    Prior is a train-only global subset audit computed on same allowed
    reference rows. The local rates shrink toward the corresponding
    prior when local support is weak. Labels ONLY from an allowed ref.
    """
    offer=np.asarray(proposal,dtype=np.float64)
    y=np.asarray(labels,dtype=int)
    if offer.ndim!=2 or offer.shape[1]!=COMBOS or len(y)!=len(offer):
        raise ValueError("proposal/label shape")
    used=np.sum(offer,axis=0)
    corrected=np.sum(offer*(y==target)[:,None],axis=0)
    regressed=np.sum(offer*(y==source)[:,None],axis=0)
    neutral=used-corrected-regressed
    if prior is None:
        # Weak, permutation-invariant uniform Dirichlet prior.
        default=(np.stack([corrected,regressed,neutral],axis=1)+1.)/(
            used[:,None]+3.)
    else:
        default=np.asarray(prior,dtype=np.float64)
        if default.shape!=(COMBOS,3):raise ValueError("prior shape")
    rates=(np.stack([corrected,regressed,neutral],axis=1)+
           pseudo*default)/(used[:,None]+pseudo)
    return np.column_stack([np.log1p(used),rates]).astype(np.float32),rates


def attach_acoustic_regime_oof_audits(options_train,mask_train,base_train,
                                      truth_train,fold_train,group_train,
                                      options_held,mask_held,base_held,
                                      group_held):
    """Train-row audit never sees its own fold label, heldout sees train only.

    The audit record is the weighted empirical probability of the
    combination DIRECTLY PROPOSING a correction, distinguishing:
    y==target -> correction; y==source -> avoidable regression; other
    true Ks -> neutral change. All conditioning is source/target +
    acoustic regime (cluster fitted without target).
    """
    yt=np.asarray(truth_train,int); bt=np.asarray(base_train,int)
    bh=np.asarray(base_held,int)
    ft=np.asarray(fold_train,int); gt=np.asarray(group_train,int)
    gh=np.asarray(group_held,int)
    n=len(yt);nh=len(bh)
    if options_train.shape[:3]!=(n,5,COMBOS) or (
            options_held.shape[:3]!=(nh,5,COMBOS)):
        raise ValueError("input option shape")
    if not np.isin(gt,range(REGIMES)).all() or (
            not np.isin(gh,range(REGIMES)).all()):
        raise ValueError("regime mismatch")
    if len(np.unique(ft))<2:raise ValueError("missing inner folds")
    output_train=np.zeros((n,5,COMBOS,CONTEXT_FEATURES),np.float32)
    output_held=np.zeros((nh,5,COMBOS,CONTEXT_FEATURES),np.float32)
    records=[]
    for source in SRC_RANGE:
        for j,target in enumerate(K_RANGE):
            if source==target:continue
            in_src=bt==source
            for heldout in (*np.unique(ft),"outer_validation"):
                ref=in_src if heldout=="outer_validation" else (
                    in_src&(ft!=heldout))
                if not ref.any():raise ValueError("no permitted reference rows")
                proposal=options_train[ref,j,:,3] > 0.5
                global_prior,global_rates=_smoothed_subset_stats(
                    proposal,yt[ref],source,target,prior=None)
                for regime in range(REGIMES):
                    subset=ref&(gt==regime)
                    local_proposal=options_train[subset,j,:,3]>0.5
                    local,rate=_smoothed_subset_stats(
                        local_proposal,yt[subset],source,target,
                        prior=global_rates)
                    if heldout=="outer_validation":
                        dest=np.flatnonzero((bh==source)&(gh==regime))
                        if len(dest):
                            output_held[dest,j,:,:]=local[None,:,:]
                        if len(dest):
                            for idx in range(COMBOS):
                                if not np.any(mask_held[dest,j,idx]):continue
                                records.append({
                                    "baseline_K":int(source),
                                    "candidate_K":int(target),
                                    "regime":int(regime),
                                    "subset_id":int(idx),
                                    "train_reference_rows":int(ref.sum()),
                                    "train_regime_rows":int(subset.sum()),
                                    "train_proposals_in_regime":int(local_proposal[:,idx].sum()),
                                    "smoothed_correct":float(local[idx,1]),
                                    "smoothed_regress":float(local[idx,2]),
                                    "smoothed_neutral":float(local[idx,3])
                                })
                    else:
                        dest=np.flatnonzero(
                            (bt==source)&(ft==heldout)&(gt==regime))
                        if len(dest):
                            output_train[dest,j,:,:]=local[None,:,:]
    output_train*=mask_train[:,:,:,None]
    output_held*=mask_held[:,:,:,None]
    if not np.isfinite(output_train).all() or (
            not np.isfinite(output_held).all()):
        raise ValueError("nonfinite contextual OOF")
    return output_train,output_held,records


def append_audit_aware_descriptors(train_features,held_features,
                                   train_mask,held_mask,base_train,base_held,
                                   truth_train,fold_train,
                                   train_ctx,held_ctx,mode):
    """Construct comparison experiment inputs with independent increments.

    mode baseline:   original 10 features
    mode identity:   +7 explicit ID bits
    mode regimes:    +4 contextual OOF audit rates
    mode both:       +7 bits and +4 regime features
    """
    if mode not in ("baseline","identity","regimes","both"):
        raise ValueError("unsupported experiment")
    a=np.asarray(train_features,np.float32)
    b=np.asarray(held_features,np.float32)
    pieces_a=[a];pieces_b=[b]
    if mode in ("identity","both"):
        pieces_a.append(identity_descriptor(a,train_mask,base_train))
        pieces_b.append(identity_descriptor(b,held_mask,base_held))
    regime_rows=[]
    if mode in ("regimes","both"):
        rtrain,rheld,kmeans=fit_acoustic_regimes(train_ctx,held_ctx)
        ca,cb,regime_rows=attach_acoustic_regime_oof_audits(
            a,train_mask,base_train,truth_train,fold_train,rtrain,
            b,held_mask,base_held,rheld)
        pieces_a.append(ca);pieces_b.append(cb)
    ta=np.concatenate(pieces_a,axis=-1)
    tb=np.concatenate(pieces_b,axis=-1)
    width={"baseline":10,"identity":17,"regimes":14,"both":21}[mode]
    if ta.shape[-1]!=width or tb.shape[-1]!=width:
        raise ValueError("augmented subset descriptor width")
    return ta,tb,regime_rows
