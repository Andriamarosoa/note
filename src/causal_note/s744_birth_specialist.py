"""S744 FIT-only supervised specialist for independent note births and active K.

This is a new specialist *inside* NOTE's multi-head architecture, NOT a
distilled Student, no shrinking/compression. FIT80 annotations are used for
training only. During validation, only past and current received acoustics,
S708 posterior and already emitted source state may enter the classifier.

A positive is a missing true onset at a causal decision fragment, not an
oracle desired-K change. Independent negative harmonic candidates prevent
blindly copying octave/fifth partials. The model's output is just H5's
candidate probability; a per-fragment 32-subset router and the S707
complete-note decoder remain responsible for events.
"""
from __future__ import annotations
import numpy as np
from causal_note.s708_cardinality_selection import GuidedSelector
from causal_note.strict_live import HOP
from causal_note.s743_fundamental_router import candidate_heads


def frozen_source_activity(post, k, ends, policy):
    """Replay the *unchanged* S708 source selector, returning its states.

    Does NOT use labels, annotation or another user's audio.
    """
    n=len(ends)
    if post.shape!=(n,49,3) or k.shape!=(n,7):
        raise ValueError("Expected S708 4-hop posterior grid")
    selector=GuidedSelector(policy)
    state=np.zeros((n,49),bool)
    for t,end in enumerate(ends):
        selector.step(post[t],k[t],int(end))
        state[t]=selector.active.copy()
    return state


def birth_features(post,native,context,sharp,k,source_active):
    """Causal per-fragment/per-pitch true acoustic features for H5 training."""
    p=np.asarray(post,np.float32)
    native=np.asarray(native,np.float32)
    ctx=np.asarray(context,np.float32)
    xx=np.asarray(sharp,np.float32)
    kk=np.asarray(k,np.float32)
    active=np.asarray(source_active,bool)
    n=len(p)
    if (p.shape!=(n,49,3) or native.shape!=(n,49,3) or
        ctx.shape!=(n,49,16) or xx.shape!=(n,49,288) or
        kk.shape!=(n,7) or active.shape!=(n,49) or
        not all(np.isfinite(z).all() for z in (p,native,ctx,xx,kk))):
        raise ValueError("S744 signed feature schema mismatch")
    prev=np.concatenate((p[:1],p[:-1]),axis=0)
    prevpart=np.concatenate((xx[:1,:,192:216],xx[:-1,:,192:216]),axis=0)
    fresh=np.maximum(xx[:,:,192:216]-prevpart,0).reshape(n,49,3,8)
    support=fresh[:,:,:3,:3].mean(axis=(2,3))
    active_count=active.sum(axis=1)
    tail=np.array([kk[i,min(6,int(active_count[i])+1):].sum()
                   for i in range(n)],np.float32)
    j=lambda v:np.broadcast_to(np.asarray(v,np.float32)[...,None],(n,49,1))
    grids=[
        ctx,p,native,
        np.broadcast_to(kk[:,None,:],(n,49,7)),
        xx[:,:,192:216], xx[:,:,126:138],
        j(active_count[:,None]/6.),
        active[...,None].astype(np.float32),
        np.broadcast_to(np.arange(49,dtype=np.float32)[None,:,None]/48,
                        (n,49,1)),
        (p[:,:,:2]-prev[:,:,:2]).astype(np.float32),
        support[...,None],
        np.broadcast_to(tail[:,None,None],(n,49,1)),
    ]
    # j(...) input is [n,1], correctly repeated over pitches.
    result=np.concatenate(grids,axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite missing-pitch model feature")
    return result


def select_fit_examples(features,onset_targets,source_active,
                        context,heads,seed,limit=1800):
    """Composition-safe bounded subsampling, retain positive / hard negatives.

    A note currently emitted by S708 is never a candidate for extra NoteOn.
    The target labels are consulted only during this FIT helper, not live.
    """
    feat=np.asarray(features,np.float32)
    truth=np.asarray(onset_targets,bool)
    active=np.asarray(source_active,bool)
    ctx=np.asarray(context,np.float32)
    hd=np.asarray(heads,np.float32)
    n=len(feat)
    if feat.ndim!=3 or feat.shape[:2]!=(n,49) or \
       truth.shape!=(n,49) or active.shape!=(n,49) or \
       ctx.shape!=(n,49,16) or hd.shape!=(n,49,6):
        raise ValueError("Wrong S744 FIT signatures")
    if limit<10:
        raise ValueError("Too few FIT rows")
    valid=~active
    positive=valid&truth
    hard=valid&(~truth)&(
        (hd.max(-1)>.12)|(ctx[:,:,4]>.015)|
        (ctx[:,:,8]>.01)|(ctx[:,:,1]>.35))
    rng=np.random.default_rng(seed)
    ix_positive=np.flatnonzero(positive.ravel())
    ix_hard=np.flatnonzero(hard.ravel())
    easy=valid&~truth&~hard
    ix_easy=np.flatnonzero(easy.ravel())
    # Protect rare true note births, yet ensure at least half the FIT
    # sample is negative for calibrated precision.
    pos_count=min(len(ix_positive),limit//2)
    hard_count=min(len(ix_hard),int((limit-pos_count)*.8))
    easy_count=min(len(ix_easy),limit-pos_count-hard_count)
    def take(x,count):
        return (np.sort(rng.choice(x,count,replace=False))
                if len(x)>count else x)
    ids=np.sort(np.concatenate((take(ix_positive,pos_count),
                                take(ix_hard,hard_count),
                                take(ix_easy,easy_count))))
    if not len(ids):
        raise ValueError("Empty FIT candidates")
    out=feat.reshape(n*49,-1)[ids]
    y=truth.ravel()[ids].astype(np.int32)
    if not np.isfinite(out).all():
        raise ValueError("Unusable acoustic FIT source")
    return out,y,dict(positives=int(y.sum()),rows=int(len(ids)),
                      true_positive_candidates=int(positive.sum()),
                      all_true_onsets=int(truth.sum()))


def boost_fit_weights(y,positive_weight=5.):
    labels=np.asarray(y,dtype=int)
    if labels.ndim!=1 or not np.isin(labels,[0,1]).all() or \
       positive_weight<1:
        raise ValueError("Supervision must be binary")
    return np.where(labels==1,float(positive_weight),1.).astype(np.float64)


def trained_head_proposals(model,base_heads,features,*,batch=65536):
    """Replace ONLY H5 as a trained, label-free causal birth proposal."""
    heads=np.asarray(base_heads,np.float32)
    x=np.asarray(features,np.float32)
    if heads.shape[:2]!=x.shape[:2] or heads.shape[-1]!=6:
        raise ValueError("H5 source mismatch")
    n=len(x)
    out=heads.copy()
    flat=x.reshape(-1,x.shape[-1])
    probs=np.empty(len(flat),np.float32)
    for i in range(0,len(flat),batch):
        y=model.predict_proba(flat[i:i+batch])
        if y.shape!=(min(batch,len(flat)-i),2):
            raise ValueError("Specialist probability ABI changed")
        probs[i:i+len(y)]=y[:,1]
    pred=probs.reshape(n,49)
    # H5 is a candidate proposal, never subtract frozen H0 onset.
    out[:,:,5]=np.maximum(heads[:,:,0],pred)
    if not np.array_equal(out[:,:,0],heads[:,:,0]):
        raise AssertionError("S744 attempted to change frozen S708 H0")
    return out
