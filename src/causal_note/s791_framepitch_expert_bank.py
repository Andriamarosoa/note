"""S791 heterogeneous Frame Pitch bank: physical causes determine expert count.

One protected reference source plus FOUR INDEPENDENTLY TRAINED candidate
specialties: general, low guitar register, harmonic confusion and temporal
transitions. These are candidate roles, not a requirement to keep four:
composition-held-out FIT contribution decides the number actually retained.

No current-time labels/GT enter expert masks, offers, or router inference.
Every frame is one complete causal 1024-sample decision (shadow, not MIDI).
"""
from __future__ import annotations
from itertools import product
import numpy as np

EXPERT_NAMES=("general","bass","harmonic","transition")
ALL_MASKS=tuple(range(1<<len(EXPERT_NAMES)))
N_EXPERTS=len(EXPERT_NAMES)
FEATURE_DIM=40
PITCHES=49

def eligibility(X,post):
    """Region definitions are fixed BEFORE FIT training, without labels."""
    t=len(post)
    sustain=post[:,:,1]
    before=np.concatenate((sustain[:1],sustain[:-1]),axis=0)
    parents=np.stack([X[:,:,FEATURE_DIM-10+i] for i in range(5)],axis=-1).max(-1)
    bass=np.broadcast_to((np.arange(PITCHES)<8)[None,:],(t,PITCHES))
    harmonic=parents>.30
    transition=((np.abs(sustain-before)>.09) |
                ((sustain>.24)&(sustain<.74)&(post[:,:,2]>.1)))
    return dict(general=np.ones_like(sustain,bool),bass=bass,
                harmonic=harmonic,transition=transition)

def sample_expert(X,y,post,expert,seed,max_pos=850,max_neg=1000):
    if X.ndim!=3 or X.shape[1:]!=(PITCHES,FEATURE_DIM):
        raise ValueError("S791 expected [time,49,40]")
    eligible=eligibility(X,post)[expert]
    positives=np.flatnonzero((eligible & y).ravel())
    negatives=np.flatnonzero((eligible & ~y).ravel())
    rng=np.random.default_rng(seed)
    positives=rng.choice(positives,min(len(positives),max_pos),replace=False)
    negatives=rng.choice(negatives,min(len(negatives),max_neg),replace=False)
    ix=np.sort(np.concatenate((positives,negatives)))
    return X.reshape(-1,FEATURE_DIM)[ix].astype(np.float32),\
           np.asarray(y,bool).ravel()[ix].astype(np.int8)

def candidate_offers(X,post,source,probabilities):
    """Return all 16 combinations, with source-only mask zero.

    Each model has its OWN fitted parameters and region; disabled/unavailable
    regional experts cannot alter any pitch. A selected subset is adapted at
    every 1024-sample FRAGMENT by the independent learned router.
    """
    n=len(post)
    source=np.asarray(source,bool)
    if source.shape!=(n,PITCHES):
        raise ValueError("Frame Pitch source occupancy shape mismatch")
    domain=eligibility(X,post)
    if set(probabilities)!=set(EXPERT_NAMES):
        raise ValueError("four physically motivated predictions required")
    base_prior=np.where(source,.25,-.25).astype(np.float32)
    supports=np.empty((N_EXPERTS,n,PITCHES),np.float32)
    for i,name in enumerate(EXPERT_NAMES):
        p=np.asarray(probabilities[name],np.float32)
        if p.shape!=(n,PITCHES) or not np.isfinite(p).all():
            raise ValueError(f"malformed {name} expert")
        supports[i]=np.where(domain[name],(p-.5)*2.,0.)
    offers=np.empty((n,len(ALL_MASKS),PITCHES),bool)
    offers[:,0,:]=source
    for mask in ALL_MASKS[1:]:
        combined=base_prior.copy()
        for i in range(N_EXPERTS):
            if mask&(1<<i):
                combined+=supports[i]
        offers[:,mask,:]=combined>=0.
    return offers,domain

def frame_descriptors(X,post,base,probabilities,domains):
    """Current-and-past-only metadata for the independent fragment selector."""
    sustain=post[:,:,1]
    prior=np.concatenate((sustain[:1],sustain[:-1]),axis=0)
    registered=[np.asarray(probabilities[k],np.float32)
                for k in EXPERT_NAMES]
    out=np.stack([
        base.mean(1),sustain.mean(1),sustain[:,:8].mean(1),
        sustain[:,8:20].mean(1),sustain[:,20:].mean(1),
        post[:,:,0].max(1),post[:,:,0].mean(1),
        np.abs(sustain-prior).mean(1),
        np.mean(X[:,:,6:30],axis=(1,2)),
        domains["harmonic"].mean(1),domains["transition"].mean(1),
        *[z.mean(1) for z in registered],
        *[(z>.65).mean(1) for z in registered]
    ],axis=1).astype(np.float32)
    if out.shape!=(len(post),19) or not np.isfinite(out).all():
        raise AssertionError(f"Frame Pitch selector feature ABI {out.shape}")
    return out

def oracle_mask_labels(offers,truth,admissible=ALL_MASKS):
    """TRAINING ONLY: labels may derive the FIT supervised router objective.

    Ground truth is never supplied to frame_descriptors, candidate_offers
    or trained router .predict(X).
    """
    n,m,p=offers.shape
    if truth.shape!=(n,p):
        raise ValueError("GT shape mismatched to FIT offers")
    tp=np.sum(offers & truth[:,None,:],axis=2)
    fp=np.sum(offers & ~truth[:,None,:],axis=2)
    fn=np.sum(~offers & truth[:,None,:],axis=2)
    per=2*tp-3*fp-2*fn
    penalties=np.array([int(x).bit_count()*.08 for x in range(m)])
    per=per-penalties[None,:]
    allowed=np.zeros(m,bool);allowed[list(admissible)]=True
    per[:,~allowed]=-1e12
    return np.argmax(per,axis=1).astype(np.int8)

def selected_offer(offers,chosen):
    idx=np.asarray(chosen,dtype=int)
    if idx.shape!=(len(offers),) or np.any(idx<0) or np.any(idx>=len(ALL_MASKS)):
        raise ValueError("one valid mask per fragment required")
    return offers[np.arange(len(offers)),idx,:]

def contribution_table(records):
    """Per-cause contributions based on FIT held-out predictions only."""
    names=["source"]+list(EXPERT_NAMES)
    out={}
    for i,name in enumerate(names):
        mask=0 if i==0 else 1<<(i-1)
        tp=fp=fn=0
        by_comp={}
        for comp in sorted(set(r["composition"] for r in records)):
            frames=[r for r in records if r["composition"]==comp]
            t=sum(int(np.sum(r["offers"][:,mask,:]&r["truth"])) for r in frames)
            f=sum(int(np.sum(r["offers"][:,mask,:]&~r["truth"])) for r in frames)
            n=sum(int(np.sum(~r["offers"][:,mask,:]&r["truth"])) for r in frames)
            tp+=t;fp+=f;fn+=n
            by_comp[comp]=2*t/max(2*t+f+n,1)
        out[name]=dict(TP=tp,FP=fp,FN=fn,
            f1=2*tp/max(2*tp+fp+fn,1),by_composition=by_comp)
    return out

def retained_from_fit(table, min_gain=.001, max_comp_drop=.01):
    base=table["source"]
    useful=[]
    for name in EXPERT_NAMES:
        row=table[name]
        if row["f1"]>=base["f1"]+min_gain and all(
            row["by_composition"][comp]>=b-max_comp_drop
            for comp,b in base["by_composition"].items()):
            useful.append(name)
    masks=tuple(mask for mask in ALL_MASKS
                if all(not(mask&(1<<i)) or name in useful
                       for i,name in enumerate(EXPERT_NAMES)))
    return useful,masks
