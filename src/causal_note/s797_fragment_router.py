"""S797 — causal fragment-wise head routing on S795 frozen experts.

RESEARCH ONLY. Each frame/pitch selects a different combination of existing
independently trained heads. No future evidence, persistence, retiming,
source MIDI modifications or production LIVE integration.
"""
from __future__ import annotations
import numpy as np
from causal_note.s795_activek_experts import HEADS,FEATURES,PITCHES,regions

MODES=("identity","consensus","harmonic_veto","poly_gap","adaptive")
CUT=.93

def route_probabilities(X, predictions, mode="adaptive", cut=CUT):
    """Return routed head maps, fragment masks and a transparent decision audit."""
    if mode not in MODES:
        raise ValueError("undeclared S797 experimental ablation")
    X=np.asarray(X)
    if X.ndim!=3 or X.shape[1:]!=(PITCHES,FEATURES) or not np.isfinite(X).all():
        raise ValueError("S797 signed causal feature dimensions/values changed")
    if not np.isfinite(cut) or cut!=CUT:
        raise ValueError("S797 only predeclared threshold supported")
    if set(predictions)!=set(HEADS):
        raise ValueError("S797 missing/extra independently trained head")
    P={}
    for name in HEADS:
        p=np.asarray(predictions[name])
        if p.shape!=X.shape[:2] or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
            raise ValueError("S797 invalid head probability map "+name)
        P[name]=p
    source=X[:,:,12]>.5
    eligibility=regions(X,source)
    high={name:eligibility[name]&(P[name]>=cut) for name in HEADS}
    votes=np.zeros(X.shape[:2],np.int16)
    for name in HEADS:
        votes+=high[name].astype(np.int16)
    rise=X[:,:,5]
    parent=X[:,:,10]
    source_count=np.rint(X[:,:,15]*6).astype(np.int16)
    predicted_k=np.rint(X[:,:,13]*6).astype(np.int16)
    native_poly=source_count>=1
    k_gap=predicted_k>source_count
    fresh=(rise>=.05)|(X[:,:,6]>=.10)
    consensus=(votes>=2)&fresh&~source
    harmonic_safe=(parent<=.45)|(rise>=.20)
    low=np.broadcast_to((np.arange(PITCHES)<8)[None,:],source.shape)
    strong=rise>=.20
    if mode=="identity":
        allowed=np.zeros(source.shape,bool)
    elif mode=="consensus":
        allowed=consensus
    elif mode=="harmonic_veto":
        allowed=consensus&harmonic_safe
    elif mode=="poly_gap":
        allowed=consensus&harmonic_safe&native_poly&k_gap
    else:
        bass=low&high["bass"]&high["general"]
        poly=native_poly&k_gap&high["polyphonic"]&high["general"]
        attack=strong&high["transient"]&high["general"]
        allowed=consensus&harmonic_safe&(bass|poly|attack)
    mask=np.zeros(source.shape,np.uint16)
    routed={}
    for i,name in enumerate(HEADS):
        chosen=allowed&high[name]
        mask|=(chosen.astype(np.uint16)<<i)
        routed[name]=np.where(chosen,P[name],0).astype(np.float32)
    hist={str(int(v)):int(n) for v,n in zip(*np.unique(mask,return_counts=True))}
    return routed,mask,dict(mode=mode,heads_available=len(HEADS),
        considered=int(consensus.sum()),allowed=int(allowed.sum()),
        blocked_harmonic=int((consensus&~harmonic_safe).sum()),
        fragment_masks=hist,extra_lookahead_samples=0,
        extra_hold_samples=0,extra_confirmation_samples=0)
