"""S691 vote from S686 novelty and S689 harmonic-series independence heads.

Neither source head changes K, pitch, sustain or release posterior. Adding
a weak onset requires consensus of two separately motivated causal criteria.
A fallback lets low-K-confident scenes use a stricter physical derivative.
TRAIN composition guards freeze one policy before external DEV check.
"""
from __future__ import annotations
import numpy as np
from causal_note.strict_live import HOP,SR,Selection,feature_track
from scripts.s613_prepare_strict_live import digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    raw_guitar_audio,native_posteriors,note_instances,native_rhythm_clock)
from scripts.s682_onset_four_lock_guard import native_metrics,score_release
from scripts.s686_causal_fundamental_novelty import rescue as slope_rescue
from scripts.s689_independent_fundamental_rescue import rescue as harmonic_rescue

POLICIES=("native","vote_0.20_0.8","vote_0.27_0.8",
          "vote_0.20_1.2","vote_0.35_1.2",
          "vote_k2_0.27_0.8","vote_k2_0.35_1.2")


def rescue(p,features,policy,native_onset,k):
    if policy=="native":return p
    _,*items=policy.split("_")
    require_k2=(items[0]=="k2")
    if require_k2:items=items[1:]
    if len(items)!=2:raise ValueError("Bad S691 policy")
    floor=float(items[0]);ratio=float(items[1])
    slope=slope_rescue(p,features,"rescue_0.30_rise_0.12",native_onset)
    harmonic=harmonic_rescue(
        p,features,f"independent_floor_{floor:.2f}_ratio_{ratio:.1f}",
        native_onset,k)
    onset_vote=(slope[:,:,0]>p[:,:,0])&(harmonic[:,:,0]>p[:,:,0])
    if require_k2:
        # Never read annotated K; use original causal 7-way K posterior.
        # For K1-dominant segments only stronger consensus is accepted.
        extra=np.maximum(features[:,:,54:57],0).mean(axis=-1)
        onset_vote&=(k[:,2:].sum(axis=1)[:,None]>=.30)|(extra>=.38)
    out=p.copy()
    out[:,:,0]=np.where(onset_vote,native_onset+.025,p[:,:,0])
    if not np.array_equal(out[:,:,1:],p[:,:,1:]):
        raise AssertionError("S691 release/sustain altered")
    return out


def record(spec,root,archive,heads,sel,policies,offmodel=None):
    wave,_=raw_guitar_audio(spec,archive)
    gt=real_notes(labels(spec,root)["notes"],len(wave)/SR)
    ends,feat=feature_track(wave,block_size=4096)
    p,k=native_posteriors(feat,heads)
    frozen=digest(p[:,:,1:].tobytes()+k.tobytes())
    scores={};release={}
    for name in policies:
        proposed=rescue(p,feat,name,sel.onset,k)
        emitted=decode(ends,proposed,k,sel,len(wave))
        scores[name]=native_metrics(emitted,gt,ends)
        if offmodel is not None:
            ons,pairs=event_score(emitted,gt,tolerance=.080)
            row=dict(truth=gt,ends=ends,feats=feat,post=p,k=k,events=emitted,
                     onset=ons,pairs=pairs,notes=note_instances(emitted,ends),
                     clock=native_rhythm_clock(emitted,ends))
            release[name]=score_release(row,offmodel)
        if digest(proposed[:,:,1:].tobytes()+k.tobytes())!=frozen:
            raise AssertionError("Native note heads modified")
    return dict(name=spec["name"],composition=spec["composition"],
                scores=scores,release=release)
