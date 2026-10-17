"""S693 causal K-posterior-conditioned independent Onset recovery.

S689 gains true notes on chords but generates extra harmonic FPs. Condition
only the NEW onset candidates on frozen native K>=2 posterior; native
head K0-K6 is neither edited nor trained on GT. All heads are causal.
Original NOTE source and S675 release model unchanged. TRAIN composition
groups freeze a single K gate before retrospective DEV check.
"""
from __future__ import annotations
import numpy as np
from causal_note.strict_live import HOP,SR,Selection,feature_track
from scripts.s613_prepare_strict_live import digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    raw_guitar_audio,native_posteriors,note_instances,native_rhythm_clock)
from scripts.s682_onset_four_lock_guard import native_metrics,score_release
from scripts.s689_independent_fundamental_rescue import rescue as independent

POLICIES=("native","poly_0.20","poly_0.30","poly_0.40","poly_0.50",
          "poly_0.65","poly_0.20_strict","poly_0.40_strict")


def rescue(p,features,policy,native_onset,k):
    if policy=="native":return p
    if policy not in POLICIES:raise ValueError("S693 non preregistered policy")
    tokens=policy.split("_")
    poly_gate=float(tokens[1]);strict=(len(tokens)>2)
    candidate=independent(p,features,
       "independent_floor_0.35_ratio_1.6" if strict
       else "independent_floor_0.35_ratio_1.2",native_onset,k)
    predicted_poly=k[:,2:].sum(axis=1)
    onset_vote=(predicted_poly[:,None]>=poly_gate)&(candidate[:,:,0]>p[:,:,0])
    out=p.copy()
    out[:,:,0]=np.where(onset_vote,candidate[:,:,0],p[:,:,0])
    if not np.array_equal(out[:,:,1:],p[:,:,1:]):
        raise AssertionError("Original Sustain/Release mutated")
    return out


def record(spec,root,archive,heads,sel,policies,offmodel=None):
    audio,_=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,features=feature_track(audio,block_size=4096)
    p,k=native_posteriors(features,heads)
    ref=digest(p[:,:,1:].tobytes()+k.tobytes())
    scores={};release={}
    for name in policies:
        proposed=rescue(p,features,name,sel.onset,k)
        events=decode(ends,proposed,k,sel,len(audio))
        scores[name]=native_metrics(events,truth,ends)
        if offmodel is not None:
            onset,pairs=event_score(events,truth,tolerance=.08)
            item=dict(truth=truth,ends=ends,feats=features,post=p,k=k,
                      events=events,onset=onset,pairs=pairs,
                      notes=note_instances(events,ends),
                      clock=native_rhythm_clock(events,ends))
            release[name]=score_release(item,offmodel)
        assert digest(proposed[:,:,1:].tobytes()+k.tobytes())==ref
    return dict(name=spec["name"],composition=spec["composition"],
                scores=scores,release=release)
