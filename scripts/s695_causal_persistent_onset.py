"""S695 require consecutive past-only NOTE physical onset evidence.

Original NOTE selector remains unchanged. This experimental *new* onset
head accepts a weak-fundamental probability boost only after 2–4 source
hops of independent evidence. Never use future frames or backdate NoteOn.
At source 44.1kHz hop256 the evidence wait is 5.8–17.4ms before boosting.
Native original Onsets (p>=original threshold) are never suppressed.
Native K, sustain and release posterior are byte-identical.
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

POLICIES={
    "native":None,
    "p2_floor_0.27_ratio_0.8":(2,.27,.8),
    "p2_floor_0.35_ratio_1.2":(2,.35,1.2),
    "p3_floor_0.27_ratio_0.8":(3,.27,.8),
    "p3_floor_0.35_ratio_1.2":(3,.35,1.2),
    "p4_floor_0.35_ratio_1.2":(4,.35,1.2),
    "p2_floor_0.35_ratio_1.6":(2,.35,1.6),
    "p3_floor_0.35_ratio_1.6":(3,.35,1.6)
}


def causal_streak(evidence,minimum_hops):
    """At time t use evidence[0:t+1] only, never future onset.”
    """
    if minimum_hops<1:raise ValueError("No current-hop streak")
    yes=np.asarray(evidence,bool)
    out=np.zeros_like(yes)
    count=np.zeros(yes.shape[1],np.int32)
    for frame in range(yes.shape[0]):
        count=np.where(yes[frame],count+1,0)
        out[frame]=count>=minimum_hops
    return out


def rescue(p,features,policy,native_onset,k):
    if policy=="native":return p
    if policy not in POLICIES:raise ValueError("Policy not precommitted")
    persistence,floor,ratio=POLICIES[policy]
    trial=independent(p,features,
        f"independent_floor_{floor:.2f}_ratio_{ratio:.1f}",
        native_onset,k)
    marginal=trial[:,:,0]>p[:,:,0]
    accepted=causal_streak(marginal,persistence)
    out=p.copy()
    out[:,:,0]=np.where(accepted,trial[:,:,0],p[:,:,0])
    if not np.array_equal(out[:,:,1:],p[:,:,1:]):
        raise AssertionError("Changed original sustain release")
    return out


def record(spec,root,archive,heads,sel,policies,offmodel=None):
    audio,_=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,feat=feature_track(audio,block_size=4096)
    p,k=native_posteriors(feat,heads)
    frozen=digest(p[:,:,1:].tobytes()+k.tobytes())
    scores={};release={}
    for name in policies:
        candidate=rescue(p,feat,name,sel.onset,k)
        native_events=decode(ends,candidate,k,sel,len(audio))
        scores[name]=native_metrics(native_events,truth,ends)
        if offmodel is not None:
            onset,pairs=event_score(native_events,truth,tolerance=.080)
            item=dict(truth=truth,ends=ends,feats=feat,post=p,k=k,
                      events=native_events,onset=onset,pairs=pairs,
                      notes=note_instances(native_events,ends),
                      clock=native_rhythm_clock(native_events,ends))
            release[name]=score_release(item,offmodel)
        assert digest(candidate[:,:,1:].tobytes()+k.tobytes())==frozen
    return dict(name=spec["name"],composition=spec["composition"],
                scores=scores,release=release)
