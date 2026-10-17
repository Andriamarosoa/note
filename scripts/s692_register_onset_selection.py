"""S692 separate native NOTE weak fundamental rescue by guitar pitch register.

Pitch 40..47=low strings, 48..59=middle, 60..88=treble.
No labels, score, future PCM or annotated K in inference. Only original
past-only NOTE features and predicted K. Original native note heads and
S675 checkpoints stay untouched. Nine TRAIN composition groups freeze one
proposal before GuitarSet DEV, with K / Onset / Offset / full-note veto.
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

POLICIES=(
 "native","bass_0.35_1.2","mid_0.35_1.2","high_0.35_1.2",
 "bassmid_0.35_1.2","basshigh_0.35_1.2","midhigh_0.35_1.2",
 "bassmid_0.35_1.6","all_0.35_1.6"
)


def rescue(p,features,policy,native_onset,k):
    if policy=="native":return p
    parts=policy.split("_")
    if len(parts)!=3:raise ValueError("S692 only preregistered register policies")
    registers,floor,ratio=parts
    if policy not in POLICIES:raise ValueError("Not a preregistered S692 policy")
    candidate=independent(p,features,
        f"independent_floor_{float(floor):.2f}_ratio_{float(ratio):.1f}",
        native_onset,k)
    membership=np.zeros(49,bool)
    if "bass" in registers:membership[:8]=True
    if "mid" in registers:membership[8:20]=True
    if "high" in registers:membership[20:]=True
    if registers=="all":membership[:]=True
    out=p.copy()
    out[:,:,0]=np.where(membership[None,:],candidate[:,:,0],p[:,:,0])
    if not np.array_equal(out[:,:,1:],p[:,:,1:]):
        raise AssertionError("Native sustain/release mutated")
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
            row=dict(truth=gt,ends=ends,feats=feat,post=p,k=k,
                     events=emitted,onset=ons,pairs=pairs,
                     notes=note_instances(emitted,ends),
                     clock=native_rhythm_clock(emitted,ends))
            release[name]=score_release(row,offmodel)
        assert digest(proposed[:,:,1:].tobytes()+k.tobytes())==frozen
    return dict(name=spec["name"],composition=spec["composition"],
                scores=scores,release=release)
