"""S689 isolate weak fundamental attacks from overtone copies.

Only the native source acoustic 126D schema and PAST native posteriors are
used. Different from S688's octave-only suppression: compare the independent
new overtone-series slope (2nd/3rd partial of candidate) against the native
lower fundamental's transient, K1/cardinality and the candidate's own
fundamental rise. Only an otherwise subthreshold Onset may be raised.

No ground truth at inference, no future audio, no change to original
pitch/K/sustain/offset neural heads. 9 disjoint TRAIN compositions freeze
one policy; S675 complete notes must not regress; DEV is audit only.
"""
from __future__ import annotations
import numpy as np
from causal_note.strict_live import HOP,SR,Selection,feature_track
from scripts.s613_prepare_strict_live import digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    raw_guitar_audio,native_posteriors,note_instances,native_rhythm_clock)
from scripts.s682_onset_four_lock_guard import native_metrics,score_release

# Native CausalPitchFeatures physical ABI:
# harmonics [0:24] 3 windows x 8, lag-2 physical change [54:60],
# lag-8 change [66:72]. K logits are frozen, unchanged.
POLICIES=("native","independent_floor_0.20_ratio_0.8",
          "independent_floor_0.20_ratio_1.2",
          "independent_floor_0.27_ratio_0.8",
          "independent_floor_0.27_ratio_1.2",
          "independent_floor_0.35_ratio_0.8",
          "independent_floor_0.35_ratio_1.2",
          "independent_floor_0.35_ratio_1.6")


def rescue(p,features,policy,native_onset,k=None):
    if policy=="native":return p
    tokens=policy.split("_")
    floor=float(tokens[2]);ratio=float(tokens[4])
    if features.shape[:2]!=p.shape[:2] or features.shape[2]!=126:
        raise ValueError("Unexpected NOTE physical feature ABI")
    if k is not None and (len(k)!=len(p) or k.shape[1]!=7):
        raise ValueError("K posterior ABI not frozen")
    # Harmonic series for a new physical string has its own increment:
    # 1st harmonic now > previous two/eight hops and partials 2/3 rise
    # independently. At source K1 a new simultaneous string must present
    # particularly strong proof, not simply be an overtone of a bass.
    f0=np.maximum(features[:,:,[0,8,16]],0).mean(axis=-1)
    f2=np.maximum(features[:,:,[1,9,17]],0).mean(axis=-1)
    f3=np.maximum(features[:,:,[2,10,18]],0).mean(axis=-1)
    d2=np.maximum(features[:,:,57:60],0).mean(axis=-1)
    d8=np.maximum(features[:,:,69:72],0).mean(axis=-1)
    # Difference-based evidence is bounded, scale free and past-only.
    d=(.6*d2+.4*d8)/(.20+f0)
    upper_independent=(.5*f2+.5*f3)/(.20+f0)
    lower=np.maximum(0,np.arange(49)-12)
    lower_sustain=p[:,lower,1]
    lower_attack=np.maximum(p[:,lower,0],.01)
    # Competing bass attack and sustain estimate harmonic-copy risk.
    harmonic_risk=(lower_sustain>.66)&(lower_attack>.40)
    independent=(d+.11*upper_independent > ratio*(.08+d[:,lower])*.5)
    independent&=(~harmonic_risk)|(d >= ratio*d[:,lower]+.05)
    onset_candidate=((p[:,:,0]>=floor)&(p[:,:,0]<native_onset)&
                     (p[:,:,1]>=.45)&(d>=.15)&independent)
    if k is not None:
        # A frozen K1 posterior >.65 requires a larger independent rise
        # before a second string is asserted.
        onset_candidate&=~((k[:,1,None]>.65)&(lower_sustain>.72)&(d<.28))
    out=p.copy()
    out[:,:,0]=np.where(onset_candidate,native_onset+.025,out[:,:,0])
    if not np.array_equal(out[:,:,1:],p[:,:,1:]):
        raise AssertionError("S689 mutated source sustain or release")
    return out


def record(spec,root,archive,heads,sel,policies,offmodel=None):
    wave,_=raw_guitar_audio(spec,archive)
    gt=real_notes(labels(spec,root)["notes"],len(wave)/SR)
    ends,feat=feature_track(wave,block_size=4096)
    p,k=native_posteriors(feat,heads)
    digest0=digest(p[:,:,1:].tobytes()+k.tobytes())
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
        if digest(proposed[:,:,1:].tobytes()+k.tobytes())!=digest0:
            raise AssertionError("S689 native K/release mutated")
    return dict(name=spec["name"],composition=spec["composition"],
                scores=scores,release=release)
