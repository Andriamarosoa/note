"""S688 original-K-preserving octave competition veto on ONLY new weak notes.

Start with S686 novelty rescue, then suppress an ADDED pitch an octave
above a sustained native lower pitch unless independently stronger spectral
change supports the high note. True upper-octave notes can still pass.
Only current and previous native NOTE frames are used. Existing native
Onsets and original K, pitch, release heads are untouched.
"""
from __future__ import annotations
import numpy as np
from causal_note.strict_live import HOP,SR,Selection,feature_track
from scripts.s613_prepare_strict_live import digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    raw_guitar_audio,native_posteriors,note_instances,native_rhythm_clock)
from scripts.s682_onset_four_lock_guard import native_metrics,score_release
from scripts.s686_causal_fundamental_novelty import rescue as original_rescue

POLICIES=("native","rescue_base","guard_0.55_ratio_0.75",
          "guard_0.55_ratio_1.00","guard_0.70_ratio_0.75",
          "guard_0.70_ratio_1.00","guard_0.70_ratio_1.25")


def rescue(p,features,policy,native_onset):
    if policy=="native":return p
    boosted=original_rescue(p,features,"rescue_0.30_rise_0.12",native_onset)
    if policy=="rescue_base":return boosted
    _,threshold,_,ratio=policy.split("_")
    threshold=float(threshold);ratio=float(ratio)
    fundamental=np.maximum(features[:,:,[0,8,16]],0).mean(axis=2)
    near=np.maximum(features[:,:,54:57],0).mean(axis=2)
    history=np.maximum(features[:,:,66:69],0).mean(axis=2)
    novelty=(.65*near+.35*history)/(.2+fundamental)
    lower=np.maximum(0,np.arange(49)-12)
    preceding=np.arange(49)[None,:]>=12
    native_bass_support=p[:,lower,1]>=threshold
    # A true independently attacked upper note must supply MORE novelty
    # than the lower note's harmonic mirror by a chosen factor.
    likely_harmonic=(novelty<=ratio*novelty[:,lower]+.018)
    only_new=boosted[:,:,0]>p[:,:,0]
    veto=only_new&preceding&native_bass_support&likely_harmonic
    out=boosted.copy()
    out[:,:,0][veto]=p[:,:,0][veto]
    assert np.array_equal(out[:,:,1:],p[:,:,1:])
    return out


def record(spec,root,archive,heads,sel,policies,offmodel=None):
    wave,_=raw_guitar_audio(spec,archive)
    gt=real_notes(labels(spec,root)["notes"],len(wave)/SR)
    ends,feat=feature_track(wave,block_size=4096)
    p,k=native_posteriors(feat,heads)
    original_sha=digest(p[:,:,1:].tobytes()+k.tobytes())
    scores={};release={}
    for name in policies:
        proposed=rescue(p,feat,name,sel.onset)
        emitted=decode(ends,proposed,k,sel,len(wave))
        scores[name]=native_metrics(emitted,gt,ends)
        if offmodel is not None:
            ons,pairs=event_score(emitted,gt,tolerance=.080)
            row=dict(truth=gt,ends=ends,feats=feat,post=p,k=k,events=emitted,
                     onset=ons,pairs=pairs,notes=note_instances(emitted,ends),
                     clock=native_rhythm_clock(emitted,ends))
            release[name]=score_release(row,offmodel)
        assert digest(proposed[:,:,1:].tobytes()+k.tobytes())==original_sha
    return dict(name=spec["name"],composition=spec["composition"],
                scores=scores,release=release)

