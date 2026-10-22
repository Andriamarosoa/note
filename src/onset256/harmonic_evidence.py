"""Onset-owned, strictly causal harmonic context for PCM256 attack events.

All observations are computed from the same current S613 [49,126] frame,
its current S613 posteriors and the current 39D S701 readout. No GT, MIDI
labels, other axis state, future PCM or 1024-trained weights are used.
Research evidence only: not a trained anti-FP model nor a runtime veto.
"""
import numpy as np
from .acoustics import HOP, SR

PITCH_MIN = 40
PITCH_MAX = 88
LOWER_HARMONIC_SEMITONES = (12, 19, 24, 28)
HARMONIC_FIELDS = (
    "own_fundamental", "strongest_lower_fundamental",
    "lower_fundamental_ratio", "strongest_lower_onset",
    "strongest_lower_sustain", "neighbor_fundamental",
    "neighbor_fundamental_ratio", "current_fundamental_rise",
    "harmonic2_ratio", "harmonic3_ratio", "k_high_probability",
    "lower_harmonic_margin",
)

def current_pitch_strength(features):
    f = np.asarray(features, np.float32)
    if f.shape != (49, 126) or not np.isfinite(f).all():
        raise ValueError("Onset harmonic evidence requires current [49,126] PCM256")
    return np.maximum(f[:, [0, 8, 16]], 0.).mean(axis=1)

def onset_event_evidence(features, posterior, count, specialist_score,
                         pitch, sample):
    if HOP != 256 or type(sample) is not int or sample <= 0 or sample % HOP:
        raise ValueError("Onset event must be located at exact PCM256 boundary")
    if type(pitch) is not int or not PITCH_MIN <= pitch <= PITCH_MAX:
        raise ValueError("Onset pitch outside E2–E6 supported range")
    p = np.asarray(posterior, np.float32)
    k = np.asarray(count, np.float32)
    s = np.asarray(specialist_score, np.float32)
    if (p.shape != (49, 3) or k.shape != (7,) or s.shape != (49,)
        or not np.isfinite(p).all() or not np.isfinite(k).all()
        or not np.isfinite(s).all()):
        raise ValueError("Invalid current trained Onset prediction")
    strength = current_pitch_strength(features)
    ix = pitch - PITCH_MIN
    lower = [ix-n for n in LOWER_HARMONIC_SEMITONES if ix >= n]
    nearest = [j for j in (ix-1, ix+1) if 0 <= j < 49]
    own = float(strength[ix])
    lower_ix = (max(lower,key=lambda j:float(strength[j])) if lower else None)
    strong_lower = float(strength[lower_ix]) if lower_ix is not None else 0.
    near = max((float(strength[j]) for j in nearest),default=0.)
    lower_onset = max((float(p[j,0]) for j in lower),default=0.)
    lower_sustain = max((float(p[j,1]) for j in lower),default=0.)
    # Offset distances are harmonic hypotheses, not supervised evidence.
    f = np.asarray(features,np.float32)
    base = max(own,.2)
    harmonic2 = max(float(f[ix,[1,9,17]].clip(0).mean()),0.)
    harmonic3 = max(float(f[ix,[2,10,18]].clip(0).mean()),0.)
    x = (
        own,strong_lower,strong_lower/base,
        lower_onset,lower_sustain,near,near/base,
        float(np.maximum(f[ix,54:57],0).mean()-
              np.maximum(f[ix,66:69],0).mean()),
        harmonic2/(.2+own),harmonic3/(.2+own),
        float(k[2:].max()),own-strong_lower,
    )
    if len(x)!=len(HARMONIC_FIELDS) or not np.isfinite(x).all():
        raise AssertionError("Onset harmonic evidence schema invalid")
    return dict(
        pitch=int(pitch),sample=int(sample),
        on=float(p[ix,0]),sustain=float(p[ix,1]),release=float(p[ix,2]),
        specialist_score=float(s[ix]),count_posterior=k.astype(float).tolist(),
        fundamentals_parent_pitch=(int(lower_ix+PITCH_MIN) if lower_ix is not None else None),
        harmonic_parent_stronger=bool(lower_ix is not None and strong_lower > own),
        context={key:float(value) for key,value in zip(HARMONIC_FIELDS,x)},
    )
