"""S725: causal *source ownership* memory for polyphonic onsets.

A single E2 note contains an E3-frequency harmonic. A new E3 can change the
shared frequency's attack and harmonic trajectory independently of E2's
predicted decay. S725 encodes residual novelty after predicting the spectral
evolution of already observed notes, gated softly by their PREVIOUS native
Sustain posterior (never true labels or future audio).

Appends 32D to the exact signed S708 288D feature ABI. The original 288D
features, Exact-K neural head and frozen S707 NoteOff are preserved.
This is a research ablation: no new detected note is emitted by this class.
"""
from __future__ import annotations
import numpy as np

BASE=288
EXTRA=32
TOTAL=BASE+EXTRA
AMP_START=192
SUSTAIN_COLUMN=126+1


class CausalSourceMemory:
    """One received 4-hop decision at a time, no lookahead or oracle K.

    8D positive innovation, 8D amplitude-normalized innovation,
    8D negative innovation, 3D parent-octave even-vs-odd innovation,
    3D third-harmonic-parent multiples novelty, 2D previous ownership
    strength. Use prior observed native Sustain only for soft gating.
    """
    def __init__(self):
        self.last=None
        self.trend=np.zeros((49,3,8),np.float32)
        self.owner=np.zeros(49,np.float32)
        pitches=np.arange(49)
        self.parent12=np.maximum(0,pitches-12)
        self.parent19=np.maximum(0,pitches-19)
        self.valid12=(pitches>=12).astype(np.float32)
        self.valid19=(pitches>=19).astype(np.float32)

    def step(self,frame):
        x=np.asarray(frame,np.float32)
        if x.shape!=(49,BASE) or not np.isfinite(x).all():
            raise ValueError("S725 requires finite S708 [49,288] current frame")
        amp=x[:,AMP_START:AMP_START+24].reshape(49,3,8)
        if self.last is None:
            residual=np.zeros_like(amp)
            change=np.zeros_like(amp)
        else:
            # Physical note decay can only predict no gain or a gradual
            # decline. Do not predict fresh growth using a prior onset.
            predicted=self.last+np.clip(self.trend,-.35,0.)
            residual=amp-predicted
            change=amp-self.last
        positive=np.maximum(residual,0.)
        negative=np.maximum(-residual,0.)
        # The model still sees absolute values (native S708 288D); these
        # new statistics add independent dynamics, not tail AGC.
        pos=positive.mean(axis=1)
        relative=(positive/(.35+np.maximum(amp,0.))).mean(axis=1)
        neg=negative.mean(axis=1)
        # Relative parity of the parent's 'unexplained' harmonic changes
        # is a cue for adding an octave note, not a hard source decision.
        parent12=positive[self.parent12]
        even=parent12[:,:,[1,3,5,7]].mean(-1)
        odd=parent12[:,:,[0,2,4,6]].mean(-1)
        octave=(even-odd)*self.valid12[:,None]*self.owner[self.parent12,None]
        parent19=positive[self.parent19]
        threes=parent19[:,:,[2,5]].mean(-1)
        other=parent19[:,:,[0,1,3,4,6,7]].mean(-1)
        third=(threes-other)*self.valid19[:,None]*self.owner[self.parent19,None]
        ownership=np.stack((
            self.owner[self.parent12]*self.valid12,
            self.owner[self.parent19]*self.valid19,
        ),axis=-1)
        out=np.concatenate((pos,relative,neg,octave,third,ownership),axis=-1)
        if out.shape!=(49,EXTRA) or not np.isfinite(out).all():
            raise AssertionError("S725 independent-source 32D ABI mismatch")
        if self.last is not None:
            # Smoothed past-only trend; adaptation neither rewrites a past
            # frame nor synthesizes a note from the previous posterior.
            self.trend=.80*self.trend+.20*np.clip(change,-.35,.35)
        self.last=amp.copy()
        self.owner=.80*self.owner+.20*np.clip(x[:,SUSTAIN_COLUMN],0,1)
        return out.astype(np.float32)


def source_memory_track(base):
    x=np.asarray(base,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,BASE) or not np.isfinite(x).all():
        raise ValueError("S725 expects original finite [T,49,288] input")
    mem=CausalSourceMemory()
    return np.stack([mem.step(frame) for frame in x],axis=0) if len(x) else np.empty((0,49,EXTRA),np.float32)


def augmented_track(base):
    x=np.asarray(base,np.float32)
    return np.concatenate((x,source_memory_track(x)),axis=-1).astype(np.float32)


def disabled_memory_track(base,extra_mean):
    """Equal-architecture ablation. New channels normalize to constant zero."""
    x=np.asarray(base,np.float32)
    mean=np.asarray(extra_mean,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,BASE) or mean.shape!=(EXTRA,):
        raise ValueError("S725 zero-memory contract mismatch")
    return np.concatenate((x,np.broadcast_to(mean,(len(x),49,EXTRA))),axis=-1).astype(np.float32)
