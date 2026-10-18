"""S729 per-note phase-reset evidence on top of signed S708 288D inputs.

E2 -> E3 can share harmonic frequency lines yet introduce another *complex*
oscillator, changing the time-continuity of phases at the overlapping partials.
Magnitude-only memory S725 cannot distinguish all such cases.

S706's existing three-window 24 partial cos and sin channels are translated
into amplitude-reliable harmonic phase discontinuity and synchronous attack.
Append 32 channels, no S708 ABI mutation. A past-only E2 owner posterior
gates octave/third-harmonic parent evidence; K remains a separate learned head.
See Bronson & Depalle ICASSP 2014, phase-constrained complex NMF overlap.
"""
from __future__ import annotations
import numpy as np

BASE=288
EXTRA=32
TOTAL=BASE+EXTRA
AMP=192
CONTRAST=216
COS=240
SUSTAIN=127


class CausalPhaseSourceEvidence:
    def __init__(self):
        self.lastlog=None
        self.owner=np.zeros(49,np.float32)
        i=np.arange(49)
        self.low12=np.maximum(0,i-12)
        self.low19=np.maximum(0,i-19)
        self.ok12=(i>=12).astype(np.float32)
        self.ok19=(i>=19).astype(np.float32)

    def step(self,row):
        x=np.asarray(row,np.float32)
        if x.shape!=(49,BASE) or not np.isfinite(x).all():
            raise ValueError("S729 expects original S708 finite [49,288] acoustic ABI")
        amp=x[:,AMP:AMP+24].reshape(49,3,8)
        contrast=x[:,CONTRAST:CONTRAST+24].reshape(49,3,8)
        cos=x[:,COS:COS+24].reshape(49,3,8)
        if self.lastlog is None:
            change=np.zeros_like(amp)
            jump=np.zeros_like(cos)
        else:
            change=np.maximum(amp-self.lastlog,0.)
            jump=np.clip((1.-cos)*.5,0.,1.)
        # Low-SNR phase may be arbitrary. Reliability attenuates pure
        # no-energy phase changes but no future signal is consulted.
        reliability=np.clip(amp/(2.+np.maximum(amp,0.)),0.,1.)
        phase=jump*reliability
        onset=phase*change
        spectral=phase*np.clip((contrast+1.)/3.,0.,1.)
        parent12=phase[self.low12]
        parent19=phase[self.low19]
        even12=parent12[:,:,[1,3,5,7]].mean(-1)
        odd12=parent12[:,:,[0,2,4,6]].mean(-1)
        third19=parent19[:,:,[2,5]].mean(-1)
        others19=parent19[:,:,[0,1,3,4,6,7]].mean(-1)
        oct_phase=(even12-odd12)*self.ok12[:,None]*self.owner[self.low12,None]
        fifth_phase=(third19-others19)*self.ok19[:,None]*self.owner[self.low19,None]
        flags=np.stack((
           self.ok12*self.owner[self.low12],
           self.ok19*self.owner[self.low19]),-1)
        out=np.concatenate((phase.mean(1),onset.mean(1),spectral.mean(1),
                     oct_phase,fifth_phase,flags),-1)
        if out.shape!=(49,EXTRA) or not np.isfinite(out).all():
            raise AssertionError("S729 32D phase onset evidence ABI error")
        self.lastlog=amp.copy()
        self.owner=.8*self.owner+.2*np.clip(x[:,SUSTAIN],0.,1.)
        return out.astype(np.float32)


def phase_track(base):
    x=np.asarray(base,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,BASE) or not np.isfinite(x).all():
        raise ValueError("S729 [T,49,288] finite ABI required")
    agent=CausalPhaseSourceEvidence()
    return np.stack([agent.step(row) for row in x],0) if len(x) else np.empty((0,49,EXTRA),np.float32)


def augmented_track(base):
    x=np.asarray(base,np.float32)
    return np.concatenate((x,phase_track(x)),axis=-1).astype(np.float32)


def disabled_phase_track(base,extra_mean):
    x=np.asarray(base,np.float32);mean=np.asarray(extra_mean,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,BASE) or mean.shape!=(EXTRA,):
        raise ValueError("S729 disabled phase control contract")
    return np.concatenate((x,np.broadcast_to(mean,(len(x),49,EXTRA))),axis=-1).astype(np.float32)
