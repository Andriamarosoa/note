"""S617 experimental causal onset bridge atop the ORIGINAL native NOTE heads.

Purpose: audit whether early high onset probabilities are being held back by
an overly stringent sustain conjunction. Only CURRENT and PAST native
probabilities / harmonics are read; there is no waiting after the attack and
no sidecar, truth or other repository model at inference.
"""
from dataclasses import dataclass
import numpy as np

from causal_note.s616_live_guard import NativeCausalGuard, GuardParameters
from causal_note.strict_live import Selection, PITCHES


@dataclass(frozen=True)
class BridgeParameters:
    min_onset: float = .75
    min_sustain: float = .25
    min_fundamental_rise: float = .15
    min_onset_rise: float = .04

    def __post_init__(self):
        if not (.5 <= self.min_onset <= 1.):
            raise ValueError("onset threshold out of range")
        for name in ("min_sustain", "min_fundamental_rise", "min_onset_rise"):
            v = getattr(self,name)
            if not np.isfinite(v) or not 0. <= v <= 1.:
                raise ValueError("invalid bridge parameter")


class NativeAttackBridge(NativeCausalGuard):
    def __init__(self,selection:Selection,bridge:BridgeParameters,
                 params:GuardParameters=GuardParameters()):
        super().__init__(selection,params)
        self.bridge=bridge
        self.previous_onset=np.zeros(len(PITCHES),np.float32)
        self.bridge_opportunities=0

    def step(self,probability,count,end_sample,available_sample,acoustic):
        p=np.asarray(probability,np.float32)
        x=np.asarray(acoustic,np.float32)
        if p.shape!=(49,3) or x.shape!=(49,126) or not np.isfinite(p).all() or not np.isfinite(x).all():
            raise ValueError("Strict native NOTE input shape required")
        curr_fundamental=np.maximum(0,x[:,(0,8,16)].mean(axis=1))
        condition=(~self.selector.active) & self.selector.armed & (
            p[:,0]>=self.bridge.min_onset) & (
            p[:,1]>=self.bridge.min_sustain) & (
            p[:,1] < self.selector.config.active_on) & (
            p[:,0]-self.previous_onset>=self.bridge.min_onset_rise) & (
            curr_fundamental-self.previous_fundamental>=self.bridge.min_fundamental_rise)
        pred=p.copy()
        pred[condition,1]=self.selector.config.active_on
        self.bridge_opportunities+=int(condition.sum())
        # Superclass can also apply validated S616 harmonic/rescue controls.
        out=super().step(pred,count,end_sample,available_sample,x)
        self.previous_onset=p[:,0].copy()
        return out
