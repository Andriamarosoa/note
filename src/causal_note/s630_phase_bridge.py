"""S630: preserve short-lived onset evidence until sustain appears in NOTE Live.

The S625 audit identified 46 missing attacks where native onset AND sustain
crossed thresholds in different frames. This selection combines only
PAST onset evidence with CURRENT sustain, no future audio lookahead, no
forced delay timer, and no modifications to the original S80/S614 heads.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import CausalSelection,Selection,PITCHES


@dataclass(frozen=True)
class PhasePolicy:
    lookback_frames:int=5
    onset_min:float=.6
    sustain_min:float=.5

    def __post_init__(self):
        if type(self.lookback_frames) is not int or not 1<=self.lookback_frames<=12:
            raise ValueError("1..12 past frames")
        if not (0<self.onset_min<=1. and 0<self.sustain_min<=1.):
            raise ValueError("Confidence threshold invalid")


class CausalPhaseBridge:
    def __init__(self,selection:Selection,policy:PhasePolicy):
        self.state=CausalSelection(selection)
        self.policy=policy
        self.history=deque(maxlen=policy.lookback_frames)
        self.helped=0

    def step(self,p,k,end):
        p=np.asarray(p,np.float32)
        if p.shape!=(49,3) or not np.isfinite(p).all():
            raise ValueError("Native per-pitch attack/sustain/release required")
        current=p.copy()
        if self.history:
            prior=np.stack(self.history).max(axis=0)
            eligible=(prior>=self.policy.onset_min)&(
                p[:,1]>=self.policy.sustain_min)&(
                p[:,0]<self.state.config.onset)&(
                ~self.state.active)&self.state.armed
            current[eligible,0]=self.state.config.onset
            self.helped+=int(eligible.sum())
        emitted=self.state.step(current,k,int(end),int(end))
        # Only CURRENT observation added *after* emission; no future audio.
        self.history.append(p[:,0].copy())
        return emitted

    def finish(self,n):
        return self.state.panic(int(n))


def variants():
    return [
      ("S614_baseline",None),
      ("phase_3_060_050",PhasePolicy(3,.6,.5)),
      ("phase_5_060_050",PhasePolicy(5,.6,.5)),
      ("phase_7_060_050",PhasePolicy(7,.6,.5)),
      ("phase_10_060_050",PhasePolicy(10,.6,.5)),
      ("phase_3_075_050",PhasePolicy(3,.75,.5)),
      ("phase_5_075_050",PhasePolicy(5,.75,.5)),
      ("phase_7_075_050",PhasePolicy(7,.75,.5)),
      ("phase_5_060_060",PhasePolicy(5,.6,.6)),
      ("phase_7_060_060",PhasePolicy(7,.6,.6)),
    ]
