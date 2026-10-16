"""S619: selectively recover weak NOTE onsets using INDEPENDENT acoustic novelty.

This is a shadow/offline DEV-only ablation, NOT a trained new official head.
It makes a strictly causal candidate onset eligible only when:
 - frozen full S613 hybrid per-pitch onset is below its usual 0.60,
 - same-frame sustain already clears the original 0.50 threshold,
 - positive causal spectral novelty at the candidate pitch clears
   an absolute + relative competition test.

All original NOTE onsets >= .60 stay untouched, as do original release,
K, armed/refractory, attack confirmations, output hop, FFT windows.
No ground-truth pitch or future frames are inputs to the gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_note.strict_live import CausalSelection, Selection, HOP


@dataclass(frozen=True)
class RescueGate:
    weak_onset_min: float = .45
    relative_novelty_min: float = .2
    absolute_novelty_min: float = .02
    novelty_mode: str = "recent"

    def __post_init__(self):
        if not (.0 < self.weak_onset_min < .6):
            raise ValueError("weak onset threshold must be in (0,.6)")
        if not (0 <= self.relative_novelty_min <= 1):
            raise ValueError("relative novelty must be within [0,1]")
        if not (self.absolute_novelty_min >= 0):
            raise ValueError("minimum absolute novelty must be nonnegative")
        if self.novelty_mode not in ("lag", "recent"):
            raise ValueError("Only causal lag/recent modes")


def weak_note_boost(probabilities, features, config):
    p=np.asarray(probabilities,dtype=np.float32)
    x=np.asarray(features,dtype=np.float32)
    if p.shape!=(49,3) or x.shape!=(49,126):
        raise ValueError("Expected original 49-pitch x 126 S613 features and onset/sustain/release")
    # S613 per-pitch 3-resolution score diffs; no extra FFT, no lookahead.
    # 60:63 = current physical pitch strength minus 2-hop history;
    # 114:117 = current strength minus minimum of PREVIOUS/RECENT 8 hops.
    span=x[:,60:63] if config.novelty_mode=="lag" else x[:,114:117]
    novelty=np.maximum(span,0.).max(axis=1)
    maximum=max(.0001,float(novelty.max()))
    eligible=(
        (p[:,0]>=config.weak_onset_min) &
        (p[:,0]<.60) &
        (p[:,1]>=.50) &
        (novelty>=config.absolute_novelty_min) &
        (novelty/maximum>=config.relative_novelty_min)
    )
    if not eligible.any():
        return p,0
    out=p.copy()
    out[eligible,0]=np.float32(.60001)
    return out,int(eligible.sum())


def causal_replay(ends, features, probability, count, length, config,
                  *, selection=None):
    """Run same CausalSelection at 256 hop; modifying only weak-onset column."""
    base=selection or Selection(onset=.6,active_on=.5)
    if base.onset!=.6 or base.active_on!=.5:
        raise ValueError("S619 needs the original unchanged S613 reference")
    selection_instance=CausalSelection(base,hop=HOP)
    all_events=[]
    promoted_rows=0
    for end,x,p,k in zip(ends,features,probability,count):
        modified,n=weak_note_boost(p,x,config)
        promoted_rows+=n
        all_events.extend(selection_instance.step(modified,k,int(end),int(end)))
    all_events.extend(selection_instance.panic(int(length)))
    return all_events,promoted_rows
