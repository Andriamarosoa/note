"""S809 independent per-axis HEAD -> SELECTOR -> DECODER pipelines.

Scientific SHADOW architecture. Not six separate released neural checkpoints.
Every axis owns its own acoustic candidate bank, selector instance, decoder
and temporal state; only a read-only causal audio snapshot is shared. A
non-intervening observer may inspect disagreements, but no axis can modify
another's predictions or the S708 source MIDI.

Two candidate *signal heads* are implemented per domain. H1 is a deterministic
current/past-frame acoustic alternative, NOT an independently trained model.
The entry points permit replacing a domain's bank with a genuine fitted model
without rewiring any other domain. No output is marked promoted.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from causal_note.s763_independent_decoders import (
    AXES, FIRST, PITCHES, DECISION_SAMPLES, Fragment, AxisDecision,
    default_decoders, ShadowArbiter
)

SCHEMA="S809_ISOLATED_SIX_AXIS_PIPELINES_V1"
HEAD_NAMES={
 "exact_k":("frozen_k_posterior","current_sustain_count_proxy"),
 "active_exact_k":("frozen_sustain","attack_supported_sustain"),
 "onset":("frozen_attack","attack_rise_supported"),
 "offset":("frozen_release","release_quiet_support"),
 "frame_pitch":("frozen_pitch_sustain","attack_supported_pitch"),
 "full_note":("frozen_joint_triplet","causal_joint_attack_sustain_release"),
}
SELECTOR_NAMES={
 "exact_k":"exact_k_private_fragment_confidence",
 "active_exact_k":"active_k_private_pitch_support",
 "onset":"onset_private_pitch_attack_rise",
 "offset":"offset_private_pitch_release_contrast",
 "frame_pitch":"frame_pitch_private_pitch_occupancy",
 "full_note":"full_note_private_joint_pitch_evidence",
}
SELECT_MODES=("native_only","causal_contrast")

def _onehot(class_id):
    arr=np.zeros(7,np.float32)
    arr[int(class_id)]=1.
    return arr

class AxisHeadBank:
    """Two genuinely distinct acoustic representations, axis-local history."""

    def __init__(self,axis):
        if axis not in AXES:raise ValueError("unknown head domain")
        self.axis=axis
        self.names=HEAD_NAMES[axis]
        self.previous=np.zeros((PITCHES,3),np.float32)
        self.last_sample=0

    def propose(self,fragment):
        if not isinstance(fragment,Fragment):raise TypeError("immutable audio fragment expected")
        if fragment.sample!=self.last_sample+DECISION_SAMPLES:
            raise ValueError("head time sequence is not causal")
        p=np.asarray(fragment.post)
        old=self.previous.copy()
        onset,sustain,release=p.T
        rise=np.maximum(onset-old[:,0],0)
        sustain_rise=np.maximum(sustain-old[:,1],0)
        q=p.copy()
        k=np.asarray(fragment.k).copy()
        if self.axis=="exact_k":
            count=min(6,int(np.count_nonzero(sustain>=.50)))
            alt_k=_onehot(count)
        elif self.axis=="active_exact_k":
            q[:,1]=np.clip(sustain+.18*onset*(1-release),0,1)
            alt_k=k
        elif self.axis=="onset":
            q[:,0]=np.clip(.83*onset+.17*rise*np.maximum(sustain,.05),0,1)
            alt_k=k
        elif self.axis=="offset":
            q[:,2]=np.clip(release+.15*(1-sustain)*release,0,1)
            alt_k=k
        elif self.axis=="frame_pitch":
            q[:,1]=np.clip(np.maximum(sustain,.80*onset*(1-release)),0,1)
            alt_k=k
        else:
            q[:,0]=np.clip(.85*onset+.15*rise,0,1)
            q[:,1]=np.clip(sustain+.10*sustain_rise*(1-release),0,1)
            q[:,2]=np.clip(release+.10*(1-sustain)*release,0,1)
            alt_k=k
        # Both proposals are independent immutable objects, never aliased.
        frozen=fragment.fork()
        alternative=Fragment(fragment.sample,q,alt_k)
        self.previous[:]=p
        self.last_sample=fragment.sample
        return frozen,alternative

@dataclass(frozen=True)
class SelectionInfo:
    axis:str
    selector:str
    selected_alternative_pitches:int
    head_names:tuple[str,str]
    mode:str
    sample:int

class AxisSelector:
    """Independent private selector state; no other axis decisions as inputs."""

    def __init__(self,axis,mode="native_only",threshold=.12):
        if axis not in AXES or mode not in SELECT_MODES:
            raise ValueError("unknown selector domain/mode")
        if not 0<=threshold<=1:raise ValueError("invalid selection cutoff")
        self.axis,self.name,self.mode=axis,SELECTOR_NAMES[axis],mode
        self.threshold=float(threshold)
        self.last_sample=0

    def select(self,original,candidate):
        if not isinstance(original,Fragment) or not isinstance(candidate,Fragment):
            raise TypeError("head candidate is not a causal fragment")
        if original.sample!=candidate.sample or (
                original.sample!=self.last_sample+DECISION_SAMPLES):
            raise ValueError("selector received mismatched/future timestamp")
        p=original.post
        onset,sustain,release=p.T
        if self.mode=="native_only":
            enable=np.zeros(PITCHES,bool)
        elif self.axis=="exact_k":
            enable=np.full(PITCHES, bool((float(original.k.max())<.80 or
                int(np.count_nonzero(sustain>=.50))!=int(original.k.argmax())) and
                np.max(sustain)>=self.threshold))
        elif self.axis=="active_exact_k":
            enable=(onset>=self.threshold)&(release<.8)
        elif self.axis=="onset":
            enable=(onset>=self.threshold)&(sustain>=.20)
        elif self.axis=="offset":
            enable=(release>=self.threshold)&(sustain<=.60)
        elif self.axis=="frame_pitch":
            enable=(onset>=self.threshold)&(release<.7)
        else:
            enable=(onset>=self.threshold)&(sustain>=.2)&(release<.75)
        post=np.where(enable[:,None],candidate.post,original.post)
        alt_k=(candidate.k if self.axis=="exact_k" and enable.any()
               else original.k)
        selected=Fragment(original.sample,post,alt_k)
        self.last_sample=original.sample
        return selected,SelectionInfo(self.axis,self.name,int(enable.sum()),
                   HEAD_NAMES[self.axis],self.mode,original.sample)

class AxisPipeline:
    """Each owned part belongs to this axis only: head, selector and decoder."""

    def __init__(self,axis,head=None,selector=None,decoder=None):
        if axis not in AXES:raise ValueError("unknown axis")
        native=default_decoders()
        self.axis=axis
        self.head=head if head is not None else AxisHeadBank(axis)
        self.selector=selector if selector is not None else AxisSelector(axis)
        self.decoder=decoder if decoder is not None else native[axis]
        if (self.head.axis!=axis or self.selector.axis!=axis):
            raise ValueError("cross-wired head or selector")

    def step(self,fragment):
        # A dedicated fragment copy even if a custom head tries to mutate it.
        first,second=self.head.propose(fragment.fork())
        selected,info=self.selector.select(first,second)
        decision=self.decoder.step(selected.fork())
        if not isinstance(decision,AxisDecision) or (
                decision.axis!=self.axis or decision.sample!=fragment.sample):
            raise AssertionError("axis decoder returned foreign or stale decision")
        if any(intent.sample!=fragment.sample for intent in decision.intents):
            raise AssertionError("future/backdated axis intent")
        return decision,info

def default_pipelines(modes=None):
    modes={} if modes is None else dict(modes)
    if any(axis not in AXES for axis in modes):
        raise ValueError("unsupported axis in selector modes")
    return {axis:AxisPipeline(axis,selector=AxisSelector(axis,modes.get(axis,"native_only")))
            for axis in AXES}

class SixIndependentAxes:
    """Run six physically distinct, testable component chains per frame.

    Renders six separate outputs; never converts them to ONE MIDI stream.
    ShadowArbiter is explicitly read-only, and final MIDI stays identical
    to the protected S708 reference unless another separately validated
    future integration explicitly replaces it.
    """

    def __init__(self,pipelines=None,order=AXES):
        self.pipelines=default_pipelines() if pipelines is None else dict(pipelines)
        if set(self.pipelines)!=set(AXES) or set(order)!=set(AXES):
            raise ValueError("six different axes and one execution order required")
        self.order=tuple(order)
        components=[]
        for axis,pipe in self.pipelines.items():
            if pipe.axis!=axis:raise ValueError("pipeline mapped to wrong axis")
            components.extend((pipe,pipe.head,pipe.selector,pipe.decoder))
        if len(set(map(id,components)))!=len(AXES)*4:
            raise ValueError("shared head/selector/decoder/axis state prohibited")
        self.last_sample=0

    def step(self,fragment):
        if not isinstance(fragment,Fragment) or (
                fragment.sample!=self.last_sample+DECISION_SAMPLES):
            raise ValueError("incorrect causal input clock")
        decisions={};selections={}
        for axis in self.order:
            # Same immutable input VALUES; distinct independent input storage.
            d,info=self.pipelines[axis].step(fragment.fork())
            decisions[axis]=d;selections[axis]=info
        self.last_sample=fragment.sample
        return decisions,selections

    @staticmethod
    def signed_midi_identity(source):
        return ShadowArbiter.preserve_signed_stream(source)
