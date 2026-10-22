"""Onset-specific S613+S701 trained-hop256 causal inference, no shared axis."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .acoustics import CausalPitchFeatures,HOP,SR
from .selector import CausalSelection,Selection
from .heads import OnsetHeadStack, OnsetHeadSelector

SCHEMA="ONSET_S613_S701_REAL_256_TRAINING_V1"
@dataclass(frozen=True)
class Intent:
    kind:str
    pitch:int
    sample:int
    confidence:float
    reason:str
@dataclass(frozen=True)
class Decision:
    sample:int
    intents:tuple=()
    candidate_intents:tuple=()
    evidence:tuple=()
    candidate_evidence:tuple=()
    predecision_intents:tuple=()
    head_doubts:tuple=()

class Stream:
    """S613 + S701 both called exactly once per received PCM256 decision."""
    def __init__(self,native,s701,*,audit_events=False,risk_model=None):
        policy=Selection(**native.metadata.get("selection",{}))
        self.native=native;self.s701=s701
        self.frontend=CausalPitchFeatures(hop=HOP)
        # Independent learned heads now form one named Onset architecture.
        # Risk is an Onset HEAD; only the private selector may use its doubt.
        self.onset_heads=OnsetHeadStack(native,s701,risk_model)
        self.s701_features=self.onset_heads.rescue.cache
        self.head_selector=OnsetHeadSelector(
            calibrated_correctness_cut=(
                risk_model.cut if risk_model is not None else 0.5))
        self.base=CausalSelection(policy,hop=HOP)
        self.shadow=CausalSelection(policy,hop=HOP)
        self.received=0;self.last_decision=0;self.calls=0;self.s701_calls=0
        self.closed=False
        self.audit_events=bool(audit_events)
        self.risk_model=risk_model
        self.risk_calls=0
        self.risk_selector=(CausalSelection(policy,hop=HOP)
                            if risk_model is not None else None)
    def _intents(self,events,post,reason):
        return tuple(Intent("note_on",int(e["pitch"]),int(e["decision_sample"]),
                     float(post[e["pitch"]-40,0]),reason)
                     for e in events if e["type"]=="note_on")
    def _step(self,end,feat):
        # Every head computes its opinion from current/past PCM first.
        opinion=self.onset_heads.predict(feat,end_sample=end)
        p,k=opinion.main_posterior,opinion.native_k
        score,x=opinion.rescue_probability,opinion.rescue_history
        self.s701_calls=self.onset_heads.rescue.calls
        candidate=self.head_selector.rescue_candidate(
            p,score,onset=self.base.config.onset,
            active_on=self.base.config.active_on)
        if not np.array_equal(candidate[:,1:],p[:,1:]):
            raise AssertionError("Rescue Onset head changed sustain/release")
        if end!=self.last_decision+HOP:raise AssertionError("Onset non-contiguous PCM256")
        self.last_decision=end;self.calls+=1

        # The MLP's P(error) is an independent learned Onset head opinion,
        # computed BEFORE any selection. Main decisions are NEVER overwritten.
        risk_intents=()
        if self.onset_heads.error is not None:
            self.risk_calls=self.onset_heads.error.calls
            gated=self.head_selector.challenged_candidate(
                p,opinion.correctness_probability)
            proposal=self.risk_selector.step(gated,k,end,end)
            risk_intents=self._intents(
                proposal,gated,"onset_error_head_challenged256")
        a=self.base.step(p,k,end,end)
        b=self.shadow.step(candidate,k,end,end)
        native_intents=self._intents(a,p,"s613_trained256")
        shadow_intents=self._intents(b,candidate,"s701_trained256_shadow")
        # A native NoteOn can be contested WITHOUT suppressing it.
        # Challenge decisions are explicitly owned by the Onset selector.
        head_doubts=self.head_selector.doubts(a,opinion)
        if self.audit_events:
            from .harmonic_evidence import onset_event_evidence
            from .predecision_risk import predecision_matrix
            before_selection=predecision_matrix(feat,p,k,score,x)
            audit_native=[]
            audit_shadow=[]
            for intent in native_intents:
                evidence=onset_event_evidence(feat,p,k,score,intent.pitch,intent.sample)
                evidence["predecision_vector"]=before_selection[intent.pitch-40].astype(float).tolist()
                audit_native.append(evidence)
            for intent in shadow_intents:
                evidence=onset_event_evidence(feat,p,k,score,intent.pitch,intent.sample)
                evidence["predecision_vector"]=before_selection[intent.pitch-40].astype(float).tolist()
                audit_shadow.append(evidence)
            audit_native=tuple(audit_native)
            audit_shadow=tuple(audit_shadow)
        else:
            audit_native=audit_shadow=()
        return Decision(end,native_intents,shadow_intents,audit_native,audit_shadow,risk_intents,head_doubts)
    def push(self,samples,*,start_sample):
        if self.closed or type(start_sample)is not int or start_sample!=self.received:
            raise ValueError("Onset PCM discontinuity or closed")
        x=np.asarray(samples)
        if x.ndim!=1 or x.dtype.kind!="f" or not np.isfinite(x).all() or len(x)>4096:
            raise ValueError("finite mono PCM float block <=4096 required")
        out=[];off=0
        while off<len(x):
            take=min(len(x)-off,HOP-self.received%HOP)
            block=np.asarray(x[off:off+take],np.float32)
            frames=self.frontend.push(block,start_sample=self.received)
            self.received+=take;off+=take
            for end,features in frames:
                if end!=self.received:raise AssertionError("Onset future read")
                out.append(self._step(end,features))
        return tuple(out)
    def finish(self,sample=None):
        self.closed=True
        self.base.panic(self.received)
        self.shadow.panic(self.received)
        if self.risk_selector is not None:
            self.risk_selector.panic(self.received)
        return ()
