"""Onset-specific S613+S701 trained-hop256 causal inference, no shared axis."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .acoustics import CausalPitchFeatures,HOP,SR
from .s701_features import LiveS701Features
from .selector import CausalSelection,Selection

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

class Stream:
    """S613 + S701 both called exactly once per received PCM256 decision."""
    def __init__(self,native,s701,*,audit_events=False,risk_model=None):
        policy=Selection(**native.metadata.get("selection",{}))
        self.native=native;self.s701=s701
        self.frontend=CausalPitchFeatures(hop=HOP)
        self.s701_features=LiveS701Features()
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
        post,k=self.native.predict(feat)
        p,k=np.asarray(post,np.float32),np.asarray(k,np.float32)
        x=self.s701_features.step(feat,p,k)
        score=np.asarray(self.s701.predict_proba(x)[:,1],np.float32)
        self.s701_calls+=1
        if score.shape!=(49,) or not np.isfinite(score).all():
            raise ValueError("S701 specialist invalid")
        candidate=p.copy()
        selected=(p[:,0]>=.20)&(p[:,0]<self.base.config.onset)&(p[:,1]>=self.base.config.active_on)&(score>=.75)
        candidate[selected,0]=min(.999,self.base.config.onset+.025)
        if not np.array_equal(candidate[:,1:],p[:,1:]):
            raise AssertionError("Onset cannot change sustain/release")
        if end!=self.last_decision+HOP:raise AssertionError("Onset non-contiguous PCM256")
        self.last_decision=end;self.calls+=1

        # A learned P(error) is computed BEFORE either Onset selector emits
        # a note. Only current and past PCM256 features are supplied.
        risk_intents=()
        if self.risk_model is not None:
            from .predecision_risk import predecision_matrix
            predicted_correct=np.asarray(
                self.risk_model.predict_correct_probability(feat,p,k,score,x),
                np.float32)
            self.risk_calls+=1
            if predicted_correct.shape!=(49,) or not np.isfinite(predicted_correct).all():
                raise ValueError("invalid before-decision neural risk")
            gated=p.copy()
            gated[predicted_correct<self.risk_model.cut,0]=0.
            proposal=self.risk_selector.step(gated,k,end,end)
            risk_intents=self._intents(proposal,gated,"neural_predecision_256")
        a=self.base.step(p,k,end,end)
        b=self.shadow.step(candidate,k,end,end)
        native_intents=self._intents(a,p,"s613_trained256")
        shadow_intents=self._intents(b,candidate,"s701_trained256_shadow")
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
        return Decision(end,native_intents,shadow_intents,audit_native,audit_shadow,risk_intents)
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
