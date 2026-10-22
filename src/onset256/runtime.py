"""Onset-specific S613+S701 trained-hop256 causal inference, no shared axis."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .acoustics import CausalPitchFeatures,HOP,SR
from .selector import CausalSelection,Selection
from .fn_recovery import CANDIDATE_POLICIES,FNRecoverySelector
from .fp_discriminator import causal_parent_matrix,select_known_onsets,VARIANTS
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
    fn_recovery_intents:tuple=()
    fn_unified_intents:tuple=()
    fp_discriminator_intents:dict=None
    fp_discriminator_unified:dict=None

class Stream:
    """S613 + S701 both called exactly once per received PCM256 decision."""
    def __init__(self,native,s701,*,audit_events=False,risk_model=None,recovery_model=None,fp_models=None):
        policy=Selection(**native.metadata.get("selection",{}))
        self.native=native;self.s701=s701
        self.frontend=CausalPitchFeatures(hop=HOP)
        # Independent learned heads now form one named Onset architecture.
        # Risk is an Onset HEAD; only the private selector may use its doubt.
        if recovery_model is not None and risk_model is None:
            raise ValueError("FNRecovery is compared against the frozen three-head Onset reference")
        self.onset_heads=OnsetHeadStack(native,s701,risk_model,recovery_model,fp_models)
        self.fp_last_native_on=np.full(49,-10**12,np.int64)
        self.fp_calls=0
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
        self.fn_recovery_selector=None
        self.fn_recovery_arbiter=None
        if recovery_model is not None:
            spec=CANDIDATE_POLICIES[recovery_model.candidate_policy]
            self.fn_recovery_selector=CausalSelection(Selection(
                onset=spec["onset"],active_on=spec["active_on"],
                onset_reset=spec["onset_reset"],
                active_off=policy.active_off,release_frames=policy.release_frames,
                refractory_frames=policy.refractory_frames,
                max_notes=policy.max_notes,
                count_confidence=policy.count_confidence,
                confirmation_frames=1,
                restrike_onset=policy.restrike_onset),hop=HOP)
            self.fn_recovery_arbiter=FNRecoverySelector(
                recovery_model.calibrated_cut,recovery_model.enabled)
        self.fn_recovery_calls=0
    def _intents(self,events,post,reason):
        return tuple(Intent("note_on",int(e["pitch"]),int(e["decision_sample"]),
                     float(post[e["pitch"]-40,0]),reason)
                     for e in events if e["type"]=="note_on")
    def _step(self,end,feat):
        # Capture private Onset state *before* Main/Rescue/Error selection.
        # The snapshot is read-only and never extends the decision horizon.
        from .error_evidence69 import capture_selector_state
        preselector_state=capture_selector_state(self.base,end)
        # 74D parent attack chronology is strictly from native NoteOns on
        # PREVIOUS 256 boundaries. Current hop's Main decision is not known.
        parent_ages=(causal_parent_matrix(self.fp_last_native_on,end)
                     if self.onset_heads.fp_heads else None)
        opinion=self.onset_heads.predict(
            feat,end_sample=end,selector_state=preselector_state,
            parent_ages=parent_ages)
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
        # Critical fix: do not zero onset *posteriors* inside a stateful
        # selector. That artificial zero re-arms the pitch and creates
        # new NoteOn events in later 256-sample hops (S69D run 39741883968).
        # Each selector always receives the untouched main probabilities.
        a=self.base.step(p,k,end,end)
        b=self.shadow.step(candidate,k,end,end)
        risk_intents=()
        if self.onset_heads.error is not None:
            self.risk_calls=self.onset_heads.error.calls
            mirror=self.risk_selector.step(p,k,end,end)
            if mirror!=a:
                raise AssertionError("ErrorPrediction changed Main Onset internal state")
            accepted=self.head_selector.challenge_emitted_onsets(
                mirror,opinion.correctness_probability)
            risk_intents=self._intents(
                accepted,p,"onset_error_head_no_feedback256")
            main_keys={(int(e["pitch"]),int(e["decision_sample"]))
                       for e in a if e["type"]=="note_on"}
            risk_keys={(int(e.pitch),int(e.sample)) for e in risk_intents}
            if not risk_keys.issubset(main_keys):
                raise AssertionError("error head cannot create new NoteOns")
        native_intents=self._intents(a,p,"s613_trained256")
        shadow_intents=self._intents(b,candidate,"s701_trained256_shadow")
        recovered_intents=()
        unified_intents=()
        if self.onset_heads.recovery is not None:
            self.fn_recovery_calls=self.onset_heads.recovery.calls
            proposals=self.fn_recovery_selector.step(p,k,end,end)
            # FNRecovery is an Onset head; it proposes only attack events,
            # without overwriting any existing main/rescue/error-head state.
            chosen=self.fn_recovery_arbiter.select(
                end_sample=end,main_raw=native_intents,
                main_accepted=risk_intents,
                proposals=proposals,
                recovery_probability=opinion.fn_recovery_probability)
            recovered_intents=self._intents(
                chosen,p,"fn_recovery_trained_neural256")
            unified_intents=risk_intents+recovered_intents
            main_keys={(e.pitch,e.sample) for e in native_intents}
            if any((e.pitch,e.sample) in main_keys for e in recovered_intents):
                raise AssertionError("FNRecovery duplicated current Main Onset")
            if len(unified_intents)!=len(risk_intents)+len(recovered_intents):
                raise AssertionError("FNRecovery main stream was overwritten")
        # Falsifiable control69/control70/temporal74 learned independent
        # Onset head arbitration. All probability forecasts were computed
        # BEFORE any Onset selector and are only allowed to remove ALREADY
        # accepted error-head NoteOns. They never touch selector state.
        fp_intents=None
        fp_unified=None
        if self.onset_heads.fp_heads:
            fp_intents={}
            fp_unified={}
            for name,head in self.onset_heads.fp_heads.items():
                picked=select_known_onsets(
                    risk_intents,
                    opinion.fp_discriminator_correctness[name],
                    head.model.cut)
                fp_intents[name]=picked
                fp_unified[name]=picked+recovered_intents
                if not {(e.pitch,e.sample) for e in picked}.issubset(
                       {(e.pitch,e.sample) for e in risk_intents}):
                    raise AssertionError("FP head invented NoteOn")
            self.fp_calls+=1
            if (self.fp_calls!=self.calls or
                any(head.calls!=self.calls
                    for head in self.onset_heads.fp_heads.values())):
                raise AssertionError("one fresh 69/70/74D neural prediction per PCM256")
            # Update parent history only AFTER this complete hop has been
            # evaluated, never before, never with GT or FNRecovery decisions.
            for event in native_intents:
                self.fp_last_native_on[event.pitch-40]=end
        # A native NoteOn can be contested WITHOUT suppressing it.
        # Challenge decisions are explicitly owned by the Onset selector.
        head_doubts=self.head_selector.doubts(a,opinion)
        if self.audit_events:
            from .harmonic_evidence import onset_event_evidence
            from .predecision_risk import predecision_matrix
            from .error_evidence69 import hard_matrix
            before_selection=predecision_matrix(feat,p,k,score,x)
            hard_before_selection=hard_matrix(
                feat,p,k,score,x,preselector_state)
            audit_native=[]
            audit_shadow=[]
            for intent in native_intents:
                evidence=onset_event_evidence(feat,p,k,score,intent.pitch,intent.sample)
                evidence["predecision_vector"]=before_selection[intent.pitch-40].astype(float).tolist()
                evidence["hard_predecision_vector"]=hard_before_selection[intent.pitch-40].astype(float).tolist()
                audit_native.append(evidence)
            for intent in shadow_intents:
                evidence=onset_event_evidence(feat,p,k,score,intent.pitch,intent.sample)
                evidence["predecision_vector"]=before_selection[intent.pitch-40].astype(float).tolist()
                evidence["hard_predecision_vector"]=hard_before_selection[intent.pitch-40].astype(float).tolist()
                audit_shadow.append(evidence)
            audit_native=tuple(audit_native)
            audit_shadow=tuple(audit_shadow)
        else:
            audit_native=audit_shadow=()
        return Decision(end,native_intents,shadow_intents,audit_native,
                        audit_shadow,risk_intents,head_doubts,recovered_intents,
                        unified_intents,fp_intents,fp_unified)
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
        if self.fn_recovery_selector is not None:
            self.fn_recovery_selector.panic(self.received)
        return ()
