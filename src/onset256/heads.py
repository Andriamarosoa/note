"""Independent trainable heads composing the *single* Onset axis, PCM256.

This is a head stack, not an external post-hoc filter:
  - main:    frozen S613 native-256 acoustic neural weights
  - rescue:  separately trained S701 native-256 specialist weights
  - doubt:   separately trained 63D Onset MLP risk weights, predicting
             P(incorrect attack) BEFORE the final Onset selector.

The heads may READ each other's Onset-only evidence but NEVER update each
other's weights/states and NEVER touch Exact-K, Pitch, Offset, Full Note or
public MIDI. The final decision belongs only to the private Onset selector.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .acoustics import HOP, PITCHES
from .s701_features import LiveS701Features

SCHEMA = "ONSET_EMBEDDED_INDEPENDENT_HEADS_PCM256_V1"
HEAD_NAMES = ("main_attack", "rescue_attack", "error_prediction")

@dataclass(frozen=True)
class HeadOpinions:
    """Scores at ONE genuinely new 256-sample time boundary, pre-selector."""
    sample: int
    main_posterior: np.ndarray
    native_k: np.ndarray
    rescue_probability: np.ndarray
    rescue_history: np.ndarray
    error_probability: Optional[np.ndarray]
    correctness_probability: Optional[np.ndarray]

class MainAttackHead:
    name="main_attack"
    trained_hop_samples=256

    def __init__(self, native_model):
        self.model=native_model
        self.calls=0

    def predict(self,acoustic_features):
        p,k=self.model.predict(acoustic_features)
        self.calls+=1
        p=np.asarray(p,np.float32)
        k=np.asarray(k,np.float32)
        if (p.shape!=(49,3) or k.shape!=(7,) or
            not np.isfinite(p).all() or not np.isfinite(k).all()):
            raise ValueError("Invalid S613 pretrained-256 main Onset head")
        return p,k

class RescueAttackHead:
    name="rescue_attack"
    trained_hop_samples=256

    def __init__(self, specialist):
        self.model=specialist
        self.cache=LiveS701Features()
        self.calls=0

    def predict(self,feat,p,k):
        history=self.cache.step(feat,p,k)
        prob=np.asarray(self.model.predict_proba(history),np.float32)
        self.calls+=1
        if (history.shape!=(49,39) or prob.shape!=(49,2) or
            not np.isfinite(prob).all()):
            raise ValueError("Invalid S701 pretrained-256 rescue Onset head")
        return prob[:,1].copy(),history

class ErrorPredictionHead:
    """True independently trained neural head with its own model weights.

    Its output is a risk opinion, NOT an automatic veto. The downstream
    Onset-owned selector alone chooses how to use this opinion. No access to
    final NoteOn, annotation truth or future frames.
    """
    name="error_prediction"
    trained_hop_samples=256

    def __init__(self, risk_model):
        if not hasattr(risk_model,"predict_correct_probability"):
            raise ValueError("Error head requires genuine learned 256 risk model")
        if not 0 <= risk_model.cut <= 1:
            raise ValueError("Risk cut must have been previously calibrated")
        self.model=risk_model
        self.calls=0

    @property
    def calibrated_correctness_cut(self):
        return float(self.model.cut)

    def predict(self,feat,p,k,rescue_probability,rescue_history,selector_state=None):
        # Hard-negative error models use six causal PRE-SELECTION Onset state
        # features. Legacy trained 63D risk models remain ABI compatible.
        if getattr(self.model,"requires_selector_state",False):
            correct=self.model.predict_correct_probability(
                feat,p,k,rescue_probability,rescue_history,selector_state)
        else:
            correct=self.model.predict_correct_probability(
                feat,p,k,rescue_probability,rescue_history)
        correct=np.asarray(correct,np.float32)
        self.calls+=1
        if (correct.shape!=(49,) or not np.isfinite(correct).all() or
            np.any(correct < 0.) or np.any(correct > 1.)):
            raise ValueError("Invalid pretrained error prediction head")
        return (1.-correct).astype(np.float32),correct.copy()

class OnsetHeadStack:
    """All three private Onset heads evaluated before the private selector."""

    def __init__(self,native,s701,risk_model=None):
        self.main=MainAttackHead(native)
        self.rescue=RescueAttackHead(s701)
        self.error=(ErrorPredictionHead(risk_model) if risk_model is not None else None)
        self.last_end=0
        self.calls=0

    @property
    def names(self):
        return HEAD_NAMES if self.error is not None else HEAD_NAMES[:2]

    def predict(self,feat,*,end_sample,selector_state=None):
        if (HOP!=256 or type(end_sample) is not int or
            end_sample!=self.last_end+256):
            raise AssertionError("each Onset head must run each fresh PCM256")
        p,k=self.main.predict(feat)
        rescue,history=self.rescue.predict(feat,p,k)
        error=correct=None
        if self.error is not None:
            error,correct=self.error.predict(feat,p,k,rescue,history,selector_state)
        self.last_end=end_sample
        self.calls+=1
        if (self.main.calls!=self.calls or self.rescue.calls!=self.calls or
            (self.error is not None and self.error.calls!=self.calls)):
            raise AssertionError("Onset head stack missed 256-sample inference")
        return HeadOpinions(sample=end_sample,main_posterior=p,native_k=k,
                            rescue_probability=rescue,rescue_history=history,
                            error_probability=error,
                            correctness_probability=correct)

@dataclass(frozen=True)
class Doubt:
    """Prediction of likely error, not a ground-truth label."""
    pitch:int
    sample:int
    predicted_error_probability:float
    predicted_correct_probability:float
    contested:bool
    origin:str

class OnsetHeadSelector:
    """Only the private Onset axis selects between independent head opinions.

    The baseline main head always remains untouched. A calibrated shadow
    route may challenge an attack when the risk model doubts it; no waiting,
    future frames or hidden changes to the primary selector are permitted.
    """

    def __init__(self, calibrated_correctness_cut=0.5):
        if not 0<=calibrated_correctness_cut<=1:
            raise ValueError("invalid private Onset calibration")
        self.correctness_cut=float(calibrated_correctness_cut)

    def rescue_candidate(self,p,rescue_probability,*,onset,active_on):
        p=np.asarray(p,np.float32)
        q=np.asarray(rescue_probability,np.float32)
        if p.shape!=(49,3) or q.shape!=(49,):
            raise ValueError("Onset rescue head dimensions incorrect")
        alternate=p.copy()
        mask=((p[:,0]>=.20)&(p[:,0]<onset)&
              (p[:,1]>=active_on)&(q>=.75))
        alternate[mask,0]=min(.999,float(onset)+.025)
        return alternate

    def challenged_candidate(self,p,correctness):
        """Experimental Onset-only route: risk is an opinion, not MAIN edit."""
        p=np.asarray(p,np.float32)
        if correctness is None:
            return p.copy()
        c=np.asarray(correctness,np.float32)
        if (p.shape!=(49,3) or c.shape!=(49,) or
            not np.isfinite(c).all()):
            raise ValueError("invalid per-pitch Onset error head")
        alternate=p.copy()
        alternate[c<self.correctness_cut,0]=0.
        if not np.array_equal(alternate[:,1:],p[:,1:]):
            raise AssertionError("risk head changed sustain/release posteriors")
        return alternate

    def doubts(self,events,opinion):
        if opinion.error_probability is None:return ()
        out=[]
        for e in events:
            if e["type"]!="note_on":continue
            i=int(e["pitch"])-40
            if i<0 or i>=49:raise ValueError("invalid challenged pitch")
            correct=float(opinion.correctness_probability[i])
            out.append(Doubt(pitch=int(e["pitch"]),
                             sample=int(e["decision_sample"]),
                             predicted_error_probability=float(opinion.error_probability[i]),
                             predicted_correct_probability=correct,
                             contested=bool(correct<self.correctness_cut),
                             origin="learned_error_prediction_head"))
        return tuple(out)
