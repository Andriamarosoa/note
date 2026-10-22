"""Onset-only learned error-head evidence for difficult polyphony/restrikes.

69 inputs: original genuine-PCM256 63D causal acoustic/harmonic evidence
plus six PRE-SELECTION state features of the SAME independent Onset axis.
This module never reads GT, the emitted NoteOn or another decoder axis.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
from .acoustics import HOP
from .predecision_risk import predecision_matrix, DIM as BASE_DIM

SCHEMA = "ONSET_ERROR_HEAD_HARDNEGATIVE_NATIVE256_69D_V1"
STATE_NAMES = ("onset_is_active","onset_is_armed",
               "onset_low_release_fraction","onset_high_confirm_fraction",
               "onset_age_since_noteon_256frames","onset_refractory_elapsed_fraction")
DIM = BASE_DIM+len(STATE_NAMES)
assert HOP==256 and DIM==69

def capture_selector_state(selector,end_sample):
    """Snapshot state BEFORE any selector makes this hop's decision.

    All inputs represent the causal, previous-hop state. Does not mutate
    selector and does not add even one 256-sample confirmation hop.
    """
    if type(end_sample) is not int or end_sample<=0 or end_sample%HOP:
        raise ValueError("native hop256 boundary required")
    if end_sample!=selector.last_end+HOP:
        raise AssertionError("Onset selector state must precede risk prediction")
    for field in ("active","armed","low","high","last_on"):
        if np.asarray(getattr(selector,field)).shape!=(49,):
            raise ValueError("Onset state ABI mismatch: "+field)
    age=np.maximum((end_sample-selector.last_on).astype(np.float64)/HOP,0.)
    state=np.column_stack((
        selector.active.astype(np.float32),
        selector.armed.astype(np.float32),
        np.minimum(1.,selector.low.astype(float)/max(1,selector.release_steps)),
        np.minimum(1.,selector.high.astype(float)/max(1,selector.confirmation_steps)),
        np.minimum(age,256.)/256.,
        np.minimum(1.,age/max(1,selector.config.refractory_frames)),
    )).astype(np.float32,copy=False)
    if state.shape!=(49,6) or not np.isfinite(state).all():
        raise AssertionError("nonfinite private Onset preselector snapshot")
    return state.copy()

def hard_matrix(features,posterior,count,rescue_quality,rescue_history,selector_state):
    base=predecision_matrix(features,posterior,count,rescue_quality,rescue_history)
    state=np.asarray(selector_state,np.float32)
    if state.shape!=(49,6) or not np.isfinite(state).all():
        raise ValueError("Missing PREDECISION Onset state (49,6)")
    out=np.concatenate((base,state),axis=1)
    if out.shape!=(49,DIM):
        raise AssertionError("hard-negative 69D feature schema mismatch")
    return np.ascontiguousarray(out)

def hard_event_vector(record):
    x=np.asarray(record.get("hard_predecision_vector",[]),np.float32)
    if x.shape!=(DIM,) or not np.isfinite(x).all():
        raise ValueError("missing signed causal 69D evidence")
    return x

class HardNegativeRiskModel:
    """Trained/scaled 69D MLP with its OWN weights (no 1024 checkpoint)."""
    requires_selector_state=True
    trained_hop_samples=256

    def __init__(self,model,*,cut):
        if getattr(model,"n_features_in_",None)!=DIM or not 0.<=cut<=1.:
            raise ValueError("untrained or incompatible native256 neural head")
        self.model=model
        self.cut=float(cut)
        self.calls=0

    def predict_correct_probability(self,features,p,k,rescue,history,selector_state):
        x=hard_matrix(features,p,k,rescue,history,selector_state)
        probabilities=np.asarray(self.model.predict_proba(x),np.float32)
        self.calls+=1
        if (probabilities.shape!=(49,2) or not np.isfinite(probabilities).all()
            or np.any(probabilities<0.) or np.any(probabilities>1.)):
            raise ValueError("invalid 69D learned risk head output")
        return probabilities[:,1].copy()

def load_hard_model(path,metadata_path,*,expected_sha256):
    """Only SHA-pinned trusted scientific joblib models may be deserialized."""
    import joblib
    path=Path(path)
    measured=hashlib.sha256(path.read_bytes()).hexdigest()
    if measured!=expected_sha256:
        raise ValueError("learned risk checkpoint SHA256 mismatch")
    meta=json.loads(Path(metadata_path).read_text())
    if (meta.get("schema")!=SCHEMA or meta.get("training_hop_samples")!=256
        or meta.get("feature_dim")!=DIM or
        meta.get("checkpoint_sha256")!=measured or
        meta.get("frozen_before_outer")!=True):
        raise ValueError("model metadata fails native256 / holdout lineage")
    return HardNegativeRiskModel(joblib.load(path),cut=float(meta["calibrated_cut"]))
