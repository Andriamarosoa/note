"""Genuinely learned, CAUSAL, pre-selection Onset correctness predictor at PCM256.

This network is a separate learned classifier. It does NOT reuse S708 weights,
future audio or downstream note/event labels. It sees acoustic S613 evidence,
the trained S701 posterior/history, and private harmonic competition, all
computed on the current available PCM256 sample boundary. The risk score is
calculated BEFORE any selector decides whether to emit NoteOn.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .acoustics import HOP, SR
from .harmonic_evidence import HARMONIC_FIELDS, current_pitch_strength

SCHEMA = "ONSET_PREDECISION_RISK_MLP_TRAINED_256_V1"
DIM = len(HARMONIC_FIELDS) + 3 + 1 + 7 + 1 + 39
assert DIM == 63
FEATURE_NAMES = (
    *HARMONIC_FIELDS,
    "s613_attack", "s613_sustain", "s613_release",
    "s701_attack_quality", *(f"s613_k{i}" for i in range(7)),
    "pitch_normalized", *(f"s701_history_{i}" for i in range(39)),
)

def predecision_matrix(features, posterior, count, s701_quality, s701_context):
    """All 49 risk forecasts from only current/past PCM; no Onset selection called."""
    if HOP != 256:
        raise AssertionError("trained predecision model only allows PCM256")
    f=np.asarray(features,np.float32)
    p=np.asarray(posterior,np.float32)
    k=np.asarray(count,np.float32)
    q=np.asarray(s701_quality,np.float32)
    history=np.asarray(s701_context,np.float32)
    if (f.shape!=(49,126) or p.shape!=(49,3) or k.shape!=(7,)
        or q.shape!=(49,) or history.shape!=(49,39)
        or not all(np.isfinite(x).all() for x in (f,p,k,q,history))):
        raise ValueError("Onset predecision ABI mismatch")
    strengths=current_pitch_strength(f)
    own=strengths
    low=np.zeros((4,49),np.float32)
    lower_on=np.zeros((4,49),np.float32)
    lower_sus=np.zeros((4,49),np.float32)
    for i,delta in enumerate((12,19,24,28)):
        low[i,delta:]=strengths[:-delta]
        lower_on[i,delta:]=p[:-delta,0]
        lower_sus[i,delta:]=p[:-delta,1]
    strongest=low.max(axis=0)
    strongest_on=lower_on.max(axis=0)
    strongest_sus=lower_sus.max(axis=0)
    neighbor=np.maximum(np.r_[0.,strengths[:-1]],
                        np.r_[strengths[1:],0.]).astype(np.float32)
    f2=np.maximum(f[:,[1,9,17]],0.).mean(axis=1)
    f3=np.maximum(f[:,[2,10,18]],0.).mean(axis=1)
    recent=np.maximum(f[:,54:57],0.).mean(axis=1)
    distant=np.maximum(f[:,66:69],0.).mean(axis=1)
    high_k=float(np.max(k[2:]))
    features12=np.column_stack((
        own,strongest,strongest/np.maximum(own,.2),
        strongest_on,strongest_sus,neighbor,neighbor/np.maximum(own,.2),
        recent-distant,f2/(.2+own),f3/(.2+own),
        np.full(49,high_k,np.float32),own-strongest,
    ))
    matrix=np.column_stack((
        features12,p,q,np.broadcast_to(k,(49,7)),
        np.arange(40,89,dtype=np.float32)/127.,history,
    )).astype(np.float32,copy=False)
    if matrix.shape!=(49,DIM) or not np.isfinite(matrix).all():
        raise AssertionError("invalid predecision matrix")
    return np.ascontiguousarray(matrix)

def event_vector(event):
    """Round-trip exported event evidence to the EXACT pre-decision inference ABI."""
    v=np.asarray(event.get("predecision_vector",[]),np.float32)
    if v.shape!=(DIM,) or not np.isfinite(v).all():
        raise ValueError("missing verified predecision risk features for event")
    return v

class FrozenRiskModel:
    """A trained 63D scikit-learn neural MLP, predicting P(candidate_is_correct)."""

    def __init__(self,model,*,cut=0.5,training_hop_samples=256):
        if training_hop_samples != 256 or HOP != 256:
            raise ValueError("risk neural weights not genuinely native256")
        if getattr(model,"n_features_in_",None)!=DIM:
            raise ValueError("risk MLP has incompatible 63D input")
        if not 0 <= cut <= 1:
            raise ValueError("invalid independently calibrated risk cut")
        self.model=model
        self.cut=float(cut)
        self.calls=0

    def predict_correct_probability(self,features,posterior,count,s701_quality,s701_context):
        x=predecision_matrix(features,posterior,count,s701_quality,s701_context)
        probs=np.asarray(self.model.predict_proba(x),np.float32)
        self.calls+=1
        if probs.shape!=(49,2) or not np.isfinite(probs).all():
            raise ValueError("incorrect neural predecision probabilities")
        return probs[:,1].copy()

def load_frozen(joblib_path,metadata_path,*,expected_sha256):
    """Never deserialize unverified weights; the source MUST be our own artifact."""
    import joblib
    path=Path(joblib_path)
    observed=hashlib.sha256(path.read_bytes()).hexdigest()
    if observed!=expected_sha256:
        raise ValueError("risk neural checkpoint digest mismatch")
    meta=json.loads(Path(metadata_path).read_text())
    if (meta.get("schema")!=SCHEMA or meta.get("trained_hop_samples")!=256
        or meta.get("dim")!=DIM or meta.get("checkpoint_sha256")!=observed):
        raise ValueError("risk model training metadata mismatch")
    return FrozenRiskModel(joblib.load(path),cut=float(meta["calibrated_cut"]))
