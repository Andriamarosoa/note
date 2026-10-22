"""Independent learned FALSE-ATTACK discrimination head inside Onset, PCM256.

Falsifiable issue #142: temporal74 (69 native acoustic + 4 ages from PREVIOUS
native main NoteOns + pretrained MLP69 correctness) compared fairly to
control69 and control70 (pretrained correctness but no temporal parents).
All three are separately trained neural weights, same heldout fold and seed.

No ground truth, future samples, new NoteOn, or state feedback to Main.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from .acoustics import HOP
from .error_evidence69 import hard_matrix

SCHEMA="ONSET_FP_TIMELINE_ABLATION_TRUE256_V1"
VARIANTS={"control69":69,"control70":70,"temporal74":74}
PARENT_INTERVALS=(12,19,24,28)
FEATURE_DIM=74
HEAD_NAMES=tuple("fp_discriminator_"+n for n in VARIANTS)

def causal_parent_matrix(last_main_noteon,end_sample):
    """Age since previous Main NoteOn at parent intervals, strictly prior hops.

    1 means absent or >= 32 hops old; no GT or same-hop Main emitted events
    may enter this matrix. O(49*4), no waiting for the future.
    """
    last=np.asarray(last_main_noteon)
    if (HOP!=256 or type(end_sample)is not int or end_sample<=0 or
        end_sample%HOP or last.shape!=(49,) or last.dtype.kind not in "iu"):
        raise ValueError("FP head needs complete prior 256 Onset state")
    out=np.ones((49,4),np.float32)
    for i,offset in enumerate(PARENT_INTERVALS):
        valid=(last[:49-offset]>=0)&(last[:49-offset]<end_sample)
        age=np.clip((end_sample-last[:49-offset].astype(np.int64))/
                    float(32*HOP),0.,1.).astype(np.float32)
        out[offset:,i]=np.where(valid,age,1.)
    return out

def discriminator_matrix(features,p,k,q,history,selector_state,
                         parent_ages,frozen_correctness):
    base=hard_matrix(features,p,k,q,history,selector_state)
    parent=np.asarray(parent_ages,np.float32)
    confidence=np.asarray(frozen_correctness,np.float32)
    if (base.shape!=(49,69) or parent.shape!=(49,4) or
        confidence.shape!=(49,) or not np.isfinite(parent).all() or
        not np.isfinite(confidence).all() or np.any(confidence<0) or
        np.any(confidence>1)):
        raise ValueError("missing Onset predecision 74D causal inputs")
    result=np.column_stack((base,parent,confidence)).astype(np.float32)
    if result.shape!=(49,FEATURE_DIM):
        raise AssertionError("FP learned feature schema unexpectedly changed")
    return np.ascontiguousarray(result)

class FPDiscriminatorModel:
    """Another independent Onset neural specialist with own 256-trained weights."""
    trained_hop_samples=256
    def __init__(self,model,variant,cut):
        if (variant not in VARIANTS or
            getattr(model,"n_features_in_",None)!=VARIANTS[variant] or
            not 0<=float(cut)<=1):
            raise ValueError("incompatible trained native256 FP discriminator")
        self.model=model
        self.variant=variant
        self.cut=float(cut)
        self.calls=0
    def predict_correct(self,all74):
        matrix=np.asarray(all74,np.float32)
        if matrix.shape!=(49,74) or not np.isfinite(matrix).all():
            raise ValueError("FP model did not see exact causal 74D ABI")
        if self.variant=="control69":
            features=matrix[:,:69]
        elif self.variant=="control70":
            features=np.concatenate((matrix[:,:69],matrix[:,73:74]),axis=1)
        else:
            features=matrix
        pred=np.asarray(self.model.predict_proba(features),np.float32)
        self.calls+=1
        if (pred.shape!=(49,2) or not np.isfinite(pred).all() or
            np.any(pred<0) or np.any(pred>1)):
            raise ValueError("learned FP predictor failed 49 pitches")
        return np.ascontiguousarray(pred[:,1])

def load_fp_discriminator(weights_path,metadata_path,*,expected_sha256):
    """Deserialize only a pinned project-owned checkpoint, never unknown .joblib."""
    import joblib
    weights_path=Path(weights_path)
    actual=hashlib.sha256(weights_path.read_bytes()).hexdigest()
    meta=json.loads(Path(metadata_path).read_text())
    if (actual!=expected_sha256 or
        meta.get("checkpoint_sha256")!=actual or
        meta.get("schema")!=SCHEMA or
        meta.get("trained_hop_samples")!=256 or
        meta.get("predecision_hop_samples")!=256 or
        meta.get("feature_dim")!=VARIANTS.get(meta.get("variant")) or
        meta.get("calibrated_only_on_CAL") is not True or
        meta.get("trained_only_on_TRAIN") is not True or
        meta.get("frozen_before_OUTER") is not True):
        raise ValueError("untrusted or leakage-risk FP neural checkpoint")
    return FPDiscriminatorModel(
        joblib.load(weights_path),meta["variant"],meta["calibrated_cut"])

def select_known_onsets(original_events,correct_by_pitch,cut):
    """Pure Onset-head arbitration; can suppress but NEVER create an attack."""
    p=np.asarray(correct_by_pitch,np.float32)
    if (p.shape!=(49,) or not np.isfinite(p).all() or
        np.any(p<0) or np.any(p>1) or not 0<=cut<=1):
        raise ValueError("invalid PRE-DECISION FP-veto probabilities")
    return tuple(e for e in original_events
                 if p[int(e.pitch)-40]>=cut)

def feature_variant(matrix74,name):
    m=np.asarray(matrix74,np.float32)
    if m.ndim!=2 or m.shape[1]!=74:
        raise ValueError("74D event vectors expected")
    if name=="control69":return m[:,:69]
    if name=="control70":return np.column_stack((m[:,:69],m[:,73]))
    if name=="temporal74":return m
    raise ValueError("unknown registered ablation")
