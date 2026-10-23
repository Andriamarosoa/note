"""Independent supervised context-dependent Onset FP-expert gate, native PCM256.

Registered issue #143, no external GT, no lookahead, no held out OUTER tuning.
The gate learns ONLY 69D↔74D disagreement, preserving unanimous expert
decisions and all old four-head states. A gate may only keep/discard
already-accepted ErrorPrediction NoteOn (never invent any new attack).
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np

SCHEMA="ONSET_FP_EXPERT_GATE_CAL_SPLIT_NATIVE256_V1"
GATE_FEATURES=11

def gate_matrix(q69,q70,q74,original_correct,parent_ages):
    a,b,c,p=(np.asarray(z,np.float32) for z in
             (q69,q70,q74,original_correct))
    parents=np.asarray(parent_ages,np.float32)
    if (a.shape!=(49,) or b.shape!=(49,) or c.shape!=(49,)
        or p.shape!=(49,) or parents.shape!=(49,4)
        or not all(np.isfinite(x).all() for x in (a,b,c,p,parents))
        or not all(np.all((0<=x)&(x<=1)) for x in (a,b,c,p,parents))):
        raise ValueError("incomplete current PCM256 prior-parent and FP expert evidence")
    pitches=(np.arange(49,dtype=np.float32)/48.).reshape(49,1)
    x=np.column_stack((a,b,c,p,c-a,b-a,parents,pitches)).astype(np.float32)
    if x.shape!=(49,GATE_FEATURES):
        raise AssertionError("expert gate feature dimension changed")
    return np.ascontiguousarray(x)

def training_events(X74,q69,q70,q74,pitches):
    m=np.asarray(X74,np.float32)
    a,b,c=(np.asarray(z,np.float32) for z in (q69,q70,q74))
    t=np.asarray(pitches,dtype=np.int64)
    n=len(m)
    if (m.shape!=(n,74) or any(q.shape!=(n,) for q in (a,b,c))
        or t.shape!=(n,) or not np.all((40<=t)&(t<=88))
        or not np.isfinite(m).all()
        or not all(np.isfinite(q).all() and np.all((0<=q)&(q<=1))
                    for q in (a,b,c))):
        raise ValueError("invalid frozen PCM256 expert gate CAL records")
    cols=np.column_stack((a,b,c,m[:,73],c-a,b-a,
                          m[:,69:73],(t-40)/48.)).astype(np.float32)
    if cols.shape!=(n,GATE_FEATURES):
        raise AssertionError("CAL gate ABI")
    return cols

class FPExpertGateModel:
    trained_hop_samples=256
    name="fp_expert_gate"
    def __init__(self,network,*,enabled,cut,cut69,cut74):
        if not 0<=cut<=1 or not 0<=cut69<=1 or not 0<=cut74<=1:
            raise ValueError("FP expert CAL cuts invalid")
        if enabled and getattr(network,"n_features_in_",None)!=GATE_FEATURES:
            raise ValueError("FP neural gate has no trained 11D weights")
        self.model=network
        self.enabled=bool(enabled)
        self.cut=float(cut)
        self.cut69=float(cut69)
        self.cut74=float(cut74)
        self.calls=0
    def predict(self,q69,q70,q74,original_correct,parents):
        x=gate_matrix(q69,q70,q74,original_correct,parents)
        self.calls+=1
        if not self.enabled:
            return np.ones(49,np.float32) # ignored: return control69 exactly
        q=np.asarray(self.model.predict_proba(x)[:,1],np.float32)
        if q.shape!=(49,) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
            raise ValueError("learned 11D expert gate invalid")
        return q

    def accepted_mask(self,q69,q74,gate_probability):
        a,b,g=(np.asarray(z,np.float32) for z in
               (q69,q74,gate_probability))
        if (a.shape!=(49,) or b.shape!=(49,) or g.shape!=(49,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(g).all()):
            raise ValueError("gate select must occur on 49 pre-decision predictions")
        accepted69=a>=self.cut69
        if not self.enabled:return accepted69
        accepted74=b>=self.cut74
        disagreement=accepted69!=accepted74
        return np.where(disagreement,g>=self.cut,accepted69)

def select_gate_onsets(accepted_original,selection):
    mask=np.asarray(selection,bool)
    if mask.shape!=(49,):
        raise ValueError("wrong Onset native pitch subset shape")
    result=tuple(e for e in accepted_original if mask[int(e.pitch)-40])
    if not {(e.pitch,e.sample) for e in result}.issubset(
           {(e.pitch,e.sample) for e in accepted_original}):
        raise AssertionError("expert gate invented a new NoteOn")
    return result

def load_gate(checkpoint,metadata_path,expected_sha256):
    import joblib
    m=json.loads(Path(metadata_path).read_text())
    if (m.get("schema")!=SCHEMA or m.get("trained_hop_samples")!=256
        or m.get("gate_input_dim")!=GATE_FEATURES
        or m.get("trained_only_on_CAL_train") is not True
        or m.get("threshold_selected_only_on_CAL_validation") is not True
        or m.get("never_tuned_on_OUTER") is not True
        or not set(m["gate_train_tracks"]).isdisjoint(m["gate_validation_tracks"])
        or m.get("checkpoint_sha256")!=expected_sha256):
        raise ValueError("FN gate calibration or split trust contract invalid")
    if m["enabled"]:
        checkpoint=Path(checkpoint)
        sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha!=expected_sha256:
            raise ValueError("FP learned gate checkpoint SHA256 drift")
        net=joblib.load(checkpoint)
    else:
        # A genuinely TRAINED but scientifically disabled gate remains archived
        # with its verified checksum; do not deserialize or deploy its weights.
        net=None
        if expected_sha256 is not None:
            checkpoint=Path(checkpoint)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected_sha256:
                raise ValueError("disabled gate checkpoint SHA256 drift")
    return FPExpertGateModel(
        net,enabled=m["enabled"],cut=m["calibrated_cut"],
        cut69=m["control69_cut"],cut74=m["temporal74_cut"])
