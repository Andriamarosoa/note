"""Fourth independently TRAINED Onset head: FN-recovery candidates, native PCM256.

This head recovers plausible missing ATTACKS; the ErrorPredictionHead is a
different specialist and only challenges already proposed attacks. FNRecovery
learns supervised P(originally-missed-GT | current-and-past-audio) on proposals.
The three permissive candidate sources have frozen, independently causal
state, without ever mutating S613/S701/error risk selector state.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
from .error_evidence69 import hard_matrix,DIM
from .acoustics import HOP,SR

SCHEMA="ONSET_FN_RECOVERY_69D_SUPERVISED_NATIVE256_V1"
CANDIDATE_POLICIES={
    "sensitive": dict(onset=.06,active_on=.10,onset_reset=.025),
    "moderate":dict(onset=.12,active_on=.20,onset_reset=.05),
    "cautious":dict(onset=.20,active_on=.30,onset_reset=.08),
}
EVAL_DEDUP_SAMPLES=round(.080*SR/HOP)*HOP

class FNRecoveryModel:
    """An independently fitted binary 69D classifier (historical MLP or HGB)."""
    trained_hop_samples=256
    name="fn_recovery"
    def __init__(self,estimator,*,candidate_policy,calibrated_cut,enabled=True):
        if (getattr(estimator,"n_features_in_",None)!=DIM or
            candidate_policy not in CANDIDATE_POLICIES or
            not 0 <= calibrated_cut <= 1):
            raise ValueError("No genuine learned PCM256 FN-rescue checkpoint")
        self.model=estimator
        self.candidate_policy=candidate_policy
        self.calibrated_cut=float(calibrated_cut)
        self.enabled=bool(enabled)
        self.calls=0
    def predict_recovery_probability(self,feat,p,k,rescue,history,selector_state):
        x=hard_matrix(feat,p,k,rescue,history,selector_state)
        q=np.asarray(self.model.predict_proba(x),np.float32)
        self.calls+=1
        if (q.shape!=(49,2) or not np.isfinite(q).all() or
            np.any(q<0) or np.any(q>1)):
            raise ValueError("FNRecovery learned probabilities must be valid 49x2")
        return np.ascontiguousarray(q[:,1])

def load_fn_recovery(checkpoint,metadata_path,*,expected_sha256):
    """Load only SHA-pinned native256 weights, never S708 trained1024."""
    import joblib
    checkpoint=Path(checkpoint)
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha!=expected_sha256:
        raise ValueError("Untrusted FNRecovery checkpoint SHA256")
    meta=json.loads(Path(metadata_path).read_text())
    if (meta.get("schema")!=SCHEMA or meta.get("checkpoint_sha256")!=sha
        or meta.get("trained_hop_samples")!=256 or meta.get("dim")!=69
        or meta.get("frozen_before_outer") is not True
        or meta.get("calibrated_only_on_cal") is not True):
        raise ValueError("FNRecovery training provenance invalid")
    return FNRecoveryModel(
        joblib.load(checkpoint),candidate_policy=meta["candidate_policy"],
        calibrated_cut=float(meta["calibrated_cut"]),
        enabled=bool(meta["enabled"]))

class FNRecoverySelector:
    """Purely causal Onset-only merge. No hold, future frames, or GT."""
    def __init__(self,cut,enabled=True):
        self.cut=float(cut)
        self.enabled=bool(enabled)
        self.last_main=np.full(49,-10**12,np.int64)
        self.last_added=np.full(49,-10**12,np.int64)
        self.calls=0

    def select(self,*,end_sample,main_raw,main_accepted,proposals,recovery_probability):
        if end_sample!=(self.calls+1)*HOP:
            raise AssertionError("FN recovery must decide every fresh 256 PCM")
        self.calls+=1
        confidence=np.asarray(recovery_probability,np.float32)
        if confidence.shape!=(49,) or not np.isfinite(confidence).all():
            raise ValueError("FNRecovery head missing 49 causal probabilities")
        mains={int(e.pitch) for e in main_raw}
        accepted={int(e.pitch) for e in main_accepted}
        for pitch in mains:
            self.last_main[pitch-40]=end_sample
        result=[]
        seen=set()
        for e in proposals:
            # CausalSelection also emits NoteOff, including immediately
            # before a same-hop restrike. Those are not recovery proposals:
            # they must not consume `seen` or advance the NoteOn cooldown.
            # Archived candidate-only records omit type and mean NoteOn.
            if e.get("type", "note_on") != "note_on":
                continue
            pitch=int(e["pitch"])
            j=pitch-40
            if not 0<=j<49 or int(e["decision_sample"])!=end_sample:
                raise ValueError("Noncausal FN rescue proposal")
            if pitch in seen:continue
            seen.add(pitch)
            if (not self.enabled or confidence[j]<self.cut or pitch in mains or
                pitch in accepted or
                end_sample-self.last_main[j]<=EVAL_DEDUP_SAMPLES or
                end_sample-self.last_added[j]<=EVAL_DEDUP_SAMPLES):
                continue
            self.last_added[j]=end_sample
            result.append(e)
        if any(int(e["pitch"]) in mains for e in result):
            raise AssertionError("FNRecovery duplicated MAIN NoteOn")
        return tuple(result)
