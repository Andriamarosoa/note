"""Learned event correctness at the current PCM256 boundary, without hold."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np

from .acoustics import HOP

SCHEMA = "ONSET_EVENT_CONFIDENCE_HGB69_PCM256_V1"


class EventConfidenceHead:
    """Own trained 69D weights; a new opinion on each available audio hop."""
    trained_hop_samples = 256

    def __init__(self, model, *, cut, selected_model):
        if getattr(model, "n_features_in_", None) != 69 or not 0 <= cut <= 1:
            raise ValueError("incompatible event-confidence model")
        self.model = model
        self.cut = float(cut)
        self.selected_model = selected_model
        self.last_sample = 0

    def predict(self, evidence, *, sample):
        x = np.asarray(evidence, np.float32)
        if (sample != self.last_sample + HOP or x.shape != (49, 69)
                or not np.isfinite(x).all()):
            raise ValueError("one current 69D observation per 256 samples required")
        q = np.asarray(self.model.predict_proba(x), np.float32)
        if (q.shape != (49, 2) or not np.isfinite(q).all()
                or np.any((q < 0) | (q > 1))):
            raise ValueError("invalid learned confidence")
        self.last_sample = sample
        return q[:, 1].copy()

    def select(self, events, confidence, *, sample):
        if sample != self.last_sample or any(e.sample != sample for e in events):
            raise ValueError("selection must use this hop, never future or backdated events")
        return tuple(e for e in events if confidence[e.pitch - 40] >= self.cut)


def load_event_head(checkpoint, metadata, *, outer):
    path = Path(checkpoint)
    meta = json.loads(Path(metadata).read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if (meta.get("schema") != SCHEMA or meta.get("checkpoint_sha256") != sha
            or meta.get("trained_hop_samples") != 256
            or meta.get("outer") != outer or outer in meta.get("train", ())
            or meta.get("cal") == outer or not meta.get("frozen_before_outer")):
        raise ValueError("checkpoint provenance or fold mismatch")
    return EventConfidenceHead(joblib.load(path), cut=meta["cut"],
                               selected_model=meta["selected_model"])
