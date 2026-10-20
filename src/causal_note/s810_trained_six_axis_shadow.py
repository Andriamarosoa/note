"""S810 FIT80: load six separately fitted OWNED classifiers, no shared decoder.

Read-only SHADOW bridge to S809. Requires six verified checkpoints for the
single held-out composition, produced by s810_six_independent_fit80.py
(the downloadable research package includes that full training script).
No output is a released MIDI event. All axes have separate weights, scaler,
feature columns, task, selector and result; no axis sees another's outputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from causal_note.s763_independent_decoders import AXES

SCHEMA = "S810_REAL_FIT80_SIX_TRAINED_INDEPENDENT_AXES_V1"
FEATURE_COLUMNS = {
    "onset": (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,22),
    "offset": (0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22),
    "frame_pitch": (0,1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,22),
    "full_note": (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22),
}

def count_features(x):
    """Same current-frame aggregation as six-head independent FIT trainer."""
    x = np.asarray(x, np.float32)
    if x.ndim != 3 or x.shape[1:] != (49,23):
        raise ValueError("S810 original 49x23 causal input required")
    result = np.concatenate((
        x.mean(axis=1), x.max(axis=1),
        np.sort(x[:,:,1], axis=1)[:,-6:],
        np.sort(x[:,:,0], axis=1)[:,-5:],
        x[:,:,12].sum(axis=1)[:,None]/6.,
        np.stack((x[:,:8,1].max(axis=1),
                  x[:,8:29,1].max(axis=1),
                  x[:,29:,1].max(axis=1)), axis=1)), axis=1)
    if not np.isfinite(result).all(): raise ValueError("nonfinite source")
    return result

@dataclass(frozen=True)
class TrainedShadowDecision:
    axis: str
    values: np.ndarray
    selector: str
    weights_digest: str

class TrainedAxis:
    """One task-specific model + selector. Never references peer objects."""

    def __init__(self, axis, model, weights_digest):
        if axis not in AXES: raise ValueError("unknown axis")
        self.axis = axis
        self.model = model
        self.weights_digest = weights_digest
        self.selector = "s810_" + axis + "_private_fitted_selector"

    def decide(self, x):
        """Private, causal per-frame output; no MIDI merger or label access."""
        if self.axis in ("exact_k","active_exact_k"):
            inp = count_features(x)
            values = self.model.predict(inp).astype(np.uint8)
        else:
            cols = FEATURE_COLUMNS[self.axis]
            inp = x[:,:,cols].reshape(-1,len(cols))
            if self.axis == "full_note":
                # 0=inactive,1=attack,2=sustain,3=release; own note identity.
                values = self.model.predict(inp).reshape(x.shape[:2]).astype(np.uint8)
            else:
                probabilities = self.model.predict_proba(inp)
                positive = list(self.model.classes_).index(1)
                values = (probabilities[:,positive] >= .50).reshape(x.shape[:2]).astype(np.uint8)
        values.setflags(write=False)
        return TrainedShadowDecision(self.axis, values, self.selector,
                                     self.weights_digest)

class TrainedSixAxisBank:
    """Six independent heads; all separate weights, private selection outputs."""

    def __init__(self, report, checkpoints, fold):
        import joblib
        data = json.loads(Path(report).read_text())
        if data.get("schema") != SCHEMA or not data.get("complete") or data.get("promotion"):
            raise ValueError("incomplete or promoted research checkpoint")
        if fold not in data["compositions"] or data["original_S708_midi_events_changed"]:
            raise ValueError("invalid FIT heldout group or altered official MIDI")
        training = data["training_provenance"][fold]
        archive = data["model_artifacts"]
        owned = {}
        for axis in AXES:
            proof = training[axis]
            if fold in proof["training_compositions"] or len(proof["training_compositions"]) != 4:
                raise ValueError("training target composition leaked")
            key = fold + "/" + axis
            entry = archive[key]
            filename = entry["file"]
            if filename != fold + "_" + axis + ".joblib" or "/" in filename or ".." in filename:
                raise ValueError("unsafe model filename")
            path = Path(checkpoints)/filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("checkpoint SHA mismatch")
            model = joblib.load(path)   # ONLY locally generated, trusted FIT checkpoints
            if self_feature_dimension(axis) != proof["feature_dimension"]:
                raise ValueError("model feature ABI mismatch")
            if model[-1].coef_.shape[1] != proof["feature_dimension"]:
                raise ValueError("wrong checkpoint shape")
            own_sha = hashlib.sha256(
                np.asarray(model[-1].coef_, dtype=np.float64).tobytes()).hexdigest()
            if own_sha != proof["model_sha256"]:
                raise ValueError("model weights changed")
            owned[axis] = TrainedAxis(axis,model,own_sha)
        if len({id(v.model) for v in owned.values()}) != 6 or (
            len({v.weights_digest for v in owned.values()}) != 6):
            raise AssertionError("shared model checkpoint between targets")
        self.axes = owned
        self.fold = fold

    def infer(self, x):
        x = np.asarray(x,np.float32)
        if x.ndim != 3 or x.shape[1:] != (49,23) or not np.isfinite(x).all():
            raise ValueError("invalid current/past acoustic evidence")
        # One independent input copy for each axis, no output fed into another.
        return {axis:head.decide(x.copy()) for axis,head in self.axes.items()}

def self_feature_dimension(axis):
    return 61 if axis in ("exact_k","active_exact_k") else len(FEATURE_COLUMNS[axis])
