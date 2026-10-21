"""S814 independent Active Exact-K teacher: acoustic MIDI256, two private heads.

No S708 posteriors, historical MIDI, labels, frame-pitch teacher, or neighbor
axis output enters inference. The 222 input features are extracted from past
and present PCM every 256 samples by the frozen S813 extractor. Counts are
K0..K6 and are never obtained by changing already emitted MIDI.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SCHEMA = "S814_ACTIVE_K_PRIVATE_DIRECT_AND_OCCUPANCY256_V1"
PITCHES, DIM = 49, 222
# Acoustic indices only; both heads own their weights and training data.
FEATURES = (0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26,
            32, 33, 34, 126, 134, 142, 150, 158, 166, 174)
SELECTORS = ("direct", "regression", "confidence_router")
SHIFTS = (0, 1, -1)
MODEL_ARGS = dict(max_iter=72, max_leaf_nodes=19, max_depth=5,
                  min_samples_leaf=55, l2_regularization=2.0,
                  learning_rate=.09, max_bins=63, early_stopping=False)


def summarize(X):
    """Current frame alone; source DSP itself only depends on past PCM."""
    x = np.asarray(X)
    if x.ndim != 3 or x.shape[1:] != (PITCHES, DIM):
        raise ValueError("fresh MIDI256 acoustic features [T,49,222] required")
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("empty/nonfinite acoustic input")
    a = np.asarray(x[:, :, FEATURES], np.float32)
    top = np.sort(a, axis=1)[:, -4:, :].reshape(len(a), -1)
    features = np.concatenate((
        top, a.mean(1), a.std(1), a[:, :8].mean(1),
        a[:, 8:29].mean(1), a[:, 29:].mean(1),
        np.count_nonzero(a > 1.0, axis=1).astype(np.float32),
    ), axis=1)
    return np.ascontiguousarray(features, dtype=np.float32)


def _validate(rows, role):
    rows = sorted(list(rows), key=lambda r: r["name"])
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError(role + " requires distinct recordings")
    for row in rows:
        x, k = np.asarray(row["X"]), np.asarray(row["k"])
        if not isinstance(row["composition"], str) or not row["composition"]:
            raise ValueError("missing composition")
        if x.ndim != 3 or x.shape[1:] != (49, 222) or k.shape != (len(x),):
            raise ValueError("native256 track and aligned K0..K6 target required")
        if not len(x) or not np.isin(k, range(7)).all():
            raise ValueError("invalid K0..K6 labels")
    return rows


def sample_frames(rows, *, cap=1400, seed=814):
    """Stratify rare K while restoring natural class weights per track."""
    if cap < 28:
        raise ValueError("at least four examples per K class budget needed")
    xs, ys, ws, audits = [], [], [], []
    for row in _validate(rows, "FIT"):
        truth = np.asarray(row["k"], dtype=np.uint8)
        h = int.from_bytes(hashlib.sha256(row["name"].encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed + h)
        chosen, weights = [], []
        for label in range(7):
            indices = np.flatnonzero(truth == label)
            n = min(len(indices), cap // 7)
            if not n:
                continue
            subset = np.sort(rng.choice(indices, size=n, replace=False))
            chosen.append(subset)
            weights.append(np.full(n, len(indices) / n, np.float64))
        if not chosen:
            raise AssertionError("empty training track")
        indices = np.concatenate(chosen)
        weight = np.concatenate(weights)
        order = np.argsort(indices)
        indices, weight = indices[order], weight[order]
        selected = np.asarray(row["X"][indices])
        xs.append(summarize(selected))
        ys.append(truth[indices])
        ws.append(weight)
        # Individual class reweighting reproduces the K distribution.
        observed = np.bincount(truth, minlength=7)
        restored = np.bincount(truth[indices], weights=weight, minlength=7)
        if not np.allclose(observed, restored, atol=1e-8, rtol=0):
            raise AssertionError("class prior changed by stratified selection")
        audits.append(dict(name=row["name"], natural=observed.tolist(),
                           selected=np.bincount(truth[indices], minlength=7).tolist()))
    X, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    # Weights normalized to preserve training-scale numerical stability.
    weight /= weight.mean()
    if len(np.unique(y)) < 2:
        raise ValueError("need at least two real K classes")
    return X, y, weight, audits


def active_score(pred, truth):
    """True S704 Active Exact-K denominator: all frames with GT count > 0."""
    p, y = np.asarray(pred), np.asarray(truth)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("misaligned K decisions")
    mask = y > 0
    return {"correct": int(np.count_nonzero((p == y) & mask)),
            "frames": int(mask.sum()),
            "undercount": int(np.count_nonzero((p < y) & mask)),
            "overcount": int(np.count_nonzero((p > y) & mask))}


def _combine(proba, regression, selector, shift):
    classes = np.argmax(proba, axis=1).astype(np.int16)
    k_reg = np.rint(regression).astype(np.int16)
    if selector == "direct":
        out = classes
    elif selector == "regression":
        out = k_reg
    elif selector == "confidence_router":
        out = np.where(proba.max(1) >= .55, classes, k_reg)
    else:
        raise ValueError("unknown independent selector")
    if shift not in SHIFTS:
        raise ValueError("undeclared calibration shift")
    return np.clip(out + shift, 0, 6).astype(np.uint8)


@dataclass
class ActiveKTeacher:
    direct: HistGradientBoostingClassifier
    regression: HistGradientBoostingRegressor
    selector: str
    shift: int
    metadata: dict

    def __post_init__(self):
        if self.direct is self.regression or self.selector not in SELECTORS or self.shift not in SHIFTS:
            raise ValueError("independent heads/declared selection required")

    @property
    def heads(self):
        return MappingProxyType({"direct": self.direct, "regression": self.regression})

    def predict(self, X):
        a = summarize(X)
        with threadpool_limits(limits=1):
            partial = self.direct.predict_proba(a)
            regression = self.regression.predict(a)
        # Some folds legitimately contain no K6. Missing class is never
        # mapped to the wrong ordinal K value.
        proba = np.zeros((len(a), 7), np.float32)
        proba[:, np.asarray(self.direct.classes_, dtype=int)] = partial
        result = _combine(proba, regression, self.selector, self.shift)
        result.setflags(write=False)
        return result


def fit_teacher(train_rows, cal_rows, *, cap=1400, seed=814):
    train = _validate(train_rows, "training")
    cal = _validate(cal_rows, "inner calibration")
    train_names, cal_names = {r["name"] for r in train}, {r["name"] for r in cal}
    train_comps = {r["composition"] for r in train}
    cal_comps = {r["composition"] for r in cal}
    if train_names & cal_names or train_comps & cal_comps:
        raise ValueError("training/calibration must be composition-disjoint")
    x, y, weights, audit = sample_frames(train, cap=cap, seed=seed)
    direct = HistGradientBoostingClassifier(**MODEL_ARGS, random_state=seed)
    regression = HistGradientBoostingRegressor(**MODEL_ARGS, random_state=seed + 1)
    with threadpool_limits(limits=1):
        direct.fit(x, y, sample_weight=weights)
        regression.fit(x, y.astype(np.float32), sample_weight=weights)
    if direct is regression:
        raise AssertionError("heads have shared fitted objects")
    choices = {(s, b): [0, 0] for s in SELECTORS for b in SHIFTS}
    for row in cal:
        z = summarize(row["X"])
        with threadpool_limits(limits=1):
            prob = direct.predict_proba(z)
            reg = regression.predict(z)
        full = np.zeros((len(z), 7), np.float32)
        full[:, np.asarray(direct.classes_, dtype=int)] = prob
        for selector, shift in choices:
            got = active_score(_combine(full, reg, selector, shift), row["k"])
            choices[selector, shift][0] += got["correct"]
            choices[selector, shift][1] += got["frames"]
    # Fixed-order tie-breaking: no outer-fold score ever selects policies.
    best = max(choices, key=lambda pair:
               (choices[pair][0] / max(choices[pair][1], 1), -SELECTORS.index(pair[0]),
                -SHIFTS.index(pair[1])))
    proof = dict(schema=SCHEMA, selector=best[0], shift=best[1],
                 train_compositions=sorted(train_comps),
                 calibration_compositions=sorted(cal_comps),
                 train_recordings=len(train), calibration_recordings=len(cal),
                 training_sampling=audit, trained_heads=["direct", "regression"],
                 calibration_results=[
                     dict(selector=s, shift=b, correct=c[0], frames=c[1])
                     for (s,b),c in choices.items()],
                 supervised_inference_inputs=False, features_per_frame=220,
                 learned_previous_axes=False, post_onset_hold_samples=0,
                 added_lookahead_samples=0)
    return ActiveKTeacher(direct, regression, best[0], best[1], proof)
