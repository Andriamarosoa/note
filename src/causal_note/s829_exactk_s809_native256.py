"""S829: train and replay an independent Exact-K readout on *fresh PCM256*.

S809 architecture: this domain owns its own head, CAL-selected selector and
count decoder. S828 acoustic clock: no 1024-as-of replay, no future/hold.
The frozen S708 encoder is REUSED (not independently trained); only the
Exact-K readout owns trainable weights. This is SHADOW, never public MIDI.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from causal_note.strict_live import HOP
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s712_s708_live import latest_s708_acoustic_features
from causal_note.s827_s708_activek_only import S827S708ActiveKOnly
from causal_note.s828_activek_s708_private_readout import (
    MODEL_ARGS, MODES, SHIFTS, summarize_one, validate, stratified_train, proposed
)

SCHEMA = "S829_S809_EXACTK_PRIVATE_TRAINED_S708_PCM256_V1"


@dataclass(frozen=True)
class ExactKDecision:
    decision_sample: int
    count: int
    confidence: float


def score_exact(pred, truth):
    """All-frame Exact-K K0..K6, including silence. Never GT-active filtering."""
    a = np.asarray(pred)
    b = np.asarray(truth)
    if a.ndim != 1 or a.shape != b.shape or not np.isin(a, range(7)).all() or not np.isin(b, range(7)).all():
        raise ValueError("Exact-K requires aligned K0..K6 predictions and truth")
    correct = int(np.count_nonzero(a == b))
    under = int(np.count_nonzero(a < b))
    over = int(np.count_nonzero(a > b))
    if correct + under + over != len(b):
        raise AssertionError("Exact-K denominator changed")
    return dict(correct=correct, frames=len(b), undercount=under,
                overcount=over, accuracy=correct / max(1, len(b)))


class S829Stream(S827S708ActiveKOnly):
    """Fresh original S708 neural K output every 256, NO Active-K decoder input.

    The legacy parent is reused for its verified causal PCM frontend and clock;
    its Active-K decoder is never stepped and its state never enters Exact-K.
    """
    def __init__(self, native_heads, original_s708_model, frozen_policy):
        super().__init__(native_heads, original_s708_model, frozen_policy)
        self.exact_history = deque([0, 0, 0, 0], maxlen=4)
        self.exact_rows = []
        self.exact_baselines = []

    def _step(self, end, f):
        p, k = self.native_heads.predict(f)
        self.native_feat.append(np.asarray(f, np.float32))
        self.native_post.append(np.asarray(p, np.float32))
        self.native_k.append(np.asarray(k, np.float32))
        acoustics = latest_s708_acoustic_features(
            self.native_feat, self.native_post, self.native_k)
        n = len(self.audio)
        sharp_ends = (np.array([n], np.int64) if end == HOP
                      else np.array([n-HOP, n], np.int64))
        spectral = sharp_track(self.audio, sharp_ends)[-1]
        X = np.concatenate((acoustics, spectral), axis=1)
        if X.shape != (49, 288) or not np.isfinite(X).all():
            raise AssertionError("fresh S708 input must be finite [49,288]")
        _post, nn_k = self.temporal.step(X)
        self.model_calls += 1
        if np.asarray(nn_k).shape != (7,) or not np.isfinite(nn_k).all():
            raise AssertionError("original S708 K0..K6 posterior ABI changed")
        base = int(np.argmax(nn_k))
        conf = float(np.max(nn_k))
        self.exact_rows.append(summarize_one(
            X, self.exact_history, base, base, conf))
        self.exact_baselines.append(base)
        self.exact_history.append(base)
        return ExactKDecision(int(end), base, conf)

    def collected(self):
        x = np.stack(self.exact_rows)
        b = np.asarray(self.exact_baselines, np.uint8)
        if x.shape[0] != self.model_calls or len(b) != self.model_calls:
            raise AssertionError("one Exact-K inference and feature per PCM256")
        return x, b

    def diagnostics(self):
        return dict(schema=SCHEMA, decision_hop_samples=HOP,
                    acoustic_feature_hop_samples=HOP,
                    model_calls=self.model_calls,
                    own_exactk_history=True,
                    other_axis_decisions_consumed=False,
                    shared_frozen_s708_encoder=True,
                    own_trained_exactk_readout=True,
                    public_midi_modified=False,
                    added_future_samples=0, added_hold_samples=0)


@dataclass
class FittedExactK:
    model: HistGradientBoostingClassifier
    mode: str
    shift: int
    metadata: dict

    def predict(self, x, baseline):
        x = np.asarray(x, np.float32)
        b = np.asarray(baseline)
        if x.ndim != 2 or b.shape != (len(x),):
            raise ValueError("Exact-K native256 rows and baseline must align")
        with threadpool_limits(limits=1):
            part = self.model.predict_proba(x)
        p = np.zeros((len(x), 7), np.float32)
        p[:, np.asarray(self.model.classes_, dtype=int)] = part
        result = proposed(p, b, self.mode, self.shift)
        result.setflags(write=False)
        return result


def fit_exact(train, cal, *, seed=829):
    """One fold: TRAIN model, independent inner-CAL selector, no outer GT."""
    tr, ca = validate(train, "TRAIN"), validate(cal, "CAL")
    tc = {r["composition"] for r in tr}
    cc = {r["composition"] for r in ca}
    if tc & cc or {r["name"] for r in tr} & {r["name"] for r in ca}:
        raise ValueError("TRAIN/CAL must be composition-disjoint")
    x, y, w = stratified_train(tr, seed=seed)
    if len(np.unique(y)) < 2:
        raise ValueError("Exact-K private learning needs >=2 classes")
    model = HistGradientBoostingClassifier(**MODEL_ARGS, random_state=seed)
    with threadpool_limits(limits=1):
        model.fit(x, y, sample_weight=w)
    scores = {(mode, shift): [0, 0, 0] for mode in MODES for shift in SHIFTS}
    for r in ca:
        with threadpool_limits(limits=1):
            part = model.predict_proba(r["X"])
        p = np.zeros((len(part), 7), np.float32)
        p[:, np.asarray(model.classes_, dtype=int)] = part
        for key, accum in scores.items():
            s = score_exact(proposed(p, r["baseline"], *key), r["k"])
            accum[0] += s["correct"]
            accum[1] += s["frames"]
            accum[2] += s["overcount"]
    best = max(scores, key=lambda key: (
        scores[key][0] / max(1, scores[key][1]),
        -scores[key][2] / max(1, scores[key][1]),
        -MODES.index(key[0]), -SHIFTS.index(key[1])))
    proof = dict(schema=SCHEMA, axis="exact_k",
        source="S809 isolated head selector decoder / S828 PCM256",
        train_compositions=sorted(tc), calibration=sorted(cc),
        training_hop_samples=256, decision_hop_samples=256,
        shared_original_s708_weights_frozen=True,
        independent_exactk_readout=True, other_five_axes_modified=False,
        other_axis_decisions_used_as_features=False,
        no_future_or_hold=True, no_public_midi=True,
        selector_cal_scores={str(k): v for k, v in scores.items()})
    return FittedExactK(model, best[0], best[1], proof)
