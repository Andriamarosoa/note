"""S829: fully isolated PCM -> DSP -> two trained Exact-K heads -> selector.

No S708/S810/S828 neural checkpoint, posteriors, emitted MIDI or decoder
state. Every model weight and mutable state is owned by Exact-K. Only
the read-only PCM signal and fixed NumPy mathematical primitives are shared.
This is a research SHADOW pipeline, not a deployed MIDI decoder.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

SR, HOP, FIRST, PITCHES = 44100, 256, 40, 49
WINDOWS = (2048, 4096, 8192)
SCHEMA = "S829_INDEPENDENT_PCM256_EXACTK_TWO_FITTED_HEADS_V2"
MODEL_ARGS = dict(max_iter=115, max_depth=5, max_leaf_nodes=25,
                  min_samples_leaf=120, learning_rate=.075,
                  l2_regularization=2.0, max_bins=63, early_stopping=False)
MODES = ("harmonic", "temporal", "fusion", "confidence_router", "acoustic")
SHIFTS = (0, -1, 1)
# 24 harmonic levels per pitch, 24 normalizations, 24 onset-differences;
# 120 aggregate statistics per family, plus 9 acoustic controls.
HARMONIC_COLS = np.r_[0:240, 360:369]
TEMPORAL_COLS = np.r_[0:120, 240:360, 360:369]


@dataclass(frozen=True)
class ExactKDecision:
    decision_sample: int
    count: int
    confidence: float


def score_exact(pred, truth):
    a = np.asarray(pred)
    b = np.asarray(truth)
    if (a.ndim != 1 or a.shape != b.shape or
            not np.isin(a, range(7)).all() or not np.isin(b, range(7)).all()):
        raise ValueError("Exact-K requires aligned K0..K6 truth and predictions")
    correct = int(np.count_nonzero(a == b))
    under = int(np.count_nonzero(a < b))
    over = int(np.count_nonzero(a > b))
    if correct + under + over != len(b):
        raise AssertionError("exact K0..K6 denominator drift")
    return dict(correct=correct, frames=int(len(b)), undercount=under,
                overcount=over, accuracy=correct / max(1, len(b)))


class S829AcousticFrontEnd:
    """Exact-K-owned right-aligned multiresolution PCM spectrum and history."""

    def __init__(self):
        self.received = 0
        self.next_end = HOP
        self.tail = np.empty(0, np.float32)
        self.previous = np.zeros((PITCHES, 24), np.float32)
        self.windows = [np.hanning(n).astype(np.float32) for n in WINDOWS]
        hz = 440. * 2. ** ((np.arange(FIRST, FIRST+PITCHES)-69)/12.)
        self.bin_indices = []
        for n in WINDOWS:
            center = np.rint(hz[:, None] *
                             np.arange(1, 9)[None, :] * n / SR).astype(int)
            self.bin_indices.append(
                np.clip(center[..., None] + np.arange(-1, 2), 1, n//2))

    def _frame(self, signal):
        levels = []
        for n, taper, bins in zip(WINDOWS, self.windows, self.bin_indices):
            wave = signal[-n:] * taper
            fft = np.abs(np.fft.rfft(wave)) * (2. / taper.sum())
            levels.append(np.log1p(600. * fft[bins].max(axis=-1)))
        h = np.stack(levels, 1).reshape(PITCHES, 24).astype(np.float32)
        relative = h / (.10 + h.max(axis=0, keepdims=True))
        delta = h - self.previous
        self.previous = h.copy()
        features = np.concatenate((h, relative, delta), axis=1)
        # Strictly private physical, positive harmonic evidence. Never NN K.
        f0 = h[:, [0, 8, 16]].mean(axis=1)
        threshold = max(.30, .65 * float(np.max(f0)))
        acoustic_k = min(6, int(np.count_nonzero(f0 >= threshold)))
        return features, acoustic_k

    def push(self, samples, *, start_sample):
        x = np.asarray(samples)
        if (type(start_sample) is not int or start_sample != self.received or
                x.ndim != 1 or x.dtype.kind != "f" or
                len(x) > 4096 or not np.isfinite(x).all()):
            raise ValueError("finite contiguous mono PCM required, max 4096")
        joined = np.concatenate((self.tail, x.astype(np.float32, copy=False)))
        origin = self.received - len(self.tail)
        end = self.received + len(x)
        results = []
        while self.next_end <= end:
            right = self.next_end - origin
            frame = joined[max(0, right-WINDOWS[-1]):right]
            if len(frame) < WINDOWS[-1]:
                frame = np.pad(frame, (WINDOWS[-1]-len(frame), 0))
            evidence, guess = self._frame(frame)
            results.append((self.next_end, evidence, guess))
            self.next_end += HOP
        self.tail = joined[-WINDOWS[-1]:].copy()
        self.received = end
        return results


def summarize_one(x, acoustic_k):
    """369 scalars made ONLY from current/past PCM, same online/fit ABI."""
    f = np.asarray(x, np.float32)
    if f.shape != (PITCHES, 72) or not np.isfinite(f).all():
        raise ValueError("S829 private [49,72] acoustic ABI required")
    families = []
    for start in (0, 24, 48):
        part = f[:, start:start+24]
        families.append(np.concatenate((
            part.mean(0), part.std(0), part.max(0),
            np.sort(part, axis=0)[-3:].mean(0),
            np.sort(part, axis=0)[-8:].mean(0))))
    f0 = f[:, [0, 8, 16]].mean(axis=1)
    controls = np.array([
        float(acoustic_k), float(f0.mean()), float(f0.std()),
        float(f0.max()), float(np.count_nonzero(f0 > .3)),
        float(np.count_nonzero(f0 > .6)),
        float(np.count_nonzero(f0 > 1.)),
        float(np.max(f[:, :24])),
        float(np.mean(np.maximum(f[:, 48:72], 0)))
    ], np.float32)
    return np.concatenate((*families, controls)).astype(np.float32)


class S829Stream:
    """Private audio input, DSP history and own-only Exact-K feature memory."""

    def __init__(self):
        self.frontend = S829AcousticFrontEnd()
        self.rows = []
        self.baselines = []
        self.model_calls = 0
        self.last_sample = 0
        self.closed = False

    def push(self, samples, *, start_sample):
        if self.closed:
            raise ValueError("stream is closed")
        result = []
        for end, feature, raw_k in self.frontend.push(
                samples, start_sample=start_sample):
            if end != self.last_sample + HOP:
                raise AssertionError("native PCM256 causal clock mismatch")
            self.rows.append(summarize_one(feature, raw_k))
            self.baselines.append(raw_k)
            result.append(ExactKDecision(end, raw_k, 1.))
            self.last_sample = end
            self.model_calls += 1
        return tuple(result)

    def collected(self):
        x = np.stack(self.rows)
        k = np.asarray(self.baselines, np.uint8)
        if len(x) != self.model_calls or len(k) != self.model_calls:
            raise AssertionError("one fresh acoustic row per decision 256")
        return x, k

    def finish(self):
        self.closed = True
        return ()

    def diagnostics(self):
        return dict(schema=SCHEMA, decision_hop_samples=HOP,
                    acoustic_feature_hop_samples=HOP, model_calls=self.model_calls,
                    shared_model_weights=False, shared_axis_state=False,
                    source_neural_encoder_used=False,
                    other_axis_posteriors_used=False, public_midi_modified=False,
                    added_future_samples=0, added_hold_samples=0)


def validate(rows, kind):
    rows = sorted(rows, key=lambda r: r["name"])
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError(kind + " records must be distinct")
    for r in rows:
        x = np.asarray(r["X"])
        y = np.asarray(r["k"])
        b = np.asarray(r["baseline"])
        if (x.ndim != 2 or x.shape[1] != 369 or len(x) == 0 or
                len(x) != len(y) or len(y) != len(b) or
                not np.isfinite(x).all() or
                not np.isin(y, range(7)).all() or
                not np.isin(b, range(7)).all() or
                not r.get("composition")):
            raise ValueError("invalid independent PCM256 K0..K6 row")
    return rows


def stratified_train(rows, *, seed=829, cap=1750):
    xs, ys, weights = [], [], []
    if cap < 70:
        raise ValueError("cap must admit seven classes")
    for r in validate(rows, "TRAIN"):
        y = np.asarray(r["k"])
        stable = int.from_bytes(hashlib.sha256(r["name"].encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed+stable)
        picks, w = [], []
        for k in range(7):
            ids = np.flatnonzero(y == k)
            if len(ids):
                n = min(len(ids), cap//7)
                sel = np.sort(rng.choice(ids, size=n, replace=False))
                picks.append(sel)
                w.append(np.full(n, len(ids)/n, np.float64))
        ix = np.concatenate(picks)
        wt = np.concatenate(w)
        order = np.argsort(ix)
        ix, wt = ix[order], wt[order]
        if not np.allclose(np.bincount(y[ix], weights=wt, minlength=7),
                           np.bincount(y, minlength=7), atol=1e-8, rtol=0):
            raise AssertionError("natural class priors not restored")
        xs.append(np.asarray(r["X"][ix], np.float32))
        ys.append(np.asarray(y[ix], np.uint8))
        weights.append(wt)
    w = np.concatenate(weights)
    w /= w.mean()
    return np.concatenate(xs), np.concatenate(ys), w


def probabilities(model, features):
    with threadpool_limits(limits=1):
        part = model.predict_proba(features)
    p = np.zeros((len(part), 7), np.float32)
    p[:, np.asarray(model.classes_, dtype=int)] = part
    return p


def proposed(harmonic, temporal, acoustic, mode, shift):
    if harmonic.shape != temporal.shape or harmonic.shape[1] != 7:
        raise ValueError("two Exact-K private head posteriors required")
    if shift not in SHIFTS or mode not in MODES:
        raise ValueError("invalid private selector")
    if mode == "harmonic":
        distribution = harmonic
    elif mode == "temporal":
        distribution = temporal
    elif mode == "fusion":
        distribution = (harmonic+temporal)*.5
    elif mode == "confidence_router":
        distribution = np.where(
            harmonic.max(1)[:, None] >= temporal.max(1)[:, None],
            harmonic, temporal)
    else:
        distribution = None
    k = (np.asarray(acoustic, np.int16) if distribution is None
         else np.argmax(distribution, axis=1).astype(np.int16))
    return np.clip(k+shift, 0, 6).astype(np.uint8)


@dataclass
class FittedExactK:
    harmonic: HistGradientBoostingClassifier
    temporal: HistGradientBoostingClassifier
    mode: str
    shift: int
    metadata: dict

    def predict(self, x, baseline):
        x = np.asarray(x, np.float32)
        b = np.asarray(baseline)
        if x.ndim != 2 or x.shape[1] != 369 or b.shape != (len(x),):
            raise ValueError("private count must use aligned native256 rows")
        h = probabilities(self.harmonic, x[:, HARMONIC_COLS])
        t = probabilities(self.temporal, x[:, TEMPORAL_COLS])
        out = proposed(h, t, b, self.mode, self.shift)
        out.setflags(write=False)
        return out


def fit_exact(train, cal, *, seed=829):
    tr, ca = validate(train, "TRAIN"), validate(cal, "CAL")
    tc = {r["composition"] for r in tr}
    cc = {r["composition"] for r in ca}
    if tc & cc or {r["name"] for r in tr} & {r["name"] for r in ca}:
        raise ValueError("TRAIN/CAL must be composition-disjoint")
    x, y, w = stratified_train(tr, seed=seed)
    if len(np.unique(y)) < 2:
        raise ValueError("at least two K classes required")
    h = HistGradientBoostingClassifier(**MODEL_ARGS, random_state=seed)
    t = HistGradientBoostingClassifier(**MODEL_ARGS, random_state=seed+100)
    with threadpool_limits(limits=1):
        h.fit(x[:, HARMONIC_COLS], y, sample_weight=w)
        t.fit(x[:, TEMPORAL_COLS], y, sample_weight=w)
    if h is t:
        raise AssertionError("two fitted heads must be separate models")
    scores = {(mode, shift): [0, 0, 0] for mode in MODES for shift in SHIFTS}
    for r in ca:
        hh = probabilities(h, r["X"][:, HARMONIC_COLS])
        tt = probabilities(t, r["X"][:, TEMPORAL_COLS])
        for key, agg in scores.items():
            sc = score_exact(proposed(hh, tt, r["baseline"], *key), r["k"])
            agg[0] += sc["correct"]
            agg[1] += sc["frames"]
            agg[2] += sc["overcount"]
    best = max(scores, key=lambda key: (
        scores[key][0]/max(1, scores[key][1]),
        -scores[key][2]/max(1, scores[key][1]),
        -MODES.index(key[0]), -SHIFTS.index(key[1])))
    proof = dict(schema=SCHEMA, axis="exact_k", train_compositions=sorted(tc),
                 calibration=sorted(cc), training_hop_samples=HOP,
                 decision_hop_samples=HOP, independent_pcm_frontend=True,
                 independent_trained_heads=2, private_selector=True,
                 shared_neural_checkpoint=False, any_other_axis_as_input=False,
                 no_future_or_hold=True, no_public_midi=True,
                 selector_cal_scores={str(k): v for k, v in scores.items()})
    return FittedExactK(h, t, best[0], best[1], proof)
