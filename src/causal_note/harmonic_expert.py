"""Experimental harmonic-parent/genuine-onset expert for NOTE S613.

Never alters the frozen S613 weights. Uses only feature rows already computed
by CausalPitchFeatures and probabilities from existing native heads.
No FFT, audio buffering, lookahead, or future samples.

SHADOW is the only mode permitted without trained weights. GUARDED can inhibit
an individual upper-pitch onset only when a *separately trained* classifier
judges the upper onset non-independent; the correct gating threshold MUST be
selected on independent development recordings. Neither mode adds NoteOn
events or assumes that an exact octave cannot be a real second note.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from causal_note.strict_live import PITCHES, HOP


SCHEMA = "NOTE_S613_harmonic_pair_genuine_onset_v1"
# Offsets of the first 7 harmonic ratios, rounded to the nearest MIDI note.
# One child pitch may have several possible lower harmonic parents.
HARMONIC_RATIOS = ((12, 2), (19, 3), (24, 4), (28, 5),
                   (31, 6), (34, 7), (36, 8))
FEATURE_NAMES = (
    "parent_first_part_8192", "parent_shared_part_8192",
    "child_first_part_8192", "child_second_part_8192",
    "parent_first_part_2048", "child_first_part_2048",
    "parent_harmonic_score_8192", "child_harmonic_score_8192",
    "parent_novelty_lag2", "child_novelty_lag2",
    "parent_onset", "parent_sustain", "child_onset", "child_sustain",
    "parent_active", "p_count_ge_2", "harmonic_index_normalized",
)
N_FEATURES = len(FEATURE_NAMES)

# Precompute pitch indices once. This preserves the existing pair order.
_PARENT_CHILD_HARMONICS = np.asarray([
    (pitch - shift - int(PITCHES[0]), pitch - int(PITCHES[0]), partial)
    for shift, partial in HARMONIC_RATIOS
    for pitch in range(int(PITCHES[0]) + shift, int(PITCHES[-1]) + 1)
], dtype=np.int16)


def harmonic_pair_features(features, probabilities, count, active, *, materialize=True):
    """Return (pairs[N,3], X[N,17]) for possible (parent, child, harmonic).

    This is fixed, modest-size and completely past-only. IMPORTANT: a parent's
    kth partial and child's fundamental frequently sample the *same* FFT bins.
    Their equality is NOT evidence of a unique parent, nor permission to
    delete the child. A genuinely new child requires a learned distinction.
    """
    x = np.asarray(features, dtype=np.float32)
    p = np.asarray(probabilities, dtype=np.float32)
    k = np.asarray(count, dtype=np.float32)
    a = np.asarray(active, dtype=bool)
    # The specialist only reads spectral indices 0..23, physical scores
    # at 50 and lag-2 novelty at 54. It must not assume/freeze the native
    # model's total column count; the underlying S613 checkpoint owns that ABI.
    if (x.ndim != 2 or x.shape[0] != 49 or x.shape[1] < 55
            or p.shape != (49, 3) or k.shape != (7,) or a.shape != (49,)):
        raise ValueError(f"Expected S613 49-pitch spectral inputs (>=55 cols), (49,3) heads and K0..6; got {x.shape}")
    if not (np.isfinite(x).all() and np.isfinite(p).all() and np.isfinite(k).all()):
        raise ValueError("Non-finite harmonic input")
    h = x[:, :24].reshape(49, 3, 8)
    topology = _PARENT_CHILD_HARMONICS
    index = topology[:, 0]
    subset = topology[h[index, 2, 0] >= .1]
    if not len(subset):
        return np.empty((0, 3), np.int16), (
            np.empty((0, N_FEATURES), np.float32) if materialize else None)
    parents, children, harmonics = subset.T
    pairs = subset.copy()
    pairs[:, :2] += int(PITCHES[0])
    if not materialize:
        return pairs, None
    rows = np.column_stack((
        h[parents, 2, 0], h[parents, 2, harmonics - 1],
        h[children, 2, 0], h[children, 2, 1],
        h[parents, 0, 0], h[children, 0, 0],
        x[parents, 50], x[children, 50],
        x[parents, 54], x[children, 54],
        p[parents, 0], p[parents, 1], p[children, 0], p[children, 1],
        a[parents].astype(np.float32),
        np.full(len(pairs), float(k[2:].sum()), dtype=np.float32),
        harmonics.astype(np.float32) / 8.0,
    ))
    return pairs, np.asarray(rows, dtype=np.float32)


class HarmonicGenuineOnsetHead:
    """Small logistic head trained on labeled onset/parent pairs.

    A missing checkpoint is NEVER replaced with a fabricated predictor.
    SHADOW with no checkpoint audits pair count only.
    """

    def __init__(self, checkpoint=None):
        self.trained = checkpoint is not None
        self.mean = self.scale = self.weights = None
        self.bias = None
        if checkpoint is not None:
            with np.load(Path(checkpoint), allow_pickle=False) as z:
                if str(z["schema"]) != SCHEMA:
                    raise ValueError("Incompatible harmonic expert schema")
                if tuple(z["feature_names"].tolist()) != FEATURE_NAMES:
                    raise ValueError("Harmonic expert feature order mismatch")
                self.mean = np.asarray(z["mean"], np.float32)
                self.scale = np.asarray(z["scale"], np.float32)
                self.weights = np.asarray(z["weights"], np.float32)
                self.bias = float(z["bias"])
            if (self.mean.shape != (N_FEATURES,) or self.scale.shape != (N_FEATURES,)
                or self.weights.shape != (N_FEATURES,)
                or not np.isfinite(self.weights).all() or not np.isfinite(self.mean).all()
                or not np.isfinite(self.scale).all() or (self.scale <= 0).any()
                or not np.isfinite(self.bias)):
                raise ValueError("Invalid harmonic expert parameters")

    def predict(self, features):
        if not self.trained:
            return None
        logits = ((features - self.mean) / self.scale) @ self.weights + self.bias
        return 1. / (1. + np.exp(-np.clip(logits, -30, 30)))


@dataclass
class HarmonicExpertStats:
    frames: int = 0
    pairs_considered: int = 0
    candidate_upper_onsets: int = 0
    low_independence_onsets: int = 0
    vetoed_upper_onsets: int = 0


class HarmonicSpecialistSelector:
    """Opt-in selector adapter, preserving original CausalSelection object.

    mode='shadow': return untouched predictions, collect candidate statistics.
    mode='guard': suppress only child onset probabilities below a trained
      genuine-onset score threshold (not a note-count rule). User must opt in.
    """

    def __init__(self, native_selector, *, checkpoint=None,
                 mode="shadow", genuine_threshold=0.10):
        if mode not in ("shadow", "guard"):
            raise ValueError("Harmonic mode must be shadow or guard")
        if not 0.0 < genuine_threshold < 1.0:
            raise ValueError("Genuine-onset threshold must be in (0,1)")
        self.native_selector = native_selector
        self.head = HarmonicGenuineOnsetHead(checkpoint)
        if mode == "guard" and not self.head.trained:
            raise ValueError("Guard mode requires a separately trained harmonic checkpoint")
        self.mode = mode
        self.genuine_threshold = genuine_threshold
        self.stats = HarmonicExpertStats()

    @property
    def active(self):
        return self.native_selector.active

    def step(self, features, probabilities, count, end_sample, available_sample):
        pairs, X = harmonic_pair_features(
            features, probabilities, count, self.native_selector.active,
            materialize=self.head.trained,
        )
        p = np.asarray(probabilities, np.float32)
        c = self.native_selector.config
        self.stats.frames += 1
        self.stats.pairs_considered += len(pairs)
        candidates = set()
        for _, child, _ in pairs:
            j = int(child) - int(PITCHES[0])
            if p[j, 0] >= c.onset and p[j, 1] >= c.active_on:
                candidates.add(j)
        self.stats.candidate_upper_onsets += len(candidates)
        if not candidates or not self.head.trained:
            # Untrained shadow MUST be observational: do not fabricate scores.
            return self.native_selector.step(p, count, end_sample, available_sample)

        genuine = self.head.predict(X)
        # A real upper note can overlap multiple potential harmonic parents.
        # Conservative aggregation: flag only if EVERY applicable parent
        # classifier votes 'not a new independent onset'. Choosing threshold
        # is delegated to out-of-sample validation, never the E2 example.
        suspect = []
        for j in candidates:
            indices = np.flatnonzero(pairs[:, 1] == j + int(PITCHES[0]))
            if len(indices) and np.all(genuine[indices] < self.genuine_threshold):
                suspect.append(j)
        self.stats.low_independence_onsets += len(suspect)

        if self.mode == "shadow":
            # A trained head can now quantify risks on an unchanged baseline.
            return self.native_selector.step(p, count, end_sample, available_sample)
        new_p = p.copy()
        for j in suspect:
            new_p[j, 0] = 0.0
            self.stats.vetoed_upper_onsets += 1
        return self.native_selector.step(new_p, count, end_sample, available_sample)

    def panic(self, sample, reason="stream_end"):
        return self.native_selector.panic(sample, reason)
