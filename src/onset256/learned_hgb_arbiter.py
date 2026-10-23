"""Causal, additive HGB-vs-direct Onset arbitration on PCM256.

The acoustic and reference heads are immutable. Every acoustic decision is
preserved; a trained gate may add selected HGB events at the current boundary.
No truth, future waveform, composition ID, or future predictions reach infer().
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

HOP = 256
PITCH_START = 40
PITCHES = 49
BASE_DIM = 195
CONTEXT_DIM = 12
DIM = BASE_DIM + CONTEXT_DIM
SCHEMA = "ONSET_LEARNED_HGB_ARBITER_PCM256_V1"


@dataclass(frozen=True)
class GatePolicy:
    cut: float = .5
    gap_hops: int = 7

    def __post_init__(self):
        if not (0.0 < self.cut <= 1.0 and
                type(self.gap_hops) is int and 1 <= self.gap_hops <= 14):
            raise ValueError("invalid gate policy")


class CausalHGBContext:
    """Generate identical causal rows for offline training and Live inference.

    last_direct refers ONLY to unchanged acoustic outputs; a newly accepted
    HGB event cannot change the frozen acoustic stream's feature distribution.
    """
    def __init__(self):
        self.end = 0
        self.last_direct = np.full(PITCHES, -10**12, np.int64)
        self.last_hgb = np.full(PITCHES, -10**12, np.int64)
        self.history = deque(maxlen=7)

    def step(self, q, evidence, baseline_pitches, direct_pitches, *, end_sample):
        q = np.asarray(q, np.float32)
        evidence = np.asarray(evidence, np.float32)
        hgb = np.asarray(tuple(baseline_pitches), np.int64)
        direct = np.asarray(tuple(direct_pitches), np.int64)
        if (type(end_sample) is not int or end_sample != self.end + HOP or
                q.shape != (PITCHES,) or evidence.shape != (PITCHES, BASE_DIM) or
                not np.isfinite(q).all() or not np.isfinite(evidence).all() or
                np.any(q < 0) or np.any(q > 1) or
                np.any((hgb < PITCH_START) | (hgb >= PITCH_START + PITCHES)) or
                np.any((direct < PITCH_START) | (direct >= PITCH_START + PITCHES))):
            raise ValueError("one contiguous finite PCM256 hop required")
        if len(np.unique(hgb)) != len(hgb) or len(np.unique(direct)) != len(direct):
            raise ValueError("duplicate pitch in a hop")
        rows = np.empty((len(hgb), DIM), np.float32)
        for i, pitch in enumerate(hgb):
            p = int(pitch) - PITCH_START
            past1 = float(self.history[-1][p]) if self.history else 0.0
            past3 = float(self.history[-3][p]) if len(self.history) >= 3 else past1
            prevmax = max(float(v[p]) for v in self.history) if self.history else 0.0
            left = float(q[p - 1]) if p else 0.0
            right = float(q[p + 1]) if p + 1 < PITCHES else 0.0
            direct_age = min(128., (end_sample - int(self.last_direct[p])) / HOP)
            hgb_age = min(128., (end_sample - int(self.last_hgb[p])) / HOP)
            rows[i, :BASE_DIM] = evidence[p]
            rows[i, BASE_DIM:] = (
                float(q[p]), past1, past3, prevmax, float(q[p]) - past1,
                float(np.sum(q > q[p])), float(np.sum(q >= .5)),
                left, right, direct_age, hgb_age, p / 48.0)
        # Update AFTER features: no same/future boundary information leaks into
        # histories, and the same reference observations exist at train and Live.
        self.last_direct[direct - PITCH_START] = end_sample
        self.last_hgb[hgb - PITCH_START] = end_sample
        self.history.append(q.copy())
        self.end = end_sample
        return rows


class LearnedHGBArbiter:
    def __init__(self, model, policy):
        if model is not None and getattr(model, "n_features_in_",None) != DIM:
            raise ValueError("gate model must use exactly 207 causal features")
        self.model = model
        self.policy = policy
        self.context = CausalHGBContext()
        self.last_added = np.full(PITCHES, -10**12, np.int64)
        self.accepted = 0

    def step(self, q, evidence, baseline_pitches, direct_pitches, *, end_sample):
        """Return (all selected pitches, newly added HGB pitches).

        Direct output always survives. Gate cannot backdate, wait, or modify
        another Onset/Offset/Exact-K/pitch stream.
        """
        baseline = tuple(baseline_pitches)
        direct = tuple(direct_pitches)
        rows = self.context.step(q, evidence, baseline, direct,
                                 end_sample=end_sample)
        emitted = set(direct)
        if self.model is None or not baseline:
            return tuple(sorted(emitted)), ()
        scores = np.asarray(self.model.predict_proba(rows), np.float32)[:, 1]
        if scores.shape != (len(baseline),) or not np.isfinite(scores).all():
            raise ValueError("invalid learned gate probabilities")
        accepted = []
        for i, pitch in enumerate(baseline):
            p = pitch - PITCH_START
            recent_direct = end_sample - int(self.context.last_direct[p])
            recent_hgb = end_sample - int(self.last_added[p])
            if (pitch not in emitted and float(scores[i]) >= self.policy.cut
                    and recent_direct >= self.policy.gap_hops * HOP
                    and recent_hgb >= self.policy.gap_hops * HOP):
                emitted.add(pitch)
                accepted.append(pitch)
                self.last_added[p] = end_sample
        self.accepted += len(accepted)
        return tuple(sorted(emitted)), tuple(accepted)


def replay_gate(q, evidence, baseline, direct, model, policy):
    """Deterministic offline replay using exactly the production state machine."""
    q = np.asarray(q, np.float32)
    if (q.ndim != 2 or q.shape[1] != PITCHES or
            baseline.shape != q.shape or direct.shape != q.shape or
            evidence.shape != (len(q), PITCHES, BASE_DIM)):
        raise ValueError("cache dimensions differ")
    stream = LearnedHGBArbiter(model, policy)
    mask = np.zeros_like(direct, bool)
    for f in range(len(q)):
        h = tuple(np.flatnonzero(baseline[f]) + PITCH_START)
        d = tuple(np.flatnonzero(direct[f]) + PITCH_START)
        pitches, _ = stream.step(q[f], evidence[f], h, d,
                                 end_sample=(f+1)*HOP)
        if pitches:
            mask[f, np.asarray(pitches) - PITCH_START] = True
    return mask


def candidate_rows(q, evidence, baseline, direct):
    """Return reference proposals and identical inference-time causal features."""
    context = CausalHGBContext()
    out = []
    locations = []
    for f in range(len(q)):
        h = tuple(np.flatnonzero(baseline[f]) + PITCH_START)
        d = tuple(np.flatnonzero(direct[f]) + PITCH_START)
        rows = context.step(q[f], evidence[f], h, d, end_sample=(f+1)*HOP)
        if h:
            out.append(rows)
            locations.extend((f, int(p)) for p in h)
    return (np.concatenate(out) if out else np.zeros((0, DIM), np.float32),
            locations)


def replay_precomputed(direct, baseline, candidate_scores, policy):
    """Fast calibration with scores frozen BEFORE exploring CAL policies.

    Exactly the same decision order/refractory state as LearnedHGBArbiter.step.
    Does not use annotation, future q, or future audio.
    """
    direct = np.asarray(direct, bool)
    baseline = np.asarray(baseline, bool)
    locations = list(zip(*np.nonzero(baseline)))
    scores = np.asarray(candidate_scores, np.float32)
    if (direct.shape != baseline.shape or direct.ndim != 2 or
            direct.shape[1] != PITCHES or len(scores) != len(locations) or
            not np.isfinite(scores).all()):
        raise ValueError("invalid precomputed gate inputs")
    accepted = direct.copy()
    last_gate = np.full(PITCHES, -10**12, np.int64)
    direct_frames = [np.flatnonzero(direct[:,p]) for p in range(PITCHES)]
    for (frame, pitch), score in zip(locations, scores):
        if direct[frame, pitch] or float(score) < policy.cut:
            continue
        previous = direct_frames[pitch]
        j = np.searchsorted(previous, frame, side="right") - 1
        last_direct = int(previous[j]) if j >= 0 else -10**12
        if (frame - last_direct >= policy.gap_hops and
                frame - last_gate[pitch] >= policy.gap_hops):
            accepted[frame, pitch] = True
            last_gate[pitch] = frame
    return accepted
