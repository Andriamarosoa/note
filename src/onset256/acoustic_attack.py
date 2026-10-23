"""Direct learned acoustic attacks, with a decision on every PCM256 boundary.

The new head sees all 126 native acoustic measurements plus the existing 69
causal context measurements. It may propose any pitch, independently of the
old onset/sustain thresholds, active-note budget, and recovery candidate gate.
"""
from dataclasses import dataclass
import numpy as np

from .acoustics import CausalPitchFeatures, HOP
from .compiled_inference import accelerated
from .error_evidence69 import capture_selector_state, hard_matrix
from .s701_features import LiveS701Features
from .selector import CausalSelection, Selection

DIM = 195
SCHEMA = "ONSET_DIRECT_ACOUSTIC_195D_PCM256_V1"


def acoustic_matrix(features, context):
    f, c = np.asarray(features, np.float32), np.asarray(context, np.float32)
    if f.shape != (49, 126) or c.shape != (49, 69):
        raise ValueError("expected native 126D and causal 69D evidence")
    x = np.concatenate((f, c), axis=1)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite acoustic evidence")
    return np.ascontiguousarray(x)


class AcousticEvidenceStream:
    """Training observer: no annotations and no learned event filtering."""
    def __init__(self, native, s701):
        self.native = native
        self.s701 = accelerated(s701)
        self.frontend = CausalPitchFeatures()
        self.history = LiveS701Features()
        self.selector = CausalSelection(Selection(**native.metadata["selection"]))
        self.received = 0

    def push(self, pcm, *, start_sample):
        pcm = np.asarray(pcm)
        if (pcm.shape != (HOP,) or type(start_sample) is not int or
                start_sample != self.received or pcm.dtype.kind != 'f'):
            raise ValueError("exactly one contiguous new PCM256 block required")
        [(end, feat)] = self.frontend.push(pcm, start_sample=start_sample)
        p, k = self.native.predict(feat)
        # Match MainAttackHead's explicit float32 ABI exactly.
        p, k = np.asarray(p, np.float32), np.asarray(k, np.float32)
        history = self.history.step(feat, p, k)
        rescue = np.asarray(self.s701.predict_proba(history), np.float32)[:, 1]
        state = capture_selector_state(self.selector, end)
        x = acoustic_matrix(feat, hard_matrix(feat, p, k, rescue, history, state))
        self.selector.step(p, k, end, end)
        self.received = end
        return x


@dataclass(frozen=True)
class AttackPolicy:
    cut: float = .5
    reset_ratio: float = .5
    gap_hops: int = 14
    use_baseline: bool = True
    baseline_cut: float = 0.
    enabled: bool = True

    def __post_init__(self):
        if (not 0 < self.cut <= 1 or not 0 < self.reset_ratio < 1 or
                type(self.gap_hops) is not int or self.gap_hops < 1 or
                not 0 <= self.baseline_cut <= 1):
            raise ValueError("invalid calibrated attack policy")


class DirectAttackSelector:
    """Immediate threshold crossing; the refractory rule rejects, never waits.

    A common past-event state prevents cross-head duplicate attacks. Nothing
    is buffered, confirmed by a future hop, or emitted with a past timestamp.
    The disabled control reproduces baseline events without dedup changes.
    """
    def __init__(self, policy):
        self.policy = policy
        self.armed = np.ones(49, bool)
        self.last = np.full(49, -10**12, np.int64)
        self.end = 0

    def step(self, probability, baseline_pitches, *, end_sample):
        q = np.asarray(probability, np.float32)
        pitches = np.asarray(baseline_pitches, np.int64)
        if (q.shape != (49,) or not np.isfinite(q).all() or
                np.any((q < 0) | (q > 1)) or
                type(end_sample) is not int or end_sample != self.end + HOP or
                np.any((pitches < 40) | (pitches > 88))):
            raise ValueError("invalid current-hop attack observations")
        self.end = end_sample
        c = self.policy
        if not c.enabled:
            return tuple(int(p) for p in pitches)
        # float64 comparison also makes archived and live scalar cuts agree.
        scores = q.astype(np.float64)
        self.armed |= scores < c.cut * c.reset_ratio
        ready = self.armed & (scores >= c.cut)
        if c.use_baseline:
            idx = pitches - 40
            ready[idx] |= scores[idx] >= c.baseline_cut
        ready &= end_sample - self.last >= c.gap_hops * HOP
        chosen = np.flatnonzero(ready)
        self.last[chosen] = end_sample
        self.armed[chosen] = False
        return tuple(int(p + 40) for p in chosen)

    def finish(self):
        return ()
