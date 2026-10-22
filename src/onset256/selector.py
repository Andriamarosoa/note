"""Private native-256 S613 attack selector, not shared with any other axis."""
from dataclasses import dataclass
import numpy as np
from .acoustics import HOP,SR,PITCHES

@dataclass(frozen=True)
class Selection:
    onset: float = .5
    active_on: float = .5
    active_off: float = .25
    onset_reset: float = .2
    release_frames: int = 5
    refractory_frames: int = 10
    max_notes: int = 6
    count_confidence: float = 1.1  # >1 disables learned count veto for ablation
    confirmation_frames: int = 1
    restrike_onset: float = .0

    def __post_init__(self):
        for key in ("onset", "active_on", "active_off", "onset_reset"):
            if not 0 < getattr(self, key) < 1:
                raise ValueError("Probability threshold must be in (0, 1)")
        if not 0 <= self.count_confidence <= 1.1:
            raise ValueError("Count confidence outside [0, 1.1]")
        if not 0 <= self.restrike_onset <= 1:
            raise ValueError("Restrike threshold outside [0, 1]")
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1
               for key in ("release_frames", "refractory_frames", "max_notes", "confirmation_frames")):
            raise ValueError("Positive integer selection bounds required")
        if self.max_notes > 6:
            raise ValueError("Native Live budget is at most six notes")


class CausalSelection:
    """Per-pitch attack/restrike/release state; output time is never backdated."""

    def __init__(self, config=Selection(), *, hop=HOP):
        if hop != HOP:
            raise ValueError("Only 256-sample Onset decisions supported")
        self.hop = hop
        self.config = config
        # Native S613 thresholds were calibrated at 256 samples. Preserve
        # release and confirmation *time* in candidate hop variants.
        self.release_steps = max(1, (config.release_frames * HOP + hop - 1) // hop)
        self.confirmation_steps = max(1, (config.confirmation_frames * HOP + hop - 1) // hop)
        self.active = np.zeros(49, bool)
        self.armed = np.ones(49, bool)
        self.low = np.zeros(49, np.int32)
        self.high = np.zeros(49, np.int32)
        self.last_on = np.full(49, -10**12, np.int64)
        self.last_end = 0

    def step(self, probability, count, end_sample, available_sample):
        p, k, c = np.asarray(probability), np.asarray(count), self.config
        if (p.shape != (49, 3) or k.shape != (7,) or not np.isfinite(p).all()
                or not np.isfinite(k).all() or np.any((p < 0) | (p > 1))
                or end_sample != self.last_end + self.hop or available_sample < end_sample):
            raise ValueError("Invalid probability stream or unavailable audio")
        self.last_end = end_sample
        onset, sustain, release = p.T
        self.armed |= onset < c.onset_reset
        self.low = np.where((sustain < c.active_off) | (release > .75), np.minimum(self.low + 1, self.release_steps), 0)
        events = []

        def emit(kind, i, reason):
            events.append(dict(type=kind, pitch=int(PITCHES[i]), decision_sample=int(end_sample),
                               available_sample=int(available_sample), time=available_sample / SR,
                               reason=reason, velocity=96 if kind == "note_on" else 0))

        for i in np.flatnonzero(self.active & (self.low >= self.release_steps)):
            emit("note_off", i, "causal_release")
            self.active[i] = False
        confident = (onset >= np.where(self.active, max(c.onset, c.restrike_onset), c.onset)) & (sustain >= c.active_on)
        self.high = np.where(confident, np.minimum(self.high + 1, self.confirmation_steps), 0)
        ready = (self.high >= self.confirmation_steps) & self.armed
        ready &= end_sample - self.last_on >= c.refractory_frames * HOP
        budget = c.max_notes
        if k.max() >= c.count_confidence:
            budget = min(budget, max(int(k.argmax()), int(self.active.sum())))
        for i in sorted(np.flatnonzero(ready), key=lambda j: (-float(onset[j] * sustain[j]), int(j))):
            if self.active[i]:
                emit("note_off", i, "restrike")
            elif self.active.sum() >= budget:
                continue
            emit("note_on", i, "causal_attack")
            self.active[i] = True
            self.armed[i] = False
            self.low[i] = 0
            self.last_on[i] = end_sample
        return events

    def panic(self, sample, reason="stream_end"):
        events = [dict(type="note_off", pitch=int(PITCHES[i]), decision_sample=int(sample),
                       available_sample=int(sample), time=sample / SR, velocity=0,
                       reason=reason) for i in np.flatnonzero(self.active)]
        self.active[:] = False
        return events


