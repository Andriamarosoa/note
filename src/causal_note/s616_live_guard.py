"""S616-LIVE: *experimental* past-only NOTE selection ablations.

Preserves the original S614/S80 weights, architecture, heads and gates.
All thresholds here are research candidates, never promoted based on an
already-opened S613/S614 test. Inference uses PCM-derived native probabilities
plus past-only harmonic features, with no imported midi repository engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from causal_note.strict_live import CausalSelection, Selection, HOP, PITCHES, SR


@dataclass(frozen=True)
class GuardParameters:
    harmonic_penalty: float = 0.0
    attack_rescue: float = 0.0

    def __post_init__(self):
        for name in ("harmonic_penalty", "attack_rescue"):
            value = getattr(self, name)
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError("Guard strengths must belong to [0,1]")


class NativeCausalGuard:
    """Optional acoustic head-selection wrapper; no future samples or labels.

    Penalise suspicious octave/fifth neighbours only while a *lower*
    fundamental is already active and acoustically stronger. True simultaneous
    chords are intentionally not hard vetoed. Attack rescue requires joint
    rise of the learned sustain and the observed physical fundamental.
    """

    def __init__(self, selection: Selection, params: GuardParameters):
        self.selector = CausalSelection(selection)
        self.params = params
        self.previous_sustain = np.zeros(len(PITCHES), np.float32)
        self.previous_fundamental = np.zeros(len(PITCHES), np.float32)
        self.calls = 0

    def step(self, probability, count, end_sample, available_sample, acoustic):
        p = np.asarray(probability, np.float32)
        x = np.asarray(acoustic, np.float32)
        if p.shape != (49, 3) or x.shape != (49, 126) or not np.isfinite(x).all():
            raise ValueError("Original native 49x3 probabilities and 49x126 acoustic features required")
        if not np.isfinite(p).all():
            raise ValueError("Non-finite probabilities")
        edited = p.copy()
        # First partial at the three right-aligned, *past-only* FFT resolutions.
        fundamental = np.maximum(0, x[:, (0, 8, 16)].mean(axis=1))
        if self.params.harmonic_penalty:
            for i in range(49):
                if self.selector.active[i] or not self.selector.armed[i]:
                    continue
                rivalry = 0.0
                for semitones in (12, 19, 24):
                    lower = i - semitones
                    if lower < 0 or not self.selector.active[lower]:
                        continue
                    if p[lower, 1] < .5:
                        continue
                    lower_energy = fundamental[lower]
                    if lower_energy <= .1:
                        continue
                    # A small fundamental of the proposed high note against
                    # an already-active low note is an ambiguity, not proof.
                    r = float(np.clip((lower_energy - fundamental[i]) /
                                      (lower_energy + .1), 0., 1.))
                    rivalry = max(rivalry, r)
                edited[i, 0] *= 1. - self.params.harmonic_penalty * rivalry
        if self.params.attack_rescue:
            # No manufactured attacks from isolated sustain alone.
            sustain_jump = np.maximum(p[:, 1] - self.previous_sustain, 0.)
            physical_jump = np.maximum(fundamental - self.previous_fundamental, 0.)
            confident_physical_jump = np.clip(physical_jump / .7, 0., 1.)
            confidence = np.minimum(np.clip(sustain_jump / .4, 0, 1),
                                    confident_physical_jump)
            eligible = (~self.selector.active) & self.selector.armed & (
                p[:, 0] >= .20) & (p[:, 1] >= .65)
            edited[eligible, 0] = np.maximum(
                edited[eligible, 0],
                np.minimum(1., p[eligible, 0] +
                    self.params.attack_rescue * confidence[eligible]))
        events = self.selector.step(edited, count, end_sample, available_sample)
        self.previous_sustain = p[:, 1].copy()
        self.previous_fundamental = fundamental.copy()
        self.calls += 1
        return events

    def finish(self, sample):
        return self.selector.panic(sample)


def candidate_variants():
    # Predeclared ablations; development only. No inspection of opened test.
    variants = [("S614_baseline", GuardParameters(), 5)]
    for weight in (.2, .4, .6):
        variants.append((f"harmonic_{weight:.1f}", GuardParameters(harmonic_penalty=weight), 5))
    for strength in (.2, .4, .6):
        variants.append((f"rescue_{strength:.1f}", GuardParameters(attack_rescue=strength), 5))
    for penalty, rescue in ((.3, .2), (.3, .4), (.5, .2), (.5, .4)):
        variants.append((f"joint_{penalty:.1f}_{rescue:.1f}",
                         GuardParameters(penalty, rescue), 5))
    for release_frames in (3, 4, 6, 8):
        variants.append((f"release_{release_frames}", GuardParameters(), release_frames))
    return variants


def replay(ends, probabilities, counts, features, selection, params, total_samples):
    stream = NativeCausalGuard(selection, params)
    events = []
    for end, p, k, x in zip(ends, probabilities, counts, features):
        events.extend(stream.step(p, k, int(end), int(end), x))
    events.extend(stream.finish(int(total_samples)))
    return events
