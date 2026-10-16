"""S632 strictly causal candidate rescue for masked low/high NOTE attacks.

Research component: changes only per-pitch onset probabilities BEFORE the
original CausalSelection; never changes native K, sustain, release, thresholds,
training checkpoints, or predicts ahead of the captured PCM. Off by default.
A competing harmonic without an independently rising fundamental must not
be silently promoted into a new NOTE.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from causal_note.strict_live import PITCHES, Selection


@dataclass(frozen=True)
class RegisterGate:
    onset_floor: float
    sustain_floor: float
    novelty_floor: float
    fundamental_floor: float
    harmonic_ratio_floor: float
    min_count_support: float = 0.0  # P(native active K > existing active notes)

    def __post_init__(self):
        if not 0 < self.onset_floor < 1 or not 0 < self.sustain_floor < 1:
            raise ValueError("Invalid pitch onset/sustain floors")
        if min(self.novelty_floor, self.fundamental_floor,
               self.harmonic_ratio_floor) < 0 or not 0 <= self.min_count_support <= 1:
            raise ValueError("Negative acoustic evidence thresholds")


@dataclass(frozen=True)
class RescuePolicy:
    bass: RegisterGate | None = None
    high: RegisterGate | None = None


# Thresholds are HYPOTHESES to audit on historical DEV. None is promoted.
BASS_CONSERVATIVE = RegisterGate(.48, .60, .12, .40, .30)
BASS_MODERATE = RegisterGate(.40, .50, .07, .27, .22)
BASS_SENSITIVE = RegisterGate(.30, .50, .03, .18, .12)
HIGH_CONSERVATIVE = RegisterGate(.50, .60, .12, .35, .30)
HIGH_MODERATE = RegisterGate(.40, .50, .07, .22, .20)
HIGH_SENSITIVE = RegisterGate(.30, .50, .03, .12, .12)

# A new bass note must have posterior support for at least one MORE active note.
# This is only an experimental veto using the native K0..6 posterior.
BASS_K10 = RegisterGate(.48, .60, .12, .40, .30, .10)
BASS_K25 = RegisterGate(.48, .60, .12, .40, .30, .25)
BASS_K50 = RegisterGate(.48, .60, .12, .40, .30, .50)
BASS_K75 = RegisterGate(.48, .60, .12, .40, .30, .75)
BASS_MOD_K25 = RegisterGate(.40, .50, .07, .27, .22, .25)
BASS_MOD_K50 = RegisterGate(.40, .50, .07, .27, .22, .50)


def variants():
    return [
        ("S614_control", RescuePolicy()),
        ("bass_conservative", RescuePolicy(bass=BASS_CONSERVATIVE)),
        ("bass_moderate", RescuePolicy(bass=BASS_MODERATE)),
        ("bass_sensitive", RescuePolicy(bass=BASS_SENSITIVE)),
        ("high_conservative", RescuePolicy(high=HIGH_CONSERVATIVE)),
        ("high_moderate", RescuePolicy(high=HIGH_MODERATE)),
        ("high_sensitive", RescuePolicy(high=HIGH_SENSITIVE)),
        ("both_conservative", RescuePolicy(BASS_CONSERVATIVE, HIGH_CONSERVATIVE)),
        ("both_moderate", RescuePolicy(BASS_MODERATE, HIGH_MODERATE)),
        ("both_sensitive", RescuePolicy(BASS_SENSITIVE, HIGH_SENSITIVE)),
        ("bass_moderate_high_sensitive", RescuePolicy(BASS_MODERATE, HIGH_SENSITIVE)),
        ("bass_sensitive_high_moderate", RescuePolicy(BASS_SENSITIVE, HIGH_MODERATE)),
        ("bass_conservative_k10", RescuePolicy(bass=BASS_K10)),
        ("bass_conservative_k25", RescuePolicy(bass=BASS_K25)),
        ("bass_conservative_k50", RescuePolicy(bass=BASS_K50)),
        ("bass_conservative_k75", RescuePolicy(bass=BASS_K75)),
        ("bass_k10_high_moderate", RescuePolicy(BASS_K10, HIGH_MODERATE)),
        ("bass_k25_high_moderate", RescuePolicy(BASS_K25, HIGH_MODERATE)),
        ("bass_k50_high_moderate", RescuePolicy(BASS_K50, HIGH_MODERATE)),
        ("bass_k75_high_moderate", RescuePolicy(BASS_K75, HIGH_MODERATE)),
        ("bass_moderate_k25_high_moderate", RescuePolicy(BASS_MOD_K25, HIGH_MODERATE)),
        ("bass_moderate_k50_high_moderate", RescuePolicy(BASS_MOD_K50, HIGH_MODERATE)),
    ]


def propose(features, native_pitch, state, policy, count=None):
    """Return (independent native probability copy, proposed masks).

    x[0:24]: three resolutions x eight partial strengths;
    x[54:60]: positive score/fundamental changes relative to past 2 hops.
    Each feature row is computed by the frozen S613 causal frontend.
    """
    x = np.asarray(features, np.float32)
    p = np.asarray(native_pitch, np.float32)
    if x.shape != (49, 126) or p.shape != (49, 3):
        raise ValueError("Frozen S613 features (49,126) and pitch (49,3) required")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("Non-finite NOTE rescue input")
    if not isinstance(policy, RescuePolicy):
        raise TypeError("Explicit S632 rescue policy required")

    modified = p.copy()
    proposed = {}
    # Fundamental evidence at 2048/4096/8192, never the 2nd partial
    # alone. Explicitly separate high-note f0 from lower-note harmonics.
    f0 = np.maximum(x[:, (0, 8, 16)], 0).mean(axis=1)
    second = np.maximum(x[:, (1, 9, 17)], 0).mean(axis=1)
    harmonic_ratio = f0 / np.maximum(second, .1)
    physical_growth = np.maximum(x[:, 54:60], 0).max(axis=1)
    # A new f0 is stronger evidence than a whole-note harmonic rise.
    fundamental_growth = np.maximum(x[:, 57:60], 0).max(axis=1)

    for label, bounds, gate in (
        ("bass", (PITCHES <= 63), policy.bass),
        ("high", (PITCHES >= 76), policy.high),
    ):
        mask = np.zeros(49, dtype=bool)
        if gate is not None:
            mask = (
                bounds & ~state.active & state.armed &
                (p[:, 0] >= gate.onset_floor) &
                (p[:, 0] < state.config.onset) &
                (p[:, 1] >= max(gate.sustain_floor, state.config.active_on)) &
                (f0 >= gate.fundamental_floor) &
                (harmonic_ratio >= gate.harmonic_ratio_floor) &
                (physical_growth >= gate.novelty_floor) &
                (fundamental_growth >= gate.novelty_floor * .25)
            )
            if gate.min_count_support > 0:
                k = np.asarray(count, dtype=np.float32)
                if k.shape != (7,) or not np.isfinite(k).all():
                    raise ValueError("Native 7-way active count posterior required")
                next_k = min(7, int(state.active.sum()) + 1)
                mask &= k[next_k:].sum() >= gate.min_count_support
            modified[mask, 0] = state.config.onset
        proposed[label] = mask
    return modified, proposed
