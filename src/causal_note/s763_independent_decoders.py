"""S763: six independently stateful, CAUSAL, shadow-only decoders.

This is an ARCHITECTURE VALIDATION, not an inference performance upgrade.
The six official S704 scores must still be measured on ONE final serialized
MIDI stream. Here each axis has a private decoder and cannot edit any other
decoder's state or proposals. The protected S708 MIDI stream passes through
an intentionally non-intervening arbiter until real paired validation.

Each fragment comes from already received audio and neural posteriors.
No ground truth, future audio, future NoteOff or source future event is used.

Exactly four 256-sample hops per decision at 44.1 kHz are expected. This
research prototype does not replace S708/S707 or the microphone LIVE path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FIRST = 40
PITCHES = 49
VOICES = 6
SAMPLE_RATE = 44100
HOP = 256
DECISION_SAMPLES = 4 * HOP

AXES = ("exact_k", "active_exact_k", "onset", "offset",
        "frame_pitch", "full_note")


@dataclass(frozen=True)
class Fragment:
    """Defensive snapshot: each decoder receives its OWN immutable copies."""

    sample: int
    post: np.ndarray
    k: np.ndarray

    def __post_init__(self):
        if isinstance(self.sample, (bool, np.bool_)) or int(self.sample) != self.sample:
            raise ValueError("decision sample must be integer")
        sample = int(self.sample)
        if sample <= 0 or sample % DECISION_SAMPLES != 0:
            raise ValueError("noncausal/nonaligned decision clock")
        post = np.array(self.post, dtype=np.float32, copy=True)
        k = np.array(self.k, dtype=np.float32, copy=True)
        if (post.shape != (PITCHES, 3) or k.shape != (7,)
            or not np.isfinite(post).all() or not np.isfinite(k).all()
            or np.any(post < 0) or np.any(post > 1)
            or np.any(k < 0) or np.any(k > 1)
            or abs(float(k.sum()) - 1.) > 1e-3):
            raise ValueError("S763 expected 49x3 probabilities and K0..6 posterior")
        post.setflags(write=False)
        k.setflags(write=False)
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "post", post)
        object.__setattr__(self, "k", k)

    def fork(self):
        """An independent copy, not a shared writable reference."""
        return Fragment(self.sample, self.post, self.k)


@dataclass(frozen=True)
class Intent:
    """Private hypothesis, NOT an official MIDI event."""

    kind: str
    pitch: int
    sample: int
    confidence: float
    reason: str

    def __post_init__(self):
        if self.kind not in ("note_on", "note_off"):
            raise ValueError("invalid intent kind")
        if not FIRST <= self.pitch < FIRST + PITCHES:
            raise ValueError("pitch outside S708 range")
        if self.sample < 0 or not 0 <= self.confidence <= 1:
            raise ValueError("invalid causal event intent")


@dataclass(frozen=True)
class AxisDecision:
    """Result cannot reference or modify another axis's state."""

    axis: str
    sample: int
    count: Optional[int] = None
    pitches: tuple[int, ...] = ()
    intents: tuple[Intent, ...] = ()
    confidence: float = 0.

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError("unknown decoder axis")
        if self.count is not None and not 0 <= self.count <= VOICES:
            raise ValueError("count outside K0..6")


class ExactKDecoder:
    """Independent K0..6 posterior -> one decision. Owns no note state."""

    def step(self, f):
        k = int(f.k.argmax())
        return AxisDecision("exact_k", f.sample, count=k,
                            confidence=float(f.k[k]))


class ActiveKDecoder:
    """Independent sustain/hysteresis state, never consults ExactK's output."""

    def __init__(self, on=.55, off=.30):
        if not (0 <= off < on <= 1):
            raise ValueError("invalid active-K hysteresis")
        self.on, self.off = on, off
        self.held = np.zeros(PITCHES, bool)

    def step(self, f):
        energy = f.post[:, 1]
        self.held = np.where(self.held, energy >= self.off, energy >= self.on)
        count = min(VOICES, int(self.held.sum()))
        return AxisDecision("active_exact_k", f.sample, count=count,
                            confidence=float(np.mean(energy[self.held])) if self.held.any() else 0.)


class OnsetDecoder:
    """Only note-on proposals, with its OWN retrigger and rearm state."""

    def __init__(self, threshold=.75, sustain=.5, rearm=.25,
                 refractory_samples=3*DECISION_SAMPLES):
        if not (0 <= rearm < threshold <= 1 and 0 <= sustain <= 1
                and refractory_samples >= 0):
            raise ValueError("invalid onset policy")
        self.threshold, self.sustain, self.rearm = threshold, sustain, rearm
        self.refractory_samples = refractory_samples
        self.armed = np.ones(PITCHES, bool)
        self.last = np.full(PITCHES, -10**12, np.int64)

    def step(self, f):
        attacks = f.post[:, 0]
        support = f.post[:, 1]
        self.armed |= attacks < self.rearm
        ready = ((attacks >= self.threshold) & (support >= self.sustain) &
                 self.armed & (f.sample - self.last >= self.refractory_samples))
        indices = np.flatnonzero(ready)
        self.armed[indices] = False
        self.last[indices] = f.sample
        result = tuple(Intent("note_on", int(i+FIRST), f.sample,
                              float(attacks[i]*support[i]), "s763_onset_head")
                       for i in indices)
        return AxisDecision("onset", f.sample, intents=result)


class OffsetDecoder:
    """Only release proposals. No state copied from Onset or FullNote."""

    def __init__(self, release=.80, low_sustain=.35, rearm=.45):
        if not (0 <= rearm < release <= 1 and 0 <= low_sustain <= 1):
            raise ValueError("invalid offset policy")
        self.release, self.low_sustain, self.rearm = release, low_sustain, rearm
        self.armed = np.ones(PITCHES, bool)

    def step(self, f):
        release = f.post[:, 2]
        active = f.post[:, 1]
        self.armed |= release < self.rearm
        ready = ((release >= self.release) & (active < self.low_sustain)
                 & self.armed)
        indices = np.flatnonzero(ready)
        self.armed[indices] = False
        return AxisDecision("offset", f.sample, intents=tuple(
            Intent("note_off", int(i+FIRST), f.sample,
                   float(release[i]), "s763_offset_head") for i in indices))


class FramePitchDecoder:
    """Independent binary-pitch state, not Onset events or K gate."""

    def __init__(self, sustain=.5):
        if not 0 <= sustain <= 1:
            raise ValueError("invalid frame-pitch threshold")
        self.sustain = sustain

    def step(self, f):
        support = f.post[:, 1]
        ranked = sorted(np.flatnonzero(support >= self.sustain),
                        key=lambda i: (-float(support[i]), int(i)))
        selected = tuple(sorted(int(i+FIRST) for i in ranked[:VOICES]))
        return AxisDecision("frame_pitch", f.sample,
                            pitches=selected, count=len(selected))


class FullNoteDecoder:
    """Independent paired MIDI hypothesis using ONLY its own note-instance state.

    This hypothesis is not the signed S707 note-off stream. Its metric must
    not be conflated with S704 Full Note F1 of the final MIDI arbiter.
    """

    def __init__(self, attack=.75, sustain=.5, release=.8, low=.30,
                 min_age_samples=2*DECISION_SAMPLES):
        if not (0 <= attack <= 1 and 0 <= sustain <= 1 and
                0 <= low < release <= 1 and min_age_samples >= 0):
            raise ValueError("invalid full-note thresholds")
        self.attack, self.sustain = attack, sustain
        self.release, self.low = release, low
        self.min_age_samples = min_age_samples
        self.held: dict[int, int] = {}
        self.armed = np.ones(PITCHES, bool)

    def step(self, f):
        attack, sustain, release = f.post.T
        intents = []
        self.armed |= attack < self.attack*.4
        # Close FIRST, without observing the onset/offset other decoders made.
        for pitch, start in sorted(self.held.items()):
            i = pitch - FIRST
            if (f.sample - start >= self.min_age_samples
                and (release[i] >= self.release or sustain[i] < self.low)):
                intents.append(Intent("note_off", pitch, f.sample,
                                      float(max(release[i],1-sustain[i])),
                                      "s763_full_note_release"))
                del self.held[pitch]
        ready = sorted(
            (int(i) for i in np.flatnonzero(
                (attack >= self.attack) & (sustain >= self.sustain) & self.armed)),
            key=lambda i: (-float(attack[i]*sustain[i]), i))
        for i in ready:
            pitch = FIRST + i
            if pitch in self.held or len(self.held) >= VOICES:
                continue
            # Self-managed NOTE pair, not a linked Onset head's prediction.
            intents.append(Intent("note_on", pitch, f.sample,
                                  float(attack[i]*sustain[i]),
                                  "s763_full_note_attack"))
            self.held[pitch] = f.sample
            self.armed[i] = False
        return AxisDecision("full_note", f.sample, count=len(self.held),
                            pitches=tuple(sorted(self.held)), intents=tuple(intents))

    def finish(self, sample):
        """Explicit panic closure, never counted as a correct predicted release."""
        if sample < 0:
            raise ValueError("invalid end of stream")
        result = tuple(Intent("note_off", pitch, int(sample), 0.,
                              "stream_end") for pitch in sorted(self.held))
        self.held.clear()
        return result


def default_decoders():
    """Each object owns fully independent state; no singleton or selector."""
    return dict(
        exact_k=ExactKDecoder(),
        active_exact_k=ActiveKDecoder(),
        onset=OnsetDecoder(),
        offset=OffsetDecoder(),
        frame_pitch=FramePitchDecoder(),
        full_note=FullNoteDecoder(),
    )


class IsolatedDecoderBank:
    """Route immutable snapshots to private axis decoders, in any order."""

    def __init__(self, decoders=None, order=AXES):
        self.decoders = dict(default_decoders() if decoders is None else decoders)
        if set(self.decoders) != set(AXES) or set(order) != set(AXES):
            raise ValueError("exactly six distinct decoder axes required")
        self.order = tuple(order)
        if len(set(map(id, self.decoders.values()))) != len(AXES):
            raise ValueError("different axes cannot share decoder state")
        self.last_sample = 0

    def step(self, f):
        if not isinstance(f, Fragment):
            raise TypeError("S763 requires immutable decision fragment")
        if f.sample != self.last_sample + DECISION_SAMPLES:
            raise ValueError("noncontiguous causal decision clock")
        responses = {}
        for name in self.order:
            result = self.decoders[name].step(f.fork())
            if result.axis != name or result.sample != f.sample:
                raise AssertionError("decoder emitted an alien axis or time")
            if any(event.sample != f.sample for event in result.intents):
                raise AssertionError("decoder backdated/future-stamped an event")
            responses[name] = result
        self.last_sample = f.sample
        return responses


@dataclass(frozen=True)
class ConflictReport:
    """Read-only reconciliation diagnostics: DO NOT rewrite signed source."""

    sample: int
    exact_k: int
    active_k: int
    active_pitch_count: int
    onset_proposals: int
    offset_proposals: int
    full_note_proposals: int


class ShadowArbiter:
    """Non-intervention in S763: this prototype can never regress S708."""

    def __init__(self):
        self.last_sample = 0

    def observe(self, decisions):
        if set(decisions) != set(AXES):
            raise ValueError("missing independent hypotheses")
        stamp = {value.sample for value in decisions.values()}
        if len(stamp) != 1:
            raise ValueError("out-of-phase decoders")
        sample = stamp.pop()
        if sample <= self.last_sample:
            raise ValueError("noncausal arbiter order")
        self.last_sample = sample
        return ConflictReport(
            sample, int(decisions["exact_k"].count),
            int(decisions["active_exact_k"].count),
            len(decisions["frame_pitch"].pitches),
            len(decisions["onset"].intents),
            len(decisions["offset"].intents),
            len(decisions["full_note"].intents))

    @staticmethod
    def preserve_signed_stream(source_events):
        """Return exact-content independent copies of S708 events."""
        return [dict(event) for event in source_events]
