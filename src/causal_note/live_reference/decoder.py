"""Low-latency global MIDI state machine for polyphonic frame predictions."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolyphonicMidiEvent:
    kind: str
    pitch: int
    velocity: int
    frame_index: int
    reason: str = ""


@dataclass(frozen=True)
class PolyphonicDecoderConfig:
    midi_min: int = 40
    midi_max: int = 76
    frame_on_threshold: float = 0.50
    strong_frame_threshold: float = 0.80
    frame_off_threshold: float = 0.25
    onset_threshold: float = 0.50
    activation_frames: int = 2
    release_frames: int = 3
    minimum_retrigger_frames: int = 14
    silence_release_frames: int = 14
    maximum_polyphony: int = 6
    harmonic_suppression_strength: float = 0.25
    harmonic_tolerance_cents: float = 35.0
    audio_onset_lookback_frames: int = 10
    unattacked_frame_threshold: float = 0.90
    harmonic_support_threshold: float = 0.60

    def __post_init__(self) -> None:
        if self.midi_max < self.midi_min:
            raise ValueError("Invalid MIDI range.")
        if not 1 <= self.maximum_polyphony <= 16:
            raise ValueError("Invalid maximum polyphony.")
        if min(
            self.activation_frames,
            self.release_frames,
            self.minimum_retrigger_frames,
            self.silence_release_frames,
            self.audio_onset_lookback_frames,
        ) < 1:
            raise ValueError("Decoder frame counts must be positive.")
        for name, value in (
            ("frame_on_threshold", self.frame_on_threshold),
            ("strong_frame_threshold", self.strong_frame_threshold),
            ("frame_off_threshold", self.frame_off_threshold),
            ("onset_threshold", self.onset_threshold),
            ("unattacked_frame_threshold", self.unattacked_frame_threshold),
            ("harmonic_support_threshold", self.harmonic_support_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.harmonic_tolerance_cents <= 0.0:
            raise ValueError("harmonic_tolerance_cents must be positive.")


def default_decoder_config(
    frame_threshold: float,
    onset_threshold: float,
    midi_min: int = 40,
    midi_max: int = 76,
    maximum_polyphony: int = 6,
) -> PolyphonicDecoderConfig:
    frame = float(frame_threshold)
    return PolyphonicDecoderConfig(
        midi_min=midi_min,
        midi_max=midi_max,
        frame_on_threshold=frame,
        strong_frame_threshold=min(0.95, max(0.80, frame + 0.25)),
        frame_off_threshold=max(0.05, frame * 0.60),
        onset_threshold=float(onset_threshold),
        maximum_polyphony=maximum_polyphony,
    )


def _harmonic_number(
    base_pitch: int,
    candidate_pitch: int,
    tolerance_cents: float = 35.0,
) -> int | None:
    if candidate_pitch <= base_pitch:
        return None
    ratio = 2.0 ** ((candidate_pitch - base_pitch) / 12.0)
    number = int(round(ratio))
    if number < 2:
        return None
    error_cents = abs(1200.0 * math.log2(ratio / number))
    return number if error_cents <= tolerance_cents else None


class PolyphonicDecoder:
    """Own one global state per MIDI pitch; guitar strings never own notes."""

    def __init__(self, config: PolyphonicDecoderConfig) -> None:
        self.config = config
        self.classes = config.midi_max - config.midi_min + 1
        self.active = np.zeros(self.classes, dtype=np.bool_)
        self.activation_count = np.zeros(self.classes, dtype=np.int16)
        self.release_count = np.zeros(self.classes, dtype=np.int16)
        self.last_note_on = np.full(self.classes, -10**9, dtype=np.int64)
        self.frame_index = -1
        self.silence_count = 0
        self.audio_onset_available = False
        self.last_audio_onset = -10**9
        self.harmonic_relations = tuple(
            tuple(
                (base_index, number - 1)
                for base_index in range(class_index)
                for number in [self._relation_harmonic_number(
                    base_index, class_index
                )]
                if number is not None
            )
            for class_index in range(self.classes)
        )

    def _relation_harmonic_number(
        self,
        base_index: int,
        candidate_index: int,
    ) -> int | None:
        return _harmonic_number(
            self.config.midi_min + base_index,
            self.config.midi_min + candidate_index,
            self.config.harmonic_tolerance_cents,
        )

    def _velocity(self, frame_probability: float, onset_probability: float) -> int:
        confidence = max(float(frame_probability), float(onset_probability))
        return int(np.clip(round(35.0 + 92.0 * confidence), 1, 127))

    def _harmonic_support(
        self,
        class_index: int,
        harmonic_amplitude: np.ndarray | None,
        base_mask: np.ndarray,
    ) -> float:
        if harmonic_amplitude is None:
            return 0.0
        support = 0.0
        for base_index, harmonic_index in self.harmonic_relations[class_index]:
            if (
                not bool(base_mask[base_index])
                or harmonic_index >= harmonic_amplitude.shape[1]
            ):
                continue
            support = max(
                support,
                float(harmonic_amplitude[base_index, harmonic_index]),
            )
        return float(np.clip(support, 0.0, 1.0))

    def _recent_audio_onset(self) -> bool:
        return bool(
            self.audio_onset_available
            and self.frame_index - self.last_audio_onset
            <= self.config.audio_onset_lookback_frames
        )

    @property
    def recent_audio_onset(self) -> bool:
        """Whether a causal physical attack is still inside the lookback."""
        return self._recent_audio_onset()

    def _legacy_on_threshold(
        self,
        class_index: int,
        onset_probability: float,
        harmonic_amplitude: np.ndarray | None,
    ) -> float:
        """Preserve the V2.2 decoder contract when audio evidence is absent."""
        threshold = self.config.frame_on_threshold
        if onset_probability >= self.config.onset_threshold:
            return threshold
        support = self._harmonic_support(
            class_index,
            harmonic_amplitude,
            self.active,
        )
        return min(
            self.config.strong_frame_threshold,
            threshold + self.config.harmonic_suppression_strength * support,
        )

    def step(
        self,
        frame_probability: np.ndarray,
        onset_probability: np.ndarray,
        harmonic_amplitude: np.ndarray | None = None,
        audio_active: bool = True,
        audio_hop_index: int | None = None,
        audio_onset: bool | None = None,
        audio_onset_hop_index: int | None = None,
    ) -> list[PolyphonicMidiEvent]:
        # ``frame_index`` remains the zero-based decoder/event clock.  Live
        # callers may supply the absolute audio-hop index so inference skips
        # advance temporal guards without requiring a synthetic prediction.
        # Offline callers retain the original one-step-per-call behaviour.
        previous_frame_index = self.frame_index
        if audio_hop_index is None:
            next_frame_index = self.frame_index + 1
        else:
            try:
                next_frame_index = operator.index(audio_hop_index)
            except TypeError as error:
                raise TypeError("audio_hop_index must be an integer.") from error
            if next_frame_index <= self.frame_index:
                raise ValueError(
                    "audio_hop_index must increase strictly between decoder steps."
                )
        self.frame_index = next_frame_index
        if audio_onset is not None:
            self.audio_onset_available = True
            if bool(audio_onset):
                if audio_onset_hop_index is None:
                    onset_hop_index = self.frame_index
                else:
                    try:
                        onset_hop_index = operator.index(audio_onset_hop_index)
                    except TypeError as error:
                        raise TypeError(
                            "audio_onset_hop_index must be an integer."
                        ) from error
                    if onset_hop_index > self.frame_index:
                        raise ValueError(
                            "audio_onset_hop_index cannot be in the future."
                        )
                self.last_audio_onset = max(
                    self.last_audio_onset, onset_hop_index
                )
        if (
            audio_hop_index is not None
            and previous_frame_index >= 0
            and next_frame_index - previous_frame_index > 1
        ):
            # A skipped inference is an unknown observation.  It advances the
            # physical retrigger clock, but must break every vote that claims
            # consecutive frame/audio evidence.  Active notes and their last
            # NoteOn clock deliberately survive the gap.
            self.activation_count[:] = 0
            self.release_count[:] = 0
        frame = np.asarray(frame_probability, dtype=np.float32)
        onset = np.asarray(onset_probability, dtype=np.float32)
        if frame.shape != (self.classes,) or onset.shape != (self.classes,):
            raise ValueError(f"Expected {self.classes} frame/onset probabilities.")
        if harmonic_amplitude is not None:
            harmonic_amplitude = np.asarray(harmonic_amplitude, np.float32)
            if harmonic_amplitude.ndim != 2 or harmonic_amplitude.shape[0] != self.classes:
                raise ValueError("Invalid harmonic amplitude shape.")

        events: list[PolyphonicMidiEvent] = []
        if not audio_active:
            # Activation votes are evidence from consecutive audible hops.
            # Even the first inactive hop must break that sequence; otherwise
            # a pre-silence vote can turn one post-silence frame into a ghost
            # NoteOn.
            self.activation_count[:] = 0
            self.silence_count += 1
            if self.silence_count >= self.config.silence_release_frames:
                return self.panic(reason="silence")
            # Keep an already active note during the short silence grace
            # period, but never create a new note from a quiet/noisy hop.
            # Previously the first inactive hop continued through the
            # activation path and could emit a frame-only ghost note.
            return events
        else:
            self.silence_count = 0

        # Release/retrigger currently active notes first.
        for class_index in np.flatnonzero(self.active):
            class_index = int(class_index)
            pitch = self.config.midi_min + class_index
            if frame[class_index] < self.config.frame_off_threshold:
                self.release_count[class_index] += 1
            else:
                self.release_count[class_index] = 0
            if self.release_count[class_index] >= self.config.release_frames:
                self.active[class_index] = False
                self.release_count[class_index] = 0
                self.activation_count[class_index] = 0
                events.append(PolyphonicMidiEvent(
                    "note_off", pitch, 0, self.frame_index, "release"
                ))
                continue
            if (
                onset[class_index] >= self.config.onset_threshold
                and frame[class_index] >= self.config.frame_on_threshold
                and self.frame_index - self.last_note_on[class_index]
                >= self.config.minimum_retrigger_frames
                and (
                    not self.audio_onset_available
                    or self._recent_audio_onset()
                )
            ):
                velocity = self._velocity(frame[class_index], onset[class_index])
                events.extend((
                    PolyphonicMidiEvent(
                        "note_off", pitch, 0, self.frame_index, "retrigger"
                    ),
                    PolyphonicMidiEvent(
                        "note_on", pitch, velocity, self.frame_index, "retrigger"
                    ),
                ))
                self.last_note_on[class_index] = self.frame_index

        available = self.config.maximum_polyphony - int(np.sum(self.active))
        if available <= 0:
            return events
        if not self.audio_onset_available:
            legacy_candidates: list[tuple[float, int]] = []
            for class_index in np.flatnonzero(~self.active):
                class_index = int(class_index)
                threshold = self._legacy_on_threshold(
                    class_index,
                    float(onset[class_index]),
                    harmonic_amplitude,
                )
                direct_onset = (
                    onset[class_index] >= self.config.onset_threshold
                    and frame[class_index] >= self.config.frame_on_threshold
                )
                if frame[class_index] >= threshold:
                    self.activation_count[class_index] += 1
                else:
                    self.activation_count[class_index] = 0
                stable_frame = (
                    self.activation_count[class_index]
                    >= self.config.activation_frames
                    and frame[class_index] >= threshold
                )
                if direct_onset or stable_frame:
                    legacy_candidates.append((
                        float(frame[class_index] + onset[class_index]),
                        class_index,
                    ))
            for _, class_index in sorted(
                legacy_candidates, reverse=True
            )[:max(0, available)]:
                pitch = self.config.midi_min + class_index
                velocity = self._velocity(
                    frame[class_index], onset[class_index]
                )
                self.active[class_index] = True
                self.activation_count[class_index] = 0
                self.release_count[class_index] = 0
                self.last_note_on[class_index] = self.frame_index
                events.append(PolyphonicMidiEvent(
                    "note_on", pitch, velocity, self.frame_index, "legacy"
                ))
            return events

        provisional: list[tuple[float, int, bool, str]] = []
        for class_index in np.flatnonzero(~self.active):
            class_index = int(class_index)
            direct_onset = (
                onset[class_index] >= self.config.onset_threshold
                and frame[class_index] >= self.config.frame_on_threshold
            )
            recent_audio_onset = self._recent_audio_onset()
            stable_threshold = (
                self.config.frame_on_threshold
                if recent_audio_onset
                else self.config.unattacked_frame_threshold
            )
            if frame[class_index] >= stable_threshold:
                self.activation_count[class_index] += 1
            else:
                self.activation_count[class_index] = 0
            stable_frame = (
                self.activation_count[class_index] >= self.config.activation_frames
                and frame[class_index] >= stable_threshold
            )
            if direct_onset or stable_frame:
                score = float(frame[class_index] + onset[class_index])
                reason = (
                    "model_onset"
                    if direct_onset
                    else (
                        "frame_attack"
                        if recent_audio_onset
                        else "frame_fallback"
                    )
                )
                provisional.append((score, class_index, direct_onset, reason))

        # A lower candidate from this same hop is valid evidence for a
        # simultaneous fundamental.  This second pass prevents a partial and
        # its fundamental from bypassing each other simply because neither was
        # active before the call.  A pitch-specific model onset remains an
        # escape hatch for intentional octave chords and natural harmonics.
        base_mask = self.active.copy()
        for _, class_index, _, _ in provisional:
            base_mask[class_index] = True
        candidates: list[tuple[float, int, str]] = []
        for score, class_index, direct_onset, reason in provisional:
            support = self._harmonic_support(
                class_index, harmonic_amplitude, base_mask
            )
            if (
                not direct_onset
                and support >= self.config.harmonic_support_threshold
            ):
                span = max(
                    0.0,
                    self.config.unattacked_frame_threshold
                    - self.config.strong_frame_threshold,
                )
                normalized_support = (
                    (support - self.config.harmonic_support_threshold)
                    / max(1.0 - self.config.harmonic_support_threshold, 1e-6)
                )
                harmonic_threshold = (
                    self.config.strong_frame_threshold
                    + span * normalized_support
                )
                if frame[class_index] < harmonic_threshold:
                    self.activation_count[class_index] = 0
                    continue
                score -= self.config.harmonic_suppression_strength * support
                reason = "harmonic_strong_frame"
            candidates.append((score, class_index, reason))

        ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
        for _, class_index, reason in ranked[:max(0, available)]:
            pitch = self.config.midi_min + class_index
            velocity = self._velocity(frame[class_index], onset[class_index])
            self.active[class_index] = True
            self.activation_count[class_index] = 0
            self.release_count[class_index] = 0
            self.last_note_on[class_index] = self.frame_index
            events.append(PolyphonicMidiEvent(
                "note_on", pitch, velocity, self.frame_index, reason
            ))
        return events

    def panic(self, reason: str = "panic") -> list[PolyphonicMidiEvent]:
        events = [
            PolyphonicMidiEvent(
                "note_off", self.config.midi_min + int(index), 0,
                self.frame_index, reason,
            )
            for index in np.flatnonzero(self.active)
        ]
        self.active[:] = False
        self.activation_count[:] = 0
        self.release_count[:] = 0
        self.silence_count = 0
        self.audio_onset_available = False
        self.last_audio_onset = -10**9
        return events
