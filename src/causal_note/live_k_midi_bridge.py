"""Non-invasive causal NoteOn/NoteOff MIDI output for K-first live events.

This is a *streaming MIDI event bridge*, not a causal S80 inference engine:
its caller MUST provide already-predicted (K, MIDI pitches, timestamp).
Without a validated low-latency S80/notes frontend, this bridge makes no
claim of live Exact-K accuracy or live full-note F1.

The original scripts.detect_live and scripts.detect_v8_live paths remain
unchanged; S80/S402/S408/S422 weights and scoring also remain unchanged.
K=0 means NO NEW NOTE, not silence: previously active notes may survive
and/or receive an offset while the incoming K is zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import struct
from typing import Iterable

SAMPLE_RATE = 44_100
MIDI_MIN = 40
MIDI_MAX = 88
MAX_K = 6


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class MidiTransition:
    """Wire-level MIDI event; each note also has a stable internal identity."""
    kind: str
    sample: int
    pitch: int
    velocity: int
    note_id: str

    def __post_init__(self) -> None:
        if self.kind not in ("note_on", "note_off"):
            raise ValueError("unknown MIDI transition")
        _integer("sample", self.sample)
        _integer("pitch", self.pitch, minimum=0)
        if self.pitch > 127:
            raise ValueError("MIDI pitch outside 0..127")
        _integer("velocity", self.velocity)
        if self.velocity > 127 or self.kind == "note_on" and self.velocity == 0:
            raise ValueError("invalid MIDI velocity")
        if not isinstance(self.note_id, str) or not self.note_id:
            raise ValueError("a nonempty note ID is required")

    def json(self) -> dict:
        return {
            "type": self.kind, "sample": self.sample, "pitch": self.pitch,
            "velocity": self.velocity, "note_id": self.note_id,
        }


@dataclass(frozen=True)
class OpenMidiNote:
    note_id: str
    pitch: int
    onset_sample: int


class LockedKLiveMidi:
    """Stateful exact-K NoteOn bridge with causal closures and retriggers.

    All K values come from the original upstream S80 (or an explicitly
    labeled replacement during a *separate* live experiment). No K estimator
    is hidden in this code, and no new note is inferred from harmonics.

    Onsets at the current decision timestamp are validated before any state
    mutation. Each request emits exactly K new NoteOns, even when pitches
    are already active (the old same-pitch instance is closed first).
    """

    def __init__(self) -> None:
        self._active: dict[str, OpenMidiNote] = {}
        self._by_pitch: dict[int, str] = {}
        self._sequence = 0
        self._last_sample = -1
        self._finalized = False
        self._input_K = 0
        self._output_note_ons = 0

    @property
    def active_notes(self) -> tuple[OpenMidiNote, ...]:
        return tuple(sorted(self._active.values(), key=lambda x: (x.onset_sample, x.note_id)))

    @property
    def total_input_K(self) -> int:
        return self._input_K

    @property
    def total_note_ons(self) -> int:
        return self._output_note_ons

    def process(
        self, *, sample: int, s80_K: int,
        midi_pitches: Iterable[int] = (),
        note_off_ids: Iterable[str] = (),
        velocity: int = 80,
    ) -> tuple[MidiTransition, ...]:
        if self._finalized:
            raise RuntimeError("the live MIDI bridge is finalized")
        time = _integer("sample", sample)
        if time < self._last_sample:
            raise ValueError("live event timestamps must be nondecreasing")
        k = _integer("s80_K", s80_K)
        if k > MAX_K:
            raise ValueError("upstream S80 K must be in K0..K6")
        speed = _integer("velocity", velocity, minimum=1)
        if speed > 127:
            raise ValueError("velocity must be <=127")
        pitches = tuple(midi_pitches)
        if len(pitches) != k or len(set(pitches)) != len(pitches):
            raise ValueError("strict S80 cardinality or distinct pitch constraint broken")
        for pitch in pitches:
            if _integer("MIDI pitch", pitch, minimum=MIDI_MIN) > MIDI_MAX:
                raise ValueError("MIDI pitch outside trained 40..88 range")
        close = tuple(note_off_ids)
        if len(set(close)) != len(close):
            raise ValueError("duplicate note_off_ids")
        for note_id in close:
            if not isinstance(note_id, str) or note_id not in self._active:
                raise ValueError("offset requested for an unknown/inactive note ID")
            if time <= self._active[note_id].onset_sample:
                raise ValueError("NoteOff must follow its NoteOn in sample time")
        # A reattack of an already active pitch closes the older event at the
        # same timestamp, then creates a fresh NoteOn, preserving K.
        retriggers = [
            self._by_pitch[p] for p in pitches
            if p in self._by_pitch and self._by_pitch[p] not in close
        ]
        if any(time <= self._active[n].onset_sample for n in retriggers):
            raise ValueError("reattack cannot end old MIDI note at its onset sample")
        transitions: list[MidiTransition] = []
        for note_id in close + tuple(retriggers):
            existing = self._active.pop(note_id)
            del self._by_pitch[existing.pitch]
            transitions.append(
                MidiTransition("note_off", time, existing.pitch, 0, note_id)
            )
        for pitch in pitches:
            self._sequence += 1
            note_id = f"note-{self._sequence:09d}"
            self._active[note_id] = OpenMidiNote(note_id, pitch, time)
            self._by_pitch[pitch] = note_id
            transitions.append(MidiTransition("note_on", time, pitch, speed, note_id))
        if sum(t.kind == "note_on" for t in transitions) != k:
            raise AssertionError("CRITICAL: emitted MIDI NoteOn count differs from provided S80 K")
        self._input_K += k
        self._output_note_ons += sum(t.kind == "note_on" for t in transitions)
        self._last_sample = time
        return tuple(transitions)

    def finalize(self, *, end_sample: int) -> tuple[MidiTransition, ...]:
        if self._finalized:
            raise RuntimeError("live MIDI bridge already finalized")
        end = _integer("end_sample", end_sample)
        if end < self._last_sample or any(end <= n.onset_sample for n in self._active.values()):
            raise ValueError("end_sample must not precede the last input and must follow open onsets")
        result = tuple(
            MidiTransition("note_off", end, n.pitch, 0, n.note_id)
            for n in self.active_notes
        )
        self._active.clear()
        self._by_pitch.clear()
        self._last_sample = end
        self._finalized = True
        if self._output_note_ons != self._input_K:
            raise AssertionError("CRITICAL: cumulative NoteOn count changed from input S80 K")
        return result


def parse_live_decision(record: dict) -> tuple[str, dict]:
    """Require explicit prediction provenance; reject anonymous V8 K guesses."""
    if not isinstance(record, dict):
        raise ValueError("JSONL records must be dictionaries")
    kind = record.get("type", "decision")
    if kind == "eof":
        if set(record) != {"type", "end_sample"}:
            raise ValueError("EOF requires exactly type and end_sample")
        return "eof", {"end_sample": record["end_sample"]}
    if kind != "decision":
        raise ValueError("unsupported live MIDI input record")
    expected = {"type", "sample", "s80_K", "midi_pitches", "note_off_ids", "velocity"}
    if set(record) - expected:
        raise ValueError("unknown live MIDI decision keys")
    if "sample" not in record or "s80_K" not in record or "midi_pitches" not in record:
        raise ValueError("live decision requires sample, predicted S80 K and predicted pitches")
    if not isinstance(record["midi_pitches"], list):
        raise ValueError("predicted pitches must be a JSON array")
    if not isinstance(record.get("note_off_ids", []), list):
        raise ValueError("note_off_ids must be a JSON array")
    return "decision", {
        "sample": record["sample"], "s80_K": record["s80_K"],
        "midi_pitches": record["midi_pitches"],
        "note_off_ids": record.get("note_off_ids", []),
        "velocity": record.get("velocity", 80),
    }


def _variable_length(value: int) -> bytes:
    _integer("MIDI tick delta", value)
    out = [value & 0x7f]
    value >>= 7
    while value:
        out.insert(0, 0x80 | (value & 0x7f))
        value >>= 7
    return bytes(out)


def save_midi_file(events: Iterable[MidiTransition], path: Path) -> None:
    """Write a standards-compatible single-track MIDI file without extra deps.

    Tempo=500000us per quarter, 480 ticks/quarter -> 960 ticks/s.
    Each event sample timestamp comes from real upstream S80 note decisions,
    NOT from wall time. This file can be reconstructed after stream EOF.
    """
    records = tuple(events)
    if any(a.sample > b.sample for a, b in zip(records, records[1:])):
        raise ValueError("MIDI events are not sorted by sample")
    track = bytearray(b"\x00\xff\x51\x03\x07\xa1\x20")
    prev = 0
    for event in records:
        tick = round(event.sample * 960 / SAMPLE_RATE)
        track.extend(_variable_length(tick-prev))
        prev = tick
        # Single-channel MIDI convention: same-pitch retriggers have an
        # explicit NoteOff before the new NoteOn at identical sample time.
        track.extend(bytes((0x90 if event.kind == "note_on" else 0x80,
                            event.pitch, event.velocity)))
    track.extend(b"\x00\xff\x2f\x00")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, 480)
    path.write_bytes(header + struct.pack(">4sI", b"MTrk", len(track)) + track)


def load_frozen_profile(path: Path) -> dict:
    """The live bridge must not silently rename/reference a challenger as official."""
    config = json.loads(path.read_text(encoding="utf8"))
    if config.get("mode") != "dual_live_and_offline_pinned":
        raise ValueError("unrecognized live/offline protection profile")
    if config.get("original_S80_ExactK_reference") != "S80_S397_K5_GUARDED":
        raise ValueError("the protected user S80 reference has changed")
    if config.get("original_active_MIDI_pitch_reference") != "S402_K_CONDITIONED_352D_MLP":
        raise ValueError("the protected S402 reference has changed")
    if config.get("original_active_Offset_reference") != "S408_PITCH_CONDITIONED_SPARSE_FUTURE_AUDIO_NOTE_DURATION_REGRESSOR":
        raise ValueError("the protected S408 reference has changed")
    if config.get("best_offline_native_challenger") != "S422_S80_K_LOCKED_5S_CONTINUOUS_MIC_PITCH_SURVIVAL_OFFSET":
        raise ValueError("the offline best native S422 challenger has changed")
    if config.get("live_quality_certified") is not False:
        raise ValueError("no low-latency S80/S422 live benchmark has been certified")
    if config.get("legacy_live_untouched") is not True:
        raise ValueError("the original causal boundary Live must be preserved")
    return config
