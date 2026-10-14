"""Live S80-K MIDI event sidecar (causal streaming protocol, not audio inference).

Usage:
  python -m scripts.stream_k_midi_live --stdin-jsonl --midi-file out.mid
  python -m scripts.stream_k_midi_live --events-jsonl decisions.jsonl --midi-file out.mid

An upstream real-time inference frontend must provide ONE decision per
native onset group as JSONL, with the S80 *predicted* K and exactly K MIDI
pitches. It may separately supply NoteOff IDs while K0. This sidecar NEVER
runs the noncausal S422 5s-future Offset estimator or estimates S80 K.
The original scripts.detect_live remains the active unchanged WAV/PCM16
causal detector. MIDI accuracy & end-to-end live latency are UNMEASURED.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TextIO

from causal_note.live_k_midi_bridge import (
    LockedKLiveMidi, MidiTransition, load_frozen_profile,
    parse_live_decision, save_midi_file,
)

DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "v273_live_and_offline_protected_modes.json"
)


def process_jsonl(source: TextIO, *, output: TextIO, profile: Path,
                  midi_file: Path | None = None, midi_port=None) -> dict:
    """Stream events immediately and validate input profile BEFORE output."""
    load_frozen_profile(profile)
    state = LockedKLiveMidi()
    history: list[MidiTransition] = []
    eof = False
    decision_count = 0

    for lineno, line in enumerate(source, 1):
        if not line.strip():
            continue
        if eof:
            raise ValueError(f"line {lineno}: data after terminal EOF")
        try:
            record = json.loads(line)
            kind, arguments = parse_live_decision(record)
            if kind == "eof":
                emitted = state.finalize(**arguments)
                eof = True
            else:
                emitted = state.process(**arguments)
                decision_count += 1
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc

        for event in emitted:
            record = event.json()
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()  # event-driven output works with pipes, not just files
            if midi_port is not None:
                import mido  # optional MIDI backend; never required for tests
                midi_port.send(mido.Message(
                    "note_on" if event.kind == "note_on" else "note_off",
                    channel=0, note=event.pitch, velocity=event.velocity))
            history.append(event)

    if not eof:
        # Do not invent the true stream end or make a false NoteOff prediction.
        raise ValueError("missing terminal EOF record with measured end_sample")

    if midi_file is not None:
        save_midi_file(history, midi_file)

    if state.active_notes or state.total_input_K != state.total_note_ons:
        raise AssertionError("invalid K-locked live MIDI state after EOF")
    return {
        "status": "K_LOCKED_LIVE_MIDI_EVENTS_ONLY",
        "decisions": decision_count,
        "note_on_events": state.total_note_ons,
        "input_S80_K_sum": state.total_input_K,
        "source_S80_model_inference_performed": False,
        "source_S80_K_must_be_provided_upstream": True,
        "original_live_boundary_model_untouched": True,
        "noncausal_S422_not_executed": True,
        "live_MIDI_accuracy_measured": False,
        "live_end_to_end_latency_measured": False,
        "midi_file": str(midi_file) if midi_file is not None else None,
    }


def parse_args(args=None):
    p = argparse.ArgumentParser(
        description="Strict per-event S80 K to MIDI NoteOn/NoteOff live bridge"
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--stdin-jsonl", action="store_true",
                     help="read causal upstream K-first predictions from stdin")
    inp.add_argument("--events-jsonl", type=Path,
                     help="replay already predicted K-first stream for tests")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--midi-file", type=Path, help="write standard MIDI file on valid EOF")
    p.add_argument("--midi-port", type=str,
                   help="optional real MIDI output port (requires mido + rtmidi)")
    return p.parse_args(args)


def main(argv=None):
    a = parse_args(argv)
    port = None
    try:
        if a.midi_port:
            import mido
            port = mido.open_output(a.midi_port)
        if a.stdin_jsonl:
            report = process_jsonl(
                sys.stdin, output=sys.stdout, profile=a.profile,
                midi_file=a.midi_file, midi_port=port
            )
        else:
            with a.events_jsonl.open(encoding="utf8") as source:
                report = process_jsonl(
                    source, output=sys.stdout, profile=a.profile,
                    midi_file=a.midi_file, midi_port=port
                )
        print(json.dumps(report, sort_keys=True), file=sys.stderr, flush=True)
    finally:
        if port is not None:
            # Safety release on EOF/error, including malformed input.
            import mido
            try:
                port.send(mido.Message("control_change", channel=0,
                                       control=123, value=0))
            finally:
                port.close()


if __name__ == "__main__":
    main()
