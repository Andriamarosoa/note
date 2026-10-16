"""Replay the *same* S613 causal model at 256/1024 input block sizes.

Diagnostic separation:
  (1) physical audio is in the WAV (provided externally),
  (2) native pitch-onset/sustain gating and NoteOn events,
  (3) callback packetization in stream.push, and
  (4) FluidSynth output (audited separately using render_live_midi_offline).

No model weights, thresholds or selection policy changed.
No microphone, piano speaker or user-specific fixed WAV assumed.

Examples:
 python -m scripts.audit_live_vs_offline --wav test_01_E2_seule.wav --expected 40@1.045
 python -m scripts.audit_live_vs_offline --wav test_02_E3_seule.wav --expected 52@1.503
 python -m scripts.audit_live_vs_offline --wav test_03_E2_E3_ensemble.wav --expected 40@1.666 52@1.666
 python -m scripts.audit_live_vs_offline --wav test_04_E2_puis_E3.wav --expected 40@1.533 52@2.525

An expected marker describes an independently measured onset, not an inference
input. The markers are used ONLY for offline evaluation after event emission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import (
    HOP, SR, CausalPitchFeatures, CausalSelection, PITCHES, Selection,
    StrictLiveStream, load_heads,
)
from scripts.live_strict_notes import DEFAULT_MODEL, blocks_wav


def parse_expected(value):
    try:
        pitch, seconds = value.split("@", 1)
        p, t = int(pitch), float(seconds)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Expected pitch@onset_s, e.g. 40@1.045") from exc
    if not (40 <= p <= 88 and np.isfinite(t) and t >= 0):
        raise argparse.ArgumentTypeError("Expected MIDI 40..88, onset >= 0")
    return (p, t)


def event_key(event):
    return (event["type"], int(event["pitch"]), int(event["decision_sample"]),
            event.get("reason"))


def summarize_notes(events):
    open_notes = {}
    notes = []
    problems = []
    for e in events:
        p = e["pitch"]
        if e["type"] == "note_on":
            if p in open_notes:
                problems.append(f"overlapping note_on for MIDI {p}")
            open_notes[p] = e
        elif e["type"] == "note_off":
            old = open_notes.pop(p, None)
            if old is None:
                problems.append(f"unpaired note_off for MIDI {p}")
                continue
            notes.append({
                "pitch": p,
                "onset_s": round(old["decision_sample"] / SR, 6),
                "offset_s": round(e["decision_sample"] / SR, 6),
                "duration_ms": round((e["decision_sample"] - old["decision_sample"]) * 1000 / SR, 3),
                "end_reason": e.get("reason", ""),
            })
    if open_notes:
        problems.extend(f"unclosed note_on MIDI {p}" for p in open_notes)
    short = Counter(
        "<=30ms" if n["duration_ms"] <= 30 else
        "<=60ms" if n["duration_ms"] <= 60 else ">60ms"
        for n in notes
    )
    return dict(note_on=sum(e["type"] == "note_on" for e in events),
                note_off=sum(e["type"] == "note_off" for e in events),
                notes=notes, short_note_counts=dict(short), pairing_problems=problems)


def replay(heads, wav, block_size):
    stream = StrictLiveStream(heads, harmonic_mode="off")
    events = []
    start = 0
    for samples in blocks_wav(wav, block_size):
        events.extend(stream.push(samples, start_sample=start))
        start += len(samples)
    events.extend(stream.finish())
    return events, np.asarray(stream.timings_ms, np.float64)


def head_audit(heads, wav, markers, *, tolerance_s=.08):
    """Offline HEADS/selector audit: label windows don't alter decisions."""
    if not markers:
        return []
    selector = CausalSelection(Selection(**heads.metadata["selection"]))
    frontend = CausalPitchFeatures()
    frames = []
    start = 0
    for audio in blocks_wav(wav, HOP):
        for end, features in frontend.push(audio, start_sample=start):
            p, k = heads.predict(features)
            # Snapshot previous selector state before stepping, not after.
            wanted = {pitch for pitch, _ in markers}
            snap = {
                "time_s": end / SR,
                "k_active_hat": int(np.argmax(k)),
                "pitch_heads": {
                    pitch: (float(p[pitch - 40, 0]), float(p[pitch - 40, 1]),
                            bool(selector.armed[pitch - 40]),
                            int(selector.last_on[pitch - 40]),
                            int(selector.high[pitch - 40]))
                    for pitch in wanted
                },
            }
            frames.append(snap)
            selector.step(p, k, end, frontend.received)
        start += len(audio)
    results = []
    c = selector.config
    for pitch, when in markers:
        # ±80ms defines a matching window, NOT extra causal lookahead.
        candidates = [row for row in frames
                      if when - tolerance_s <= row["time_s"] <= when + tolerance_s]
        after = [row for row in frames if when <= row["time_s"] <= when + tolerance_s]
        if not candidates:
            results.append(dict(pitch=pitch, marked_onset_s=when,
                                diagnosis="No model frames around event"))
            continue
        onsets = np.asarray([row["pitch_heads"][pitch][0] for row in candidates])
        sustain = np.asarray([row["pitch_heads"][pitch][1] for row in candidates])
        joint = (onsets >= c.onset) & (sustain >= c.active_on)
        results.append({
            "pitch": pitch,
            "marked_onset_s": when,
            "max_onset_nearby": round(float(onsets.max()), 5),
            "max_sustain_nearby": round(float(sustain.max()), 5),
            "frames_with_simultaneous_onset_and_sustain": int(joint.sum()),
            "earliest_joint_time_s": next(
                (round(row["time_s"], 6) for row, ok in zip(candidates, joint) if ok), None),
            "max_joint_onset_given_sustain": round(float(onsets[sustain >= c.active_on].max()), 5)
                   if np.any(sustain >= c.active_on) else None,
            "armed_on_joint_frames": [bool(row["pitch_heads"][pitch][2])
                        for row, ok in zip(candidates, joint) if ok][:12],
            "k_active_predictions_after_onset": [row["k_active_hat"] for row in after[:12]],
            "cause_if_missing": (
                "heads do not jointly clear native thresholds"
                if not np.any(joint) else
                "selector armed/refractory/active/budget or timing; inspect emitted events"
            ),
        })
    return results


def audit(wav, model, markers):
    heads = load_heads(model)
    with threadpool_limits(limits=1):
        events256, t256 = replay(heads, wav, HOP)
        events1024, t1024 = replay(heads, wav, 1024)
        heads_report = head_audit(heads, wav, markers)
    keys256 = [event_key(e) for e in events256]
    keys1024 = [event_key(e) for e in events1024]
    report = {
        "wav": str(wav),
        "model": str(model),
        "causal_lookahead_ms": 0,
        "decision_hop_ms": round(1000 * HOP / SR, 5),
        "input_packet_ms_256": round(1000 * 256 / SR, 5),
        "input_packet_ms_1024": round(1000 * 1024 / SR, 5),
        "identical_decision_events_256_vs_1024": keys256 == keys1024,
        "event_count_256": len(events256),
        "event_count_1024": len(events1024),
        "event_timestamp_packetization_ms": {
            str(size): {
                "max_available_minus_decision": round(
                    max((e["available_sample"] - e["decision_sample"]) / SR * 1000
                        for e in ev), 5) if ev else 0,
                "events_delayed_by_packetization": sum(
                    e["available_sample"] != e["decision_sample"] for e in ev),
            }
            for size, ev in ((256, events256), (1024, events1024))
        },
        "cpu_worker_per_packet_ms": {
            str(size): {
                "p95": round(float(np.percentile(vals, 95)), 4),
                "p99": round(float(np.percentile(vals, 99)), 4),
            } for size, vals in ((256, t256), (1024, t1024)) if len(vals)
        },
        "notes": summarize_notes(events256),
        "expected_events": [],
        "pitch_head_gating": heads_report,
        "important": "Same S613 causal predictor in both input modes. This does not test alternate noncausal offline transcription or FluidSynth sound.",
    }
    for pitch, when in markers:
        matched = [e for e in events256 if e["type"] == "note_on" and
                   e["pitch"] == pitch and
                   abs(e["decision_sample"] / SR - when) <= .08]
        nearest = min((abs(e["decision_sample"] / SR - when) for e in events256
                       if e["type"] == "note_on" and e["pitch"] == pitch),
                      default=None)
        report["expected_events"].append({
            "pitch": pitch, "onset_s": when,
            "correct_note_on_in_80ms_window": bool(matched),
            "nearest_same_pitch_attack_error_ms": (
                round(nearest * 1000, 3) if nearest is not None else None),
        })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expected", type=parse_expected, action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.wav, args.model, args.expected)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
