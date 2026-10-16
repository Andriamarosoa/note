"""Audit false chords on a single-struck guitar note without changing NOTE weights.

Runs the same causal frontend and native trained heads as LIVE, comparing
original selections against two counterfactual count policies. These are
diagnostics, NOT approved production threshold changes or accuracy estimates.

Examples:
  python -m scripts.audit_live_pitch_count --wav irig_E2_wasapi.wav --expected-pitch 40
  python -m scripts.audit_live_pitch_count --wav irig_E2_wasapi.wav --expected-pitch 40 --auto-level
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from causal_note.strict_live import (
    CausalPitchFeatures, CausalSelection, HOP, PITCHES, SR, Selection,
    load_heads,
)
from scripts.live_input_level import NoteAutoLevel
from scripts.live_strict_notes import DEFAULT_MODEL, blocks_wav


def pitch_name(pitch):
    return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[pitch % 12] + str(pitch // 12 - 1)


def process_wav(wav_path, model_path, *, auto_level=False, expected_pitch=40):
    heads = load_heads(model_path)
    selection = Selection(**heads.metadata["selection"])
    variants = {
        "original": selection,
        # Real K distribution confidence veto, previously 1.1 (> every
        # possible softmax probability, therefore never consulted).
        "count_gate_050": replace(selection, count_confidence=0.5),
        # Single-note budget *only* as a diagnostic bound. This is NOT a
        # recommended LIVE setting: it would suppress real polyphony.
        "mono_budget_1": replace(selection, max_notes=1),
    }
    selectors = {key: CausalSelection(value) for key, value in variants.items()}
    frontend = CausalPitchFeatures()
    leveler = NoteAutoLevel(sample_rate=SR, block_size=HOP) if auto_level else None
    outputs = {key: [] for key in variants}
    first_attack = None
    strongest = None
    start = 0
    observed_frames = 0

    for raw in blocks_wav(wav_path, HOP):
        audio = leveler.apply(raw) if leveler is not None else raw
        for end, features in frontend.push(audio, start_sample=start):
            observed_frames += 1
            p, k = heads.predict(features)
            per_frame_events = {
                key: selector.step(p, k, end, frontend.received)
                for key, selector in selectors.items()
            }
            for key, events in per_frame_events.items():
                outputs[key].extend(events)

            candidates = np.argsort(-(p[:, 0] * p[:, 1]))[:12]
            candidate_rows = []
            for ix in candidates:
                candidate_rows.append({
                    "pitch": int(PITCHES[ix]),
                    "name": pitch_name(int(PITCHES[ix])),
                    "onset": round(float(p[ix, 0]), 5),
                    "sustain": round(float(p[ix, 1]), 5),
                    "release": round(float(p[ix, 2]), 5),
                    "onset_x_sustain": round(float(p[ix, 0] * p[ix, 1]), 5),
                })
            expected_index = int(expected_pitch) - int(PITCHES[0])
            expected_row = (
                {"pitch": expected_pitch, "name": pitch_name(expected_pitch),
                 "onset": round(float(p[expected_index, 0]), 5),
                 "sustain": round(float(p[expected_index, 1]), 5),
                 "release": round(float(p[expected_index, 2]), 5)}
                if 0 <= expected_index < len(PITCHES) else None
            )
            snapshot = {
                "sample": int(end),
                "time_s": round(float(end / SR), 6),
                "predicted_k": int(np.argmax(k)),
                "k_confidence": round(float(np.max(k)), 5),
                "k_probabilities_0_through_6": [
                    round(float(v), 5) for v in k
                ],
                "expected": expected_row,
                "pitch_ranking": candidate_rows,
                "original_note_on": [
                    x["pitch"] for x in per_frame_events["original"]
                    if x["type"] == "note_on"
                ],
                "count_gate_050_note_on": [
                    x["pitch"] for x in per_frame_events["count_gate_050"]
                    if x["type"] == "note_on"
                ],
                "mono_budget_1_note_on": [
                    x["pitch"] for x in per_frame_events["mono_budget_1"]
                    if x["type"] == "note_on"
                ],
            }
            if snapshot["original_note_on"] and first_attack is None:
                first_attack = snapshot
            score = float(np.max(p[:, 0] * p[:, 1]))
            if strongest is None or score > strongest[0]:
                strongest = (score, snapshot)
        start += len(audio)

    note_events = {}
    for key, events in outputs.items():
        note_events[key] = [
            {
                "pitch": int(e["pitch"]),
                "name": pitch_name(int(e["pitch"])),
                "time_s": round(float(e["time"]), 6),
            }
            for e in events if e["type"] == "note_on"
        ]
    return {
        "model": str(model_path),
        "wav": str(wav_path),
        "expected_pitch": int(expected_pitch),
        "expected_name": pitch_name(int(expected_pitch)),
        "auto_level": bool(auto_level),
        "selection_original": vars(selection),
        "count_veto_disabled": selection.count_confidence > 1.0,
        "frames_analyzed": observed_frames,
        "note_on_by_selection": note_events,
        "first_original_attack_frame": first_attack,
        "strongest_onset_frame": strongest[1] if strongest else None,
        "auto_level_diagnostics": leveler.diagnostics() if leveler is not None else None,
        "interpretation": (
            "Count gate may suppress spurious simultaneity, but a K head "
            "predicting the wrong K or a pitch head preferring harmonics "
            "needs separate training/audit. Mono budget cannot be promoted "
            "as a general polyphonic LIVE policy."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expected-pitch", type=int, default=40)
    parser.add_argument("--auto-level", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = process_wav(
        args.wav, args.model, auto_level=args.auto_level,
        expected_pitch=args.expected_pitch,
    )
    payload = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(payload + "\n", encoding="utf-8")
        print(f"Audit saved to {args.report}")
    print(payload)


if __name__ == "__main__":
    main()
