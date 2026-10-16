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


def raw_harmonic_evidence(features, expected_pitch):
    """Inspect actual NOTE log-spectral features (not postprocessed audio).

    The first 24 dimensions are raw 3-window x 8-partial harmonic features
    of each pitch. These are model frontend features, *not* inferred pitch
    probabilities; scores here are diagnostic only.
    """
    h = np.asarray(features[:, :24], dtype=np.float32).reshape(49, 3, 8)
    weights = (1. / np.arange(1, 9) ** .66).astype(np.float32)
    harmonic = .75 * (h * weights).sum(-1) / weights.sum() + .25 * h[..., 0]
    salience = harmonic.mean(axis=1)
    candidates = {int(expected_pitch), 51, 52, 59, 61, 64, 68}
    candidates = sorted(p for p in candidates if int(PITCHES[0]) <= p <= int(PITCHES[-1]))
    best = np.argsort(-salience)[:12]
    all_indices = sorted(set(int(i) for i in best) | {p - int(PITCHES[0]) for p in candidates})
    return {
        "method": "native_3_res_8_harmonics_logged_fft; nonprobabilistic",
        "salience_top_pitches": [
            {"pitch": int(PITCHES[i]), "name": pitch_name(int(PITCHES[i])),
             "score": round(float(salience[i]), 5)}
            for i in best
        ],
        "focus": [
            {"pitch": int(PITCHES[i]), "name": pitch_name(int(PITCHES[i])),
             "score": round(float(salience[i]), 5),
             "window_2048_score": round(float(harmonic[i, 0]), 5),
             "window_4096_score": round(float(harmonic[i, 1]), 5),
             "window_8192_score": round(float(harmonic[i, 2]), 5),
             "h1_to_h4_long_window": [round(float(v), 5) for v in h[i, 2, :4]]}
            for i in all_indices if int(PITCHES[i]) in candidates
        ],
    }


def summarize_trajectory(frames, attack_time, *, expected_pitch=40):
    """Summarize ONLY already-seen causal frames relative to first emission.

    Sampling a future frame here is offline AUDIT, never used to backdate a
    causal detection. Helps diagnose attack-vs-sustain harmonic evidence.
    """
    if not frames or attack_time is None:
        return {"available": False, "reason": "No original note_on in this WAV"}
    offsets_ms = (0, 25, 50, 100, 200, 350, 500)
    selected = []
    for offset in offsets_ms:
        target = attack_time + offset / 1000.0
        future = next((row for row in frames if row["time_s"] >= target), None)
        if future is None:
            break
        selected.append({"offset_requested_ms": offset, **future})
    observed = [row for row in frames
                if attack_time <= row["time_s"] <= attack_time + .5]
    maxima = {}
    for key in ("e2_onset", "e2_sustain", "e2_h1_8192",
                "e2_harmonic_score_8192", "e3_harmonic_score_8192"):
        if observed:
            row = max(observed, key=lambda v: v[key])
            maxima[key] = {"value": row[key], "time_s": row["time_s"]}
    return {
        "available": True,
        "first_original_attack_s": attack_time,
        "expected_pitch": expected_pitch,
        "latency_note": "Offline audit samples later causal frames; NOT a LIVE lookahead or delayed decision",
        "sampled_causal_frames": selected,
        "maxima_during_next_500_ms": maxima,
        "frames_examined": len(observed),
    }


def audit_joint_head_overlap(frames, *, anchor_s=None, window_s=0.5):
    """Locate causal frames meeting *both* onset and sustain requirements.

    This is a measurement only; later frames do not become earlier predictions.
    The anchor is the first ORIGINAL emitted pitch, not a ground-truth attack.
    """
    if anchor_s is None:
        return {"available": False, "reason": "No original attack anchor"}
    rows = [
        row for row in frames
        if anchor_s <= row["time_s"] <= anchor_s + window_s
        and row["e2_onset"] is not None and row["e2_sustain"] is not None
    ]
    if not rows:
        return {"available": False, "reason": "No frames in analysis window"}
    maxima = []
    for sustain in (0.25, 0.4, 0.5, 0.7, 0.9):
        eligible = [row for row in rows if row["e2_sustain"] >= sustain]
        if eligible:
            row = max(eligible, key=lambda r: r["e2_onset"])
            maxima.append({
                "sustain_min": sustain,
                "max_simultaneous_onset": row["e2_onset"],
                "at_sustain": row["e2_sustain"],
                "time_s": row["time_s"],
                "after_first_false_attack_ms": round(
                    1000 * (row["time_s"] - anchor_s), 2),
            })
        else:
            maxima.append({"sustain_min": sustain, "max_simultaneous_onset": None})
    return {
        "available": True,
        "anchor_s": anchor_s,
        "window_s": window_s,
        "frames": len(rows),
        "note": "Head overlap alone; selector arming and note budget may still veto emission",
        "conditional_maxima": maxima,
    }


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
        # Counterfactual: would a lower causal onset threshold rescue the
        # expected note? Do NOT promote without false-positive/accord audits.
        # Keep onset_reset BELOW onset, otherwise the note immediately rearms.
        "lower_onset_015": replace(selection, onset=0.15, onset_reset=0.03),
    }
    # Experimental sweep on this ONE WAV; diagnostic, never a promotion.
    # Distinct names expose onset/sustain interaction at the SAME frame.
    for onset in (0.05, 0.10, 0.15):
        for sustain in (0.25, 0.40, 0.50):
            name = f"grid_on{int(onset * 100):02d}_sus{int(sustain * 100):02d}"
            variants[name] = replace(
                selection, onset=onset, active_on=sustain,
                onset_reset=min(0.02, onset / 2),
            )
    selectors = {key: CausalSelection(value) for key, value in variants.items()}
    frontend = CausalPitchFeatures()
    leveler = NoteAutoLevel(sample_rate=SR, block_size=HOP) if auto_level else None
    outputs = {key: [] for key in variants}
    first_attack = None
    trajectory_frames = []
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
                "raw_harmonic_evidence": raw_harmonic_evidence(features, expected_pitch),
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
                "lower_onset_015_note_on": [
                    x["pitch"] for x in per_frame_events["lower_onset_015"]
                    if x["type"] == "note_on"
                ],
            }
            # Keep a small frame-level trajectory so we can distinguish
            # sparse early E2 evidence from an unrecognized later sustain.
            # The piano/LIVE decoder never sees this audit data.
            focus = {row["pitch"]: row for row in snapshot["raw_harmonic_evidence"]["focus"]}
            e2_features = focus.get(int(expected_pitch), {})
            e3_features = focus.get(52, {})
            trajectory_frames.append({
                "time_s": snapshot["time_s"],
                "predicted_k": snapshot["predicted_k"],
                "k_confidence": snapshot["k_confidence"],
                "e2_onset": snapshot["expected"]["onset"] if snapshot["expected"] else None,
                "e2_sustain": snapshot["expected"]["sustain"] if snapshot["expected"] else None,
                "e2_h1_8192": (e2_features.get("h1_to_h4_long_window") or [0])[0],
                "e2_harmonic_score_8192": e2_features.get("window_8192_score", 0),
                "e3_harmonic_score_8192": e3_features.get("window_8192_score", 0),
                "original_note_on": snapshot["original_note_on"],
            })
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
        "summary": {
            key: {
                "all_note_on": len(events),
                "expected_pitch_on": sum(row["pitch"] == expected_pitch for row in events),
                "other_pitch_on": sum(row["pitch"] != expected_pitch for row in events),
                "distinct_pitches": sorted({row["pitch"] for row in events}),
                "first_expected_time_s": next(
                    (row["time_s"] for row in events if row["pitch"] == expected_pitch), None
                ),
            }
            for key, events in note_events.items()
        },
        "first_original_attack_frame": first_attack,
        "strongest_onset_frame": strongest[1] if strongest else None,
        "attack_to_sustain_trajectory": summarize_trajectory(
            trajectory_frames,
            first_attack["time_s"] if first_attack else None,
            expected_pitch=expected_pitch,
        ),
        "joint_head_overlap": audit_joint_head_overlap(
            trajectory_frames, anchor_s=first_attack["time_s"] if first_attack else None,
        ),
        "grid_diagnostic_note": (
            "One labeled monophonic WAV is insufficient to tune LIVE thresholds. "
            "Compare true polyphony, quiet passages, and note releases before any promotion."
        ),
        "auto_level_diagnostics": leveler.diagnostics() if leveler is not None else None,
        "additional_diagnostic": ("If native spectral E2 evidence is substantial yet E2 onset and sustain probabilities remain near zero, inspect pitch-head training and microphone domain shift; if spectral evidence itself is absent, inspect capture/front-end. This audit deliberately does not inject any note."),
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
