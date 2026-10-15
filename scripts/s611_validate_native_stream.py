"""Compare original S430 batch decisions against NOTE streaming on a real WAV.

S80 event IDs and K are provided, not inferred here. This is an execution-parity
audit, never a new musical accuracy benchmark. No labels are opened.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.live_native_notes import DEFAULT_MODELS, read_decisions, replay
from scripts.s82_external_pr16_neural_inference import load_wav
from scripts.extract_v273_harmonic_trajectory import _frames, features_from_spectrogram
from scripts.s430_infer_note_names_from_new_S80_sound_events import run as original_s430
from scripts.s351_S71_S72_dataset1_477 import geometry46
from scripts.s97_external_s45_original_features import geometry
from causal_note.native_stream import NativeEvent, NativeFeatureStream, POST


def validate(wav, decisions, out, models=DEFAULT_MODELS, legacy_notes=None, native_events=None):
    if out.exists():
        raise ValueError("Immutable audit output already exists")
    out.mkdir(parents=True)
    source = read_decisions(decisions)
    audio = load_wav(wav)
    if source[-1][0].start_sample + POST > len(audio):
        raise ValueError("Incomplete real audio horizon")
    sequence, summary = [], []
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):
        for event, k in source:
            spectrum, times = _frames(audio, event.start_sample)
            values, activation = features_from_spectrogram(spectrum, times, return_states=True)
            sequence.append(np.log1p(activation).astype(np.float32))
            summary.append([values[key] for key in sorted(values)])
        sequence = np.asarray(sequence, np.float32)
        summary = np.asarray(summary, np.float32)
        row_id = np.asarray([event.row_id for event, k in source])
        counts = np.asarray([k for event, k in source], np.int8)
        np.savez_compressed(out / "features.npz", row_id=row_id, sequence42x49=sequence)
        np.savez_compressed(out / "s80.npz", row_id=row_id, s80_K=counts)
        original_s430(out / "features.npz", out / "s80.npz", models / "S402", models / "S416",
                      models / "S428", models / "S419/report.json", models / "S428/report.json", out / "oracle")
        reference = [json.loads(line) for line in (out / "oracle/notes_from_new_sound.jsonl").read_text().splitlines()]
        replays = []
        for block in (256, 997):
            result = replay(wav, decisions, models, out / f"blocks_{block}", block)
            rows = [json.loads(line) for line in (out / f"blocks_{block}/notes.jsonl").read_text().splitlines()]
            mismatches = [a["source_row_id"] for a, b in zip(reference, rows)
                          if any(a[key] != b[key] for key in ("source_row_id", "K", "notes", "MIDI_pitches"))]
            if mismatches or len(rows) != len(reference):
                raise AssertionError("Native batch/streaming note mismatch: " + str(mismatches))
            result["original_S430_note_mismatches"] = len(mismatches)
            result["selected_heads"] = dict(Counter(row["selected_head"] for row in rows))
            replays.append(result)
        # Geometry needs ALL neighboring canonical events, including those not
        # selected for the S434 pitch smoke. Never certify gaps in a subset.
        geometry_rows = None
        if native_events:
            full = [NativeEvent(**json.loads(line)) for line in native_events.read_text().splitlines()]
            stream = NativeFeatureStream(include_geometry46=True, history_samples=32768)
            packets = []; index = at = 0
            while at < len(audio):
                end = min(at + 256, len(audio)); new = []
                while index < len(full) and full[index].end_sample < end:
                    new.append(full[index]); index += 1
                packets.extend(stream.push(audio[at:end], start_sample=at, events=new,
                                           event_watermark=max(0, end - 1764)))
                at = end
            # Pending tail events are unscored, not padded with fictitious audio.
            expected_rows = sum(e.start_sample + 10584 <= max(0, len(audio) - 1764) for e in full)
            if not packets or len(packets) != expected_rows:
                raise AssertionError("Lost or prematurely emitted native geometry event")
            metadata = dict(member=np.asarray(["fixture"] * len(full)),
                            start_samples=np.asarray([e.start_sample for e in full]),
                            end_samples=np.asarray([e.end_sample for e in full]))
            n = len(packets)
            np.testing.assert_array_equal(np.stack([p.geometry46 for p in packets]), geometry46(metadata)[:n])
            np.testing.assert_array_equal(np.stack([p.geometry3 for p in packets]), geometry(
                metadata["member"], metadata["start_samples"], metadata["end_samples"])[:n])
            for packet in packets:
                spectrum, times = _frames(audio, packet.event.start_sample)
                values, activation = features_from_spectrogram(spectrum, times, return_states=True)
                np.testing.assert_array_equal(packet.sequence42x49, np.log1p(activation).astype(np.float32))
                np.testing.assert_array_equal(packet.summary58, np.asarray([values[key] for key in sorted(values)], np.float32))
            geometry_rows = n
    historical_matches = None
    if legacy_notes:
        historical = [json.loads(line) for line in legacy_notes.read_text().splitlines()]
        by_id = {r["source_row_id"]: r for r in reference}
        for row in historical:
            found = by_id[row["source_row_id"]]
            if row["K"] != found["K"] or row["notes"] != found["notes"]:
                raise AssertionError("Historical S434 note names changed")
        historical_matches = len(historical)
    result = dict(status="S611_NATIVE_BATCH_STREAM_PARITY_PASS",
                  source_audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                  source_decisions_sha256=hashlib.sha256(decisions.read_bytes()).hexdigest(),
                  events=len(source), K_histogram=np.bincount(counts, minlength=7).tolist(),
                  frozen_S434_note_rows_identical=historical_matches,
                  native_acoustic_and_geometry_rows_bit_identical=geometry_rows, note_mismatches=0,
                  K_regressions=0, real_audio=True, source_native_K_precomputed=True,
                  S80_microphone_frontend_integrated=False, learned_offsets_integrated=False,
                  independent_accuracy_measured=False, official_promotion=False,
                  replay_results=replays)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "replay_results"}, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("wav", "decisions", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--legacy-notes", type=Path)
    p.add_argument("--native-events", type=Path, help="All canonical events of the recording, without K/labels")
    a = p.parse_args()
    validate(a.wav, a.decisions, a.out, a.models, a.legacy_notes, a.native_events)
