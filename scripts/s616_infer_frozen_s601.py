"""S616: run the ORIGINAL frozen S601 gate on new audio (post-onset, OFFLINE).

No approximation or fresh spectral model. Uses the exact retained Basic Pitch
pretrained weights + trained S601 ExtraTrees candidate classifier and original
candidate/scoring implementations. This is inference only; not a new F1
benchmark, as the song has no reference MIDI annotations.

Typical:
  python -m scripts.s616_infer_frozen_s601 --audio song.wav --out /tmp/s616
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import librosa
import mido
import numpy as np
from scipy.io import wavfile

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, run_inference
from scripts.s601_bp_candidate_gate import candidates, feature_matrix, emit
from scripts.s585_attack_sustain_restrike import sha


CONTRACT = Path("analysis/v273_s602_selected_candidate_freeze.json")
SR = 44100
CHUNK = 24.0
STRIDE = 22.0


def infer(audio_path: Path, out: Path):
    if out.exists():
        raise FileExistsError("Existing output: refusing to overwrite")
    contract = json.loads(CONTRACT.read_text())
    assert contract["model_sha256"] == sha(contract["model_path"]), "S601 model integrity mismatch"
    # Strict reuse of frozen training feature names and decoder policy, no song tuning.
    model = joblib.load(contract["model_path"])
    assert hasattr(model, "predict_proba") and hasattr(model, "n_features_in_")
    assert model.n_features_in_ == len(contract["feature_names"])
    basic_pitch_hashes = {str(p.relative_to(ICASSP_2022_MODEL_PATH)): sha(p)
                          for p in sorted(Path(ICASSP_2022_MODEL_PATH).rglob("*")) if p.is_file()}
    assert basic_pitch_hashes == contract["basic_pitch_weight_hashes"], "BasicPitch original weight mismatch"
    network = Model(ICASSP_2022_MODEL_PATH)
    signal, samplerate = librosa.load(str(audio_path), sr=SR, mono=True)
    assert samplerate == SR and len(signal) > SR, "Nonempty real audio required"
    total_duration = len(signal) / SR
    out.mkdir(parents=True)
    rows, segments = [], []
    starts = np.arange(0., total_duration, STRIDE)
    for i, origin in enumerate(starts):
        start = round(float(origin) * SR)
        stop = min(len(signal), start + round(CHUNK * SR))
        cut = np.asarray(signal[start:stop], np.float32)
        if len(cut) < SR:  # avoid an extremely short last chunk
            continue
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "original_audio_pcm.wav"
            wavfile.write(src, SR, np.int16(np.clip(cut, -1., 1.) * 32767))
            raw = run_inference(str(src), network)
        events = candidates(raw)
        if not events:
            segments.append({"start": float(origin), "candidates": 0, "selected": 0})
            continue
        matrix, names = feature_matrix(raw, events)
        assert names == contract["feature_names"], "Original S601 feature ABI mismatch"
        p = model.predict_proba(matrix)[:, 1]
        # Original threshold, NMS and timestamp policy: no retuning.
        chosen = emit(events, p, **contract["learned_decoder"])
        index = {(round(float(event["time"]), 7), event["pitch"]): event
                 for event in events}
        selection_count = 0
        for note in chosen:
            local_time = float(note["time"])
            if local_time >= STRIDE and stop < len(signal):
                continue
            pitch = int(note["notes"][0])
            # S601 trains/scores ONLY NoteOn. Durations below are unvalidated
            # BasicPitch decoder candidate proposals, never an S601 NoteOff score.
            candidate = min((v for v in events if v["pitch"] == pitch),
                            key=lambda v: abs(v["time"] - local_time))
            length = max([v for v in candidate["ends"] if v > 0] or [.12])
            rows.append({"onset_seconds": round(float(origin) + local_time, 6),
                         "offset_estimate_seconds": round(min(total_duration, float(origin) + local_time + max(.08, min(2., length))), 6),
                         "midi_pitch": pitch,
                         "note": mido.get_note_name(pitch) if hasattr(mido,"get_note_name") else str(pitch),
                         "S601_probability": round(float(note["probability"]), 5),
                         "chunk_start_seconds": float(origin)})
            selection_count += 1
        segments.append({"start": float(origin), "candidates": len(events),
                         "selected": selection_count})
        print("S616 FROZEN INFERENCE", json.dumps(segments[-1]), flush=True)
    rows.sort(key=lambda v: (v["onset_seconds"], v["midi_pitch"]))
    with (out/"notes_s601_original.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
            ["onset_seconds","offset_estimate_seconds","midi_pitch","note","S601_probability","chunk_start_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack();mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    absolute = []
    for note in rows:
        p=note["midi_pitch"]
        absolute.append((int(round(note["onset_seconds"]*960)), 1, p))
        absolute.append((int(round(note["offset_estimate_seconds"]*960)), 0, p))
    last = 0
    for when,kind,pitch in sorted(absolute, key=lambda e: (e[0],e[1],e[2])):
        track.append(mido.Message("note_on" if kind else "note_off",
            note=pitch, velocity=86 if kind else 0, time=max(0,when-last)))
        last=when
    mid.save(out/"notes_s601_original.mid")
    report={"status":"ORIGINAL_S601_FROZEN_MODEL_APPLIED_TO_USER_AUDIO",
            "audio_filename":audio_path.name,
            "audio_sha256":hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            "duration_seconds":total_duration,
            "model_sha256":contract["model_sha256"],
            "basic_pitch_weight_hashes":basic_pitch_hashes,
            "parameters":contract["learned_decoder"],
            "noteon_count":len(rows),
            "midi_range":[40,88],
            "ground_truth_available":False,
            "F1_on_this_song":None,
            "S80_original_run":False,
            "noteoff_accuracy_validated":False,
            "segments":segments}
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print("S616 COMPLETE",json.dumps({"events":len(rows),"seconds":total_duration}), flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--audio",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    infer(a.audio,a.out)
