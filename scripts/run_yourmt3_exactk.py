"""Offline YourMT3+ transcription, evaluated on the frozen native Exact-K groups."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import numpy as np

from scripts.yourmt3_exactk_common import (FOLDS, SAMPLE_RATE, MODEL_ARGS, SPACE_REVISION,
    CHECKPOINT_PATH, CHECKPOINT_SHA256, assign_onsets, digest, metrics, paired,
    report_markdown, require, write_json)


def load_model(source):
    import torch
    source = source.resolve()
    require(digest(source / CHECKPOINT_PATH) == CHECKPOINT_SHA256, "checkpoint identity drift")
    verified = json.loads((source / "verified.json").read_text())
    require(verified["space_revision"] == SPACE_REVISION, "unverified source")
    sys.path.insert(0, str(source / "amt/src"))
    sys.path.insert(0, str(source))
    from model_helper import load_model_checkpoint
    previous = Path.cwd()
    try:
        os.chdir(source)
        model = load_model_checkpoint(args=MODEL_ARGS, device="cpu")
    finally:
        os.chdir(previous)
    # The demo uses strict=False. Explicitly audit all weights used by inference.
    state = torch.load(source / CHECKPOINT_PATH, map_location="cpu", weights_only=False)["state_dict"]
    state = {k: v for k, v in state.items() if "pitchshift" not in k}
    incompatible = model.load_state_dict(state, strict=False)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,
            f"checkpoint/model mismatch: {incompatible}")
    del state
    return model.eval()


def transcribe(model, samples, batch_size, token_path=None):
    import torch
    import torchaudio
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes
    audio = torch.from_numpy(np.asarray(samples, np.float32)).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, SAMPLE_RATE, model.audio_cfg["sample_rate"])
    length = model.audio_cfg["input_frames"]
    segments = slice_padded_array(audio, length, length)
    segments = torch.from_numpy(segments.astype("float32")).unsqueeze(1)
    with torch.inference_mode():
        tokens, _ = model.inference_file(bsz=batch_size, audio_segments=segments)
    if token_path is not None:
        np.savez_compressed(token_path, **{f"batch_{i}": t for i, t in enumerate(tokens)})
    start_secs = [length * i / model.audio_cfg["sample_rate"] for i in range(len(segments))]
    channels, errors = [], Counter()
    for ch in range(model.task_manager.num_decoding_channels):
        batches = [arr[:, ch, :] for arr in tokens]
        zipped, _, token_errors = model.task_manager.detokenize_list_batches(batches, start_secs, return_events=True)
        notes, note_errors = merge_zipped_note_events_and_ties_to_notes(zipped)
        channels.append(notes)
        errors.update(token_errors); errors.update(note_errors)
    return mix_notes(channels), dict(errors)


def main(a):
    import torch
    import soundfile as sf
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    require(a.fold in FOLDS, "excluded fold")
    require(not a.output.exists(), "refusing to overwrite results")
    a.output.mkdir(parents=True)
    (a.output / "tokens").mkdir()
    manifest = json.loads((a.cohort / "manifest.json").read_text())
    require(manifest["status"] == "verified", "unverified cohort")
    cohort_file = a.cohort / f"fold-{a.fold}.npz"
    require(digest(cohort_file) == manifest["by_fold"][str(a.fold)]["cohort_sha256"], "cohort checksum drift")
    with np.load(cohort_file, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    tracks = json.loads((a.cohort / f"tracks-{a.fold}.json").read_text())
    members = sorted(tracks)
    if a.max_tracks:
        members = members[:a.max_tracks]
    model = load_model(a.source)
    raw_counts = np.full(len(data["k"]), -1, np.int32)
    timings, errors_all, unmatched, overflow, total_events = [], Counter(), 0, 0, 0
    started = time.monotonic()
    with zipfile.ZipFile(a.dataset / "audio_mono-pickup_mix.zip") as archive, (a.output / "notes.jsonl").open("w") as note_out:
        for i, member in enumerate(members):
            require(member[:2] in {"00", "01", "02", "03", "04"}, "excluded player")
            rows = np.flatnonzero(data["member"] == member)
            audio_bytes = archive.read(tracks[member]["audio_member"])
            samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            require(sr == SAMPLE_RATE and samples.ndim == 1, "audio format drift")
            require(len(samples) == tracks[member]["frames"], "audio length drift")
            begin = time.monotonic()
            notes, errors = transcribe(model, samples, a.batch_size,
                                       a.output / "tokens" / (Path(member).stem + ".npz"))
            require(all(np.isfinite(n.onset) and n.onset >= 0 for n in notes), "invalid note onset")
            onset_samples = [round(n.onset * SAMPLE_RATE) for n in notes]
            counts, assignments = assign_onsets(onset_samples, data["starts"][rows], data["ends"][rows])
            raw_counts[rows] = counts
            events = []
            for n, sample, assigned in zip(notes, onset_samples, assignments):
                value = asdict(n)
                value["onset_sample"] = int(sample)
                value["global_index"] = int(data["global_index"][rows[assigned]]) if assigned >= 0 else None
                events.append(value)
            note_out.write(json.dumps({"member": member, "notes": events, "decode_errors": errors}) + "\n")
            note_out.flush()
            unmatched += int(np.sum(assignments < 0)); overflow += int(np.sum(counts > 6))
            total_events += len(notes); errors_all.update(errors)
            timing = {"member": member, "audio_seconds": len(samples)/sr,
                      "inference_seconds": time.monotonic()-begin, "notes": len(notes),
                      "rows": len(rows), "unmatched_notes": int(np.sum(assignments < 0))}
            timings.append(timing)
            write_json(a.output / "progress.json", {"fold": a.fold, "completed_tracks": i+1,
                       "total_tracks": len(members), "seconds": time.monotonic()-started, "latest": timing})
            np.savez_compressed(a.output / "partial-predictions.npz", global_index=data["global_index"], raw_count=raw_counts)
            print(json.dumps({"fold": a.fold, "completed_tracks": i+1, "total_tracks": len(members), **timing}), flush=True)
    used = raw_counts >= 0
    require(a.max_tracks > 0 or np.all(used), "incomplete cohort")
    y, baseline = data["k"][used], data["baseline"][used]
    pred = np.minimum(raw_counts[used], 6)
    np.savez_compressed(a.output / "predictions.npz", **{k: v[used] for k, v in data.items()},
                        predicted=pred, raw_count=raw_counts[used])
    report = {"status": "completed", "fold": a.fold, "smoke_subset": bool(a.max_tracks),
              "protocol": {"experiment": "yourmt3_plus_exactk", "space_revision": SPACE_REVISION,
                           "checkpoint_sha256": CHECKPOINT_SHA256, "model_args": MODEL_ARGS,
                           "cohort_sha256": digest(cohort_file), "pretrained_guitarset_overlap_excluded": False,
                           "offline_context": True, "automatic_promotion": False,
                           "assignment_radius_samples": 882, "note_filter": "all decoded notes",
                           "input_frames": model.audio_cfg["input_frames"],
                           "model_sample_rate": model.audio_cfg["sample_rate"],
                           "source_manifest_sha256": json.loads((a.source / "verified.json").read_text())["source_manifest_sha256"]},
              "freeze_local_combo": metrics(y, baseline), "yourmt3_plus": metrics(y, pred),
              "paired": paired(y, baseline, pred), "raw_event_count": total_events,
              "unmatched_predicted_events": unmatched, "rows_above_six_raw": overflow,
              "decode_errors": dict(errors_all), "timings": timings,
              "runtime": {name: importlib.metadata.version(name) for name in
                          ("torch", "torchaudio", "numpy", "transformers", "pytorch-lightning", "librosa")}}
    write_json(a.output / "report.json", report)
    (a.output / "report.md").write_text(report_markdown(report))
    print(report_markdown(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cohort", "dataset", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--fold", type=int, choices=FOLDS, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-tracks", type=int, default=0)
    main(p.parse_args())
