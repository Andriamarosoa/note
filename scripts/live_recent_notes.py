"""Replay real PCM through NOTE S511/S515 -> S569 -> S578/S585.

Original V88 events and finalized Basic Pitch outputs are external replay
inputs. Their true live computation delay is unknown; this is not microphone
transcription. The native heads and selections are never replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import wavfile

from causal_note.recent_audio import RecentEvent, RecentAudioStream, reference_profile, SR
from causal_note.recent_native import RecentHeads, RecentNoteStream, POLICIES


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def replay_events(rows, notes):
    """S527/S589 observed birth, active-at-30ms and BP support, no annotations.

    producer_ready_sample below is the earliest *observation* time only. BP's
    original offline decoder has not been ported; this is a replay-only adapter.
    """
    result = []
    for row in sorted(rows, key=lambda r: (r["original_V88_candidate_start_s"], r["V88_original_event_id"])):
        t = float(row["original_V88_candidate_start_s"])
        born = [n for n in notes if t-.0002 <= float(n["predicted_onset_s"]) <= t+.0202]
        active = tuple(sorted({int(n["pitch_MIDI"]) for n in notes
                               if float(n["predicted_onset_s"]) <= t+.030 < float(n["predicted_offset_s"])}))
        births = tuple(sorted({int(n["pitch_MIDI"]) for n in born}))
        support = tuple(max((float(n.get("pretrained_note_score", 0)) for n in born
                             if int(n["pitch_MIDI"]) == p), default=0.) for p in range(40, 89))
        result.append(RecentEvent(row["source_recording_sha"], row["V88_original_event_id"], t,
                                  row["original_V88_K_soft_prior_ONLY"], births, active, support,
                                  int(np.ceil((t+.030)*SR))))
    return result


def replay_acoustics(audio, rows, notes, *, normalization="rolling3s", blocks=(256,)):
    if not blocks or any(type(b) is not int or not 0 < b <= SR for b in blocks):
        raise ValueError("Block sizes must be integers in 1..44100")
    events = replay_events(rows, notes)
    if not events or len({e.source for e in events}) != 1:
        raise ValueError("Exactly one nonempty original source per audio stream")
    if any(e.time >= len(audio)/SR or e.producer_ready_sample > len(audio) for e in events):
        raise ValueError("WAV ends before the required BP +30ms observation; provide original context")
    stream = RecentAudioStream(normalization=normalization,
                               profile=reference_profile(audio) if normalization == "reference" else None)
    at = index = step = 0
    while at < len(audio):
        end = min(len(audio), at + blocks[step % len(blocks)])
        ready = []
        while index < len(events) and events[index].producer_ready_sample <= end:
            ready.append(events[index]); index += 1
        yield from stream.push(audio[at:end], start_sample=at, events=ready)
        at, step = end, step+1
    yield from stream.finish()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status", action="store_true")
    for name in ("wav", "events", "bp", "models", "out"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--normalization", choices=("rolling3s", "reference"), default="rolling3s")
    p.add_argument("--block-size", type=int, default=256)
    a = p.parse_args(argv)
    if a.status:
        print(json.dumps(dict(status="NOTE_RECENT_NATIVE_HEADS_STREAM_ADAPTER", policies=POLICIES,
                              temporal_features="S569_52D", S80_preserved=True,
                              upstream="original V88 events and finalized BP observations required",
                              default_normalization="rolling3s_experimental",
                              microphone_to_S80_complete=False, learned_NoteOff_integrated=False,
                              live_accuracy_certified=False), indent=2))
        return
    if any(getattr(a, k) is None for k in ("wav", "events", "bp", "models", "out")):
        p.error("--wav, --events, --bp, --models and --out are required")
    if a.out.exists():
        p.error("Output directory exists; use a new run directory")
    sr, audio = wavfile.read(a.wav)
    if sr != SR or audio.ndim != 1 or audio.dtype not in (np.dtype("int16"), np.dtype("float32")):
        p.error("WAV must be mono 44100 Hz PCM16 or float32")
    audio = audio.astype(np.float32) / 32768. if audio.dtype == np.int16 else audio
    heads = RecentHeads(a.models)
    stream = RecentNoteStream(heads)
    events = read_rows(a.events)
    sources = {r['source_recording_sha'] for r in events}
    if len(sources) != 1:
        p.error('Provide canonical events for exactly one WAV source')
    source = next(iter(sources))
    bp = read_rows(a.bp)
    if any('WAV_filename_SHA256' not in n for n in bp):
        p.error('Original BP ledger must retain WAV_filename_SHA256 source identities')
    bp = [n for n in bp if n['WAV_filename_SHA256'] == source]
    start = time.perf_counter()
    results = []
    for packet in replay_acoustics(audio, events, bp,
                                   normalization=a.normalization, blocks=(a.block_size,)):
        result, _, _, _ = stream.push(packet["row"], available_sample=packet["available_sample"])
        result.update({k:v for k,v in packet.items() if k != "row"})
        results.append(result)
    a.out.mkdir(parents=True)
    (a.out / "notes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results))
    report = dict(status="S612_RECENT_NOTE_PCM_REPLAY", source_events=len(results),
                  notes_by_policy={p:sum(len(r["notes"][p]) for r in results) for p in POLICIES},
                  normalization=a.normalization, models_SHA256=heads.hashes,
                  acoustic_seconds=len(audio)/SR, elapsed_replay_seconds=time.perf_counter()-start,
                  upstream_events_and_BP_precomputed=True, upstream_latency_unknown=True,
                  reference_mode_uses_future=a.normalization == "reference",
                  live_accuracy_certified=False, no_promotion=True,
                  learned_NoteOff_integrated=False, S80_ExactK_unchanged=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
