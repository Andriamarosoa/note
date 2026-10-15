"""Replay WAV blocks through NOTE heads and gates with original upstream S80 K.

This diagnostic is not a complete microphone-to-S80 frontend. No note detector
from the midi repository, invented K or zero-padded future is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import wave
import numpy as np
from causal_note.native_stream import NativeEvent, NativeNoteStream, NativePitchHeads, SR

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "model/live/native_note"


def read_decisions(path):
    """Strict, label-free S432/S433 input ABI; preserve original IDs and K."""
    result = []
    ids = set()
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"row_id", "start_sample", "end_sample", "s80_K"}:
            raise ValueError("Require original S80 row_id/start_sample/end_sample/s80_K only")
        event = NativeEvent(row["row_id"], row["start_sample"], row["end_sample"])
        k = row["s80_K"]
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 6:
            raise ValueError("Original predicted integer K0..K6 is mandatory")
        if event.row_id in ids or (result and event.start_sample <= result[-1][0].end_sample):
            raise ValueError("Duplicate or overlapping original native event")
        ids.add(event.row_id)
        result.append((event, k))
    if not result:
        raise ValueError("No original native decisions supplied")
    return result


def load_heads(root):
    manifest = json.loads((root / "provenance.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid model path in provenance")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Native checkpoint/gate changed: " + relative)
    return NativePitchHeads(root / "S402", root / "S416", root / "S428",
                            root / "S419/report.json", root / "S428/report.json")


def replay(wav_path, decisions_path, models, out, block_size=256):
    import torch
    from threadpoolctl import threadpool_limits
    if out.exists():
        raise ValueError("Use a new output folder to retain prior evidence")
    if not isinstance(block_size, int) or not 1 <= block_size <= 4096:
        raise ValueError("Replay block size must be 1..4096 samples")
    source = read_decisions(decisions_path)
    torch.set_num_threads(2)
    with threadpool_limits(limits=1), wave.open(str(wav_path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, SR, 2, "NONE"):
            raise ValueError("Original NOTE frontend requires mono PCM16 WAV at 44100 Hz")
        heads = load_heads(models)
        runtime = NativeNoteStream(heads)
        rows, timings = [], []
        at = index = 0
        while raw := wav.readframes(block_size):
            audio = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.
            end = at + len(audio)
            new = []
            while index < len(source) and source[index][0].end_sample < end:
                new.append(source[index])
                index += 1
            start = time.perf_counter()
            rows.extend(runtime.push(audio, start_sample=at, decisions=new))
            timings.append((time.perf_counter() - start) * 1000.)
            at = end
    if index != len(source) or runtime.features.pending_events:
        raise ValueError("WAV ended before all events had real +160 ms context; no zero-padding flush")
    if [r["K"] for r in rows] != [k for e, k in source] or any(len(r["notes"]) != r["K"] for r in rows):
        raise AssertionError("Streaming altered the original S80 note budget")
    out.mkdir(parents=True)
    (out / "notes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    report = dict(status="S611_NATIVE_NOTE_BLOCK_REPLAY_WITH_UPSTREAM_S80",
                  native_heads=["S402", "S416", "S428"], selectors=["S419", "S428"],
                  model_sha256=heads.hashes, per_K_selection=heads.gates,
                  events=len(rows), new_notes=sum(r["K"] for r in rows),
                  original_K_regressions=0, foreign_recognition_model=False,
                  S80_decisions_precomputed_upstream=True,
                  raw_microphone_to_S80_integrated=False, learned_NoteOff_integrated=False,
                  independent_accuracy_measured=False, official_promotion=False,
                  sample_rate=SR, block_size=block_size, audio_seconds=at / SR,
                  minimum_acoustic_context_ms=160,
                  available_delay_ms=dict(min=min(r["acoustic_wait_ms"] for r in rows),
                                          max=max(r["acoustic_wait_ms"] for r in rows)),
                  replay_compute_ms=dict(total=sum(timings), p95=float(np.percentile(timings, 95)),
                                         maximum=max(timings)),
                  physical_microphone_latency_measured=False,
                  audio_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                  upstream_decisions_sha256=hashlib.sha256(decisions_path.read_bytes()).hexdigest())
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--decisions", type=Path, help="Original S433 S80 JSONL; never annotations")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(dict(native_heads=["S402", "S416", "S428"],
                              native_selection=["S419", "S428"],
                              original_S80_S397_required_upstream=True,
                              raw_microphone_to_S80_integrated=False,
                              acoustic_context_ms=160, PR16_geometry_context_ms=240,
                              foreign_recognition_model=False,
                              independent_live_accuracy_measured=False), indent=2))
        return
    if not all((args.wav, args.decisions, args.out)):
        parser.error("Native replay requires --wav, --decisions and --out; see --status")
    print(json.dumps(replay(args.wav, args.decisions, args.models, args.out, args.block_size), indent=2))


if __name__ == "__main__":
    main()
