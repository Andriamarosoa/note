"""Real-PCM causality, bounded-state, model and preservation audit for S613."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import wave
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import HOP, SR, load_heads, StrictLiveStream
from scripts.live_strict_notes import DEFAULT_MODEL


def audit(wav, out):
    if out.exists():
        raise FileExistsError("Immutable audit output")
    with wave.open(str(wav), "rb") as source:
        assert source.getframerate() == SR and source.getnchannels() == 1 and source.getsampwidth() == 2
        audio = np.frombuffer(source.readframes(source.getnframes()), "<i2").astype(np.float32) / 32768
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "analysis/evidence/v273-s613-strict-live/selection_frozen.json").read_text())
    heads = load_heads(DEFAULT_MODEL, expected_sha256=contract["model_sha256"])
    def run(signal, block):
        stream = StrictLiveStream(heads)
        result = []
        timing = []
        for start in range(0, len(signal), block):
            result.extend(stream.push(signal[start:start + block], start_sample=start))
            timing.append(stream.timings_ms[-1])
        result.extend(stream.finish())
        assert len(stream.frontend.tail) <= 8192 and len(stream.frontend.history) <= 32
        assert not stream.selector.active.any()
        return result, timing
    with threadpool_limits(limits=1):
        reference, timing = run(audio, 256)
        rechunked, _ = run(audio, 997)
        project = lambda events: [(e["type"], e["pitch"], e["decision_sample"], e["reason"]) for e in events]
        assert project(reference) == project(rechunked)
        cuts = [int(len(audio) * fraction) // HOP * HOP for fraction in (.2, .5, .8)]
        for cut in cuts:
            changed = audio.copy()
            changed[cut:] = np.random.default_rng(cut).normal(0, .3, len(audio) - cut)
            altered, _ = run(changed, HOP)
            assert [e for e in reference if e["decision_sample"] <= cut] == [e for e in altered if e["decision_sample"] <= cut]
        assert any(e["type"] == "note_on" for e in reference), "Real-audio event test cannot be vacuous"
    protected = json.loads((root / "analysis/evidence/v273-s612-recent-native-stream/protected_recent_sources_SHA256.json").read_text())
    for name, expected in protected["sha256"].items():
        file = root / name
        blob = file.read_bytes() if file.exists() else subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
        assert hashlib.sha256(blob).hexdigest() == expected, f"Original protected source changed: {name}"
    steady = np.asarray(timing[8:])
    report = dict(status="S613_REAL_PCM_CAUSALITY_AND_PRESERVATION_PASS", model_sha256=heads.sha256,
                  wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(), seconds=len(audio) / SR,
                  input_block_samples=HOP, input_block_ms=HOP / SR * 1000,
                  original_files_unchanged=len(protected["sha256"]),
                  future_mutation_cut_samples=cuts, future_mutation_event_prefix_exact=True,
                  block_256_vs_997_decision_events_exact=True, note_events=len(reference),
                  bounded_audio_history_samples=8192, bounded_feature_history_frames=32,
                  maximum_active_notes=6, no_event_or_BP_sidecars=True,
                  steady_compute_ms=dict(zip(("median", "p95", "p99", "max"), np.percentile(steady, [50, 95, 99, 100]).tolist())),
                  steady_blocks_exceeding_block_period=int((steady > HOP / SR * 1000).sum()),
                  timed_blocks=len(steady), excluded_warmup_blocks=8,
                  noteoff_on_shutdown=True, microphone_device_tested=False, midi_device_tested=False)
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "real_audio_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in reference))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.wav, args.out)
