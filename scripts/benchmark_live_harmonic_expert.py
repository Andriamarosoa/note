"""Measure harmonic SHADOW overhead on real WAV, with the same trained heads.

Uses normal causal stream, compares frame-by-frame emitted events and measures
wall-clock CPU compute times. Does NOT measure capture/output-device latency.
Example:
  python -m scripts.benchmark_live_harmonic_expert --wav test_03_E2_E3_ensemble.wav
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from threadpoolctl import threadpool_limits
from causal_note.strict_live import HOP, SR, StrictLiveStream, load_heads
from scripts.live_strict_notes import DEFAULT_MODEL, blocks_wav


def run_once(heads, audio, mode):
    stream = StrictLiveStream(heads, harmonic_mode=mode)
    events = []
    for start in range(0, len(audio), HOP):
        events.extend(stream.push(audio[start:start + HOP], start_sample=start))
    events.extend(stream.finish())
    return events, np.asarray(stream.timings_ms, np.float64), stream


def benchmark(wav, model):
    chunks = list(blocks_wav(wav, HOP))
    audio = np.concatenate(chunks)
    heads = load_heads(model)
    # Warm NumPy, model and FFT implementations without polluting benchmark.
    for mode in ("off", "shadow"):
        warm = StrictLiveStream(heads, harmonic_mode=mode)
        for start in range(0, min(8 * HOP, len(audio)), HOP):
            warm.push(audio[start:start + HOP], start_sample=start)
    metrics = {"off": [], "shadow": []}
    events = {}
    # Alternate order to reduce warm-cache/order bias.
    with threadpool_limits(limits=1):
        for mode in ("off", "shadow", "shadow", "off"):
            emitted, times, stream = run_once(heads, audio, mode)
            if mode in events and events[mode] != emitted:
                raise AssertionError("Non-deterministic replay in " + mode)
            events[mode] = emitted
            metrics[mode].append(times)
    if events["off"] != events["shadow"]:
        raise AssertionError("Harmonic shadow changed native NoteOn/NoteOff events")
    values = {name: np.concatenate(runs) for name, runs in metrics.items()}
    summary = {
        "wav": str(wav),
        "model": str(model),
        "frames": int(len(values["off"]) // len(metrics["off"])),
        "decision_hop_ms": 1000. * HOP / SR,
        "causal_lookahead_ms": 0.,
        "shadow_events_exactly_identical": True,
        "compute_ms": {
            name: {
                "median": round(float(np.median(v)), 4),
                "p95": round(float(np.percentile(v, 95)), 4),
                "p99": round(float(np.percentile(v, 99)), 4),
                "max": round(float(v.max()), 4),
            }
            for name, v in values.items()
        },
    }
    summary["extra_p95_compute_ms"] = round(
        summary["compute_ms"]["shadow"]["p95"] -
        summary["compute_ms"]["off"]["p95"], 4)
    summary["shadow_p95_below_hop"] = bool(
        summary["compute_ms"]["shadow"]["p95"] < summary["decision_hop_ms"])
    summary["note"] = (
        "CPU/offline end-to-end inference benchmark only. Does NOT imply zero "
        "hardware latency; test on the target Windows/iRig system under load."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.wav, args.model), indent=2))


if __name__ == "__main__":
    main()
