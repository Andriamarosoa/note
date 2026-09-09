"""Bounded CPU benchmark for the V28 causal pitch-aligned front end.

This script uses deterministic synthetic audio only.  It neither indexes a
dataset nor trains a model, so it is safe to run before the V28 protocol opens
any outer fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.v280_causal_cqt import (
    IMPLEMENTATION,
    CausalCQTConfig,
    causal_cqt,
    cluster_feature_map,
    harmonic_shift_stack,
)


def synthetic_audio(seconds: float, sample_rate: int, seed: int) -> np.ndarray:
    """Create repeatable attacks and overlapping harmonic families."""
    count = int(round(float(seconds) * sample_rate))
    time_axis = np.arange(count, dtype=np.float64) / float(sample_rate)
    rng = np.random.default_rng(seed)
    audio = np.zeros(count, dtype=np.float64)
    frequencies = (82.4069, 110.0, 146.832, 195.998, 246.942, 329.628)
    starts = np.linspace(0.08, max(0.09, seconds * 0.58), len(frequencies))
    for frequency, start in zip(frequencies, starts):
        relative = np.maximum(time_axis - start, 0.0)
        envelope = (time_axis >= start) * np.exp(-2.8 * relative)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        for harmonic in range(1, 7):
            if frequency * harmonic >= sample_rate * 0.49:
                break
            audio += (
                0.10
                * envelope
                * np.sin(2.0 * np.pi * frequency * harmonic * time_axis + phase)
                / harmonic
            )
    audio += rng.normal(0.0, 2e-4, count)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def benchmark(seconds: float, repeats: int, seed: int) -> dict:
    if not 0.25 <= float(seconds) <= 10.0:
        raise ValueError("seconds must be in [0.25, 10]")
    if isinstance(repeats, bool) or not 1 <= int(repeats) <= 5:
        raise ValueError("repeats must be in [1, 5]")
    config = CausalCQTConfig()
    audio = synthetic_audio(seconds, config.sample_rate, seed)
    elapsed = []
    track = None
    for _ in range(int(repeats)):
        started = time.perf_counter()
        track = causal_cqt(audio, config)
        elapsed.append(time.perf_counter() - started)
    assert track is not None

    latest_start = max(0, len(audio) - config.cluster_post_samples)
    features = cluster_feature_map(track, latest_start, config, decision_end=len(audio))
    one_semitone = features[:, : config.output_bin_count : config.bins_per_semitone]
    harmonic = harmonic_shift_stack(
        one_semitone,
        bins_per_octave=12,
        max_harmonic_order=config.max_harmonic_order,
    )
    median = float(np.median(np.asarray(elapsed, dtype=np.float64)))
    cache_bytes = int(track.magnitude.size * np.dtype(np.float16).itemsize)
    return {
        "schema_version": 1,
        "benchmark": "bounded deterministic synthetic CPU front-end",
        "implementation": IMPLEMENTATION,
        "configuration_sha256": config.sha256,
        "audio_seconds": float(len(audio) / config.sample_rate),
        "sample_count": int(len(audio)),
        "repeats": int(repeats),
        "elapsed_seconds": [float(value) for value in elapsed],
        "median_elapsed_seconds": median,
        "audio_seconds_per_wall_second": float(seconds / median),
        "cqt_shape": list(track.magnitude.shape),
        "cluster_feature_shape": list(features.shape),
        "harmonic_stack_shape_at_one_bin_per_semitone": list(harmonic.shape),
        "frequency_bins": int(len(track.frequencies_hz)),
        "output_pitch_bins_at_three_per_semitone": int(config.output_bin_count),
        "maximum_window_samples": int(config.maximum_window_samples),
        "maximum_window_ms": float(1000.0 * config.maximum_window_seconds),
        "estimated_float16_cache_bytes": cache_bytes,
        "estimated_float16_cache_mib": float(cache_bytes / (1024.0**2)),
        "finite": bool(np.isfinite(track.magnitude).all() and np.isfinite(features).all()),
        "checksum": float(np.sum(features, dtype=np.float64)),
        "future_samples_used": False,
        "dataset_indexed": False,
        "training_started": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--seconds", type=float, default=3.0)
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--seed", type=int, default=28031)
    result.add_argument("--output", type=Path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    report = benchmark(args.seconds, args.repeats, args.seed)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
