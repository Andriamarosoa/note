"""Preflight CPU-only tuner for experimental 128/256/512 NOTE decision hops.

Checks compute headroom on THIS machine with the existing frozen S613 weights.
No device, no driver buffers and no future note samples are used. Picks a hop
ONCE before the real microphone opens; changing mid-note would corrupt lag and
selector history. A faster hop is NOT an established accuracy improvement.
"""
from __future__ import annotations

import numpy as np

from causal_note.strict_live import HOP, SR, StrictLiveStream


CANDIDATES = (128, 256, 512)


def parse_hop_size(value):
    if str(value).lower() == "auto":
        return "auto"
    try:
        hop = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("hop-size must be auto, 128, 256 or 512") from exc
    if hop not in CANDIDATES:
        raise ValueError("hop-size must be auto, 128, 256 or 512")
    return hop


def choose_cpu_hop(stats, *, utilization=0.55, supported= CANDIDATES):
    """Return lowest hop with enough compute budget, otherwise 512.

    The conservative per-frame p95 deadline allows for audio I/O, FluidSynth,
    Windows scheduling jitter and the callback->worker context switch. This
    is a CPU headroom gate, never an acoustic correctness or Exact-K test.
    """
    if not 0 < utilization < 1:
        raise ValueError("utilization must be within (0,1)")
    allowed = tuple(h for h in CANDIDATES if h in supported)
    if not allowed:
        raise ValueError("No valid trained hop for auto selection")
    for hop in allowed:
        estimate = float(stats[hop]["p95_ms"])
        if np.isfinite(estimate) and estimate <= utilization * (1000. * hop / SR):
            return hop
    return max(allowed)


def benchmark_hop(heads, *, harmonic_mode="off", harmonic_checkpoint=None,
                  harmonic_threshold=.1, frames=64):
    """Benchmark realistic compute on a deterministic test signal, pre-mic.

    No tuning on the actual user's played notes, no adaptivity mid-stream.
    64 baseline frames = 371 ms acoustic duration; a small preflight.
    """
    if frames < 16:
        raise ValueError("Need >=16 frames for stable preflight")
    total = frames * HOP
    t = np.arange(total, dtype=np.float64) / SR
    # Deterministic harmonic chord with separate gradual amplitude envelopes.
    signal = (.05 * np.sin(2*np.pi*82.2*t)
              + .04 * np.sin(2*np.pi*164.9*t)
              + .03 * np.sin(2*np.pi*247.4*t)
              + .02 * np.sin(2*np.pi*329.8*t))
    signal *= np.minimum(1., t * 10.)
    signal = signal.astype(np.float32)
    stats = {}
    for hop in CANDIDATES:
        stream = StrictLiveStream(
            heads, harmonic_mode=harmonic_mode,
            harmonic_checkpoint=harmonic_checkpoint,
            harmonic_threshold=harmonic_threshold,
            hop=hop,
        )
        costs = []
        # Always submit transport-sized 256 chunks, independent of model hop.
        # Exclude startup frames while retaining measurements even in short
        # smoke tests. The normal 64-frame preflight skips 16; a 16-frame
        # test skips only 4, avoiding empty percentile arrays.
        warmup_blocks = min(16, max(1, frames // 4))
        for start in range(0, len(signal), HOP):
            stream.push(signal[start:start+HOP], start_sample=start)
            if start >= warmup_blocks * HOP:
                # A 256-frame input yields 2, 1, or 0 decision frames
                # for respective model hops. Account for empty 512 blocks.
                nframes = (start + HOP)//hop - start//hop
                if nframes:
                    costs.append(stream.timings_ms[-1] / nframes)
        stream.finish()
        stats[hop] = {
            "p95_ms": round(float(np.percentile(costs, 95)), 4),
            "median_ms": round(float(np.median(costs)), 4),
            "decision_period_ms": round(1000. * hop/SR, 4),
            "sampled_frames": len(costs),
        }
    return stats


def resolve_hop(heads, option, *, harmonic_mode="off", harmonic_checkpoint=None,
                harmonic_threshold=.1):
    value = parse_hop_size(option)
    if value != "auto":
        return value, {"method": "manual", "selected": value}
    stats = benchmark_hop(
        heads, harmonic_mode=harmonic_mode, harmonic_checkpoint=harmonic_checkpoint,
        harmonic_threshold=harmonic_threshold,
    )
    # An old S613 checkpoint was trained ONLY at 256. Do not silently
    # pretend that CPU headroom validates its spectral/onset model at 128.
    trained = tuple(heads.metadata.get("trained_hops", [256]))
    if any(int(h) not in CANDIDATES for h in trained):
        raise ValueError("Checkpoint advertises unsupported hop")
    choice = choose_cpu_hop(stats, supported=trained)
    return choice, {
        "method": "cpu_preflight",
        "selected": choice,
        "checkpoint_trained_hops": list(trained),
        "candidate_p95": stats,
        "cpu_utilization_budget": 0.55,
        "warning": "CPU-only selection among hops trained for this checkpoint. F1/Exact-K still require independent validation.",
    }
