"""Shared manual capture-gain policy for polyphonic live and WAV replay."""

from __future__ import annotations

import math

import numpy as np


def validate_manual_audio_gain(gain: float) -> float:
    """Return a finite positive gain or raise a user-facing value error."""
    value = float(gain)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("audio gain must be finite and positive")
    return value


def apply_manual_audio_gain(
    samples: np.ndarray,
    gain: float,
) -> tuple[np.ndarray, int]:
    """Apply capture gain once and report samples clipped by that operation."""
    value = validate_manual_audio_gain(gain)
    source = np.asarray(samples, dtype=np.float32)
    scaled = source * value
    induced_clipping = int(np.count_nonzero(np.abs(scaled) > 1.0))
    return (
        np.clip(scaled, -1.0, 1.0).astype(np.float32, copy=False),
        induced_clipping,
    )
