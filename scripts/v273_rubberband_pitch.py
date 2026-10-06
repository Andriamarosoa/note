"""Rubber Band pitch-shift helper matching the V27.3 spectral replay path."""
from __future__ import annotations
import numpy as np

from causal_note.guitarset import SAMPLE_RATE
from scripts.train_v100_spectral_string_slots import _pcm_window, _spectral_map_from_segment
from scripts.spectral_window import COVERED
from scripts.v273_window_experiment import require

PAD = 4096

def rubberband_shifted_map(samples, start, step):
    require(step != 0, "step zero should use frozen reference probability")
    import pyrubberband as pyrb

    left = PAD + COVERED.pre_samples
    right = PAD + COVERED.post_samples
    local = _pcm_window(samples, int(start) - left, left + right).astype(np.float32)
    shifted = np.asarray(
        pyrb.pitch_shift(local, SAMPLE_RATE, float(step)),
        np.float32,
    )
    if len(shifted) < len(local):
        shifted = np.pad(shifted, (0, len(local) - len(shifted)))
    elif len(shifted) > len(local):
        shifted = shifted[:len(local)]
    segment = shifted[PAD:PAD + COVERED.segment_samples]
    require(segment.shape == (COVERED.segment_samples,), "Rubber Band crop drift")
    return _spectral_map_from_segment(segment, window=COVERED).astype(np.float16).astype(np.float32)
