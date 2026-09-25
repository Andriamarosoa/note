"""Versioned native acoustic windows; historical callers retain 23 frames.

The covered window ends at the existing group + verifier horizon, not at a
duration selected from held-out annotations. It requires finalization at that
horizon; this is not an instantaneous prediction at the group start.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpectralWindow:
    name: str
    pre_samples: int
    post_samples: int

    @property
    def segment_samples(self):
        return self.pre_samples + self.post_samples

    @property
    def time_frames(self):
        return 1 + (self.segment_samples - 256) // 128

    @property
    def centers(self):
        return np.arange(self.time_frames, dtype=np.float64) * 128 + 128 - self.pre_samples


LEGACY = SpectralWindow("legacy_40ms", 1308, 1764)
COVERED = SpectralWindow("group_plus_verifier", 1308, 2788)
WINDOWS = {w.name: w for w in (LEGACY, COVERED)}
WINDOW_KEYS = ("spectral_window", "spectral_pre_samples", "spectral_post_samples")


def window_metadata(window):
    if WINDOWS.get(window.name) != window:
        raise ValueError("unknown spectral window")
    return {"spectral_window": np.asarray([window.name]),
            "spectral_pre_samples": np.asarray([window.pre_samples], np.int64),
            "spectral_post_samples": np.asarray([window.post_samples], np.int64)}


def cache_window(cache, *, version=None):
    present = [key in cache for key in WINDOW_KEYS]
    if any(present) and not all(present):
        raise ValueError("incomplete spectral window metadata")
    if version in (1, 2) and any(present):
        raise ValueError("legacy schema has unversioned spectral window metadata")
    if version == 3 and not all(present):
        raise ValueError("schema 3 requires spectral window metadata")
    window = LEGACY
    if all(present):
        if any(np.asarray(cache[key]).shape != (1,) for key in WINDOW_KEYS):
            raise ValueError("invalid spectral window metadata shape")
        window = WINDOWS.get(str(cache["spectral_window"][0]))
        if window is None:
            raise ValueError("unknown spectral window name")
        expected = window_metadata(window)
        if any(not np.array_equal(cache[key], expected[key]) for key in WINDOW_KEYS):
            raise ValueError("spectral window metadata mismatch")
    if "spectral" in cache and np.asarray(cache["spectral"]).shape != (len(cache["target"]), window.time_frames, 64, 3):
        raise ValueError("spectral shape does not match window metadata")
    return window


def time_coordinates(time_frames):
    """Keep historical coordinates exactly, extrapolate for appended frames."""
    if time_frames not in (LEGACY.time_frames, COVERED.time_frames):
        raise ValueError("unsupported spectral frame count")
    old = np.linspace(-1, 1, LEGACY.time_frames, dtype=np.float32)
    if time_frames == len(old):
        return old
    extra = 1 + np.arange(1, time_frames - len(old) + 1, dtype=np.float64) * 2 / (len(old) - 1)
    return np.concatenate([old, extra.astype(np.float32)])
