"""Causal, pitch-aligned acoustic features for the V28 count expert.

The extractor deliberately exposes sample endpoints.  Every frame is computed
from ``audio[end - window_length:end]`` with left zero padding only.  This
makes the no-lookahead contract testable instead of relying on a library's
implicit centering convention.

The implementation is an octave-grouped, multi-resolution constant-Q
filterbank.  Bins are spaced exactly on a constant-Q grid (three bins per
semitone by default), while bins in one octave share a right-aligned FFT.  The
shared FFTs keep track-level cache mining practical on CPU without adding a
runtime dependency such as librosa.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .guitarset import SAMPLE_RATE


CACHE_SCHEMA_VERSION = 1
IMPLEMENTATION = "v280_causal_octave_cqt_v1"


class V280FeatureError(ValueError):
    """Raised when a V28 feature or cache contract is violated."""


def midi_to_hz(midi: float | np.ndarray) -> float | np.ndarray:
    """Convert MIDI pitch to frequency without quantizing fractional bins."""
    value = np.asarray(midi, dtype=np.float64)
    result = 440.0 * np.power(2.0, (value - 69.0) / 12.0)
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class CausalCQTConfig:
    """Frozen signal contract for V28 feature mining."""

    sample_rate: int = SAMPLE_RATE
    hop_length: int = 256
    bins_per_semitone: int = 3
    input_min_midi: float = 40.0
    output_max_midi: float = 83.0
    max_harmonic_order: int = 8
    cluster_post_samples: int = 1764
    cluster_frames: int = 24
    block_frames: int = 24
    log_gain: float = 100.0
    nyquist_margin: float = 0.98

    def __post_init__(self) -> None:
        integer_fields = (
            "sample_rate",
            "hop_length",
            "bins_per_semitone",
            "max_harmonic_order",
            "cluster_post_samples",
            "cluster_frames",
            "block_frames",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise V280FeatureError(f"{name} must be a positive integer")
        if self.max_harmonic_order < 2:
            raise V280FeatureError("max_harmonic_order must be >= 2")
        for name in ("input_min_midi", "output_max_midi", "log_gain", "nyquist_margin"):
            if not math.isfinite(float(getattr(self, name))):
                raise V280FeatureError(f"{name} must be finite")
        if self.output_max_midi <= self.input_min_midi:
            raise V280FeatureError("output_max_midi must exceed input_min_midi")
        output_steps = (self.output_max_midi - self.input_min_midi) * self.bins_per_semitone
        if not math.isclose(output_steps, round(output_steps), abs_tol=1e-9):
            raise V280FeatureError("output pitch range must align to the CQT grid")
        if self.log_gain <= 0.0:
            raise V280FeatureError("log_gain must be positive")
        if not 0.5 <= self.nyquist_margin < 1.0:
            raise V280FeatureError("nyquist_margin must be in [0.5, 1)")
        if float(midi_to_hz(self.output_max_midi)) >= self.sample_rate / 2.0:
            raise V280FeatureError("output_max_midi must be below Nyquist")

    @property
    def bins_per_octave(self) -> int:
        return 12 * self.bins_per_semitone

    @property
    def q_factor(self) -> float:
        return 1.0 / (2.0 ** (1.0 / self.bins_per_octave) - 1.0)

    @property
    def input_min_hz(self) -> float:
        return float(midi_to_hz(self.input_min_midi))

    @property
    def input_max_hz(self) -> float:
        harmonic_limit = float(midi_to_hz(self.output_max_midi)) * self.max_harmonic_order
        return min(harmonic_limit, self.sample_rate * 0.5 * self.nyquist_margin)

    @property
    def output_bin_count(self) -> int:
        steps = int(round((self.output_max_midi - self.input_min_midi) * self.bins_per_semitone))
        return steps + 1

    @property
    def center_frequencies_hz(self) -> np.ndarray:
        ratio = self.input_max_hz / self.input_min_hz
        count = int(math.floor(self.bins_per_octave * math.log2(ratio) + 1e-10)) + 1
        index = np.arange(count, dtype=np.float64)
        return self.input_min_hz * np.power(2.0, index / self.bins_per_octave)

    @property
    def maximum_window_samples(self) -> int:
        return int(math.ceil(self.q_factor * self.sample_rate / self.input_min_hz))

    @property
    def maximum_window_seconds(self) -> float:
        return self.maximum_window_samples / float(self.sample_rate)

    def serializable(self) -> dict[str, int | float | str]:
        return {"implementation": IMPLEMENTATION, **asdict(self)}

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.serializable(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CausalCQT:
    """A complete track cache with explicit causal frame endpoints."""

    magnitude: np.ndarray
    frame_ends: np.ndarray
    frequencies_hz: np.ndarray
    window_lengths: np.ndarray
    sample_count: int
    config_sha256: str

    def __post_init__(self) -> None:
        magnitude = np.asarray(self.magnitude)
        frame_ends = np.asarray(self.frame_ends)
        frequencies = np.asarray(self.frequencies_hz)
        windows = np.asarray(self.window_lengths)
        if magnitude.ndim != 2:
            raise V280FeatureError("CQT magnitude must be a 2-D frame-by-frequency array")
        if frame_ends.shape != (magnitude.shape[0],):
            raise V280FeatureError("frame endpoint count does not match CQT frames")
        if frequencies.shape != windows.shape or frequencies.shape != (magnitude.shape[1],):
            raise V280FeatureError("frequency metadata does not match CQT bins")
        if isinstance(self.sample_count, bool) or int(self.sample_count) < 0:
            raise V280FeatureError("sample_count must be >= 0")
        if len(frame_ends) == 0 or frame_ends[0] != 0 or np.any(np.diff(frame_ends) <= 0):
            raise V280FeatureError("frame endpoints must start at zero and increase")
        if frame_ends[-1] > int(self.sample_count):
            raise V280FeatureError("frame endpoint exceeds available audio")
        if np.any(windows <= 0) or np.any(np.diff(frequencies) <= 0):
            raise V280FeatureError("invalid CQT frequency metadata")
        if not np.isfinite(magnitude).all() or np.any(magnitude < 0.0):
            raise V280FeatureError("CQT magnitude must be finite and non-negative")
        if not isinstance(self.config_sha256, str) or len(self.config_sha256) != 64:
            raise V280FeatureError("invalid feature configuration digest")


def _audio_float32(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim != 1:
        raise V280FeatureError("audio must be mono and one-dimensional")
    if array.dtype == np.int16:
        result = array.astype(np.float32) / 32768.0
    elif np.issubdtype(array.dtype, np.floating):
        result = array.astype(np.float32, copy=False)
    else:
        raise V280FeatureError("audio must use float samples or signed PCM16")
    if not np.isfinite(result).all():
        raise V280FeatureError("audio contains a non-finite sample")
    return np.ascontiguousarray(result)


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _octave_groups(config: CausalCQTConfig, frequencies: np.ndarray):
    for start in range(0, len(frequencies), config.bins_per_octave):
        stop = min(start + config.bins_per_octave, len(frequencies))
        lowest = float(frequencies[start])
        window_length = int(math.ceil(config.q_factor * config.sample_rate / lowest))
        yield start, stop, window_length, _next_power_of_two(window_length)


def _frame_endpoints(sample_count: int, hop_length: int) -> np.ndarray:
    return np.arange(0, int(sample_count) + 1, int(hop_length), dtype=np.int64)


def causal_cqt(
    samples: Sequence[float] | np.ndarray,
    config: CausalCQTConfig = CausalCQTConfig(),
) -> CausalCQT:
    """Compute a track-level CQT cache with no right padding or lookahead.

    Frame ``i`` uses exactly the half-open sample interval ending at
    ``frame_ends[i]``.  A frame is emitted only after that endpoint exists.
    """
    audio = _audio_float32(samples)
    frame_ends = _frame_endpoints(len(audio), config.hop_length)
    frequencies = config.center_frequencies_hz
    output = np.zeros((len(frame_ends), len(frequencies)), dtype=np.float32)
    window_lengths = np.zeros(len(frequencies), dtype=np.int32)

    for start, stop, window_length, fft_size in _octave_groups(config, frequencies):
        window_lengths[start:stop] = window_length
        taper = np.hanning(window_length).astype(np.float32)
        normalization = 2.0 / max(float(np.sum(taper, dtype=np.float64)), 1e-12)
        padded = np.pad(audio, (window_length, 0), mode="constant")
        track_windows = np.lib.stride_tricks.sliding_window_view(padded, window_length)
        group_frequencies = frequencies[start:stop]
        fft_positions = group_frequencies * fft_size / float(config.sample_rate)
        low = np.floor(fft_positions).astype(np.int64)
        high = low + 1
        alpha = (fft_positions - low).astype(np.float32)
        if np.any(high > fft_size // 2):
            raise V280FeatureError("CQT bin interpolation exceeds Nyquist")

        for block_start in range(0, len(frame_ends), config.block_frames):
            block_stop = min(block_start + config.block_frames, len(frame_ends))
            endpoints = frame_ends[block_start:block_stop]
            frames = np.asarray(track_windows[endpoints], dtype=np.float32) * taper[None, :]
            spectrum = np.abs(np.fft.rfft(frames, n=fft_size, axis=1)).astype(np.float32)
            interpolated = (
                spectrum[:, low] * (1.0 - alpha[None, :])
                + spectrum[:, high] * alpha[None, :]
            )
            output[block_start:block_stop, start:stop] = interpolated * normalization

    return CausalCQT(
        magnitude=output,
        frame_ends=frame_ends,
        frequencies_hz=frequencies.astype(np.float32),
        window_lengths=window_lengths,
        sample_count=len(audio),
        config_sha256=config.sha256,
    )


def cluster_feature_map(
    track: CausalCQT,
    cluster_start: int,
    config: CausalCQTConfig = CausalCQTConfig(),
    *,
    decision_end: int | None = None,
) -> np.ndarray:
    """Return ``log magnitude / positive baseline delta / positive flux``.

    The fixed-length crop ends at the last cached frame whose endpoint is not
    later than ``decision_end``.  Missing history at track start is left-zero
    padded.  The output is float32 with shape ``(cluster_frames, bins, 3)``.
    """
    if track.config_sha256 != config.sha256:
        raise V280FeatureError("track cache and feature configuration differ")
    if isinstance(cluster_start, bool) or not isinstance(cluster_start, int) or cluster_start < 0:
        raise V280FeatureError("cluster_start must be an integer >= 0")
    if decision_end is None:
        decision_end = cluster_start + config.cluster_post_samples
    if isinstance(decision_end, bool) or not isinstance(decision_end, int):
        raise V280FeatureError("decision_end must be an integer")
    if decision_end < cluster_start:
        raise V280FeatureError("decision_end precedes cluster_start")
    if decision_end > track.sample_count:
        raise V280FeatureError("decision_end exceeds available causal audio")

    last = int(np.searchsorted(track.frame_ends, decision_end, side="right") - 1)
    logical = np.arange(last - config.cluster_frames + 1, last + 1, dtype=np.int64)
    magnitude = np.zeros((config.cluster_frames, track.magnitude.shape[1]), dtype=np.float32)
    valid = logical >= 0
    magnitude[valid] = np.asarray(track.magnitude[logical[valid]], dtype=np.float32)
    crop_ends = logical * config.hop_length

    log_magnitude = np.log1p(config.log_gain * magnitude).astype(np.float32)
    pre = crop_ends <= cluster_start
    if not np.any(pre):
        raise V280FeatureError("cluster crop contains no pre-cluster frame")
    baseline = np.mean(log_magnitude[pre], axis=0, dtype=np.float32)
    positive_pre = np.maximum(log_magnitude - baseline[None, :], 0.0)
    flux = np.zeros_like(log_magnitude)
    flux[1:] = np.maximum(log_magnitude[1:] - log_magnitude[:-1], 0.0)
    result = np.stack((log_magnitude, positive_pre, flux), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise V280FeatureError("cluster feature map contains non-finite values")
    return result


def harmonic_ratios(max_harmonic_order: int = 8) -> tuple[float, ...]:
    """Return Harmonica's seven subharmonic then seven harmonic ratios."""
    if isinstance(max_harmonic_order, bool) or not isinstance(max_harmonic_order, int):
        raise V280FeatureError("max_harmonic_order must be an integer")
    if max_harmonic_order < 2:
        raise V280FeatureError("max_harmonic_order must be >= 2")
    return tuple(1.0 / order for order in range(2, max_harmonic_order + 1)) + tuple(
        float(order) for order in range(2, max_harmonic_order + 1)
    )


def harmonic_offsets(
    bins_per_octave: int = 12,
    max_harmonic_order: int = 8,
) -> np.ndarray:
    """Fractional frequency-bin offsets for shift-and-aggregate."""
    if isinstance(bins_per_octave, bool) or not isinstance(bins_per_octave, int) or bins_per_octave <= 0:
        raise V280FeatureError("bins_per_octave must be a positive integer")
    ratios = np.asarray(harmonic_ratios(max_harmonic_order), dtype=np.float64)
    return (bins_per_octave * np.log2(ratios)).astype(np.float32)


def _shift_frequency_axis(values: np.ndarray, offset: float) -> np.ndarray:
    """Gather ``output[..., f, :] = input[..., f + offset, :]`` linearly."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 2:
        raise V280FeatureError("harmonic features need frequency and channel axes")
    bins = source.shape[-2]
    coordinates = np.arange(bins, dtype=np.float32) + float(offset)
    low = np.floor(coordinates).astype(np.int64)
    high = low + 1
    alpha = coordinates - low
    valid_low = (low >= 0) & (low < bins)
    valid_high = (high >= 0) & (high < bins)
    low_clipped = np.clip(low, 0, max(bins - 1, 0))
    high_clipped = np.clip(high, 0, max(bins - 1, 0))
    low_value = np.take(source, low_clipped, axis=-2)
    high_value = np.take(source, high_clipped, axis=-2)
    broadcast = (1,) * (source.ndim - 2) + (bins, 1)
    low_weight = ((1.0 - alpha) * valid_low).reshape(broadcast)
    high_weight = (alpha * valid_high).reshape(broadcast)
    return (low_value * low_weight + high_value * high_weight).astype(np.float32)


def harmonic_shift_stack(
    values: np.ndarray,
    *,
    bins_per_octave: int = 12,
    max_harmonic_order: int = 8,
) -> np.ndarray:
    """Concatenate the 14 shifted copies used by harmonic convolution.

    The unshifted center is intentionally absent: the neural block supplies it
    through its residual connection, matching the parameter-efficient design.
    """
    source = np.asarray(values, dtype=np.float32)
    shifted = [
        _shift_frequency_axis(source, float(offset))
        for offset in harmonic_offsets(bins_per_octave, max_harmonic_order)
    ]
    return np.concatenate(shifted, axis=-1)


def harmonic_validity_mask(
    frequency_bins: int,
    *,
    bins_per_octave: int = 12,
    max_harmonic_order: int = 8,
) -> np.ndarray:
    """Expose which shifted harmonic positions are inside the feature map."""
    if isinstance(frequency_bins, bool) or not isinstance(frequency_bins, int) or frequency_bins <= 0:
        raise V280FeatureError("frequency_bins must be a positive integer")
    ones = np.ones((frequency_bins, 1), dtype=np.float32)
    return harmonic_shift_stack(
        ones,
        bins_per_octave=bins_per_octave,
        max_harmonic_order=max_harmonic_order,
    )


def write_track_cache(
    path: Path,
    track: CausalCQT,
    config: CausalCQTConfig = CausalCQTConfig(),
) -> None:
    """Atomically write one float16 track cache, refusing replacement."""
    path = Path(path)
    if path.suffix != ".npz":
        raise V280FeatureError("track cache path must end in .npz")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if track.config_sha256 != config.sha256:
        raise V280FeatureError("track cache and feature configuration differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                schema_version=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int16),
                implementation=np.asarray([IMPLEMENTATION]),
                config_json=np.asarray([json.dumps(config.serializable(), sort_keys=True)]),
                config_sha256=np.asarray([config.sha256]),
                magnitude=np.asarray(track.magnitude, dtype=np.float16),
                frame_ends=np.asarray(track.frame_ends, dtype=np.int64),
                frequencies_hz=np.asarray(track.frequencies_hz, dtype=np.float32),
                window_lengths=np.asarray(track.window_lengths, dtype=np.int32),
                sample_count=np.asarray([track.sample_count], dtype=np.int64),
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_track_cache(
    path: Path,
    config: CausalCQTConfig = CausalCQTConfig(),
) -> CausalCQT:
    """Load and validate a V28 track cache without pickle support."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "implementation",
            "config_json",
            "config_sha256",
            "magnitude",
            "frame_ends",
            "frequencies_hz",
            "window_lengths",
            "sample_count",
        }
        if set(data.files) != required:
            raise V280FeatureError("unexpected V28 track-cache schema")
        if int(data["schema_version"][0]) != CACHE_SCHEMA_VERSION:
            raise V280FeatureError("unsupported V28 track-cache version")
        if str(data["implementation"][0]) != IMPLEMENTATION:
            raise V280FeatureError("unexpected V28 feature implementation")
        saved_config = json.loads(str(data["config_json"][0]))
        if saved_config != config.serializable() or str(data["config_sha256"][0]) != config.sha256:
            raise V280FeatureError("V28 track-cache configuration mismatch")
        return CausalCQT(
            magnitude=np.asarray(data["magnitude"], dtype=np.float32),
            frame_ends=np.asarray(data["frame_ends"], dtype=np.int64),
            frequencies_hz=np.asarray(data["frequencies_hz"], dtype=np.float32),
            window_lengths=np.asarray(data["window_lengths"], dtype=np.int32),
            sample_count=int(data["sample_count"][0]),
            config_sha256=str(data["config_sha256"][0]),
        )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "IMPLEMENTATION",
    "CausalCQT",
    "CausalCQTConfig",
    "V280FeatureError",
    "causal_cqt",
    "cluster_feature_map",
    "harmonic_offsets",
    "harmonic_ratios",
    "harmonic_shift_stack",
    "harmonic_validity_mask",
    "midi_to_hz",
    "read_track_cache",
    "write_track_cache",
]
