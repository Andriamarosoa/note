"""Training-only harmonic state features for mining difficult onset negatives.

These features never enter NOTE's public API. They use annotated active-note
frequencies only to identify training positions where the waveform changes
strongly but the change is still dominated by harmonics of notes that are
already active. Those positions are exactly the failure mode exposed by the
V8.1 acoustic audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .guitarset import SAMPLE_RATE


@dataclass(frozen=True)
class HarmonicChange:
    positive_flux_over_pre_energy: float
    active_harmonic_flux_fraction: float
    rms_delta_db: float

    def __post_init__(self) -> None:
        for name, value in (
            ("positive_flux_over_pre_energy", self.positive_flux_over_pre_energy),
            ("active_harmonic_flux_fraction", self.active_harmonic_flux_fraction),
            ("rms_delta_db", self.rms_delta_db),
        ):
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
        if self.positive_flux_over_pre_energy < 0.0:
            raise ValueError("positive_flux_over_pre_energy must be >= 0")
        if not 0.0 <= self.active_harmonic_flux_fraction <= 1.0:
            raise ValueError("active_harmonic_flux_fraction must be in [0, 1]")

    @property
    def hardness(self) -> float:
        """Rank large changes that are still explained by active harmonics."""
        return self.positive_flux_over_pre_energy * self.active_harmonic_flux_fraction


def _pcm_window(np, samples: Sequence[int], start: int, length: int):
    result = np.zeros(length, dtype=np.float64)
    source_start = max(int(start), 0)
    source_end = min(int(start) + int(length), len(samples))
    if source_end > source_start:
        destination = source_start - int(start)
        result[destination : destination + source_end - source_start] = (
            np.asarray(samples[source_start:source_end], dtype=np.float64) / 32768.0
        )
    return result


def _harmonic_mask(np, frequencies_hz: Iterable[float], *, fft_size: int, max_hz: float):
    max_bin = fft_size // 2 + 1
    mask = np.zeros(max_bin, dtype=bool)
    resolution = SAMPLE_RATE / float(fft_size)
    for raw_frequency in frequencies_hz:
        f0 = float(raw_frequency)
        if not math.isfinite(f0) or f0 <= 0.0:
            continue
        harmonic = 1
        while harmonic * f0 <= max_hz:
            frequency = harmonic * f0
            bandwidth_hz = max(12.0, frequency * 0.008)
            low = max(0, int(math.floor((frequency - bandwidth_hz) / resolution)))
            high = min(max_bin, int(math.ceil((frequency + bandwidth_hz) / resolution)) + 1)
            if high > low:
                mask[low:high] = True
            harmonic += 1
    return mask


def harmonic_change(
    np,
    samples: Sequence[int],
    sample: int,
    active_frequencies_hz: Iterable[float],
    *,
    window_samples: int = 1024,
    fft_size: int = 4096,
    max_hz: float = 8000.0,
) -> HarmonicChange:
    """Measure spectral change across ``sample`` and how much active notes explain it."""
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise ValueError("sample must be an integer >= 0")
    if window_samples <= 0 or fft_size < window_samples:
        raise ValueError("fft_size must be >= positive window_samples")
    if not math.isfinite(float(max_hz)) or not 0.0 < float(max_hz) <= SAMPLE_RATE / 2:
        raise ValueError("max_hz must be in (0, Nyquist]")

    pre = _pcm_window(np, samples, sample - window_samples, window_samples)
    post = _pcm_window(np, samples, sample, window_samples)
    taper = np.hanning(window_samples)
    pre_power = np.abs(np.fft.rfft(pre * taper, n=fft_size)) ** 2
    post_power = np.abs(np.fft.rfft(post * taper, n=fft_size)) ** 2
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / SAMPLE_RATE)
    valid = (frequencies >= 50.0) & (frequencies <= float(max_hz))

    flux = np.maximum(post_power - pre_power, 0.0)
    flux[~valid] = 0.0
    total_flux = float(flux.sum())
    pre_energy = float(pre_power[valid].sum()) + 1e-20
    active_mask = _harmonic_mask(
        np,
        active_frequencies_hz,
        fft_size=fft_size,
        max_hz=float(max_hz),
    ) & valid
    explained = float(flux[active_mask].sum()) if total_flux > 0.0 else 0.0
    pre_rms = float(np.sqrt(np.mean(pre * pre)) + 1e-12)
    post_rms = float(np.sqrt(np.mean(post * post)) + 1e-12)
    return HarmonicChange(
        positive_flux_over_pre_energy=total_flux / pre_energy,
        active_harmonic_flux_fraction=explained / total_flux if total_flux > 0.0 else 0.0,
        rms_delta_db=20.0 * math.log10(post_rms / pre_rms),
    )


__all__ = ["HarmonicChange", "harmonic_change"]
