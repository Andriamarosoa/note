#!/usr/bin/env python3
"""Adaptive causal onset detector for mono streaming audio."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

EPSILON = 1e-12


@dataclass(frozen=True)
class OnsetResult:
    is_onset: bool
    tick_index: int
    time_s: float
    rms: float
    rms_dbfs: float
    rms_growth: float
    spectral_flux: float
    rms_threshold: float
    growth_threshold: float
    flux_threshold: float
    score: float
    confidence: float
    calibrated: bool
    cooldown_remaining: int
    armed: bool


class RunningRobustStats:
    def __init__(self, capacity: int) -> None:
        if capacity < 4:
            raise ValueError("capacity must be >= 4")
        self.capacity = int(capacity)
        self.values = np.zeros(self.capacity, dtype=np.float64)
        self.count = 0
        self.index = 0

    def reset(self) -> None:
        self.values.fill(0.0)
        self.count = 0
        self.index = 0

    def add(self, value: float) -> None:
        self.values[self.index] = float(value)
        self.index = (self.index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def snapshot(self) -> np.ndarray:
        return self.values[: self.count]

    def median(self, default: float = 0.0) -> float:
        if self.count == 0:
            return float(default)
        return float(np.median(self.snapshot()))

    def mad(self, default: float = 0.0) -> float:
        if self.count < 2:
            return float(default)
        values = self.snapshot()
        center = np.median(values)
        return float(np.median(np.abs(values - center)))

    def robust_sigma(self, minimum: float = 0.0) -> float:
        return max(float(minimum), 1.4826 * self.mad())


class AdaptiveOnsetDetector:
    """Hop-by-hop mono onset detector with adaptive thresholds and hysteresis."""

    def __init__(
        self,
        sample_rate: int,
        hop_samples: int,
        fft_size: int = 512,
        calibration_s: float = 1.0,
        history_s: float = 2.0,
        cooldown_ms: float = 80.0,
        rearm_ratio: float = 1.35,
        rms_sigma: float = 5.0,
        growth_sigma: float = 4.0,
        flux_sigma: float = 4.0,
        minimum_rms_dbfs: float = -72.0,
        trigger_score: float = 2.2,
        require_joint_temporal_evidence: bool = False,
        enable_peak_rearm: bool = False,
        robust_rearm: bool = False,
        rearm_stable_hops: int = 3,
        rearm_attack_ratio: float = 1.20,
        rearm_flux_ratio: float = 2.0,
        rearm_growth_ratio: float = 8.0,
    ) -> None:
        if sample_rate <= 0 or hop_samples <= 0:
            raise ValueError("sample_rate and hop_samples must be > 0")
        if fft_size < max(128, hop_samples):
            raise ValueError("fft_size must be >= max(128, hop_samples)")
        if rearm_ratio <= 1.0:
            raise ValueError("rearm_ratio must be > 1")
        if rearm_stable_hops < 1:
            raise ValueError("rearm_stable_hops must be positive")
        if rearm_attack_ratio <= 1.0:
            raise ValueError("rearm_attack_ratio must be > 1")
        if rearm_flux_ratio <= 1.0:
            raise ValueError("rearm_flux_ratio must be > 1")
        if rearm_growth_ratio <= 1.0:
            raise ValueError("rearm_growth_ratio must be > 1")

        self.sample_rate = int(sample_rate)
        self.hop_samples = int(hop_samples)
        self.fft_size = int(fft_size)

        hops_per_second = self.sample_rate / float(self.hop_samples)
        self.calibration_hops = max(4, int(round(calibration_s * hops_per_second)))
        history_hops = max(self.calibration_hops, int(round(history_s * hops_per_second)))
        self.cooldown_hops = max(1, int(round(cooldown_ms / 1000.0 * hops_per_second)))

        self.rearm_ratio = float(rearm_ratio)
        self.rms_sigma = float(rms_sigma)
        self.growth_sigma = float(growth_sigma)
        self.flux_sigma = float(flux_sigma)
        self.minimum_rms = 10.0 ** (float(minimum_rms_dbfs) / 20.0)
        self.trigger_score = float(trigger_score)
        self.require_joint_temporal_evidence = bool(
            require_joint_temporal_evidence
        )
        self.enable_peak_rearm = bool(enable_peak_rearm)
        self.robust_rearm = bool(robust_rearm)
        self.rearm_stable_hops = int(rearm_stable_hops)
        self.rearm_attack_ratio = float(rearm_attack_ratio)
        self.rearm_flux_ratio = float(rearm_flux_ratio)
        self.rearm_growth_ratio = float(rearm_growth_ratio)

        self.rms_stats = RunningRobustStats(history_hops)
        self.growth_stats = RunningRobustStats(history_hops)
        self.flux_stats = RunningRobustStats(history_hops)

        self.analysis_buffer = np.zeros(self.fft_size, dtype=np.float64)
        self.window = np.hanning(self.fft_size).astype(np.float64)
        self.previous_spectrum = np.zeros(self.fft_size // 2 + 1, dtype=np.float64)

        self.previous_rms = 0.0
        self.tick_index = 0
        self.cooldown_remaining = 0
        self.armed = True
        self.unarmed_peak_rms = 0.0
        self.unarmed_peak_flux = 0.0
        self.rearm_stable_count = 0
        self.rearm_reference_rms = 0.0
        self.last_rearmed_this_hop = False
        self._has_previous_spectrum = False
        self._continuity_suppression_remaining = 0

    @property
    def calibrated(self) -> bool:
        return self.rms_stats.count >= self.calibration_hops

    def reset(self) -> None:
        self.rms_stats.reset()
        self.growth_stats.reset()
        self.flux_stats.reset()
        self.analysis_buffer.fill(0.0)
        self.previous_spectrum.fill(0.0)
        self.previous_rms = 0.0
        self.tick_index = 0
        self.cooldown_remaining = 0
        self.armed = True
        self.unarmed_peak_rms = 0.0
        self.unarmed_peak_flux = 0.0
        self.rearm_stable_count = 0
        self.rearm_reference_rms = 0.0
        self.last_rearmed_this_hop = False
        self._has_previous_spectrum = False
        self._continuity_suppression_remaining = 0

    def reset_continuity(self) -> None:
        """Forget pre-gap temporal state without losing noise calibration."""
        self.analysis_buffer.fill(0.0)
        self.previous_spectrum.fill(0.0)
        self.previous_rms = self.rms_stats.median(0.0)
        self.cooldown_remaining = 0
        self.armed = True
        self.unarmed_peak_rms = 0.0
        self.unarmed_peak_flux = 0.0
        self.rearm_stable_count = 0
        self.rearm_reference_rms = 0.0
        self.last_rearmed_this_hop = False
        self._has_previous_spectrum = False
        self._continuity_suppression_remaining = int(np.ceil(
            self.fft_size / float(self.hop_samples)
        ))

    def _append_hop(self, hop: np.ndarray) -> np.ndarray:
        samples = np.asarray(hop, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            raise ValueError("hop must not be empty")
        if samples.size >= self.fft_size:
            self.analysis_buffer[:] = samples[-self.fft_size :]
        else:
            shift = int(samples.size)
            self.analysis_buffer[:-shift] = self.analysis_buffer[shift:]
            self.analysis_buffer[-shift:] = samples
        return samples

    def _spectral_flux(self) -> float:
        frame = self.analysis_buffer - float(np.mean(self.analysis_buffer))
        magnitude = np.abs(np.fft.rfft(frame * self.window))
        magnitude /= max(float(np.sum(self.window)), EPSILON)
        compressed = np.log1p(1000.0 * magnitude)

        if not self._has_previous_spectrum:
            flux = 0.0
            self._has_previous_spectrum = True
        else:
            diff = np.maximum(compressed - self.previous_spectrum, 0.0)
            flux = float(np.sqrt(np.mean(diff * diff)))

        self.previous_spectrum[:] = compressed
        return flux

    @staticmethod
    def _normalized_excess(value: float, threshold: float, scale: float) -> float:
        if value <= threshold:
            return 0.0
        return (value - threshold) / max(scale, EPSILON)

    def process(self, hop: np.ndarray) -> OnsetResult:
        samples = self._append_hop(hop)
        self.tick_index += 1

        rms = float(np.sqrt(np.mean(samples * samples) + EPSILON))
        rms_dbfs = float(20.0 * np.log10(max(rms, EPSILON)))
        rms_growth = max(0.0, rms - self.previous_rms)
        spectral_flux = self._spectral_flux()

        rms_median = self.rms_stats.median(self.minimum_rms)
        growth_median = self.growth_stats.median(0.0)
        flux_median = self.flux_stats.median(0.0)

        rms_scale = self.rms_stats.robust_sigma(max(self.minimum_rms * 0.25, 1e-7))
        growth_scale = self.growth_stats.robust_sigma(max(self.minimum_rms * 0.10, 1e-8))
        flux_scale = self.flux_stats.robust_sigma(1e-4)

        rms_threshold = max(self.minimum_rms, rms_median + self.rms_sigma * rms_scale)
        growth_threshold = growth_median + self.growth_sigma * growth_scale
        flux_threshold = flux_median + self.flux_sigma * flux_scale

        was_unarmed = not self.armed
        rearmed_this_hop = False
        calm_temporal_hop = False
        if was_unarmed:
            self.unarmed_peak_rms = max(self.unarmed_peak_rms, rms)
            self.unarmed_peak_flux = max(
                self.unarmed_peak_flux,
                spectral_flux,
            )
            release_level = self.unarmed_peak_rms / self.rearm_ratio
            calm_flux_threshold = max(
                flux_threshold,
                self.unarmed_peak_flux / self.rearm_flux_ratio,
            )
            calm_growth_threshold = max(
                growth_threshold,
                self.unarmed_peak_rms / self.rearm_growth_ratio,
            )
            calm_temporal_hop = (
                rms_growth <= calm_growth_threshold
                and spectral_flux <= calm_flux_threshold
            )
            if calm_temporal_hop:
                self.rearm_stable_count += 1
            else:
                self.rearm_stable_count = 0
            noise_floor_release = rms <= rms_threshold * self.rearm_ratio
            peak_valley_release = (
                self.enable_peak_rearm
                and rms <= release_level
                and rms <= self.previous_rms
            )
            stable_release = (
                self.robust_rearm
                and self.rearm_stable_count >= self.rearm_stable_hops
            )
            if self.cooldown_remaining == 0 and (
                noise_floor_release
                or peak_valley_release
                or stable_release
            ):
                self.armed = True
                self.unarmed_peak_rms = 0.0
                self.unarmed_peak_flux = 0.0
                self.rearm_stable_count = 0
                self.rearm_reference_rms = rms
                rearmed_this_hop = True
        elif self.robust_rearm and self.rearm_reference_rms > 0.0:
            self.rearm_reference_rms = min(
                self.rearm_reference_rms,
                rms,
            )
        self.last_rearmed_this_hop = rearmed_this_hop

        rms_excess = self._normalized_excess(rms, rms_threshold, rms_scale)
        growth_excess = self._normalized_excess(rms_growth, growth_threshold, growth_scale)
        flux_excess = self._normalized_excess(spectral_flux, flux_threshold, flux_scale)

        score = (
            0.75 * min(rms_excess, 6.0)
            + 0.65 * min(growth_excess, 6.0)
            + 0.85 * min(flux_excess, 6.0)
        )

        temporal_evidence = (
            growth_excess > 0.0 and flux_excess > 0.0
            if self.require_joint_temporal_evidence
            else growth_excess > 0.0 or flux_excess > 0.0
        )
        suppress_discontinuous_onset = (
            self._continuity_suppression_remaining > 0
        )
        if self._continuity_suppression_remaining > 0:
            self._continuity_suppression_remaining -= 1
        robust_attack_ready = (
            not self.robust_rearm
            or self.rearm_reference_rms <= EPSILON
            or rms >= self.rearm_reference_rms * self.rearm_attack_ratio
        )
        is_onset = (
            not suppress_discontinuous_onset
            and not (self.robust_rearm and rearmed_this_hop)
            and self.calibrated
            and self.armed
            and self.cooldown_remaining == 0
            and rms > rms_threshold
            and robust_attack_ready
            and temporal_evidence
            and score >= self.trigger_score
        )

        if is_onset:
            self.cooldown_remaining = self.cooldown_hops
            self.armed = False
            self.unarmed_peak_rms = rms
            self.unarmed_peak_flux = spectral_flux
            self.rearm_stable_count = 0
            self.rearm_reference_rms = 0.0
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        update_background = not is_onset and (
            not self.calibrated
            or (
                self.armed
                and rms <= rms_threshold * 1.25
                and spectral_flux <= flux_threshold * 1.25
            )
        )
        if update_background:
            self.rms_stats.add(rms)
            self.growth_stats.add(rms_growth)
            self.flux_stats.add(spectral_flux)
        self.previous_rms = rms
        confidence = float(np.clip(score / max(self.trigger_score * 2.0, EPSILON), 0.0, 1.0))
        time_s = self.tick_index * self.hop_samples / float(self.sample_rate)

        return OnsetResult(
            is_onset=is_onset,
            tick_index=self.tick_index,
            time_s=time_s,
            rms=rms,
            rms_dbfs=rms_dbfs,
            rms_growth=rms_growth,
            spectral_flux=spectral_flux,
            rms_threshold=rms_threshold,
            growth_threshold=growth_threshold,
            flux_threshold=flux_threshold,
            score=float(score),
            confidence=confidence,
            calibrated=self.calibrated,
            cooldown_remaining=self.cooldown_remaining,
            armed=self.armed,
        )
