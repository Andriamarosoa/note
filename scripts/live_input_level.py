"""Causal NOTE model-input level control, adapted from midi/src/polyphonic/input_level.py.

Only the inference audio is amplified. No per-note tail normalization.
The gain controller is bounded by the original midi default of +18 dB,
plus real-window headroom. The output is intentionally experimental: it
does not repair missing frequencies or recover near-silent PCM16 recordings.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class InputLevelResult:
    gain: float
    gain_db: float
    session_gain_db: float
    safety_gain_db: float
    observed_peak_dbfs: float
    observed_window_peak: float
    projected_model_peak: float
    initialized: bool


class CausalModelInputLeveler:
    """Match microphone sensitivity without normalizing every guitar tail.

    The controller observes only audible raw/capture hops.  It reduces gain
    immediately when a louder peak arrives, but can increase it by at most a
    small number of dB per second.  A decaying note therefore cannot be held at
    a constant level and mistaken for a fresh attack.  The selected scalar is
    applied uniformly to the complete causal model window by the caller; the
    audio activity gate and onset detector continue to see unlevelled audio.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_samples: int,
        model_normalization_gain: float,
        *,
        input_samples: int = 4096,
        target_capture_peak_dbfs: float = -12.0,
        minimum_gain_db: float = 0.0,
        maximum_gain_db: float = 18.0,
        recovery_db_per_second: float = 1.0,
        model_headroom_peak: float = 0.98,
    ) -> None:
        if sample_rate <= 0 or hop_samples <= 0:
            raise ValueError("sample_rate and hop_samples must be positive")
        if model_normalization_gain <= 0.0:
            raise ValueError("model_normalization_gain must be positive")
        if input_samples < 1:
            raise ValueError("input_samples must be positive")
        if minimum_gain_db > maximum_gain_db:
            raise ValueError("minimum_gain_db must not exceed maximum_gain_db")
        if recovery_db_per_second <= 0.0:
            raise ValueError("recovery_db_per_second must be positive")
        if not 0.0 < model_headroom_peak < 1.0:
            raise ValueError("model_headroom_peak must be between zero and one")
        self.sample_rate = int(sample_rate)
        self.hop_samples = int(hop_samples)
        self.input_samples = int(input_samples)
        self.window_hops = int(np.ceil(
            self.input_samples / float(self.hop_samples)
        ))
        self.model_normalization_gain = float(model_normalization_gain)
        self.target_capture_peak_dbfs = float(target_capture_peak_dbfs)
        self.minimum_gain_db = float(minimum_gain_db)
        self.maximum_gain_db = float(maximum_gain_db)
        self.recovery_db_per_second = float(recovery_db_per_second)
        self.model_headroom_peak = float(model_headroom_peak)
        self.maximum_recovery_db_per_hop = (
            self.recovery_db_per_second
            * self.hop_samples
            / float(self.sample_rate)
        )
        self.session_gain_db = 0.0
        self.current_gain_db = 0.0
        self.current_safety_gain_db = float("inf")
        self.minimum_applied_gain_db: float | None = None
        self.maximum_applied_gain_db: float | None = None
        self.initialized = False
        self.audible_hops_seen = 0
        self.inactive_hops = 0
        self.hop_index = -1
        self.window_peaks: deque[tuple[int, float]] = deque()

    @classmethod
    def from_metadata(
        cls,
        sample_rate: int,
        hop_samples: int,
        model_normalization_gain: float,
        input_samples: int,
        metadata: Mapping[str, object],
    ) -> "CausalModelInputLeveler":
        config = metadata.get("automatic_model_input_level", {})
        values = config if isinstance(config, Mapping) else {}
        return cls(
            sample_rate,
            hop_samples,
            model_normalization_gain,
            input_samples=input_samples,
            target_capture_peak_dbfs=float(
                values.get("target_capture_peak_dbfs", -12.0)
            ),
            minimum_gain_db=float(
                values.get("minimum_gain_db", 0.0)
            ),
            maximum_gain_db=float(
                values.get("maximum_gain_db", 18.0)
            ),
            recovery_db_per_second=float(
                values.get("recovery_db_per_second", 1.0)
            ),
            model_headroom_peak=float(
                values.get("model_headroom_peak", 0.98)
            ),
        )

    @staticmethod
    def _dbfs(value: float) -> float:
        return float(20.0 * np.log10(max(float(value), EPSILON)))

    @property
    def gain(self) -> float:
        return float(10.0 ** (self.current_gain_db / 20.0))

    def process(self, hop: np.ndarray, *, audio_active: bool) -> InputLevelResult:
        samples = np.asarray(hop, dtype=np.float32).reshape(-1)
        if samples.shape != (self.hop_samples,):
            raise ValueError(f"Expected one audio hop of {self.hop_samples} samples.")
        peak = float(np.max(np.abs(samples), initial=0.0))
        self.hop_index += 1
        oldest = self.hop_index - self.window_hops + 1
        while self.window_peaks and self.window_peaks[0][0] < oldest:
            self.window_peaks.popleft()
        while self.window_peaks and self.window_peaks[-1][1] <= peak:
            self.window_peaks.pop()
        self.window_peaks.append((self.hop_index, peak))
        window_peak = self.window_peaks[0][1]
        peak_dbfs = self._dbfs(peak)
        if audio_active and peak > EPSILON:
            self.audible_hops_seen += 1
            self.inactive_hops = 0
            desired_gain_db = float(np.clip(
                self.target_capture_peak_dbfs - peak_dbfs,
                self.minimum_gain_db,
                self.maximum_gain_db,
            ))
            if not self.initialized:
                self.session_gain_db = desired_gain_db
                self.initialized = True
            elif desired_gain_db < self.session_gain_db:
                # Protect headroom immediately when the player strikes harder.
                self.session_gain_db = desired_gain_db
            # Never raise gain on an active tail.  Recovery is reserved for a
            # stable closed gate so it prepares the next note, not the decay.
        else:
            self.inactive_hops += 1
            if (
                self.initialized
                and self.inactive_hops >= self.window_hops
            ):
                self.session_gain_db = min(
                    self.maximum_gain_db,
                    self.session_gain_db
                    + self.maximum_recovery_db_per_hop,
                )

        if window_peak > EPSILON:
            maximum_safe_gain = (
                self.model_headroom_peak
                / (window_peak * self.model_normalization_gain)
            )
            self.current_safety_gain_db = self._dbfs(
                maximum_safe_gain
            )
        else:
            self.current_safety_gain_db = float("inf")
        self.current_gain_db = min(
            self.session_gain_db,
            self.current_safety_gain_db,
        )
        if self.initialized:
            if self.minimum_applied_gain_db is None:
                self.minimum_applied_gain_db = self.current_gain_db
                self.maximum_applied_gain_db = self.current_gain_db
            else:
                self.minimum_applied_gain_db = min(
                    self.minimum_applied_gain_db, self.current_gain_db
                )
                self.maximum_applied_gain_db = max(
                    float(self.maximum_applied_gain_db),
                    self.current_gain_db,
                )
        projected_model_peak = (
            window_peak * self.gain * self.model_normalization_gain
        )
        return InputLevelResult(
            gain=self.gain,
            gain_db=float(self.current_gain_db),
            session_gain_db=float(self.session_gain_db),
            safety_gain_db=float(self.current_safety_gain_db),
            observed_peak_dbfs=peak_dbfs,
            observed_window_peak=float(window_peak),
            projected_model_peak=float(projected_model_peak),
            initialized=bool(self.initialized),
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "enabled": True,
            "initialized": bool(self.initialized),
            "audible_hops_seen": int(self.audible_hops_seen),
            "gain": self.gain,
            "gain_db": float(self.current_gain_db),
            "session_gain_db": float(self.session_gain_db),
            "safety_gain_db": (
                None
                if not np.isfinite(self.current_safety_gain_db)
                else float(self.current_safety_gain_db)
            ),
            "minimum_applied_gain_db": (
                None
                if self.minimum_applied_gain_db is None
                else float(self.minimum_applied_gain_db)
            ),
            "maximum_applied_gain_db": (
                None
                if self.maximum_applied_gain_db is None
                else float(self.maximum_applied_gain_db)
            ),
            "target_capture_peak_dbfs": self.target_capture_peak_dbfs,
            "input_samples": self.input_samples,
            "window_hops": self.window_hops,
            "inactive_hops": self.inactive_hops,
            "minimum_gain_db": self.minimum_gain_db,
            "maximum_gain_db": self.maximum_gain_db,
            "recovery_db_per_second": self.recovery_db_per_second,
            "model_headroom_peak": self.model_headroom_peak,
            "model_normalization_gain": self.model_normalization_gain,
        }



class NoteAutoLevel:
    """Live/WAV adapter for NOTE's variable 128/256/512/1024-sample blocks.

    Activity is estimated from *raw* microphone samples, with conservative
    fixed peak+RMS floors. Unlike midi, NOTE strict has no separate calibrated
    activity gate, so this is an intentionally simpler, optional control.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        max_gain_db: float = 18.0,
        peak_gate_dbfs: float = -55.0,
        rms_gate_dbfs: float = -65.0,
        target_peak_dbfs: float = -12.0,
    ):
        if not np.isfinite(max_gain_db) or not 0 <= max_gain_db <= 24:
            raise ValueError("max_gain_db must be finite and within [0,24]")
        if not np.isfinite(peak_gate_dbfs) or not -120 <= peak_gate_dbfs <= 0:
            raise ValueError("Invalid peak_gate_dbfs")
        if not np.isfinite(rms_gate_dbfs) or not -120 <= rms_gate_dbfs <= 0:
            raise ValueError("Invalid rms_gate_dbfs")
        self.peak_gate = 10.0 ** (peak_gate_dbfs / 20.0)
        self.rms_gate = 10.0 ** (rms_gate_dbfs / 20.0)
        self.leveler = CausalModelInputLeveler(
            sample_rate=sample_rate,
            hop_samples=block_size,
            model_normalization_gain=1.0,
            input_samples=8192,
            target_capture_peak_dbfs=target_peak_dbfs,
            minimum_gain_db=0.0,
            maximum_gain_db=max_gain_db,
            recovery_db_per_second=1.0,
            model_headroom_peak=0.98,
        )
        self.total_blocks = 0
        self.active_blocks = 0
        self.max_raw_peak = 0.0
        self.over_limit_input_samples = 0
        self.latest = None

    def apply(self, raw):
        x = np.asarray(raw, dtype=np.float32)
        if x.ndim != 1 or not (0 < x.shape[0] <= self.leveler.hop_samples):
            raise ValueError("Expected one contiguous capture block at most one hop long")
        if not np.isfinite(x).all():
            raise ValueError("Non-finite microphone audio")
        # A WAV/stdin file can end with a partial block; pad ONLY the leveler's
        # observable current chunk and return only the original real samples.
        real_samples = x.shape[0]
        if real_samples < self.leveler.hop_samples:
            x = np.pad(x, (0, self.leveler.hop_samples - real_samples))
        peak = float(np.max(np.abs(x), initial=0.0))
        rms = float(np.sqrt(np.mean(x[:real_samples].astype(np.float64) ** 2)))
        active = peak >= self.peak_gate and rms >= self.rms_gate
        result = self.leveler.process(x, audio_active=active)
        self.latest = result
        self.total_blocks += 1
        self.active_blocks += int(active)
        self.max_raw_peak = max(self.max_raw_peak, peak)
        scaled = x * result.gain
        self.over_limit_input_samples += int(np.count_nonzero(np.abs(scaled) > 1.0))
        return np.clip(scaled[:real_samples], -1.0, 1.0).astype(np.float32)

    def diagnostics(self):
        return {
            **self.leveler.diagnostics(),
            "blocks": self.total_blocks,
            "gate_active_blocks": self.active_blocks,
            "raw_max_peak": self.max_raw_peak,
            "peak_gate_dbfs": float(20 * np.log10(self.peak_gate)),
            "rms_gate_dbfs": float(20 * np.log10(self.rms_gate)),
            "clipped_after_gain": self.over_limit_input_samples,
        }
