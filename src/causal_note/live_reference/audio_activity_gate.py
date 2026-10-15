"""Session-calibrated causal audio activity gate for live guitar input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class AudioActivityResult:
    active: bool
    calibrated: bool
    calibration_stable: bool | None
    rms_dbfs: float
    open_threshold_dbfs: float | None
    close_threshold_dbfs: float | None


class CalibratedAudioActivityGate:
    """Freeze a robust noise gate after a stable, silence-only calibration.

    The onset detector deliberately keeps adapting its attack thresholds.  A
    sustain/activity decision must not reuse those mutable thresholds because
    they can learn a long guitar passage as background and veto valid model
    predictions.  This gate estimates two RMS thresholds once per session and
    then uses only current-hop evidence with hysteresis.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_samples: int,
        *,
        calibration_s: float = 1.0,
        calibration_tail_s: float = 0.4,
        maximum_calibration_s: float = 3.0,
        maximum_tail_spread_db: float = 8.0,
        open_quantile: float = 0.99,
        close_quantile: float = 0.90,
        minimum_open_dbfs: float = -72.0,
        minimum_close_dbfs: float = -75.0,
        minimum_hysteresis_db: float = 0.75,
    ) -> None:
        if sample_rate <= 0 or hop_samples <= 0:
            raise ValueError("sample_rate and hop_samples must be positive")
        if calibration_s <= 0 or calibration_tail_s <= 0:
            raise ValueError("calibration durations must be positive")
        if maximum_calibration_s < calibration_s:
            raise ValueError("maximum calibration must cover minimum calibration")
        if not 0.0 < close_quantile < open_quantile < 1.0:
            raise ValueError("activity quantiles must satisfy 0 < close < open < 1")
        if maximum_tail_spread_db <= 0 or minimum_hysteresis_db <= 0:
            raise ValueError("spread and hysteresis must be positive")
        if minimum_close_dbfs >= minimum_open_dbfs:
            raise ValueError("minimum close threshold must be below open threshold")

        hops_per_second = float(sample_rate) / float(hop_samples)
        self.minimum_calibration_hops = max(
            4, int(round(calibration_s * hops_per_second))
        )
        self.tail_hops = max(
            4,
            min(
                self.minimum_calibration_hops,
                int(round(calibration_tail_s * hops_per_second)),
            ),
        )
        self.maximum_calibration_hops = max(
            self.minimum_calibration_hops,
            int(round(maximum_calibration_s * hops_per_second)),
        )
        self.sample_rate = int(sample_rate)
        self.hop_samples = int(hop_samples)
        self.maximum_tail_spread_db = float(maximum_tail_spread_db)
        self.open_quantile = float(open_quantile)
        self.close_quantile = float(close_quantile)
        self.minimum_open_dbfs = float(minimum_open_dbfs)
        self.minimum_close_dbfs = float(minimum_close_dbfs)
        self.minimum_hysteresis_db = float(minimum_hysteresis_db)

        self._calibration_dbfs: list[float] = []
        self._calibrated = False
        self._calibration_stable: bool | None = None
        self._tail_spread_db: float | None = None
        self._open_threshold_dbfs: float | None = None
        self._close_threshold_dbfs: float | None = None
        self._active = False

    @staticmethod
    def rms_to_dbfs(rms: float) -> float:
        value = float(rms)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("rms must be a finite non-negative value")
        return float(20.0 * np.log10(max(value, EPSILON)))

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def active(self) -> bool:
        return self._active

    @property
    def open_threshold_dbfs(self) -> float | None:
        return self._open_threshold_dbfs

    @property
    def close_threshold_dbfs(self) -> float | None:
        return self._close_threshold_dbfs

    def _finish_calibration(self, stable: bool) -> None:
        tail = np.asarray(self._calibration_dbfs[-self.tail_hops :], np.float64)
        open_threshold = max(
            self.minimum_open_dbfs,
            float(np.quantile(tail, self.open_quantile)),
        )
        close_threshold = max(
            self.minimum_close_dbfs,
            float(np.quantile(tail, self.close_quantile)),
        )
        close_threshold = min(
            close_threshold,
            open_threshold - self.minimum_hysteresis_db,
        )
        self._open_threshold_dbfs = open_threshold
        self._close_threshold_dbfs = close_threshold
        self._calibration_stable = bool(stable)
        self._calibrated = True
        self._active = False

    def process_rms(self, rms: float) -> AudioActivityResult:
        rms_dbfs = self.rms_to_dbfs(rms)
        if not self._calibrated:
            self._calibration_dbfs.append(rms_dbfs)
            count = len(self._calibration_dbfs)
            if count >= self.minimum_calibration_hops:
                tail = np.asarray(
                    self._calibration_dbfs[-self.tail_hops :], np.float64
                )
                self._tail_spread_db = float(
                    np.quantile(tail, 0.95) - np.quantile(tail, 0.05)
                )
                stable = self._tail_spread_db <= self.maximum_tail_spread_db
                forced = count >= self.maximum_calibration_hops
                if stable or forced:
                    self._finish_calibration(stable)
        else:
            assert self._open_threshold_dbfs is not None
            assert self._close_threshold_dbfs is not None
            if self._active:
                if rms_dbfs < self._close_threshold_dbfs:
                    self._active = False
            elif rms_dbfs >= self._open_threshold_dbfs:
                self._active = True

        return AudioActivityResult(
            active=self._active,
            calibrated=self._calibrated,
            calibration_stable=self._calibration_stable,
            rms_dbfs=rms_dbfs,
            open_threshold_dbfs=self._open_threshold_dbfs,
            close_threshold_dbfs=self._close_threshold_dbfs,
        )

    def reset_continuity(self) -> None:
        """Close the gate after an audio gap without losing calibration."""
        self._active = False

    def reset(self) -> None:
        self._calibration_dbfs.clear()
        self._calibrated = False
        self._calibration_stable = None
        self._tail_spread_db = None
        self._open_threshold_dbfs = None
        self._close_threshold_dbfs = None
        self._active = False

    def diagnostics(self) -> dict[str, object]:
        return {
            "calibrated": self._calibrated,
            "calibration_stable": self._calibration_stable,
            "minimum_calibration_hops": self.minimum_calibration_hops,
            "maximum_calibration_hops": self.maximum_calibration_hops,
            "calibration_hops_seen": len(self._calibration_dbfs),
            "calibration_extended_hops": max(
                0, len(self._calibration_dbfs) - self.minimum_calibration_hops
            ),
            "tail_hops": self.tail_hops,
            "tail_spread_db": self._tail_spread_db,
            "open_threshold_dbfs": self._open_threshold_dbfs,
            "close_threshold_dbfs": self._close_threshold_dbfs,
            "active": self._active,
        }
