"""Shared causal audio-evidence policy for polyphonic live and WAV paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .audio_activity_gate import (
    AudioActivityResult,
    CalibratedAudioActivityGate,
)
from .onset_detector import AdaptiveOnsetDetector, OnsetResult


@dataclass(frozen=True)
class PolyphonicAudioEvidence:
    """Onset and audibility evidence for one absolute audio hop."""

    audio_hop_index: int
    time_s: float
    onset: OnsetResult
    activity: AudioActivityResult

    @property
    def calibrated(self) -> bool:
        return bool(self.onset.calibrated and self.activity.calibrated)


class PolyphonicAudioEvidencePolicy:
    """Own the exact onset/activity state used by every polyphonic frontend.

    Live calibration consumes real microphone hops.  Offline transcription
    can call :meth:`prime_silence` first: synthetic calibration then advances
    only the internal estimators, never the real audio clock, so the first WAV
    hop is still processed at index zero.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_samples: int,
        *,
        fft_size: int = 512,
        calibration_s: float = 1.0,
        onset_cooldown_ms: float = 80.0,
        onset_rearm_ratio: float = 1.35,
        onset_rearm_stable_hops: int = 3,
        onset_rearm_attack_ratio: float = 3.0,
        onset_rearm_flux_ratio: float = 2.0,
        onset_rearm_growth_ratio: float = 8.0,
    ) -> None:
        if sample_rate <= 0 or hop_samples <= 0:
            raise ValueError("sample_rate and hop_samples must be positive")
        self.sample_rate = int(sample_rate)
        self.hop_samples = int(hop_samples)
        self.onset_detector = AdaptiveOnsetDetector(
            self.sample_rate,
            self.hop_samples,
            fft_size=fft_size,
            calibration_s=calibration_s,
            cooldown_ms=onset_cooldown_ms,
            rearm_ratio=onset_rearm_ratio,
            enable_peak_rearm=True,
            robust_rearm=True,
            rearm_stable_hops=onset_rearm_stable_hops,
            rearm_attack_ratio=onset_rearm_attack_ratio,
            rearm_flux_ratio=onset_rearm_flux_ratio,
            rearm_growth_ratio=onset_rearm_growth_ratio,
            require_joint_temporal_evidence=True,
        )
        self.activity_gate = CalibratedAudioActivityGate(
            self.sample_rate,
            self.hop_samples,
            calibration_s=calibration_s,
        )
        self.audio_hop_index = -1
        self.synthetic_priming_hops = 0

    @classmethod
    def from_metadata(
        cls,
        sample_rate: int,
        hop_samples: int,
        metadata: Mapping[str, object],
        *,
        calibration_s: float = 1.0,
    ) -> "PolyphonicAudioEvidencePolicy":
        config = metadata.get("audio_evidence", {})
        values = config if isinstance(config, Mapping) else {}
        return cls(
            sample_rate,
            hop_samples,
            fft_size=int(values.get("fft_size", 512)),
            calibration_s=calibration_s,
            onset_cooldown_ms=float(
                values.get("onset_cooldown_ms", 80.0)
            ),
            onset_rearm_ratio=float(
                values.get("onset_rearm_ratio", 1.35)
            ),
            onset_rearm_stable_hops=int(
                values.get("onset_rearm_stable_hops", 3)
            ),
            onset_rearm_attack_ratio=float(
                values.get("onset_rearm_attack_ratio", 3.0)
            ),
            onset_rearm_flux_ratio=float(
                values.get("onset_rearm_flux_ratio", 2.0)
            ),
            onset_rearm_growth_ratio=float(
                values.get("onset_rearm_growth_ratio", 8.0)
            ),
        )

    @property
    def calibrated(self) -> bool:
        return bool(
            self.onset_detector.calibrated and self.activity_gate.calibrated
        )

    @staticmethod
    def rms_to_dbfs(rms: float) -> float:
        return CalibratedAudioActivityGate.rms_to_dbfs(rms)

    def _process_estimators(
        self, hop: np.ndarray
    ) -> tuple[OnsetResult, AudioActivityResult]:
        onset = self.onset_detector.process(hop)
        activity = self.activity_gate.process_rms(onset.rms)
        return onset, activity

    def process(self, hop: np.ndarray) -> PolyphonicAudioEvidence:
        samples = np.asarray(hop, dtype=np.float32).reshape(-1)
        if samples.shape != (self.hop_samples,):
            raise ValueError(f"Expected one audio hop of {self.hop_samples} samples.")
        self.audio_hop_index += 1
        onset, activity = self._process_estimators(samples)
        return PolyphonicAudioEvidence(
            audio_hop_index=self.audio_hop_index,
            time_s=(self.audio_hop_index + 1)
            * self.hop_samples
            / float(self.sample_rate),
            onset=onset,
            activity=activity,
        )

    def prime_silence(self) -> int:
        """Deterministically calibrate offline without consuming WAV time."""
        if self.calibrated:
            return 0
        if self.audio_hop_index >= 0:
            raise RuntimeError(
                "Silent priming must happen before the first real audio hop."
            )
        silence = np.zeros(self.hop_samples, dtype=np.float32)
        maximum_hops = max(
            self.onset_detector.calibration_hops,
            self.activity_gate.maximum_calibration_hops,
        )
        primed = 0
        while not self.calibrated and primed < maximum_hops:
            self._process_estimators(silence)
            primed += 1
        if not self.calibrated:
            raise RuntimeError("Synthetic silence did not calibrate audio evidence.")
        self.synthetic_priming_hops += primed
        return primed

    def reset_continuity(self) -> None:
        """Forget cross-gap evidence while preserving calibration and time."""
        self.onset_detector.reset_continuity()
        self.activity_gate.reset_continuity()

    def reset(self) -> None:
        """Return both estimators and the real audio clock to startup state."""
        self.onset_detector.reset()
        self.activity_gate.reset()
        self.audio_hop_index = -1
        self.synthetic_priming_hops = 0

    def activity_diagnostics(self) -> dict[str, object]:
        return self.activity_gate.diagnostics()

    def diagnostics(self) -> dict[str, object]:
        return {
            "calibrated": self.calibrated,
            "audio_hop_index": self.audio_hop_index,
            "synthetic_priming_hops": self.synthetic_priming_hops,
            "onset_detector_tick_index": self.onset_detector.tick_index,
            "onset_detector": {
                "cooldown_hops": self.onset_detector.cooldown_hops,
                "rearm_ratio": self.onset_detector.rearm_ratio,
                "enable_peak_rearm": self.onset_detector.enable_peak_rearm,
                "robust_rearm": self.onset_detector.robust_rearm,
                "rearm_stable_hops": (
                    self.onset_detector.rearm_stable_hops
                ),
                "rearm_attack_ratio": (
                    self.onset_detector.rearm_attack_ratio
                ),
                "rearm_flux_ratio": (
                    self.onset_detector.rearm_flux_ratio
                ),
                "rearm_growth_ratio": (
                    self.onset_detector.rearm_growth_ratio
                ),
                "require_joint_temporal_evidence": (
                    self.onset_detector.require_joint_temporal_evidence
                ),
            },
            "audio_activity_gate": self.activity_diagnostics(),
        }


def offline_audio_evidence_masks(
    waveform: np.ndarray,
    sample_rate: int,
    hop_samples: int,
    *,
    frame_count: int | None = None,
    calibration_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Replay live activity and attack evidence without consulting labels.

    Silence priming represents the live instruction to remain quiet during
    calibration. It changes estimator state only: frame zero remains the
    first real WAV hop and no evaluation timestamp is shifted.
    """
    source = np.asarray(waveform).reshape(-1)
    complete_frames = len(source) // hop_samples
    padded_frames = int(np.ceil(len(source) / float(hop_samples)))
    if frame_count is None:
        frame_count = padded_frames
    frame_count = int(frame_count)
    if frame_count not in {complete_frames, padded_frames}:
        raise ValueError(
            f"frame_count {frame_count} does not match waveform hops "
            f"({complete_frames} complete, {padded_frames} with tail padding)."
        )

    policy = PolyphonicAudioEvidencePolicy(
        sample_rate, hop_samples, fft_size=512,
        calibration_s=calibration_s,
    )
    priming_hops = policy.prime_silence()
    active = np.zeros(frame_count, dtype=np.bool_)
    onset = np.zeros(frame_count, dtype=np.bool_)
    integer_scale = (
        float(max(abs(np.iinfo(source.dtype).min), 1))
        if np.issubdtype(source.dtype, np.integer)
        else 1.0
    )
    for frame_index in range(frame_count):
        start = frame_index * hop_samples
        hop = np.zeros(hop_samples, dtype=np.float32)
        part = np.asarray(
            source[start:start + hop_samples], dtype=np.float32,
        )
        if integer_scale != 1.0:
            part = part / integer_scale
        hop[:len(part)] = part
        evidence = policy.process(hop)
        active[frame_index] = evidence.activity.active
        onset[frame_index] = evidence.onset.is_onset

    report = policy.diagnostics()
    active_hops = int(np.count_nonzero(active))
    report.update({
        "policy": "shared_live_audio_evidence_with_synthetic_silence_priming",
        "label_leakage": False,
        "silent_priming_hops": priming_hops,
        "real_audio_hops": frame_count,
        "active_hops": active_hops,
        "inactive_hops": frame_count - active_hops,
        "onset_hops": int(np.count_nonzero(onset)),
    })
    return active, onset, report


def offline_audio_activity_mask(
    waveform: np.ndarray,
    sample_rate: int,
    hop_samples: int,
    *,
    frame_count: int | None = None,
    calibration_s: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Backward-compatible activity-only view of the shared live policy."""
    active, _, report = offline_audio_evidence_masks(
        waveform,
        sample_rate,
        hop_samples,
        frame_count=frame_count,
        calibration_s=calibration_s,
    )
    return active, report
