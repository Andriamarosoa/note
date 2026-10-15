"""S610 streaming audio-to-notes engine, isolated from frozen NOTE models.

The reference backend is causal MIDI v2.2.1. Event times are decision times
(exclusive sample ends), never backdated acoustic onsets. No S80 K is inferred.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
import queue
import time

import numpy as np

from .live_reference.audio_evidence import PolyphonicAudioEvidencePolicy
from .live_reference.audio_gain import apply_manual_audio_gain, validate_manual_audio_gain
from .live_reference.decoder import PolyphonicDecoder, PolyphonicDecoderConfig
from .live_reference.ring_buffer import MonoRingBuffer
from .live_reference.tflite_runtime import PolyphonicBundle, TFLitePolyphonicModel

SAMPLE_RATE = 44100
HOP = 256
WINDOW = 4096
BACKEND = "midi_v221_causal_reference"
DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "model/live/midi_v221"


def note_name(pitch: int) -> str:
    return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[pitch % 12] + str(pitch // 12 - 1)


class TimingStats:
    """Whole-session counts/mean/max, bounded rolling quantile sample."""

    def __init__(self, capacity: int = 4096) -> None:
        self.values: deque[float] = deque(maxlen=capacity)
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.over_hop = 0

    def add(self, ms: float) -> None:
        self.values.append(float(ms))
        self.count += 1
        self.total += ms
        self.maximum = max(self.maximum, ms)
        self.over_hop += ms > HOP / SAMPLE_RATE * 1000

    def report(self) -> dict:
        return {
            "count": self.count,
            "mean_ms": self.total / self.count if self.count else None,
            "max_ms": self.maximum if self.count else None,
            "over_hop_budget": self.over_hop,
            "quantile_scope": "last_4096_observations",
            "quantile_count": len(self.values),
            **{f"p{p}_ms": float(np.percentile(self.values, p)) if self.values else None for p in (50, 95, 99)},
        }


@dataclass(frozen=True)
class LiveEvent:
    type: str
    sample: int
    pitch: int
    velocity: int
    note_id: str
    reason: str

    def json(self) -> dict:
        return {**asdict(self), "note": note_name(self.pitch),
                "time_s": self.sample / SAMPLE_RATE, "clock": "decision_sample_end",
                "backend": BACKEND}


@dataclass(frozen=True)
class AudioBlock:
    start_sample: int
    samples: np.ndarray
    captured_at: float
    discontinuity: bool = False


class LatestAudioQueue:
    """One capture producer, one worker; retain recent audio under overload.

    The callback never performs inference or MIDI I/O. Sample positions advance
    even when a callback is rejected. Driver-reported loss has unknown length;
    it is counted explicitly and never represented as continuous source audio.
    """

    def __init__(self, capacity: int = 16, max_backlog: int = 8) -> None:
        if capacity < 2 or not 0 <= max_backlog < capacity:
            raise ValueError("Invalid queue bounds")
        self.queue: queue.Queue[AudioBlock] = queue.Queue(maxsize=capacity)
        self.max_backlog = max_backlog
        self.next_sample = 0
        self.last_callback_at = time.perf_counter()
        self.last_valid_callback_at = self.last_callback_at
        self.callback_drops = 0
        self.backlog_drops = 0
        self.status_events = 0
        self.invalid_blocks = 0
        self._gap = False

    def callback(self, indata, frames, time_info=None, status=False) -> None:
        captured = time.perf_counter()
        start = self.next_sample
        self.next_sample += max(0, int(frames))
        self.last_callback_at = captured
        if status:
            self.status_events += 1
            self._gap = True
            return
        if frames != HOP or indata.shape != (HOP, 1):
            self.invalid_blocks += 1
            self._gap = True
            return
        self.last_valid_callback_at = captured
        block = AudioBlock(start, np.asarray(indata[:, 0], np.float32).copy(), captured, self._gap)
        self._gap = False
        try:
            self.queue.put_nowait(block)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.callback_drops += 1
            except queue.Empty:
                pass
            # Only this callback produces; the worker can only make more room.
            self.queue.put_nowait(block)

    def get(self, timeout: float = 0.1) -> AudioBlock:
        block = self.queue.get(timeout=timeout)
        if self.queue.qsize() > self.max_backlog:
            # Snapshot the size: concurrent callbacks cannot prolong this loop.
            for _ in range(self.queue.qsize()):
                try:
                    block = self.queue.get_nowait()
                    self.backlog_drops += 1
                except queue.Empty:
                    break
            block = AudioBlock(block.start_sample, block.samples, block.captured_at, True)
        return block

    def report(self) -> dict:
        return {"capacity_hops": self.queue.maxsize,
                "max_backlog_hops": self.max_backlog,
                "callback_queue_drops": self.callback_drops,
                "worker_backlog_drops": self.backlog_drops,
                "driver_status_events_unknown_lost_duration": self.status_events,
                "invalid_callback_blocks": self.invalid_blocks,
                "sample_clock": "delivered_callback_samples; driver loss duration unknown"}


class LiveEngine:
    """Bounded causal DSP/inference/decoder state. Only the worker calls it."""

    def __init__(self, runtime, metadata: dict, *, gain: float = 1.0,
                 calibration_s: float = 1.0, prime_silence: bool = False) -> None:
        if (metadata["sample_rate"], metadata["hop_samples"], metadata["max_window_samples"],
            metadata["min_pitch"], metadata["max_pitch"]) != (SAMPLE_RATE, HOP, WINDOW, 40, 76):
            raise ValueError("Unsupported reference model contract")
        validate_manual_audio_gain(gain)
        self.runtime, self.metadata, self.gain = runtime, metadata, gain
        self.decoder_config = PolyphonicDecoderConfig(**metadata["decoder"])
        self.decoder = PolyphonicDecoder(self.decoder_config)
        self.evidence = PolyphonicAudioEvidencePolicy.from_metadata(
            SAMPLE_RATE, HOP, metadata, calibration_s=calibration_s)
        if prime_silence:
            self.evidence.prime_silence()
        self.ring = MonoRingBuffer(WINDOW)
        self.window = np.zeros(WINDOW, np.float32)
        self.last_end = 0
        self.closed = False
        self.active_ids: dict[int, str] = {}
        self.sequence = 0
        self.events = Counter()
        self.reasons = Counter()
        self.recoveries = 0
        self.recovery_history = deque(maxlen=32)
        self.missing_samples = 0
        self.clipped_samples = 0
        self.processed_hops = 0
        self.skipped_inferences = 0
        self.pending_onset_hop = None
        self.maximum_active = 0
        self.inference_timing = TimingStats()
        self.processing_timing = TimingStats()

    @classmethod
    def from_bundle(cls, path: Path = DEFAULT_BUNDLE, *, threads: int = 2, **kwargs):
        if threads < 1:
            raise ValueError("threads must be positive")
        bundle = PolyphonicBundle(path)
        runtime = TFLitePolyphonicModel(bundle, threads)
        for _ in range(20):
            runtime.infer(np.zeros(WINDOW, np.float32))
        return cls(runtime, bundle.metadata, **kwargs)

    def _events(self, events, sample: int) -> list[LiveEvent]:
        output = []
        for event in events:
            if event.kind == "note_on":
                if event.pitch in self.active_ids:
                    raise RuntimeError("Duplicate active pitch without NoteOff")
                self.sequence += 1
                note_id = f"live-{self.sequence}"
                self.active_ids[event.pitch] = note_id
            else:
                note_id = self.active_ids.pop(event.pitch)
            output.append(LiveEvent(event.kind, sample, event.pitch, event.velocity,
                                    note_id, event.reason))
            self.events[event.kind] += 1
            self.reasons[event.reason] += 1
        self.maximum_active = max(self.maximum_active, len(self.active_ids))
        return output

    def recover(self, sample: int, reason: str = "audio_gap") -> list[LiveEvent]:
        missing = max(0, sample - self.last_end)
        self.missing_samples += missing
        self.recovery_history.append({"resume_sample": sample, "last_processed_sample": self.last_end,
                                      "missing_delivered_samples": missing, "reason": reason})
        result = self._events(self.decoder.panic(reason=reason), sample)
        self.decoder = PolyphonicDecoder(self.decoder_config)
        self.ring.reset()
        self.evidence.reset_continuity()
        self.pending_onset_hop = None
        self.recoveries += 1
        return result

    def process(self, samples: np.ndarray, *, start_sample: int,
                discontinuity: bool = False, skip_inference: bool = False) -> list[LiveEvent]:
        if self.closed:
            raise RuntimeError("Stream is finalized")
        if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < self.last_end:
            raise ValueError("Audio sample clock must increase without overlap")
        hop = np.asarray(samples, np.float32)
        if hop.shape != (HOP,) or not np.all(np.isfinite(hop)):
            raise ValueError("Expected 256 finite mono samples")
        started = time.perf_counter()
        output = []
        if discontinuity or start_sample != self.last_end:
            output.extend(self.recover(start_sample))
        hop, clipping = apply_manual_audio_gain(hop, self.gain)
        self.clipped_samples += clipping
        # Align evidence and decoder clocks to capture, including discarded hops.
        self.evidence.audio_hop_index = start_sample // HOP - 1
        evidence = self.evidence.process(hop)
        if evidence.onset.is_onset:
            self.pending_onset_hop = evidence.audio_hop_index
        self.ring.write(hop)
        self.last_end = int(start_sample) + HOP
        self.processed_hops += 1
        if evidence.calibrated and self.ring.available >= 512:
            if skip_inference:
                self.skipped_inferences += 1
                self.processing_timing.add((time.perf_counter() - started) * 1000)
                return output
            self.ring.copy_latest_into(self.window)
            prediction = self.runtime.infer(self.window, self.ring.available)
            for values, shape in ((prediction.frame_probability, (37,)),
                                  (prediction.onset_probability, (37,)),
                                  (prediction.harmonic_amplitude, (37, 20))):
                if np.shape(values) != shape or not np.all(np.isfinite(values)):
                    raise ValueError("Invalid causal model output")
            self.inference_timing.add(prediction.inference_ms)
            events = self.decoder.step(
                prediction.frame_probability, prediction.onset_probability,
                prediction.harmonic_amplitude, audio_active=evidence.activity.active,
                audio_hop_index=evidence.audio_hop_index,
                audio_onset=self.pending_onset_hop is not None,
                audio_onset_hop_index=self.pending_onset_hop)
            self.pending_onset_hop = None
            output.extend(self._events(events, self.last_end))
        self.processing_timing.add((time.perf_counter() - started) * 1000)
        return output

    def finalize(self, end_sample: int | None = None, reason: str = "stop") -> list[LiveEvent]:
        if self.closed:
            return []
        end_sample = self.last_end if end_sample is None else end_sample
        if end_sample < self.last_end:
            raise ValueError("End precedes processed audio")
        result = self._events(self.decoder.panic(reason=reason), end_sample)
        self.closed = True
        return result

    def report(self) -> dict:
        return {"backend": BACKEND, "offline_S601_scores_apply": False,
                "S80_integrated": False, "sample_rate": SAMPLE_RATE,
                "hop_samples": HOP, "hop_ms": HOP / SAMPLE_RATE * 1000,
                "lookahead_samples": 0, "maximum_past_context_samples": WINDOW,
                "pitch_range": [40, 76], "maximum_polyphony": self.decoder_config.maximum_polyphony,
                "gain": self.gain, "automatic_input_level": False,
                "model_sha256": self.metadata["artifact"]["sha256"],
                "decoder": asdict(self.decoder_config),
                "processed_hops": self.processed_hops, "last_processed_sample": self.last_end,
                "skipped_inferences_audio_retained": self.skipped_inferences,
                "events": dict(self.events), "event_reasons": dict(self.reasons),
                "maximum_active_notes": self.maximum_active,
                "active_notes_at_end": sorted(self.active_ids),
                "continuity_resets": self.recoveries,
                "missing_delivered_samples": self.missing_samples,
                "last_32_recoveries": list(self.recovery_history),
                "gain_induced_clipped_samples": self.clipped_samples,
                "inference": self.inference_timing.report(),
                "processing_without_output": self.processing_timing.report(),
                "audio_evidence": self.evidence.diagnostics()}
