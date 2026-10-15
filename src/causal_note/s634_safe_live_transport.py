"""S634: bounded, discontinuity-safe PCM transport for NOTE LIVE.

Adaptation of the *transport principles* from Andriamarosoa/midi:
nonblocking capture, bounded backpressure, fail-closed MIDI panic,
reinitialised causal acoustics after loss, and observable latency.

Recognition, candidate scoring, NOTE S80, and S614/S632 selectors are
untouched. This class is a host-side transport, NOT a new recogniser.

The input callback MUST NOT call the neural model or a MIDI device.
All inference and MIDI emission happens in the consumer thread.
It accepts absolute sample indices supplied by a continuous capture clock.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import time
from typing import Callable

import numpy as np

from causal_note.strict_live import SR, HOP


@dataclass(frozen=True)
class TransportConfig:
    queue_capacity: int = 16
    max_backlog_blocks: int = 3
    stall_seconds: float = 1.0

    def __post_init__(self):
        if type(self.queue_capacity) is not int or not 2 <= self.queue_capacity <= 512:
            raise ValueError("queue_capacity must be 2..512")
        if (type(self.max_backlog_blocks) is not int
                or not 1 <= self.max_backlog_blocks < self.queue_capacity):
            raise ValueError("max_backlog_blocks must be smaller than queue_capacity")
        if not 0 < self.stall_seconds < 60:
            raise ValueError("invalid microphone stall deadline")


class SafeLiveTransport:
    """Single-producer / single-consumer NOTE bridge.

    engine_factory returns a freshly reset StrictLiveStream, S632Live,
    or another implementation of push(samples,start_sample=) / finish().
    emit(event) is called only by the consumer and may forward NoteOn/Off.
    Timing in returned event dictionaries is in ABSOLUTE input samples.

    Unlike padding a lost block with zeros, dropping audio ALWAYS closes
    previously emitted notes and rebuilds the acoustic and MIDI states.
    """

    def __init__(self, engine_factory: Callable, emit: Callable,
                 config: TransportConfig = TransportConfig(), clock=time.monotonic):
        if not callable(engine_factory) or not callable(emit):
            raise TypeError("A native engine factory and an event sink are required")
        self.factory = engine_factory
        self.emit = emit
        self.config = config
        self.clock = clock
        self.pending = queue.Queue(maxsize=config.queue_capacity)
        self.engine = None
        self.epoch_start = None
        self.next_absolute = None
        self.loss_reason = None
        self.closed = False
        self.events_emitted = 0
        self.recoveries = 0
        self.queue_drops = 0
        self.discarded_backlog = 0
        self.invalid_blocks = 0
        self.audio_gap_samples = 0
        self.blocks_processed = 0
        self.last_callback_at = None
        self._capture_next = None
        self.elapsed_blocks_ms = deque(maxlen=4096)
        self.queue_delays_ms = deque(maxlen=4096)
        self.recovery_ledger = []
        self._consecutive_errors = 0

    def capture(self, pcm, *, start_sample: int, status_ok=True):
        """Only copy and enqueue; safe to call in a realtime audio callback.

        Must supply a mono float32 block at HOP and an absolute capture index.
        The index advances even if a callback was invalid or its block dropped.
        """
        if self.closed:
            raise RuntimeError("Audio transport is closed")
        if type(start_sample) is not int or start_sample < 0:
            raise ValueError("Capture start sample must be a nonnegative integer")
        if self._capture_next is not None and start_sample != self._capture_next:
            self.loss_reason = "capture_gap"
        x = np.asarray(pcm)
        self._capture_next = start_sample + (len(x) if x.ndim else 0)
        self.last_callback_at = self.clock()
        if (not status_ok or x.shape != (HOP,) or x.dtype != np.float32
                or not np.isfinite(x).all()):
            self.invalid_blocks += 1
            self.loss_reason = "invalid_capture"
            return False
        item = (start_sample, x.copy(), self.last_callback_at)
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            # Keep the newest audio without ever treating the lost hop as
            # continuous. The consumer handles the discontinuity and panic.
            self.queue_drops += 1
            self.loss_reason = "queue_overflow"
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(item)
        return True

    def _deliver(self, event, offset, *, force_at=None, reason=None):
        result = dict(event)
        result["decision_sample"] = int(result["decision_sample"]) + offset
        result["available_sample"] = int(result["available_sample"]) + offset
        if force_at is not None:
            result["available_sample"] = max(result["available_sample"], int(force_at))
        result["time"] = result["available_sample"] / SR
        if reason is not None:
            result["reason"] = reason
        if result["decision_sample"] > result["available_sample"]:
            raise AssertionError("Noncausal MIDI event")
        self.emit(result)
        self.events_emitted += 1

    def _panic(self, *, at_absolute, reason):
        if self.engine is None:
            return
        # Both S614 and S632 expose finish(). Only S614 accepts 'reason'.
        for event in self.engine.finish():
            self._deliver(event, self.epoch_start,
                          force_at=at_absolute,
                          reason=("transport_"+reason if event["type"]=="note_off"
                                  else None))
        self.engine = None
        self.epoch_start = None
        self.next_absolute = None

    def _recover(self, absolute, reason):
        previous = self.next_absolute
        self._panic(at_absolute=absolute, reason=reason)
        self.recoveries += 1
        missing = max(0, absolute-previous) if previous is not None else 0
        self.audio_gap_samples += missing
        self.recovery_ledger.append(dict(
            at_absolute_sample=int(absolute),reason=reason,
            observed_gap_samples=int(missing)))
        self.loss_reason = None

    def _consume_one(self, item):
        start, pcm, captured_at = item
        now = self.clock()
        self.queue_delays_ms.append(max(0.,(now-captured_at)*1000))
        if self.loss_reason is not None:
            self._recover(start,self.loss_reason)
        elif self.next_absolute is not None and start != self.next_absolute:
            self._recover(start,"discontinuous_input")
        if self.engine is None:
            self.engine = self.factory()
            self.epoch_start = start
            self.next_absolute = start
        before = self.clock()
        try:
            events = self.engine.push(pcm,start_sample=start-self.epoch_start)
        except Exception:
            # Fail closed on any native model / MIDI selector failure.
            self._recover(start+len(pcm),"engine_error")
            raise
        self.next_absolute = start+len(pcm)
        for event in events:
            self._deliver(event,self.epoch_start)
        self.blocks_processed += 1
        self.elapsed_blocks_ms.append((self.clock()-before)*1000)

    def poll(self, *, max_blocks=1):
        """Consumer-side deterministic workload; never block capture."""
        if self.closed:
            raise RuntimeError("Audio transport is closed")
        if type(max_blocks) is not int or max_blocks < 1:
            raise ValueError("max_blocks must be positive")
        processed=0
        for _ in range(max_blocks):
            # If behind, drop old captured frames. Their timesteps must NOT
            # go through the native model and generate stale NoteOn events.
            if self.pending.qsize()>self.config.max_backlog_blocks:
                while self.pending.qsize()>1:
                    self.pending.get_nowait()
                    self.discarded_backlog += 1
                self.loss_reason = "backpressure"
            try:
                item=self.pending.get_nowait()
            except queue.Empty:
                break
            self._consume_one(item)
            processed += 1
        return processed

    def check_stall(self, *, now=None):
        """Fail closed when the microphone stops delivering audio."""
        if self.closed or self.last_callback_at is None or self.engine is None:
            return False
        observed = self.clock() if now is None else float(now)
        if observed-self.last_callback_at < self.config.stall_seconds:
            return False
        self._recover(self.next_absolute,"microphone_stall")
        return True

    def close(self, *, drain=True):
        """Idempotent final all-notes-off; drop queue if explicitly requested."""
        if self.closed:
            return
        if drain:
            while self.poll(max_blocks=self.config.queue_capacity):
                pass
        else:
            while not self.pending.empty():
                self.pending.get_nowait()
        absolute=self.next_absolute
        if absolute is not None:
            self._panic(at_absolute=absolute,reason="shutdown")
        self.closed=True

    def report(self):
        def dist(samples):
            if not samples:
                return dict(p50=0.,p95=0.,max=0.)
            return dict(p50=float(np.percentile(samples,50)),
                        p95=float(np.percentile(samples,95)),
                        max=float(max(samples)))
        return dict(sample_rate=SR,hop_samples=HOP,
            blocks_processed=self.blocks_processed,
            queue_drops=self.queue_drops,
            backlog_drops=self.discarded_backlog,
            invalid_blocks=self.invalid_blocks,
            recoveries=self.recoveries,
            gap_samples=self.audio_gap_samples,
            events_emitted=self.events_emitted,
            processing_block_ms=dist(self.elapsed_blocks_ms),
            queue_wait_ms=dist(self.queue_delays_ms),
            events=self.recovery_ledger,
            note="CPU and queue timing only; no physical mic-to-MIDI loopback metric")
