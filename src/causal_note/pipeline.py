"""Integrated live boundary detection and exact offset-frame renewal."""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .detector import BoundaryEvent
from .scheduler import RestartFrame, RestartScheduler


@dataclass(frozen=True)
class LiveChunkResult:
    """Boundaries and restart frames produced after one live audio chunk."""

    events: Tuple[BoundaryEvent, ...]
    requested_frames: Tuple[RestartFrame, ...]
    ready_frames: Tuple[RestartFrame, ...]


class LiveOnsetOffsetPipeline:
    """Keep the detector and restart scheduler on one contiguous timeline."""

    def __init__(self, detector, scheduler: Optional[RestartScheduler] = None) -> None:
        if not callable(getattr(detector, "process_chunk", None)):
            raise ValueError("detector must implement process_chunk")
        if not hasattr(detector, "next_sample"):
            raise ValueError("detector must expose next_sample")
        if not callable(getattr(detector, "finalize_stream", None)):
            raise ValueError("detector must implement finalize_stream")
        if not callable(getattr(detector, "active_events", None)):
            raise ValueError("detector must implement active_events")
        if scheduler is not None and not isinstance(scheduler, RestartScheduler):
            raise ValueError("scheduler must be a RestartScheduler")
        self._detector = detector
        self._scheduler = scheduler or RestartScheduler()
        self._finalized = False

    @property
    def next_sample(self) -> int:
        return self._scheduler.audio_end_sample

    @property
    def scheduler(self) -> RestartScheduler:
        return self._scheduler

    def process_chunk(
        self,
        samples: Iterable[float],
        *,
        start_sample: Optional[int] = None,
    ) -> LiveChunkResult:
        if self._finalized:
            raise RuntimeError("live pipeline is finalized")
        values = tuple(samples)
        expected = self._scheduler.audio_end_sample
        start = expected if start_sample is None else start_sample
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("start_sample must be an integer >= 0")
        if start != expected:
            raise ValueError(
                f"expected contiguous pipeline chunk at {expected}, got {start}"
            )
        detector_next = self._detector.next_sample
        if detector_next is not None and detector_next != expected:
            raise ValueError("detector and scheduler timelines are inconsistent")

        events = self._detector.process_chunk(values, start_sample=start)
        self._scheduler.append_audio(values, start_sample=start)
        requested = []
        for event in events:
            frame = self._scheduler.accept_event(event)
            if frame is not None:
                requested.append(frame)
        ready = self._scheduler.ready_frames()
        # Even before a frame completes, audio older than the earliest pending
        # offset frame is irrelevant to the scheduler. This bounds long-stream
        # memory while open events remain tracked by ID.
        self._scheduler.prune_completed()
        return LiveChunkResult(
            events=tuple(events),
            requested_frames=tuple(requested),
            ready_frames=ready,
        )

    def finalize_stream(self) -> LiveChunkResult:
        """Close every open event at EOF without creating restart frames."""

        if self._finalized:
            raise RuntimeError("live pipeline is already finalized")
        end = self._scheduler.audio_end_sample
        detector_next = self._detector.next_sample
        if detector_next is not None and detector_next != end:
            raise ValueError("detector and scheduler timelines are inconsistent")

        detector_active = tuple(self._detector.active_events())
        detector_ids = tuple(sorted(event.event_id for event in detector_active))
        scheduler_ids = self._scheduler.open_event_ids()
        if detector_ids != scheduler_ids:
            raise ValueError("detector and scheduler open event IDs are inconsistent")
        if any(end <= event.onset_sample for event in detector_active):
            raise ValueError("stream end must be after every open onset")

        events = tuple(self._detector.finalize_stream(end_sample=end))
        self._scheduler.accept_terminal_events(events)
        self._finalized = True
        return LiveChunkResult(
            events=events,
            requested_frames=(),
            ready_frames=self._scheduler.ready_frames(),
        )

    def complete_frame(self, frame_start: int, *, prune: bool = True) -> None:
        """Acknowledge an analysed restart frame and optionally compact state."""

        self._scheduler.complete_frame(frame_start)
        if prune:
            self._scheduler.prune_completed()
