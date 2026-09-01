"""Minimal renewal of 512-sample frames at detected offsets."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set, Tuple

from .audio_buffer import CausalAudioBuffer, FrameAvailability
from .detector import BoundaryEvent, BoundaryType


class SchedulerError(Exception):
    """Base class for renewal failures."""


class UnknownEventError(SchedulerError, LookupError):
    """Raised when an event identifier is unknown."""


@dataclass
class _Event:
    event_id: str
    onset_sample: int
    offset_sample: Optional[int] = None


@dataclass
class _Request:
    frame_start: int
    source_event_ids: Set[str] = field(default_factory=set)
    completed: bool = False


@dataclass(frozen=True)
class RestartFrame:
    """One complete frame beginning exactly at a detected offset."""

    frame_start: int
    frame_end: int
    source_event_ids: Tuple[str, ...]
    availability: FrameAvailability
    samples: Optional[Tuple[float, ...]]

    @property
    def ready(self) -> bool:
        return self.samples is not None


class RestartScheduler:
    """Keep open events and retain every frame requested by an offset."""

    def __init__(self, frame_size: int = 512) -> None:
        if frame_size != 512 or isinstance(frame_size, bool):
            raise ValueError("RestartScheduler requires exactly 512 samples")
        self._audio = CausalAudioBuffer(frame_size)
        self._events: Dict[str, _Event] = {}
        self._requests: Dict[int, _Request] = {}
        self._finalized = False

    @property
    def audio_end_sample(self) -> int:
        return self._audio.end_sample

    @property
    def audio_start_sample(self) -> int:
        return self._audio.start_sample

    def append_audio(
        self,
        samples: Iterable[float],
        *,
        start_sample: Optional[int] = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("restart scheduler is finalized")
        self._audio.append(samples, start_sample=start_sample)

    def accept_event(self, event: BoundaryEvent) -> Optional[RestartFrame]:
        """Apply one live boundary event to the renewal state."""

        if self._finalized:
            raise RuntimeError("restart scheduler is finalized")
        if not isinstance(event, BoundaryEvent):
            raise ValueError("event must be a BoundaryEvent")
        if event.sample > self._audio.end_sample:
            raise ValueError("event sample has not been received yet")
        if event.kind is BoundaryType.ONSET:
            self.open_event(event.event_id, event.sample)
            return None
        return self.close_event(event.event_id, event.sample)

    def open_event(self, event_id: str, onset_sample: int) -> None:
        if self._finalized:
            raise RuntimeError("restart scheduler is finalized")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if (
            isinstance(onset_sample, bool)
            or not isinstance(onset_sample, int)
            or onset_sample < 0
        ):
            raise ValueError("onset_sample must be an integer >= 0")
        if event_id in self._events:
            raise SchedulerError("event_id is already open or closed")
        self._events[event_id] = _Event(event_id, onset_sample)

    def close_event(self, event_id: str, offset_sample: int) -> RestartFrame:
        if self._finalized:
            raise RuntimeError("restart scheduler is finalized")
        event = self._events.get(event_id)
        if event is None:
            raise UnknownEventError(f"unknown event: {event_id}")
        if event.offset_sample is not None:
            raise SchedulerError("event is already closed")
        if (
            isinstance(offset_sample, bool)
            or not isinstance(offset_sample, int)
            or not event.onset_sample < offset_sample <= self._audio.end_sample
        ):
            raise ValueError(
                "offset_sample must be after onset and already received"
            )
        event.offset_sample = offset_sample
        request = self._requests.get(offset_sample)
        if request is None:
            request = _Request(offset_sample)
            self._requests[offset_sample] = request
        request.source_event_ids.add(event_id)
        return self._snapshot(request)

    def accept_terminal_events(
        self,
        events: Iterable[BoundaryEvent],
    ) -> None:
        """Close the complete open-ID set at EOF without requesting frames."""

        if self._finalized:
            raise RuntimeError("restart scheduler is already finalized")
        terminal_events = tuple(events)
        terminal_sample = self._audio.end_sample
        open_events = {
            event.event_id: event
            for event in self._events.values()
            if event.offset_sample is None
        }
        seen = set()
        for boundary in terminal_events:
            if not isinstance(boundary, BoundaryEvent):
                raise ValueError("terminal events must be BoundaryEvent objects")
            if boundary.kind is not BoundaryType.OFFSET:
                raise ValueError("terminal events must all be offsets")
            if boundary.sample != terminal_sample:
                raise ValueError(
                    "terminal event sample must equal the exclusive audio end"
                )
            if boundary.event_id in seen:
                raise SchedulerError("duplicate terminal event_id")
            seen.add(boundary.event_id)
            event = open_events.get(boundary.event_id)
            if event is None:
                raise UnknownEventError(
                    f"terminal event is not open: {boundary.event_id}"
                )
            if boundary.sample <= event.onset_sample:
                raise ValueError("terminal offset must be after its onset")

        expected = set(open_events)
        if seen != expected:
            missing = tuple(sorted(expected - seen))
            raise SchedulerError(
                f"terminal events do not contain the complete open-ID set: {missing}"
            )

        # Validation is complete. Terminal control closures deliberately bypass
        # close_event(), because no future 512-sample restart frame can exist.
        for boundary in terminal_events:
            self._events[boundary.event_id].offset_sample = boundary.sample
        self._finalized = True

    def open_event_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                event.event_id
                for event in self._events.values()
                if event.offset_sample is None
            )
        )

    def restart_frames(self) -> Tuple[RestartFrame, ...]:
        return tuple(
            self._snapshot(self._requests[start])
            for start in sorted(self._requests)
        )

    def ready_frames(self) -> Tuple[RestartFrame, ...]:
        return tuple(
            frame
            for frame in self.restart_frames()
            if frame.ready and not self._requests[frame.frame_start].completed
        )

    def complete_frame(self, frame_start: int) -> None:
        request = self._requests.get(frame_start)
        if request is None:
            raise SchedulerError("unknown restart frame")
        if request.completed:
            raise SchedulerError("restart frame is already completed")
        if not self._audio.availability(frame_start).ready:
            raise SchedulerError("restart frame is not ready")
        request.completed = True

    def prune_completed(self) -> int:
        """Remove completed requests/closed events and compact retained audio.

        Returns the new first retained absolute sample. Open events need no
        historical scheduler audio: any future restart frame begins at their
        future offset.
        """

        completed_starts = tuple(
            start
            for start, request in self._requests.items()
            if request.completed
        )
        completed_event_ids = set()
        for start in completed_starts:
            completed_event_ids.update(self._requests[start].source_event_ids)
            del self._requests[start]
        for event_id in completed_event_ids:
            event = self._events.get(event_id)
            if event is not None and event.offset_sample is not None:
                del self._events[event_id]

        retain_from = min(
            self._requests,
            default=self._audio.end_sample,
        )
        if retain_from > self._audio.start_sample:
            self._audio.discard_before(retain_from)
        return self._audio.start_sample

    def _snapshot(self, request: _Request) -> RestartFrame:
        availability = self._audio.availability(request.frame_start)
        samples = (
            self._audio.frame_at(request.frame_start)
            if availability.ready and not request.completed
            else None
        )
        return RestartFrame(
            frame_start=request.frame_start,
            frame_end=request.frame_start + self._audio.frame_size,
            source_event_ids=tuple(sorted(request.source_event_ids)),
            availability=availability,
            samples=samples,
        )
