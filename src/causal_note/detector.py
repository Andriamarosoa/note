"""Causal live onset/offset events with opaque event association."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Iterable, Optional, Protocol, Tuple


FRAME_SIZE = 512
SAMPLE_RATE = 44_100


def _index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value


def _stream_end_sample(
    end_sample: Optional[int],
    next_sample: Optional[int],
) -> int:
    """Resolve an exclusive stream end without inventing unreceived audio."""

    expected = 0 if next_sample is None else next_sample
    end = expected if end_sample is None else _index("end_sample", end_sample)
    if end != expected:
        raise ValueError(f"expected stream end at {expected}, got {end}")
    return end


def _samples(values: Iterable[float]) -> Tuple[float, ...]:
    converted = []
    for sample in values:
        if isinstance(sample, bool):
            raise ValueError("audio samples must be finite numbers")
        try:
            value = float(sample)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("audio samples must be finite numbers") from exc
        if not math.isfinite(value):
            raise ValueError("audio samples must be finite numbers")
        converted.append(value)
    return tuple(converted)


class BoundaryType(str, Enum):
    ONSET = "onset"
    OFFSET = "offset"


@dataclass(frozen=True)
class BoundaryEvent:
    """One event emitted immediately by the live pipeline."""

    kind: BoundaryType
    event_id: str
    sample: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryType):
            raise ValueError("kind must be a BoundaryType")
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        _index("sample", self.sample)


@dataclass(frozen=True)
class BoundaryCandidate:
    """One unassociated boundary peak emitted by the candidate decoder."""

    kind: BoundaryType
    sample: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryType):
            raise ValueError("kind must be a BoundaryType")
        _index("sample", self.sample)


@dataclass(frozen=True)
class ActiveEvent:
    """Minimal state retained between one onset and its offset."""

    event_id: str
    onset_sample: int
    representation: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        _index("onset_sample", self.onset_sample)
        object.__setattr__(self, "representation", _samples(self.representation))


class LiveEventTracker:
    """Allocate onset ids and close exactly the selected open event."""

    def __init__(self, event_prefix: str = "event") -> None:
        if not isinstance(event_prefix, str) or not event_prefix.strip():
            raise ValueError("event_prefix must be a non-empty string")
        self._event_prefix = event_prefix
        self._next_id = 1
        self._active: Dict[str, ActiveEvent] = {}
        self._finalized = False

    def start_event(
        self,
        onset_sample: int,
        representation: Iterable[float] = (),
    ) -> BoundaryEvent:
        if self._finalized:
            raise RuntimeError("event tracker is finalized")
        onset = _index("onset_sample", onset_sample)
        event_id = f"{self._event_prefix}-{self._next_id:06d}"
        active = ActiveEvent(event_id, onset, tuple(representation))
        self._active[event_id] = active
        self._next_id += 1
        return BoundaryEvent(BoundaryType.ONSET, event_id, onset)

    def finish_event(self, event_id: str, offset_sample: int) -> BoundaryEvent:
        if self._finalized:
            raise RuntimeError("event tracker is finalized")
        active = self._active.get(event_id)
        if active is None:
            raise KeyError(f"unknown active event: {event_id}")
        offset = _index("offset_sample", offset_sample)
        if offset <= active.onset_sample:
            raise ValueError("offset_sample must be after its onset")
        del self._active[event_id]
        return BoundaryEvent(BoundaryType.OFFSET, event_id, offset)

    def finish_all(self, end_sample: int) -> Tuple[BoundaryEvent, ...]:
        """Close every open ID at one validated exclusive stream end."""

        if self._finalized:
            raise RuntimeError("event tracker is already finalized")
        end = _index("end_sample", end_sample)
        active = self.active_events()
        if any(end <= item.onset_sample for item in active):
            raise ValueError("end_sample must be after every open onset")

        # All validation precedes mutation so a bad terminal position leaves the
        # complete open-ID set available for a corrected finalization attempt.
        events = tuple(
            BoundaryEvent(BoundaryType.OFFSET, item.event_id, end)
            for item in active
        )
        self._active.clear()
        self._finalized = True
        return events

    def active_events(self) -> Tuple[ActiveEvent, ...]:
        return tuple(
            sorted(
                self._active.values(),
                key=lambda item: (item.onset_sample, item.event_id),
            )
        )


class LiveEnergyDetector:
    """Causal monophonic smoke detector operating on contiguous audio chunks."""

    def __init__(
        self,
        threshold: float = 0.02,
        release_samples: int = 16,
        *,
        event_prefix: str = "event",
    ) -> None:
        if isinstance(threshold, bool):
            raise ValueError("threshold must be finite and > 0")
        try:
            converted_threshold = float(threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("threshold must be finite and > 0") from exc
        if not math.isfinite(converted_threshold) or converted_threshold <= 0:
            raise ValueError("threshold must be finite and > 0")
        if (
            isinstance(release_samples, bool)
            or not isinstance(release_samples, int)
            or release_samples <= 0
        ):
            raise ValueError("release_samples must be an integer > 0")
        self._threshold = converted_threshold
        self._release_samples = release_samples
        self._tracker = LiveEventTracker(event_prefix)
        self._next_sample: Optional[int] = None
        self._current_event_id: Optional[str] = None
        self._silence_start: Optional[int] = None
        self._finalized = False

    @property
    def next_sample(self) -> Optional[int]:
        return self._next_sample

    def active_events(self) -> Tuple[ActiveEvent, ...]:
        return self._tracker.active_events()

    def process_chunk(
        self,
        samples: Iterable[float],
        *,
        start_sample: Optional[int] = None,
    ) -> Tuple[BoundaryEvent, ...]:
        """Emit onset now and the associated offset when release is confirmed."""

        if self._finalized:
            raise RuntimeError("energy detector is finalized")
        values = _samples(samples)
        if self._next_sample is None:
            start = 0 if start_sample is None else _index(
                "start_sample",
                start_sample,
            )
        else:
            start = self._next_sample if start_sample is None else _index(
                "start_sample",
                start_sample,
            )
            if start != self._next_sample:
                raise ValueError(
                    f"expected contiguous chunk at {self._next_sample}, got {start}"
                )

        emitted = []
        for relative_sample, sample in enumerate(values):
            absolute_sample = start + relative_sample
            active = abs(sample) >= self._threshold
            if self._current_event_id is None:
                if active:
                    event = self._tracker.start_event(absolute_sample)
                    self._current_event_id = event.event_id
                    self._silence_start = None
                    emitted.append(event)
                continue

            if active:
                self._silence_start = None
                continue
            if self._silence_start is None:
                self._silence_start = absolute_sample
            silent_length = absolute_sample - self._silence_start + 1
            if silent_length >= self._release_samples:
                event = self._tracker.finish_event(
                    self._current_event_id,
                    self._silence_start,
                )
                emitted.append(event)
                self._current_event_id = None
                self._silence_start = None

        self._next_sample = start + len(values)
        return tuple(emitted)

    def finalize_stream(
        self,
        end_sample: Optional[int] = None,
    ) -> Tuple[BoundaryEvent, ...]:
        """Emit control offsets for events still open at the exclusive EOF."""

        if self._finalized:
            raise RuntimeError("energy detector is already finalized")
        end = _stream_end_sample(end_sample, self._next_sample)
        events = self._tracker.finish_all(end)
        self._current_event_id = None
        self._silence_start = None
        self._finalized = True
        return events


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite probability")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _release_probability(
    name: str,
    value: Optional[float],
    *,
    entry_name: str,
    entry_threshold: float,
) -> float:
    converted = (
        entry_threshold
        if value is None
        else _probability(name, value)
    )
    if converted == 0.0 or converted > entry_threshold:
        raise ValueError(
            f"{name} must satisfy 0 < {name} <= {entry_name}"
        )
    return converted


def _score_rows(
    name: str,
    rows: Iterable[Iterable[float]],
) -> Tuple[Tuple[float, ...], ...]:
    converted = []
    width: Optional[int] = None
    for row in rows:
        values = tuple(
            _probability(f"{name} score", value) for value in row
        )
        if width is None:
            width = len(values)
            if width == 0:
                raise ValueError("boundary scores must contain at least one slot")
        elif len(values) != width:
            raise ValueError("all boundary score rows must have the same width")
        converted.append(values)
    return tuple(converted)


@dataclass(frozen=True)
class BoundaryScoreChunk:
    """Model scores for anonymous internal slots over one contiguous chunk.

    Slots are an internal association mechanism.  They never appear in emitted
    :class:`BoundaryEvent` objects, which expose only opaque event identifiers.
    """

    start_sample: int
    onset: Tuple[Tuple[float, ...], ...]
    offset: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        _index("start_sample", self.start_sample)
        onset = _score_rows("onset", self.onset)
        offset = _score_rows("offset", self.offset)
        if len(onset) != len(offset):
            raise ValueError("onset and offset scores must have equal lengths")
        if onset and len(onset[0]) != len(offset[0]):
            raise ValueError("onset and offset scores must use equal slot counts")
        object.__setattr__(self, "onset", onset)
        object.__setattr__(self, "offset", offset)

    @property
    def sample_count(self) -> int:
        return len(self.onset)

    @property
    def slot_count(self) -> int:
        if self.onset:
            return len(self.onset[0])
        if self.offset:
            return len(self.offset[0])
        return 0


class BoundaryScorePredictor(Protocol):
    """Minimal interface implemented by a trained causal boundary model."""

    @property
    def slot_count(self) -> int:
        ...

    def predict_chunk(
        self,
        samples: Tuple[float, ...],
        *,
        start_sample: int,
    ) -> BoundaryScoreChunk:
        ...


class LiveBoundaryScoreDecoder:
    """Turn causal per-slot model scores into associated live boundaries.

    A slot is allowed to contain at most one open event.  Offset decisions are
    applied before onset decisions at the same sample, so a retrigger can close
    the old event and immediately allocate a new opaque identifier.
    """

    def __init__(
        self,
        slot_count: int = 6,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.5,
        *,
        onset_release_threshold: Optional[float] = None,
        offset_release_threshold: Optional[float] = None,
        event_prefix: str = "event",
    ) -> None:
        if (
            isinstance(slot_count, bool)
            or not isinstance(slot_count, int)
            or slot_count <= 0
        ):
            raise ValueError("slot_count must be an integer > 0")
        self._slot_count = slot_count
        self._onset_threshold = _probability(
            "onset_threshold", onset_threshold
        )
        self._offset_threshold = _probability(
            "offset_threshold", offset_threshold
        )
        if self._onset_threshold == 0.0 or self._offset_threshold == 0.0:
            raise ValueError("boundary thresholds must be > 0")
        self._onset_release_threshold = _release_probability(
            "onset_release_threshold",
            onset_release_threshold,
            entry_name="onset_threshold",
            entry_threshold=self._onset_threshold,
        )
        self._offset_release_threshold = _release_probability(
            "offset_release_threshold",
            offset_release_threshold,
            entry_name="offset_threshold",
            entry_threshold=self._offset_threshold,
        )
        self._tracker = LiveEventTracker(event_prefix)
        self._slot_event_ids = [None] * slot_count
        self._onset_high = [False] * slot_count
        self._offset_high = [False] * slot_count
        self._next_sample: Optional[int] = None
        self._finalized = False

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def onset_threshold(self) -> float:
        return self._onset_threshold

    @property
    def offset_threshold(self) -> float:
        return self._offset_threshold

    @property
    def onset_release_threshold(self) -> float:
        return self._onset_release_threshold

    @property
    def offset_release_threshold(self) -> float:
        return self._offset_release_threshold

    @property
    def next_sample(self) -> Optional[int]:
        return self._next_sample

    def active_events(self) -> Tuple[ActiveEvent, ...]:
        return self._tracker.active_events()

    def process_chunk(
        self,
        scores: BoundaryScoreChunk,
    ) -> Tuple[BoundaryEvent, ...]:
        """Decode one completely validated, contiguous score chunk."""

        if self._finalized:
            raise RuntimeError("boundary decoder is finalized")
        if not isinstance(scores, BoundaryScoreChunk):
            raise ValueError("scores must be a BoundaryScoreChunk")
        if scores.sample_count and scores.slot_count != self._slot_count:
            raise ValueError(
                f"expected {self._slot_count} slots, got {scores.slot_count}"
            )
        if self._next_sample is not None and scores.start_sample != self._next_sample:
            raise ValueError(
                f"expected contiguous scores at {self._next_sample}, "
                f"got {scores.start_sample}"
            )

        emitted = []
        for relative_sample, (onset_row, offset_row) in enumerate(
            zip(scores.onset, scores.offset)
        ):
            absolute_sample = scores.start_sample + relative_sample

            for slot, score in enumerate(offset_row):
                if self._offset_high[slot]:
                    high = score >= self._offset_release_threshold
                else:
                    high = score >= self._offset_threshold
                event_id = self._slot_event_ids[slot]
                if high and not self._offset_high[slot] and event_id is not None:
                    emitted.append(
                        self._tracker.finish_event(event_id, absolute_sample)
                    )
                    self._slot_event_ids[slot] = None
                self._offset_high[slot] = high

            for slot, score in enumerate(onset_row):
                if self._onset_high[slot]:
                    high = score >= self._onset_release_threshold
                else:
                    high = score >= self._onset_threshold
                if (
                    high
                    and not self._onset_high[slot]
                    and self._slot_event_ids[slot] is None
                ):
                    event = self._tracker.start_event(absolute_sample)
                    self._slot_event_ids[slot] = event.event_id
                    emitted.append(event)
                self._onset_high[slot] = high

        self._next_sample = scores.start_sample + scores.sample_count
        return tuple(emitted)

    def finalize_stream(
        self,
        end_sample: Optional[int] = None,
    ) -> Tuple[BoundaryEvent, ...]:
        """Close all associated open events at the exclusive stream end."""

        if self._finalized:
            raise RuntimeError("boundary decoder is already finalized")
        end = _stream_end_sample(end_sample, self._next_sample)
        events = self._tracker.finish_all(end)
        self._slot_event_ids = [None] * self._slot_count
        self._finalized = True
        return events


class LiveBoundaryPeakDecoder:
    """Emit unassociated rising onset and offset peaks from causal scores.

    Model slots remain private and are used only to preserve an independent
    high/low state for every score channel.  Candidate multiplicity is
    preserved when several channels cross their thresholds at the same sample.
    """

    def __init__(
        self,
        slot_count: int = 6,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.5,
        *,
        onset_release_threshold: Optional[float] = None,
        offset_release_threshold: Optional[float] = None,
        rearm_low_samples: int = 1,
        consolidation_samples: int = 0,
    ) -> None:
        if (
            isinstance(slot_count, bool)
            or not isinstance(slot_count, int)
            or slot_count <= 0
        ):
            raise ValueError("slot_count must be an integer > 0")
        if (
            isinstance(rearm_low_samples, bool)
            or not isinstance(rearm_low_samples, int)
            or rearm_low_samples <= 0
        ):
            raise ValueError("rearm_low_samples must be an integer > 0")
        if (
            isinstance(consolidation_samples, bool)
            or not isinstance(consolidation_samples, int)
            or consolidation_samples < 0
        ):
            raise ValueError("consolidation_samples must be an integer >= 0")
        self._slot_count = slot_count
        self._rearm_low_samples = rearm_low_samples
        self._consolidation_samples = consolidation_samples
        self._onset_threshold = _probability(
            "onset_threshold", onset_threshold
        )
        self._offset_threshold = _probability(
            "offset_threshold", offset_threshold
        )
        if self._onset_threshold == 0.0 or self._offset_threshold == 0.0:
            raise ValueError("boundary thresholds must be > 0")
        self._onset_release_threshold = _release_probability(
            "onset_release_threshold",
            onset_release_threshold,
            entry_name="onset_threshold",
            entry_threshold=self._onset_threshold,
        )
        self._offset_release_threshold = _release_probability(
            "offset_release_threshold",
            offset_release_threshold,
            entry_name="offset_threshold",
            entry_threshold=self._offset_threshold,
        )
        self._onset_high = [False] * slot_count
        self._offset_high = [False] * slot_count
        self._onset_low_counts = [0] * slot_count
        self._offset_low_counts = [0] * slot_count
        self._last_kept_onset_samples: list[Optional[int]] = [None] * slot_count
        self._last_kept_offset_samples: list[Optional[int]] = [None] * slot_count
        self._next_sample: Optional[int] = None

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def onset_threshold(self) -> float:
        return self._onset_threshold

    @property
    def offset_threshold(self) -> float:
        return self._offset_threshold

    @property
    def onset_release_threshold(self) -> float:
        return self._onset_release_threshold

    @property
    def offset_release_threshold(self) -> float:
        return self._offset_release_threshold

    @property
    def rearm_low_samples(self) -> int:
        return self._rearm_low_samples

    @property
    def consolidation_samples(self) -> int:
        return self._consolidation_samples

    @property
    def next_sample(self) -> Optional[int]:
        return self._next_sample

    def process_chunk(
        self,
        scores: BoundaryScoreChunk,
    ) -> Tuple[BoundaryCandidate, ...]:
        """Decode every rising score edge in one contiguous score chunk."""

        if not isinstance(scores, BoundaryScoreChunk):
            raise ValueError("scores must be a BoundaryScoreChunk")
        if scores.sample_count and scores.slot_count != self._slot_count:
            raise ValueError(
                f"expected {self._slot_count} slots, got {scores.slot_count}"
            )
        if self._next_sample is not None and scores.start_sample != self._next_sample:
            raise ValueError(
                f"expected contiguous scores at {self._next_sample}, "
                f"got {scores.start_sample}"
            )

        emitted = []
        for relative_sample, (onset_row, offset_row) in enumerate(
            zip(scores.onset, scores.offset)
        ):
            absolute_sample = scores.start_sample + relative_sample

            for slot, score in enumerate(offset_row):
                if self._offset_high[slot]:
                    if score < self._offset_release_threshold:
                        self._offset_low_counts[slot] += 1
                        if (
                            self._offset_low_counts[slot]
                            >= self._rearm_low_samples
                        ):
                            self._offset_high[slot] = False
                            self._offset_low_counts[slot] = 0
                    else:
                        self._offset_low_counts[slot] = 0
                elif score >= self._offset_threshold:
                    last_kept = self._last_kept_offset_samples[slot]
                    if (
                        self._consolidation_samples == 0
                        or last_kept is None
                        or absolute_sample - last_kept
                        > self._consolidation_samples
                    ):
                        emitted.append(
                            BoundaryCandidate(BoundaryType.OFFSET, absolute_sample)
                        )
                        self._last_kept_offset_samples[slot] = absolute_sample
                    self._offset_high[slot] = True
                    self._offset_low_counts[slot] = 0

            for slot, score in enumerate(onset_row):
                if self._onset_high[slot]:
                    if score < self._onset_release_threshold:
                        self._onset_low_counts[slot] += 1
                        if (
                            self._onset_low_counts[slot]
                            >= self._rearm_low_samples
                        ):
                            self._onset_high[slot] = False
                            self._onset_low_counts[slot] = 0
                    else:
                        self._onset_low_counts[slot] = 0
                elif score >= self._onset_threshold:
                    last_kept = self._last_kept_onset_samples[slot]
                    if (
                        self._consolidation_samples == 0
                        or last_kept is None
                        or absolute_sample - last_kept
                        > self._consolidation_samples
                    ):
                        emitted.append(
                            BoundaryCandidate(BoundaryType.ONSET, absolute_sample)
                        )
                        self._last_kept_onset_samples[slot] = absolute_sample
                    self._onset_high[slot] = True
                    self._onset_low_counts[slot] = 0

        self._next_sample = scores.start_sample + scores.sample_count
        return tuple(emitted)


class LiveModelDetector:
    """Raw-audio live detector backed by a trained causal score predictor."""

    def __init__(
        self,
        predictor: BoundaryScorePredictor,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.5,
        *,
        onset_release_threshold: Optional[float] = None,
        offset_release_threshold: Optional[float] = None,
        event_prefix: str = "event",
    ) -> None:
        slot_count = getattr(predictor, "slot_count", None)
        if (
            isinstance(slot_count, bool)
            or not isinstance(slot_count, int)
            or slot_count <= 0
        ):
            raise ValueError("predictor.slot_count must be an integer > 0")
        if not callable(getattr(predictor, "predict_chunk", None)):
            raise ValueError("predictor must implement predict_chunk")
        self._predictor = predictor
        self._decoder = LiveBoundaryScoreDecoder(
            slot_count,
            onset_threshold,
            offset_threshold,
            onset_release_threshold=onset_release_threshold,
            offset_release_threshold=offset_release_threshold,
            event_prefix=event_prefix,
        )
        self._finalized = False

    @property
    def next_sample(self) -> Optional[int]:
        return self._decoder.next_sample

    @property
    def onset_threshold(self) -> float:
        return self._decoder.onset_threshold

    @property
    def offset_threshold(self) -> float:
        return self._decoder.offset_threshold

    @property
    def onset_release_threshold(self) -> float:
        return self._decoder.onset_release_threshold

    @property
    def offset_release_threshold(self) -> float:
        return self._decoder.offset_release_threshold

    def active_events(self) -> Tuple[ActiveEvent, ...]:
        return self._decoder.active_events()

    def process_chunk(
        self,
        samples: Iterable[float],
        *,
        start_sample: Optional[int] = None,
    ) -> Tuple[BoundaryEvent, ...]:
        """Predict and emit boundaries for one contiguous raw-audio chunk."""

        if self._finalized:
            raise RuntimeError("model detector is finalized")
        values = _samples(samples)
        expected = self._decoder.next_sample
        if expected is None:
            start = 0 if start_sample is None else _index(
                "start_sample", start_sample
            )
        else:
            start = expected if start_sample is None else _index(
                "start_sample", start_sample
            )
            if start != expected:
                raise ValueError(
                    f"expected contiguous chunk at {expected}, got {start}"
                )

        scores = self._predictor.predict_chunk(values, start_sample=start)
        if not isinstance(scores, BoundaryScoreChunk):
            raise ValueError("predictor must return a BoundaryScoreChunk")
        if scores.start_sample != start:
            raise ValueError("predictor returned the wrong start sample")
        if scores.sample_count != len(values):
            raise ValueError("predictor returned the wrong number of samples")
        return self._decoder.process_chunk(scores)

    def finalize_stream(
        self,
        end_sample: Optional[int] = None,
    ) -> Tuple[BoundaryEvent, ...]:
        """Finalize decoder association without predicting padded audio."""

        if self._finalized:
            raise RuntimeError("model detector is already finalized")
        events = self._decoder.finalize_stream(end_sample)
        self._finalized = True
        return events
