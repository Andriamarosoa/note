"""Runtime decoding for V8 anonymous hierarchical boundary scores."""
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple
import math


class BoundaryKind(str, Enum):
    ONSET = "onset"
    OFFSET = "offset"


@dataclass(frozen=True)
class AnonymousBoundary:
    kind: BoundaryKind
    sample: int
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryKind):
            raise ValueError("kind must be a BoundaryKind")
        if isinstance(self.sample, bool) or not isinstance(self.sample, int) or self.sample < 0:
            raise ValueError("sample must be an integer >= 0")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("count must be an integer > 0")


@dataclass(frozen=True)
class V8ScoreChunk:
    start_sample: int
    onset_presence: Tuple[float, ...]
    offset_presence: Tuple[float, ...]
    onset_multiplicity: Tuple[Tuple[float, float, float], ...]
    offset_multiplicity: Tuple[Tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if isinstance(self.start_sample, bool) or not isinstance(self.start_sample, int) or self.start_sample < 0:
            raise ValueError("start_sample must be an integer >= 0")
        n = len(self.onset_presence)
        if not (
            len(self.offset_presence) == n
            and len(self.onset_multiplicity) == n
            and len(self.offset_multiplicity) == n
        ):
            raise ValueError("all V8 score sequences must have equal temporal length")
        for probability in self.onset_presence + self.offset_presence:
            value = float(probability)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("presence scores must be finite probabilities")
        for rows in (self.onset_multiplicity, self.offset_multiplicity):
            for row in rows:
                if len(row) != 3:
                    raise ValueError("multiplicity rows must contain exactly 3 classes")
                values = tuple(float(value) for value in row)
                if any(not math.isfinite(value) or value < 0.0 for value in values):
                    raise ValueError("multiplicity scores must be finite and nonnegative")
                if sum(values) <= 0.0:
                    raise ValueError("multiplicity scores must have positive mass")

    @property
    def sample_count(self) -> int:
        return len(self.onset_presence)


def _threshold(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite probability")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 < converted <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return converted


def _decode_count(row: Sequence[float]) -> int:
    return max(range(3), key=lambda index: float(row[index])) + 1


class V8BoundaryDecoder:
    """Convert hierarchical scores to rising-edge anonymous boundaries.

    Presence controls whether a boundary exists. Multiplicity is consulted only
    at a new presence rising edge, so rare-cardinality training cannot create
    additional temporal triggers by itself.
    """

    def __init__(
        self,
        *,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.5,
        onset_release_threshold: Optional[float] = None,
        offset_release_threshold: Optional[float] = None,
    ) -> None:
        self._onset_threshold = _threshold(onset_threshold, "onset_threshold")
        self._offset_threshold = _threshold(offset_threshold, "offset_threshold")
        self._onset_release = (
            self._onset_threshold
            if onset_release_threshold is None
            else _threshold(onset_release_threshold, "onset_release_threshold")
        )
        self._offset_release = (
            self._offset_threshold
            if offset_release_threshold is None
            else _threshold(offset_release_threshold, "offset_release_threshold")
        )
        if self._onset_release > self._onset_threshold:
            raise ValueError("onset_release_threshold must be <= onset_threshold")
        if self._offset_release > self._offset_threshold:
            raise ValueError("offset_release_threshold must be <= offset_threshold")
        self._onset_high = False
        self._offset_high = False
        self._next_sample: Optional[int] = None

    @property
    def next_sample(self) -> Optional[int]:
        return self._next_sample

    def process_chunk(self, scores: V8ScoreChunk) -> Tuple[AnonymousBoundary, ...]:
        if not isinstance(scores, V8ScoreChunk):
            raise ValueError("scores must be a V8ScoreChunk")
        if self._next_sample is not None and scores.start_sample != self._next_sample:
            raise ValueError(
                f"expected contiguous scores at {self._next_sample}, got {scores.start_sample}"
            )

        emitted: List[AnonymousBoundary] = []
        for index in range(scores.sample_count):
            sample = scores.start_sample + index

            offset_score = float(scores.offset_presence[index])
            offset_high = (
                offset_score >= self._offset_release
                if self._offset_high
                else offset_score >= self._offset_threshold
            )
            if offset_high and not self._offset_high:
                emitted.append(
                    AnonymousBoundary(
                        BoundaryKind.OFFSET,
                        sample,
                        _decode_count(scores.offset_multiplicity[index]),
                    )
                )
            self._offset_high = offset_high

            onset_score = float(scores.onset_presence[index])
            onset_high = (
                onset_score >= self._onset_release
                if self._onset_high
                else onset_score >= self._onset_threshold
            )
            if onset_high and not self._onset_high:
                emitted.append(
                    AnonymousBoundary(
                        BoundaryKind.ONSET,
                        sample,
                        _decode_count(scores.onset_multiplicity[index]),
                    )
                )
            self._onset_high = onset_high

        self._next_sample = scores.start_sample + scores.sample_count
        return tuple(emitted)


@dataclass(frozen=True)
class AssociatedBoundary:
    kind: BoundaryKind
    event_id: str
    sample: int


@dataclass(frozen=True)
class OpenAnonymousEvent:
    event_id: str
    onset_sample: int


class AnonymousEventAssociator:
    """Deterministic fallback association for anonymous V8 boundaries.

    Offset multiplicity closes oldest open events first (FIFO). Boundary metrics
    must be reported separately from association metrics so this policy can be
    replaced later without retraining the detector.
    """

    def __init__(self, event_prefix: str = "event") -> None:
        if not isinstance(event_prefix, str) or not event_prefix.strip():
            raise ValueError("event_prefix must be a non-empty string")
        self._prefix = event_prefix
        self._next_id = 1
        self._open: List[OpenAnonymousEvent] = []

    def active_events(self) -> Tuple[OpenAnonymousEvent, ...]:
        return tuple(self._open)

    def process(
        self, boundaries: Iterable[AnonymousBoundary]
    ) -> Tuple[AssociatedBoundary, ...]:
        emitted: List[AssociatedBoundary] = []
        for boundary in boundaries:
            if boundary.kind is BoundaryKind.OFFSET:
                close_count = min(boundary.count, len(self._open))
                for _ in range(close_count):
                    event = self._open.pop(0)
                    emitted.append(
                        AssociatedBoundary(
                            BoundaryKind.OFFSET, event.event_id, boundary.sample
                        )
                    )
            else:
                for _ in range(boundary.count):
                    event_id = f"{self._prefix}-{self._next_id:06d}"
                    self._next_id += 1
                    self._open.append(OpenAnonymousEvent(event_id, boundary.sample))
                    emitted.append(
                        AssociatedBoundary(BoundaryKind.ONSET, event_id, boundary.sample)
                    )
        return tuple(emitted)

    def finalize_stream(self, end_sample: int) -> Tuple[AssociatedBoundary, ...]:
        if isinstance(end_sample, bool) or not isinstance(end_sample, int) or end_sample < 0:
            raise ValueError("end_sample must be an integer >= 0")
        if any(end_sample <= event.onset_sample for event in self._open):
            raise ValueError("end_sample must be after every open onset")
        emitted = tuple(
            AssociatedBoundary(BoundaryKind.OFFSET, event.event_id, end_sample)
            for event in self._open
        )
        self._open.clear()
        return emitted


__all__ = [
    "AnonymousBoundary",
    "AnonymousEventAssociator",
    "AssociatedBoundary",
    "BoundaryKind",
    "OpenAnonymousEvent",
    "V8BoundaryDecoder",
    "V8ScoreChunk",
]
