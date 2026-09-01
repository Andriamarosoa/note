"""Contiguous absolute-sample buffer for causal 512-sample reframing."""

from array import array
from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple


class AudioBufferError(Exception):
    """Base class for explicit audio-buffer failures."""


class NonContiguousAudioError(AudioBufferError, ValueError):
    """Raised when appended audio would create a gap or overlap."""


class AudioDiscardedError(AudioBufferError, LookupError):
    """Raised when a requested frame begins before retained live audio."""


def _sample_index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value


@dataclass(frozen=True)
class FrameAvailability:
    """Availability of one half-open frame ``[start, end)``."""

    frame_start: int
    frame_end: int
    reusable_samples: int
    stream_samples_until_ready: int

    def __post_init__(self) -> None:
        _sample_index("frame_start", self.frame_start)
        _sample_index("frame_end", self.frame_end)
        _sample_index("reusable_samples", self.reusable_samples)
        _sample_index(
            "stream_samples_until_ready", self.stream_samples_until_ready
        )
        frame_length = self.frame_end - self.frame_start
        if frame_length <= 0:
            raise ValueError("frame_end must be strictly after frame_start")
        if self.stream_samples_until_ready == 0:
            expected_reusable = frame_length
        elif self.stream_samples_until_ready < frame_length:
            expected_reusable = frame_length - self.stream_samples_until_ready
        else:
            expected_reusable = 0
        if self.reusable_samples != expected_reusable:
            raise ValueError("reusable and missing sample counts are inconsistent")

    @property
    def ready(self) -> bool:
        return self.stream_samples_until_ready == 0


class InsufficientAudioError(AudioBufferError):
    """Raised instead of padding when a complete frame is not yet received."""

    def __init__(self, availability: FrameAvailability) -> None:
        self.availability = availability
        super().__init__(
            "frame is not ready: "
            f"{availability.stream_samples_until_ready} additional stream samples "
            "must be received"
        )


class CausalAudioBuffer:
    """Retain a contiguous stream and expose exact frames at absolute offsets.

    Several open sounds may request ``frame_at(their_offset)`` independently.
    """

    def __init__(self, frame_size: int = 512) -> None:
        self._frame_size = _sample_index("frame_size", frame_size)
        if self._frame_size == 0:
            raise ValueError("frame_size must be > 0")
        self._samples = array("f")
        self._start_sample = 0
        self._end_sample = 0

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def start_sample(self) -> int:
        return self._start_sample

    @property
    def end_sample(self) -> int:
        """First sample not yet received (the exclusive stream end)."""

        return self._end_sample

    def __len__(self) -> int:
        return len(self._samples)

    def append(
        self, samples: Iterable[float], *, start_sample: Optional[int] = None
    ) -> None:
        """Append finite samples at exactly the current exclusive stream end."""

        append_start = self._end_sample if start_sample is None else _sample_index(
            "start_sample", start_sample
        )
        if append_start != self._end_sample:
            raise NonContiguousAudioError(
                f"expected append at sample {self._end_sample}, got {append_start}"
            )

        converted = array("f")
        for sample in samples:
            try:
                value = float(sample)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("audio samples must be finite numbers") from exc
            if not math.isfinite(value):
                raise ValueError("audio samples must be finite numbers")
            try:
                converted.append(value)
            except OverflowError as exc:
                raise ValueError("audio samples must fit finite float32") from exc
            if not math.isfinite(converted[-1]):
                raise ValueError("audio samples must fit finite float32")

        self._samples.extend(converted)
        self._end_sample += len(converted)

    def discard_before(self, sample: int) -> None:
        """Compact retained audio while preserving absolute sample indices."""

        position = _sample_index("sample", sample)
        if not self._start_sample <= position <= self._end_sample:
            raise ValueError(
                "discard position must be inside the retained audio interval"
            )
        discard_count = position - self._start_sample
        if discard_count:
            del self._samples[:discard_count]
            self._start_sample = position

    def availability(self, frame_start: int) -> FrameAvailability:
        """Describe reuse and future input needed for a frame at ``frame_start``."""

        start = _sample_index("frame_start", frame_start)
        if start < self._start_sample:
            raise AudioDiscardedError(
                f"audio before sample {self._start_sample} was discarded"
            )
        frame_end = start + self._frame_size
        received_end_in_frame = min(self._end_sample, frame_end)
        reusable = max(0, received_end_in_frame - start)
        stream_samples_until_ready = max(0, frame_end - self._end_sample)
        return FrameAvailability(
            frame_start=start,
            frame_end=frame_end,
            reusable_samples=reusable,
            stream_samples_until_ready=stream_samples_until_ready,
        )

    def frame_at(self, frame_start: int) -> Tuple[float, ...]:
        """Return exactly one frame, or fail explicitly if it is not ready."""

        availability = self.availability(frame_start)
        if not availability.ready:
            raise InsufficientAudioError(availability)

        local_start = availability.frame_start - self._start_sample
        local_end = local_start + self._frame_size
        frame = tuple(self._samples[local_start:local_end])
        if len(frame) != self._frame_size:
            raise AssertionError("internal buffer invariant violated")
        return frame

    def try_frame_at(self, frame_start: int) -> Optional[Tuple[float, ...]]:
        """Return a frame when ready and ``None`` when future audio is required."""

        try:
            return self.frame_at(frame_start)
        except InsufficientAudioError:
            return None
