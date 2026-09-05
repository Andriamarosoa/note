"""Reusable V8 exact-point context and label assembly helpers."""
from dataclasses import dataclass
from typing import Mapping, Tuple

from .v8_targets import exact_count_to_hierarchical


@dataclass(frozen=True)
class V8PointExample:
    track_index: int
    position: int
    onset_count: int
    offset_count: int
    stratum: str
    presence_weight: float
    multiplicity_weight: float = 1.0


def causal_context_bounds(position: int, receptive_field: int) -> Tuple[int, int, int]:
    """Return source [start,end) and left zero padding for query sample t."""

    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("position must be an integer >= 0")
    if (
        isinstance(receptive_field, bool)
        or not isinstance(receptive_field, int)
        or receptive_field <= 0
    ):
        raise ValueError("receptive_field must be an integer > 0")
    end = position + 1
    start = max(0, end - receptive_field)
    left_padding = receptive_field - (end - start)
    return start, end, left_padding


def hierarchical_targets(example: V8PointExample) -> Mapping[str, int]:
    onset = exact_count_to_hierarchical(example.onset_count)
    offset = exact_count_to_hierarchical(example.offset_count)
    return {
        "onset_presence": onset.presence,
        "offset_presence": offset.presence,
        "onset_multiplicity": onset.multiplicity_class,
        "offset_multiplicity": offset.multiplicity_class,
    }


def hierarchical_weights(example: V8PointExample) -> Mapping[str, float]:
    """Presence follows live-prior correction; multiplicity exists only if present."""

    return {
        "onset_presence": float(example.presence_weight),
        "offset_presence": float(example.presence_weight),
        "onset_multiplicity": (
            float(example.multiplicity_weight) if example.onset_count > 0 else 0.0
        ),
        "offset_multiplicity": (
            float(example.multiplicity_weight) if example.offset_count > 0 else 0.0
        ),
    }


__all__ = [
    "V8PointExample",
    "causal_context_bounds",
    "hierarchical_targets",
    "hierarchical_weights",
]
