"""Data-driven V8.1 causal burst target helpers.

V8.1 replaces exact-sample cardinality with fixed-span anonymous bursts.  The
chosen defaults come from the GuitarSet audit on the locked split:
- 20 ms onset bursts: 52.8% of validation onset instances belong to a
  multi-boundary burst;
- 30 ms offset bursts: 48.7% of validation offset instances belong to a
  multi-boundary burst.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


SAMPLE_RATE = 44_100
DEFAULT_ONSET_HORIZON_MS = 20.0
DEFAULT_OFFSET_HORIZON_MS = 30.0


def milliseconds_to_samples(milliseconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    value = float(milliseconds)
    if value < 0.0:
        raise ValueError("milliseconds must be >= 0")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be an integer > 0")
    return int(round(value * sample_rate / 1000.0))


DEFAULT_ONSET_HORIZON_SAMPLES = milliseconds_to_samples(DEFAULT_ONSET_HORIZON_MS)
DEFAULT_OFFSET_HORIZON_SAMPLES = milliseconds_to_samples(DEFAULT_OFFSET_HORIZON_MS)


@dataclass(frozen=True)
class BoundaryBurst:
    start_sample: int
    end_sample: int
    count: int

    @property
    def count_class(self) -> int:
        """Return 0,1,2 for anonymous cardinality 1,2,3+."""
        return min(self.count, 3) - 1

    @property
    def span_samples(self) -> int:
        return self.end_sample - self.start_sample


def cluster_fixed_span(
    positions: Iterable[int],
    horizon_samples: int,
) -> Tuple[BoundaryBurst, ...]:
    """Cluster sorted boundary instances without chain-growing the horizon.

    A burst starts at the first still-unassigned boundary. Every following
    boundary at most ``horizon_samples`` after that fixed start belongs to the
    same burst. This is intentionally the same definition used by the audit
    that selected the V8.1 horizons.
    """
    if isinstance(horizon_samples, bool) or not isinstance(horizon_samples, int) or horizon_samples < 0:
        raise ValueError("horizon_samples must be an integer >= 0")
    values = tuple(sorted(int(position) for position in positions))
    if any(position < 0 for position in values):
        raise ValueError("boundary positions must be >= 0")
    bursts = []
    index = 0
    while index < len(values):
        start = values[index]
        end_index = index + 1
        while end_index < len(values) and values[end_index] - start <= horizon_samples:
            end_index += 1
        selected = values[index:end_index]
        bursts.append(BoundaryBurst(start, selected[-1], len(selected)))
        index = end_index
    return tuple(bursts)


def response_window_is_empty(
    sorted_positions: Sequence[int],
    start_sample: int,
    horizon_samples: int,
    *,
    margin_samples: int = 0,
) -> bool:
    """Return whether an expanded causal response window contains no boundary."""
    if start_sample < 0:
        return False
    if horizon_samples < 0 or margin_samples < 0:
        raise ValueError("horizon_samples and margin_samples must be >= 0")
    low = max(0, start_sample - margin_samples)
    high = start_sample + horizon_samples + margin_samples
    index = bisect_left(sorted_positions, low)
    return index >= len(sorted_positions) or sorted_positions[index] > high


def training_context_bounds(
    response_start_sample: int,
    receptive_field: int,
    maximum_horizon_samples: int,
):
    """Return source bounds and zero padding for a causal response bag.

    The returned tensor spans enough past context for the response at
    ``response_start_sample`` and enough real-time future progression to score
    through ``maximum_horizon_samples``. Every stream output remains causal.
    """
    if response_start_sample < 0:
        raise ValueError("response_start_sample must be >= 0")
    if receptive_field <= 0 or maximum_horizon_samples < 0:
        raise ValueError("invalid receptive field or maximum horizon")
    source_start = response_start_sample - (receptive_field - 1)
    source_end = response_start_sample + maximum_horizon_samples + 1
    left_padding = max(0, -source_start)
    return max(0, source_start), source_end, left_padding


__all__ = [
    "BoundaryBurst",
    "DEFAULT_OFFSET_HORIZON_MS",
    "DEFAULT_OFFSET_HORIZON_SAMPLES",
    "DEFAULT_ONSET_HORIZON_MS",
    "DEFAULT_ONSET_HORIZON_SAMPLES",
    "cluster_fixed_span",
    "milliseconds_to_samples",
    "response_window_is_empty",
    "training_context_bounds",
]
