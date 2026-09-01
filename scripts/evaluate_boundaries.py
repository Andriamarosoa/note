"""Evaluate a trained NOTE model on complete validation tracks.

The evaluator deliberately ignores pitch and string labels in its public
metrics.  GuitarSet string slots are used only to reconstruct reference note
intervals; model events are associated exclusively through their opaque event
identifiers.

The validation set is rebuilt with the same composition-group split as
``train_boundaries.py`` (seed 1337 and validation fraction 0.2 by default).
Player 05 is never an accepted CLI value and is rejected by the shared split
guard before any annotation or audio member could be opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.detector import (  # noqa: E402 - local source bootstrap above
    BoundaryEvent,
    BoundaryType,
    LiveModelDetector,
)
from causal_note.guitarset import (  # noqa: E402
    ALLOWED_PLAYERS,
    NoteBoundary,
    SAMPLE_RATE,
    index_guitarset,
    load_boundary_slots,
)
from causal_note.keras_predictor import KerasBoundaryPredictor  # noqa: E402
from scripts.train_boundaries import (  # noqa: E402
    decode_pcm16_mono_wav,
    group_stem,
    split_tracks_by_group,
)


DEFAULT_SEED = 1337
DEFAULT_VALIDATION_FRACTION = 0.2


class EvaluationError(ValueError):
    """Raised when an evaluation request or artifact is inconsistent."""


@dataclass(frozen=True, order=True)
class EventInterval:
    """One complete anonymous event reconstructed from an opaque event ID."""

    onset_sample: int
    offset_sample: int
    event_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise EvaluationError("event_id must be a non-empty string")
        for name, value in (
            ("onset_sample", self.onset_sample),
            ("offset_sample", self.offset_sample),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvaluationError(f"{name} must be an integer >= 0")
        if self.offset_sample <= self.onset_sample:
            raise EvaluationError("offset_sample must be after onset_sample")


@dataclass(frozen=True)
class CountMetrics:
    reference_count: int
    prediction_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class LatencyMetrics:
    """Matched boundary timing relative to the corresponding reference.

    Signed values include early predictions (negative values).  ``causal_*``
    values include only non-negative delays, which are the delays observable in
    a causal live system rather than anticipatory errors.
    """

    matched_count: int
    early_match_count: int
    causal_match_count: int
    signed_p50_samples: Optional[float]
    signed_p90_samples: Optional[float]
    causal_p50_samples: Optional[float]
    causal_p90_samples: Optional[float]
    causal_max_samples: Optional[int]
    signed_p50_ms: Optional[float]
    signed_p90_ms: Optional[float]
    causal_p50_ms: Optional[float]
    causal_p90_ms: Optional[float]
    causal_max_ms: Optional[float]


@dataclass(frozen=True)
class OnsetMultiplicityMetrics:
    """Exact-sample onset multiplicity without temporal neighbourhood merging."""

    onset_count: int
    unique_position_count: int
    simultaneous_position_count: int
    simultaneous_onset_count: int
    extra_simultaneous_onset_count: int
    maximum_multiplicity: int
    position_histogram: Mapping[str, int]


@dataclass(frozen=True)
class OnsetMultiplicityComparison:
    reference: OnsetMultiplicityMetrics
    prediction: OnsetMultiplicityMetrics
    matched_position_count: int
    exact_multiplicity_match_count: int
    underpredicted_onset_count: int
    overpredicted_onset_count: int
    absolute_multiplicity_error: int


@dataclass(frozen=True)
class TrackEvaluation:
    reference_complete_events: int
    predicted_event_ids: int
    predicted_complete_events: int
    predicted_incomplete_events: int
    predicted_onset_without_offset_events: int
    predicted_offset_without_onset_events: int
    predicted_malformed_events: int
    raw_predicted_onsets: int
    raw_predicted_offsets: int
    onset: CountMetrics
    offset: CountMetrics
    associated_interval: CountMetrics
    onset_latency: LatencyMetrics
    offset_latency: LatencyMetrics
    onset_multiplicity: OnsetMultiplicityComparison


@dataclass(frozen=True)
class PredictionParse:
    intervals: Tuple[EventInterval, ...]
    incomplete_event_ids: Tuple[str, ...]
    onset_without_offset_event_ids: Tuple[str, ...]
    offset_without_onset_event_ids: Tuple[str, ...]
    malformed_event_ids: Tuple[str, ...]
    onset_samples: Tuple[int, ...]
    offset_samples: Tuple[int, ...]
    event_id_count: int


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{name} must be an integer >= 0")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer > 0") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be an integer > 0")
    return value


def _probability(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in (0, 1]") from exc
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("must be a number in (0, 1]")
    return value


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number strictly between 0 and 1") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must be a number strictly between 0 and 1")
    return value


def _milliseconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number >= 0") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return value


def milliseconds_to_samples(milliseconds: float) -> int:
    if (
        isinstance(milliseconds, bool)
        or not isinstance(milliseconds, (int, float))
        or not math.isfinite(float(milliseconds))
        or float(milliseconds) < 0.0
    ):
        raise EvaluationError("milliseconds must be a finite number >= 0")
    return round(float(milliseconds) * SAMPLE_RATE / 1000.0)


def _percentile(values: Sequence[int], percentile: float) -> Optional[float]:
    """Return a deterministic linearly interpolated percentile."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _samples_to_ms(value: Optional[float]) -> Optional[float]:
    return None if value is None else value * 1000.0 / SAMPLE_RATE


def _count_metrics(
    reference_count: int,
    prediction_count: int,
    true_positive: int,
) -> CountMetrics:
    reference_count = _non_negative_int("reference_count", reference_count)
    prediction_count = _non_negative_int("prediction_count", prediction_count)
    true_positive = _non_negative_int("true_positive", true_positive)
    if true_positive > min(reference_count, prediction_count):
        raise EvaluationError("true_positive exceeds the available events")
    false_positive = prediction_count - true_positive
    false_negative = reference_count - true_positive
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / reference_count if reference_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return CountMetrics(
        reference_count=reference_count,
        prediction_count=prediction_count,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _maximum_matching(
    reference_count: int,
    prediction_count: int,
    candidates: Mapping[int, Sequence[int]],
) -> Tuple[Tuple[int, int], ...]:
    """Find a deterministic maximum-cardinality bipartite matching."""

    matched_reference_by_prediction: Dict[int, int] = {}

    def assign(reference_index: int, visited: set) -> bool:
        for prediction_index in candidates.get(reference_index, ()):
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = matched_reference_by_prediction.get(prediction_index)
            if previous is None or assign(previous, visited):
                matched_reference_by_prediction[prediction_index] = reference_index
                return True
        return False

    for reference_index in range(reference_count):
        assign(reference_index, set())
    pairs = tuple(
        sorted(
            (
                (reference_index, prediction_index)
                for prediction_index, reference_index
                in matched_reference_by_prediction.items()
            ),
            key=lambda pair: (pair[0], pair[1]),
        )
    )
    if any(
        reference_index < 0
        or reference_index >= reference_count
        or prediction_index < 0
        or prediction_index >= prediction_count
        for reference_index, prediction_index in pairs
    ):
        raise AssertionError("matching returned an invalid index")
    return pairs


def match_boundaries(
    references: Sequence[int],
    predictions: Sequence[int],
    tolerance_samples: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return matched ``(reference_sample, prediction_sample)`` pairs."""

    tolerance = _non_negative_int("tolerance_samples", tolerance_samples)
    reference_values = tuple(
        _non_negative_int("reference sample", value) for value in references
    )
    prediction_values = tuple(
        _non_negative_int("prediction sample", value) for value in predictions
    )
    candidates = {
        reference_index: tuple(
            sorted(
                (
                    prediction_index
                    for prediction_index, prediction in enumerate(prediction_values)
                    if abs(prediction - reference) <= tolerance
                ),
                key=lambda prediction_index: (
                    abs(prediction_values[prediction_index] - reference),
                    prediction_values[prediction_index],
                    prediction_index,
                ),
            )
        )
        for reference_index, reference in enumerate(reference_values)
    }
    index_pairs = _maximum_matching(
        len(reference_values),
        len(prediction_values),
        candidates,
    )
    return tuple(
        (reference_values[reference_index], prediction_values[prediction_index])
        for reference_index, prediction_index in index_pairs
    )


def match_intervals(
    references: Sequence[NoteBoundary],
    predictions: Sequence[EventInterval],
    onset_tolerance_samples: int,
    offset_tolerance_samples: int,
) -> Tuple[Tuple[int, int], ...]:
    """Match only when one predicted ID satisfies both reference boundaries."""

    onset_tolerance = _non_negative_int(
        "onset_tolerance_samples", onset_tolerance_samples
    )
    offset_tolerance = _non_negative_int(
        "offset_tolerance_samples", offset_tolerance_samples
    )
    reference_values = tuple(references)
    prediction_values = tuple(predictions)
    if any(not isinstance(reference, NoteBoundary) for reference in reference_values):
        raise EvaluationError("references must contain NoteBoundary values")
    if any(not isinstance(prediction, EventInterval) for prediction in prediction_values):
        raise EvaluationError("predictions must contain EventInterval values")
    candidates = {
        reference_index: tuple(
            sorted(
                (
                    prediction_index
                    for prediction_index, prediction in enumerate(prediction_values)
                    if abs(prediction.onset_sample - reference.onset_sample)
                    <= onset_tolerance
                    and abs(prediction.offset_sample - reference.offset_sample)
                    <= offset_tolerance
                ),
                key=lambda prediction_index: (
                    abs(
                        prediction_values[prediction_index].onset_sample
                        - reference.onset_sample
                    )
                    + abs(
                        prediction_values[prediction_index].offset_sample
                        - reference.offset_sample
                    ),
                    prediction_values[prediction_index].onset_sample,
                    prediction_values[prediction_index].offset_sample,
                    prediction_values[prediction_index].event_id,
                ),
            )
        )
        for reference_index, reference in enumerate(reference_values)
    }
    return _maximum_matching(
        len(reference_values),
        len(prediction_values),
        candidates,
    )


def parse_predicted_events(events: Iterable[BoundaryEvent]) -> PredictionParse:
    """Reconstruct complete events and explicitly count malformed/incomplete IDs."""

    grouped: Dict[str, Dict[str, List[int]]] = {}
    onset_samples: List[int] = []
    offset_samples: List[int] = []
    for event in events:
        if not isinstance(event, BoundaryEvent):
            raise EvaluationError("predictions must contain BoundaryEvent values")
        buckets = grouped.setdefault(event.event_id, {"onset": [], "offset": []})
        if event.kind is BoundaryType.ONSET:
            buckets["onset"].append(event.sample)
            onset_samples.append(event.sample)
        elif event.kind is BoundaryType.OFFSET:
            buckets["offset"].append(event.sample)
            offset_samples.append(event.sample)
        else:  # Defensive: BoundaryEvent already validates the enum.
            raise EvaluationError(f"unsupported boundary type: {event.kind!r}")

    complete: List[EventInterval] = []
    incomplete: List[str] = []
    onset_without_offset: List[str] = []
    offset_without_onset: List[str] = []
    malformed: List[str] = []
    for event_id, buckets in sorted(grouped.items()):
        onsets = buckets["onset"]
        offsets = buckets["offset"]
        if len(onsets) == 1 and len(offsets) == 1 and offsets[0] > onsets[0]:
            complete.append(EventInterval(onsets[0], offsets[0], event_id))
        else:
            incomplete.append(event_id)
            if onsets and not offsets:
                onset_without_offset.append(event_id)
            elif offsets and not onsets:
                offset_without_onset.append(event_id)
            else:
                malformed.append(event_id)
    return PredictionParse(
        intervals=tuple(sorted(complete)),
        incomplete_event_ids=tuple(incomplete),
        onset_without_offset_event_ids=tuple(onset_without_offset),
        offset_without_onset_event_ids=tuple(offset_without_onset),
        malformed_event_ids=tuple(malformed),
        onset_samples=tuple(sorted(onset_samples)),
        offset_samples=tuple(sorted(offset_samples)),
        event_id_count=len(grouped),
    )


def onset_multiplicity_metrics(
    onset_samples: Sequence[int],
) -> OnsetMultiplicityMetrics:
    """Summarise multiplicity at exactly equal sample positions.

    Consecutive samples are intentionally separate positions.  This makes the
    result a property of emitted event cardinality rather than a peak-clustering
    heuristic.
    """

    position_counts: Dict[int, int] = {}
    for sample in onset_samples:
        value = _non_negative_int("onset sample", sample)
        position_counts[value] = position_counts.get(value, 0) + 1
    histogram: Dict[str, int] = {}
    for multiplicity in position_counts.values():
        key = str(multiplicity)
        histogram[key] = histogram.get(key, 0) + 1
    simultaneous = tuple(
        multiplicity
        for multiplicity in position_counts.values()
        if multiplicity >= 2
    )
    return OnsetMultiplicityMetrics(
        onset_count=sum(position_counts.values()),
        unique_position_count=len(position_counts),
        simultaneous_position_count=len(simultaneous),
        simultaneous_onset_count=sum(simultaneous),
        extra_simultaneous_onset_count=sum(
            multiplicity - 1 for multiplicity in simultaneous
        ),
        maximum_multiplicity=max(position_counts.values(), default=0),
        position_histogram=dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
    )


def compare_onset_multiplicity(
    reference_onsets: Sequence[int],
    predicted_onsets: Sequence[int],
    tolerance_samples: int,
) -> OnsetMultiplicityComparison:
    """Compare exact-position cardinalities after one-to-one position matching.

    Repeated onsets are grouped only when their sample positions are exactly
    equal.  The tolerance is used solely to associate distinct reference and
    prediction positions; it never clusters neighbouring positions together.
    """

    tolerance = _non_negative_int("tolerance_samples", tolerance_samples)
    reference_counts: Dict[int, int] = {}
    prediction_counts: Dict[int, int] = {}
    for sample in reference_onsets:
        value = _non_negative_int("reference onset sample", sample)
        reference_counts[value] = reference_counts.get(value, 0) + 1
    for sample in predicted_onsets:
        value = _non_negative_int("prediction onset sample", sample)
        prediction_counts[value] = prediction_counts.get(value, 0) + 1

    matched_positions = match_boundaries(
        tuple(sorted(reference_counts)),
        tuple(sorted(prediction_counts)),
        tolerance,
    )
    matched_references = {reference for reference, _ in matched_positions}
    matched_predictions = {prediction for _, prediction in matched_positions}
    underpredicted = sum(
        max(reference_counts[reference] - prediction_counts[prediction], 0)
        for reference, prediction in matched_positions
    ) + sum(
        count
        for position, count in reference_counts.items()
        if position not in matched_references
    )
    overpredicted = sum(
        max(prediction_counts[prediction] - reference_counts[reference], 0)
        for reference, prediction in matched_positions
    ) + sum(
        count
        for position, count in prediction_counts.items()
        if position not in matched_predictions
    )
    return OnsetMultiplicityComparison(
        reference=onset_multiplicity_metrics(reference_onsets),
        prediction=onset_multiplicity_metrics(predicted_onsets),
        matched_position_count=len(matched_positions),
        exact_multiplicity_match_count=sum(
            reference_counts[reference] == prediction_counts[prediction]
            for reference, prediction in matched_positions
        ),
        underpredicted_onset_count=underpredicted,
        overpredicted_onset_count=overpredicted,
        absolute_multiplicity_error=underpredicted + overpredicted,
    )


def latency_metrics(
    matched_boundaries: Sequence[Tuple[int, int]],
) -> LatencyMetrics:
    delays = [prediction - reference for reference, prediction in matched_boundaries]
    causal_delays = [delay for delay in delays if delay >= 0]
    signed_p50 = _percentile(delays, 0.5)
    signed_p90 = _percentile(delays, 0.9)
    causal_p50 = _percentile(causal_delays, 0.5)
    causal_p90 = _percentile(causal_delays, 0.9)
    causal_max = max(causal_delays) if causal_delays else None
    return LatencyMetrics(
        matched_count=len(delays),
        early_match_count=sum(delay < 0 for delay in delays),
        causal_match_count=len(causal_delays),
        signed_p50_samples=signed_p50,
        signed_p90_samples=signed_p90,
        causal_p50_samples=causal_p50,
        causal_p90_samples=causal_p90,
        causal_max_samples=causal_max,
        signed_p50_ms=_samples_to_ms(signed_p50),
        signed_p90_ms=_samples_to_ms(signed_p90),
        causal_p50_ms=_samples_to_ms(causal_p50),
        causal_p90_ms=_samples_to_ms(causal_p90),
        causal_max_ms=_samples_to_ms(causal_max),
    )


def evaluate_track_events(
    references: Sequence[NoteBoundary],
    predictions: Iterable[BoundaryEvent],
    *,
    onset_tolerance_samples: int,
    offset_tolerance_samples: int,
) -> TrackEvaluation:
    """Evaluate one track without exposing GuitarSet slots as output labels."""

    reference_values = tuple(sorted(references))
    if any(not isinstance(reference, NoteBoundary) for reference in reference_values):
        raise EvaluationError("references must contain NoteBoundary values")
    parsed = parse_predicted_events(predictions)
    reference_onsets = tuple(reference.onset_sample for reference in reference_values)
    reference_offsets = tuple(reference.offset_sample for reference in reference_values)
    onset_pairs = match_boundaries(
        reference_onsets,
        parsed.onset_samples,
        onset_tolerance_samples,
    )
    offset_pairs = match_boundaries(
        reference_offsets,
        parsed.offset_samples,
        offset_tolerance_samples,
    )
    interval_pairs = match_intervals(
        reference_values,
        parsed.intervals,
        onset_tolerance_samples,
        offset_tolerance_samples,
    )
    return TrackEvaluation(
        reference_complete_events=len(reference_values),
        predicted_event_ids=parsed.event_id_count,
        predicted_complete_events=len(parsed.intervals),
        predicted_incomplete_events=len(parsed.incomplete_event_ids),
        predicted_onset_without_offset_events=len(
            parsed.onset_without_offset_event_ids
        ),
        predicted_offset_without_onset_events=len(
            parsed.offset_without_onset_event_ids
        ),
        predicted_malformed_events=len(parsed.malformed_event_ids),
        raw_predicted_onsets=len(parsed.onset_samples),
        raw_predicted_offsets=len(parsed.offset_samples),
        onset=_count_metrics(
            len(reference_onsets), len(parsed.onset_samples), len(onset_pairs)
        ),
        offset=_count_metrics(
            len(reference_offsets), len(parsed.offset_samples), len(offset_pairs)
        ),
        associated_interval=_count_metrics(
            len(reference_values), len(parsed.intervals), len(interval_pairs)
        ),
        onset_latency=latency_metrics(onset_pairs),
        offset_latency=latency_metrics(offset_pairs),
        onset_multiplicity=compare_onset_multiplicity(
            reference_onsets,
            parsed.onset_samples,
            onset_tolerance_samples,
        ),
    )


def _flatten_references(slots) -> Tuple[NoteBoundary, ...]:
    return tuple(sorted(note for slot in slots for note in slot))


def _metadata_path(model_path: Path, explicit: Optional[Path]) -> Path:
    return explicit if explicit is not None else model_path.with_suffix(".metadata.json")


def _load_and_validate_metadata(
    metadata_path: Path,
    *,
    model_path: Path,
    validation_members: Sequence[str],
    selected_players: Sequence[str],
    seed: int,
    validation_fraction: float,
) -> Mapping[str, object]:
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read training metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise EvaluationError("training metadata root must be an object")
    recorded_model = metadata.get("model_path")
    if recorded_model is not None and Path(str(recorded_model)).resolve() != model_path:
        raise EvaluationError("requested model does not match training metadata model_path")
    if metadata.get("seed") != seed:
        raise EvaluationError(
            f"evaluation seed {seed} does not match training metadata seed "
            f"{metadata.get('seed')!r}"
        )
    metadata_players = metadata.get("selected_players")
    if list(selected_players) != metadata_players:
        raise EvaluationError(
            "evaluation players do not match training metadata selected_players"
        )
    split = metadata.get("split")
    if not isinstance(split, dict):
        raise EvaluationError("training metadata has no split object")
    metadata_fraction = metadata.get("validation_fraction")
    if metadata_fraction is None:
        # Existing metadata records the members but not the fraction directly.
        metadata_fraction = validation_fraction
    if not math.isclose(float(metadata_fraction), validation_fraction):
        raise EvaluationError("evaluation validation fraction does not match metadata")
    if list(validation_members) != split.get("validation_members"):
        raise EvaluationError(
            "reconstructed validation members do not match training metadata"
        )
    return metadata


def _receptive_field(metadata: Mapping[str, object]) -> int:
    direct = metadata.get("receptive_field")
    model = metadata.get("model")
    nested = model.get("receptive_field_samples") if isinstance(model, dict) else None
    value = direct if direct is not None else nested
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationError("metadata must contain a positive model receptive field")
    return value


def _aggregate_count_metrics(
    track_evaluations: Sequence[TrackEvaluation],
    attribute: str,
) -> CountMetrics:
    metrics = [getattr(track, attribute) for track in track_evaluations]
    return _count_metrics(
        sum(item.reference_count for item in metrics),
        sum(item.prediction_count for item in metrics),
        sum(item.true_positive for item in metrics),
    )


def _aggregate_latency(
    matched_pairs_by_track: Sequence[Sequence[Tuple[int, int]]],
) -> LatencyMetrics:
    return latency_metrics(
        tuple(pair for track_pairs in matched_pairs_by_track for pair in track_pairs)
    )


def _aggregate_multiplicity(
    metrics_by_track: Sequence[OnsetMultiplicityMetrics],
) -> OnsetMultiplicityMetrics:
    histogram: Dict[str, int] = {}
    for metrics in metrics_by_track:
        for multiplicity, position_count in metrics.position_histogram.items():
            histogram[multiplicity] = histogram.get(multiplicity, 0) + position_count
    return OnsetMultiplicityMetrics(
        onset_count=sum(item.onset_count for item in metrics_by_track),
        unique_position_count=sum(
            item.unique_position_count for item in metrics_by_track
        ),
        simultaneous_position_count=sum(
            item.simultaneous_position_count for item in metrics_by_track
        ),
        simultaneous_onset_count=sum(
            item.simultaneous_onset_count for item in metrics_by_track
        ),
        extra_simultaneous_onset_count=sum(
            item.extra_simultaneous_onset_count for item in metrics_by_track
        ),
        maximum_multiplicity=max(
            (item.maximum_multiplicity for item in metrics_by_track),
            default=0,
        ),
        position_histogram=dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
    )


def _aggregate_multiplicity_comparisons(
    comparisons: Sequence[OnsetMultiplicityComparison],
) -> OnsetMultiplicityComparison:
    return OnsetMultiplicityComparison(
        reference=_aggregate_multiplicity(
            tuple(item.reference for item in comparisons)
        ),
        prediction=_aggregate_multiplicity(
            tuple(item.prediction for item in comparisons)
        ),
        matched_position_count=sum(
            item.matched_position_count for item in comparisons
        ),
        exact_multiplicity_match_count=sum(
            item.exact_multiplicity_match_count for item in comparisons
        ),
        underpredicted_onset_count=sum(
            item.underpredicted_onset_count for item in comparisons
        ),
        overpredicted_onset_count=sum(
            item.overpredicted_onset_count for item in comparisons
        ),
        absolute_multiplicity_error=sum(
            item.absolute_multiplicity_error for item in comparisons
        ),
    )


def _aggregate_counts(
    track_evaluations: Sequence[TrackEvaluation],
) -> Dict[str, int]:
    return {
        "tracks": len(track_evaluations),
        "reference_complete_events": sum(
            item.reference_complete_events for item in track_evaluations
        ),
        "predicted_event_ids": sum(
            item.predicted_event_ids for item in track_evaluations
        ),
        "predicted_complete_events": sum(
            item.predicted_complete_events for item in track_evaluations
        ),
        "predicted_incomplete_events": sum(
            item.predicted_incomplete_events for item in track_evaluations
        ),
        "predicted_onset_without_offset_events": sum(
            item.predicted_onset_without_offset_events
            for item in track_evaluations
        ),
        "predicted_offset_without_onset_events": sum(
            item.predicted_offset_without_onset_events
            for item in track_evaluations
        ),
        "predicted_malformed_events": sum(
            item.predicted_malformed_events for item in track_evaluations
        ),
        "raw_predicted_onsets": sum(
            item.raw_predicted_onsets for item in track_evaluations
        ),
        "raw_predicted_offsets": sum(
            item.raw_predicted_offsets for item in track_evaluations
        ),
    }


def _rate_per_hour(count: int, duration_hours: float) -> Optional[float]:
    return count / duration_hours if duration_hours > 0.0 else None


def aggregate_track_evaluations(
    track_evaluations: Sequence[TrackEvaluation],
    audio_samples_by_track: Sequence[int],
    matched_onsets_by_track: Sequence[Sequence[Tuple[int, int]]],
    matched_offsets_by_track: Sequence[Sequence[Tuple[int, int]]],
) -> Dict[str, object]:
    """Aggregate raw track results into one duration-normalised regime.

    Latency percentiles are computed after concatenating raw matched pairs.
    They are never averages of per-track percentiles.
    """

    evaluations = tuple(track_evaluations)
    audio_samples = tuple(
        _non_negative_int("audio samples", value)
        for value in audio_samples_by_track
    )
    onset_pairs = tuple(tuple(pairs) for pairs in matched_onsets_by_track)
    offset_pairs = tuple(tuple(pairs) for pairs in matched_offsets_by_track)
    lengths = {
        len(evaluations),
        len(audio_samples),
        len(onset_pairs),
        len(offset_pairs),
    }
    if len(lengths) != 1:
        raise EvaluationError("aggregate inputs must contain the same tracks")

    counts = _aggregate_counts(evaluations)
    onset = _aggregate_count_metrics(evaluations, "onset")
    offset = _aggregate_count_metrics(evaluations, "offset")
    associated_interval = _aggregate_count_metrics(
        evaluations, "associated_interval"
    )
    total_audio_samples = sum(audio_samples)
    duration_seconds = total_audio_samples / SAMPLE_RATE
    duration_hours = duration_seconds / 3600.0
    false_event_ids = (
        counts["predicted_event_ids"] - associated_interval.true_positive
    )
    if false_event_ids < 0:
        raise AssertionError("associated true positives exceed predicted event IDs")
    multiplicity = _aggregate_multiplicity_comparisons(
        tuple(item.onset_multiplicity for item in evaluations)
    )
    return {
        "duration": {
            "audio_samples": total_audio_samples,
            "audio_seconds": duration_seconds,
            "audio_hours": duration_hours,
        },
        "counts": counts,
        "metrics": {
            "onset": asdict(onset),
            "offset": asdict(offset),
            "associated_interval": asdict(associated_interval),
            "onset_latency": asdict(_aggregate_latency(onset_pairs)),
            "offset_latency": asdict(_aggregate_latency(offset_pairs)),
            "onset_multiplicity": asdict(multiplicity),
        },
        "rates_per_hour": {
            "false_onsets": _rate_per_hour(onset.false_positive, duration_hours),
            "false_offsets": _rate_per_hour(offset.false_positive, duration_hours),
            "false_complete_intervals": _rate_per_hour(
                associated_interval.false_positive, duration_hours
            ),
            "orphan_ids": _rate_per_hour(
                counts["predicted_incomplete_events"], duration_hours
            ),
            "false_event_ids": _rate_per_hour(false_event_ids, duration_hours),
        },
    }


def _track_arrangement(annotation_member: str) -> Optional[str]:
    stem = Path(annotation_member).stem
    for arrangement in ("comp", "solo"):
        if stem.endswith(f"_{arrangement}"):
            return arrangement
    return None


def run_evaluation(arguments: argparse.Namespace) -> Dict[str, object]:
    """Run full-track streaming inference on the locked validation partition."""

    dataset_dir = Path(arguments.dataset_dir).resolve()
    model_path = Path(arguments.model).resolve()
    metadata_path = _metadata_path(model_path, arguments.metadata).resolve()
    output_path = (
        Path(arguments.output).resolve() if arguments.output is not None else None
    )
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"refusing to replace evaluation output: {output_path}")
    selected_players = tuple(dict.fromkeys(arguments.players))
    if not selected_players or any(player not in ALLOWED_PLAYERS for player in selected_players):
        raise EvaluationError("players must be selected from 00 through 04")
    if arguments.seed != DEFAULT_SEED:
        raise EvaluationError(
            f"full-track evaluation is locked to split seed {DEFAULT_SEED}"
        )

    indexed = tuple(
        track
        for track in index_guitarset(dataset_dir)
        if track.player_id in selected_players
    )
    _, validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
    validation_members = tuple(track.annotation_member for track in validation_tracks)
    metadata = _load_and_validate_metadata(
        metadata_path,
        model_path=model_path,
        validation_members=validation_members,
        selected_players=selected_players,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
    )

    predictor = KerasBoundaryPredictor.from_path(
        str(model_path),
        receptive_field=_receptive_field(metadata),
    )
    predictor.warm_up(arguments.chunk_size)
    onset_tolerance_samples = milliseconds_to_samples(
        arguments.onset_tolerance_ms
    )
    offset_tolerance_samples = milliseconds_to_samples(
        arguments.offset_tolerance_ms
    )
    onset_release_threshold = (
        arguments.onset_threshold
        if arguments.onset_release_threshold is None
        else arguments.onset_release_threshold
    )
    offset_release_threshold = (
        arguments.offset_threshold
        if arguments.offset_release_threshold is None
        else arguments.offset_release_threshold
    )

    track_results: List[Dict[str, object]] = []
    track_evaluations: List[TrackEvaluation] = []
    matched_onsets_by_track: List[Tuple[Tuple[int, int], ...]] = []
    matched_offsets_by_track: List[Tuple[Tuple[int, int], ...]] = []
    audio_samples_by_track: List[int] = []
    arrangements_by_track: List[Optional[str]] = []
    inference_durations_ms: List[float] = []
    for track in validation_tracks:
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        references = _flatten_references(slots)
        decoded = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        if any(
            reference.offset_sample > decoded.frame_count
            for reference in references
        ):
            raise EvaluationError(
                f"reference boundary exceeds audio length for {track.annotation_member!r}"
            )
        audio_samples_by_track.append(decoded.frame_count)
        arrangement = _track_arrangement(track.annotation_member)
        arrangements_by_track.append(arrangement)
        predictor.reset()
        detector = LiveModelDetector(
            predictor,
            onset_threshold=arguments.onset_threshold,
            offset_threshold=arguments.offset_threshold,
            onset_release_threshold=arguments.onset_release_threshold,
            offset_release_threshold=arguments.offset_release_threshold,
        )
        predictions: List[BoundaryEvent] = []
        for start in range(0, decoded.frame_count, arguments.chunk_size):
            integer_chunk = decoded.samples[start : start + arguments.chunk_size]
            samples = tuple(sample / 32768.0 for sample in integer_chunk)
            started = time.perf_counter()
            predictions.extend(detector.process_chunk(samples, start_sample=start))
            inference_durations_ms.append((time.perf_counter() - started) * 1000.0)

        evaluation = evaluate_track_events(
            references,
            predictions,
            onset_tolerance_samples=onset_tolerance_samples,
            offset_tolerance_samples=offset_tolerance_samples,
        )
        track_evaluations.append(evaluation)
        matched_onsets = match_boundaries(
            tuple(reference.onset_sample for reference in references),
            tuple(
                event.sample
                for event in predictions
                if event.kind is BoundaryType.ONSET
            ),
            onset_tolerance_samples,
        )
        matched_onsets_by_track.append(matched_onsets)
        matched_offsets = match_boundaries(
            tuple(reference.offset_sample for reference in references),
            tuple(
                event.sample
                for event in predictions
                if event.kind is BoundaryType.OFFSET
            ),
            offset_tolerance_samples,
        )
        matched_offsets_by_track.append(matched_offsets)
        track_results.append(
            {
                "annotation_member": track.annotation_member,
                "audio_member": track.audio_member,
                "arrangement": arrangement,
                "audio_samples": decoded.frame_count,
                "metrics": asdict(evaluation),
            }
        )

    global_aggregate = aggregate_track_evaluations(
        track_evaluations,
        audio_samples_by_track,
        matched_onsets_by_track,
        matched_offsets_by_track,
    )
    aggregates: Dict[str, Dict[str, object]] = {"global": global_aggregate}
    for arrangement in ("comp", "solo"):
        indices = tuple(
            index
            for index, value in enumerate(arrangements_by_track)
            if value == arrangement
        )
        aggregates[arrangement] = aggregate_track_evaluations(
            tuple(track_evaluations[index] for index in indices),
            tuple(audio_samples_by_track[index] for index in indices),
            tuple(matched_onsets_by_track[index] for index in indices),
            tuple(matched_offsets_by_track[index] for index in indices),
        )
    elapsed_inference_ms = sum(inference_durations_ms)
    total_audio_samples = sum(audio_samples_by_track)
    audio_duration_seconds = total_audio_samples / SAMPLE_RATE
    result: Dict[str, object] = {
        "schema_version": 2,
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "split": {
            "seed": arguments.seed,
            "validation_fraction": arguments.validation_fraction,
            "players": list(selected_players),
            "player_05_read": False,
            "validation_tracks": len(validation_tracks),
            "validation_groups": len(
                {group_stem(track) for track in validation_tracks}
            ),
            "validation_members": list(validation_members),
        },
        "configuration": {
            "chunk_size": arguments.chunk_size,
            "onset_threshold": arguments.onset_threshold,
            "offset_threshold": arguments.offset_threshold,
            "onset_release_threshold": onset_release_threshold,
            "offset_release_threshold": offset_release_threshold,
            "onset_tolerance_ms": arguments.onset_tolerance_ms,
            "onset_tolerance_samples": onset_tolerance_samples,
            "offset_tolerance_ms": arguments.offset_tolerance_ms,
            "offset_tolerance_samples": offset_tolerance_samples,
        },
        # Root aliases preserve the schema-v1 access paths while v2 exposes
        # duration-normalised global/comp/solo aggregates below.
        "counts": global_aggregate["counts"],
        "metrics": global_aggregate["metrics"],
        "rates_per_hour": global_aggregate["rates_per_hour"],
        "aggregates": aggregates,
        "runtime": {
            "audio_duration_seconds": audio_duration_seconds,
            "inference_elapsed_seconds": elapsed_inference_ms / 1000.0,
            "realtime_factor": (
                elapsed_inference_ms / 1000.0 / audio_duration_seconds
                if audio_duration_seconds
                else None
            ),
            "chunks": len(inference_durations_ms),
            "chunk_compute_p50_ms": _percentile(
                [round(value * 1000) for value in inference_durations_ms], 0.5
            )
            / 1000.0
            if inference_durations_ms
            else None,
            "chunk_compute_p95_ms": _percentile(
                [round(value * 1000) for value in inference_durations_ms], 0.95
            )
            / 1000.0
            if inference_durations_ms
            else None,
            "chunk_compute_max_ms": max(inference_durations_ms)
            if inference_durations_ms
            else None,
        },
        "tracks": track_results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        result["output_path"] = str(output_path)
    return result


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate anonymous onset/offset associations over complete "
            "GuitarSet validation tracks."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "GuitarSet",
        help="directory containing the two GuitarSet ZIP archives",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPOSITORY_ROOT / "model" / "causal-boundaries.keras",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="training metadata JSON (default: beside --model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new JSON file; existing files are never overwritten",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=_fraction,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--players",
        nargs="+",
        choices=sorted(ALLOWED_PLAYERS),
        default=sorted(ALLOWED_PLAYERS),
        help="allowed development players; player 05 cannot be selected",
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=512)
    parser.add_argument("--onset-threshold", type=_probability, default=0.5)
    parser.add_argument("--offset-threshold", type=_probability, default=0.5)
    parser.add_argument("--onset-release-threshold", type=_probability)
    parser.add_argument("--offset-release-threshold", type=_probability)
    parser.add_argument("--onset-tolerance-ms", type=_milliseconds, default=50.0)
    parser.add_argument("--offset-tolerance-ms", type=_milliseconds, default=50.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    result = run_evaluation(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
