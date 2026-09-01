"""Compare associated V7 decoding with unassociated boundary candidates.

Each audio track and each model score chunk is produced exactly once.  The
same immutable :class:`BoundaryScoreChunk` instance is delivered first to the
official associated decoder and then to the unassociated peak decoder.  This
script reports boundary metrics only: it deliberately has no event-interval
metric and never invokes the restart scheduler.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.detector import (  # noqa: E402 - local bootstrap above
    BoundaryCandidate,
    BoundaryEvent,
    BoundaryScoreChunk,
    BoundaryType,
    LiveBoundaryPeakDecoder,
    LiveBoundaryScoreDecoder,
)
from causal_note.guitarset import (  # noqa: E402
    ALLOWED_PLAYERS,
    NoteBoundary,
    SAMPLE_RATE,
    index_guitarset,
    load_boundary_slots,
)
from causal_note.keras_predictor import KerasBoundaryPredictor  # noqa: E402
from scripts.evaluate_boundaries import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    CountMetrics,
    EvaluationError,
    LatencyMetrics,
    OnsetMultiplicityComparison,
    _aggregate_multiplicity_comparisons,
    _count_metrics,
    _flatten_references,
    _fraction,
    _load_and_validate_metadata,
    _metadata_path,
    _milliseconds,
    _positive_int,
    _probability,
    _receptive_field,
    _track_arrangement,
    compare_onset_multiplicity,
    latency_metrics,
    match_boundaries,
    milliseconds_to_samples,
)
from scripts.train_boundaries import (  # noqa: E402
    decode_pcm16_mono_wav,
    group_stem,
    split_tracks_by_group,
)


OFFICIAL_THRESHOLD = 0.55
LOCKED_PLAYERS = tuple(sorted(ALLOWED_PLAYERS))
DEFAULT_MODEL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7.epochs"
    / "epoch-08.keras"
)
DEFAULT_METADATA = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.recovery.metadata.json"
)


@dataclass(frozen=True)
class BoundaryOnlyEvaluation:
    """Metrics for independent onset and offset lists, without association."""

    onset: CountMetrics
    offset: CountMetrics
    onset_latency: LatencyMetrics
    offset_latency: LatencyMetrics
    onset_multiplicity: OnsetMultiplicityComparison


@dataclass(frozen=True)
class SharedStreamDecoding:
    """Outputs from two decoders consuming one shared stream of score chunks."""

    control_events: Tuple[BoundaryEvent, ...]
    candidates: Tuple[BoundaryCandidate, ...]
    control_open_events: int
    chunks: int
    inference_elapsed_ns: int
    control_decoding_elapsed_ns: int
    candidate_decoding_elapsed_ns: int


@dataclass(frozen=True)
class TrackOutcome:
    annotation_member: str
    audio_member: str
    family: str
    arrangement: str
    audio_samples: int
    control: BoundaryOnlyEvaluation
    treatment: BoundaryOnlyEvaluation
    control_open_events: int
    control_onset_pairs: Tuple[Tuple[int, int], ...]
    control_offset_pairs: Tuple[Tuple[int, int], ...]
    treatment_onset_pairs: Tuple[Tuple[int, int], ...]
    treatment_offset_pairs: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class SourceExpectations:
    """Canonical integer counts extracted from a source report or protocol."""

    global_metrics: Mapping[str, Mapping[str, int]]
    global_open_events: Optional[int]
    tracks: Mapping[str, Mapping[str, object]]
    kind: str


def _same_probability(left: object, right: float) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
    )


def _locked_thresholds(arguments: argparse.Namespace) -> None:
    values = {
        "onset threshold": arguments.onset_threshold,
        "offset threshold": arguments.offset_threshold,
        "onset release threshold": arguments.onset_release_threshold,
        "offset release threshold": arguments.offset_release_threshold,
    }
    for name, value in values.items():
        if not _same_probability(value, OFFICIAL_THRESHOLD):
            raise EvaluationError(f"{name} is locked to {OFFICIAL_THRESHOLD}")


def _audio_chunks(decoded, chunk_size: int):
    for start in range(0, decoded.frame_count, chunk_size):
        integer_chunk = decoded.samples[start : start + chunk_size]
        yield start, tuple(sample / 32768.0 for sample in integer_chunk)


def _decode_shared_score_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    control_decoder,
    candidate_decoder,
) -> SharedStreamDecoding:
    """Infer once and pass the identical score object to both decoders."""

    control_events: List[BoundaryEvent] = []
    candidates: List[BoundaryCandidate] = []
    inference_elapsed_ns = 0
    control_decoding_elapsed_ns = 0
    candidate_decoding_elapsed_ns = 0
    chunk_count = 0

    for start_sample, samples in chunks:
        inference_started = time.perf_counter_ns()
        scores = predictor.predict_chunk(samples, start_sample=start_sample)
        inference_elapsed_ns += time.perf_counter_ns() - inference_started
        if not isinstance(scores, BoundaryScoreChunk):
            raise EvaluationError("predictor must return a BoundaryScoreChunk")
        if scores.start_sample != start_sample:
            raise EvaluationError("predictor returned the wrong start sample")
        if scores.sample_count != len(samples):
            raise EvaluationError("predictor returned the wrong number of samples")
        chunk_count += 1

        control_started = time.perf_counter_ns()
        control_events.extend(control_decoder.process_chunk(scores))
        control_decoding_elapsed_ns += time.perf_counter_ns() - control_started

        candidate_started = time.perf_counter_ns()
        candidates.extend(candidate_decoder.process_chunk(scores))
        candidate_decoding_elapsed_ns += (
            time.perf_counter_ns() - candidate_started
        )

    active_events = getattr(control_decoder, "active_events", None)
    open_event_count = len(active_events()) if callable(active_events) else 0
    return SharedStreamDecoding(
        control_events=tuple(control_events),
        candidates=tuple(candidates),
        control_open_events=open_event_count,
        chunks=chunk_count,
        inference_elapsed_ns=inference_elapsed_ns,
        control_decoding_elapsed_ns=control_decoding_elapsed_ns,
        candidate_decoding_elapsed_ns=candidate_decoding_elapsed_ns,
    )


def decode_candidate_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    *,
    onset_threshold: float = OFFICIAL_THRESHOLD,
    offset_threshold: float = OFFICIAL_THRESHOLD,
    onset_release_threshold: float = OFFICIAL_THRESHOLD,
    offset_release_threshold: float = OFFICIAL_THRESHOLD,
) -> SharedStreamDecoding:
    """Create both V7 decoders and consume one shared model-score stream."""

    slot_count = getattr(predictor, "slot_count", None)
    if (
        isinstance(slot_count, bool)
        or not isinstance(slot_count, int)
        or slot_count <= 0
    ):
        raise EvaluationError("predictor must expose a positive slot_count")
    if not callable(getattr(predictor, "predict_chunk", None)):
        raise EvaluationError("predictor must implement predict_chunk")

    control_decoder = LiveBoundaryScoreDecoder(
        slot_count=slot_count,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        onset_release_threshold=onset_release_threshold,
        offset_release_threshold=offset_release_threshold,
    )
    candidate_decoder = LiveBoundaryPeakDecoder(
        slot_count=slot_count,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        onset_release_threshold=onset_release_threshold,
        offset_release_threshold=offset_release_threshold,
    )
    return _decode_shared_score_stream(
        predictor,
        chunks,
        control_decoder,
        candidate_decoder,
    )


def _boundary_samples(
    predictions: Iterable[object],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    onsets: List[int] = []
    offsets: List[int] = []
    for prediction in predictions:
        kind = getattr(prediction, "kind", None)
        sample = getattr(prediction, "sample", None)
        if not isinstance(kind, BoundaryType):
            raise EvaluationError("prediction kind must be a BoundaryType")
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            raise EvaluationError("prediction sample must be an integer >= 0")
        if kind is BoundaryType.ONSET:
            onsets.append(sample)
        elif kind is BoundaryType.OFFSET:
            offsets.append(sample)
    return tuple(sorted(onsets)), tuple(sorted(offsets))


def evaluate_boundary_lists(
    references: Sequence[NoteBoundary],
    predicted_onsets: Sequence[int],
    predicted_offsets: Sequence[int],
    *,
    onset_tolerance_samples: int,
    offset_tolerance_samples: int,
) -> Tuple[
    BoundaryOnlyEvaluation,
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
]:
    """Evaluate multiplicity-preserving boundary lists without pairing them."""

    reference_values = tuple(sorted(references))
    if any(not isinstance(reference, NoteBoundary) for reference in reference_values):
        raise EvaluationError("references must contain NoteBoundary values")
    reference_onsets = tuple(
        reference.onset_sample for reference in reference_values
    )
    reference_offsets = tuple(
        reference.offset_sample for reference in reference_values
    )
    onset_pairs = match_boundaries(
        reference_onsets,
        predicted_onsets,
        onset_tolerance_samples,
    )
    offset_pairs = match_boundaries(
        reference_offsets,
        predicted_offsets,
        offset_tolerance_samples,
    )
    evaluation = BoundaryOnlyEvaluation(
        onset=_count_metrics(
            len(reference_onsets), len(predicted_onsets), len(onset_pairs)
        ),
        offset=_count_metrics(
            len(reference_offsets), len(predicted_offsets), len(offset_pairs)
        ),
        onset_latency=latency_metrics(onset_pairs),
        offset_latency=latency_metrics(offset_pairs),
        onset_multiplicity=compare_onset_multiplicity(
            reference_onsets,
            predicted_onsets,
            onset_tolerance_samples,
        ),
    )
    return evaluation, onset_pairs, offset_pairs


def _aggregate_regime(
    outcomes: Sequence[TrackOutcome],
    regime: str,
) -> Dict[str, object]:
    if regime not in ("control", "treatment"):
        raise EvaluationError("regime must be control or treatment")
    evaluations = tuple(getattr(outcome, regime) for outcome in outcomes)
    onset_pairs = tuple(
        getattr(outcome, f"{regime}_onset_pairs") for outcome in outcomes
    )
    offset_pairs = tuple(
        getattr(outcome, f"{regime}_offset_pairs") for outcome in outcomes
    )
    onset = _count_metrics(
        sum(item.onset.reference_count for item in evaluations),
        sum(item.onset.prediction_count for item in evaluations),
        sum(item.onset.true_positive for item in evaluations),
    )
    offset = _count_metrics(
        sum(item.offset.reference_count for item in evaluations),
        sum(item.offset.prediction_count for item in evaluations),
        sum(item.offset.true_positive for item in evaluations),
    )
    total_audio_samples = sum(item.audio_samples for item in outcomes)
    result: Dict[str, object] = {
        "tracks": len(outcomes),
        "audio_samples": total_audio_samples,
        "metrics": {
            "onset": asdict(onset),
            "offset": asdict(offset),
            "onset_latency": asdict(
                latency_metrics(
                    tuple(pair for pairs in onset_pairs for pair in pairs)
                )
            ),
            "offset_latency": asdict(
                latency_metrics(
                    tuple(pair for pairs in offset_pairs for pair in pairs)
                )
            ),
            "onset_multiplicity": asdict(
                _aggregate_multiplicity_comparisons(
                    tuple(item.onset_multiplicity for item in evaluations)
                )
            ),
        },
    }
    if regime == "control":
        result["open_events_at_track_end"] = sum(
            item.control_open_events for item in outcomes
        )
    return result


def _aggregate_outcomes(outcomes: Sequence[TrackOutcome]) -> Dict[str, object]:
    values = tuple(outcomes)
    return {
        "control": _aggregate_regime(values, "control"),
        "treatment": _aggregate_regime(values, "treatment"),
    }


def _family(annotation_member: str) -> str:
    composition = group_stem(annotation_member)
    family = composition.split("-", 1)[0]
    if not family:
        raise EvaluationError("track family must not be empty")
    return family


def aggregate_all_groups(
    outcomes: Sequence[TrackOutcome],
) -> Dict[str, object]:
    """Build global, arrangement, and family-arrangement aggregates."""

    values = tuple(outcomes)
    result: Dict[str, object] = {
        "global": _aggregate_outcomes(values),
        "comp": _aggregate_outcomes(
            tuple(item for item in values if item.arrangement == "comp")
        ),
        "solo": _aggregate_outcomes(
            tuple(item for item in values if item.arrangement == "solo")
        ),
    }
    pairs = sorted({(item.family, item.arrangement) for item in values})
    result["family_arrangement"] = [
        {
            "family": family,
            "arrangement": arrangement,
            **_aggregate_outcomes(
                tuple(
                    item
                    for item in values
                    if item.family == family and item.arrangement == arrangement
                )
            ),
        }
        for family, arrangement in pairs
    ]
    return result


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"source count {name!r} must be an integer >= 0")
    return value


def _canonical_head(metrics: Mapping[str, object], head: str) -> Dict[str, int]:
    value = metrics.get(head)
    if not isinstance(value, Mapping):
        raise EvaluationError(f"source report has no {head} metrics")
    return {
        key: _integer(f"{head}.{key}", value.get(key))
        for key in (
            "reference_count",
            "prediction_count",
            "true_positive",
            "false_positive",
            "false_negative",
        )
    }


def _candidate_at_official_threshold(report: Mapping[str, object]):
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise EvaluationError("source sweep has no candidate list")
    matches = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if report.get("kind") == "boundary_threshold_sweep":
            selected = _same_probability(
                candidate.get("threshold"), OFFICIAL_THRESHOLD
            )
        else:
            selected = (
                _same_probability(
                    candidate.get("entry_threshold"), OFFICIAL_THRESHOLD
                )
                and _same_probability(
                    candidate.get("onset_release_threshold"),
                    OFFICIAL_THRESHOLD,
                )
                and _same_probability(
                    candidate.get("offset_release_threshold"),
                    OFFICIAL_THRESHOLD,
                )
            )
        if selected:
            matches.append(candidate)
    if len(matches) != 1:
        raise EvaluationError(
            "source sweep must contain exactly one official 0.55 control"
        )
    return matches[0]


def _canonical_report_expectations(
    report: Mapping[str, object],
) -> SourceExpectations:
    kind = str(report.get("kind") or "boundary_evaluation")
    if kind in ("boundary_threshold_sweep", "boundary_hysteresis_sweep"):
        selected = _candidate_at_official_threshold(report)
    else:
        selected = report

    aggregates = selected.get("aggregates")
    global_value = (
        aggregates.get("global")
        if isinstance(aggregates, Mapping)
        else selected
    )
    if not isinstance(global_value, Mapping):
        raise EvaluationError("source report has no global aggregate")
    metrics = global_value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvaluationError("source report has no global metrics")
    counts = global_value.get("counts")
    open_events = None
    if isinstance(counts, Mapping) and "predicted_incomplete_events" in counts:
        open_events = _integer(
            "predicted_incomplete_events",
            counts.get("predicted_incomplete_events"),
        )

    tracks: Dict[str, Mapping[str, object]] = {}
    raw_tracks = selected.get("tracks")
    if isinstance(raw_tracks, list):
        for track in raw_tracks:
            if not isinstance(track, Mapping):
                raise EvaluationError("source report contains an invalid track")
            member = track.get("annotation_member")
            track_metrics = track.get("metrics")
            if (
                not isinstance(member, str)
                or member in tracks
                or not isinstance(track_metrics, Mapping)
            ):
                raise EvaluationError("source report track identities are invalid")
            track_open_events = track_metrics.get("predicted_incomplete_events")
            tracks[member] = {
                "onset": _canonical_head(track_metrics, "onset"),
                "offset": _canonical_head(track_metrics, "offset"),
                "open_events": (
                    _integer("track open events", track_open_events)
                    if track_open_events is not None
                    else None
                ),
            }
    return SourceExpectations(
        global_metrics={
            "onset": _canonical_head(metrics, "onset"),
            "offset": _canonical_head(metrics, "offset"),
        },
        global_open_events=open_events,
        tracks=tracks,
        kind=kind,
    )


def _protocol_expectations(report: Mapping[str, object]) -> SourceExpectations:
    baseline = report.get("baseline")
    if not isinstance(baseline, Mapping):
        raise EvaluationError("source protocol has no baseline")
    decoder = baseline.get("associated_decoder")
    if not isinstance(decoder, Mapping):
        raise EvaluationError("source protocol has no associated decoder counts")

    def head(name: str) -> Dict[str, int]:
        reference_count = _integer(
            f"reference_{name}s", baseline.get(f"reference_{name}s")
        )
        prediction_count = _integer(
            f"predicted_{name}s", decoder.get(f"predicted_{name}s")
        )
        true_positive = _integer(
            f"{name}_true_positive", decoder.get(f"{name}_true_positive")
        )
        false_positive = _integer(
            f"{name}_false_positive", decoder.get(f"{name}_false_positive")
        )
        false_negative = _integer(
            f"{name}_false_negative", decoder.get(f"{name}_false_negative")
        )
        return {
            "reference_count": reference_count,
            "prediction_count": prediction_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    open_events = decoder.get("incomplete_events")
    return SourceExpectations(
        global_metrics={"onset": head("onset"), "offset": head("offset")},
        global_open_events=(
            _integer("incomplete_events", open_events)
            if open_events is not None
            else None
        ),
        tracks={},
        kind="unassociated_boundary_candidates_protocol",
    )


def load_source_expectations(path: Path) -> SourceExpectations:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read source report: {path}") from exc
    if not isinstance(report, Mapping):
        raise EvaluationError("source report root must be an object")
    if isinstance(report.get("baseline"), Mapping):
        return _protocol_expectations(report)
    return _canonical_report_expectations(report)


def _metric_integer_counts(value: BoundaryOnlyEvaluation, head: str):
    metrics = getattr(value, head)
    return {
        key: int(getattr(metrics, key))
        for key in (
            "reference_count",
            "prediction_count",
            "true_positive",
            "false_positive",
            "false_negative",
        )
    }


def validate_control_reproduction(
    expectations: SourceExpectations,
    outcomes: Sequence[TrackOutcome],
) -> Dict[str, object]:
    aggregate = _aggregate_regime(tuple(outcomes), "control")
    aggregate_metrics = aggregate["metrics"]
    if not isinstance(aggregate_metrics, Mapping):
        raise AssertionError("control aggregate lacks metrics")
    for head in ("onset", "offset"):
        actual_value = aggregate_metrics.get(head)
        if not isinstance(actual_value, Mapping):
            raise AssertionError(f"control aggregate lacks {head} metrics")
        actual = {
            key: int(actual_value[key]) for key in expectations.global_metrics[head]
        }
        expected = dict(expectations.global_metrics[head])
        if actual != expected:
            raise EvaluationError(
                f"control {head} counts differ from source: "
                f"actual={actual}, source={expected}"
            )
    if (
        expectations.global_open_events is not None
        and aggregate.get("open_events_at_track_end")
        != expectations.global_open_events
    ):
        raise EvaluationError(
            "control open-event count differs from source: "
            f"actual={aggregate.get('open_events_at_track_end')}, "
            f"source={expectations.global_open_events}"
        )

    by_member = {item.annotation_member: item for item in outcomes}
    if expectations.tracks and set(expectations.tracks) != set(by_member):
        raise EvaluationError("source and evaluated track identities differ")
    checked_tracks = 0
    for member, expected in expectations.tracks.items():
        outcome = by_member[member]
        for head in ("onset", "offset"):
            actual = _metric_integer_counts(outcome.control, head)
            if actual != expected[head]:
                raise EvaluationError(
                    f"control {head} counts differ from source for {member!r}: "
                    f"actual={actual}, source={expected[head]}"
                )
        expected_open = expected.get("open_events")
        if expected_open is not None and outcome.control_open_events != expected_open:
            raise EvaluationError(
                f"control open events differ from source for {member!r}"
            )
        checked_tracks += 1
    return {
        "matched": True,
        "source_kind": expectations.kind,
        "global_heads_checked": ["onset", "offset"],
        "per_track_counts_checked": checked_tracks,
        "open_event_count_checked": expectations.global_open_events is not None,
    }


def refuse_output_overwrite(output_path: Path) -> Path:
    resolved = Path(output_path).resolve()
    if resolved.exists():
        raise FileExistsError(
            f"refusing to replace candidate evaluation output: {resolved}"
        )
    return resolved


def write_json_atomically(output_path: Path, value: Mapping[str, object]) -> None:
    """Publish a complete JSON file atomically without replacing a report."""

    output = refuse_output_overwrite(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace candidate evaluation output: {output}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _track_result(outcome: TrackOutcome) -> Dict[str, object]:
    return {
        "annotation_member": outcome.annotation_member,
        "audio_member": outcome.audio_member,
        "family": outcome.family,
        "arrangement": outcome.arrangement,
        "audio_samples": outcome.audio_samples,
        "control": {
            "metrics": asdict(outcome.control),
            "open_events_at_track_end": outcome.control_open_events,
        },
        "treatment": {"metrics": asdict(outcome.treatment)},
    }


def run_candidate_evaluation(arguments: argparse.Namespace) -> Dict[str, object]:
    """Run the locked V7 full-track control/treatment comparison."""

    wall_started_ns = time.perf_counter_ns()
    _locked_thresholds(arguments)
    output_path = refuse_output_overwrite(arguments.output)
    dataset_dir = Path(arguments.dataset_dir).resolve()
    model_path = Path(arguments.model).resolve()
    metadata_path = _metadata_path(model_path, arguments.metadata).resolve()
    selected_players = tuple(dict.fromkeys(arguments.players))
    if selected_players != LOCKED_PLAYERS:
        raise EvaluationError("players are locked to 00 through 04")
    if arguments.seed != DEFAULT_SEED:
        raise EvaluationError(
            f"full-track evaluation is locked to split seed {DEFAULT_SEED}"
        )
    print("[init] arguments validated", file=sys.stderr, flush=True)

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
    validation_members = tuple(
        track.annotation_member for track in validation_tracks
    )
    metadata = _load_and_validate_metadata(
        metadata_path,
        model_path=model_path,
        validation_members=validation_members,
        selected_players=selected_players,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
    )
    print(
        f"[init] split and metadata validated: {len(validation_tracks)} tracks",
        file=sys.stderr,
        flush=True,
    )

    source_path = Path(arguments.source_report).resolve()
    source_expectations = load_source_expectations(source_path)
    if not source_expectations.tracks:
        raise EvaluationError(
            "source report must contain per-track control counts"
        )
    if set(source_expectations.tracks) != set(validation_members):
        raise EvaluationError(
            "source and reconstructed validation track identities differ"
        )
    print("[init] source counts loaded", file=sys.stderr, flush=True)

    print("[init] loading model", file=sys.stderr, flush=True)
    predictor = KerasBoundaryPredictor.from_path(
        str(model_path), receptive_field=_receptive_field(metadata)
    )
    predictor.warm_up(arguments.chunk_size)
    print("[init] model ready", file=sys.stderr, flush=True)
    onset_tolerance_samples = milliseconds_to_samples(
        arguments.onset_tolerance_ms
    )
    offset_tolerance_samples = milliseconds_to_samples(
        arguments.offset_tolerance_ms
    )

    outcomes: List[TrackOutcome] = []
    inference_elapsed_ns = 0
    control_decoding_elapsed_ns = 0
    candidate_decoding_elapsed_ns = 0
    chunk_count = 0
    track_total = len(validation_tracks)
    for track_index, track in enumerate(validation_tracks, start=1):
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} start",
            file=sys.stderr,
            flush=True,
        )
        # Exactly one annotation read and one audio read for this track.
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        references = _flatten_references(slots)
        decoded = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        if any(
            reference.offset_sample > decoded.frame_count
            for reference in references
        ):
            raise EvaluationError(
                "reference boundary exceeds audio length for "
                f"{track.annotation_member!r}"
            )
        arrangement = _track_arrangement(track.annotation_member)
        if arrangement not in ("comp", "solo"):
            raise EvaluationError(
                f"unknown arrangement for {track.annotation_member!r}"
            )

        predictor.reset()
        decoding = decode_candidate_stream(
            predictor,
            _audio_chunks(decoded, arguments.chunk_size),
            onset_threshold=arguments.onset_threshold,
            offset_threshold=arguments.offset_threshold,
            onset_release_threshold=arguments.onset_release_threshold,
            offset_release_threshold=arguments.offset_release_threshold,
        )
        inference_elapsed_ns += decoding.inference_elapsed_ns
        control_decoding_elapsed_ns += decoding.control_decoding_elapsed_ns
        candidate_decoding_elapsed_ns += decoding.candidate_decoding_elapsed_ns
        chunk_count += decoding.chunks

        control_onsets, control_offsets = _boundary_samples(
            decoding.control_events
        )
        candidate_onsets, candidate_offsets = _boundary_samples(
            decoding.candidates
        )
        control, control_onset_pairs, control_offset_pairs = (
            evaluate_boundary_lists(
                references,
                control_onsets,
                control_offsets,
                onset_tolerance_samples=onset_tolerance_samples,
                offset_tolerance_samples=offset_tolerance_samples,
            )
        )
        treatment, treatment_onset_pairs, treatment_offset_pairs = (
            evaluate_boundary_lists(
                references,
                candidate_onsets,
                candidate_offsets,
                onset_tolerance_samples=onset_tolerance_samples,
                offset_tolerance_samples=offset_tolerance_samples,
            )
        )
        outcomes.append(
            TrackOutcome(
                annotation_member=track.annotation_member,
                audio_member=track.audio_member,
                family=_family(track.annotation_member),
                arrangement=arrangement,
                audio_samples=decoded.frame_count,
                control=control,
                treatment=treatment,
                control_open_events=decoding.control_open_events,
                control_onset_pairs=control_onset_pairs,
                control_offset_pairs=control_offset_pairs,
                treatment_onset_pairs=treatment_onset_pairs,
                treatment_offset_pairs=treatment_offset_pairs,
            )
        )
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} complete",
            file=sys.stderr,
            flush=True,
        )

    aggregates = aggregate_all_groups(outcomes)
    source_validation = {
        "provided": True,
        "path": str(source_path),
        **validate_control_reproduction(source_expectations, outcomes),
    }
    print(
        "[audit] control counts reproduce source",
        file=sys.stderr,
        flush=True,
    )

    audio_duration_seconds = (
        sum(item.audio_samples for item in outcomes) / SAMPLE_RATE
    )
    compute_ns = (
        inference_elapsed_ns
        + control_decoding_elapsed_ns
        + candidate_decoding_elapsed_ns
    )
    result: Dict[str, object] = {
        "schema_version": 1,
        "kind": "boundary_candidate_evaluation",
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
            "onset_release_threshold": arguments.onset_release_threshold,
            "offset_release_threshold": arguments.offset_release_threshold,
            "onset_tolerance_ms": arguments.onset_tolerance_ms,
            "onset_tolerance_samples": onset_tolerance_samples,
            "offset_tolerance_ms": arguments.offset_tolerance_ms,
            "offset_tolerance_samples": offset_tolerance_samples,
            "model_passes": 1,
            "shared_immutable_score_chunk": True,
            "association": {
                "control": True,
                "treatment": False,
            },
            "treatment_public_fields": ["type", "position"],
            "interval_metrics": False,
            "scheduler_used": False,
        },
        "source_validation": source_validation,
        "aggregates": aggregates,
        "runtime": {
            "audio_duration_seconds": audio_duration_seconds,
            "chunks": chunk_count,
            "predictor_calls": chunk_count,
            "control_score_chunk_deliveries": chunk_count,
            "treatment_score_chunk_deliveries": chunk_count,
            "model_inference_elapsed_seconds": (
                inference_elapsed_ns / 1_000_000_000.0
            ),
            "control_decoding_elapsed_seconds": (
                control_decoding_elapsed_ns / 1_000_000_000.0
            ),
            "treatment_decoding_elapsed_seconds": (
                candidate_decoding_elapsed_ns / 1_000_000_000.0
            ),
            "compute_realtime_factor": (
                compute_ns / 1_000_000_000.0 / audio_duration_seconds
                if audio_duration_seconds
                else None
            ),
            "wall_elapsed_seconds": (
                time.perf_counter_ns() - wall_started_ns
            )
            / 1_000_000_000.0,
        },
        "tracks": [_track_result(outcome) for outcome in outcomes],
    }
    write_json_atomically(output_path, result)
    result["output_path"] = str(output_path)
    print(f"[done] wrote {output_path}", file=sys.stderr, flush=True)
    return result


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official associated V7 boundaries with unassociated "
            "onset/offset candidates from one immutable score stream."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "GuitarSet",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-report",
        type=Path,
        required=True,
        help="report with official global and per-track control counts",
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
        choices=LOCKED_PLAYERS,
        default=list(LOCKED_PLAYERS),
        help="locked development players; player 05 cannot be selected",
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=512)
    parser.add_argument(
        "--onset-threshold", type=_probability, default=OFFICIAL_THRESHOLD
    )
    parser.add_argument(
        "--offset-threshold", type=_probability, default=OFFICIAL_THRESHOLD
    )
    parser.add_argument(
        "--onset-release-threshold",
        type=_probability,
        default=OFFICIAL_THRESHOLD,
    )
    parser.add_argument(
        "--offset-release-threshold",
        type=_probability,
        default=OFFICIAL_THRESHOLD,
    )
    parser.add_argument(
        "--onset-tolerance-ms", type=_milliseconds, default=50.0
    )
    parser.add_argument(
        "--offset-tolerance-ms", type=_milliseconds, default=50.0
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    result = run_candidate_evaluation(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
