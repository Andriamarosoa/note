"""Compare one-sample and confirmed rearming for V7 boundary candidates.

The model is evaluated once per causal audio chunk.  The identical immutable
score object is delivered to two unassociated boundary decoders: the Exp06
control rearms after one low sample and the treatment requires sixteen
consecutive low samples.  No event association, interval metric, scheduler,
or live activation is part of this diagnostic experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
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

from causal_note.detector import (  # noqa: E402 - local bootstrap above
    BoundaryCandidate,
    BoundaryScoreChunk,
    LiveBoundaryPeakDecoder,
)
from causal_note.guitarset import (  # noqa: E402
    ALLOWED_PLAYERS,
    SAMPLE_RATE,
    index_guitarset,
    load_boundary_slots,
)
from causal_note.keras_predictor import KerasBoundaryPredictor  # noqa: E402
from scripts.evaluate_boundaries import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    EvaluationError,
    _flatten_references,
    _fraction,
    _load_and_validate_metadata,
    _metadata_path,
    _milliseconds,
    _positive_int,
    _probability,
    _receptive_field,
    _track_arrangement,
    milliseconds_to_samples,
)
from scripts.evaluate_boundary_candidates import (  # noqa: E402
    BoundaryOnlyEvaluation,
    OFFICIAL_THRESHOLD,
    _aggregate_regime,
    _audio_chunks,
    _boundary_samples,
    _canonical_head,
    _family,
    _metric_integer_counts,
    _same_probability,
    aggregate_all_groups,
    evaluate_boundary_lists,
    refuse_output_overwrite,
    write_json_atomically,
)
from scripts.train_boundaries import (  # noqa: E402
    decode_pcm16_mono_wav,
    group_stem,
    split_tracks_by_group,
)


LOCKED_PLAYERS = tuple(sorted(ALLOWED_PLAYERS))
CONTROL_REARM_LOW_SAMPLES = 1
TREATMENT_REARM_LOW_SAMPLES = 16
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
DEFAULT_SOURCE_REPORT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.unassociated-boundary-candidates.json"
)
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.unassociated-boundary-rearm-low1-16-protocol.json"
)
EXPECTED_MODEL_SHA256 = (
    "5634ADD0E112A6889B65D5245AD051AD850A1FFFE66FEB8D9E5E74472BA114BF"
)
EXPECTED_SOURCE_SHA256 = (
    "3725ABDB75383AC51C2B800C1232FDAE9A3E5F271F8D3D40743C498AAA07AB81"
)
EXPECTED_PROTOCOL_SHA256 = (
    "794D54EC573B355CC77F96EA0DFD7242E82D1A6571F39BD65EFC4418A0EFD7CF"
)


@dataclass(frozen=True)
class RearmStreamDecoding:
    """Two candidate streams decoded from the same model-score objects."""

    control_candidates: Tuple[BoundaryCandidate, ...]
    treatment_candidates: Tuple[BoundaryCandidate, ...]
    morphology: Mapping[str, Mapping[str, object]]
    chunks: int
    inference_elapsed_ns: int
    control_decoding_elapsed_ns: int
    treatment_decoding_elapsed_ns: int
    morphology_elapsed_ns: int


@dataclass(frozen=True)
class RearmTrackOutcome:
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
    morphology: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class CandidateSourceExpectations:
    global_metrics: Mapping[str, Mapping[str, int]]
    tracks: Mapping[str, Mapping[str, Mapping[str, int]]]
    kind: str


class LowRunMorphology:
    """Count score-low runs bounded by entry-high samples.

    Entry and release are both locked to 0.55 in Exp07.  Consequently every
    bounded run shorter than sixteen samples produces one extra N=1 edge that
    the treatment must suppress.
    """

    def __init__(self, slot_count: int, threshold: float) -> None:
        self._slot_count = slot_count
        self._threshold = threshold
        self._seen_high = {
            "onset": [False] * slot_count,
            "offset": [False] * slot_count,
        }
        self._low_counts = {
            "onset": [0] * slot_count,
            "offset": [0] * slot_count,
        }
        self._histograms = {
            "onset": Counter(),
            "offset": Counter(),
        }
        self._next_sample: Optional[int] = None

    def process_chunk(self, scores: BoundaryScoreChunk) -> None:
        if not isinstance(scores, BoundaryScoreChunk):
            raise EvaluationError("morphology requires a BoundaryScoreChunk")
        if scores.sample_count and scores.slot_count != self._slot_count:
            raise EvaluationError("morphology score slot count differs")
        if self._next_sample is not None and scores.start_sample != self._next_sample:
            raise EvaluationError("morphology scores must be contiguous")
        for onset_row, offset_row in zip(scores.onset, scores.offset):
            self._process_row("onset", onset_row)
            self._process_row("offset", offset_row)
        self._next_sample = scores.start_sample + scores.sample_count

    def _process_row(self, head: str, row: Sequence[float]) -> None:
        seen_high = self._seen_high[head]
        low_counts = self._low_counts[head]
        histogram = self._histograms[head]
        for slot, score in enumerate(row):
            if not math.isfinite(float(score)):
                raise EvaluationError("morphology scores must be finite")
            if score < self._threshold:
                if seen_high[slot]:
                    low_counts[slot] += 1
                continue
            if seen_high[slot] and low_counts[slot]:
                histogram[low_counts[slot]] += 1
            seen_high[slot] = True
            low_counts[slot] = 0

    def summary(self) -> Dict[str, Mapping[str, object]]:
        return {
            head: _morphology_summary(histogram)
            for head, histogram in self._histograms.items()
        }


def _morphology_summary(histogram: Mapping[int, int]):
    run_count = sum(histogram.values())
    sample_count = sum(length * count for length, count in histogram.items())
    maximum = max(histogram, default=0)
    short_count = sum(
        count
        for length, count in histogram.items()
        if length < TREATMENT_REARM_LOW_SAMPLES
    )
    compact_histogram = {
        str(length): int(histogram.get(length, 0))
        for length in range(1, TREATMENT_REARM_LOW_SAMPLES)
    }
    compact_histogram[f"{TREATMENT_REARM_LOW_SAMPLES}_plus"] = sum(
        count
        for length, count in histogram.items()
        if length >= TREATMENT_REARM_LOW_SAMPLES
    )
    return {
        "bounded_low_run_count": run_count,
        "bounded_low_run_samples": sample_count,
        "mean_low_run_samples": sample_count / run_count if run_count else None,
        "mean_low_run_ms": (
            sample_count / run_count / SAMPLE_RATE * 1000.0
            if run_count
            else None
        ),
        "maximum_low_run_samples": maximum,
        "maximum_low_run_ms": maximum / SAMPLE_RATE * 1000.0,
        "shorter_than_treatment_run_count": short_count,
        "at_least_treatment_run_count": run_count - short_count,
        "histogram": compact_histogram,
    }


def _decode_shared_score_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    control_decoder,
    treatment_decoder,
    morphology: LowRunMorphology,
) -> RearmStreamDecoding:
    """Infer once and give the identical score object to both decoders."""

    control: List[BoundaryCandidate] = []
    treatment: List[BoundaryCandidate] = []
    inference_ns = 0
    control_ns = 0
    treatment_ns = 0
    morphology_ns = 0
    chunk_count = 0
    for start_sample, samples in chunks:
        started = time.perf_counter_ns()
        scores = predictor.predict_chunk(samples, start_sample=start_sample)
        inference_ns += time.perf_counter_ns() - started
        if not isinstance(scores, BoundaryScoreChunk):
            raise EvaluationError("predictor must return a BoundaryScoreChunk")
        if scores.start_sample != start_sample:
            raise EvaluationError("predictor returned the wrong start sample")
        if scores.sample_count != len(samples):
            raise EvaluationError("predictor returned the wrong number of samples")
        chunk_count += 1

        started = time.perf_counter_ns()
        control.extend(control_decoder.process_chunk(scores))
        control_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        treatment.extend(treatment_decoder.process_chunk(scores))
        treatment_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        morphology.process_chunk(scores)
        morphology_ns += time.perf_counter_ns() - started

    control_values = tuple(control)
    treatment_values = tuple(treatment)
    control_multiset = Counter(control_values)
    treatment_multiset = Counter(treatment_values)
    if any(
        count > control_multiset[candidate]
        for candidate, count in treatment_multiset.items()
    ):
        raise AssertionError("N=16 candidates must be a multiset subset of N=1")
    morphology_value = morphology.summary()
    for head in ("onset", "offset"):
        control_count = sum(candidate.kind.value == head for candidate in control_values)
        treatment_count = sum(
            candidate.kind.value == head for candidate in treatment_values
        )
        observed = morphology_value[head]["shorter_than_treatment_run_count"]
        if control_count - treatment_count != observed:
            raise AssertionError(
                f"{head} suppression differs from short-low-run count"
            )
    return RearmStreamDecoding(
        control_candidates=control_values,
        treatment_candidates=treatment_values,
        morphology=morphology_value,
        chunks=chunk_count,
        inference_elapsed_ns=inference_ns,
        control_decoding_elapsed_ns=control_ns,
        treatment_decoding_elapsed_ns=treatment_ns,
        morphology_elapsed_ns=morphology_ns,
    )


def decode_rearm_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    *,
    onset_threshold: float = OFFICIAL_THRESHOLD,
    offset_threshold: float = OFFICIAL_THRESHOLD,
    onset_release_threshold: float = OFFICIAL_THRESHOLD,
    offset_release_threshold: float = OFFICIAL_THRESHOLD,
    control_rearm_low_samples: int = CONTROL_REARM_LOW_SAMPLES,
    treatment_rearm_low_samples: int = TREATMENT_REARM_LOW_SAMPLES,
) -> RearmStreamDecoding:
    slot_count = getattr(predictor, "slot_count", None)
    if (
        isinstance(slot_count, bool)
        or not isinstance(slot_count, int)
        or slot_count <= 0
    ):
        raise EvaluationError("predictor must expose a positive slot_count")
    if not callable(getattr(predictor, "predict_chunk", None)):
        raise EvaluationError("predictor must implement predict_chunk")
    if not _same_probability(onset_threshold, offset_threshold):
        raise EvaluationError("Exp07 requires one common entry threshold")
    if not _same_probability(onset_threshold, onset_release_threshold) or not (
        _same_probability(offset_threshold, offset_release_threshold)
    ):
        raise EvaluationError("Exp07 requires release equal to entry")

    common = {
        "slot_count": slot_count,
        "onset_threshold": onset_threshold,
        "offset_threshold": offset_threshold,
        "onset_release_threshold": onset_release_threshold,
        "offset_release_threshold": offset_release_threshold,
    }
    control_decoder = LiveBoundaryPeakDecoder(
        **common, rearm_low_samples=control_rearm_low_samples
    )
    treatment_decoder = LiveBoundaryPeakDecoder(
        **common, rearm_low_samples=treatment_rearm_low_samples
    )
    morphology = LowRunMorphology(slot_count, onset_threshold)
    return _decode_shared_score_stream(
        predictor,
        chunks,
        control_decoder,
        treatment_decoder,
        morphology,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _locked_configuration(arguments: argparse.Namespace) -> None:
    probability_values = {
        "onset threshold": arguments.onset_threshold,
        "offset threshold": arguments.offset_threshold,
        "onset release threshold": arguments.onset_release_threshold,
        "offset release threshold": arguments.offset_release_threshold,
    }
    for name, value in probability_values.items():
        if not _same_probability(value, OFFICIAL_THRESHOLD):
            raise EvaluationError(f"{name} is locked to {OFFICIAL_THRESHOLD}")
    if arguments.control_rearm_low_samples != CONTROL_REARM_LOW_SAMPLES:
        raise EvaluationError("control rearm is locked to one low sample")
    if arguments.treatment_rearm_low_samples != TREATMENT_REARM_LOW_SAMPLES:
        raise EvaluationError("treatment rearm is locked to sixteen low samples")
    if arguments.chunk_size != 512:
        raise EvaluationError("chunk size is locked to 512 samples")


def load_candidate_source(path: Path) -> CandidateSourceExpectations:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read source candidate report: {path}") from exc
    if not isinstance(report, Mapping):
        raise EvaluationError("source candidate report root must be an object")
    if report.get("kind") != "boundary_candidate_evaluation":
        raise EvaluationError("source must be the completed Exp06 candidate report")
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise EvaluationError("source candidate report lacks configuration")
    for name in (
        "onset_threshold",
        "offset_threshold",
        "onset_release_threshold",
        "offset_release_threshold",
    ):
        if not _same_probability(configuration.get(name), OFFICIAL_THRESHOLD):
            raise EvaluationError(f"source {name} is not locked to 0.55")
    association = configuration.get("association")
    if not isinstance(association, Mapping) or association.get("treatment") is not False:
        raise EvaluationError("source treatment must be unassociated")
    if configuration.get("interval_metrics") is not False:
        raise EvaluationError("source must not contain interval metrics")

    aggregates = report.get("aggregates")
    global_value = aggregates.get("global") if isinstance(aggregates, Mapping) else None
    treatment = (
        global_value.get("treatment") if isinstance(global_value, Mapping) else None
    )
    global_metrics = treatment.get("metrics") if isinstance(treatment, Mapping) else None
    if not isinstance(global_metrics, Mapping):
        raise EvaluationError("source has no global treatment metrics")

    tracks: Dict[str, Mapping[str, Mapping[str, int]]] = {}
    raw_tracks = report.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise EvaluationError("source must contain per-track candidate metrics")
    for track in raw_tracks:
        if not isinstance(track, Mapping):
            raise EvaluationError("source contains an invalid track")
        member = track.get("annotation_member")
        treatment = track.get("treatment")
        metrics = treatment.get("metrics") if isinstance(treatment, Mapping) else None
        if (
            not isinstance(member, str)
            or member in tracks
            or not isinstance(metrics, Mapping)
        ):
            raise EvaluationError("source track candidate metrics are invalid")
        tracks[member] = {
            "onset": _canonical_head(metrics, "onset"),
            "offset": _canonical_head(metrics, "offset"),
        }
    return CandidateSourceExpectations(
        global_metrics={
            "onset": _canonical_head(global_metrics, "onset"),
            "offset": _canonical_head(global_metrics, "offset"),
        },
        tracks=tracks,
        kind=str(report.get("kind")),
    )


def validate_control_reproduction(
    expectations: CandidateSourceExpectations,
    outcomes: Sequence[RearmTrackOutcome],
) -> Dict[str, object]:
    aggregate = _aggregate_regime(tuple(outcomes), "control")
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise AssertionError("control aggregate lacks metrics")
    for head in ("onset", "offset"):
        actual = {
            key: int(metrics[head][key])
            for key in expectations.global_metrics[head]
        }
        expected = dict(expectations.global_metrics[head])
        if actual != expected:
            raise EvaluationError(
                f"control {head} differs from Exp06: actual={actual}, source={expected}"
            )
    by_member = {outcome.annotation_member: outcome for outcome in outcomes}
    if set(by_member) != set(expectations.tracks):
        raise EvaluationError("source and evaluated track identities differ")
    for member, expected in expectations.tracks.items():
        for head in ("onset", "offset"):
            actual = _metric_integer_counts(by_member[member].control, head)
            if actual != expected[head]:
                raise EvaluationError(
                    f"control {head} differs from Exp06 for {member!r}"
                )
    return {
        "matched": True,
        "source_kind": expectations.kind,
        "global_heads_checked": ["onset", "offset"],
        "per_track_counts_checked": len(expectations.tracks),
        "open_event_count_checked": False,
    }


def _strip_open_event_fields(aggregates: Mapping[str, object]) -> None:
    groups = [aggregates.get(name) for name in ("global", "comp", "solo")]
    family_groups = aggregates.get("family_arrangement")
    if isinstance(family_groups, list):
        groups.extend(family_groups)
    for group in groups:
        if not isinstance(group, dict):
            continue
        control = group.get("control")
        if isinstance(control, dict):
            control.pop("open_events_at_track_end", None)
        if isinstance(control, Mapping) and isinstance(group.get("treatment"), Mapping):
            treatment = group["treatment"]
            comparison = {}
            for head in ("onset", "offset"):
                control_count = control["metrics"][head]["prediction_count"]
                treatment_count = treatment["metrics"][head]["prediction_count"]
                comparison[head] = {
                    "control_prediction_count": control_count,
                    "treatment_prediction_count": treatment_count,
                    "suppressed_candidate_count": control_count - treatment_count,
                    "suppressed_candidate_percent": (
                        (control_count - treatment_count) / control_count * 100.0
                        if control_count
                        else 0.0
                    ),
                }
            group["comparison"] = comparison


def _merge_morphology(
    values: Sequence[Mapping[str, Mapping[str, object]]]
) -> Dict[str, Mapping[str, object]]:
    result = {}
    for head in ("onset", "offset"):
        histogram: Counter[int] = Counter()
        for value in values:
            compact = value[head]["histogram"]
            for key, count in compact.items():
                if key.endswith("_plus"):
                    # Only totals and short-run counts require exact lengths.
                    continue
                histogram[int(key)] += int(count)
        short_count = sum(histogram.values())
        total_runs = sum(int(value[head]["bounded_low_run_count"]) for value in values)
        total_samples = sum(
            int(value[head]["bounded_low_run_samples"]) for value in values
        )
        maximum = max(
            (int(value[head]["maximum_low_run_samples"]) for value in values),
            default=0,
        )
        compact_histogram = {
            str(length): histogram[length]
            for length in range(1, TREATMENT_REARM_LOW_SAMPLES)
        }
        compact_histogram[f"{TREATMENT_REARM_LOW_SAMPLES}_plus"] = (
            total_runs - short_count
        )
        result[head] = {
            "bounded_low_run_count": total_runs,
            "bounded_low_run_samples": total_samples,
            "mean_low_run_samples": total_samples / total_runs if total_runs else None,
            "mean_low_run_ms": (
                total_samples / total_runs / SAMPLE_RATE * 1000.0
                if total_runs
                else None
            ),
            "maximum_low_run_samples": maximum,
            "maximum_low_run_ms": maximum / SAMPLE_RATE * 1000.0,
            "shorter_than_treatment_run_count": short_count,
            "at_least_treatment_run_count": total_runs - short_count,
            "histogram": compact_histogram,
        }
    return result


def _track_result(outcome: RearmTrackOutcome) -> Dict[str, object]:
    comparison = {}
    for head in ("onset", "offset"):
        control_count = getattr(outcome.control, head).prediction_count
        treatment_count = getattr(outcome.treatment, head).prediction_count
        comparison[head] = {
            "control_prediction_count": control_count,
            "treatment_prediction_count": treatment_count,
            "suppressed_candidate_count": control_count - treatment_count,
        }
    return {
        "annotation_member": outcome.annotation_member,
        "audio_member": outcome.audio_member,
        "family": outcome.family,
        "arrangement": outcome.arrangement,
        "audio_samples": outcome.audio_samples,
        "control": {"metrics": asdict(outcome.control)},
        "treatment": {"metrics": asdict(outcome.treatment)},
        "comparison": comparison,
        "low_run_morphology": outcome.morphology,
    }


def run_rearm_evaluation(arguments: argparse.Namespace) -> Dict[str, object]:
    wall_started_ns = time.perf_counter_ns()
    _locked_configuration(arguments)
    output_path = refuse_output_overwrite(arguments.output)
    dataset_dir = Path(arguments.dataset_dir).resolve()
    model_path = Path(arguments.model).resolve()
    metadata_path = _metadata_path(model_path, arguments.metadata).resolve()
    source_path = Path(arguments.source_report).resolve()
    protocol_path = Path(arguments.protocol).resolve()
    selected_players = tuple(dict.fromkeys(arguments.players))
    if selected_players != LOCKED_PLAYERS:
        raise EvaluationError("players are locked to 00 through 04")
    if arguments.seed != DEFAULT_SEED:
        raise EvaluationError(f"seed is locked to {DEFAULT_SEED}")
    if not math.isclose(
        arguments.validation_fraction,
        DEFAULT_VALIDATION_FRACTION,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise EvaluationError(
            f"validation fraction is locked to {DEFAULT_VALIDATION_FRACTION}"
        )
    if _sha256(model_path) != EXPECTED_MODEL_SHA256:
        raise EvaluationError("model SHA-256 differs from locked V7-e8")
    if _sha256(source_path) != EXPECTED_SOURCE_SHA256:
        raise EvaluationError("source SHA-256 differs from completed Exp06")
    if _sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise EvaluationError("protocol SHA-256 differs from preregistered Exp07")
    print("[init] arguments and hashes validated", file=sys.stderr, flush=True)

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
    expectations = load_candidate_source(source_path)
    if set(expectations.tracks) != set(validation_members):
        raise EvaluationError("source and reconstructed validation tracks differ")
    print(
        f"[init] split and source validated: {len(validation_tracks)} tracks",
        file=sys.stderr,
        flush=True,
    )

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

    outcomes: List[RearmTrackOutcome] = []
    inference_ns = control_ns = treatment_ns = morphology_ns = 0
    chunk_count = 0
    track_total = len(validation_tracks)
    for track_index, track in enumerate(validation_tracks, start=1):
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} start",
            file=sys.stderr,
            flush=True,
        )
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        references = _flatten_references(slots)
        decoded = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        if any(reference.offset_sample > decoded.frame_count for reference in references):
            raise EvaluationError(
                f"reference boundary exceeds audio for {track.annotation_member!r}"
            )
        arrangement = _track_arrangement(track.annotation_member)
        if arrangement not in ("comp", "solo"):
            raise EvaluationError(f"unknown arrangement for {track.annotation_member!r}")

        predictor.reset()
        decoding = decode_rearm_stream(
            predictor,
            _audio_chunks(decoded, arguments.chunk_size),
            onset_threshold=arguments.onset_threshold,
            offset_threshold=arguments.offset_threshold,
            onset_release_threshold=arguments.onset_release_threshold,
            offset_release_threshold=arguments.offset_release_threshold,
            control_rearm_low_samples=arguments.control_rearm_low_samples,
            treatment_rearm_low_samples=arguments.treatment_rearm_low_samples,
        )
        inference_ns += decoding.inference_elapsed_ns
        control_ns += decoding.control_decoding_elapsed_ns
        treatment_ns += decoding.treatment_decoding_elapsed_ns
        morphology_ns += decoding.morphology_elapsed_ns
        chunk_count += decoding.chunks

        control_onsets, control_offsets = _boundary_samples(
            decoding.control_candidates
        )
        treatment_onsets, treatment_offsets = _boundary_samples(
            decoding.treatment_candidates
        )
        control, control_onset_pairs, control_offset_pairs = evaluate_boundary_lists(
            references,
            control_onsets,
            control_offsets,
            onset_tolerance_samples=onset_tolerance_samples,
            offset_tolerance_samples=offset_tolerance_samples,
        )
        treatment, treatment_onset_pairs, treatment_offset_pairs = (
            evaluate_boundary_lists(
                references,
                treatment_onsets,
                treatment_offsets,
                onset_tolerance_samples=onset_tolerance_samples,
                offset_tolerance_samples=offset_tolerance_samples,
            )
        )
        outcomes.append(
            RearmTrackOutcome(
                annotation_member=track.annotation_member,
                audio_member=track.audio_member,
                family=_family(track.annotation_member),
                arrangement=arrangement,
                audio_samples=decoded.frame_count,
                control=control,
                treatment=treatment,
                control_open_events=0,
                control_onset_pairs=control_onset_pairs,
                control_offset_pairs=control_offset_pairs,
                treatment_onset_pairs=treatment_onset_pairs,
                treatment_offset_pairs=treatment_offset_pairs,
                morphology=decoding.morphology,
            )
        )
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} complete",
            file=sys.stderr,
            flush=True,
        )

    aggregates = aggregate_all_groups(outcomes)
    _strip_open_event_fields(aggregates)
    source_validation = {
        "provided": True,
        "path": str(source_path),
        "sha256": EXPECTED_SOURCE_SHA256,
        **validate_control_reproduction(expectations, outcomes),
    }
    print("[audit] N=1 reproduces Exp06", file=sys.stderr, flush=True)

    global_morphology = _merge_morphology(
        tuple(outcome.morphology for outcome in outcomes)
    )
    for head in ("onset", "offset"):
        suppressed = aggregates["global"]["comparison"][head][
            "suppressed_candidate_count"
        ]
        short_runs = global_morphology[head][
            "shorter_than_treatment_run_count"
        ]
        if suppressed != short_runs:
            raise AssertionError("global suppression and low-run audit differ")

    audio_seconds = sum(outcome.audio_samples for outcome in outcomes) / SAMPLE_RATE
    compute_ns = inference_ns + control_ns + treatment_ns + morphology_ns
    result: Dict[str, object] = {
        "schema_version": 1,
        "kind": "boundary_candidate_rearm_evaluation",
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "metadata_path": str(metadata_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
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
            "control_rearm_low_samples": arguments.control_rearm_low_samples,
            "treatment_rearm_low_samples": arguments.treatment_rearm_low_samples,
            "treatment_rearm_duration_ms": (
                arguments.treatment_rearm_low_samples / SAMPLE_RATE * 1000.0
            ),
            "onset_tolerance_ms": arguments.onset_tolerance_ms,
            "onset_tolerance_samples": onset_tolerance_samples,
            "offset_tolerance_ms": arguments.offset_tolerance_ms,
            "offset_tolerance_samples": offset_tolerance_samples,
            "model_passes": 1,
            "shared_immutable_score_chunk": True,
            "association": {"control": False, "treatment": False},
            "public_fields": ["type", "position"],
            "interval_metrics": False,
            "scheduler_used": False,
            "official_live_activation": False,
        },
        "source_validation": source_validation,
        "aggregates": aggregates,
        "low_run_morphology": global_morphology,
        "runtime": {
            "audio_duration_seconds": audio_seconds,
            "chunks": chunk_count,
            "predictor_calls": chunk_count,
            "control_score_chunk_deliveries": chunk_count,
            "treatment_score_chunk_deliveries": chunk_count,
            "morphology_score_chunk_deliveries": chunk_count,
            "model_inference_elapsed_seconds": inference_ns / 1_000_000_000.0,
            "control_decoding_elapsed_seconds": control_ns / 1_000_000_000.0,
            "treatment_decoding_elapsed_seconds": treatment_ns / 1_000_000_000.0,
            "morphology_elapsed_seconds": morphology_ns / 1_000_000_000.0,
            "compute_realtime_factor": (
                compute_ns / 1_000_000_000.0 / audio_seconds
                if audio_seconds
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
            "Compare N=1 and N=16 event-independent rearming on immutable "
            "V7 boundary scores."
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
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--source-report",
        type=Path,
        required=True,
        help="completed Exp06 unassociated-candidate report",
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
        "--control-rearm-low-samples",
        type=_positive_int,
        default=CONTROL_REARM_LOW_SAMPLES,
    )
    parser.add_argument(
        "--treatment-rearm-low-samples",
        type=_positive_int,
        default=TREATMENT_REARM_LOW_SAMPLES,
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
    result = run_rearm_evaluation(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
