"""Evaluate a fixed causal refractory period on accepted N=16 candidates.

The V7-e8 model is inferred once for every causal audio chunk.  The identical
immutable score object is delivered to an N=16 control decoder, an N=16 decoder
with a fixed 2205-sample first-candidate refractory period, and the locked Exp08
identity observer.  Model channels remain audit-only; public candidates remain
``type, position`` and no onset/offset association is attempted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.detector import (  # noqa: E402
    BoundaryCandidate,
    BoundaryType,
    LiveBoundaryPeakDecoder,
)
from causal_note.guitarset import (  # noqa: E402
    ALLOWED_PLAYERS,
    SAMPLE_RATE,
    index_guitarset,
    load_boundary_slots,
)
from causal_note.keras_predictor import KerasBoundaryPredictor  # noqa: E402
from scripts.audit_boundary_candidate_relations import (  # noqa: E402
    CandidateRelationObserver,
    IdentifiedReference,
    TracedCandidate,
    _candidate_neighborhoods,
    _family,
    _histogram_summary,
    _integer_evaluation,
    _multiplicity_histogram,
    _percentile,
    identified_references,
    note_support_counts,
    official_index_matching,
    summarize_note_support,
)
from scripts.evaluate_boundaries import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    EvaluationError,
    _fraction,
    _load_and_validate_metadata,
    _metadata_path,
    _milliseconds,
    _positive_int,
    _probability,
    _receptive_field,
    _track_arrangement,
    latency_metrics,
    milliseconds_to_samples,
)
from scripts.evaluate_boundary_candidates import (  # noqa: E402
    OFFICIAL_THRESHOLD,
    _audio_chunks,
    _boundary_samples,
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
REARM_LOW_SAMPLES = 16
CONTROL_REFRACTORY_SAMPLES = 0
TREATMENT_REFRACTORY_SAMPLES = 2205
DISTANCE_BINS = (
    ("1-44", 1, 44),
    ("45-220", 45, 220),
    ("221-441", 221, 441),
    ("442-882", 442, 882),
    ("883-2205", 883, 2205),
)
STATUS_LABELS = (
    "matched",
    "isolated",
    "same_channel_same_reference_successor_excess",
    "single_reference_near_excess",
    "ambiguous_near",
)

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
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-relations.json"
)
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-refractory-first50ms-protocol.json"
)
DEFAULT_PROTOCOL_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-refractory-first50ms-protocol-amendment-01.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-refractory-first50ms.json"
)
DEFAULT_DETECTOR_SOURCE = SOURCE_ROOT / "causal_note" / "detector.py"
DEFAULT_RELATION_EVALUATOR = REPOSITORY_ROOT / "scripts" / "audit_boundary_candidate_relations.py"

EXPECTED_MODEL_SHA256 = "5634ADD0E112A6889B65D5245AD051AD850A1FFFE66FEB8D9E5E74472BA114BF"
EXPECTED_METADATA_SHA256 = "8AB99DC6DD191B39043A303D41BB405A0C84C2CE607C608E330E4DEF33F52A25"
EXPECTED_SOURCE_SHA256 = "C117BA2C15B0FF9B5A907831C38CEDB71B4BAB67E7158FBA8DE88F743BBBE25B"
EXPECTED_RELATION_EVALUATOR_SHA256 = "2C0E732CD3DB4534C00840D79A55C3970DBC025F1DC588DC7EDB390DC2758402"
EXPECTED_PROTOCOL_SHA256 = "A8C5B1BF5D9D05BFE59C7F2800B5BC756771485045F043CFB0B0B210781A5576"
EXPECTED_PROTOCOL_AMENDMENT_SHA256 = "9881D3113054F31795CA14EE6B7859194CDDBD20DC105A3476ADDE8C9821FF02"
EXPECTED_DETECTOR_SHA256 = "BE4F15D4999D802640074D91F1A772EAC764F91F391F6E35EAA6CF2FAB5460FB"


@dataclass(frozen=True)
class Suppression:
    candidate: TracedCandidate
    anchor: TracedCandidate
    distance_samples: int


@dataclass(frozen=True)
class ConsolidationTrace:
    kept: Tuple[TracedCandidate, ...]
    suppressed: Tuple[Suppression, ...]


@dataclass(frozen=True)
class StreamDecoding:
    control_candidates: Tuple[BoundaryCandidate, ...]
    treatment_candidates: Tuple[BoundaryCandidate, ...]
    traced_control: Tuple[TracedCandidate, ...]
    traced_treatment: Tuple[TracedCandidate, ...]
    suppressions: Tuple[Suppression, ...]
    chunks: int
    inference_elapsed_ns: int
    control_elapsed_ns: int
    treatment_elapsed_ns: int
    observer_elapsed_ns: int
    trace_elapsed_ns: int


@dataclass
class HeadRaw:
    control_metrics: Counter = field(default_factory=Counter)
    treatment_metrics: Counter = field(default_factory=Counter)
    control_delays: List[int] = field(default_factory=list)
    treatment_delays: List[int] = field(default_factory=list)
    control_relations: Counter = field(default_factory=Counter)
    treatment_relations: Counter = field(default_factory=Counter)
    suppressed_status: Counter = field(default_factory=Counter)
    anchor_status: Counter = field(default_factory=Counter)
    suppression_cross: Counter = field(default_factory=Counter)
    suppression_distance_bins: Counter = field(default_factory=Counter)
    control_multiplicity: Counter = field(default_factory=Counter)
    treatment_multiplicity: Counter = field(default_factory=Counter)
    reference_multiplicity: Counter = field(default_factory=Counter)
    simultaneous_reference_instances: int = 0
    control_simultaneous_support: int = 0
    treatment_simultaneous_support: int = 0
    suppressed_count: int = 0

    def merge(self, other: "HeadRaw") -> None:
        for name in (
            "control_metrics",
            "treatment_metrics",
            "control_relations",
            "treatment_relations",
            "suppressed_status",
            "anchor_status",
            "suppression_cross",
            "suppression_distance_bins",
            "control_multiplicity",
            "treatment_multiplicity",
            "reference_multiplicity",
        ):
            getattr(self, name).update(getattr(other, name))
        self.control_delays.extend(other.control_delays)
        self.treatment_delays.extend(other.treatment_delays)
        self.simultaneous_reference_instances += other.simultaneous_reference_instances
        self.control_simultaneous_support += other.control_simultaneous_support
        self.treatment_simultaneous_support += other.treatment_simultaneous_support
        self.suppressed_count += other.suppressed_count


@dataclass
class TrackOutcome:
    annotation_member: str
    audio_member: str
    family: str
    arrangement: str
    audio_samples: int
    heads: Mapping[str, HeadRaw]
    control_note_support: Counter
    treatment_note_support: Counter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _public_projection(values: Iterable[TracedCandidate]) -> Tuple[BoundaryCandidate, ...]:
    return tuple(BoundaryCandidate(item.kind, item.sample) for item in values)


def trace_fixed_refractory(
    candidates: Iterable[TracedCandidate],
    refractory_samples: int = TREATMENT_REFRACTORY_SAMPLES,
) -> ConsolidationTrace:
    """Apply the causal non-extending first-candidate rule to traced values."""

    if (
        isinstance(refractory_samples, bool)
        or not isinstance(refractory_samples, int)
        or refractory_samples < 0
    ):
        raise EvaluationError("refractory_samples must be an integer >= 0")
    kept: List[TracedCandidate] = []
    suppressed: List[Suppression] = []
    last_kept: Dict[Tuple[BoundaryType, int], TracedCandidate] = {}
    previous: Dict[Tuple[BoundaryType, int], TracedCandidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, TracedCandidate):
            raise EvaluationError("trace requires TracedCandidate values")
        key = candidate.kind, candidate.channel
        prior = previous.get(key)
        if prior is not None and candidate.sample <= prior.sample:
            raise AssertionError("same-key N=16 candidates must have a positive gap")
        previous[key] = candidate
        anchor = last_kept.get(key)
        if anchor is None or refractory_samples == 0:
            kept.append(candidate)
            last_kept[key] = candidate
            continue
        distance = candidate.sample - anchor.sample
        if distance <= 0:
            raise AssertionError("refractory distance must be positive")
        if distance <= refractory_samples:
            suppressed.append(Suppression(candidate, anchor, distance))
        else:
            kept.append(candidate)
            last_kept[key] = candidate
    return ConsolidationTrace(tuple(kept), tuple(suppressed))


def _distance_bin(distance: int) -> str:
    for label, lower, upper in DISTANCE_BINS:
        if lower <= distance <= upper:
            return label
    raise AssertionError(f"suppression distance {distance} is outside locked bins")


def decode_refractory_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    *,
    threshold: float = OFFICIAL_THRESHOLD,
    rearm_low_samples: int = REARM_LOW_SAMPLES,
    treatment_refractory_samples: int = TREATMENT_REFRACTORY_SAMPLES,
) -> StreamDecoding:
    """Infer once and decode control/treatment causally from identical scores."""

    slot_count = getattr(predictor, "slot_count", None)
    if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count <= 0:
        raise EvaluationError("predictor must expose a positive slot_count")
    common = {
        "slot_count": slot_count,
        "onset_threshold": threshold,
        "offset_threshold": threshold,
        "onset_release_threshold": threshold,
        "offset_release_threshold": threshold,
        "rearm_low_samples": rearm_low_samples,
    }
    control_decoder = LiveBoundaryPeakDecoder(
        **common, consolidation_samples=CONTROL_REFRACTORY_SAMPLES
    )
    treatment_decoder = LiveBoundaryPeakDecoder(
        **common, consolidation_samples=treatment_refractory_samples
    )
    observer = CandidateRelationObserver(
        slot_count=slot_count,
        threshold=threshold,
        n16_low_samples=rearm_low_samples,
    )
    control: List[BoundaryCandidate] = []
    treatment: List[BoundaryCandidate] = []
    inference_ns = control_ns = treatment_ns = observer_ns = 0
    chunk_count = 0
    for start_sample, samples in chunks:
        started = time.perf_counter_ns()
        scores = predictor.predict_chunk(samples, start_sample=start_sample)
        inference_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        control.extend(control_decoder.process_chunk(scores))
        control_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        treatment.extend(treatment_decoder.process_chunk(scores))
        treatment_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        observer.process_chunk(scores)
        observer_ns += time.perf_counter_ns() - started
        chunk_count += 1

    traced_control = observer.n16_candidates
    trace_started = time.perf_counter_ns()
    trace = trace_fixed_refractory(
        traced_control, refractory_samples=treatment_refractory_samples
    )
    trace_ns = time.perf_counter_ns() - trace_started
    control_values = tuple(control)
    treatment_values = tuple(treatment)
    if _public_projection(traced_control) != control_values:
        raise AssertionError("identity observer does not reproduce N=16 control")
    if _public_projection(trace.kept) != treatment_values:
        raise AssertionError("traced refractory projection differs from treatment")
    if len(traced_control) != len(trace.kept) + len(trace.suppressed):
        raise AssertionError("control does not partition into kept and suppressed")
    control_by_key = Counter((item.kind, item.channel) for item in traced_control)
    kept_by_key = Counter((item.kind, item.channel) for item in trace.kept)
    suppressed_by_key = Counter(
        (item.candidate.kind, item.candidate.channel) for item in trace.suppressed
    )
    if control_by_key != kept_by_key + suppressed_by_key:
        raise AssertionError("per-key control count does not equal kept plus suppressed")
    kept_ids = {item.episode_id for item in trace.kept}
    for suppression in trace.suppressed:
        if (
            suppression.candidate.kind is not suppression.anchor.kind
            or suppression.candidate.channel != suppression.anchor.channel
            or not 1
            <= suppression.distance_samples
            <= treatment_refractory_samples
            or suppression.anchor.episode_id not in kept_ids
        ):
            raise AssertionError("suppression violates the locked anchor rule")
    last_retained: Dict[Tuple[BoundaryType, int], int] = {}
    for candidate in trace.kept:
        key = candidate.kind, candidate.channel
        previous_sample = last_retained.get(key)
        if (
            previous_sample is not None
            and candidate.sample - previous_sample <= treatment_refractory_samples
        ):
            raise AssertionError("retained same-key candidates violate refractory gap")
        last_retained[key] = candidate.sample
    return StreamDecoding(
        control_candidates=control_values,
        treatment_candidates=treatment_values,
        traced_control=traced_control,
        traced_treatment=trace.kept,
        suppressions=trace.suppressed,
        chunks=chunk_count,
        inference_elapsed_ns=inference_ns,
        control_elapsed_ns=control_ns,
        treatment_elapsed_ns=treatment_ns,
        observer_elapsed_ns=observer_ns,
        trace_elapsed_ns=trace_ns,
    )


def classify_candidate_statuses(
    references: Sequence[IdentifiedReference],
    candidates: Sequence[TracedCandidate],
    *,
    tolerance_samples: int,
) -> Tuple[Mapping[int, str], Counter, Tuple[Tuple[int, int], ...]]:
    """Return the amended Exp08 status of every candidate occurrence."""

    references = tuple(references)
    values = tuple(sorted(candidates, key=lambda item: (item.sample, item.channel)))
    if len({item.episode_id for item in values}) != len(values):
        raise AssertionError("candidate episode identities must be unique")
    neighborhoods = _candidate_neighborhoods(references, values, tolerance_samples)
    pairs = official_index_matching(references, values, tolerance_samples)
    matched_predictions = {prediction_index for _, prediction_index in pairs}
    strong_repeat_fp_indices = set()
    by_channel: DefaultDict[int, List[int]] = defaultdict(list)
    for index, candidate in enumerate(values):
        by_channel[candidate.channel].append(index)
    for indices in by_channel.values():
        for previous_index, current_index in zip(indices, indices[1:]):
            left = neighborhoods[previous_index]
            right = neighborhoods[current_index]
            if (
                len(left) == 1
                and len(right) == 1
                and left[0] == right[0]
            ):
                if previous_index not in matched_predictions:
                    strong_repeat_fp_indices.add(previous_index)
                if current_index not in matched_predictions:
                    strong_repeat_fp_indices.add(current_index)

    statuses: Dict[int, str] = {}
    partition = Counter()
    for index, (candidate, eligible) in enumerate(zip(values, neighborhoods)):
        if index in matched_predictions:
            label = "matched"
        elif not eligible:
            label = "isolated"
        elif len(eligible) >= 2:
            label = "ambiguous_near"
        elif index in strong_repeat_fp_indices:
            label = "same_channel_same_reference_successor_excess"
        else:
            label = "single_reference_near_excess"
        statuses[candidate.episode_id] = label
        if label != "matched":
            partition[label] += 1
    if sum(partition.values()) != len(values) - len(pairs):
        raise AssertionError("false-positive relation partition is not exhaustive")
    return statuses, partition, pairs


def _simultaneous_reference_support(
    references: Sequence[IdentifiedReference],
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    multiplicity = Counter(reference.sample for reference in references)
    simultaneous_indices = {
        index
        for index, reference in enumerate(references)
        if multiplicity[reference.sample] > 1
    }
    matched_indices = {reference_index for reference_index, _ in pairs}
    return len(simultaneous_indices), len(simultaneous_indices & matched_indices)


def _metrics_counter(values: Mapping[str, int]) -> Counter:
    return Counter(
        {
            name: int(values[name])
            for name in (
                "reference_count",
                "prediction_count",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        }
    )


def _complete_metrics(values: Mapping[str, int]) -> Dict[str, object]:
    reference_count = int(values["reference_count"])
    prediction_count = int(values["prediction_count"])
    true_positive = int(values["true_positive"])
    false_positive = int(values["false_positive"])
    false_negative = int(values["false_negative"])
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / reference_count if reference_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "reference_count": reference_count,
        "prediction_count": prediction_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _samples_to_ms(value: Optional[float]) -> Optional[float]:
    return None if value is None else value / SAMPLE_RATE * 1000.0


def summarize_delays(delays: Sequence[int]) -> Dict[str, object]:
    values = tuple(int(value) for value in delays)
    pairs = tuple((0, value) for value in values)
    signed_and_causal = asdict(latency_metrics(pairs))
    absolute = tuple(abs(value) for value in values)
    absolute_p50 = _percentile(absolute, 0.5)
    absolute_p90 = _percentile(absolute, 0.9)
    signed_and_causal.update(
        {
            "absolute_p50_samples": absolute_p50,
            "absolute_p90_samples": absolute_p90,
            "absolute_max_samples": max(absolute) if absolute else None,
            "absolute_p50_ms": _samples_to_ms(absolute_p50),
            "absolute_p90_ms": _samples_to_ms(absolute_p90),
            "absolute_max_ms": _samples_to_ms(max(absolute) if absolute else None),
        }
    )
    return signed_and_causal


def load_relation_source(path: Path) -> Mapping[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read Exp08 source report: {path}") from exc
    if not isinstance(report, Mapping) or report.get("kind") != "boundary_candidate_relation_audit":
        raise EvaluationError("source must be the completed Exp08 relation audit")
    split = report.get("split")
    if not isinstance(split, Mapping) or split.get("player_05_read") is not False:
        raise EvaluationError("Exp08 source does not preserve the player05 lock")
    tracks = report.get("tracks")
    if not isinstance(tracks, list) or len(tracks) != 60:
        raise EvaluationError("Exp08 source must contain exactly 60 tracks")
    return report


def _source_tracks(report: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    result: Dict[str, Mapping[str, object]] = {}
    for raw in report["tracks"]:
        if not isinstance(raw, Mapping):
            raise EvaluationError("Exp08 source contains an invalid track")
        member = raw.get("annotation_member")
        if not isinstance(member, str) or not member or member in result:
            raise EvaluationError("Exp08 track identity is invalid or duplicated")
        result[member] = raw
    return result


def _track_head_raw(
    references: Sequence[IdentifiedReference],
    control: Sequence[TracedCandidate],
    treatment: Sequence[TracedCandidate],
    suppressions: Sequence[Suppression],
    control_metrics: Mapping[str, int],
    treatment_metrics: Mapping[str, int],
    control_pairs: Sequence[Tuple[int, int]],
    treatment_pairs: Sequence[Tuple[int, int]],
    *,
    tolerance_samples: int,
) -> HeadRaw:
    control_status, control_relations, control_identity_pairs = (
        classify_candidate_statuses(
            references, control, tolerance_samples=tolerance_samples
        )
    )
    _, treatment_relations, treatment_identity_pairs = classify_candidate_statuses(
        references, treatment, tolerance_samples=tolerance_samples
    )
    if len(control_identity_pairs) != int(control_metrics["true_positive"]):
        raise AssertionError("identity and official control matching counts differ")
    if len(treatment_identity_pairs) != int(treatment_metrics["true_positive"]):
        raise AssertionError("identity and official treatment matching counts differ")

    simultaneous_instances, control_simultaneous_support = (
        _simultaneous_reference_support(references, control_identity_pairs)
    )
    treatment_instances, treatment_simultaneous_support = (
        _simultaneous_reference_support(references, treatment_identity_pairs)
    )
    if treatment_instances != simultaneous_instances:
        raise AssertionError("simultaneous reference population changed")

    suppressed_status = Counter()
    anchor_status = Counter()
    suppression_cross = Counter()
    distance_bins = Counter({label: 0 for label, _, _ in DISTANCE_BINS})
    for suppression in suppressions:
        candidate_label = control_status[suppression.candidate.episode_id]
        anchor_label = control_status[suppression.anchor.episode_id]
        suppressed_status[candidate_label] += 1
        anchor_status[anchor_label] += 1
        suppression_cross[f"{candidate_label}|{anchor_label}"] += 1
        distance_bins[_distance_bin(suppression.distance_samples)] += 1
    if sum(suppressed_status.values()) != len(suppressions):
        raise AssertionError("suppression status audit is not exhaustive")

    return HeadRaw(
        control_metrics=_metrics_counter(control_metrics),
        treatment_metrics=_metrics_counter(treatment_metrics),
        control_delays=[prediction - reference for reference, prediction in control_pairs],
        treatment_delays=[
            prediction - reference for reference, prediction in treatment_pairs
        ],
        control_relations=control_relations,
        treatment_relations=treatment_relations,
        suppressed_status=suppressed_status,
        anchor_status=anchor_status,
        suppression_cross=suppression_cross,
        suppression_distance_bins=distance_bins,
        control_multiplicity=_multiplicity_histogram(
            candidate.sample for candidate in control
        ),
        treatment_multiplicity=_multiplicity_histogram(
            candidate.sample for candidate in treatment
        ),
        reference_multiplicity=_multiplicity_histogram(
            reference.sample for reference in references
        ),
        simultaneous_reference_instances=simultaneous_instances,
        control_simultaneous_support=control_simultaneous_support,
        treatment_simultaneous_support=treatment_simultaneous_support,
        suppressed_count=len(suppressions),
    )


def _with_density(metrics: Mapping[str, object], audio_seconds: float) -> Dict[str, object]:
    result = dict(metrics)
    predictions = int(result["prediction_count"])
    references = int(result["reference_count"])
    result["predictions_per_second"] = (
        predictions / audio_seconds if audio_seconds else None
    )
    result["prediction_to_reference_ratio"] = (
        predictions / references if references else None
    )
    return result


def merge_outcomes(outcomes: Sequence[TrackOutcome]) -> Dict[str, object]:
    values = tuple(outcomes)
    if not values:
        raise AssertionError("cannot aggregate an empty outcome set")
    audio_samples = sum(item.audio_samples for item in values)
    audio_seconds = audio_samples / SAMPLE_RATE
    merged_heads = {head: HeadRaw() for head in ("onset", "offset")}
    control_support = Counter()
    treatment_support = Counter()
    for outcome in values:
        for head in merged_heads:
            merged_heads[head].merge(outcome.heads[head])
        control_support.update(outcome.control_note_support)
        treatment_support.update(outcome.treatment_note_support)

    heads: Dict[str, object] = {}
    for head, raw in merged_heads.items():
        control = _complete_metrics(raw.control_metrics)
        treatment = _complete_metrics(raw.treatment_metrics)
        control_count = int(control["prediction_count"])
        treatment_count = int(treatment["prediction_count"])
        if control_count != treatment_count + raw.suppressed_count:
            raise AssertionError("aggregate kept and suppressed counts do not partition")
        heads[head] = {
            "control": _with_density(control, audio_seconds),
            "treatment": _with_density(treatment, audio_seconds),
            "change": {
                "prediction_count": treatment_count - control_count,
                "suppressed_count": raw.suppressed_count,
                "suppressed_percent": (
                    raw.suppressed_count / control_count * 100.0
                    if control_count
                    else 0.0
                ),
                "true_positive": int(treatment["true_positive"])
                - int(control["true_positive"]),
                "false_positive": int(treatment["false_positive"])
                - int(control["false_positive"]),
                "false_negative": int(treatment["false_negative"])
                - int(control["false_negative"]),
                "precision_points": float(treatment["precision"])
                - float(control["precision"]),
                "recall_points": float(treatment["recall"])
                - float(control["recall"]),
                "f1_points": float(treatment["f1"]) - float(control["f1"]),
                "true_positive_retention_percent": (
                    int(treatment["true_positive"])
                    / int(control["true_positive"])
                    * 100.0
                    if int(control["true_positive"])
                    else None
                ),
                "false_positive_reduction_percent": (
                    (int(control["false_positive"]) - int(treatment["false_positive"]))
                    / int(control["false_positive"])
                    * 100.0
                    if int(control["false_positive"])
                    else None
                ),
            },
            "control_timing": summarize_delays(raw.control_delays),
            "treatment_timing": summarize_delays(raw.treatment_delays),
            "control_false_positive_relations": {
                label: int(raw.control_relations[label])
                for label in STATUS_LABELS
                if label != "matched"
            },
            "treatment_false_positive_relations": {
                label: int(raw.treatment_relations[label])
                for label in STATUS_LABELS
                if label != "matched"
            },
            "suppressed_original_control_status": {
                label: int(raw.suppressed_status[label]) for label in STATUS_LABELS
            },
            "anchor_original_control_status": {
                label: int(raw.anchor_status[label]) for label in STATUS_LABELS
            },
            "suppressed_status_by_anchor_status": {
                key: int(raw.suppression_cross[key])
                for key in sorted(raw.suppression_cross)
            },
            "suppression_distance_bins_samples": {
                label: int(raw.suppression_distance_bins[label])
                for label, _, _ in DISTANCE_BINS
            },
            "same_sample_multiplicity": {
                "reference": _histogram_summary(raw.reference_multiplicity),
                "control": _histogram_summary(raw.control_multiplicity),
                "treatment": _histogram_summary(raw.treatment_multiplicity),
            },
            "simultaneous_reference_support": {
                "reference_instances": raw.simultaneous_reference_instances,
                "control_matched_instances": raw.control_simultaneous_support,
                "treatment_matched_instances": raw.treatment_simultaneous_support,
            },
        }

    track_f1_direction = {
        head: Counter(
            "improved"
            if outcome.heads[head].treatment_metrics["true_positive"]
            * (
                outcome.heads[head].control_metrics["prediction_count"]
                + outcome.heads[head].control_metrics["reference_count"]
            )
            > outcome.heads[head].control_metrics["true_positive"]
            * (
                outcome.heads[head].treatment_metrics["prediction_count"]
                + outcome.heads[head].treatment_metrics["reference_count"]
            )
            else "equal"
            if outcome.heads[head].treatment_metrics["true_positive"]
            * (
                outcome.heads[head].control_metrics["prediction_count"]
                + outcome.heads[head].control_metrics["reference_count"]
            )
            == outcome.heads[head].control_metrics["true_positive"]
            * (
                outcome.heads[head].treatment_metrics["prediction_count"]
                + outcome.heads[head].treatment_metrics["reference_count"]
            )
            else "worse"
            for outcome in values
        )
        for head in ("onset", "offset")
    }
    return {
        "tracks": len(values),
        "audio_samples": audio_samples,
        "audio_duration_seconds": audio_seconds,
        "heads": heads,
        "control_same_annotated_note_onset_offset_support": summarize_note_support(
            control_support
        ),
        "treatment_same_annotated_note_onset_offset_support": summarize_note_support(
            treatment_support
        ),
        "track_f1_direction": {
            head: {
                label: int(track_f1_direction[head][label])
                for label in ("improved", "equal", "worse")
            }
            for head in ("onset", "offset")
        },
    }


def aggregate_outcomes(outcomes: Sequence[TrackOutcome]) -> Dict[str, object]:
    values = tuple(outcomes)
    family_keys = sorted({(item.family, item.arrangement) for item in values})
    return {
        "global": merge_outcomes(values),
        "comp": merge_outcomes(tuple(item for item in values if item.arrangement == "comp")),
        "solo": merge_outcomes(tuple(item for item in values if item.arrangement == "solo")),
        "family_arrangement": [
            {
                "family": family,
                "arrangement": arrangement,
                **merge_outcomes(
                    tuple(
                        item
                        for item in values
                        if item.family == family and item.arrangement == arrangement
                    )
                ),
            }
            for family, arrangement in family_keys
        ],
    }


def _locked_configuration(arguments: argparse.Namespace) -> None:
    if arguments.chunk_size != 512:
        raise EvaluationError("chunk size is locked to 512")
    for name in (
        "onset_threshold",
        "offset_threshold",
        "onset_release_threshold",
        "offset_release_threshold",
    ):
        if not math.isclose(
            float(getattr(arguments, name)),
            OFFICIAL_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise EvaluationError(f"{name} is locked to {OFFICIAL_THRESHOLD}")
    if arguments.rearm_low_samples != REARM_LOW_SAMPLES:
        raise EvaluationError("rearm low samples is locked to 16")
    if arguments.control_refractory_samples != CONTROL_REFRACTORY_SAMPLES:
        raise EvaluationError("control refractory period is locked to zero")
    if arguments.treatment_refractory_samples != TREATMENT_REFRACTORY_SAMPLES:
        raise EvaluationError("treatment refractory period is locked to 2205")
    if not math.isclose(arguments.onset_tolerance_ms, 50.0, rel_tol=0.0, abs_tol=1e-12):
        raise EvaluationError("onset tolerance is locked to 50 ms")
    if not math.isclose(arguments.offset_tolerance_ms, 50.0, rel_tol=0.0, abs_tol=1e-12):
        raise EvaluationError("offset tolerance is locked to 50 ms")


def _expected_relations(source_track: Mapping[str, object], head: str) -> Dict[str, int]:
    try:
        relations = source_track["heads"][head]["false_positive_relations"]
        return {
            label: int(relations[label]["count"])
            for label in STATUS_LABELS
            if label != "matched"
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("Exp08 source relation metrics are invalid") from exc


def _decision(aggregates: Mapping[str, object]) -> Dict[str, object]:
    global_value = aggregates["global"]
    comp = aggregates["comp"]
    solo = aggregates["solo"]
    groups = aggregates["family_arrangement"]
    group_improvements = {
        head: sum(
            float(group["heads"][head]["treatment"]["f1"])
            > float(group["heads"][head]["control"]["f1"])
            for group in groups
        )
        for head in ("onset", "offset")
    }
    useful_checks = {
        "onset_false_positive_at_most_126553": int(
            global_value["heads"]["onset"]["treatment"]["false_positive"]
        )
        <= 126553,
        "offset_false_positive_at_most_213049": int(
            global_value["heads"]["offset"]["treatment"]["false_positive"]
        )
        <= 213049,
        "onset_true_positive_at_least_9055": int(
            global_value["heads"]["onset"]["treatment"]["true_positive"]
        )
        >= 9055,
        "offset_true_positive_at_least_8896": int(
            global_value["heads"]["offset"]["treatment"]["true_positive"]
        )
        >= 8896,
        "global_f1_improves_both_heads": all(
            float(global_value["heads"][head]["treatment"]["f1"])
            > float(global_value["heads"][head]["control"]["f1"])
            for head in ("onset", "offset")
        ),
        "comp_f1_improves_both_heads": all(
            float(comp["heads"][head]["treatment"]["f1"])
            > float(comp["heads"][head]["control"]["f1"])
            for head in ("onset", "offset")
        ),
        "solo_f1_improves_both_heads": all(
            float(solo["heads"][head]["treatment"]["f1"])
            > float(solo["heads"][head]["control"]["f1"])
            for head in ("onset", "offset")
        ),
        "at_least_10_of_12_groups_improve_each_head": all(
            group_improvements[head] >= 10 for head in ("onset", "offset")
        ),
        "simultaneous_reference_support_does_not_decrease": all(
            int(
                global_value["heads"][head]["simultaneous_reference_support"][
                    "treatment_matched_instances"
                ]
            )
            >= int(
                global_value["heads"][head]["simultaneous_reference_support"][
                    "control_matched_instances"
                ]
            )
            for head in ("onset", "offset")
        ),
        "both_boundary_note_support_at_least_7198": int(
            global_value["treatment_same_annotated_note_onset_offset_support"][
                "both"
            ]
        )
        >= 7198,
    }
    resolved_checks = {
        "onset_prediction_count_at_most_19082": int(
            global_value["heads"]["onset"]["treatment"]["prediction_count"]
        )
        <= 19082,
        "offset_prediction_count_at_most_19082": int(
            global_value["heads"]["offset"]["treatment"]["prediction_count"]
        )
        <= 19082,
        "onset_recall_at_least_0_9": float(
            global_value["heads"]["onset"]["treatment"]["recall"]
        )
        >= 0.9,
        "offset_recall_at_least_0_9": float(
            global_value["heads"]["offset"]["treatment"]["recall"]
        )
        >= 0.9,
    }
    return {
        "technical_integrity_valid": True,
        "family_arrangement_group_f1_improvements": group_improvements,
        "useful_filter_checks": useful_checks,
        "useful_filter_accepted": all(useful_checks.values()),
        "overprediction_resolved_on_adaptation_validation_checks": resolved_checks,
        "overprediction_resolved_on_adaptation_validation": all(
            resolved_checks.values()
        ),
        "adaptive_validation_result_only": True,
        "generalization_claim_allowed": False,
        "live_promotion_allowed": False,
        "training_started": False,
    }


def _track_result(outcome: TrackOutcome) -> Dict[str, object]:
    return {
        "annotation_member": outcome.annotation_member,
        "audio_member": outcome.audio_member,
        "family": outcome.family,
        "arrangement": outcome.arrangement,
        **merge_outcomes((outcome,)),
    }


def run_refractory_evaluation(arguments: argparse.Namespace) -> Dict[str, object]:
    wall_started_ns = time.perf_counter_ns()
    _locked_configuration(arguments)
    output_path = refuse_output_overwrite(arguments.output)
    dataset_dir = Path(arguments.dataset_dir).resolve()
    model_path = Path(arguments.model).resolve()
    metadata_path = _metadata_path(model_path, arguments.metadata).resolve()
    source_path = Path(arguments.source_report).resolve()
    protocol_path = Path(arguments.protocol).resolve()
    amendment_path = Path(arguments.protocol_amendment).resolve()
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

    locked_hashes = {
        "model": (_sha256(model_path), EXPECTED_MODEL_SHA256),
        "metadata": (_sha256(metadata_path), EXPECTED_METADATA_SHA256),
        "source": (_sha256(source_path), EXPECTED_SOURCE_SHA256),
        "relation_evaluator": (
            _sha256(DEFAULT_RELATION_EVALUATOR),
            EXPECTED_RELATION_EVALUATOR_SHA256,
        ),
        "protocol": (_sha256(protocol_path), EXPECTED_PROTOCOL_SHA256),
        "protocol_amendment": (
            _sha256(amendment_path),
            EXPECTED_PROTOCOL_AMENDMENT_SHA256,
        ),
        "detector": (_sha256(DEFAULT_DETECTOR_SOURCE), EXPECTED_DETECTOR_SHA256),
    }
    for name, (observed, expected) in locked_hashes.items():
        if observed != expected:
            raise EvaluationError(f"{name} SHA-256 differs from the locked value")
    print("[init] arguments and hashes validated", file=sys.stderr, flush=True)

    source = load_relation_source(source_path)
    source_tracks = _source_tracks(source)
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
    if set(validation_members) != set(source_tracks):
        raise EvaluationError("reconstructed validation split differs from Exp08")
    metadata = _load_and_validate_metadata(
        metadata_path,
        model_path=model_path,
        validation_members=validation_members,
        selected_players=selected_players,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
    )
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

    onset_tolerance = milliseconds_to_samples(arguments.onset_tolerance_ms)
    offset_tolerance = milliseconds_to_samples(arguments.offset_tolerance_ms)
    if onset_tolerance != 2205 or offset_tolerance != 2205:
        raise EvaluationError("official tolerances must resolve to 2205 samples")

    outcomes: List[TrackOutcome] = []
    inference_ns = control_ns = treatment_ns = observer_ns = trace_ns = analysis_ns = 0
    chunk_count = 0
    for track_index, track in enumerate(validation_tracks, start=1):
        print(
            f"[{track_index}/{len(validation_tracks)}] {track.annotation_member} start",
            file=sys.stderr,
            flush=True,
        )
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        flat_references = tuple(sorted(note for slot in slots for note in slot))
        references_by_head = identified_references(slots)
        decoded = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        if any(reference.offset_sample > decoded.frame_count for reference in flat_references):
            raise EvaluationError(
                f"reference boundary exceeds audio for {track.annotation_member!r}"
            )
        arrangement = _track_arrangement(track.annotation_member)
        if arrangement not in ("comp", "solo"):
            raise EvaluationError(f"unknown arrangement for {track.annotation_member!r}")

        predictor.reset()
        decoding = decode_refractory_stream(
            predictor,
            _audio_chunks(decoded, arguments.chunk_size),
            threshold=arguments.onset_threshold,
            rearm_low_samples=arguments.rearm_low_samples,
            treatment_refractory_samples=arguments.treatment_refractory_samples,
        )
        inference_ns += decoding.inference_elapsed_ns
        control_ns += decoding.control_elapsed_ns
        treatment_ns += decoding.treatment_elapsed_ns
        observer_ns += decoding.observer_elapsed_ns
        trace_ns += decoding.trace_elapsed_ns
        chunk_count += decoding.chunks

        control_onsets, control_offsets = _boundary_samples(
            decoding.control_candidates
        )
        treatment_onsets, treatment_offsets = _boundary_samples(
            decoding.treatment_candidates
        )
        control_evaluation, control_onset_pairs, control_offset_pairs = (
            evaluate_boundary_lists(
                flat_references,
                control_onsets,
                control_offsets,
                onset_tolerance_samples=onset_tolerance,
                offset_tolerance_samples=offset_tolerance,
            )
        )
        treatment_evaluation, treatment_onset_pairs, treatment_offset_pairs = (
            evaluate_boundary_lists(
                flat_references,
                treatment_onsets,
                treatment_offsets,
                onset_tolerance_samples=onset_tolerance,
                offset_tolerance_samples=offset_tolerance,
            )
        )
        control_metrics = _integer_evaluation(control_evaluation)
        treatment_metrics = _integer_evaluation(treatment_evaluation)
        source_track = source_tracks[track.annotation_member]
        expected_metrics = source_track.get("official_metrics", {}).get("treatment")
        if control_metrics != expected_metrics:
            raise AssertionError(
                f"N=16 control differs from Exp08 for {track.annotation_member!r}"
            )

        analysis_started = time.perf_counter_ns()
        control_by_head = {
            head: tuple(
                sorted(
                    (
                        item
                        for item in decoding.traced_control
                        if item.kind.value == head
                    ),
                    key=lambda item: (item.sample, item.channel),
                )
            )
            for head in ("onset", "offset")
        }
        treatment_by_head = {
            head: tuple(
                sorted(
                    (
                        item
                        for item in decoding.traced_treatment
                        if item.kind.value == head
                    ),
                    key=lambda item: (item.sample, item.channel),
                )
            )
            for head in ("onset", "offset")
        }
        suppressions_by_head = {
            head: tuple(
                item
                for item in decoding.suppressions
                if item.candidate.kind.value == head
            )
            for head in ("onset", "offset")
        }
        track_heads: Dict[str, HeadRaw] = {}
        pairs = {
            "onset": (control_onset_pairs, treatment_onset_pairs),
            "offset": (control_offset_pairs, treatment_offset_pairs),
        }
        for head in ("onset", "offset"):
            raw = _track_head_raw(
                references_by_head[head],
                control_by_head[head],
                treatment_by_head[head],
                suppressions_by_head[head],
                control_metrics[head],
                treatment_metrics[head],
                pairs[head][0],
                pairs[head][1],
                tolerance_samples=onset_tolerance,
            )
            expected_relations = _expected_relations(source_track, head)
            observed_relations = {
                label: int(raw.control_relations[label])
                for label in STATUS_LABELS
                if label != "matched"
            }
            if observed_relations != expected_relations:
                raise AssertionError(
                    f"Exp08 relation partition differs for {track.annotation_member!r} {head}"
                )
            track_heads[head] = raw

        control_note_support = note_support_counts(
            references_by_head,
            control_by_head,
            tolerance_samples=onset_tolerance,
        )
        treatment_note_support = note_support_counts(
            references_by_head,
            treatment_by_head,
            tolerance_samples=onset_tolerance,
        )
        expected_support = source_track.get(
            "same_annotated_note_onset_offset_support"
        )
        if summarize_note_support(control_note_support) != expected_support:
            raise AssertionError(
                f"Exp08 note support differs for {track.annotation_member!r}"
            )
        analysis_ns += time.perf_counter_ns() - analysis_started
        outcomes.append(
            TrackOutcome(
                annotation_member=track.annotation_member,
                audio_member=track.audio_member,
                family=_family(track.annotation_member),
                arrangement=arrangement,
                audio_samples=decoded.frame_count,
                heads=track_heads,
                control_note_support=control_note_support,
                treatment_note_support=treatment_note_support,
            )
        )
        print(
            f"[{track_index}/{len(validation_tracks)}] {track.annotation_member} complete",
            file=sys.stderr,
            flush=True,
        )

    aggregates = aggregate_outcomes(outcomes)
    if len(aggregates["family_arrangement"]) != 12:
        raise AssertionError("expected exactly twelve family-arrangement groups")
    source_global = source["aggregates"]["global"]
    for head in ("onset", "offset"):
        observed = {
            name: int(aggregates["global"]["heads"][head]["control"][name])
            for name in (
                "reference_count",
                "prediction_count",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        }
        expected = source_global["official_metrics"]["treatment"][head]
        if observed != expected:
            raise AssertionError(f"global Exp08 reproduction differs for {head}")
    if (
        aggregates["global"]["control_same_annotated_note_onset_offset_support"]
        != source_global["same_annotated_note_onset_offset_support"]
    ):
        raise AssertionError("global Exp08 note support reproduction differs")
    print("[audit] Exp08 reproduced globally and on 60/60 tracks", file=sys.stderr, flush=True)

    audio_seconds = aggregates["global"]["audio_duration_seconds"]
    compute_ns = (
        inference_ns
        + control_ns
        + treatment_ns
        + observer_ns
        + trace_ns
        + analysis_ns
    )
    result: Dict[str, object] = {
        "schema_version": 1,
        "kind": "boundary_candidate_fixed_refractory_evaluation",
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "source_report_path": str(source_path),
        "source_report_sha256": EXPECTED_SOURCE_SHA256,
        "protocol_path": str(protocol_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_amendment_path": str(amendment_path),
        "protocol_amendment_sha256": EXPECTED_PROTOCOL_AMENDMENT_SHA256,
        "detector_source_sha256": EXPECTED_DETECTOR_SHA256,
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
            "adaptive_duration_selection_on_same_validation_tracks": True,
            "independent_confirmation": False,
        },
        "configuration": {
            "chunk_size": arguments.chunk_size,
            "threshold": arguments.onset_threshold,
            "release_threshold": arguments.onset_release_threshold,
            "rearm_low_samples": arguments.rearm_low_samples,
            "control_refractory_samples": arguments.control_refractory_samples,
            "treatment_refractory_samples": arguments.treatment_refractory_samples,
            "treatment_refractory_ms": arguments.treatment_refractory_samples
            / SAMPLE_RATE
            * 1000.0,
            "anchor_updates_on_suppressed_candidate": False,
            "official_tolerance_samples": onset_tolerance,
            "official_tolerance_ms": arguments.onset_tolerance_ms,
            "model_passes": 1,
            "same_immutable_score_chunk": True,
            "association_enabled": False,
            "event_id_enabled": False,
            "scheduler_enabled": False,
            "official_live_activation_enabled": False,
            "public_fields": ["type", "position"],
            "private_channel_persisted_in_public_output": False,
        },
        "integrity": {
            "source_reproduced_globally": True,
            "source_tracks_reproduced": len(outcomes),
            "observer_control_projection_exact": True,
            "observer_treatment_projection_exact": True,
            "treatment_multiset_subset_of_control": True,
            "control_equals_kept_plus_suppressed": True,
            "same_type_channel_positive_gap": True,
            "suppression_distance_and_anchor_rule_exact": True,
            "retained_same_type_channel_gap_exceeds_2205": True,
            "positions_shifted": False,
            "cross_channel_suppression": False,
            "raw_scores_written": False,
            "raw_candidates_written": False,
        },
        "training_audit": {
            "training_change": False,
            "target_density": "not_applicable_unchanged",
            "sample_weights": "not_applicable_unchanged",
            "sampler_distribution": "not_applicable_unchanged",
            "loss_constant_output_optimum": "not_applicable_unchanged",
        },
        "aggregates": aggregates,
        "decision": _decision(aggregates),
        "runtime": {
            "audio_duration_seconds": audio_seconds,
            "chunks": chunk_count,
            "predictor_calls": chunk_count,
            "control_score_chunk_deliveries": chunk_count,
            "treatment_score_chunk_deliveries": chunk_count,
            "observer_score_chunk_deliveries": chunk_count,
            "model_inference_elapsed_seconds": inference_ns / 1_000_000_000.0,
            "control_decoding_elapsed_seconds": control_ns / 1_000_000_000.0,
            "treatment_decoding_elapsed_seconds": treatment_ns / 1_000_000_000.0,
            "observer_elapsed_seconds": observer_ns / 1_000_000_000.0,
            "trace_elapsed_seconds": trace_ns / 1_000_000_000.0,
            "analysis_elapsed_seconds": analysis_ns / 1_000_000_000.0,
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
            "Compare accepted N=16 candidates with a fixed causal 50 ms "
            "first-candidate refractory period."
        )
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=REPOSITORY_ROOT / "data" / "GuitarSet"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--protocol-amendment", type=Path, default=DEFAULT_PROTOCOL_AMENDMENT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--players", nargs="+", default=list(LOCKED_PLAYERS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction", type=_fraction, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=512)
    parser.add_argument("--onset-threshold", type=_probability, default=OFFICIAL_THRESHOLD)
    parser.add_argument("--offset-threshold", type=_probability, default=OFFICIAL_THRESHOLD)
    parser.add_argument(
        "--onset-release-threshold", type=_probability, default=OFFICIAL_THRESHOLD
    )
    parser.add_argument(
        "--offset-release-threshold", type=_probability, default=OFFICIAL_THRESHOLD
    )
    parser.add_argument(
        "--rearm-low-samples", type=_positive_int, default=REARM_LOW_SAMPLES
    )
    parser.add_argument(
        "--control-refractory-samples",
        type=int,
        default=CONTROL_REFRACTORY_SAMPLES,
    )
    parser.add_argument(
        "--treatment-refractory-samples",
        type=_positive_int,
        default=TREATMENT_REFRACTORY_SAMPLES,
    )
    parser.add_argument("--onset-tolerance-ms", type=_milliseconds, default=50.0)
    parser.add_argument("--offset-tolerance-ms", type=_milliseconds, default=50.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    result = run_refractory_evaluation(arguments)
    global_value = result["aggregates"]["global"]
    decision = result["decision"]
    print(
        json.dumps(
            {
                "output": result["output_path"],
                "onset": global_value["heads"]["onset"]["treatment"],
                "offset": global_value["heads"]["offset"]["treatment"],
                "useful_filter_accepted": decision["useful_filter_accepted"],
                "overprediction_resolved": decision[
                    "overprediction_resolved_on_adaptation_validation"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
