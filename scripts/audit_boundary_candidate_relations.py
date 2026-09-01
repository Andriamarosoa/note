"""Audit relations among V7 boundary candidates without changing decoding.

The trained model is run once per causal chunk.  The identical immutable score
object is delivered to the historical N=1 decoder, the accepted diagnostic
N=16 decoder, and an audit-only observer which retains private channel and
score morphology temporarily.  Only aggregate diagnostics are written; the
public candidate contract remains ``type, position``.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
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
    BoundaryScoreChunk,
    BoundaryType,
    LiveBoundaryPeakDecoder,
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
    EvaluationError,
    _fraction,
    _load_and_validate_metadata,
    _maximum_matching,
    _metadata_path,
    _milliseconds,
    _positive_int,
    _probability,
    _receptive_field,
    _track_arrangement,
    milliseconds_to_samples,
)
from scripts.evaluate_boundary_candidates import (  # noqa: E402
    OFFICIAL_THRESHOLD,
    _audio_chunks,
    _boundary_samples,
    _family,
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
DIAGNOSTIC_REARM_LOW_SAMPLES = 16
PROXIMITY_TOLERANCES_MS = (1.0, 5.0, 10.0, 20.0, 50.0)
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
    / "causal-boundaries-weight28-window512-v7-epoch08.unassociated-boundary-rearm-low1-16.json"
)
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-relations-protocol.json"
)
DEFAULT_PREAUDIT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-relations-preaudit.json"
)
DEFAULT_PROTOCOL_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-relations-protocol-amendment-01.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.candidate-relations.json"
)
EXPECTED_MODEL_SHA256 = (
    "5634ADD0E112A6889B65D5245AD051AD850A1FFFE66FEB8D9E5E74472BA114BF"
)
EXPECTED_SOURCE_SHA256 = (
    "4E5249C4AB2E6C3B170AA450BE8E27F6DFCF95692DF0917623C268485E20AD88"
)
EXPECTED_PROTOCOL_SHA256 = (
    "4EB624BC721A457B0CB512F2AD9F684AA32B70EF979FBC711A401DCEAB91C8D7"
)
EXPECTED_PREAUDIT_SHA256 = (
    "14A1E5B63B96FDEFC01163361B3E4E96229D8150541E05ED963699B49AE6922C"
)
EXPECTED_PROTOCOL_AMENDMENT_SHA256 = (
    "87D6521CC99642C1FF971EA4694826D7E9FF532DC4E9A71510E3EADBDC242011"
)


@dataclass(frozen=True)
class TracedCandidate:
    """One audit-only occurrence; channel is never a public output field."""

    kind: BoundaryType
    sample: int
    channel: int
    entry_score: float
    preceding_low_run_samples: Optional[int]
    previous_same_channel_gap_samples: Optional[int]
    episode_id: int
    survives_n16: bool


@dataclass(frozen=True)
class IdentifiedReference:
    kind: BoundaryType
    sample: int
    channel: int
    note_index: int

    @property
    def note_id(self) -> Tuple[int, int]:
        return self.channel, self.note_index


@dataclass(frozen=True)
class SourceExpectations:
    global_metrics: Mapping[str, Mapping[str, Mapping[str, int]]]
    tracks: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int]]]]


class CandidateRelationObserver:
    """Mirror N=1/N=16 rising edges while retaining diagnostic identity."""

    def __init__(
        self,
        slot_count: int,
        threshold: float = OFFICIAL_THRESHOLD,
        n16_low_samples: int = DIAGNOSTIC_REARM_LOW_SAMPLES,
    ) -> None:
        if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count <= 0:
            raise EvaluationError("observer slot_count must be an integer > 0")
        if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) <= 1.0:
            raise EvaluationError("observer threshold must be in (0, 1]")
        if (
            isinstance(n16_low_samples, bool)
            or not isinstance(n16_low_samples, int)
            or n16_low_samples <= 0
        ):
            raise EvaluationError("observer n16_low_samples must be an integer > 0")
        self._slot_count = slot_count
        self._threshold = float(threshold)
        self._n16_low_samples = n16_low_samples
        self._n1_high = {head: [False] * slot_count for head in ("onset", "offset")}
        self._n16_high = {head: [False] * slot_count for head in ("onset", "offset")}
        self._n16_low_counts = {
            head: [0] * slot_count for head in ("onset", "offset")
        }
        self._n16_low_starts: Dict[str, List[Optional[int]]] = {
            head: [None] * slot_count for head in ("onset", "offset")
        }
        self._last_n16_samples: Dict[str, List[Optional[int]]] = {
            head: [None] * slot_count for head in ("onset", "offset")
        }
        self._current_episode_ids: Dict[str, List[Optional[int]]] = {
            head: [None] * slot_count for head in ("onset", "offset")
        }
        self._next_episode_id = 0
        self._n1_candidates: List[TracedCandidate] = []
        self._n16_candidates: List[TracedCandidate] = []
        self._episode_front_counts: Counter = Counter()
        self._next_sample: Optional[int] = None

    @property
    def n1_candidates(self) -> Tuple[TracedCandidate, ...]:
        return tuple(self._n1_candidates)

    @property
    def n16_candidates(self) -> Tuple[TracedCandidate, ...]:
        return tuple(self._n16_candidates)

    @property
    def episode_front_counts(self) -> Mapping[int, int]:
        return dict(self._episode_front_counts)

    def process_chunk(self, scores: BoundaryScoreChunk) -> None:
        if not isinstance(scores, BoundaryScoreChunk):
            raise EvaluationError("observer requires BoundaryScoreChunk")
        if scores.sample_count and scores.slot_count != self._slot_count:
            raise EvaluationError("observer score slot count differs")
        if self._next_sample is not None and scores.start_sample != self._next_sample:
            raise EvaluationError("observer scores must be contiguous")
        for relative, (onset_row, offset_row) in enumerate(
            zip(scores.onset, scores.offset)
        ):
            sample = scores.start_sample + relative
            self._process_row("offset", BoundaryType.OFFSET, sample, offset_row)
            self._process_row("onset", BoundaryType.ONSET, sample, onset_row)
        self._next_sample = scores.start_sample + scores.sample_count

    def _process_row(
        self,
        head: str,
        kind: BoundaryType,
        sample: int,
        row: Sequence[float],
    ) -> None:
        for channel, raw_score in enumerate(row):
            score = float(raw_score)
            if not math.isfinite(score):
                raise EvaluationError("observer scores must be finite")

            starts_n16_episode = False
            preceding_low: Optional[int] = None
            if self._n16_high[head][channel]:
                if score < self._threshold:
                    if self._n16_low_counts[head][channel] == 0:
                        self._n16_low_starts[head][channel] = sample
                    self._n16_low_counts[head][channel] += 1
                    if self._n16_low_counts[head][channel] >= self._n16_low_samples:
                        self._n16_high[head][channel] = False
                else:
                    self._n16_low_counts[head][channel] = 0
                    self._n16_low_starts[head][channel] = None
            elif score >= self._threshold:
                low_start = self._n16_low_starts[head][channel]
                if low_start is not None:
                    preceding_low = sample - low_start
                previous = self._last_n16_samples[head][channel]
                previous_gap = None if previous is None else sample - previous
                episode_id = self._next_episode_id
                self._next_episode_id += 1
                traced = TracedCandidate(
                    kind=kind,
                    sample=sample,
                    channel=channel,
                    entry_score=score,
                    preceding_low_run_samples=preceding_low,
                    previous_same_channel_gap_samples=previous_gap,
                    episode_id=episode_id,
                    survives_n16=True,
                )
                self._n16_candidates.append(traced)
                self._current_episode_ids[head][channel] = episode_id
                self._last_n16_samples[head][channel] = sample
                self._n16_high[head][channel] = True
                self._n16_low_counts[head][channel] = 0
                self._n16_low_starts[head][channel] = None
                starts_n16_episode = True

            emits_n1 = False
            if self._n1_high[head][channel]:
                if score < self._threshold:
                    self._n1_high[head][channel] = False
            elif score >= self._threshold:
                self._n1_high[head][channel] = True
                emits_n1 = True

            if emits_n1:
                episode_id = self._current_episode_ids[head][channel]
                if episode_id is None:
                    raise AssertionError("N=1 edge has no N=16 episode")
                previous = self._last_n16_samples[head][channel]
                n1 = TracedCandidate(
                    kind=kind,
                    sample=sample,
                    channel=channel,
                    entry_score=score,
                    preceding_low_run_samples=preceding_low if starts_n16_episode else None,
                    previous_same_channel_gap_samples=(
                        None if previous is None or starts_n16_episode else sample - previous
                    ),
                    episode_id=episode_id,
                    survives_n16=starts_n16_episode,
                )
                self._n1_candidates.append(n1)
                self._episode_front_counts[episode_id] += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def gap_bin(samples: int) -> str:
    if samples == 0:
        return "0"
    if 1 <= samples <= 15:
        return "1-15"
    if samples <= 63:
        return "16-63"
    if samples <= 255:
        return "64-255"
    if samples <= 511:
        return "256-511"
    if samples <= 2205:
        return "512-2205"
    return "2206_plus"


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_numeric(
    values: Sequence[float], *, samples_to_ms: bool = False
) -> Dict[str, Optional[float]]:
    numeric = [float(value) for value in values]
    summary: Dict[str, Optional[float]] = {
        "count": len(numeric),
        "minimum": min(numeric) if numeric else None,
        "mean": sum(numeric) / len(numeric) if numeric else None,
        "p50": _percentile(numeric, 0.5),
        "p90": _percentile(numeric, 0.9),
        "maximum": max(numeric) if numeric else None,
    }
    if samples_to_ms:
        for name in ("minimum", "mean", "p50", "p90", "maximum"):
            value = summary[name]
            summary[f"{name}_ms"] = (
                None if value is None else value * 1000.0 / SAMPLE_RATE
            )
    return summary


def _canonical_integer_metrics(value: object) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        raise EvaluationError("source head metrics must be an object")
    result: Dict[str, int] = {}
    for name in (
        "reference_count",
        "prediction_count",
        "true_positive",
        "false_positive",
        "false_negative",
    ):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise EvaluationError(f"source metric {name} must be an integer >= 0")
        result[name] = item
    return result


def load_source_expectations(path: Path) -> SourceExpectations:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read source relation report: {path}") from exc
    if not isinstance(report, Mapping) or report.get("kind") != (
        "boundary_candidate_rearm_evaluation"
    ):
        raise EvaluationError("source must be the completed Exp07 report")
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise EvaluationError("source report lacks configuration")
    required_configuration = {
        "chunk_size": 512,
        "control_rearm_low_samples": CONTROL_REARM_LOW_SAMPLES,
        "treatment_rearm_low_samples": DIAGNOSTIC_REARM_LOW_SAMPLES,
        "onset_threshold": OFFICIAL_THRESHOLD,
        "offset_threshold": OFFICIAL_THRESHOLD,
        "onset_release_threshold": OFFICIAL_THRESHOLD,
        "offset_release_threshold": OFFICIAL_THRESHOLD,
        "official_live_activation": False,
    }
    for name, expected in required_configuration.items():
        observed = configuration.get(name)
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise EvaluationError(f"source configuration differs for {name}")
        elif observed != expected:
            raise EvaluationError(f"source configuration differs for {name}")

    aggregates = report.get("aggregates")
    global_value = aggregates.get("global") if isinstance(aggregates, Mapping) else None
    if not isinstance(global_value, Mapping):
        raise EvaluationError("source report lacks global aggregates")
    global_metrics: Dict[str, Dict[str, Dict[str, int]]] = {}
    for regime in ("control", "treatment"):
        value = global_value.get(regime)
        metrics = value.get("metrics") if isinstance(value, Mapping) else None
        if not isinstance(metrics, Mapping):
            raise EvaluationError(f"source report lacks global {regime} metrics")
        global_metrics[regime] = {
            head: _canonical_integer_metrics(metrics.get(head))
            for head in ("onset", "offset")
        }

    raw_tracks = report.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise EvaluationError("source report lacks per-track metrics")
    tracks: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    for raw in raw_tracks:
        if not isinstance(raw, Mapping):
            raise EvaluationError("source contains an invalid track")
        member = raw.get("annotation_member")
        if not isinstance(member, str) or not member or member in tracks:
            raise EvaluationError("source track identity is invalid or duplicated")
        tracks[member] = {}
        for regime in ("control", "treatment"):
            value = raw.get(regime)
            metrics = value.get("metrics") if isinstance(value, Mapping) else None
            if not isinstance(metrics, Mapping):
                raise EvaluationError(f"source track lacks {regime} metrics")
            tracks[member][regime] = {
                head: _canonical_integer_metrics(metrics.get(head))
                for head in ("onset", "offset")
            }
    return SourceExpectations(global_metrics=global_metrics, tracks=tracks)


def identified_references(slots) -> Mapping[str, Tuple[IdentifiedReference, ...]]:
    identified_notes = [
        (note, channel, note_index)
        for channel, slot in enumerate(slots)
        for note_index, note in enumerate(slot)
    ]
    identified_notes.sort(key=lambda item: item[0])
    return {
        "onset": tuple(
            IdentifiedReference(
                BoundaryType.ONSET,
                note.onset_sample,
                channel,
                note_index,
            )
            for note, channel, note_index in identified_notes
        ),
        "offset": tuple(
            IdentifiedReference(
                BoundaryType.OFFSET,
                note.offset_sample,
                channel,
                note_index,
            )
            for note, channel, note_index in identified_notes
        ),
    }


def _candidate_neighborhoods(
    references: Sequence[IdentifiedReference],
    candidates: Sequence[TracedCandidate],
    tolerance_samples: int,
) -> Tuple[Tuple[int, ...], ...]:
    chronological = sorted(
        (reference.sample, index) for index, reference in enumerate(references)
    )
    samples = [item[0] for item in chronological]
    neighborhoods: List[Tuple[int, ...]] = []
    for candidate in candidates:
        left = bisect_left(samples, candidate.sample - tolerance_samples)
        right = bisect_right(samples, candidate.sample + tolerance_samples)
        eligible = [chronological[index][1] for index in range(left, right)]
        eligible.sort(
            key=lambda reference_index: (
                abs(references[reference_index].sample - candidate.sample),
                references[reference_index].sample,
                reference_index,
            )
        )
        neighborhoods.append(tuple(eligible))
    return tuple(neighborhoods)


def official_index_matching(
    references: Sequence[IdentifiedReference],
    candidates: Sequence[TracedCandidate],
    tolerance_samples: int,
) -> Tuple[Tuple[int, int], ...]:
    neighborhoods = _candidate_neighborhoods(references, candidates, tolerance_samples)
    prediction_indices_by_reference: DefaultDict[int, List[int]] = defaultdict(list)
    for prediction_index, eligible in enumerate(neighborhoods):
        for reference_index in eligible:
            prediction_indices_by_reference[reference_index].append(prediction_index)
    ordered = {
        reference_index: tuple(
            sorted(
                indices,
                key=lambda prediction_index: (
                    abs(
                        candidates[prediction_index].sample
                        - references[reference_index].sample
                    ),
                    candidates[prediction_index].sample,
                    prediction_index,
                ),
            )
        )
        for reference_index, indices in prediction_indices_by_reference.items()
    }
    return _maximum_matching(len(references), len(candidates), ordered)


def _nearest_distance_to_samples(
    sample: int, values: Sequence[int]
) -> Optional[int]:
    if not values:
        return None
    position = bisect_left(values, sample)
    distances = []
    if position < len(values):
        distances.append(abs(values[position] - sample))
    if position:
        distances.append(abs(values[position - 1] - sample))
    return min(distances) if distances else None


@dataclass
class HeadAuditRaw:
    reference_count: int = 0
    candidate_count: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    fp_partition: Counter = field(default_factory=Counter)
    neighborhood_degrees: Counter = field(default_factory=Counter)
    channel_relation: Counter = field(default_factory=Counter)
    proximity_any: Counter = field(default_factory=Counter)
    proximity_same_channel: Counter = field(default_factory=Counter)
    successor_relations: Counter = field(default_factory=Counter)
    successor_gap_bins: Counter = field(default_factory=Counter)
    successor_relation_gap_bins: DefaultDict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    successor_gaps: List[int] = field(default_factory=list)
    preceding_low_runs: List[int] = field(default_factory=list)
    entry_scores: List[float] = field(default_factory=list)
    episode_front_histogram: Counter = field(default_factory=Counter)
    episode_n1_front_count: int = 0
    episode_maximum_n1_fronts: int = 0
    candidate_multiplicity_histogram: Counter = field(default_factory=Counter)
    reference_multiplicity_histogram: Counter = field(default_factory=Counter)

    def merge(self, other: "HeadAuditRaw") -> None:
        for name in (
            "reference_count",
            "candidate_count",
            "true_positive",
            "false_positive",
            "false_negative",
            "episode_n1_front_count",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.episode_maximum_n1_fronts = max(
            self.episode_maximum_n1_fronts,
            other.episode_maximum_n1_fronts,
        )
        for name in (
            "fp_partition",
            "neighborhood_degrees",
            "channel_relation",
            "proximity_any",
            "proximity_same_channel",
            "successor_relations",
            "successor_gap_bins",
            "episode_front_histogram",
            "candidate_multiplicity_histogram",
            "reference_multiplicity_histogram",
        ):
            getattr(self, name).update(getattr(other, name))
        for relation, counts in other.successor_relation_gap_bins.items():
            self.successor_relation_gap_bins[relation].update(counts)
        self.successor_gaps.extend(other.successor_gaps)
        self.preceding_low_runs.extend(other.preceding_low_runs)
        self.entry_scores.extend(other.entry_scores)


@dataclass
class TrackAudit:
    annotation_member: str
    audio_member: str
    family: str
    arrangement: str
    audio_samples: int
    heads: Mapping[str, HeadAuditRaw]
    note_support: Counter
    n1_counts: Mapping[str, int]
    n16_counts: Mapping[str, int]
    regime_metrics: Mapping[str, Mapping[str, Mapping[str, int]]]


def _multiplicity_histogram(samples: Iterable[int]) -> Counter:
    counts = Counter(samples)
    return Counter(counts.values())


def analyze_head(
    references: Sequence[IdentifiedReference],
    candidates: Sequence[TracedCandidate],
    n1_candidates: Sequence[TracedCandidate],
    episode_front_counts: Mapping[int, int],
    *,
    tolerance_samples: int,
    proximity_tolerances: Mapping[str, int],
) -> Tuple[HeadAuditRaw, Tuple[Tuple[int, ...], ...]]:
    references = tuple(references)
    candidates = tuple(sorted(candidates, key=lambda item: (item.sample, item.channel)))
    neighborhoods = _candidate_neighborhoods(
        references, candidates, tolerance_samples
    )
    pairs = official_index_matching(references, candidates, tolerance_samples)
    matched_predictions = {prediction_index for _, prediction_index in pairs}

    raw = HeadAuditRaw(
        reference_count=len(references),
        candidate_count=len(candidates),
        true_positive=len(pairs),
        false_positive=len(candidates) - len(pairs),
        false_negative=len(references) - len(pairs),
    )
    for label in proximity_tolerances:
        raw.proximity_any[label] += 0
        raw.proximity_same_channel[label] += 0
    for name in ("0", "1-15", "16-63", "64-255", "256-511", "512-2205", "2206_plus"):
        raw.successor_gap_bins[name] += 0
    chronological_samples = sorted(reference.sample for reference in references)
    channel_samples = {
        channel: sorted(
            reference.sample
            for reference in references
            if reference.channel == channel
        )
        for channel in {candidate.channel for candidate in candidates}
        | {reference.channel for reference in references}
    }
    for index, (candidate, eligible) in enumerate(zip(candidates, neighborhoods)):
        degree = len(eligible)
        raw.neighborhood_degrees[str(degree) if degree < 5 else "5_plus"] += 1
        same_channel_degree = sum(
            references[reference_index].channel == candidate.channel
            for reference_index in eligible
        )
        if degree == 0:
            raw.channel_relation["no_reference"] += 1
        elif same_channel_degree == 0:
            raw.channel_relation["other_channel_only"] += 1
        elif same_channel_degree == degree:
            raw.channel_relation["same_channel_available"] += 1
        else:
            raw.channel_relation["mixed_same_and_other_channel"] += 1
        any_distance = _nearest_distance_to_samples(
            candidate.sample, chronological_samples
        )
        same_distance = _nearest_distance_to_samples(
            candidate.sample, channel_samples.get(candidate.channel, ())
        )
        for label, tolerance in proximity_tolerances.items():
            if any_distance is not None and any_distance <= tolerance:
                raw.proximity_any[label] += 1
            if same_distance is not None and same_distance <= tolerance:
                raw.proximity_same_channel[label] += 1

    by_channel: DefaultDict[int, List[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_channel[candidate.channel].append(index)
        raw.entry_scores.append(candidate.entry_score)
        if candidate.preceding_low_run_samples is not None:
            raw.preceding_low_runs.append(candidate.preceding_low_run_samples)
    strong_repeat_fp_indices = set()
    for indices in by_channel.values():
        for previous_index, current_index in zip(indices, indices[1:]):
            previous = candidates[previous_index]
            current = candidates[current_index]
            gap = current.sample - previous.sample
            if gap < 0:
                raise AssertionError("same-channel candidates are not chronological")
            left = neighborhoods[previous_index]
            right = neighborhoods[current_index]
            if not left or not right:
                relation = "one_or_both_isolated"
            elif len(left) == 1 and len(right) == 1:
                relation = (
                    "same_unique_reference"
                    if left[0] == right[0]
                    else "distinct_unique_references"
                )
                if relation == "same_unique_reference":
                    if previous_index not in matched_predictions:
                        strong_repeat_fp_indices.add(previous_index)
                    if current_index not in matched_predictions:
                        strong_repeat_fp_indices.add(current_index)
            else:
                relation = "ambiguous"
            bin_name = gap_bin(gap)
            raw.successor_relations[relation] += 1
            raw.successor_gap_bins[bin_name] += 1
            raw.successor_relation_gap_bins[relation][bin_name] += 1
            raw.successor_gaps.append(gap)

    for index, eligible in enumerate(neighborhoods):
        if index in matched_predictions:
            continue
        if not eligible:
            raw.fp_partition["isolated"] += 1
        elif len(eligible) >= 2:
            raw.fp_partition["ambiguous_near"] += 1
        elif index in strong_repeat_fp_indices:
            raw.fp_partition[
                "same_channel_same_reference_successor_excess"
            ] += 1
        else:
            raw.fp_partition["single_reference_near_excess"] += 1
    if sum(raw.fp_partition.values()) != raw.false_positive:
        raise AssertionError("false-positive relation partition is not exhaustive")

    for candidate in candidates:
        count = episode_front_counts.get(candidate.episode_id)
        if count is None or count <= 0:
            raise AssertionError("N=16 episode lacks its N=1 front")
        raw.episode_front_histogram[str(count) if count < 5 else "5_plus"] += 1
        raw.episode_n1_front_count += count
        raw.episode_maximum_n1_fronts = max(raw.episode_maximum_n1_fronts, count)
    if sum(episode_front_counts.get(item.episode_id, 0) for item in candidates) != len(
        n1_candidates
    ):
        raise AssertionError("N=1 fronts do not partition into N=16 episodes")

    raw.candidate_multiplicity_histogram.update(
        _multiplicity_histogram(candidate.sample for candidate in candidates)
    )
    raw.reference_multiplicity_histogram.update(
        _multiplicity_histogram(reference.sample for reference in references)
    )
    return raw, neighborhoods


def note_support_counts(
    references_by_head: Mapping[str, Sequence[IdentifiedReference]],
    candidates_by_head: Mapping[str, Sequence[TracedCandidate]],
    *,
    tolerance_samples: int,
) -> Counter:
    support: Dict[Tuple[int, int], Dict[str, int]] = {}
    for head in ("onset", "offset"):
        for reference in references_by_head[head]:
            support.setdefault(reference.note_id, {})[head] = 0
        channels = {
            reference.channel for reference in references_by_head[head]
        } | {candidate.channel for candidate in candidates_by_head[head]}
        for channel in channels:
            references = tuple(
                reference
                for reference in references_by_head[head]
                if reference.channel == channel
            )
            candidates = tuple(
                sorted(
                    (
                        candidate
                        for candidate in candidates_by_head[head]
                        if candidate.channel == channel
                    ),
                    key=lambda item: item.sample,
                )
            )
            eligible = {
                reference_index: tuple(
                    sorted(
                        (
                            candidate_index
                            for candidate_index, candidate in enumerate(candidates)
                            if abs(candidate.sample - reference.sample)
                            <= tolerance_samples
                        ),
                        key=lambda candidate_index: (
                            abs(
                                candidates[candidate_index].sample
                                - reference.sample
                            ),
                            candidates[candidate_index].sample,
                            candidate_index,
                        ),
                    )
                )
                for reference_index, reference in enumerate(references)
            }
            pairs = _maximum_matching(len(references), len(candidates), eligible)
            for reference_index, _ in pairs:
                support[references[reference_index].note_id][head] = 1

    result = Counter()
    for values in support.values():
        onset = values.get("onset", 0)
        offset = values.get("offset", 0)
        if onset and offset:
            result["both"] += 1
        elif onset:
            result["onset_only"] += 1
        elif offset:
            result["offset_only"] += 1
        else:
            result["neither"] += 1
        result["onset_matched_supports"] += onset
        result["offset_matched_supports"] += offset
    result["notes"] = len(support)
    if (
        result["both"]
        + result["onset_only"]
        + result["offset_only"]
        + result["neither"]
        != result["notes"]
    ):
        raise AssertionError("note support partition is not exhaustive")
    return result


def _histogram_summary(histogram: Mapping[object, int]) -> Dict[str, object]:
    histogram = Counter(histogram)
    total_positions = sum(histogram.values())
    total_candidates = sum(int(key) * count for key, count in histogram.items())
    simultaneous_positions = sum(
        count for key, count in histogram.items() if int(key) >= 2
    )
    simultaneous_candidates = sum(
        int(key) * count for key, count in histogram.items() if int(key) >= 2
    )
    return {
        "event_count": total_candidates,
        "unique_position_count": total_positions,
        "simultaneous_position_count": simultaneous_positions,
        "simultaneous_event_count": simultaneous_candidates,
        "extra_simultaneous_event_count": simultaneous_candidates
        - simultaneous_positions,
        "maximum_multiplicity": max((int(key) for key in histogram), default=0),
        "position_histogram": {
            str(key): histogram[key]
            for key in sorted(histogram, key=lambda value: int(value))
        },
    }


def summarize_head(raw: HeadAuditRaw) -> Dict[str, object]:
    if raw.true_positive + raw.false_positive != raw.candidate_count:
        raise AssertionError("candidate metric counts are inconsistent")
    if raw.true_positive + raw.false_negative != raw.reference_count:
        raise AssertionError("reference metric counts are inconsistent")
    fp = raw.false_positive
    candidates = raw.candidate_count
    precision = raw.true_positive / candidates if candidates else 0.0
    recall = (
        raw.true_positive / raw.reference_count if raw.reference_count else 0.0
    )
    episode_count = sum(raw.episode_front_histogram.values())
    return {
        "official_metrics": {
            "reference_count": raw.reference_count,
            "prediction_count": raw.candidate_count,
            "true_positive": raw.true_positive,
            "false_positive": raw.false_positive,
            "false_negative": raw.false_negative,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        },
        "false_positive_relations": {
            name: {
                "count": int(raw.fp_partition.get(name, 0)),
                "percent_of_false_positives": (
                    raw.fp_partition.get(name, 0) / fp * 100.0 if fp else 0.0
                ),
            }
            for name in (
                "isolated",
                "same_channel_same_reference_successor_excess",
                "single_reference_near_excess",
                "ambiguous_near",
            )
        },
        "legacy_false_positive_relation": {
            "single_reference_excess": int(
                raw.fp_partition.get(
                    "same_channel_same_reference_successor_excess", 0
                )
                + raw.fp_partition.get("single_reference_near_excess", 0)
            ),
            "definition": "sum of the two one-reference classes retained for original protocol comparability",
        },
        "candidate_neighborhood_degree_histogram": dict(raw.neighborhood_degrees),
        "private_channel_relation": dict(raw.channel_relation),
        "proximity": {
            "any_reference": {
                label: {
                    "count": int(raw.proximity_any.get(label, 0)),
                    "percent": raw.proximity_any.get(label, 0) / candidates * 100.0
                    if candidates
                    else 0.0,
                }
                for label in sorted(raw.proximity_any, key=float)
            },
            "same_private_channel_reference": {
                label: {
                    "count": int(raw.proximity_same_channel.get(label, 0)),
                    "percent": raw.proximity_same_channel.get(label, 0)
                    / candidates
                    * 100.0
                    if candidates
                    else 0.0,
                }
                for label in sorted(raw.proximity_same_channel, key=float)
            },
        },
        "same_type_same_channel_successors": {
            "relations": dict(raw.successor_relations),
            "gap_bins": dict(raw.successor_gap_bins),
            "relation_gap_bins": {
                relation: dict(counts)
                for relation, counts in raw.successor_relation_gap_bins.items()
            },
            "gap_summary": summarize_numeric(
                raw.successor_gaps, samples_to_ms=True
            ),
        },
        "score_morphology": {
            "entry_score": summarize_numeric(raw.entry_scores),
            "preceding_low_run": summarize_numeric(
                raw.preceding_low_runs, samples_to_ms=True
            ),
        },
        "n1_fronts_per_n16_episode": {
            "episode_count": episode_count,
            "n1_front_count": raw.episode_n1_front_count,
            "suppressed_n1_front_count": raw.episode_n1_front_count
            - episode_count,
            "mean_n1_fronts": raw.episode_n1_front_count / episode_count
            if episode_count
            else 0.0,
            "maximum_n1_fronts": raw.episode_maximum_n1_fronts,
            "display_histogram": dict(raw.episode_front_histogram),
        },
        "same_sample_multiplicity": {
            "candidate": _histogram_summary(raw.candidate_multiplicity_histogram),
            "reference": _histogram_summary(raw.reference_multiplicity_histogram),
        },
    }


@dataclass(frozen=True)
class RelationStreamDecoding:
    control_candidates: Tuple[BoundaryCandidate, ...]
    diagnostic_candidates: Tuple[BoundaryCandidate, ...]
    traced_n1: Tuple[TracedCandidate, ...]
    traced_n16: Tuple[TracedCandidate, ...]
    episode_front_counts: Mapping[int, int]
    chunks: int
    inference_elapsed_ns: int
    control_decoding_elapsed_ns: int
    diagnostic_decoding_elapsed_ns: int
    observer_elapsed_ns: int


def decode_relation_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    *,
    threshold: float = OFFICIAL_THRESHOLD,
    control_rearm_low_samples: int = CONTROL_REARM_LOW_SAMPLES,
    diagnostic_rearm_low_samples: int = DIAGNOSTIC_REARM_LOW_SAMPLES,
) -> RelationStreamDecoding:
    slot_count = getattr(predictor, "slot_count", None)
    if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count <= 0:
        raise EvaluationError("predictor must expose a positive slot_count")
    common = {
        "slot_count": slot_count,
        "onset_threshold": threshold,
        "offset_threshold": threshold,
        "onset_release_threshold": threshold,
        "offset_release_threshold": threshold,
    }
    control_decoder = LiveBoundaryPeakDecoder(
        **common, rearm_low_samples=control_rearm_low_samples
    )
    diagnostic_decoder = LiveBoundaryPeakDecoder(
        **common, rearm_low_samples=diagnostic_rearm_low_samples
    )
    observer = CandidateRelationObserver(
        slot_count, threshold=threshold, n16_low_samples=diagnostic_rearm_low_samples
    )
    control: List[BoundaryCandidate] = []
    diagnostic: List[BoundaryCandidate] = []
    inference_ns = control_ns = diagnostic_ns = observer_ns = 0
    chunk_count = 0
    for start_sample, samples in chunks:
        started = time.perf_counter_ns()
        scores = predictor.predict_chunk(samples, start_sample=start_sample)
        inference_ns += time.perf_counter_ns() - started
        if not isinstance(scores, BoundaryScoreChunk):
            raise EvaluationError("predictor must return BoundaryScoreChunk")
        if scores.start_sample != start_sample or scores.sample_count != len(samples):
            raise EvaluationError("predictor returned a misaligned score chunk")
        chunk_count += 1

        started = time.perf_counter_ns()
        control.extend(control_decoder.process_chunk(scores))
        control_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        diagnostic.extend(diagnostic_decoder.process_chunk(scores))
        diagnostic_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        observer.process_chunk(scores)
        observer_ns += time.perf_counter_ns() - started

    traced_n1 = observer.n1_candidates
    traced_n16 = observer.n16_candidates
    public_control = tuple(control)
    public_diagnostic = tuple(diagnostic)
    traced_n1_projection = Counter(
        BoundaryCandidate(item.kind, item.sample) for item in traced_n1
    )
    traced_n16_projection = Counter(
        BoundaryCandidate(item.kind, item.sample) for item in traced_n16
    )
    if traced_n1_projection != Counter(public_control):
        raise AssertionError("relation observer does not reproduce N=1")
    if traced_n16_projection != Counter(public_diagnostic):
        raise AssertionError("relation observer does not reproduce N=16")
    if any(
        count > traced_n1_projection[candidate]
        for candidate, count in traced_n16_projection.items()
    ):
        raise AssertionError("N=16 candidate multiset is not contained in N=1")
    if sum(observer.episode_front_counts.values()) != len(traced_n1):
        raise AssertionError("observer episodes do not contain every N=1 front")
    return RelationStreamDecoding(
        control_candidates=public_control,
        diagnostic_candidates=public_diagnostic,
        traced_n1=traced_n1,
        traced_n16=traced_n16,
        episode_front_counts=observer.episode_front_counts,
        chunks=chunk_count,
        inference_elapsed_ns=inference_ns,
        control_decoding_elapsed_ns=control_ns,
        diagnostic_decoding_elapsed_ns=diagnostic_ns,
        observer_elapsed_ns=observer_ns,
    )


def _integer_evaluation(evaluation) -> Dict[str, Dict[str, int]]:
    return {
        head: {
            name: int(getattr(getattr(evaluation, head), name))
            for name in (
                "reference_count",
                "prediction_count",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        }
        for head in ("onset", "offset")
    }


NOTE_SUPPORT_KEYS = (
    "notes",
    "both",
    "onset_only",
    "offset_only",
    "neither",
    "onset_matched_supports",
    "offset_matched_supports",
)


def summarize_note_support(values: Mapping[str, int]) -> Dict[str, int]:
    return {name: int(values.get(name, 0)) for name in NOTE_SUPPORT_KEYS}


def _validate_track_source(
    member: str,
    observed: Mapping[str, Mapping[str, Mapping[str, int]]],
    expected: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int]]]],
) -> None:
    if member not in expected:
        raise EvaluationError(f"source report lacks track {member!r}")
    if observed != expected[member]:
        raise AssertionError(f"Exp07 reproduction differs for {member!r}")


def merge_tracks(outcomes: Sequence[TrackAudit]) -> Dict[str, object]:
    heads = {"onset": HeadAuditRaw(), "offset": HeadAuditRaw()}
    note_support = Counter()
    n1 = Counter()
    n16 = Counter()
    for outcome in outcomes:
        for head in ("onset", "offset"):
            heads[head].merge(outcome.heads[head])
        note_support.update(outcome.note_support)
        n1.update(outcome.n1_counts)
        n16.update(outcome.n16_counts)
    official_metrics = {
        regime: _sum_regime_metrics(outcomes, regime)
        for regime in ("control", "treatment")
    }
    for head in ("onset", "offset"):
        relation_metrics = {
            name: getattr(heads[head], name)
            for name in (
                "reference_count",
                "candidate_count",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        }
        treatment = official_metrics["treatment"][head]
        if relation_metrics != {
            "reference_count": treatment["reference_count"],
            "candidate_count": treatment["prediction_count"],
            "true_positive": treatment["true_positive"],
            "false_positive": treatment["false_positive"],
            "false_negative": treatment["false_negative"],
        }:
            raise AssertionError("relation and treatment aggregates differ")
    return {
        "tracks": len(outcomes),
        "audio_samples": sum(outcome.audio_samples for outcome in outcomes),
        "n1_candidate_counts": dict(n1),
        "n16_candidate_counts": dict(n16),
        "official_metrics": official_metrics,
        "heads": {head: summarize_head(raw) for head, raw in heads.items()},
        "same_annotated_note_onset_offset_support": summarize_note_support(
            note_support
        ),
    }


def aggregate_outcomes(outcomes: Sequence[TrackAudit]) -> Dict[str, object]:
    values = tuple(outcomes)
    family_keys = sorted({(item.family, item.arrangement) for item in values})
    return {
        "global": merge_tracks(values),
        "comp": merge_tracks(tuple(item for item in values if item.arrangement == "comp")),
        "solo": merge_tracks(tuple(item for item in values if item.arrangement == "solo")),
        "family_arrangement": [
            {
                "family": family,
                "arrangement": arrangement,
                **merge_tracks(
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


def _sum_regime_metrics(
    outcomes: Sequence[TrackAudit], regime: str
) -> Dict[str, Dict[str, int]]:
    result = {
        head: {
            name: 0
            for name in (
                "reference_count",
                "prediction_count",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        }
        for head in ("onset", "offset")
    }
    for outcome in outcomes:
        for head in ("onset", "offset"):
            for name, value in outcome.regime_metrics[regime][head].items():
                result[head][name] += value
    return result


def _same_probability(left: object, right: float) -> bool:
    return isinstance(left, (int, float)) and math.isclose(
        float(left), right, rel_tol=0.0, abs_tol=1e-12
    )


def _locked_configuration(arguments: argparse.Namespace) -> None:
    if arguments.chunk_size != 512:
        raise EvaluationError("relation audit chunk size is locked to 512")
    for name in (
        "onset_threshold",
        "offset_threshold",
        "onset_release_threshold",
        "offset_release_threshold",
    ):
        if not _same_probability(getattr(arguments, name), OFFICIAL_THRESHOLD):
            raise EvaluationError(f"{name} is locked to {OFFICIAL_THRESHOLD}")
    if arguments.control_rearm_low_samples != CONTROL_REARM_LOW_SAMPLES:
        raise EvaluationError("control rearm is locked to one low sample")
    if arguments.diagnostic_rearm_low_samples != DIAGNOSTIC_REARM_LOW_SAMPLES:
        raise EvaluationError("diagnostic rearm is locked to sixteen low samples")
    if not _same_probability(arguments.onset_tolerance_ms, 50.0) or not (
        _same_probability(arguments.offset_tolerance_ms, 50.0)
    ):
        raise EvaluationError("official relation tolerance is locked to 50 ms")


def _track_result(outcome: TrackAudit) -> Dict[str, object]:
    return {
        "annotation_member": outcome.annotation_member,
        "audio_member": outcome.audio_member,
        "family": outcome.family,
        "arrangement": outcome.arrangement,
        "audio_samples": outcome.audio_samples,
        "n1_candidate_counts": dict(outcome.n1_counts),
        "n16_candidate_counts": dict(outcome.n16_counts),
        "official_metrics": outcome.regime_metrics,
        "heads": {
            head: summarize_head(outcome.heads[head])
            for head in ("onset", "offset")
        },
        "same_annotated_note_onset_offset_support": summarize_note_support(
            outcome.note_support
        ),
    }


def run_relation_audit(arguments: argparse.Namespace) -> Dict[str, object]:
    wall_started_ns = time.perf_counter_ns()
    _locked_configuration(arguments)
    output_path = refuse_output_overwrite(arguments.output)
    dataset_dir = Path(arguments.dataset_dir).resolve()
    model_path = Path(arguments.model).resolve()
    metadata_path = _metadata_path(model_path, arguments.metadata).resolve()
    source_path = Path(arguments.source_report).resolve()
    protocol_path = Path(arguments.protocol).resolve()
    amendment_path = Path(arguments.protocol_amendment).resolve()
    preaudit_path = Path(arguments.preaudit).resolve()
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
        raise EvaluationError("source SHA-256 differs from completed Exp07")
    if _sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise EvaluationError("protocol SHA-256 differs from preregistered Exp08")
    amendment_sha256 = _sha256(amendment_path)
    if amendment_sha256 != EXPECTED_PROTOCOL_AMENDMENT_SHA256:
        raise EvaluationError("protocol amendment SHA-256 differs from reviewed Exp08 amendment")
    preaudit_sha256 = _sha256(preaudit_path)
    if preaudit_sha256 != EXPECTED_PREAUDIT_SHA256:
        raise EvaluationError("preaudit SHA-256 differs from committed Exp08 preaudit")
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
    source = load_source_expectations(source_path)
    if set(source.tracks) != set(validation_members):
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
    onset_tolerance_samples = milliseconds_to_samples(arguments.onset_tolerance_ms)
    offset_tolerance_samples = milliseconds_to_samples(arguments.offset_tolerance_ms)
    if onset_tolerance_samples != offset_tolerance_samples:
        raise EvaluationError("relation audit requires one common tolerance")
    tolerance_samples = onset_tolerance_samples
    proximity_tolerances = {
        f"{milliseconds:g}": milliseconds_to_samples(milliseconds)
        for milliseconds in PROXIMITY_TOLERANCES_MS
    }

    outcomes: List[TrackAudit] = []
    inference_ns = control_ns = diagnostic_ns = observer_ns = relation_ns = 0
    chunk_count = 0
    track_total = len(validation_tracks)
    for track_index, track in enumerate(validation_tracks, start=1):
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} start",
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
        decoding = decode_relation_stream(
            predictor,
            _audio_chunks(decoded, arguments.chunk_size),
            threshold=arguments.onset_threshold,
            control_rearm_low_samples=arguments.control_rearm_low_samples,
            diagnostic_rearm_low_samples=arguments.diagnostic_rearm_low_samples,
        )
        inference_ns += decoding.inference_elapsed_ns
        control_ns += decoding.control_decoding_elapsed_ns
        diagnostic_ns += decoding.diagnostic_decoding_elapsed_ns
        observer_ns += decoding.observer_elapsed_ns
        chunk_count += decoding.chunks

        control_onsets, control_offsets = _boundary_samples(
            decoding.control_candidates
        )
        diagnostic_onsets, diagnostic_offsets = _boundary_samples(
            decoding.diagnostic_candidates
        )
        control_evaluation, _, _ = evaluate_boundary_lists(
            flat_references,
            control_onsets,
            control_offsets,
            onset_tolerance_samples=tolerance_samples,
            offset_tolerance_samples=tolerance_samples,
        )
        diagnostic_evaluation, _, _ = evaluate_boundary_lists(
            flat_references,
            diagnostic_onsets,
            diagnostic_offsets,
            onset_tolerance_samples=tolerance_samples,
            offset_tolerance_samples=tolerance_samples,
        )
        regime_metrics = {
            "control": _integer_evaluation(control_evaluation),
            "treatment": _integer_evaluation(diagnostic_evaluation),
        }
        _validate_track_source(track.annotation_member, regime_metrics, source.tracks)

        started = time.perf_counter_ns()
        traced_n1_by_head = {
            head: tuple(
                item
                for item in decoding.traced_n1
                if item.kind.value == head
            )
            for head in ("onset", "offset")
        }
        traced_n16_by_head = {
            head: tuple(
                sorted(
                    (
                        item
                        for item in decoding.traced_n16
                        if item.kind.value == head
                    ),
                    key=lambda item: (item.sample, item.channel),
                )
            )
            for head in ("onset", "offset")
        }
        head_audits: Dict[str, HeadAuditRaw] = {}
        for head in ("onset", "offset"):
            raw, _ = analyze_head(
                references_by_head[head],
                traced_n16_by_head[head],
                traced_n1_by_head[head],
                decoding.episode_front_counts,
                tolerance_samples=tolerance_samples,
                proximity_tolerances=proximity_tolerances,
            )
            expected_head = regime_metrics["treatment"][head]
            observed_head = {
                "reference_count": raw.reference_count,
                "prediction_count": raw.candidate_count,
                "true_positive": raw.true_positive,
                "false_positive": raw.false_positive,
                "false_negative": raw.false_negative,
            }
            if observed_head != expected_head:
                raise AssertionError(
                    f"identity-aware matching differs for {track.annotation_member!r} {head}"
                )
            head_audits[head] = raw
        support = note_support_counts(
            references_by_head,
            traced_n16_by_head,
            tolerance_samples=tolerance_samples,
        )
        relation_ns += time.perf_counter_ns() - started
        n1_counts = {
            head: len(traced_n1_by_head[head]) for head in ("onset", "offset")
        }
        n16_counts = {
            head: len(traced_n16_by_head[head]) for head in ("onset", "offset")
        }
        outcomes.append(
            TrackAudit(
                annotation_member=track.annotation_member,
                audio_member=track.audio_member,
                family=_family(track.annotation_member),
                arrangement=arrangement,
                audio_samples=decoded.frame_count,
                heads=head_audits,
                note_support=support,
                n1_counts=n1_counts,
                n16_counts=n16_counts,
                regime_metrics=regime_metrics,
            )
        )
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} complete",
            file=sys.stderr,
            flush=True,
        )

    observed_global = {
        regime: _sum_regime_metrics(outcomes, regime)
        for regime in ("control", "treatment")
    }
    if observed_global != source.global_metrics:
        raise AssertionError("global Exp07 reproduction differs")
    aggregates = aggregate_outcomes(outcomes)
    if len(aggregates["family_arrangement"]) != 12:
        raise AssertionError("expected exactly twelve family-arrangement groups")
    print("[audit] Exp07 reproduced globally and on 60/60 tracks", file=sys.stderr, flush=True)

    audio_seconds = sum(outcome.audio_samples for outcome in outcomes) / SAMPLE_RATE
    compute_ns = inference_ns + control_ns + diagnostic_ns + observer_ns + relation_ns
    result: Dict[str, object] = {
        "schema_version": 1,
        "kind": "boundary_candidate_relation_audit",
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "metadata_path": str(metadata_path),
        "source_report_path": str(source_path),
        "source_report_sha256": EXPECTED_SOURCE_SHA256,
        "protocol_path": str(protocol_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_amendment_path": str(amendment_path),
        "protocol_amendment_sha256": amendment_sha256,
        "preaudit_path": str(preaudit_path),
        "preaudit_sha256": preaudit_sha256,
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
            "threshold": arguments.onset_threshold,
            "release_threshold": arguments.onset_release_threshold,
            "control_rearm_low_samples": arguments.control_rearm_low_samples,
            "diagnostic_rearm_low_samples": arguments.diagnostic_rearm_low_samples,
            "official_tolerance_samples": tolerance_samples,
            "official_tolerance_ms": arguments.onset_tolerance_ms,
            "proximity_tolerances_ms": list(PROXIMITY_TOLERANCES_MS),
            "model_passes": 1,
            "same_immutable_score_chunk": True,
            "association": False,
            "event_id": False,
            "interval_metrics": False,
            "scheduler_used": False,
            "official_live_activation": False,
            "public_fields": ["type", "position"],
            "private_channel_persisted_in_public_output": False,
        },
        "integrity": {
            "source_reproduced_globally": True,
            "source_tracks_reproduced": len(outcomes),
            "observer_n1_projection_exact": True,
            "observer_n16_projection_exact": True,
            "n16_multiset_subset_of_n1": True,
            "raw_scores_written": False,
            "raw_candidates_written": False,
        },
        "aggregates": aggregates,
        "runtime": {
            "audio_duration_seconds": audio_seconds,
            "chunks": chunk_count,
            "predictor_calls": chunk_count,
            "control_score_chunk_deliveries": chunk_count,
            "diagnostic_score_chunk_deliveries": chunk_count,
            "observer_score_chunk_deliveries": chunk_count,
            "model_inference_elapsed_seconds": inference_ns / 1_000_000_000.0,
            "control_decoding_elapsed_seconds": control_ns / 1_000_000_000.0,
            "diagnostic_decoding_elapsed_seconds": diagnostic_ns
            / 1_000_000_000.0,
            "observer_elapsed_seconds": observer_ns / 1_000_000_000.0,
            "relation_analysis_elapsed_seconds": relation_ns / 1_000_000_000.0,
            "compute_realtime_factor": compute_ns
            / 1_000_000_000.0
            / audio_seconds
            if audio_seconds
            else None,
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
        description="Audit temporal, channel and annotation relations among V7 candidates."
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPOSITORY_ROOT / "data" / "GuitarSet")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--protocol-amendment", type=Path, default=DEFAULT_PROTOCOL_AMENDMENT
    )
    parser.add_argument("--preaudit", type=Path, default=DEFAULT_PREAUDIT)
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
        "--control-rearm-low-samples",
        type=_positive_int,
        default=CONTROL_REARM_LOW_SAMPLES,
    )
    parser.add_argument(
        "--diagnostic-rearm-low-samples",
        type=_positive_int,
        default=DIAGNOSTIC_REARM_LOW_SAMPLES,
    )
    parser.add_argument("--onset-tolerance-ms", type=_milliseconds, default=50.0)
    parser.add_argument("--offset-tolerance-ms", type=_milliseconds, default=50.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    result = run_relation_audit(arguments)
    global_value = result["aggregates"]["global"]
    for head in ("onset", "offset"):
        metrics = global_value["heads"][head]["official_metrics"]
        relations = global_value["heads"][head]["false_positive_relations"]
        print(
            f"{head}: predictions={metrics['prediction_count']} "
            f"fp={metrics['false_positive']} "
            f"isolated={relations['isolated']['count']} "
            "same_channel_same_reference_successor_excess="
            f"{relations['same_channel_same_reference_successor_excess']['count']} "
            f"single_reference_near_excess={relations['single_reference_near_excess']['count']} "
            f"ambiguous={relations['ambiguous_near']['count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
