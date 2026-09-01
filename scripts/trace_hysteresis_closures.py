"""Trace same-event offset closures in the locked offset-only experiment.

The trained model is called exactly once for each causal audio chunk.  The
same immutable :class:`BoundaryScoreChunk` is then consumed by the official
control and treatment decoders and by an instrumented mirror of their state
machine.  The mirror must reproduce both public event sequences exactly before
any closure attribution is accepted.

An offset closure is called *suppressed* only when all of the following hold:

* the control emits an offset for ``(track, internal slot, exact onset)``;
* the treatment still has that exact event active in the same slot;
* the treatment emits no offset because its offset latch remained high.

Closures observed after identities have diverged are reported as cascades and
are never counted as same-event suppressions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    BoundaryEvent,
    BoundaryScoreChunk,
    BoundaryType,
    LiveBoundaryScoreDecoder,
)
from causal_note.guitarset import (  # noqa: E402
    ALLOWED_PLAYERS,
    SAMPLE_RATE,
    index_guitarset,
)
from causal_note.keras_predictor import KerasBoundaryPredictor  # noqa: E402
from scripts.evaluate_boundaries import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    EvaluationError,
    _load_and_validate_metadata,
    _metadata_path,
    _positive_int,
    _receptive_field,
    _track_arrangement,
)
from scripts.train_boundaries import (  # noqa: E402
    decode_pcm16_mono_wav,
    group_stem,
    split_tracks_by_group,
)


ENTRY_THRESHOLD = 0.55
ONSET_RELEASE_THRESHOLD = 0.55
CONTROL_OFFSET_RELEASE_THRESHOLD = 0.55
TREATMENT_OFFSET_RELEASE_THRESHOLD = 0.50
LOCKED_PLAYERS = ("00", "01", "02", "03", "04")
CONTROL = "control"
TREATMENT = "treatment"
PREDICTED_COUNT_KEYS = (
    "predicted_event_ids",
    "predicted_complete_events",
    "predicted_incomplete_events",
    "predicted_onset_without_offset_events",
    "predicted_offset_without_onset_events",
    "predicted_malformed_events",
    "raw_predicted_onsets",
    "raw_predicted_offsets",
)


@dataclass(frozen=True, order=True)
class EventIdentity:
    """Internal identity used only by this audit, never by public output."""

    track: str
    slot: int
    onset_sample: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "track": self.track,
            "internal_slot": self.slot,
            "onset_sample": self.onset_sample,
        }


@dataclass(frozen=True)
class _InternalEvent:
    kind: BoundaryType
    event_id: str
    identity: EventIdentity
    sample: int

    def public(self) -> BoundaryEvent:
        return BoundaryEvent(self.kind, self.event_id, self.sample)


@dataclass
class _Suppression:
    identity: EventIdentity
    control_close_sample: int
    offset_score: float
    rearmed_sample: Optional[int] = None
    treatment_close_sample: Optional[int] = None

    def example(self) -> Dict[str, object]:
        value = self.identity.as_dict()
        value.update(
            {
                "control_close_sample": self.control_close_sample,
                "offset_score_at_suppressed_closure": self.offset_score,
                "treatment_rearmed_sample": self.rearmed_sample,
                "treatment_close_sample": self.treatment_close_sample,
            }
        )
        return value


@dataclass(frozen=True)
class TraceSummary:
    summary: Mapping[str, object]
    recovery_latencies: Tuple[int, ...]
    permanent_examples: Tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class TraceStreamResult:
    events: Mapping[str, Tuple[BoundaryEvent, ...]]
    trace: TraceSummary
    chunks: int
    inference_elapsed_ns: int
    tracing_elapsed_ns: int
    official_public_sequences_equal: bool


class _CandidateTraceDecoder:
    """Instrumented exact mirror of ``LiveBoundaryScoreDecoder``."""

    def __init__(
        self,
        *,
        track: str,
        slot_count: int,
        onset_release_threshold: float,
        offset_release_threshold: float,
    ) -> None:
        self.track = track
        self.slot_count = slot_count
        self.onset_release_threshold = onset_release_threshold
        self.offset_release_threshold = offset_release_threshold
        self._slot_events: List[Optional[EventIdentity]] = [None] * slot_count
        self._slot_event_ids: List[Optional[str]] = [None] * slot_count
        self._onset_high = [False] * slot_count
        self._offset_high = [False] * slot_count
        self._next_id = 1
        self.started = 0
        self.closed = 0

    def active_identity(self, slot: int) -> Optional[EventIdentity]:
        return self._slot_events[slot]

    def offset_high(self, slot: int) -> bool:
        return self._offset_high[slot]

    def process_offset(
        self,
        slot: int,
        score: float,
        sample: int,
    ) -> Optional[_InternalEvent]:
        previous_high = self._offset_high[slot]
        high = score >= (
            self.offset_release_threshold
            if previous_high
            else ENTRY_THRESHOLD
        )
        identity = self._slot_events[slot]
        event_id = self._slot_event_ids[slot]
        emitted = None
        if high and not previous_high and identity is not None:
            if event_id is None:
                raise AssertionError("active trace identity has no public event id")
            if sample <= identity.onset_sample:
                raise AssertionError("offset must be after its traced onset")
            emitted = _InternalEvent(
                BoundaryType.OFFSET,
                event_id,
                identity,
                sample,
            )
            self._slot_events[slot] = None
            self._slot_event_ids[slot] = None
            self.closed += 1
        self._offset_high[slot] = high
        return emitted

    def process_onset(
        self,
        slot: int,
        score: float,
        sample: int,
    ) -> Optional[_InternalEvent]:
        previous_high = self._onset_high[slot]
        high = score >= (
            self.onset_release_threshold
            if previous_high
            else ENTRY_THRESHOLD
        )
        emitted = None
        if high and not previous_high and self._slot_events[slot] is None:
            identity = EventIdentity(self.track, slot, sample)
            event_id = f"event-{self._next_id:06d}"
            self._next_id += 1
            self._slot_events[slot] = identity
            self._slot_event_ids[slot] = event_id
            self.started += 1
            emitted = _InternalEvent(
                BoundaryType.ONSET,
                event_id,
                identity,
                sample,
            )
        self._onset_high[slot] = high
        return emitted

    def active_identities(self) -> Tuple[EventIdentity, ...]:
        return tuple(
            identity for identity in self._slot_events if identity is not None
        )


class OffsetOnlyClosureTracer:
    """Compare control and offset-only treatment at exact event identity."""

    def __init__(self, track: str, slot_count: int) -> None:
        if not isinstance(track, str) or not track:
            raise ValueError("track must be a non-empty string")
        if (
            isinstance(slot_count, bool)
            or not isinstance(slot_count, int)
            or slot_count <= 0
        ):
            raise ValueError("slot_count must be an integer > 0")
        self.track = track
        self.slot_count = slot_count
        self._candidate = {
            CONTROL: _CandidateTraceDecoder(
                track=track,
                slot_count=slot_count,
                onset_release_threshold=ONSET_RELEASE_THRESHOLD,
                offset_release_threshold=CONTROL_OFFSET_RELEASE_THRESHOLD,
            ),
            TREATMENT: _CandidateTraceDecoder(
                track=track,
                slot_count=slot_count,
                onset_release_threshold=ONSET_RELEASE_THRESHOLD,
                offset_release_threshold=TREATMENT_OFFSET_RELEASE_THRESHOLD,
            ),
        }
        self._events: Dict[str, List[BoundaryEvent]] = {
            CONTROL: [],
            TREATMENT: [],
        }
        self._next_sample: Optional[int] = None
        self._pending: Dict[EventIdentity, _Suppression] = {}
        self._recovered: List[_Suppression] = []
        self._same_event_control_closures = 0
        self._same_sample_closures = 0
        self._suppressed = 0
        self._control_identity_divergences = 0
        self._treatment_identity_divergences = 0
        self._finalized = False

    def process_chunk(
        self,
        scores: BoundaryScoreChunk,
    ) -> Mapping[str, Tuple[BoundaryEvent, ...]]:
        if self._finalized:
            raise RuntimeError("trace has already been finalized")
        if not isinstance(scores, BoundaryScoreChunk):
            raise ValueError("scores must be a BoundaryScoreChunk")
        if scores.sample_count and scores.slot_count != self.slot_count:
            raise ValueError(
                f"expected {self.slot_count} slots, got {scores.slot_count}"
            )
        if self._next_sample is not None and scores.start_sample != self._next_sample:
            raise ValueError(
                f"expected contiguous scores at {self._next_sample}, "
                f"got {scores.start_sample}"
            )

        emitted: Dict[str, List[BoundaryEvent]] = {
            CONTROL: [],
            TREATMENT: [],
        }
        control = self._candidate[CONTROL]
        treatment = self._candidate[TREATMENT]

        for relative_sample, (onset_row, offset_row) in enumerate(
            zip(scores.onset, scores.offset)
        ):
            sample = scores.start_sample + relative_sample

            # This loop intentionally precedes the onset loop, exactly as in
            # LiveBoundaryScoreDecoder, including at a retrigger sample.
            for slot, score in enumerate(offset_row):
                treatment_identity_before = treatment.active_identity(slot)
                treatment_high_before = treatment.offset_high(slot)

                control_event = control.process_offset(slot, score, sample)
                treatment_event = treatment.process_offset(slot, score, sample)
                treatment_high_after = treatment.offset_high(slot)

                if treatment_high_before and not treatment_high_after:
                    for suppression in self._pending.values():
                        if (
                            suppression.identity.slot == slot
                            and suppression.rearmed_sample is None
                            and treatment.active_identity(slot)
                            == suppression.identity
                        ):
                            suppression.rearmed_sample = sample

                if control_event is not None:
                    public = control_event.public()
                    emitted[CONTROL].append(public)
                    self._events[CONTROL].append(public)
                    if treatment_identity_before == control_event.identity:
                        self._same_event_control_closures += 1
                        if (
                            treatment_event is not None
                            and treatment_event.identity == control_event.identity
                        ):
                            self._same_sample_closures += 1
                        elif (
                            treatment_event is None
                            and treatment_high_before
                            and treatment_high_after
                            and score >= TREATMENT_OFFSET_RELEASE_THRESHOLD
                        ):
                            if control_event.identity in self._pending:
                                raise AssertionError(
                                    "same event received more than one suppression"
                                )
                            self._pending[control_event.identity] = _Suppression(
                                identity=control_event.identity,
                                control_close_sample=sample,
                                offset_score=score,
                            )
                            self._suppressed += 1
                        else:
                            raise AssertionError(
                                "same-event control closure has no valid "
                                "treatment explanation"
                            )
                    else:
                        # The treatment is empty or holds a different onset in
                        # this slot.  This is a downstream identity cascade,
                        # not a same-event closure suppressed by hysteresis.
                        self._control_identity_divergences += 1

                if treatment_event is not None:
                    public = treatment_event.public()
                    emitted[TREATMENT].append(public)
                    self._events[TREATMENT].append(public)
                    suppression = self._pending.pop(
                        treatment_event.identity,
                        None,
                    )
                    if suppression is not None:
                        suppression.treatment_close_sample = sample
                        self._recovered.append(suppression)
                    elif not (
                        control_event is not None
                        and control_event.identity == treatment_event.identity
                    ):
                        self._treatment_identity_divergences += 1

            for slot, score in enumerate(onset_row):
                for candidate_name, candidate in self._candidate.items():
                    event = candidate.process_onset(slot, score, sample)
                    if event is not None:
                        public = event.public()
                        emitted[candidate_name].append(public)
                        self._events[candidate_name].append(public)

        self._next_sample = scores.start_sample + scores.sample_count
        return {
            candidate: tuple(candidate_events)
            for candidate, candidate_events in emitted.items()
        }

    def events(self, candidate: str) -> Tuple[BoundaryEvent, ...]:
        return tuple(self._events[candidate])

    def finalize(self) -> TraceSummary:
        if self._finalized:
            raise RuntimeError("trace has already been finalized")
        self._finalized = True
        treatment_active = set(
            self._candidate[TREATMENT].active_identities()
        )
        if any(identity not in treatment_active for identity in self._pending):
            raise AssertionError(
                "suppressed event disappeared without a traced treatment offset"
            )

        permanent = tuple(
            sorted(
                self._pending.values(),
                key=lambda item: item.identity,
            )
        )
        recovered = tuple(self._recovered)
        latencies = tuple(
            int(item.treatment_close_sample) - item.control_close_sample
            for item in recovered
            if item.treatment_close_sample is not None
        )
        if any(value <= 0 for value in latencies):
            raise AssertionError("recovered offset latency must be positive")
        never_rearmed = sum(item.rearmed_sample is None for item in permanent)
        rearmed_no_later_offset = len(permanent) - never_rearmed

        candidate_counts = {}
        for name, candidate in self._candidate.items():
            incomplete = len(candidate.active_identities())
            candidate_counts[name] = {
                "predicted_event_ids": candidate.started,
                "predicted_complete_events": candidate.closed,
                "predicted_incomplete_events": incomplete,
                "predicted_onset_without_offset_events": incomplete,
                "predicted_offset_without_onset_events": 0,
                "predicted_malformed_events": 0,
                "raw_predicted_onsets": candidate.started,
                "raw_predicted_offsets": candidate.closed,
            }
            if candidate.started != candidate.closed + incomplete:
                raise AssertionError("traced candidate event accounting is invalid")

        summary: Dict[str, object] = {
            "candidate_counts": candidate_counts,
            "control_offset_closures": self._candidate[CONTROL].closed,
            "treatment_offset_closures": self._candidate[TREATMENT].closed,
            "same_event_control_closures": self._same_event_control_closures,
            "same_event_closed_at_same_sample": self._same_sample_closures,
            "same_event_offset_closure_opportunities_suppressed": self._suppressed,
            "suppressed_but_event_closed_later": len(recovered),
            "suppressed_and_event_remained_open_at_track_end": len(permanent),
            "permanent_never_rearmed_after_suppression": never_rearmed,
            "permanent_rearmed_but_no_later_offset": rearmed_no_later_offset,
            "control_closure_identity_divergence_or_cascade_not_attributed": (
                self._control_identity_divergences
            ),
            "treatment_closure_identity_divergence_or_cascade_not_attributed": (
                self._treatment_identity_divergences
            ),
            "recovery_latency": _latency_summary(latencies),
        }
        if self._same_sample_closures + self._suppressed != (
            self._same_event_control_closures
        ):
            raise AssertionError("same-event closure accounting is invalid")
        if len(recovered) + len(permanent) != self._suppressed:
            raise AssertionError("suppression outcome accounting is invalid")
        return TraceSummary(
            summary=summary,
            recovery_latencies=latencies,
            permanent_examples=tuple(item.example() for item in permanent),
        )


def _percentile(values: Sequence[int], percentile: float) -> Optional[float]:
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


def _latency_summary(values: Sequence[int]) -> Dict[str, object]:
    p50 = _percentile(values, 0.50)
    p90 = _percentile(values, 0.90)
    maximum = max(values) if values else None
    return {
        "count": len(values),
        "p50_samples": p50,
        "p90_samples": p90,
        "max_samples": maximum,
        "p50_ms": None if p50 is None else p50 * 1000.0 / SAMPLE_RATE,
        "p90_ms": None if p90 is None else p90 * 1000.0 / SAMPLE_RATE,
        "max_ms": (
            None if maximum is None else maximum * 1000.0 / SAMPLE_RATE
        ),
    }


def trace_offset_only_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    *,
    track: str,
) -> TraceStreamResult:
    """Infer once per chunk and prove the traced and official outputs equal."""

    slot_count = getattr(predictor, "slot_count", None)
    if (
        isinstance(slot_count, bool)
        or not isinstance(slot_count, int)
        or slot_count <= 0
    ):
        raise EvaluationError("predictor must expose a positive slot_count")
    predict_chunk = getattr(predictor, "predict_chunk", None)
    if not callable(predict_chunk):
        raise EvaluationError("predictor must implement predict_chunk")

    official = {
        CONTROL: LiveBoundaryScoreDecoder(
            slot_count=slot_count,
            onset_threshold=ENTRY_THRESHOLD,
            offset_threshold=ENTRY_THRESHOLD,
            onset_release_threshold=ONSET_RELEASE_THRESHOLD,
            offset_release_threshold=CONTROL_OFFSET_RELEASE_THRESHOLD,
        ),
        TREATMENT: LiveBoundaryScoreDecoder(
            slot_count=slot_count,
            onset_threshold=ENTRY_THRESHOLD,
            offset_threshold=ENTRY_THRESHOLD,
            onset_release_threshold=ONSET_RELEASE_THRESHOLD,
            offset_release_threshold=TREATMENT_OFFSET_RELEASE_THRESHOLD,
        ),
    }
    official_events: Dict[str, List[BoundaryEvent]] = {
        CONTROL: [],
        TREATMENT: [],
    }
    tracer = OffsetOnlyClosureTracer(track, slot_count)
    inference_elapsed_ns = 0
    tracing_elapsed_ns = 0
    chunk_count = 0

    for start_sample, samples in chunks:
        started = time.perf_counter_ns()
        scores = predict_chunk(samples, start_sample=start_sample)
        inference_elapsed_ns += time.perf_counter_ns() - started
        if not isinstance(scores, BoundaryScoreChunk):
            raise EvaluationError("predictor must return a BoundaryScoreChunk")
        chunk_count += 1

        official_chunk = {
            name: decoder.process_chunk(scores)
            for name, decoder in official.items()
        }
        started = time.perf_counter_ns()
        traced_chunk = tracer.process_chunk(scores)
        tracing_elapsed_ns += time.perf_counter_ns() - started
        for name in (CONTROL, TREATMENT):
            official_events[name].extend(official_chunk[name])
            if traced_chunk[name] != official_chunk[name]:
                raise AssertionError(
                    f"instrumented {name} sequence differs from "
                    f"LiveBoundaryScoreDecoder on {track!r} at chunk "
                    f"{start_sample}"
                )

    trace = tracer.finalize()
    events = {
        name: tuple(candidate_events)
        for name, candidate_events in official_events.items()
    }
    for name in (CONTROL, TREATMENT):
        if tracer.events(name) != events[name]:
            raise AssertionError(
                f"instrumented {name} full sequence differs from official decoder"
            )
        if len(official[name].active_events()) != int(
            trace.summary["candidate_counts"][name][
                "predicted_incomplete_events"
            ]
        ):
            raise AssertionError("official and traced active event counts differ")
    return TraceStreamResult(
        events=events,
        trace=trace,
        chunks=chunk_count,
        inference_elapsed_ns=inference_elapsed_ns,
        tracing_elapsed_ns=tracing_elapsed_ns,
        official_public_sequences_equal=True,
    )


def refuse_output_overwrite(output_path: Path) -> Path:
    resolved = Path(output_path).resolve()
    if resolved.exists():
        raise FileExistsError(
            f"refusing to replace hysteresis closure trace output: {resolved}"
        )
    return resolved


def _audio_chunks(decoded, chunk_size: int):
    for start in range(0, decoded.frame_count, chunk_size):
        integer_chunk = decoded.samples[start : start + chunk_size]
        yield start, tuple(sample / 32768.0 for sample in integer_chunk)


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON report: {path}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationError(f"JSON report must contain an object: {path}")
    return value


def _same_probability(left: object, right: float) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
    )


def _candidate_by_thresholds(
    report: Mapping[str, object],
    *,
    offset_release_threshold: float,
) -> Mapping[str, object]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise EvaluationError("source sweep has no candidate list")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and _same_probability(
            candidate.get("onset_release_threshold"),
            ONSET_RELEASE_THRESHOLD,
        )
        and _same_probability(
            candidate.get("offset_release_threshold"),
            offset_release_threshold,
        )
    ]
    if len(matches) != 1:
        raise EvaluationError(
            "source sweep must contain exactly one candidate with "
            f"onset release {ONSET_RELEASE_THRESHOLD} and offset release "
            f"{offset_release_threshold}"
        )
    return matches[0]


def _validate_source_report(
    report: Mapping[str, object],
    *,
    dataset_dir: Path,
    model_path: Path,
    metadata_path: Path,
    chunk_size: int,
) -> Mapping[str, Mapping[str, object]]:
    if report.get("kind") != "boundary_hysteresis_sweep":
        raise EvaluationError("source report is not a boundary hysteresis sweep")
    configuration = report.get("configuration")
    split = report.get("split")
    if not isinstance(configuration, Mapping) or not isinstance(split, Mapping):
        raise EvaluationError("source sweep lacks configuration or split")
    if configuration.get("release_sweep_mode") != "fixed_onset_offset_only":
        raise EvaluationError("source sweep is not the offset-only experiment")
    if not _same_probability(
        configuration.get("common_entry_threshold"), ENTRY_THRESHOLD
    ):
        raise EvaluationError("source sweep entry threshold is not locked to 0.55")
    if not _same_probability(
        configuration.get("fixed_onset_release_threshold"),
        ONSET_RELEASE_THRESHOLD,
    ):
        raise EvaluationError("source sweep onset release is not locked to 0.55")
    if configuration.get("chunk_size") != chunk_size:
        raise EvaluationError("trace chunk size must equal the source sweep")
    if configuration.get("model_passes") != 1:
        raise EvaluationError("source sweep did not use one model pass")

    if split.get("seed") != DEFAULT_SEED:
        raise EvaluationError("source sweep seed is not locked to 1337")
    if tuple(split.get("players", ())) != LOCKED_PLAYERS:
        raise EvaluationError("source sweep players are not exactly 00 through 04")
    if split.get("player_05_read") is not False:
        raise EvaluationError("source sweep does not prove player 05 remained locked")
    if not _same_probability(
        split.get("validation_fraction"), DEFAULT_VALIDATION_FRACTION
    ):
        raise EvaluationError("source sweep validation fraction differs")

    expected_paths = {
        "dataset_dir": dataset_dir,
        "model_path": model_path,
        "metadata_path": metadata_path,
    }
    for key, expected in expected_paths.items():
        raw = report.get(key)
        if not isinstance(raw, str) or Path(raw).resolve() != expected:
            raise EvaluationError(f"source sweep {key} differs from this trace")

    return {
        CONTROL: _candidate_by_thresholds(
            report,
            offset_release_threshold=CONTROL_OFFSET_RELEASE_THRESHOLD,
        ),
        TREATMENT: _candidate_by_thresholds(
            report,
            offset_release_threshold=TREATMENT_OFFSET_RELEASE_THRESHOLD,
        ),
    }


def _track_map(candidate: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    tracks = candidate.get("tracks")
    if not isinstance(tracks, list):
        raise EvaluationError("source candidate lacks per-track results")
    result: Dict[str, Mapping[str, object]] = {}
    for track in tracks:
        if not isinstance(track, Mapping):
            raise EvaluationError("source candidate contains an invalid track")
        member = track.get("annotation_member")
        if not isinstance(member, str) or member in result:
            raise EvaluationError("source candidate track identities are invalid")
        result[member] = track
    return result


def _predicted_counts(value: Mapping[str, object]) -> Dict[str, int]:
    result = {}
    for key in PREDICTED_COUNT_KEYS:
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise EvaluationError(f"invalid source count {key!r}")
        result[key] = raw
    return result


def _validate_track_counts(
    *,
    member: str,
    candidate_name: str,
    traced: Mapping[str, object],
    source_track: Mapping[str, object],
) -> None:
    metrics = source_track.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvaluationError(f"source track {member!r} has no metrics")
    expected = _predicted_counts(metrics)
    actual = {key: int(traced[key]) for key in PREDICTED_COUNT_KEYS}
    if actual != expected:
        raise EvaluationError(
            f"{candidate_name} trace counts differ from source sweep for "
            f"{member!r}: trace={actual}, source={expected}"
        )


def _merge_trace_summaries(
    traces: Sequence[TraceSummary],
) -> Tuple[Dict[str, object], Tuple[Mapping[str, object], ...]]:
    scalar_keys = (
        "control_offset_closures",
        "treatment_offset_closures",
        "same_event_control_closures",
        "same_event_closed_at_same_sample",
        "same_event_offset_closure_opportunities_suppressed",
        "suppressed_but_event_closed_later",
        "suppressed_and_event_remained_open_at_track_end",
        "permanent_never_rearmed_after_suppression",
        "permanent_rearmed_but_no_later_offset",
        "control_closure_identity_divergence_or_cascade_not_attributed",
        "treatment_closure_identity_divergence_or_cascade_not_attributed",
    )
    result: Dict[str, object] = {
        key: sum(int(trace.summary[key]) for trace in traces)
        for key in scalar_keys
    }
    candidate_counts = {}
    for candidate in (CONTROL, TREATMENT):
        candidate_counts[candidate] = {
            key: sum(
                int(trace.summary["candidate_counts"][candidate][key])
                for trace in traces
            )
            for key in PREDICTED_COUNT_KEYS
        }
    result["candidate_counts"] = candidate_counts
    latencies = tuple(
        value for trace in traces for value in trace.recovery_latencies
    )
    result["recovery_latency"] = _latency_summary(latencies)
    result["net_incomplete_event_increase"] = (
        candidate_counts[TREATMENT]["predicted_incomplete_events"]
        - candidate_counts[CONTROL]["predicted_incomplete_events"]
    )
    permanent_examples = tuple(
        sorted(
            (
                example
                for trace in traces
                for example in trace.permanent_examples
            ),
            key=lambda item: (
                str(item["track"]),
                int(item["internal_slot"]),
                int(item["onset_sample"]),
            ),
        )
    )
    return result, permanent_examples


def run_closure_trace(arguments: argparse.Namespace) -> Dict[str, object]:
    wall_started_ns = time.perf_counter_ns()
    output_path = refuse_output_overwrite(arguments.output)
    source_path = Path(arguments.source_report).resolve()
    dataset_dir = Path(arguments.dataset_dir).resolve()
    model_path = Path(arguments.model).resolve()
    metadata_path = _metadata_path(model_path, arguments.metadata).resolve()
    report = _load_json(source_path)
    source_candidates = _validate_source_report(
        report,
        dataset_dir=dataset_dir,
        model_path=model_path,
        metadata_path=metadata_path,
        chunk_size=arguments.chunk_size,
    )
    source_tracks = {
        name: _track_map(candidate)
        for name, candidate in source_candidates.items()
    }
    print("[init] offset-only source report validated", file=sys.stderr, flush=True)

    indexed = tuple(index_guitarset(dataset_dir))
    if any(track.player_id not in ALLOWED_PLAYERS for track in indexed):
        raise AssertionError("GuitarSet index admitted a locked player")
    _, validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        seed=DEFAULT_SEED,
    )
    validation_members = tuple(
        track.annotation_member for track in validation_tracks
    )
    split = report["split"]
    if tuple(split.get("validation_members", ())) != validation_members:
        raise EvaluationError("current locked validation split differs from source")
    if set(source_tracks[CONTROL]) != set(validation_members) or set(
        source_tracks[TREATMENT]
    ) != set(validation_members):
        raise EvaluationError("source per-track members differ from locked split")
    metadata = _load_and_validate_metadata(
        metadata_path,
        model_path=model_path,
        validation_members=validation_members,
        selected_players=LOCKED_PLAYERS,
        seed=DEFAULT_SEED,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
    )
    print(
        f"[init] seed 1337, players 00-04, {len(validation_tracks)} tracks; "
        "player 05 not loaded",
        file=sys.stderr,
        flush=True,
    )

    predictor = KerasBoundaryPredictor.from_path(
        str(model_path),
        receptive_field=_receptive_field(metadata),
    )
    predictor.warm_up(arguments.chunk_size)
    print("[init] model ready", file=sys.stderr, flush=True)

    track_rows: List[Dict[str, object]] = []
    trace_summaries: List[TraceSummary] = []
    total_audio_samples = 0
    inference_elapsed_ns = 0
    tracing_elapsed_ns = 0
    chunk_count = 0
    for index, track in enumerate(validation_tracks, start=1):
        print(
            f"[{index}/{len(validation_tracks)}] {track.annotation_member}",
            file=sys.stderr,
            flush=True,
        )
        if track.player_id == "05":
            raise AssertionError("locked player 05 reached audio loading")
        decoded = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        control_source_track = source_tracks[CONTROL][track.annotation_member]
        treatment_source_track = source_tracks[TREATMENT][track.annotation_member]
        for source_track in (control_source_track, treatment_source_track):
            if source_track.get("audio_member") != track.audio_member:
                raise EvaluationError("source audio member differs from current index")
            if source_track.get("audio_samples") != decoded.frame_count:
                raise EvaluationError("source audio length differs from current data")

        predictor.reset()
        stream = trace_offset_only_stream(
            predictor,
            _audio_chunks(decoded, arguments.chunk_size),
            track=track.annotation_member,
        )
        counts = stream.trace.summary["candidate_counts"]
        _validate_track_counts(
            member=track.annotation_member,
            candidate_name=CONTROL,
            traced=counts[CONTROL],
            source_track=control_source_track,
        )
        _validate_track_counts(
            member=track.annotation_member,
            candidate_name=TREATMENT,
            traced=counts[TREATMENT],
            source_track=treatment_source_track,
        )
        track_rows.append(
            {
                "annotation_member": track.annotation_member,
                "audio_member": track.audio_member,
                "arrangement": _track_arrangement(track.annotation_member),
                "audio_samples": decoded.frame_count,
                "trace": dict(stream.trace.summary),
                "source_counts_equal": {
                    CONTROL: True,
                    TREATMENT: True,
                },
                "official_public_sequences_equal": True,
            }
        )
        trace_summaries.append(stream.trace)
        total_audio_samples += decoded.frame_count
        inference_elapsed_ns += stream.inference_elapsed_ns
        tracing_elapsed_ns += stream.tracing_elapsed_ns
        chunk_count += stream.chunks

    answer, permanent_examples = _merge_trace_summaries(trace_summaries)
    for candidate_name in (CONTROL, TREATMENT):
        source_counts = source_candidates[candidate_name].get("counts")
        if not isinstance(source_counts, Mapping):
            raise EvaluationError("source candidate lacks global counts")
        expected = _predicted_counts(source_counts)
        actual = {
            key: int(answer["candidate_counts"][candidate_name][key])
            for key in PREDICTED_COUNT_KEYS
        }
        if actual != expected:
            raise EvaluationError(
                f"global {candidate_name} trace differs from source: "
                f"trace={actual}, source={expected}"
            )

    audio_duration_seconds = total_audio_samples / SAMPLE_RATE
    wall_elapsed_ns = time.perf_counter_ns() - wall_started_ns
    result: Dict[str, object] = {
        "schema_version": 1,
        "kind": "offset_only_hysteresis_closure_trace",
        "source_report": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest().upper(),
        },
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "split": {
            "seed": DEFAULT_SEED,
            "validation_fraction": DEFAULT_VALIDATION_FRACTION,
            "players": list(LOCKED_PLAYERS),
            "player_05_read": False,
            "validation_tracks": len(validation_tracks),
            "validation_groups": len(
                {group_stem(track) for track in validation_tracks}
            ),
            "validation_members": list(validation_members),
        },
        "configuration": {
            "chunk_size": arguments.chunk_size,
            "entry_thresholds": {
                "onset": ENTRY_THRESHOLD,
                "offset": ENTRY_THRESHOLD,
            },
            "control_release_thresholds": {
                "onset": ONSET_RELEASE_THRESHOLD,
                "offset": CONTROL_OFFSET_RELEASE_THRESHOLD,
            },
            "treatment_release_thresholds": {
                "onset": ONSET_RELEASE_THRESHOLD,
                "offset": TREATMENT_OFFSET_RELEASE_THRESHOLD,
            },
            "model_predictions_per_audio_chunk": 1,
            "event_identity": "(annotation_member, internal_slot, exact_onset_sample)",
            "sample_order": "offset_before_onset",
            "suppression_definition": (
                "The control closes an event while the treatment still has "
                "the exact same event active and emits no offset because its "
                "offset latch remained high. Identity divergences are excluded."
            ),
        },
        "answer": answer,
        "trace_validation": {
            "official_live_boundary_score_decoder_sequences_equal": True,
            "source_counts_equal_per_track": True,
            "source_counts_equal_global": True,
            "model_predict_calls": chunk_count,
            "model_predictions_per_audio_chunk": 1,
            "same_score_chunk_passed_to_both_official_decoders_and_trace": True,
            "player_05_read": False,
        },
        "per_track": track_rows,
        "permanent_examples": list(
            permanent_examples[: arguments.max_permanent_examples]
        ),
        "permanent_examples_total": len(permanent_examples),
        "permanent_examples_truncated": (
            len(permanent_examples) > arguments.max_permanent_examples
        ),
        "runtime": {
            "audio_duration_seconds": audio_duration_seconds,
            "chunks": chunk_count,
            "model_inference_elapsed_seconds": (
                inference_elapsed_ns / 1_000_000_000.0
            ),
            "trace_elapsed_seconds": tracing_elapsed_ns / 1_000_000_000.0,
            "model_inference_realtime_factor": (
                inference_elapsed_ns / 1_000_000_000.0 / audio_duration_seconds
                if audio_duration_seconds
                else None
            ),
            "wall_elapsed_seconds": wall_elapsed_ns / 1_000_000_000.0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    result["output_path"] = str(output_path)
    return result


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace exact offset closure suppression in the locked offset-only "
            "0.55/0.50 hysteresis experiment."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "GuitarSet",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPOSITORY_ROOT / "model" / "causal-boundaries.keras",
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=_positive_int, default=512)
    parser.add_argument(
        "--max-permanent-examples",
        type=_positive_int,
        default=50,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    result = run_closure_trace(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
