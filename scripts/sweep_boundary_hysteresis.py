"""One-pass release-threshold audit for the calibrated causal decoder.

The model is evaluated exactly once for every audio chunk.  The immutable
score chunk is then passed to one independent ``LiveBoundaryScoreDecoder`` per
release candidate.  By default, a candidate applies the same release threshold
to onset and offset, preserving the historical common-threshold sweep.  A
fixed onset release threshold can instead isolate an offset-only sweep.
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
    load_boundary_slots,
)
from causal_note.keras_predictor import KerasBoundaryPredictor  # noqa: E402
from scripts.evaluate_boundaries import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    EvaluationError,
    TrackEvaluation,
    _flatten_references,
    _fraction,
    _load_and_validate_metadata,
    _metadata_path,
    _milliseconds,
    _positive_int,
    _probability,
    _receptive_field,
    _track_arrangement,
    aggregate_track_evaluations,
    evaluate_track_events,
    match_boundaries,
    milliseconds_to_samples,
)
from scripts.train_boundaries import (  # noqa: E402
    decode_pcm16_mono_wav,
    group_stem,
    split_tracks_by_group,
)


DEFAULT_ENTRY_THRESHOLD = 0.55
DEFAULT_RELEASE_THRESHOLDS = (0.55, 0.50)


@dataclass(frozen=True)
class HysteresisThresholds:
    """Resolved per-head release thresholds for one candidate."""

    onset_release_threshold: float
    offset_release_threshold: float


@dataclass(frozen=True)
class HysteresisDecoding:
    """Events, morphology, and timing from one streaming model pass."""

    events: Mapping[float, Tuple[BoundaryEvent, ...]]
    morphology: Mapping[float, Mapping[str, object]]
    inference_elapsed_ns: int
    decoding_elapsed_ns: Mapping[float, int]
    morphology_elapsed_ns: Mapping[float, int]
    chunks: int
    candidate_thresholds: Mapping[float, HysteresisThresholds]


class _HeadDipTracker:
    """Count entry-high runs joined by a release-band dip per slot."""

    def __init__(
        self,
        slot_count: int,
        entry_threshold: float,
        release_threshold: float,
    ) -> None:
        self._entry_threshold = entry_threshold
        self._release_threshold = release_threshold
        self._eligible_high = [False] * slot_count
        self._pending_samples = [0] * slot_count
        self.count = 0
        self.samples = 0
        self.maximum_samples = 0

    def process_rows(self, rows: Sequence[Sequence[float]]) -> None:
        for row in rows:
            for slot, score in enumerate(row):
                if score >= self._entry_threshold:
                    duration = self._pending_samples[slot]
                    if duration:
                        self.count += 1
                        self.samples += duration
                        self.maximum_samples = max(
                            self.maximum_samples,
                            duration,
                        )
                    self._pending_samples[slot] = 0
                    self._eligible_high[slot] = True
                elif score >= self._release_threshold:
                    if self._eligible_high[slot]:
                        self._pending_samples[slot] += 1
                else:
                    # This dip crossed the release threshold, so a later high
                    # run is a genuine re-arm rather than a bridged dip.
                    self._pending_samples[slot] = 0
                    self._eligible_high[slot] = False

    def raw_summary(self) -> Dict[str, int]:
        return {
            "bridged_dip_count": self.count,
            "bridged_dip_samples": self.samples,
            "maximum_bridged_dip_samples": self.maximum_samples,
        }


class _DipMorphologyTracker:
    """Track onset and offset dips without retaining dense model scores."""

    def __init__(
        self,
        slot_count: int,
        entry_threshold: float,
        onset_release_threshold: float,
        offset_release_threshold: float,
    ) -> None:
        self._entry_threshold = entry_threshold
        self._onset_release_threshold = onset_release_threshold
        self._offset_release_threshold = offset_release_threshold
        self._onset = _HeadDipTracker(
            slot_count,
            entry_threshold,
            onset_release_threshold,
        )
        self._offset = _HeadDipTracker(
            slot_count,
            entry_threshold,
            offset_release_threshold,
        )

    def process_chunk(self, scores: BoundaryScoreChunk) -> None:
        # An entry-equal release has an empty release band.  Skip that head's
        # dense scan independently so an offset-only sweep never attributes
        # offset morphology to the unchanged onset head.
        if self._onset_release_threshold < self._entry_threshold:
            self._onset.process_rows(scores.onset)
        if self._offset_release_threshold < self._entry_threshold:
            self._offset.process_rows(scores.offset)

    def summary(self) -> Dict[str, object]:
        onset = self._onset.raw_summary()
        offset = self._offset.raw_summary()
        combined = _merge_raw_morphology((onset, offset))
        return {
            "onset": _format_morphology(onset),
            "offset": _format_morphology(offset),
            "combined": _format_morphology(combined),
        }


def _finite_probability(name: str, raw_value: object) -> float:
    if isinstance(raw_value, bool):
        raise EvaluationError(f"{name} must be a finite probability in (0, 1]")
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError(
            f"{name} must be a finite probability in (0, 1]"
        ) from exc
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise EvaluationError(
            f"{name} must be a finite probability in (0, 1]"
        )
    return value


def validate_hysteresis_thresholds(
    entry_threshold: float,
    release_thresholds: Sequence[float],
) -> Tuple[float, Tuple[float, ...]]:
    """Validate a common entry threshold and ordered unique releases."""

    entry = _finite_probability("entry threshold", entry_threshold)
    releases: List[float] = []
    for raw_release in release_thresholds:
        release = _finite_probability("release thresholds", raw_release)
        if release > entry:
            raise EvaluationError(
                "release thresholds must be less than or equal to the entry "
                "threshold"
            )
        if release in releases:
            raise EvaluationError("release thresholds must not contain duplicates")
        releases.append(release)
    if not releases:
        raise EvaluationError("at least one release threshold is required")
    return entry, tuple(releases)


def _resolve_candidate_thresholds(
    entry_threshold: float,
    release_thresholds: Sequence[float],
    fixed_onset_release_threshold: Optional[float],
) -> Tuple[float, Tuple[float, ...], Dict[float, HysteresisThresholds]]:
    """Resolve historical common or fixed-onset/offset-only candidates."""

    entry, releases = validate_hysteresis_thresholds(
        entry_threshold,
        release_thresholds,
    )
    fixed_onset = None
    if fixed_onset_release_threshold is not None:
        fixed_onset = _finite_probability(
            "fixed onset release threshold",
            fixed_onset_release_threshold,
        )
        if fixed_onset > entry:
            raise EvaluationError(
                "fixed onset release threshold must be less than or equal "
                "to the entry threshold"
            )
    return entry, releases, {
        release: HysteresisThresholds(
            onset_release_threshold=(
                release if fixed_onset is None else fixed_onset
            ),
            offset_release_threshold=release,
        )
        for release in releases
    }


def refuse_output_overwrite(output_path: Path) -> Path:
    """Resolve an output path and fail before inference if it exists."""

    resolved = Path(output_path).resolve()
    if resolved.exists():
        raise FileExistsError(
            f"refusing to replace hysteresis sweep output: {resolved}"
        )
    return resolved


def _merge_raw_morphology(
    summaries: Iterable[Mapping[str, object]],
) -> Dict[str, int]:
    count = 0
    samples = 0
    maximum = 0
    for summary in summaries:
        count += int(summary["bridged_dip_count"])
        samples += int(summary["bridged_dip_samples"])
        maximum = max(
            maximum,
            int(summary["maximum_bridged_dip_samples"]),
        )
    return {
        "bridged_dip_count": count,
        "bridged_dip_samples": samples,
        "maximum_bridged_dip_samples": maximum,
    }


def _format_morphology(raw: Mapping[str, object]) -> Dict[str, object]:
    count = int(raw["bridged_dip_count"])
    samples = int(raw["bridged_dip_samples"])
    maximum = int(raw["maximum_bridged_dip_samples"])
    mean_samples = samples / count if count else None
    return {
        "bridged_dip_count": count,
        "bridged_dip_samples": samples,
        "bridged_dip_duration_seconds": samples / SAMPLE_RATE,
        "mean_bridged_dip_samples": mean_samples,
        "mean_bridged_dip_ms": (
            mean_samples * 1000.0 / SAMPLE_RATE
            if mean_samples is not None
            else None
        ),
        "maximum_bridged_dip_samples": maximum,
        "maximum_bridged_dip_ms": maximum * 1000.0 / SAMPLE_RATE,
    }


def decode_hysteresis_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    entry_threshold: float,
    release_thresholds: Sequence[float],
    *,
    fixed_onset_release_threshold: Optional[float] = None,
) -> HysteresisDecoding:
    """Infer each chunk once and decode independent release candidates.

    Without ``fixed_onset_release_threshold``, every candidate retains the
    historical behavior of applying its release threshold to both heads.  If
    fixed, only the offset release threshold varies across candidates.
    """

    entry, releases, candidate_thresholds = _resolve_candidate_thresholds(
        entry_threshold,
        release_thresholds,
        fixed_onset_release_threshold,
    )
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

    decoders = {
        release: LiveBoundaryScoreDecoder(
            slot_count=slot_count,
            onset_threshold=entry,
            offset_threshold=entry,
            onset_release_threshold=(
                candidate_thresholds[release].onset_release_threshold
            ),
            offset_release_threshold=(
                candidate_thresholds[release].offset_release_threshold
            ),
        )
        for release in releases
    }
    morphologies = {
        release: _DipMorphologyTracker(
            slot_count,
            entry,
            candidate_thresholds[release].onset_release_threshold,
            candidate_thresholds[release].offset_release_threshold,
        )
        for release in releases
    }
    emitted: Dict[float, List[BoundaryEvent]] = {
        release: [] for release in releases
    }
    decoding_elapsed_ns = {release: 0 for release in releases}
    morphology_elapsed_ns = {release: 0 for release in releases}
    inference_elapsed_ns = 0
    chunk_count = 0

    for start_sample, samples in chunks:
        inference_started = time.perf_counter_ns()
        scores = predict_chunk(samples, start_sample=start_sample)
        inference_elapsed_ns += time.perf_counter_ns() - inference_started
        chunk_count += 1

        for release in releases:
            decoding_started = time.perf_counter_ns()
            emitted[release].extend(decoders[release].process_chunk(scores))
            decoding_elapsed_ns[release] += (
                time.perf_counter_ns() - decoding_started
            )

            thresholds = candidate_thresholds[release]
            if (
                thresholds.onset_release_threshold < entry
                or thresholds.offset_release_threshold < entry
            ):
                morphology_started = time.perf_counter_ns()
                morphologies[release].process_chunk(scores)
                morphology_elapsed_ns[release] += (
                    time.perf_counter_ns() - morphology_started
                )

    return HysteresisDecoding(
        events={release: tuple(emitted[release]) for release in releases},
        morphology={
            release: morphologies[release].summary() for release in releases
        },
        inference_elapsed_ns=inference_elapsed_ns,
        decoding_elapsed_ns=dict(decoding_elapsed_ns),
        morphology_elapsed_ns=dict(morphology_elapsed_ns),
        chunks=chunk_count,
        candidate_thresholds=dict(candidate_thresholds),
    )


def _audio_chunks(decoded, chunk_size: int):
    for start in range(0, decoded.frame_count, chunk_size):
        integer_chunk = decoded.samples[start : start + chunk_size]
        yield start, tuple(sample / 32768.0 for sample in integer_chunk)


def _new_release_store(releases: Sequence[float]):
    return {
        release: {
            "evaluations": [],
            "matched_onsets": [],
            "matched_offsets": [],
            "tracks": [],
            "morphologies": [],
            "decoding_elapsed_ns": 0,
            "morphology_elapsed_ns": 0,
        }
        for release in releases
    }


def _aggregate_arrangements(
    evaluations: Sequence[TrackEvaluation],
    audio_samples_by_track: Sequence[int],
    matched_onsets_by_track: Sequence[Sequence[Tuple[int, int]]],
    matched_offsets_by_track: Sequence[Sequence[Tuple[int, int]]],
    arrangements_by_track: Sequence[Optional[str]],
) -> Dict[str, Dict[str, object]]:
    aggregates = {
        "global": aggregate_track_evaluations(
            evaluations,
            audio_samples_by_track,
            matched_onsets_by_track,
            matched_offsets_by_track,
        )
    }
    for arrangement in ("comp", "solo"):
        indices = tuple(
            index
            for index, value in enumerate(arrangements_by_track)
            if value == arrangement
        )
        aggregates[arrangement] = aggregate_track_evaluations(
            tuple(evaluations[index] for index in indices),
            tuple(audio_samples_by_track[index] for index in indices),
            tuple(matched_onsets_by_track[index] for index in indices),
            tuple(matched_offsets_by_track[index] for index in indices),
        )
    return aggregates


def _raw_head_morphology(
    morphology: Mapping[str, object],
    head: str,
) -> Dict[str, int]:
    value = morphology[head]
    if not isinstance(value, Mapping):
        raise EvaluationError("invalid morphology summary")
    return {
        "bridged_dip_count": int(value["bridged_dip_count"]),
        "bridged_dip_samples": int(value["bridged_dip_samples"]),
        "maximum_bridged_dip_samples": int(
            value["maximum_bridged_dip_samples"]
        ),
    }


def _aggregate_morphology(
    morphologies: Sequence[Mapping[str, object]],
    arrangements_by_track: Sequence[Optional[str]],
) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for arrangement in ("global", "comp", "solo"):
        indices = tuple(
            range(len(morphologies))
            if arrangement == "global"
            else (
                index
                for index, value in enumerate(arrangements_by_track)
                if value == arrangement
            )
        )
        head_raw = {
            head: _merge_raw_morphology(
                _raw_head_morphology(morphologies[index], head)
                for index in indices
            )
            for head in ("onset", "offset")
        }
        combined = _merge_raw_morphology(head_raw.values())
        result[arrangement] = {
            "onset": _format_morphology(head_raw["onset"]),
            "offset": _format_morphology(head_raw["offset"]),
            "combined": _format_morphology(combined),
        }
    return result


def run_hysteresis_sweep(arguments: argparse.Namespace) -> Dict[str, object]:
    """Run the locked full-track hysteresis audit with one model pass."""

    wall_started_ns = time.perf_counter_ns()
    output_path = refuse_output_overwrite(arguments.output)
    fixed_onset_release_threshold = getattr(
        arguments,
        "fixed_onset_release_threshold",
        None,
    )
    entry, releases, candidate_thresholds = _resolve_candidate_thresholds(
        arguments.entry_threshold,
        arguments.release_thresholds,
        fixed_onset_release_threshold,
    )
    dataset_dir = Path(arguments.dataset_dir).resolve()
    model_path = Path(arguments.model).resolve()
    metadata_path = _metadata_path(model_path, arguments.metadata).resolve()
    selected_players = tuple(dict.fromkeys(arguments.players))
    if not selected_players or any(
        player not in ALLOWED_PLAYERS for player in selected_players
    ):
        raise EvaluationError("players must be selected from 00 through 04")
    if arguments.seed != DEFAULT_SEED:
        raise EvaluationError(
            f"full-track sweep is locked to split seed {DEFAULT_SEED}"
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

    print("[init] loading model", file=sys.stderr, flush=True)
    predictor = KerasBoundaryPredictor.from_path(
        str(model_path),
        receptive_field=_receptive_field(metadata),
    )
    predictor.warm_up(arguments.chunk_size)
    print("[init] model ready", file=sys.stderr, flush=True)
    onset_tolerance_samples = milliseconds_to_samples(
        arguments.onset_tolerance_ms
    )
    offset_tolerance_samples = milliseconds_to_samples(
        arguments.offset_tolerance_ms
    )

    stores = _new_release_store(releases)
    audio_samples_by_track: List[int] = []
    arrangements_by_track: List[Optional[str]] = []
    inference_elapsed_ns = 0
    chunk_count = 0
    track_total = len(validation_tracks)

    for track_index, track in enumerate(validation_tracks, start=1):
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} start",
            file=sys.stderr,
            flush=True,
        )
        slots = load_boundary_slots(
            track.annotation_zip,
            track.annotation_member,
        )
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
        audio_samples_by_track.append(decoded.frame_count)
        arrangement = _track_arrangement(track.annotation_member)
        arrangements_by_track.append(arrangement)

        predictor.reset()
        decoded_releases = decode_hysteresis_stream(
            predictor,
            _audio_chunks(decoded, arguments.chunk_size),
            entry,
            releases,
            fixed_onset_release_threshold=(
                fixed_onset_release_threshold
            ),
        )
        print(
            f"[{track_index}/{track_total}] {track.annotation_member} decoded",
            file=sys.stderr,
            flush=True,
        )
        inference_elapsed_ns += decoded_releases.inference_elapsed_ns
        chunk_count += decoded_releases.chunks

        reference_onsets = tuple(
            reference.onset_sample for reference in references
        )
        reference_offsets = tuple(
            reference.offset_sample for reference in references
        )
        for release in releases:
            predictions = decoded_releases.events[release]
            evaluation = evaluate_track_events(
                references,
                predictions,
                onset_tolerance_samples=onset_tolerance_samples,
                offset_tolerance_samples=offset_tolerance_samples,
            )
            predicted_onsets = tuple(
                event.sample
                for event in predictions
                if event.kind is BoundaryType.ONSET
            )
            predicted_offsets = tuple(
                event.sample
                for event in predictions
                if event.kind is BoundaryType.OFFSET
            )
            matched_onsets = match_boundaries(
                reference_onsets,
                predicted_onsets,
                onset_tolerance_samples,
            )
            matched_offsets = match_boundaries(
                reference_offsets,
                predicted_offsets,
                offset_tolerance_samples,
            )
            store = stores[release]
            store["evaluations"].append(evaluation)
            store["matched_onsets"].append(matched_onsets)
            store["matched_offsets"].append(matched_offsets)
            store["decoding_elapsed_ns"] += (
                decoded_releases.decoding_elapsed_ns[release]
            )
            store["morphology_elapsed_ns"] += (
                decoded_releases.morphology_elapsed_ns[release]
            )
            track_morphology = decoded_releases.morphology[release]
            store["morphologies"].append(track_morphology)
            store["tracks"].append(
                {
                    "annotation_member": track.annotation_member,
                    "audio_member": track.audio_member,
                    "arrangement": arrangement,
                    "audio_samples": decoded.frame_count,
                    "metrics": asdict(evaluation),
                    "morphology": track_morphology,
                }
            )
            print(
                f"[{track_index}/{track_total}] {track.annotation_member} "
                "onset_release="
                f"{candidate_thresholds[release].onset_release_threshold:.2f} "
                "offset_release="
                f"{candidate_thresholds[release].offset_release_threshold:.2f} "
                "evaluated",
                file=sys.stderr,
                flush=True,
            )

        print(
            f"[{track_index}/{track_total}] {track.annotation_member} complete",
            file=sys.stderr,
            flush=True,
        )

    total_audio_samples = sum(audio_samples_by_track)
    audio_duration_seconds = total_audio_samples / SAMPLE_RATE
    candidates = []
    total_decoding_ns = 0
    total_morphology_ns = 0
    for release in releases:
        store = stores[release]
        aggregates = _aggregate_arrangements(
            store["evaluations"],
            audio_samples_by_track,
            store["matched_onsets"],
            store["matched_offsets"],
            arrangements_by_track,
        )
        morphology = _aggregate_morphology(
            store["morphologies"],
            arrangements_by_track,
        )
        decoding_ns = int(store["decoding_elapsed_ns"])
        morphology_ns = int(store["morphology_elapsed_ns"])
        total_decoding_ns += decoding_ns
        total_morphology_ns += morphology_ns
        global_aggregate = aggregates["global"]
        candidate_compute_ns = decoding_ns + morphology_ns
        candidates.append(
            {
                "entry_threshold": entry,
                # Retained as the historical candidate key.  It is identical
                # to offset_release_threshold in offset-only mode.
                "release_threshold": release,
                "onset_release_threshold": (
                    candidate_thresholds[release].onset_release_threshold
                ),
                "offset_release_threshold": (
                    candidate_thresholds[release].offset_release_threshold
                ),
                "counts": global_aggregate["counts"],
                "metrics": global_aggregate["metrics"],
                "rates_per_hour": global_aggregate["rates_per_hour"],
                "aggregates": aggregates,
                "morphology": morphology,
                "runtime": {
                    "decoding_elapsed_seconds": decoding_ns / 1_000_000_000.0,
                    "morphology_elapsed_seconds": (
                        morphology_ns / 1_000_000_000.0
                    ),
                    "decoding_plus_morphology_realtime_factor": (
                        candidate_compute_ns
                        / 1_000_000_000.0
                        / audio_duration_seconds
                        if audio_duration_seconds
                        else None
                    ),
                },
                "tracks": store["tracks"],
            }
        )

    compute_ns = (
        inference_elapsed_ns + total_decoding_ns + total_morphology_ns
    )
    result: Dict[str, object] = {
        "schema_version": 1,
        "kind": "boundary_hysteresis_sweep",
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
            "common_entry_threshold": entry,
            "release_sweep_mode": (
                "common_onset_and_offset"
                if fixed_onset_release_threshold is None
                else "fixed_onset_offset_only"
            ),
            "common_release_thresholds": (
                list(releases)
                if fixed_onset_release_threshold is None
                else None
            ),
            "fixed_onset_release_threshold": (
                fixed_onset_release_threshold
            ),
            "offset_release_thresholds": list(releases),
            "candidate_release_thresholds": [
                {
                    "release_threshold": release,
                    "onset_release_threshold": (
                        candidate_thresholds[release].onset_release_threshold
                    ),
                    "offset_release_threshold": (
                        candidate_thresholds[release].offset_release_threshold
                    ),
                }
                for release in releases
            ],
            "onset_tolerance_ms": arguments.onset_tolerance_ms,
            "onset_tolerance_samples": onset_tolerance_samples,
            "offset_tolerance_ms": arguments.offset_tolerance_ms,
            "offset_tolerance_samples": offset_tolerance_samples,
            "model_passes": 1,
            "morphology_definition": (
                "A bridged dip is a contiguous per-slot score run below the "
                "entry threshold but at or above the candidate release "
                "threshold, bounded before and after by entry-high runs."
            ),
        },
        "runtime": {
            "audio_duration_seconds": audio_duration_seconds,
            "chunks": chunk_count,
            "model_inference_elapsed_seconds": (
                inference_elapsed_ns / 1_000_000_000.0
            ),
            "model_inference_realtime_factor": (
                inference_elapsed_ns
                / 1_000_000_000.0
                / audio_duration_seconds
                if audio_duration_seconds
                else None
            ),
            "all_candidate_decoding_elapsed_seconds": (
                total_decoding_ns / 1_000_000_000.0
            ),
            "all_candidate_morphology_elapsed_seconds": (
                total_morphology_ns / 1_000_000_000.0
            ),
            "inference_plus_decoding_plus_morphology_elapsed_seconds": (
                compute_ns / 1_000_000_000.0
            ),
            "inference_plus_decoding_plus_morphology_realtime_factor": (
                compute_ns / 1_000_000_000.0 / audio_duration_seconds
                if audio_duration_seconds
                else None
            ),
            "wall_elapsed_seconds": (
                time.perf_counter_ns() - wall_started_ns
            )
            / 1_000_000_000.0,
        },
        "candidates": candidates,
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
            "Compare causal release thresholds at one calibrated common entry "
            "threshold over one full-track model pass."
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--entry-threshold",
        type=_probability,
        default=DEFAULT_ENTRY_THRESHOLD,
        metavar="P",
    )
    parser.add_argument(
        "--release-thresholds",
        nargs="+",
        type=_probability,
        default=DEFAULT_RELEASE_THRESHOLDS,
        metavar="P",
    )
    parser.add_argument(
        "--fixed-onset-release-threshold",
        type=_probability,
        help=(
            "keep onset re-arming fixed at P while --release-thresholds "
            "sweeps offset re-arming only; omit for the historical common "
            "onset/offset sweep"
        ),
        metavar="P",
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=512)
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
    )
    parser.add_argument(
        "--onset-tolerance-ms",
        type=_milliseconds,
        default=50.0,
    )
    parser.add_argument(
        "--offset-tolerance-ms",
        type=_milliseconds,
        default=50.0,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    result = run_hysteresis_sweep(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
