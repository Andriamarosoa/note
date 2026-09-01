"""One-pass common-threshold sweep for causal boundary models.

The Keras predictor is called exactly once for each audio chunk.  Its immutable
score chunk is then decoded by one independent ``LiveBoundaryScoreDecoder`` per
threshold, preserving the complete rising-edge and open-slot state for every
candidate without repeating model inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
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


DEFAULT_THRESHOLDS = tuple(index / 100.0 for index in range(50, 91, 5))


@dataclass(frozen=True)
class ThresholdDecoding:
    """Events and measured compute time from one streaming model pass."""

    events: Mapping[float, Tuple[BoundaryEvent, ...]]
    inference_elapsed_ns: int
    decoding_elapsed_ns: Mapping[float, int]
    chunks: int


def validate_thresholds(values: Sequence[float]) -> Tuple[float, ...]:
    """Return finite unique common thresholds in their requested order."""

    thresholds: List[float] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise EvaluationError("thresholds must be finite numbers in (0, 1]")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError(
                "thresholds must be finite numbers in (0, 1]"
            ) from exc
        # Let the production decoder remain the authority for probability
        # semantics while reporting a sweep-specific error here.
        try:
            LiveBoundaryScoreDecoder(
                slot_count=1,
                onset_threshold=value,
                offset_threshold=value,
            )
        except ValueError as exc:
            raise EvaluationError(
                "thresholds must be finite numbers in (0, 1]"
            ) from exc
        if value in thresholds:
            raise EvaluationError("thresholds must not contain duplicates")
        thresholds.append(value)
    if not thresholds:
        raise EvaluationError("at least one threshold is required")
    return tuple(thresholds)


def refuse_output_overwrite(output_path: Path) -> Path:
    """Resolve an output path and fail before inference if it already exists."""

    resolved = Path(output_path).resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to replace sweep output: {resolved}")
    return resolved


def decode_threshold_stream(
    predictor,
    chunks: Iterable[Tuple[int, Tuple[float, ...]]],
    thresholds: Sequence[float],
) -> ThresholdDecoding:
    """Infer each chunk once and decode it independently at every threshold.

    The caller owns predictor reset boundaries.  A fresh set of decoders is
    deliberately allocated for this stream because rising-edge, active-slot,
    and opaque-ID state all depend on the selected threshold.
    """

    selected = validate_thresholds(thresholds)
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
        threshold: LiveBoundaryScoreDecoder(
            slot_count=slot_count,
            onset_threshold=threshold,
            offset_threshold=threshold,
        )
        for threshold in selected
    }
    emitted: Dict[float, List[BoundaryEvent]] = {
        threshold: [] for threshold in selected
    }
    decoding_elapsed_ns = {threshold: 0 for threshold in selected}
    inference_elapsed_ns = 0
    chunk_count = 0

    for start_sample, samples in chunks:
        inference_started = time.perf_counter_ns()
        scores = predict_chunk(samples, start_sample=start_sample)
        inference_elapsed_ns += time.perf_counter_ns() - inference_started
        chunk_count += 1

        for threshold in selected:
            decoding_started = time.perf_counter_ns()
            emitted[threshold].extend(
                decoders[threshold].process_chunk(scores)
            )
            decoding_elapsed_ns[threshold] += (
                time.perf_counter_ns() - decoding_started
            )

    return ThresholdDecoding(
        events={
            threshold: tuple(emitted[threshold]) for threshold in selected
        },
        inference_elapsed_ns=inference_elapsed_ns,
        decoding_elapsed_ns=dict(decoding_elapsed_ns),
        chunks=chunk_count,
    )


def _audio_chunks(decoded, chunk_size: int):
    for start in range(0, decoded.frame_count, chunk_size):
        integer_chunk = decoded.samples[start : start + chunk_size]
        yield start, tuple(sample / 32768.0 for sample in integer_chunk)


def _new_threshold_store(thresholds: Sequence[float]):
    return {
        threshold: {
            "evaluations": [],
            "matched_onsets": [],
            "matched_offsets": [],
            "tracks": [],
            "decoding_elapsed_ns": 0,
        }
        for threshold in thresholds
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


def run_threshold_sweep(arguments: argparse.Namespace) -> Dict[str, object]:
    """Run a locked full-track threshold sweep with one model pass."""

    wall_started_ns = time.perf_counter_ns()
    output_path = refuse_output_overwrite(arguments.output)
    thresholds = validate_thresholds(arguments.thresholds)
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

    stores = _new_threshold_store(thresholds)
    audio_samples_by_track: List[int] = []
    arrangements_by_track: List[Optional[str]] = []
    inference_elapsed_ns = 0
    chunk_count = 0

    for track in validation_tracks:
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
        decoded_thresholds = decode_threshold_stream(
            predictor,
            _audio_chunks(decoded, arguments.chunk_size),
            thresholds,
        )
        inference_elapsed_ns += decoded_thresholds.inference_elapsed_ns
        chunk_count += decoded_thresholds.chunks

        reference_onsets = tuple(
            reference.onset_sample for reference in references
        )
        reference_offsets = tuple(
            reference.offset_sample for reference in references
        )
        for threshold in thresholds:
            predictions = decoded_thresholds.events[threshold]
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
            store = stores[threshold]
            store["evaluations"].append(evaluation)
            store["matched_onsets"].append(matched_onsets)
            store["matched_offsets"].append(matched_offsets)
            store["decoding_elapsed_ns"] += (
                decoded_thresholds.decoding_elapsed_ns[threshold]
            )
            store["tracks"].append(
                {
                    "annotation_member": track.annotation_member,
                    "audio_member": track.audio_member,
                    "arrangement": arrangement,
                    "audio_samples": decoded.frame_count,
                    "metrics": asdict(evaluation),
                }
            )

    total_audio_samples = sum(audio_samples_by_track)
    audio_duration_seconds = total_audio_samples / SAMPLE_RATE
    candidates = []
    total_decoding_ns = 0
    for threshold in thresholds:
        store = stores[threshold]
        aggregates = _aggregate_arrangements(
            store["evaluations"],
            audio_samples_by_track,
            store["matched_onsets"],
            store["matched_offsets"],
            arrangements_by_track,
        )
        decoding_ns = int(store["decoding_elapsed_ns"])
        total_decoding_ns += decoding_ns
        global_aggregate = aggregates["global"]
        candidates.append(
            {
                "threshold": threshold,
                "counts": global_aggregate["counts"],
                "metrics": global_aggregate["metrics"],
                "rates_per_hour": global_aggregate["rates_per_hour"],
                "aggregates": aggregates,
                "runtime": {
                    "decoding_elapsed_seconds": decoding_ns / 1_000_000_000.0,
                    "decoding_realtime_factor": (
                        decoding_ns / 1_000_000_000.0 / audio_duration_seconds
                        if audio_duration_seconds
                        else None
                    ),
                },
                "tracks": store["tracks"],
            }
        )

    compute_ns = inference_elapsed_ns + total_decoding_ns
    result: Dict[str, object] = {
        "schema_version": 1,
        "kind": "boundary_threshold_sweep",
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
            "common_thresholds": list(thresholds),
            "onset_tolerance_ms": arguments.onset_tolerance_ms,
            "onset_tolerance_samples": onset_tolerance_samples,
            "offset_tolerance_ms": arguments.offset_tolerance_ms,
            "offset_tolerance_samples": offset_tolerance_samples,
            "model_passes": 1,
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
            "all_threshold_decoding_elapsed_seconds": (
                total_decoding_ns / 1_000_000_000.0
            ),
            "inference_plus_decoding_elapsed_seconds": (
                compute_ns / 1_000_000_000.0
            ),
            "inference_plus_decoding_realtime_factor": (
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
            "Sweep one common onset/offset threshold grid over one causal "
            "full-track model pass."
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
        "--thresholds",
        nargs="+",
        type=_probability,
        default=DEFAULT_THRESHOLDS,
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
    result = run_threshold_sweep(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
