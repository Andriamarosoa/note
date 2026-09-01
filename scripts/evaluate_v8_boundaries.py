"""Evaluate V8 anonymous boundary detection on complete validation tracks.

This evaluator intentionally scores boundary detection independently from FIFO
interval association. The dominant V7 failure is false temporal triggering, so
V8 must first prove that onset/offset precision and false-boundary rates improve
on the same locked validation split and ±50 ms matching tolerance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import (
    ALLOWED_PLAYERS,
    SAMPLE_RATE,
    index_guitarset,
    load_boundary_slots,
)
from causal_note.v8_predictor import V8KerasPredictor
from causal_note.v8_runtime import BoundaryKind, V8BoundaryDecoder
from scripts.evaluate_boundaries import (
    _count_metrics,
    _percentile,
    compare_onset_multiplicity,
    match_boundaries,
    milliseconds_to_samples,
)
from scripts.train_boundaries import (
    decode_pcm16_mono_wav,
    group_stem,
    split_tracks_by_group,
)


DEFAULT_SEED = 1337
DEFAULT_VALIDATION_FRACTION = 0.2


def _arrangement(member: str) -> str:
    stem = Path(member).stem
    if stem.endswith("_comp"):
        return "comp"
    if stem.endswith("_solo"):
        return "solo"
    return "unknown"


def _reference_positions(track, frame_count: int):
    slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
    onsets: List[int] = []
    offsets: List[int] = []
    for notes in slots:
        for note in notes:
            if note.onset_sample < frame_count:
                onsets.append(note.onset_sample)
            if note.offset_sample < frame_count:
                offsets.append(note.offset_sample)
    return tuple(sorted(onsets)), tuple(sorted(offsets))


def _predict_track(
    predictor,
    audio,
    *,
    chunk_size: int,
    onset_threshold: float,
    offset_threshold: float,
    onset_release_threshold: Optional[float],
    offset_release_threshold: Optional[float],
):
    predictor.reset()
    decoder = V8BoundaryDecoder(
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        onset_release_threshold=onset_release_threshold,
        offset_release_threshold=offset_release_threshold,
    )
    predicted_onsets: List[int] = []
    predicted_offsets: List[int] = []
    durations_ms: List[float] = []
    position = 0
    while position < audio.frame_count:
        end = min(audio.frame_count, position + chunk_size)
        integer = audio.samples[position:end]
        values = tuple(sample / 32768.0 for sample in integer)
        started = time.perf_counter()
        scores = predictor.predict_chunk(values, start_sample=position)
        boundaries = decoder.process_chunk(scores)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        for boundary in boundaries:
            target = (
                predicted_onsets
                if boundary.kind is BoundaryKind.ONSET
                else predicted_offsets
            )
            target.extend([boundary.sample] * boundary.count)
        position = end
    return tuple(predicted_onsets), tuple(predicted_offsets), tuple(durations_ms)


def _track_metrics(references, predictions, tolerance_samples: int):
    pairs = match_boundaries(references, predictions, tolerance_samples)
    metrics = _count_metrics(len(references), len(predictions), len(pairs))
    return metrics, pairs


def _aggregate(records, key: str):
    reference = sum(record[key]["reference"] for record in records)
    prediction = sum(record[key]["prediction"] for record in records)
    true_positive = sum(record[key]["true_positive"] for record in records)
    return _count_metrics(reference, prediction, true_positive)


def _aggregate_regime(records, audio_samples: int):
    onset = _aggregate(records, "onset")
    offset = _aggregate(records, "offset")
    duration_hours = audio_samples / SAMPLE_RATE / 3600.0
    return {
        "duration": {
            "audio_samples": audio_samples,
            "audio_seconds": audio_samples / SAMPLE_RATE,
            "audio_hours": duration_hours,
        },
        "onset": asdict(onset),
        "offset": asdict(offset),
        "prediction_reference_ratio": {
            "onset": (
                onset.prediction_count / onset.reference_count
                if onset.reference_count
                else None
            ),
            "offset": (
                offset.prediction_count / offset.reference_count
                if offset.reference_count
                else None
            ),
        },
        "rates_per_hour": {
            "false_onsets": (
                onset.false_positive / duration_hours if duration_hours else None
            ),
            "false_offsets": (
                offset.false_positive / duration_hours if duration_hours else None
            ),
        },
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate V8 anonymous boundaries on locked validation tracks."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "GuitarSet",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "model" / "causal-boundaries-v8.keras",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--players",
        nargs="+",
        choices=sorted(ALLOWED_PLAYERS),
        default=sorted(ALLOWED_PLAYERS),
    )
    parser.add_argument("--receptive-field", type=int, default=4093)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.5)
    parser.add_argument("--onset-release-threshold", type=float)
    parser.add_argument("--offset-release-threshold", type=float)
    parser.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--offset-tolerance-ms", type=float, default=50.0)
    return parser


def run_evaluation(args) -> Dict[str, object]:
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"V8 evaluation is locked to split seed {DEFAULT_SEED}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if args.chunk_size <= 0 or args.receptive_field <= 0:
        raise ValueError("chunk-size and receptive-field must be > 0")
    for name in ("onset_threshold", "offset_threshold"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    selected_players = tuple(dict.fromkeys(args.players))
    if any(player not in ALLOWED_PLAYERS for player in selected_players):
        raise ValueError("player 05 or unknown players are not allowed")

    indexed = tuple(
        track
        for track in index_guitarset(args.dataset_dir)
        if track.player_id in selected_players
    )
    _, validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    predictor = V8KerasPredictor.from_path(
        str(args.model),
        receptive_field=args.receptive_field,
    )
    predictor.warm_up(args.chunk_size)

    onset_tolerance = milliseconds_to_samples(args.onset_tolerance_ms)
    offset_tolerance = milliseconds_to_samples(args.offset_tolerance_ms)
    records = []
    inference_durations: List[float] = []

    for track in validation_tracks:
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        reference_onsets, reference_offsets = _reference_positions(
            track, audio.frame_count
        )
        predicted_onsets, predicted_offsets, durations = _predict_track(
            predictor,
            audio,
            chunk_size=args.chunk_size,
            onset_threshold=args.onset_threshold,
            offset_threshold=args.offset_threshold,
            onset_release_threshold=args.onset_release_threshold,
            offset_release_threshold=args.offset_release_threshold,
        )
        inference_durations.extend(durations)
        onset_metrics, onset_pairs = _track_metrics(
            reference_onsets, predicted_onsets, onset_tolerance
        )
        offset_metrics, offset_pairs = _track_metrics(
            reference_offsets, predicted_offsets, offset_tolerance
        )
        multiplicity = compare_onset_multiplicity(
            reference_onsets,
            predicted_onsets,
            onset_tolerance,
        )
        records.append(
            {
                "member": track.annotation_member,
                "arrangement": _arrangement(track.annotation_member),
                "audio_samples": audio.frame_count,
                "onset": {
                    "reference": onset_metrics.reference_count,
                    "prediction": onset_metrics.prediction_count,
                    "true_positive": onset_metrics.true_positive,
                    "false_positive": onset_metrics.false_positive,
                    "false_negative": onset_metrics.false_negative,
                    "precision": onset_metrics.precision,
                    "recall": onset_metrics.recall,
                    "f1": onset_metrics.f1,
                },
                "offset": {
                    "reference": offset_metrics.reference_count,
                    "prediction": offset_metrics.prediction_count,
                    "true_positive": offset_metrics.true_positive,
                    "false_positive": offset_metrics.false_positive,
                    "false_negative": offset_metrics.false_negative,
                    "precision": offset_metrics.precision,
                    "recall": offset_metrics.recall,
                    "f1": offset_metrics.f1,
                },
                "onset_multiplicity": asdict(multiplicity),
                "matched_onsets": len(onset_pairs),
                "matched_offsets": len(offset_pairs),
            }
        )

    total_samples = sum(record["audio_samples"] for record in records)
    global_result = _aggregate_regime(records, total_samples)
    by_arrangement = {}
    for arrangement in ("comp", "solo"):
        selected = [
            record for record in records if record["arrangement"] == arrangement
        ]
        by_arrangement[arrangement] = _aggregate_regime(
            selected,
            sum(record["audio_samples"] for record in selected),
        )

    audio_seconds = total_samples / SAMPLE_RATE
    inference_seconds = sum(inference_durations) / 1000.0
    result = {
        "schema_version": 1,
        "model": str(Path(args.model).resolve()),
        "split": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "players": list(selected_players),
            "player_05_read": False,
            "validation_tracks": len(validation_tracks),
            "validation_groups": len(
                {group_stem(track) for track in validation_tracks}
            ),
            "validation_members": [
                track.annotation_member for track in validation_tracks
            ],
        },
        "configuration": {
            "receptive_field": args.receptive_field,
            "chunk_size": args.chunk_size,
            "onset_threshold": args.onset_threshold,
            "offset_threshold": args.offset_threshold,
            "onset_release_threshold": args.onset_release_threshold,
            "offset_release_threshold": args.offset_release_threshold,
            "onset_tolerance_ms": args.onset_tolerance_ms,
            "onset_tolerance_samples": onset_tolerance,
            "offset_tolerance_ms": args.offset_tolerance_ms,
            "offset_tolerance_samples": offset_tolerance,
            "association_scored": False,
        },
        "aggregates": {
            "global": global_result,
            **by_arrangement,
        },
        "runtime": {
            "audio_duration_seconds": audio_seconds,
            "inference_elapsed_seconds": inference_seconds,
            "realtime_factor": (
                inference_seconds / audio_seconds if audio_seconds else None
            ),
            "chunks": len(inference_durations),
            "chunk_compute_p50_ms": (
                _percentile(
                    [round(value * 1000) for value in inference_durations], 0.5
                )
                / 1000.0
                if inference_durations
                else None
            ),
            "chunk_compute_p95_ms": (
                _percentile(
                    [round(value * 1000) for value in inference_durations], 0.95
                )
                / 1000.0
                if inference_durations
                else None
            ),
            "chunk_compute_max_ms": (
                max(inference_durations) if inference_durations else None
            ),
        },
        "tracks": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        result["output_path"] = str(args.output.resolve())
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    print(json.dumps(run_evaluation(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
