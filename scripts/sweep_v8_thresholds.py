"""Single-pass threshold sweep for V8 on the locked validation split.

The neural predictor runs once per audio chunk. Multiple boundary decoders then
consume the same scores, so threshold calibration does not multiply inference
cost. A deterministic validation-track limit is available for pilot runs; omit
it for the full 60-track locked validation evaluation.
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

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from causal_note.v8_predictor import V8KerasPredictor
from causal_note.v8_runtime import BoundaryKind, V8BoundaryDecoder
from scripts.evaluate_boundaries import _percentile, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import (
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    _aggregate_regime,
    _arrangement,
    _reference_positions,
    _track_metrics,
)
from scripts.train_boundaries import decode_pcm16_mono_wav, group_stem, split_tracks_by_group


DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def _thresholds(values: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise ValueError("at least one threshold is required")
    if any(not 0.0 < value <= 1.0 for value in converted):
        raise ValueError("thresholds must be in (0, 1]")
    if len(set(converted)) != len(converted):
        raise ValueError("thresholds must be unique")
    return tuple(sorted(converted))


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep V8 presence thresholds with one neural inference pass."
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
    parser.add_argument("--receptive-field", type=int, default=4093)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--limit-tracks",
        type=int,
        help="deterministic prefix of the locked validation split; omit for full evaluation",
    )
    return parser


def run_sweep(args) -> Dict[str, object]:
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"V8 sweep is locked to split seed {DEFAULT_SEED}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if args.chunk_size <= 0 or args.receptive_field <= 0:
        raise ValueError("chunk-size and receptive-field must be > 0")
    if args.limit_tracks is not None and args.limit_tracks <= 0:
        raise ValueError("limit-tracks must be > 0")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    thresholds = _thresholds(args.thresholds)
    indexed = tuple(
        track for track in index_guitarset(args.dataset_dir)
        if track.player_id in ALLOWED_PLAYERS
    )
    _, locked_validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    validation_tracks = locked_validation_tracks
    if args.limit_tracks is not None:
        validation_tracks = validation_tracks[: args.limit_tracks]

    predictor = V8KerasPredictor.from_path(
        str(args.model), receptive_field=args.receptive_field
    )
    predictor.warm_up(args.chunk_size)
    tolerance_samples = milliseconds_to_samples(args.tolerance_ms)

    records_by_threshold: Dict[float, List[dict]] = {
        threshold: [] for threshold in thresholds
    }
    inference_durations: List[float] = []

    for track_number, track in enumerate(validation_tracks, start=1):
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        reference_onsets, reference_offsets = _reference_positions(track, audio.frame_count)
        predictor.reset()
        decoders = {
            threshold: V8BoundaryDecoder(
                onset_threshold=threshold,
                offset_threshold=threshold,
            )
            for threshold in thresholds
        }
        predictions = {
            threshold: {"onset": [], "offset": []}
            for threshold in thresholds
        }

        position = 0
        while position < audio.frame_count:
            end = min(audio.frame_count, position + args.chunk_size)
            values = tuple(sample / 32768.0 for sample in audio.samples[position:end])
            started = time.perf_counter()
            scores = predictor.predict_chunk(values, start_sample=position)
            inference_durations.append((time.perf_counter() - started) * 1000.0)
            for threshold, decoder in decoders.items():
                for boundary in decoder.process_chunk(scores):
                    key = "onset" if boundary.kind is BoundaryKind.ONSET else "offset"
                    predictions[threshold][key].extend([boundary.sample] * boundary.count)
            position = end

        for threshold in thresholds:
            onset_metrics, _ = _track_metrics(
                reference_onsets,
                tuple(predictions[threshold]["onset"]),
                tolerance_samples,
            )
            offset_metrics, _ = _track_metrics(
                reference_offsets,
                tuple(predictions[threshold]["offset"]),
                tolerance_samples,
            )
            records_by_threshold[threshold].append(
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
                }
            )
        print(
            f"evaluated {track_number}/{len(validation_tracks)}: "
            f"{track.annotation_member}"
        )

    total_samples = sum(record["audio_samples"] for record in records_by_threshold[thresholds[0]])
    sweep = {}
    for threshold in thresholds:
        records = records_by_threshold[threshold]
        aggregates = {
            "global": _aggregate_regime(records, total_samples),
        }
        for arrangement in ("comp", "solo"):
            selected = [record for record in records if record["arrangement"] == arrangement]
            aggregates[arrangement] = _aggregate_regime(
                selected,
                sum(record["audio_samples"] for record in selected),
            )
        sweep[f"{threshold:.6f}"] = {"aggregates": aggregates}

    audio_seconds = total_samples / SAMPLE_RATE
    inference_seconds = sum(inference_durations) / 1000.0
    result = {
        "schema_version": 1,
        "model": str(Path(args.model).resolve()),
        "split": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "locked_validation_tracks": len(locked_validation_tracks),
            "evaluated_validation_tracks": len(validation_tracks),
            "validation_groups": len({group_stem(track) for track in validation_tracks}),
            "pilot_subset": args.limit_tracks is not None,
            "player_05_read": False,
            "validation_members": [track.annotation_member for track in validation_tracks],
        },
        "configuration": {
            "receptive_field": args.receptive_field,
            "chunk_size": args.chunk_size,
            "tolerance_ms": args.tolerance_ms,
            "tolerance_samples": tolerance_samples,
            "thresholds": list(thresholds),
        },
        "runtime": {
            "audio_duration_seconds": audio_seconds,
            "inference_elapsed_seconds": inference_seconds,
            "realtime_factor": inference_seconds / audio_seconds if audio_seconds else None,
            "chunks": len(inference_durations),
            "chunk_compute_p50_ms": (
                _percentile([round(value * 1000) for value in inference_durations], 0.5) / 1000.0
                if inference_durations else None
            ),
            "chunk_compute_p95_ms": (
                _percentile([round(value * 1000) for value in inference_durations], 0.95) / 1000.0
                if inference_durations else None
            ),
            "chunk_compute_max_ms": max(inference_durations) if inference_durations else None,
        },
        "sweep": sweep,
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
    result = run_sweep(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
