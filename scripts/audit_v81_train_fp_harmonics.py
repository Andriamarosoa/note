"""Train-only acoustic audit of real V8.1 false-onset predictions.

This diagnostic intentionally performs no training.  It freezes the V8.1 epoch-03
stream model, runs it only on the original train split, extracts actual false
onset predictions, and compares them with within-track controls matched by
polyphony and pre-event RMS.

The audit measures the exact V8.2 1024-sample / 4096-FFT proxy plus diagnostics
requested after review: harmonic-mask coverage, enrichment over coverage,
positive and negative spectral flux, absolute RMS, network onset score, and an
instantaneous pitch-contour version of the harmonic mask.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from causal_note.guitarset_acoustics import PitchContourPoint, RichNote, load_rich_annotations
from causal_note.v8_predictor import V8KerasPredictor
from causal_note.v8_runtime import BoundaryKind, V8BoundaryDecoder
from scripts.evaluate_boundaries import match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import _arrangement
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group


DEFAULT_SEED = 1337
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_THRESHOLD = 0.40
FFT_WINDOW = 1024
FFT_SIZE = 4096
MIN_HZ = 50.0
MAX_HZ = 8000.0
DEFAULT_BOUNDARY_GUARD_MS = 50.0
DEFAULT_CONTROL_STRIDE = 256
DEFAULT_MAX_CONTROL_RMS_DELTA_DB = 3.0
DEFAULT_CONTOUR_MAX_DISTANCE_MS = 50.0


class AuditError(RuntimeError):
    pass


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _false_predictions(references: Sequence[int], predictions: Sequence[int], tolerance_samples: int) -> Tuple[int, ...]:
    pairs = match_boundaries(references, predictions, tolerance_samples)
    matched = Counter(prediction for _, prediction in pairs)
    result: List[int] = []
    for prediction in predictions:
        if matched[prediction] > 0:
            matched[prediction] -= 1
        else:
            result.append(prediction)
    return tuple(result)


def _nearest_distance(sample: int, values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    index = bisect.bisect_left(values, sample)
    candidates = []
    if index < len(values):
        candidates.append(abs(values[index] - sample))
    if index:
        candidates.append(abs(values[index - 1] - sample))
    return min(candidates) if candidates else None


def _far_from(sample: int, values: Sequence[int], margin: int) -> bool:
    distance = _nearest_distance(sample, values)
    return distance is None or distance > margin


def _active_notes(notes: Sequence[RichNote], sample: int) -> Tuple[RichNote, ...]:
    return tuple(note for note in notes if note.onset_sample <= sample < note.offset_sample)


def _harmonic_mask(frequencies_hz: Iterable[float], valid: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(valid), dtype=bool)
    resolution = SAMPLE_RATE / float(FFT_SIZE)
    for raw_frequency in frequencies_hz:
        f0 = float(raw_frequency)
        if not math.isfinite(f0) or f0 <= 0.0:
            continue
        harmonic = 1
        while harmonic * f0 <= MAX_HZ:
            frequency = harmonic * f0
            bandwidth_hz = max(12.0, frequency * 0.008)
            low = max(0, int(math.floor((frequency - bandwidth_hz) / resolution)))
            high = min(len(mask), int(math.ceil((frequency + bandwidth_hz) / resolution)) + 1)
            if high > low:
                mask[low:high] = True
            harmonic += 1
    return mask & valid


def _contour_indices(contours_by_slot) -> Tuple[Tuple[Tuple[int, ...], Tuple[PitchContourPoint, ...]], ...]:
    result = []
    for points in contours_by_slot:
        frozen = tuple(points)
        result.append((tuple(point.sample for point in frozen), frozen))
    return tuple(result)


def _nearest_voiced_contour_frequency(
    contour_index,
    slot: int,
    sample: int,
    max_distance_samples: int,
) -> Optional[float]:
    samples, points = contour_index[slot]
    if not samples:
        return None
    index = bisect.bisect_left(samples, sample)
    candidates = []
    for candidate_index in range(max(0, index - 4), min(len(points), index + 5)):
        point = points[candidate_index]
        if point.voiced and point.frequency_hz > 0.0:
            distance = abs(point.sample - sample)
            if distance <= max_distance_samples:
                candidates.append((distance, point.sample, point.frequency_hz))
    if not candidates:
        return None
    return float(min(candidates)[2])


def _pcm_window(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    source_start = max(start, 0)
    source_end = min(start + length, len(samples))
    if source_end > source_start:
        destination = source_start - start
        result[destination : destination + source_end - source_start] = samples[source_start:source_end]
    return result


def _dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(float(rms), 1e-12))


def _mask_metrics(flux: np.ndarray, mask: np.ndarray, valid_count: int) -> Tuple[float, float, Optional[float]]:
    coverage = float(mask.sum() / valid_count) if valid_count else 0.0
    total = float(flux.sum())
    fraction = float(flux[mask].sum() / total) if total > 1e-20 else 0.0
    enrichment = fraction / coverage if coverage > 0.0 else None
    return coverage, fraction, enrichment


def _spectral_features(
    normalized_samples: np.ndarray,
    sample: int,
    active: Sequence[RichNote],
    contour_index,
    contour_max_distance_samples: int,
) -> Dict[str, object]:
    pre_values = _pcm_window(normalized_samples, sample - FFT_WINDOW, FFT_WINDOW)
    post_values = _pcm_window(normalized_samples, sample, FFT_WINDOW)
    taper = np.hanning(FFT_WINDOW)
    pre = np.abs(np.fft.rfft(pre_values * taper, n=FFT_SIZE)) ** 2
    post = np.abs(np.fft.rfft(post_values * taper, n=FFT_SIZE)) ** 2
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    valid = (frequencies >= MIN_HZ) & (frequencies <= MAX_HZ)
    valid_count = int(valid.sum())

    positive = np.maximum(post - pre, 0.0)
    negative = np.maximum(pre - post, 0.0)
    positive[~valid] = 0.0
    negative[~valid] = 0.0
    pre_energy = float(pre[valid].sum()) + 1e-20

    fixed_frequencies = tuple(note.frequency_hz for note in active)
    fixed_mask = _harmonic_mask(fixed_frequencies, valid)
    fixed_coverage, fixed_positive_fraction, fixed_positive_enrichment = _mask_metrics(positive, fixed_mask, valid_count)
    _, fixed_negative_fraction, fixed_negative_enrichment = _mask_metrics(negative, fixed_mask, valid_count)

    contour_frequencies = []
    contour_deviations = []
    for note in active:
        contour_frequency = _nearest_voiced_contour_frequency(
            contour_index,
            note.slot,
            sample,
            contour_max_distance_samples,
        )
        if contour_frequency is None:
            continue
        contour_frequencies.append(contour_frequency)
        if note.frequency_hz > 0.0:
            contour_deviations.append(abs(1200.0 * math.log2(contour_frequency / note.frequency_hz)))
    contour_mask = _harmonic_mask(contour_frequencies, valid)
    contour_coverage, contour_positive_fraction, contour_positive_enrichment = _mask_metrics(positive, contour_mask, valid_count)

    pre_rms = float(np.sqrt(np.mean(pre_values * pre_values)) + 1e-12)
    post_rms = float(np.sqrt(np.mean(post_values * post_values)) + 1e-12)
    return {
        "active_note_count": len(active),
        "active_midi": [float(note.midi) for note in active],
        "pre_rms_dbfs": _dbfs(pre_rms),
        "post_rms_dbfs": _dbfs(post_rms),
        "rms_delta_db": 20.0 * math.log10(post_rms / pre_rms),
        "positive_flux_over_pre_energy": float(positive.sum()) / pre_energy,
        "negative_flux_over_pre_energy": float(negative.sum()) / pre_energy,
        "fixed_mask_coverage": fixed_coverage,
        "fixed_positive_flux_fraction": fixed_positive_fraction,
        "fixed_positive_flux_enrichment": fixed_positive_enrichment,
        "fixed_negative_flux_fraction": fixed_negative_fraction,
        "fixed_negative_flux_enrichment": fixed_negative_enrichment,
        "contour_frequency_count": len(contour_frequencies),
        "contour_mean_abs_cents_from_midi": None if not contour_deviations else float(np.mean(contour_deviations)),
        "contour_mask_coverage": contour_coverage,
        "contour_positive_flux_fraction": contour_positive_fraction,
        "contour_positive_flux_enrichment": contour_positive_enrichment,
    }


def _rms_prefix(normalized_samples: np.ndarray) -> np.ndarray:
    squared = normalized_samples * normalized_samples
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(squared, dtype=np.float64)))


def _pre_rms_dbfs(prefix: np.ndarray, sample: int) -> float:
    start = sample - FFT_WINDOW
    energy = float(prefix[sample] - prefix[start]) / FFT_WINDOW
    return _dbfs(math.sqrt(max(energy, 0.0)) + 1e-12)


def _control_candidates(
    normalized_samples: np.ndarray,
    notes: Sequence[RichNote],
    boundaries: Sequence[int],
    predicted_onsets: Sequence[int],
    *,
    boundary_guard: int,
    stride: int,
) -> List[dict]:
    prefix = _rms_prefix(normalized_samples)
    candidates = []
    high = len(normalized_samples) - FFT_WINDOW
    for sample in range(FFT_WINDOW, high, stride):
        if not _far_from(sample, boundaries, boundary_guard):
            continue
        if not _far_from(sample, predicted_onsets, boundary_guard):
            continue
        active = _active_notes(notes, sample)
        candidates.append(
            {
                "sample": sample,
                "active_note_count": len(active),
                "pre_rms_dbfs": _pre_rms_dbfs(prefix, sample),
            }
        )
    return candidates


def _match_controls(fp_records: Sequence[dict], candidates: Sequence[dict], max_rms_delta_db: float) -> Tuple[List[Tuple[dict, dict]], List[dict]]:
    by_polyphony: Dict[int, List[dict]] = defaultdict(list)
    for candidate in candidates:
        by_polyphony[int(candidate["active_note_count"])].append(candidate)
    used = set()
    pairs = []
    unmatched = []
    for fp in sorted(fp_records, key=lambda row: (row["sample"], row["pre_rms_dbfs"])):
        pool = by_polyphony.get(int(fp["active_note_count"]), ())
        best = None
        for candidate in pool:
            if candidate["sample"] in used:
                continue
            delta = abs(float(candidate["pre_rms_dbfs"]) - float(fp["pre_rms_dbfs"]))
            if delta > max_rms_delta_db:
                continue
            key = (delta, abs(int(candidate["sample"]) - int(fp["sample"])), int(candidate["sample"]))
            if best is None or key < best[0]:
                best = (key, candidate)
        if best is None:
            unmatched.append(fp)
            continue
        used.add(best[1]["sample"])
        pairs.append((fp, best[1]))
    return pairs, unmatched


def _summarize(records: Sequence[dict]) -> Dict[str, object]:
    numeric = (
        "model_onset_score",
        "pre_rms_dbfs",
        "post_rms_dbfs",
        "rms_delta_db",
        "positive_flux_over_pre_energy",
        "negative_flux_over_pre_energy",
        "fixed_mask_coverage",
        "fixed_positive_flux_fraction",
        "fixed_positive_flux_enrichment",
        "fixed_negative_flux_fraction",
        "fixed_negative_flux_enrichment",
        "contour_mean_abs_cents_from_midi",
        "contour_mask_coverage",
        "contour_positive_flux_fraction",
        "contour_positive_flux_enrichment",
    )
    summary: Dict[str, object] = {
        "positions": len(records),
        "raw_multiplicity": sum(int(record.get("multiplicity", 1)) for record in records),
        "tracks": len({record["member"] for record in records}),
        "with_active_notes": sum(int(record["active_note_count"]) > 0 for record in records),
        "polyphony": dict(sorted(Counter(int(record["active_note_count"]) for record in records).items())),
    }
    for key in numeric:
        values = [float(record[key]) for record in records if record.get(key) is not None and math.isfinite(float(record[key]))]
        summary[key] = _quantiles(values)
    summary["v82_raw_proxy_pass"] = sum(
        float(record["positive_flux_over_pre_energy"]) >= 0.50
        and float(record["fixed_positive_flux_fraction"]) >= 0.70
        for record in records
    )
    summary["fixed_enrichment_ge_1_25"] = sum(
        record.get("fixed_positive_flux_enrichment") is not None and float(record["fixed_positive_flux_enrichment"]) >= 1.25
        for record in records
    )
    summary["fixed_enrichment_ge_1_50"] = sum(
        record.get("fixed_positive_flux_enrichment") is not None and float(record["fixed_positive_flux_enrichment"]) >= 1.50
        for record in records
    )
    summary["fixed_enrichment_ge_2"] = sum(
        record.get("fixed_positive_flux_enrichment") is not None and float(record["fixed_positive_flux_enrichment"]) >= 2.0
        for record in records
    )
    summary["score_ge_0_50"] = sum(float(record.get("model_onset_score", 0.0)) >= 0.50 for record in records)
    summary["score_ge_0_70"] = sum(float(record.get("model_onset_score", 0.0)) >= 0.70 for record in records)
    summary["score_ge_0_90"] = sum(float(record.get("model_onset_score", 0.0)) >= 0.90 for record in records)
    return summary


def _proxy_distribution(records: Sequence[dict]) -> Dict[str, object]:
    passing = [
        record for record in records
        if float(record["positive_flux_over_pre_energy"]) >= 0.50
        and float(record["fixed_positive_flux_fraction"]) >= 0.70
    ]
    track_counts = Counter(record["member"] for record in passing)
    arrangement_counts = Counter(record["arrangement"] for record in passing)
    return {
        "passing_unique_positions": len(passing),
        "passing_tracks": len(track_counts),
        "arrangement": dict(sorted(arrangement_counts.items())),
        "max_positions_from_one_track": max(track_counts.values()) if track_counts else 0,
        "top_tracks": track_counts.most_common(10),
        "enrichment_quantiles": _quantiles([
            float(record["fixed_positive_flux_enrichment"])
            for record in passing
            if record.get("fixed_positive_flux_enrichment") is not None
        ]),
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit real frozen V8.1 train false positives and matched controls.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--receptive-field", type=int, default=4093)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--boundary-guard-ms", type=float, default=DEFAULT_BOUNDARY_GUARD_MS)
    parser.add_argument("--target-fp", type=int, default=4000)
    parser.add_argument("--min-tracks", type=int, default=24)
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument("--control-stride", type=int, default=DEFAULT_CONTROL_STRIDE)
    parser.add_argument("--max-control-rms-delta-db", type=float, default=DEFAULT_MAX_CONTROL_RMS_DELTA_DB)
    parser.add_argument("--contour-max-distance-ms", type=float, default=DEFAULT_CONTOUR_MAX_DISTANCE_MS)
    return parser


def run(args) -> Dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if args.min_tracks <= 0 or args.max_tracks < args.min_tracks or args.target_fp <= 0:
        raise ValueError("invalid track/FP limits")
    if args.control_stride <= 0 or args.max_control_rms_delta_db < 0.0:
        raise ValueError("invalid control matching configuration")

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    train_tracks, validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        seed=DEFAULT_SEED,
    )
    train_tracks = list(train_tracks)
    random.Random(DEFAULT_SEED + 8201).shuffle(train_tracks)

    tolerance_samples = milliseconds_to_samples(args.tolerance_ms)
    boundary_guard_samples = milliseconds_to_samples(args.boundary_guard_ms)
    contour_max_distance_samples = milliseconds_to_samples(args.contour_max_distance_ms)
    predictor = V8KerasPredictor.from_path(str(args.model), receptive_field=args.receptive_field)
    predictor.warm_up(args.chunk_size)

    false_records: List[dict] = []
    control_records: List[dict] = []
    unmatched_controls: List[dict] = []
    track_summaries = []

    for ordinal, track in enumerate(train_tracks[: args.max_tracks], start=1):
        rich = load_rich_annotations(track.annotation_zip, track.annotation_member)
        notes = tuple(sorted(rich.notes, key=lambda note: (note.onset_sample, note.slot, note.offset_sample)))
        contour_index = _contour_indices(rich.contours_by_slot)
        reference_onsets = tuple(sorted(note.onset_sample for note in notes))
        reference_offsets = tuple(sorted(note.offset_sample for note in notes))
        boundaries = tuple(sorted(set(reference_onsets + reference_offsets)))
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        normalized = np.asarray(audio.samples, dtype=np.float64) / 32768.0

        predictor.reset()
        decoder = V8BoundaryDecoder(onset_threshold=args.threshold, offset_threshold=args.threshold)
        onset_scores = np.zeros(audio.frame_count, dtype=np.float32)
        predicted_onsets: List[int] = []
        predicted_score_by_sample: Dict[int, float] = {}
        position = 0
        while position < audio.frame_count:
            end = min(audio.frame_count, position + args.chunk_size)
            values = tuple(float(value) for value in normalized[position:end])
            scores = predictor.predict_chunk(values, start_sample=position)
            onset_scores[position:end] = np.asarray(scores.onset_presence, dtype=np.float32)
            for boundary in decoder.process_chunk(scores):
                if boundary.kind is not BoundaryKind.ONSET:
                    continue
                local = boundary.sample - scores.start_sample
                score = float(scores.onset_presence[local])
                predicted_score_by_sample[boundary.sample] = score
                predicted_onsets.extend([boundary.sample] * boundary.count)
            position = end

        false_onsets = _false_predictions(reference_onsets, tuple(predicted_onsets), tolerance_samples)
        false_positions = Counter(false_onsets)
        arrangement = _arrangement(track.annotation_member)
        per_track_fp = []
        for sample, multiplicity in sorted(false_positions.items()):
            if sample < FFT_WINDOW or sample + FFT_WINDOW >= audio.frame_count:
                continue
            active = _active_notes(notes, sample)
            features = _spectral_features(normalized, sample, active, contour_index, contour_max_distance_samples)
            record = {
                "member": track.annotation_member,
                "player_id": track.player_id,
                "arrangement": arrangement,
                "sample": int(sample),
                "time_seconds": float(sample / SAMPLE_RATE),
                "multiplicity": int(multiplicity),
                "model_onset_score": float(predicted_score_by_sample.get(sample, onset_scores[sample])),
                **features,
            }
            per_track_fp.append(record)
            false_records.append(record)

        candidates = _control_candidates(
            normalized,
            notes,
            boundaries,
            tuple(sorted(set(predicted_onsets))),
            boundary_guard=boundary_guard_samples,
            stride=args.control_stride,
        )
        pairs, unmatched = _match_controls(per_track_fp, candidates, args.max_control_rms_delta_db)
        for fp, candidate in pairs:
            sample = int(candidate["sample"])
            active = _active_notes(notes, sample)
            features = _spectral_features(normalized, sample, active, contour_index, contour_max_distance_samples)
            control_records.append(
                {
                    "member": track.annotation_member,
                    "player_id": track.player_id,
                    "arrangement": arrangement,
                    "sample": sample,
                    "time_seconds": float(sample / SAMPLE_RATE),
                    "multiplicity": 1,
                    "model_onset_score": float(onset_scores[sample]),
                    "matched_fp_sample": int(fp["sample"]),
                    "match_rms_delta_db": abs(float(features["pre_rms_dbfs"]) - float(fp["pre_rms_dbfs"])),
                    **features,
                }
            )
        unmatched_controls.extend(unmatched)

        track_summaries.append(
            {
                "member": track.annotation_member,
                "player_id": track.player_id,
                "arrangement": arrangement,
                "reference_onsets": len(reference_onsets),
                "predicted_onsets": len(predicted_onsets),
                "false_onsets_raw": len(false_onsets),
                "false_onset_unique_positions": len(per_track_fp),
                "matched_controls": len(pairs),
                "unmatched_controls": len(unmatched),
            }
        )
        print(
            f"audited train {ordinal}/{min(len(train_tracks), args.max_tracks)}: {track.annotation_member} "
            f"fp_unique={len(per_track_fp)} cumulative={len(false_records)} controls={len(pairs)}"
        )
        if ordinal >= args.min_tracks and len(false_records) >= args.target_fp:
            break

    by_arrangement_fp = {
        name: _summarize([record for record in false_records if record["arrangement"] == name])
        for name in ("comp", "solo")
    }
    by_arrangement_controls = {
        name: _summarize([record for record in control_records if record["arrangement"] == name])
        for name in ("comp", "solo")
    }
    result = {
        "schema_version": 1,
        "scope": {
            "seed": DEFAULT_SEED,
            "validation_fraction": DEFAULT_VALIDATION_FRACTION,
            "train_tracks_total": len(train_tracks),
            "validation_tracks_total": len(validation_tracks),
            "evaluated_train_tracks": len(track_summaries),
            "player_05_read": False,
            "members": [row["member"] for row in track_summaries],
            "stop_target_unique_fp": args.target_fp,
            "target_reached": len(false_records) >= args.target_fp,
        },
        "configuration": {
            "model": str(Path(args.model).resolve()),
            "threshold": args.threshold,
            "boundary_tolerance_ms": args.tolerance_ms,
            "boundary_guard_ms": args.boundary_guard_ms,
            "fft_window_samples": FFT_WINDOW,
            "fft_window_ms": FFT_WINDOW * 1000.0 / SAMPLE_RATE,
            "fft_size": FFT_SIZE,
            "frequency_range_hz": [MIN_HZ, MAX_HZ],
            "control_stride": args.control_stride,
            "max_control_rms_delta_db": args.max_control_rms_delta_db,
            "contour_max_distance_ms": args.contour_max_distance_ms,
        },
        "interpretation_guard": (
            "All harmonic categories remain diagnostics, not relabels. The fixed-MIDI and instantaneous-contour masks are both reported because harmonic-family overlap can make raw mask fractions misleading."
        ),
        "false_positive_onsets": {
            "global": _summarize(false_records),
            **by_arrangement_fp,
            "v82_proxy_distribution": _proxy_distribution(false_records),
        },
        "matched_controls": {
            "global": _summarize(control_records),
            **by_arrangement_controls,
            "v82_proxy_distribution": _proxy_distribution(control_records),
            "matched_pairs": len(control_records),
            "unmatched_false_positive_positions": len(unmatched_controls),
            "match_rate": len(control_records) / len(false_records) if false_records else None,
            "rms_delta_db": _quantiles([float(record["match_rms_delta_db"]) for record in control_records]),
        },
        "tracks": track_summaries,
        "false_positive_records": false_records,
        "control_records": control_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    result = run(args)
    compact = {
        "scope": result["scope"],
        "false_positive_onsets": result["false_positive_onsets"],
        "matched_controls": result["matched_controls"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
