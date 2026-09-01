"""Audit whether V8.1 false positives and note offsets agree with acoustic evidence.

This is deliberately diagnostic.  It does not silently relabel GuitarSet.
False positives are first defined against the existing +/-50 ms note boundary
metric, then described using independent-ish acoustic features:

* per-string pitch-contour activation / large pitch jumps;
* spectral novelty explained by harmonics of already-active annotated notes;
* strongest residual harmonic series after subtracting those active harmonics;
* harmonic energy remaining after annotated note offsets.

Pitch contours are not a fully independent ground truth because GuitarSet used
note regions to facilitate pitch-track estimation.  The mono pickup-mix spectral
measure is therefore kept alongside contour evidence instead of replacing it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
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
from causal_note.guitarset_acoustics import RichNote, load_rich_annotations
from causal_note.v8_predictor import V8KerasPredictor
from causal_note.v8_runtime import BoundaryKind, V8BoundaryDecoder
from scripts.evaluate_boundaries import match_boundaries, milliseconds_to_samples
from scripts.evaluate_v8_boundaries import _arrangement
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group


DEFAULT_SEED = 1337
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_THRESHOLD = 0.40
FFT_WINDOW = 2048
FFT_SIZE = 8192
MAX_SPECTRAL_HZ = 8000.0
CONTOUR_WINDOW_MS = 50.0
PITCH_JUMP_CENTS = 100.0


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _false_predictions(
    references: Sequence[int], predictions: Sequence[int], tolerance_samples: int
) -> Tuple[int, ...]:
    pairs = match_boundaries(references, predictions, tolerance_samples)
    matched = Counter(prediction for _, prediction in pairs)
    false_values: List[int] = []
    for prediction in predictions:
        if matched[prediction] > 0:
            matched[prediction] -= 1
        else:
            false_values.append(prediction)
    return tuple(false_values)


def _nearest_distance(sample: int, candidates: Sequence[int]) -> Optional[int]:
    if not candidates:
        return None
    import bisect

    index = bisect.bisect_left(candidates, sample)
    choices = []
    if index < len(candidates):
        choices.append(abs(candidates[index] - sample))
    if index:
        choices.append(abs(candidates[index - 1] - sample))
    return min(choices) if choices else None


def _contour_change_samples(contours_by_slot) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    activations: List[int] = []
    jumps: List[int] = []
    for points in contours_by_slot:
        if not points:
            continue
        steps = [
            current.sample - previous.sample
            for previous, current in zip(points, points[1:])
            if current.sample > previous.sample
        ]
        median_step = float(np.median(steps)) if steps else 0.0
        max_contiguous_gap = max(round(median_step * 3.0), 1) if median_step else 1
        previous = None
        for point in points:
            if point.voiced:
                if (
                    previous is None
                    or not previous.voiced
                    or point.sample - previous.sample > max_contiguous_gap
                ):
                    activations.append(point.sample)
                elif previous.frequency_hz > 0.0:
                    cents = abs(1200.0 * math.log2(point.frequency_hz / previous.frequency_hz))
                    if cents >= PITCH_JUMP_CENTS:
                        jumps.append(point.sample)
            previous = point
    return tuple(sorted(activations)), tuple(sorted(jumps))


def _note_positions(notes_by_slot) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    onsets = sorted(note.onset_sample for slot in notes_by_slot for note in slot)
    offsets = sorted(note.offset_sample for slot in notes_by_slot for note in slot)
    return tuple(onsets), tuple(offsets)


def _active_notes(notes: Sequence[RichNote], sample: int) -> Tuple[RichNote, ...]:
    return tuple(note for note in notes if note.onset_sample <= sample < note.offset_sample)


def _pcm_window(samples, start: int, length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    source_start = max(start, 0)
    source_end = min(start + length, len(samples))
    if source_end > source_start:
        destination_start = source_start - start
        result[destination_start : destination_start + source_end - source_start] = np.asarray(
            samples[source_start:source_end], dtype=np.float64
        ) / 32768.0
    return result


def _power_spectrum(values: np.ndarray) -> np.ndarray:
    window = np.hanning(len(values))
    transformed = np.fft.rfft(values * window, n=FFT_SIZE)
    return np.abs(transformed) ** 2


def _harmonic_bin_ranges(f0: float, max_hz: float = MAX_SPECTRAL_HZ) -> Tuple[Tuple[int, int], ...]:
    if not math.isfinite(f0) or f0 <= 0.0:
        return ()
    resolution = SAMPLE_RATE / FFT_SIZE
    ranges = []
    harmonic = 1
    while harmonic * f0 <= max_hz:
        frequency = harmonic * f0
        bandwidth_hz = max(12.0, frequency * 0.008)
        low = max(0, int(math.floor((frequency - bandwidth_hz) / resolution)))
        high = min(FFT_SIZE // 2, int(math.ceil((frequency + bandwidth_hz) / resolution)))
        if high >= low:
            ranges.append((low, high + 1))
        harmonic += 1
    return tuple(ranges)


def _mask_for_frequencies(frequencies: Iterable[float], max_bin: int) -> np.ndarray:
    mask = np.zeros(max_bin, dtype=bool)
    for frequency in frequencies:
        for low, high in _harmonic_bin_ranges(float(frequency)):
            mask[low:min(high, max_bin)] = True
    return mask


def _candidate_residual_series(residual_flux: np.ndarray) -> Tuple[Optional[float], float]:
    resolution = SAMPLE_RATE / FFT_SIZE
    total = float(residual_flux.sum())
    if total <= 1e-20:
        return None, 0.0
    best_midi = None
    best_score = 0.0
    for midi in range(40, 89):
        f0 = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
        score = 0.0
        weight = 0.0
        harmonic = 1
        while harmonic * f0 <= MAX_SPECTRAL_HZ:
            index = int(round(harmonic * f0 / resolution))
            low = max(index - 2, 0)
            high = min(index + 3, len(residual_flux))
            harmonic_weight = 1.0 / math.sqrt(harmonic)
            score += float(residual_flux[low:high].sum()) * harmonic_weight
            weight += harmonic_weight
            harmonic += 1
        if weight:
            score /= weight
        if score > best_score:
            best_score = score
            best_midi = float(midi)
    return best_midi, best_score / total


def _spectral_onset_features(samples, sample: int, active_notes: Sequence[RichNote]) -> Dict[str, object]:
    pre = _pcm_window(samples, sample - FFT_WINDOW, FFT_WINDOW)
    post = _pcm_window(samples, sample, FFT_WINDOW)
    pre_power = _power_spectrum(pre)
    post_power = _power_spectrum(post)
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    valid = (frequencies >= 50.0) & (frequencies <= MAX_SPECTRAL_HZ)
    positive_flux = np.maximum(post_power - pre_power, 0.0)
    positive_flux[~valid] = 0.0
    total_flux = float(positive_flux.sum())
    pre_energy = float(pre_power[valid].sum()) + 1e-20
    active_frequencies = tuple(note.frequency_hz for note in active_notes)
    harmonic_mask = _mask_for_frequencies(active_frequencies, len(positive_flux)) & valid
    explained_flux = float(positive_flux[harmonic_mask].sum()) if active_frequencies else 0.0
    residual_flux = positive_flux.copy()
    residual_flux[harmonic_mask] = 0.0
    candidate_midi, candidate_fraction = _candidate_residual_series(residual_flux)
    pre_rms = float(np.sqrt(np.mean(pre * pre)) + 1e-12)
    post_rms = float(np.sqrt(np.mean(post * post)) + 1e-12)
    return {
        "active_note_count": len(active_notes),
        "active_midi": [float(note.midi) for note in active_notes],
        "positive_flux_over_pre_energy": total_flux / pre_energy,
        "active_harmonic_flux_fraction": explained_flux / total_flux if total_flux > 0.0 else None,
        "residual_harmonic_candidate_midi": candidate_midi,
        "residual_harmonic_series_fraction": candidate_fraction,
        "rms_delta_db": 20.0 * math.log10(post_rms / pre_rms),
    }


def _offset_features(samples, note: RichNote, all_notes: Sequence[RichNote]) -> Optional[Dict[str, object]]:
    sample = note.offset_sample
    if sample < FFT_WINDOW or sample + FFT_WINDOW > len(samples):
        return None
    pre = _power_spectrum(_pcm_window(samples, sample - FFT_WINDOW, FFT_WINDOW))
    post = _power_spectrum(_pcm_window(samples, sample, FFT_WINDOW))
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    valid = (frequencies >= 50.0) & (frequencies <= MAX_SPECTRAL_HZ)
    mask = _mask_for_frequencies((note.frequency_hz,), len(pre)) & valid
    pre_energy = float(pre[mask].sum()) + 1e-20
    post_energy = float(post[mask].sum())
    probe_sample = sample + FFT_WINDOW // 2
    other_active = tuple(
        candidate
        for candidate in all_notes
        if candidate is not note and candidate.onset_sample <= probe_sample < candidate.offset_sample
    )
    return {
        "member_note_slot": note.slot,
        "midi": float(note.midi),
        "harmonic_post_pre_ratio": post_energy / pre_energy,
        "isolated_post_window": not other_active,
        "other_active_note_count": len(other_active),
    }


def _threshold_counts(records: Sequence[dict]) -> Dict[str, int]:
    harmonic_fractions = [
        record["spectral"]["active_harmonic_flux_fraction"]
        for record in records
        if record["spectral"]["active_harmonic_flux_fraction"] is not None
    ]
    residual = [record["spectral"]["residual_harmonic_series_fraction"] for record in records]
    return {
        "active_harmonic_flux_ge_0_50": sum(value >= 0.50 for value in harmonic_fractions),
        "active_harmonic_flux_ge_0_70": sum(value >= 0.70 for value in harmonic_fractions),
        "residual_series_ge_0_10": sum(value >= 0.10 for value in residual),
        "residual_series_ge_0_20": sum(value >= 0.20 for value in residual),
        "residual_series_ge_0_30": sum(value >= 0.30 for value in residual),
    }


def _summarize_fp_records(records: Sequence[dict]) -> Dict[str, object]:
    if not records:
        return {
            "unique_positions": 0,
            "raw_false_positive_count": 0,
            "near_unannotated_contour_activation_positions": 0,
            "near_unannotated_pitch_jump_positions": 0,
            "threshold_counts": _threshold_counts(records),
        }
    raw_count = sum(record["multiplicity"] for record in records)
    harmonic = [
        record["spectral"]["active_harmonic_flux_fraction"]
        for record in records
        if record["spectral"]["active_harmonic_flux_fraction"] is not None
    ]
    residual = [record["spectral"]["residual_harmonic_series_fraction"] for record in records]
    flux = [record["spectral"]["positive_flux_over_pre_energy"] for record in records]
    rms = [record["spectral"]["rms_delta_db"] for record in records]
    return {
        "unique_positions": len(records),
        "raw_false_positive_count": raw_count,
        "multiplicity_excess": raw_count - len(records),
        "with_active_annotated_note": sum(record["spectral"]["active_note_count"] > 0 for record in records),
        "near_unannotated_contour_activation_positions": sum(record["near_unannotated_contour_activation"] for record in records),
        "near_unannotated_pitch_jump_positions": sum(record["near_unannotated_pitch_jump"] for record in records),
        "active_harmonic_flux_fraction": _quantiles(harmonic),
        "residual_harmonic_series_fraction": _quantiles(residual),
        "positive_flux_over_pre_energy": _quantiles(flux),
        "rms_delta_db": _quantiles(rms),
        "threshold_counts": _threshold_counts(records),
    }


def _summarize_offsets(records: Sequence[dict]) -> Dict[str, object]:
    ratios = [record["harmonic_post_pre_ratio"] for record in records]
    isolated = [record for record in records if record["isolated_post_window"]]
    isolated_ratios = [record["harmonic_post_pre_ratio"] for record in isolated]
    return {
        "evaluated_offsets": len(records),
        "ratio_quantiles": _quantiles(ratios),
        "post_pre_ge_0_10": sum(value >= 0.10 for value in ratios),
        "post_pre_ge_0_25": sum(value >= 0.25 for value in ratios),
        "post_pre_ge_0_50": sum(value >= 0.50 for value in ratios),
        "post_pre_ge_0_80": sum(value >= 0.80 for value in ratios),
        "isolated_offsets": len(isolated),
        "isolated_ratio_quantiles": _quantiles(isolated_ratios),
        "isolated_post_pre_ge_0_25": sum(value >= 0.25 for value in isolated_ratios),
        "isolated_post_pre_ge_0_50": sum(value >= 0.50 for value in isolated_ratios),
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit V8.1 false positives against GuitarSet acoustic evidence.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--receptive-field", type=int, default=4093)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--limit-tracks", type=int, default=12)
    return parser


def run_audit(args) -> Dict[str, object]:
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    _, locked_validation = split_tracks_by_group(
        indexed,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        seed=DEFAULT_SEED,
    )
    validation_tracks = locked_validation[: args.limit_tracks]
    tolerance_samples = milliseconds_to_samples(args.tolerance_ms)
    contour_window_samples = milliseconds_to_samples(CONTOUR_WINDOW_MS)

    predictor = V8KerasPredictor.from_path(str(args.model), receptive_field=args.receptive_field)
    predictor.warm_up(args.chunk_size)

    fp_records: List[dict] = []
    offset_records: List[dict] = []
    contour_gap_counts = Counter()
    track_summaries = []

    for track_number, track in enumerate(validation_tracks, start=1):
        rich = load_rich_annotations(track.annotation_zip, track.annotation_member)
        all_notes = tuple(sorted((note for slot in rich.notes_by_slot for note in slot), key=lambda note: (note.onset_sample, note.slot, note.offset_sample)))
        reference_onsets, reference_offsets = _note_positions(rich.notes_by_slot)
        activations, jumps = _contour_change_samples(rich.contours_by_slot)
        unannotated_activations = tuple(
            sample for sample in activations
            if (_nearest_distance(sample, reference_onsets) or 10**18) > tolerance_samples
        )
        unannotated_jumps = tuple(
            sample for sample in jumps
            if (_nearest_distance(sample, reference_onsets) or 10**18) > tolerance_samples
        )
        contour_gap_counts["activations"] += len(activations)
        contour_gap_counts["jumps_ge_100_cents"] += len(jumps)
        contour_gap_counts["unannotated_activations"] += len(unannotated_activations)
        contour_gap_counts["unannotated_jumps_ge_100_cents"] += len(unannotated_jumps)

        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        predictor.reset()
        decoder = V8BoundaryDecoder(onset_threshold=args.threshold, offset_threshold=args.threshold)
        predicted_onsets: List[int] = []
        predicted_offsets: List[int] = []
        position = 0
        while position < audio.frame_count:
            end = min(audio.frame_count, position + args.chunk_size)
            values = tuple(sample / 32768.0 for sample in audio.samples[position:end])
            scores = predictor.predict_chunk(values, start_sample=position)
            for boundary in decoder.process_chunk(scores):
                target = predicted_onsets if boundary.kind is BoundaryKind.ONSET else predicted_offsets
                target.extend([boundary.sample] * boundary.count)
            position = end

        false_onsets = _false_predictions(reference_onsets, tuple(predicted_onsets), tolerance_samples)
        false_positions = Counter(false_onsets)
        arrangement = _arrangement(track.annotation_member)
        for sample, multiplicity in sorted(false_positions.items()):
            active = _active_notes(all_notes, sample)
            spectral = _spectral_onset_features(audio.samples, sample, active)
            activation_distance = _nearest_distance(sample, unannotated_activations)
            jump_distance = _nearest_distance(sample, unannotated_jumps)
            fp_records.append(
                {
                    "member": track.annotation_member,
                    "arrangement": arrangement,
                    "sample": sample,
                    "time_seconds": sample / SAMPLE_RATE,
                    "multiplicity": multiplicity,
                    "near_unannotated_contour_activation": activation_distance is not None and activation_distance <= contour_window_samples,
                    "nearest_unannotated_contour_activation_ms": None if activation_distance is None else activation_distance * 1000.0 / SAMPLE_RATE,
                    "near_unannotated_pitch_jump": jump_distance is not None and jump_distance <= contour_window_samples,
                    "nearest_unannotated_pitch_jump_ms": None if jump_distance is None else jump_distance * 1000.0 / SAMPLE_RATE,
                    "spectral": spectral,
                }
            )

        for note in all_notes:
            features = _offset_features(audio.samples, note, all_notes)
            if features is not None:
                features.update(
                    {
                        "member": track.annotation_member,
                        "arrangement": arrangement,
                        "sample": note.offset_sample,
                    }
                )
                offset_records.append(features)

        track_summaries.append(
            {
                "member": track.annotation_member,
                "arrangement": arrangement,
                "reference_onsets": len(reference_onsets),
                "predicted_onsets": len(predicted_onsets),
                "false_onsets": len(false_onsets),
                "false_onset_unique_positions": len(false_positions),
                "contour_activations": len(activations),
                "unannotated_contour_activations": len(unannotated_activations),
                "pitch_jumps_ge_100_cents": len(jumps),
                "unannotated_pitch_jumps_ge_100_cents": len(unannotated_jumps),
            }
        )
        print(f"audited {track_number}/{len(validation_tracks)}: {track.annotation_member}")

    result = {
        "schema_version": 1,
        "scope": {
            "seed": DEFAULT_SEED,
            "validation_fraction": DEFAULT_VALIDATION_FRACTION,
            "locked_validation_tracks": len(locked_validation),
            "evaluated_validation_tracks": len(validation_tracks),
            "player_05_read": False,
            "members": [track.annotation_member for track in validation_tracks],
        },
        "configuration": {
            "model": str(Path(args.model).resolve()),
            "threshold": args.threshold,
            "boundary_tolerance_ms": args.tolerance_ms,
            "contour_proximity_ms": CONTOUR_WINDOW_MS,
            "pitch_jump_cents": PITCH_JUMP_CENTS,
            "fft_window_samples": FFT_WINDOW,
            "fft_window_ms": FFT_WINDOW * 1000.0 / SAMPLE_RATE,
            "fft_size": FFT_SIZE,
            "max_spectral_hz": MAX_SPECTRAL_HZ,
        },
        "important_limitation": (
            "GuitarSet pitch contours are not fully independent labels because note regions were used to facilitate pitch-track estimation. "
            "Mono pickup-mix spectral evidence is therefore reported separately and all categories remain heuristic diagnostics."
        ),
        "contour_annotation_gap_candidates": dict(contour_gap_counts),
        "false_positive_onsets": {
            "global": _summarize_fp_records(fp_records),
            "comp": _summarize_fp_records([record for record in fp_records if record["arrangement"] == "comp"]),
            "solo": _summarize_fp_records([record for record in fp_records if record["arrangement"] == "solo"]),
        },
        "annotated_offsets": {
            "global": _summarize_offsets(offset_records),
            "comp": _summarize_offsets([record for record in offset_records if record["arrangement"] == "comp"]),
            "solo": _summarize_offsets([record for record in offset_records if record["arrangement"] == "solo"]),
        },
        "tracks": track_summaries,
        "false_positive_onset_records": fp_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    result = run_audit(args)
    compact = {
        "contour_annotation_gap_candidates": result["contour_annotation_gap_candidates"],
        "false_positive_onsets": result["false_positive_onsets"],
        "annotated_offsets": result["annotated_offsets"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
