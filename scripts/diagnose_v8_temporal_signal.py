"""Diagnose where V8 boundary evidence appears relative to reference events.

This is a data/model diagnostic, not a threshold tuner. It measures whether the
current causal model already separates true boundaries from nearby/background
samples, and how much post-boundary evidence is needed before scores rise.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset, load_boundary_slots
from causal_note.v8_predictor import V8KerasPredictor
from scripts.train_boundaries import decode_pcm16_mono_wav, split_tracks_by_group


DEFAULT_WINDOWS_MS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)
NEGATIVE_BANDS = (
    ("d1_15", 1, 15),
    ("d16_63", 16, 63),
    ("d64_511", 64, 511),
    ("d512_2204", 512, 2204),
    ("d2205_plus", 2205, None),
)
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
PROBES = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)


def _ms_to_samples(value: float) -> int:
    return int(round(float(value) * SAMPLE_RATE / 1000.0))


def _reference_positions(track, frame_count: int):
    slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
    onsets: List[int] = []
    offsets: List[int] = []
    for notes in slots:
        for note in notes:
            if 0 <= note.onset_sample < frame_count:
                onsets.append(note.onset_sample)
            if 0 <= note.offset_sample < frame_count:
                offsets.append(note.offset_sample)
    return tuple(sorted(onsets)), tuple(sorted(offsets))


def _quantiles(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {f"q{int(q * 100):02d}": None for q in QUANTILES}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(q * 100):02d}": float(np.quantile(array, q))
        for q in QUANTILES
    }


def _summary(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {"count": 0, "mean": None, "quantiles": _quantiles(values), "above": {}}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "quantiles": _quantiles(values),
        "above": {
            f"{probe:.4g}": float(np.mean(array >= probe))
            for probe in PROBES
        },
    }


def _nearest_distance(sorted_positions: Sequence[int], position: int) -> int:
    if not sorted_positions:
        return 10**18
    index = bisect.bisect_left(sorted_positions, position)
    best = 10**18
    if index < len(sorted_positions):
        best = min(best, abs(sorted_positions[index] - position))
    if index:
        best = min(best, abs(sorted_positions[index - 1] - position))
    return int(best)


def _negative_band(distance: int) -> str | None:
    if distance == 0:
        return None
    for name, low, high in NEGATIVE_BANDS:
        if distance >= low and (high is None or distance <= high):
            return name
    return None


def _event_profile(scores: np.ndarray, references: Sequence[int], windows_samples: Sequence[int]):
    by_window = {window: [] for window in windows_samples}
    peak_delays = []
    pre50 = []
    post50 = windows_samples[-1]
    n = int(scores.shape[0])

    for reference in references:
        if not 0 <= reference < n:
            continue
        for window in windows_samples:
            end = min(n, reference + window + 1)
            by_window[window].append(float(np.max(scores[reference:end])))

        end = min(n, reference + post50 + 1)
        local = scores[reference:end]
        if local.size:
            peak_delays.append(int(np.argmax(local)))
        start = max(0, reference - post50)
        if start < reference:
            pre50.append(float(np.max(scores[start:reference])))

    return by_window, peak_delays, pre50


def _sample_negative_scores(
    scores: np.ndarray,
    references: Sequence[int],
    *,
    seed: int,
    count: int,
):
    rng = random.Random(seed)
    result = {name: [] for name, _low, _high in NEGATIVE_BANDS}
    n = int(scores.shape[0])
    attempts = 0
    max_attempts = max(count * 30, 1000)
    while sum(len(values) for values in result.values()) < count and attempts < max_attempts:
        attempts += 1
        position = rng.randrange(n)
        band = _negative_band(_nearest_distance(references, position))
        if band is not None:
            result[band].append(float(scores[position]))
    return result


def _merge_dict_lists(destination, source):
    for key, values in source.items():
        destination.setdefault(key, []).extend(values)


def _score_track(predictor, audio, chunk_size: int):
    onset = np.empty(audio.frame_count, dtype=np.float32)
    offset = np.empty(audio.frame_count, dtype=np.float32)
    predictor.reset()
    position = 0
    while position < audio.frame_count:
        end = min(audio.frame_count, position + chunk_size)
        values = np.asarray(audio.samples[position:end], dtype=np.float32) / 32768.0
        chunk = predictor.predict_chunk(values, start_sample=position)
        onset[position:end] = np.asarray(chunk.onset_presence, dtype=np.float32)
        offset[position:end] = np.asarray(chunk.offset_presence, dtype=np.float32)
        position = end
    return onset, offset


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Diagnose V8 temporal score evidence.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-tracks", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--negative-samples-per-track", type=int, default=12000)
    parser.add_argument("--windows-ms", nargs="+", type=float, default=list(DEFAULT_WINDOWS_MS))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.limit_tracks <= 0 or args.chunk_size <= 0 or args.negative_samples_per_track <= 0:
        raise ValueError("limit-tracks, chunk-size and negative sample count must be > 0")
    windows_ms = tuple(sorted(set(float(value) for value in args.windows_ms)))
    if not windows_ms or windows_ms[0] < 0.0:
        raise ValueError("windows-ms must contain nonnegative values")
    windows_samples = tuple(_ms_to_samples(value) for value in windows_ms)

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    _train, validation = split_tracks_by_group(indexed, validation_fraction=0.2, seed=args.seed)
    selected = validation[: args.limit_tracks]

    predictor = V8KerasPredictor.from_path(str(args.model), receptive_field=4093)
    predictor.warm_up(args.chunk_size)

    aggregate = {
        "onset": {
            "windows": {window: [] for window in windows_samples},
            "peak_delays": [],
            "pre50": [],
            "negative": {name: [] for name, _low, _high in NEGATIVE_BANDS},
        },
        "offset": {
            "windows": {window: [] for window in windows_samples},
            "peak_delays": [],
            "pre50": [],
            "negative": {name: [] for name, _low, _high in NEGATIVE_BANDS},
        },
    }
    members = []

    for track_index, track in enumerate(selected):
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        references = _reference_positions(track, audio.frame_count)
        onset_scores, offset_scores = _score_track(predictor, audio, args.chunk_size)
        members.append(track.annotation_member)

        for kind_index, (kind, scores) in enumerate((("onset", onset_scores), ("offset", offset_scores))):
            event_windows, peak_delays, pre50 = _event_profile(scores, references[kind_index], windows_samples)
            for window, values in event_windows.items():
                aggregate[kind]["windows"][window].extend(values)
            aggregate[kind]["peak_delays"].extend(peak_delays)
            aggregate[kind]["pre50"].extend(pre50)
            negatives = _sample_negative_scores(
                scores,
                references[kind_index],
                seed=args.seed + track_index * 101 + kind_index * 100003,
                count=args.negative_samples_per_track,
            )
            _merge_dict_lists(aggregate[kind]["negative"], negatives)

    report = {
        "schema_version": 1,
        "model": str(args.model),
        "tracks": members,
        "configuration": {
            "windows_ms": list(windows_ms),
            "windows_samples": list(windows_samples),
            "chunk_size": args.chunk_size,
            "negative_samples_per_track": args.negative_samples_per_track,
            "negative_bands_samples": [
                {"name": name, "low": low, "high": high}
                for name, low, high in NEGATIVE_BANDS
            ],
        },
        "kinds": {},
    }

    for kind in ("onset", "offset"):
        payload = aggregate[kind]
        window_report = {}
        for window_ms, window_samples in zip(windows_ms, windows_samples):
            window_report[f"{window_ms:g}ms"] = _summary(payload["windows"][window_samples])
        delay_ms = [delay * 1000.0 / SAMPLE_RATE for delay in payload["peak_delays"]]
        report["kinds"][kind] = {
            "event_score_by_causal_window": window_report,
            "peak_delay_within_max_window_ms": _summary(delay_ms),
            "pre_event_max_within_max_window": _summary(payload["pre50"]),
            "negative_score_by_distance": {
                name: _summary(values)
                for name, values in payload["negative"].items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("V8 temporal signal diagnostic")
    for kind in ("onset", "offset"):
        print(kind.upper())
        for window, summary in report["kinds"][kind]["event_score_by_causal_window"].items():
            q = summary["quantiles"]
            above = summary["above"]
            print(
                f"  {window:>5} q50={q['q50']:.6g} q90={q['q90']:.6g} q99={q['q99']:.6g} "
                f">=.001={above['0.001']:.3f} >=.01={above['0.01']:.3f} >=.1={above['0.1']:.3f}"
            )
        delay = report["kinds"][kind]["peak_delay_within_max_window_ms"]["quantiles"]
        print(f"  peak delay ms q50={delay['q50']:.3f} q90={delay['q90']:.3f} q99={delay['q99']:.3f}")
        for band, summary in report["kinds"][kind]["negative_score_by_distance"].items():
            q = summary["quantiles"]
            print(f"  neg {band:>11} q90={q['q90']:.6g} q99={q['q99']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
