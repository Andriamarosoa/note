"""Compare V8.1 false onsets with true onsets and within-note background.

The first acoustic audit showed that most V8.1 false-onset spectral flux falls
on harmonic families of already-active notes.  This script tests whether that
observation is actually discriminative rather than a consequence of harmonic
masks covering a large part of the spectrum.

Three populations are measured on the same locked 12-track pilot subset:

* false onset positions from the completed V8.1 epoch-03 audit;
* every unique annotated onset position;
* deterministic within-note background positions at least +/-50 ms from every
  annotated onset/offset, sampled per track to match the number of FP positions.

For each point we report harmonic-mask bin coverage, spectral-flux fraction in
that mask, flux enrichment over chance coverage, short-term harmonic-profile
cosine stability, RMS change, and residual flux attributable to genuinely new
annotated notes (true-onset population only).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import wave
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from causal_note.guitarset_acoustics import RichNote, load_rich_annotations


FFT_WINDOW = 2048
FFT_SIZE = 8192
MIN_HZ = 50.0
MAX_HZ = 8000.0
BOUNDARY_GUARD_MS = 50.0
BOUNDARY_GUARD_SAMPLES = round(BOUNDARY_GUARD_MS * SAMPLE_RATE / 1000.0)
RANDOM_SEED = 1337


class AuditError(RuntimeError):
    pass


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


def _read_pcm16_mono(track) -> np.ndarray:
    try:
        with zipfile.ZipFile(track.audio_zip, "r") as archive:
            raw = archive.read(track.audio_member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise AuditError(f"cannot read {track.audio_member}") from exc
    with wave.open(io.BytesIO(raw), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
            raise AuditError(
                f"expected mono PCM16 {SAMPLE_RATE} Hz for {track.audio_member}, got "
                f"channels={wav.getnchannels()} width={wav.getsampwidth()} rate={wav.getframerate()}"
            )
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0


def _window(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    source_start = max(start, 0)
    source_end = min(start + length, len(samples))
    if source_end > source_start:
        destination = source_start - start
        result[destination : destination + source_end - source_start] = samples[source_start:source_end]
    return result


def _power(values: np.ndarray) -> np.ndarray:
    transformed = np.fft.rfft(values * np.hanning(len(values)), n=FFT_SIZE)
    return np.abs(transformed) ** 2


def _harmonic_ranges(f0: float) -> Tuple[Tuple[int, int], ...]:
    if not math.isfinite(f0) or f0 <= 0.0:
        return ()
    resolution = SAMPLE_RATE / FFT_SIZE
    result = []
    harmonic = 1
    while harmonic * f0 <= MAX_HZ:
        frequency = harmonic * f0
        bandwidth = max(12.0, frequency * 0.008)
        low = max(0, int(math.floor((frequency - bandwidth) / resolution)))
        high = min(FFT_SIZE // 2, int(math.ceil((frequency + bandwidth) / resolution)))
        result.append((low, high + 1))
        harmonic += 1
    return tuple(result)


def _mask(frequencies: Iterable[float], length: int) -> np.ndarray:
    result = np.zeros(length, dtype=bool)
    for frequency in frequencies:
        for low, high in _harmonic_ranges(float(frequency)):
            result[low:min(high, length)] = True
    return result


def _active_before(notes: Sequence[RichNote], sample: int) -> Tuple[RichNote, ...]:
    return tuple(note for note in notes if note.onset_sample < sample < note.offset_sample)


def _starting(notes: Sequence[RichNote], sample: int) -> Tuple[RichNote, ...]:
    return tuple(note for note in notes if note.onset_sample == sample)


def _harmonic_profile(power: np.ndarray, f0: float) -> np.ndarray:
    values = []
    for low, high in _harmonic_ranges(f0):
        values.append(float(power[low:high].sum()))
    if not values:
        return np.zeros(0, dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-20:
        return np.zeros_like(vector)
    return vector / norm


def _profile_cosine(pre_power: np.ndarray, post_power: np.ndarray, active: Sequence[RichNote]) -> Optional[float]:
    cosines = []
    for note in active:
        pre = _harmonic_profile(pre_power, note.frequency_hz)
        post = _harmonic_profile(post_power, note.frequency_hz)
        if not len(pre) or len(pre) != len(post):
            continue
        pre_norm = float(np.linalg.norm(pre))
        post_norm = float(np.linalg.norm(post))
        if pre_norm <= 1e-20 or post_norm <= 1e-20:
            continue
        cosines.append(float(np.clip(np.dot(pre, post) / (pre_norm * post_norm), -1.0, 1.0)))
    return float(np.mean(cosines)) if cosines else None


def _features(samples: np.ndarray, sample: int, active: Sequence[RichNote], starting: Sequence[RichNote]) -> Dict[str, object]:
    pre_values = _window(samples, sample - FFT_WINDOW, FFT_WINDOW)
    post_values = _window(samples, sample, FFT_WINDOW)
    pre = _power(pre_values)
    post = _power(post_values)
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    valid = (frequencies >= MIN_HZ) & (frequencies <= MAX_HZ)
    flux = np.maximum(post - pre, 0.0)
    flux[~valid] = 0.0
    total_flux = float(flux.sum())

    active_mask = _mask((note.frequency_hz for note in active), len(flux)) & valid
    valid_bins = int(valid.sum())
    active_bin_fraction = float(active_mask.sum() / valid_bins) if valid_bins else 0.0
    active_flux_fraction = float(flux[active_mask].sum() / total_flux) if total_flux > 1e-20 else 0.0
    flux_enrichment = active_flux_fraction / active_bin_fraction if active_bin_fraction > 0.0 else None

    starting_mask = _mask((note.frequency_hz for note in starting), len(flux)) & valid & ~active_mask
    new_note_residual_fraction = (
        float(flux[starting_mask].sum() / total_flux) if total_flux > 1e-20 and starting else None
    )

    pre_energy = float(pre[valid].sum()) + 1e-20
    pre_rms = float(np.sqrt(np.mean(pre_values * pre_values)) + 1e-12)
    post_rms = float(np.sqrt(np.mean(post_values * post_values)) + 1e-12)
    return {
        "active_note_count": len(active),
        "starting_note_count": len(starting),
        "active_harmonic_bin_fraction": active_bin_fraction,
        "active_harmonic_flux_fraction": active_flux_fraction,
        "active_harmonic_flux_enrichment": flux_enrichment,
        "active_harmonic_profile_cosine": _profile_cosine(pre, post, active),
        "new_note_residual_harmonic_flux_fraction": new_note_residual_fraction,
        "positive_flux_over_pre_energy": total_flux / pre_energy,
        "rms_delta_db": 20.0 * math.log10(post_rms / pre_rms),
    }


def _boundary_positions(notes: Sequence[RichNote]) -> Tuple[int, ...]:
    return tuple(sorted({position for note in notes for position in (note.onset_sample, note.offset_sample)}))


def _far_from_boundaries(sample: int, boundaries: Sequence[int]) -> bool:
    import bisect
    index = bisect.bisect_left(boundaries, sample)
    if index < len(boundaries) and abs(boundaries[index] - sample) <= BOUNDARY_GUARD_SAMPLES:
        return False
    if index and abs(boundaries[index - 1] - sample) <= BOUNDARY_GUARD_SAMPLES:
        return False
    return True


def _sample_active_background(
    notes: Sequence[RichNote], boundaries: Sequence[int], frame_count: int, count: int, rng: random.Random
) -> Tuple[int, ...]:
    if count <= 0:
        return ()
    low = FFT_WINDOW
    high = frame_count - FFT_WINDOW - 1
    if high <= low:
        return ()
    selected = set()
    attempts = 0
    maximum_attempts = max(50_000, count * 1000)
    while len(selected) < count and attempts < maximum_attempts:
        attempts += 1
        sample = rng.randint(low, high)
        if sample in selected or not _far_from_boundaries(sample, boundaries):
            continue
        if not _active_before(notes, sample):
            continue
        selected.add(sample)
    if len(selected) < count:
        raise AuditError(f"could only sample {len(selected)}/{count} active background points")
    return tuple(sorted(selected))


def _summarize(records: Sequence[dict]) -> Dict[str, object]:
    metrics = (
        "active_harmonic_bin_fraction",
        "active_harmonic_flux_fraction",
        "active_harmonic_flux_enrichment",
        "active_harmonic_profile_cosine",
        "new_note_residual_harmonic_flux_fraction",
        "positive_flux_over_pre_energy",
        "rms_delta_db",
    )
    summary: Dict[str, object] = {
        "positions": len(records),
        "with_active_notes": sum(record["active_note_count"] > 0 for record in records),
        "with_starting_notes": sum(record["starting_note_count"] > 0 for record in records),
    }
    for metric in metrics:
        values = [float(record[metric]) for record in records if record.get(metric) is not None and math.isfinite(float(record[metric]))]
        summary[metric] = _quantiles(values)
    summary["active_flux_ge_0_50"] = sum(record["active_harmonic_flux_fraction"] >= 0.50 for record in records)
    summary["active_flux_ge_0_70"] = sum(record["active_harmonic_flux_fraction"] >= 0.70 for record in records)
    summary["enrichment_ge_2"] = sum(
        record.get("active_harmonic_flux_enrichment") is not None and record["active_harmonic_flux_enrichment"] >= 2.0
        for record in records
    )
    summary["enrichment_ge_4"] = sum(
        record.get("active_harmonic_flux_enrichment") is not None and record["active_harmonic_flux_enrichment"] >= 4.0
        for record in records
    )
    summary["profile_cosine_ge_0_90"] = sum(
        record.get("active_harmonic_profile_cosine") is not None and record["active_harmonic_profile_cosine"] >= 0.90
        for record in records
    )
    return summary


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run harmonic controls for the V8.1 acoustic-label audit.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--fp-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args) -> Dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with args.fp_audit.open("r", encoding="utf-8") as stream:
        fp_audit = json.load(stream)

    members = tuple(fp_audit["scope"]["members"])
    fp_by_member: Dict[str, List[int]] = defaultdict(list)
    for record in fp_audit["false_positive_onset_records"]:
        fp_by_member[record["member"]].append(int(record["sample"]))
    fp_by_member = {member: sorted(set(values)) for member, values in fp_by_member.items()}

    tracks = {track.annotation_member: track for track in index_guitarset(args.dataset_dir)}
    rng = random.Random(RANDOM_SEED)
    populations: Dict[str, List[dict]] = {"false_positive": [], "true_onset": [], "active_background": []}
    per_track = []

    for index, member in enumerate(members, start=1):
        track = tracks.get(member)
        if track is None:
            raise AuditError(f"missing indexed track {member}")
        rich = load_rich_annotations(track.annotation_zip, member)
        notes = tuple(sorted((note for slot in rich.notes_by_slot for note in slot), key=lambda note: (note.onset_sample, note.slot)))
        samples = _read_pcm16_mono(track)
        boundaries = _boundary_positions(notes)
        fp_positions = tuple(fp_by_member.get(member, ()))
        onset_positions = tuple(sorted({note.onset_sample for note in notes}))
        background_positions = _sample_active_background(notes, boundaries, len(samples), len(fp_positions), rng)

        for population, positions in (
            ("false_positive", fp_positions),
            ("true_onset", onset_positions),
            ("active_background", background_positions),
        ):
            for sample in positions:
                active = _active_before(notes, sample)
                starting = _starting(notes, sample) if population == "true_onset" else ()
                record = _features(samples, sample, active, starting)
                record.update({"member": member, "sample": sample})
                populations[population].append(record)

        per_track.append(
            {
                "member": member,
                "fp_unique_positions": len(fp_positions),
                "true_onset_unique_positions": len(onset_positions),
                "active_background_positions": len(background_positions),
            }
        )
        print(f"controlled {index}/{len(members)}: {member}")

    result = {
        "schema_version": 1,
        "source_fp_audit": str(args.fp_audit),
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "fft_window_samples": FFT_WINDOW,
            "fft_window_ms": FFT_WINDOW * 1000.0 / SAMPLE_RATE,
            "fft_size": FFT_SIZE,
            "boundary_guard_ms": BOUNDARY_GUARD_MS,
            "random_seed": RANDOM_SEED,
        },
        "interpretation_guard": (
            "These are acoustic diagnostics, not automatic relabels. Harmonic families can overlap and pickup/timbre dynamics change harmonic amplitudes."
        ),
        "populations": {name: _summarize(records) for name, records in populations.items()},
        "tracks": per_track,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result["populations"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
