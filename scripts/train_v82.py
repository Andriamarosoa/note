"""Train V8.2 with harmonic-state hard-negative replay.

V8.2 deliberately keeps the V8.1 network and burst objective unchanged for the
first experiment. The isolated variable is the training distribution: 25% of
examples are mined from strong within-note spectral changes that are dominated
by harmonics of already-active annotated notes and are far from any annotated
boundary. This directly targets the dominant V8.1 false-positive mode.

If this improves stream precision, the existing backbone has enough capacity and
future work should refine data/representation. If it does not, the next step is
a dedicated latent harmonic-state branch rather than another threshold tweak.
"""
from __future__ import annotations

import argparse
import bisect
from pathlib import Path
import random
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from causal_note.guitarset_acoustics import load_rich_annotations
from causal_note.harmonic_state import harmonic_change
from causal_note.v81_model import build_v81_bag_model
from causal_note.v81_targets import DEFAULT_OFFSET_HORIZON_SAMPLES, DEFAULT_ONSET_HORIZON_SAMPLES
from scripts.audit_anonymous_boundary_targets import _prepare_tracks
from scripts.train_boundaries import TrainingDataError, _configure_cpu_dependencies, decode_pcm16_mono_wav, split_tracks_by_group
from scripts.train_v81 import (
    BagExample,
    SMOKE_FILTERS,
    SMOKE_TRAIN_EXAMPLES,
    SMOKE_VALIDATION_EXAMPLES,
    _assemble_numpy,
    _clean_background_example,
    _draw_examples,
    _epoch_path,
    _hard_negative_pool,
    _positive_pool,
    _prepare_burst_tracks,
)


DEFAULT_TRAIN_EXAMPLES = 1600
DEFAULT_VALIDATION_EXAMPLES = 400
DEFAULT_NEGATIVE_MARGIN = 16
DEFAULT_MINING_TRACKS = 80
DEFAULT_PROBES_PER_TRACK = 96
DEFAULT_MINED_PER_TRACK = 12
DEFAULT_BOUNDARY_MARGIN_MS = 50.0
DEFAULT_MIN_FLUX_RATIO = 0.50
DEFAULT_MIN_ACTIVE_HARMONIC_FRACTION = 0.70


def _nearest_distance(position: int, values: Sequence[int]):
    if not values:
        return None
    index = bisect.bisect_left(values, position)
    candidates = []
    if index < len(values):
        candidates.append(abs(values[index] - position))
    if index:
        candidates.append(abs(values[index - 1] - position))
    return min(candidates) if candidates else None


def _far_from(position: int, values: Sequence[int], margin: int) -> bool:
    distance = _nearest_distance(position, values)
    return distance is None or distance > margin


def _active_frequencies(notes, position: int):
    return tuple(
        note.frequency_hz
        for note in notes
        if note.onset_sample <= position < note.offset_sample
    )


def _mine_track(
    np,
    item,
    *,
    seed: int,
    probes: int,
    maximum: int,
    boundary_margin: int,
    min_flux_ratio: float,
    min_active_fraction: float,
):
    track = item.audit_track.track
    rich = load_rich_annotations(track.annotation_zip, track.annotation_member)
    notes = tuple(note for note in rich.notes if note.offset_sample < item.frame_count)
    if not notes:
        return ()
    decoded = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
    window = 1024
    eligible = tuple(
        note for note in notes
        if note.onset_sample + boundary_margin + window < note.offset_sample - boundary_margin - window
    )
    if not eligible:
        return ()

    all_onsets = tuple(sorted(item.onset_positions))
    all_offsets = tuple(sorted(item.offset_positions))
    rng = random.Random(seed)
    seen = set()
    ranked = []
    for _ in range(probes):
        note = rng.choice(eligible)
        low = note.onset_sample + boundary_margin
        high = note.offset_sample - boundary_margin
        if high <= low:
            continue
        position = rng.randint(low, high)
        if position in seen:
            continue
        seen.add(position)
        if position < window or position + window >= item.frame_count:
            continue
        if not _far_from(position, all_onsets, boundary_margin):
            continue
        if not _far_from(position, all_offsets, boundary_margin):
            continue
        frequencies = _active_frequencies(notes, position)
        if not frequencies:
            continue
        change = harmonic_change(np, decoded.samples, position, frequencies)
        if (
            change.positive_flux_over_pre_energy >= min_flux_ratio
            and change.active_harmonic_flux_fraction >= min_active_fraction
        ):
            ranked.append((change.hardness, position, change))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return tuple(ranked[:maximum])


def mine_harmonic_negative_pool(
    np,
    tracks,
    *,
    seed: int,
    track_limit: int,
    probes_per_track: int,
    per_track: int,
    boundary_margin_ms: float,
    min_flux_ratio: float,
    min_active_fraction: float,
):
    if track_limit <= 0 or probes_per_track <= 0 or per_track <= 0:
        raise TrainingDataError("V8.2 mining counts must be positive")
    margin = round(float(boundary_margin_ms) * SAMPLE_RATE / 1000.0)
    if margin <= 0:
        raise TrainingDataError("V8.2 boundary mining margin must be positive")
    if min_flux_ratio < 0.0 or not 0.0 <= min_active_fraction <= 1.0:
        raise TrainingDataError("invalid V8.2 harmonic mining thresholds")

    indices = list(range(len(tracks)))
    random.Random(seed).shuffle(indices)
    selected = indices[: min(track_limit, len(indices))]
    pool = []
    diagnostics = []
    for ordinal, track_index in enumerate(selected, start=1):
        rows = _mine_track(
            np,
            tracks[track_index],
            seed=seed + track_index * 7919,
            probes=probes_per_track,
            maximum=per_track,
            boundary_margin=margin,
            min_flux_ratio=min_flux_ratio,
            min_active_fraction=min_active_fraction,
        )
        for hardness, position, change in rows:
            pool.append(BagExample(track_index, int(position), "harmonic_transform"))
            diagnostics.append((float(hardness), float(change.active_harmonic_flux_fraction), float(change.positive_flux_over_pre_energy)))
        print(f"V8.2 mined {ordinal}/{len(selected)} tracks: {len(rows)} candidates")
    if not pool:
        raise TrainingDataError("V8.2 harmonic hard-negative pool is empty")
    hardness = sorted(row[0] for row in diagnostics)
    median = hardness[len(hardness) // 2]
    print(
        "V8.2 harmonic pool:",
        f"examples={len(pool)}",
        f"tracks={len(selected)}",
        f"hardness_median={median:.6f}",
        f"active_fraction_min={min(row[1] for row in diagnostics):.6f}",
        f"flux_ratio_min={min(row[2] for row in diagnostics):.6f}",
    )
    return tuple(pool)


def _draw_v82_examples(
    tracks,
    harmonic_pool,
    *,
    count: int,
    seed: int,
    maximum_horizon: int,
    negative_margin: int,
):
    if count < 8:
        raise TrainingDataError("V8.2 example count must be >= 8")
    rng = random.Random(seed)
    positives = _positive_pool(tracks, maximum_horizon)
    pre_boundary = _hard_negative_pool(tracks, maximum_horizon)
    positive_count = count // 2
    harmonic_count = count // 4
    pre_count = count // 8
    background_count = count - positive_count - harmonic_count - pre_count
    result = [rng.choice(positives) for _ in range(positive_count)]
    result.extend(rng.choice(harmonic_pool) for _ in range(harmonic_count))
    result.extend(rng.choice(pre_boundary) for _ in range(pre_count))
    result.extend(
        _clean_background_example(tracks, rng, maximum_horizon, negative_margin)
        for _ in range(background_count)
    )
    rng.shuffle(result)
    return tuple(result)


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Train V8.2 with harmonic hard-negative replay.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "causal-boundaries-v82.keras")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--validation-examples", type=int, default=DEFAULT_VALIDATION_EXAMPLES)
    parser.add_argument("--negative-margin", type=int, default=DEFAULT_NEGATIVE_MARGIN)
    parser.add_argument("--mine-tracks", type=int, default=DEFAULT_MINING_TRACKS)
    parser.add_argument("--probes-per-track", type=int, default=DEFAULT_PROBES_PER_TRACK)
    parser.add_argument("--mined-per-track", type=int, default=DEFAULT_MINED_PER_TRACK)
    parser.add_argument("--boundary-margin-ms", type=float, default=DEFAULT_BOUNDARY_MARGIN_MS)
    parser.add_argument("--min-flux-ratio", type=float, default=DEFAULT_MIN_FLUX_RATIO)
    parser.add_argument("--min-active-harmonic-fraction", type=float, default=DEFAULT_MIN_ACTIVE_HARMONIC_FRACTION)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None):
    args = create_argument_parser().parse_args(argv)
    if args.epochs <= 0 or args.filters <= 0 or args.learning_rate <= 0:
        raise TrainingDataError("epochs, filters and learning rate must be positive")
    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    train_tracks, validation_tracks = split_tracks_by_group(indexed, validation_fraction=0.2, seed=args.seed)
    train = _prepare_burst_tracks(_prepare_tracks(train_tracks))
    validation = _prepare_burst_tracks(_prepare_tracks(validation_tracks))

    np, tf = _configure_cpu_dependencies(args.seed)
    filters = min(args.filters, SMOKE_FILTERS) if args.smoke else args.filters
    model = build_v81_bag_model(filters=filters)
    receptive_field = int(model.receptive_field)
    maximum_horizon = int(model.maximum_horizon_samples)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss={
            "onset_bag_presence": tf.keras.losses.BinaryCrossentropy(),
            "offset_bag_presence": tf.keras.losses.BinaryCrossentropy(),
            "onset_mass": tf.keras.losses.MeanSquaredError(),
            "offset_mass": tf.keras.losses.MeanSquaredError(),
            "onset_delay": tf.keras.losses.MeanAbsoluteError(),
            "offset_delay": tf.keras.losses.MeanAbsoluteError(),
            "onset_count": tf.keras.losses.SparseCategoricalCrossentropy(),
            "offset_count": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={
            "onset_bag_presence": 2.0,
            "offset_bag_presence": 2.0,
            "onset_mass": 1.0,
            "offset_mass": 1.0,
            "onset_delay": 0.20,
            "offset_delay": 0.20,
            "onset_count": 0.50,
            "offset_count": 0.50,
        },
        metrics={
            "onset_bag_presence": [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
            "offset_bag_presence": [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
        },
    )

    mine_tracks = min(args.mine_tracks, 4) if args.smoke else args.mine_tracks
    probes = min(args.probes_per_track, 12) if args.smoke else args.probes_per_track
    per_track = min(args.mined_per_track, 4) if args.smoke else args.mined_per_track
    harmonic_pool = mine_harmonic_negative_pool(
        np,
        train,
        seed=args.seed + 4242,
        track_limit=mine_tracks,
        probes_per_track=probes,
        per_track=per_track,
        boundary_margin_ms=args.boundary_margin_ms,
        min_flux_ratio=args.min_flux_ratio,
        min_active_fraction=args.min_active_harmonic_fraction,
    )

    validation_count = min(args.validation_examples, SMOKE_VALIDATION_EXAMPLES) if args.smoke else args.validation_examples
    validation_examples = _draw_examples(
        validation,
        count=validation_count,
        seed=args.seed + 991,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )
    validation_data = _assemble_numpy(
        np, validation, validation_examples,
        receptive_field=receptive_field,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.smoke else args.epochs
    train_count = min(args.train_examples, SMOKE_TRAIN_EXAMPLES) if args.smoke else args.train_examples
    print(
        "V8.2 configuration:",
        f"backbone=v81-identical",
        f"receptive_field={receptive_field}",
        f"filters={filters}",
        f"train_examples={train_count}",
        f"validation_examples={validation_count}",
        f"harmonic_pool={len(harmonic_pool)}",
    )

    for epoch in range(1, epochs + 1):
        examples = _draw_v82_examples(
            train,
            harmonic_pool,
            count=train_count,
            seed=args.seed + epoch * 10007,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        counts = {}
        for example in examples:
            counts[example.stratum] = counts.get(example.stratum, 0) + 1
        print(f"V8.2 epoch {epoch} strata:", counts)
        train_data = _assemble_numpy(
            np, train, examples,
            receptive_field=receptive_field,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        model.fit(
            train_data[0], train_data[1], sample_weight=train_data[2],
            validation_data=(validation_data[0], validation_data[1], validation_data[2]),
            initial_epoch=epoch - 1, epochs=epoch, batch_size=8, shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()], verbose=2,
        )
        stream = model.get_layer(model.stream_model_name)
        epoch_path = _epoch_path(args.output, epoch)
        stream.save(epoch_path)
        stream.save(args.output)
        print(f"saved V8.2 stream epoch {epoch}: {epoch_path}")

    if not args.output.is_file():
        raise RuntimeError("V8.2 training completed without a saved stream model")
    print(f"latest V8.2 stream model: {args.output}")


if __name__ == "__main__":
    main()
