"""Train V8.2b with replay of actual frozen-V8.1 train false onsets.

V8.2b keeps the V8.1 architecture and losses.  The only training-data change is
that one quarter of each epoch is drawn from real V8.1 false-positive onset
positions extracted on the train split.  Replay positions are unique within an
epoch, capped per track, preserve the measured solo/comp mixture, and supervise
only the onset presence/mass heads.  All offset weights are zero on replay rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, index_guitarset
from causal_note.v81_model import build_v81_bag_model
from causal_note.v81_targets import (
    DEFAULT_OFFSET_HORIZON_SAMPLES,
    DEFAULT_ONSET_HORIZON_SAMPLES,
    response_window_is_empty,
)
from causal_note.v82b_replay import (
    ReplayError,
    ReplayPoint,
    load_replay_points,
    select_replay_points,
    summarize_replay,
)
from scripts.audit_anonymous_boundary_targets import _prepare_tracks
from scripts.train_boundaries import TrainingDataError, _configure_cpu_dependencies, split_tracks_by_group
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


DEFAULT_TRAIN_EXAMPLES = 800
DEFAULT_VALIDATION_EXAMPLES = 200
DEFAULT_MAX_REPLAY_PER_TRACK = 16
DEFAULT_NEGATIVE_MARGIN = 16


def _member(item) -> str:
    return str(item.audit_track.track.annotation_member)


def _clean_replay_pool(
    tracks: Sequence[object],
    raw_points: Sequence[ReplayPoint],
    *,
    maximum_horizon: int,
    negative_margin: int,
) -> Tuple[ReplayPoint, ...]:
    member_to_track = {_member(item): item for item in tracks}
    unknown = sorted({point.member for point in raw_points if point.member not in member_to_track})
    if unknown:
        raise TrainingDataError(
            "V8.2b replay audit contains members outside the train split: " + ", ".join(unknown[:5])
        )

    cleaned = []
    for point in raw_points:
        item = member_to_track[point.member]
        latest = item.frame_count - maximum_horizon - 1
        if point.sample < 0 or point.sample > latest:
            continue
        if not response_window_is_empty(
            item.onset_positions,
            point.sample,
            DEFAULT_ONSET_HORIZON_SAMPLES,
            margin_samples=negative_margin,
        ):
            # An unmatched prediction can still lie close to a reference already
            # claimed by another prediction.  Do not turn that ambiguity into a
            # negative label.
            continue
        cleaned.append(point)
    if not cleaned:
        raise TrainingDataError("V8.2b clean true-FP replay pool is empty")
    return tuple(cleaned)


def _draw_v82b_examples(
    tracks: Sequence[object],
    replay_pool: Sequence[ReplayPoint],
    *,
    count: int,
    seed: int,
    maximum_horizon: int,
    negative_margin: int,
    max_replay_per_track: int,
):
    if count < 8:
        raise TrainingDataError("V8.2b example count must be >= 8")
    rng = random.Random(seed)
    positives = _positive_pool(tracks, maximum_horizon)
    hard = _hard_negative_pool(tracks, maximum_horizon)
    member_to_index = {_member(item): index for index, item in enumerate(tracks)}

    positive_count = count // 2
    replay_count = count // 4
    hard_count = count // 8
    background_count = count - positive_count - replay_count - hard_count

    try:
        selected_replay = select_replay_points(
            replay_pool,
            count=replay_count,
            seed=seed + 4703,
            max_per_track=max_replay_per_track,
        )
    except ReplayError as exc:
        raise TrainingDataError(str(exc)) from exc

    examples = [rng.choice(positives) for _ in range(positive_count)]
    examples.extend(
        BagExample(
            member_to_index[point.member],
            point.sample,
            "true_fp_replay_harmonic" if point.harmonic_proxy else "true_fp_replay",
        )
        for point in selected_replay
    )
    examples.extend(rng.choice(hard) for _ in range(hard_count))
    examples.extend(
        _clean_background_example(tracks, rng, maximum_horizon, negative_margin)
        for _ in range(background_count)
    )
    rng.shuffle(examples)
    return tuple(examples), selected_replay


def _assemble_v82b_numpy(
    np,
    tracks: Sequence[object],
    examples: Sequence[BagExample],
    *,
    receptive_field: int,
    maximum_horizon: int,
    negative_margin: int,
):
    audio, targets, weights, _ = _assemble_numpy(
        np,
        tracks,
        examples,
        receptive_field=receptive_field,
        maximum_horizon=maximum_horizon,
        negative_margin=negative_margin,
    )

    for row, example in enumerate(examples):
        if not example.stratum.startswith("true_fp_replay"):
            continue
        # Actual V8.1 FP replay is an onset-only negative.  Force that contract
        # explicitly rather than inheriting generic two-head background labels.
        targets["onset_bag_presence"][row, 0] = 0.0
        targets["onset_mass"][row, 0] = 0.0
        targets["onset_delay"][row, 0] = 0.0
        targets["onset_count"][row] = 0
        weights["onset_bag_presence"][row] = 1.0
        weights["onset_mass"][row] = 1.0
        weights["onset_delay"][row] = 0.0
        weights["onset_count"][row] = 0.0

        for suffix in ("bag_presence", "mass", "delay", "count"):
            weights[f"offset_{suffix}"][row] = 0.0

    supervised_counts = {
        name: int(np.count_nonzero(values)) for name, values in weights.items()
    }
    return audio, targets, weights, supervised_counts


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Train V8.2b with true V8.1 train-FP onset replay.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model" / "causal-boundaries-v82b.keras",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--validation-examples", type=int, default=DEFAULT_VALIDATION_EXAMPLES)
    parser.add_argument("--negative-margin", type=int, default=DEFAULT_NEGATIVE_MARGIN)
    parser.add_argument("--max-replay-per-track", type=int, default=DEFAULT_MAX_REPLAY_PER_TRACK)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None):
    args = create_argument_parser().parse_args(argv)
    if args.epochs <= 0 or args.filters <= 0 or args.learning_rate <= 0:
        raise TrainingDataError("epochs, filters and learning rate must be positive")
    if args.train_examples <= 0 or args.validation_examples <= 0:
        raise TrainingDataError("example counts must be positive")
    if args.negative_margin < 0 or args.max_replay_per_track <= 0:
        raise TrainingDataError("negative margin must be >=0 and replay cap must be positive")

    indexed = tuple(
        track for track in index_guitarset(args.dataset_dir)
        if track.player_id in ALLOWED_PLAYERS
    )
    train_tracks, validation_tracks = split_tracks_by_group(
        indexed, validation_fraction=0.2, seed=args.seed
    )
    train = _prepare_burst_tracks(_prepare_tracks(train_tracks))
    validation = _prepare_burst_tracks(_prepare_tracks(validation_tracks))

    np, tf = _configure_cpu_dependencies(args.seed)
    filters = min(args.filters, SMOKE_FILTERS) if args.smoke else args.filters
    model = build_v81_bag_model(filters=filters)
    receptive_field = int(model.receptive_field)
    maximum_horizon = int(model.maximum_horizon_samples)

    raw_replay = load_replay_points(args.replay_audit)
    replay_pool = _clean_replay_pool(
        train,
        raw_replay,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )

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
        weighted_metrics={
            "onset_bag_presence": [
                tf.keras.metrics.Precision(name="weighted_precision"),
                tf.keras.metrics.Recall(name="weighted_recall"),
            ],
            "offset_bag_presence": [
                tf.keras.metrics.Precision(name="weighted_precision"),
                tf.keras.metrics.Recall(name="weighted_recall"),
            ],
        },
    )

    validation_count = (
        min(args.validation_examples, SMOKE_VALIDATION_EXAMPLES)
        if args.smoke else args.validation_examples
    )
    validation_examples = _draw_examples(
        validation,
        count=validation_count,
        seed=args.seed + 991,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )
    validation_data = _assemble_numpy(
        np,
        validation,
        validation_examples,
        receptive_field=receptive_field,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.smoke else args.epochs
    train_count = min(args.train_examples, SMOKE_TRAIN_EXAMPLES) if args.smoke else args.train_examples
    replay_manifest = {
        "schema_version": 1,
        "source_audit": str(args.replay_audit),
        "raw_pool": summarize_replay(raw_replay),
        "clean_pool": summarize_replay(replay_pool),
        "configuration": {
            "seed": args.seed,
            "train_examples": train_count,
            "max_replay_per_track": args.max_replay_per_track,
            "onset_only": True,
            "without_replacement_within_epoch": True,
        },
        "epochs": [],
    }

    print(
        "V8.2b configuration:",
        f"receptive_field={receptive_field}",
        f"onset_horizon={DEFAULT_ONSET_HORIZON_SAMPLES}",
        f"offset_horizon={DEFAULT_OFFSET_HORIZON_SAMPLES}",
        f"filters={filters}",
        f"train_examples={train_count}",
        f"validation_examples={validation_count}",
        f"raw_replay={len(raw_replay)}",
        f"clean_replay={len(replay_pool)}",
    )
    print("V8.2b raw replay summary:", summarize_replay(raw_replay))
    print("V8.2b clean replay summary:", summarize_replay(replay_pool))
    print("V8.2b validation supervision:", validation_data[3])

    for epoch in range(1, epochs + 1):
        train_examples, selected_replay = _draw_v82b_examples(
            train,
            replay_pool,
            count=train_count,
            seed=args.seed + epoch * 10007,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
            max_replay_per_track=args.max_replay_per_track,
        )
        train_data = _assemble_v82b_numpy(
            np,
            train,
            train_examples,
            receptive_field=receptive_field,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        summary = summarize_replay(selected_replay)
        print(f"V8.2b epoch {epoch} replay:", summary)
        print(f"V8.2b epoch {epoch} supervision:", train_data[3])
        replay_manifest["epochs"].append(
            {
                "epoch": epoch,
                "summary": summary,
                "points": [
                    {
                        "member": point.member,
                        "sample": point.sample,
                        "arrangement": point.arrangement,
                        "model_onset_score": point.model_onset_score,
                        "harmonic_proxy": point.harmonic_proxy,
                    }
                    for point in selected_replay
                ],
            }
        )

        model.fit(
            train_data[0],
            train_data[1],
            sample_weight=train_data[2],
            validation_data=(validation_data[0], validation_data[1], validation_data[2]),
            initial_epoch=epoch - 1,
            epochs=epoch,
            batch_size=8,
            shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            verbose=2,
        )
        stream = model.get_layer(model.stream_model_name)
        epoch_path = _epoch_path(args.output, epoch)
        stream.save(epoch_path)
        stream.save(args.output)
        print(f"saved V8.2b stream epoch {epoch}: {epoch_path}")

    manifest_path = args.output.with_name(f"{args.output.stem}.replay.json")
    manifest_path.write_text(json.dumps(replay_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.output.is_file():
        raise RuntimeError("V8.2b training completed without a saved stream model")
    print(f"latest V8.2b stream model: {args.output}")
    print(f"V8.2b replay manifest: {manifest_path}")


if __name__ == "__main__":
    main()
