"""Controlled V8.1 continuation A/B: annotation control vs true-FP onset replay.

Both arms start from the exact same frozen V8.1 epoch-03 stream weights.  They
use the same optimizer configuration, number of examples/steps, seed schedule,
validation set, and per-head total sample-weight mass.  The control arm keeps
V8.1 sampling (50% burst / 25% pre-boundary / 25% background).  The replay arm
keeps 50% positive bursts but replaces half of the ordinary negatives with
actual frozen-V8.1 train false-onset crops (25% replay / 12.5% pre-boundary /
12.5% background).

Unlike V8.2b, replay rows do NOT mask or rewrite offset supervision.  Offset
targets and masks come directly from the annotations at the replay crop.  Only
the onset replay contract is forced negative.  Replay sample weights are then
scaled per head so each replay-arm head has exactly the same total supervision
mass as the corresponding control-arm head for that epoch.

The source artifact stores only the deployable stream model, so Adam moment
state cannot be restored.  Both arms therefore start with identical stream
weights and fresh, identically configured Adam optimizers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Dict, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, index_guitarset
from causal_note.v81_model import build_v81_bag_model
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
from scripts.train_v82b import _clean_replay_pool


DEFAULT_TRAIN_EXAMPLES = 800
DEFAULT_VALIDATION_EXAMPLES = 200
DEFAULT_MAX_REPLAY_PER_TRACK = 16
DEFAULT_NEGATIVE_MARGIN = 16
DEFAULT_REPLAY_FRACTION = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights_digest(model) -> str:
    digest = hashlib.sha256()
    for value in model.get_weights():
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _compile_model(tf, model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
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


def _draw_replay_examples(
    tracks: Sequence[object],
    replay_pool: Sequence[ReplayPoint],
    *,
    count: int,
    seed: int,
    replay_fraction: float,
    maximum_horizon: int,
    negative_margin: int,
    max_replay_per_track: int,
) -> Tuple[Tuple[BagExample, ...], Tuple[ReplayPoint, ...]]:
    if count < 8:
        raise TrainingDataError("V8.2c example count must be >= 8")
    if not 0.0 < replay_fraction < 0.5:
        raise TrainingDataError("replay_fraction must be in (0, 0.5)")

    rng = random.Random(seed)
    positives = _positive_pool(tracks, maximum_horizon)
    hard = _hard_negative_pool(tracks, maximum_horizon)
    member_to_index = {
        str(item.audit_track.track.annotation_member): index
        for index, item in enumerate(tracks)
    }

    positive_count = count // 2
    replay_count = int(round(count * replay_fraction))
    negative_remainder = count - positive_count - replay_count
    hard_count = negative_remainder // 2
    background_count = negative_remainder - hard_count

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
        BagExample(member_to_index[point.member], point.sample, "true_fp_replay")
        for point in selected_replay
    )
    examples.extend(rng.choice(hard) for _ in range(hard_count))
    examples.extend(
        _clean_background_example(tracks, rng, maximum_horizon, negative_margin)
        for _ in range(background_count)
    )
    rng.shuffle(examples)
    return tuple(examples), selected_replay


def _apply_onset_replay_contract(targets, weights, examples: Sequence[BagExample]) -> None:
    """Force only onset replay labels; leave every offset tensor untouched."""
    for row, example in enumerate(examples):
        if example.stratum != "true_fp_replay":
            continue
        targets["onset_bag_presence"][row, 0] = 0.0
        targets["onset_mass"][row, 0] = 0.0
        targets["onset_delay"][row, 0] = 0.0
        targets["onset_count"][row] = 0
        weights["onset_bag_presence"][row] = 1.0
        weights["onset_mass"][row] = 1.0
        weights["onset_delay"][row] = 0.0
        weights["onset_count"][row] = 0.0


def _weight_mass(weights) -> Dict[str, float]:
    return {name: float(values.sum()) for name, values in weights.items()}


def _match_supervision_mass(control_weights, replay_weights) -> Dict[str, dict]:
    """Scale replay active weights so every head matches control total mass."""
    report = {}
    for name, control_values in control_weights.items():
        control_mass = float(control_values.sum())
        replay_values = replay_weights[name]
        replay_mass = float(replay_values.sum())
        if control_mass == 0.0 and replay_mass == 0.0:
            scale = 1.0
        elif control_mass <= 0.0 or replay_mass <= 0.0:
            raise TrainingDataError(
                f"cannot match supervision mass for {name}: control={control_mass} replay={replay_mass}"
            )
        else:
            scale = control_mass / replay_mass
            replay_values *= scale
        matched = float(replay_values.sum())
        tolerance = max(1e-5, abs(control_mass) * 1e-6)
        if abs(matched - control_mass) > tolerance:
            raise TrainingDataError(
                f"supervision mass mismatch for {name}: control={control_mass} replay={matched}"
            )
        report[name] = {
            "control_mass": control_mass,
            "replay_raw_mass": replay_mass,
            "replay_scale": scale,
            "replay_matched_mass": matched,
        }
    return report


def _supervised_counts(np, weights) -> Dict[str, int]:
    return {name: int(np.count_nonzero(values)) for name, values in weights.items()}


def _initialize_from_source(tf, source_model: Path, filters: int, learning_rate: float):
    source = tf.keras.models.load_model(source_model, compile=False)
    control = build_v81_bag_model(filters=filters)
    replay = build_v81_bag_model(filters=filters)
    control_stream = control.get_layer(control.stream_model_name)
    replay_stream = replay.get_layer(replay.stream_model_name)
    try:
        control_stream.set_weights(source.get_weights())
        replay_stream.set_weights(source.get_weights())
    except ValueError as exc:
        raise TrainingDataError(
            "source V8.1 stream weights do not match requested V8.2c architecture"
        ) from exc

    source_digest = _weights_digest(source)
    control_digest = _weights_digest(control_stream)
    replay_digest = _weights_digest(replay_stream)
    if source_digest != control_digest or source_digest != replay_digest:
        raise TrainingDataError("A/B arms did not initialize identically from source checkpoint")

    _compile_model(tf, control, learning_rate)
    _compile_model(tf, replay, learning_rate)
    return source_digest, control, replay


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Controlled continuation from V8.1 epoch-03: control vs true-FP onset replay."
    )
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model" / "v82c-ab")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--validation-examples", type=int, default=DEFAULT_VALIDATION_EXAMPLES)
    parser.add_argument("--negative-margin", type=int, default=DEFAULT_NEGATIVE_MARGIN)
    parser.add_argument("--replay-fraction", type=float, default=DEFAULT_REPLAY_FRACTION)
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
    if not args.source_model.is_file() or not args.replay_audit.is_file():
        raise TrainingDataError("source model and replay audit must exist")

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
    source_weights_sha256, control_model, replay_model = _initialize_from_source(
        tf, args.source_model, args.filters, args.learning_rate
    )
    receptive_field = int(control_model.receptive_field)
    maximum_horizon = int(control_model.maximum_horizon_samples)
    if (
        receptive_field != int(replay_model.receptive_field)
        or maximum_horizon != int(replay_model.maximum_horizon_samples)
    ):
        raise TrainingDataError("A/B model geometry differs")

    raw_replay = load_replay_points(args.replay_audit)
    replay_pool = _clean_replay_pool(
        train,
        raw_replay,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
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

    epochs = 1 if args.smoke else args.epochs
    train_count = min(args.train_examples, SMOKE_TRAIN_EXAMPLES) if args.smoke else args.train_examples
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_output = args.output_dir / "control.keras"
    replay_output = args.output_dir / "replay.keras"

    manifest = {
        "schema_version": 1,
        "experiment": "v82c_controlled_continuation_ab",
        "source": {
            "stream_model": str(args.source_model),
            "stream_model_file_sha256": _sha256(args.source_model),
            "stream_weights_sha256": source_weights_sha256,
            "replay_audit": str(args.replay_audit),
            "replay_audit_sha256": _sha256(args.replay_audit),
            "optimizer_state_restored": False,
        },
        "configuration": {
            "seed": args.seed,
            "epochs": epochs,
            "filters": args.filters,
            "learning_rate": args.learning_rate,
            "train_examples": train_count,
            "validation_examples": validation_count,
            "batch_size": 8,
            "steps_per_epoch": (train_count + 7) // 8,
            "replay_fraction": args.replay_fraction,
            "max_replay_per_track": args.max_replay_per_track,
            "offset_replay_masking": False,
            "per_head_weight_mass_matched": True,
            "backbone_frozen": False,
        },
        "raw_replay": summarize_replay(raw_replay),
        "clean_replay": summarize_replay(replay_pool),
        "validation_supervised_counts": validation_data[3],
        "epochs": [],
    }

    print("V8.2c source model sha256:", manifest["source"]["stream_model_file_sha256"])
    print("V8.2c source weights sha256:", source_weights_sha256)
    print("V8.2c replay audit sha256:", manifest["source"]["replay_audit_sha256"])
    print("V8.2c clean replay:", summarize_replay(replay_pool))
    print("V8.2c validation supervision:", validation_data[3])

    for epoch in range(1, epochs + 1):
        epoch_seed = args.seed + epoch * 10007
        control_examples = _draw_examples(
            train,
            count=train_count,
            seed=epoch_seed,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        replay_examples, selected_replay = _draw_replay_examples(
            train,
            replay_pool,
            count=train_count,
            seed=epoch_seed,
            replay_fraction=args.replay_fraction,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
            max_replay_per_track=args.max_replay_per_track,
        )

        control_data = _assemble_numpy(
            np,
            train,
            control_examples,
            receptive_field=receptive_field,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        replay_data = _assemble_numpy(
            np,
            train,
            replay_examples,
            receptive_field=receptive_field,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )

        # Snapshot offsets before the replay contract so the experiment can
        # assert that onset replay does not rewrite or mask them.
        offset_targets_before = {
            name: values.copy()
            for name, values in replay_data[1].items()
            if name.startswith("offset_")
        }
        offset_weights_before = {
            name: values.copy()
            for name, values in replay_data[2].items()
            if name.startswith("offset_")
        }
        _apply_onset_replay_contract(replay_data[1], replay_data[2], replay_examples)
        for name, before in offset_targets_before.items():
            if not np.array_equal(before, replay_data[1][name]):
                raise TrainingDataError(f"replay unexpectedly changed offset target {name}")
        for name, before in offset_weights_before.items():
            if not np.array_equal(before, replay_data[2][name]):
                raise TrainingDataError(f"replay unexpectedly changed offset mask {name}")

        replay_raw_mass = _weight_mass(replay_data[2])
        mass_report = _match_supervision_mass(control_data[2], replay_data[2])
        control_counts = _supervised_counts(np, control_data[2])
        replay_counts = _supervised_counts(np, replay_data[2])

        manifest["epochs"].append(
            {
                "epoch": epoch,
                "seed": epoch_seed,
                "control_supervised_counts": control_counts,
                "replay_supervised_counts": replay_counts,
                "control_weight_mass": _weight_mass(control_data[2]),
                "replay_raw_weight_mass": replay_raw_mass,
                "mass_matching": mass_report,
                "replay_summary": summarize_replay(selected_replay),
                "replay_points": [
                    {
                        "member": point.member,
                        "sample": point.sample,
                        "arrangement": point.arrangement,
                        "model_onset_score": point.model_onset_score,
                    }
                    for point in selected_replay
                ],
            }
        )

        print(f"V8.2c epoch {epoch} control counts:", control_counts)
        print(f"V8.2c epoch {epoch} replay counts:", replay_counts)
        print(f"V8.2c epoch {epoch} mass match:", mass_report)
        print(f"V8.2c epoch {epoch} replay:", summarize_replay(selected_replay))

        # Reset the stochastic seed to the same value immediately before each
        # arm fit so shuffling/dropout-style randomness cannot favor one arm.
        fit_seed = args.seed + epoch * 19001
        tf.keras.utils.set_random_seed(fit_seed)
        control_model.fit(
            control_data[0],
            control_data[1],
            sample_weight=control_data[2],
            validation_data=(validation_data[0], validation_data[1], validation_data[2]),
            initial_epoch=epoch - 1,
            epochs=epoch,
            batch_size=8,
            shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            verbose=2,
        )
        control_stream = control_model.get_layer(control_model.stream_model_name)
        control_epoch_path = _epoch_path(control_output, epoch)
        control_stream.save(control_epoch_path)
        control_stream.save(control_output)

        tf.keras.utils.set_random_seed(fit_seed)
        replay_model.fit(
            replay_data[0],
            replay_data[1],
            sample_weight=replay_data[2],
            validation_data=(validation_data[0], validation_data[1], validation_data[2]),
            initial_epoch=epoch - 1,
            epochs=epoch,
            batch_size=8,
            shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            verbose=2,
        )
        replay_stream = replay_model.get_layer(replay_model.stream_model_name)
        replay_epoch_path = _epoch_path(replay_output, epoch)
        replay_stream.save(replay_epoch_path)
        replay_stream.save(replay_output)
        print(f"saved V8.2c epoch {epoch}: control={control_epoch_path} replay={replay_epoch_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not control_output.is_file() or not replay_output.is_file():
        raise RuntimeError("V8.2c completed without both saved stream models")
    print("V8.2c manifest:", manifest_path)


if __name__ == "__main__":
    main()
