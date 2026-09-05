"""V8.4 controlled experiment with fully independent task pipelines.

The audit after V8.3 found two distinct leaks:
1. onset-selected replay crops also changed the data distribution seen by offset;
2. onset replay gradients still changed shared transient/stem parameters used by offset.

V8.4 removes both.  Three independently initialized V8.1 bag models are
continued from the same frozen V8.1 epoch-03 stream:
- onset_control: normal V8.1 sampler, onset losses only;
- onset_replay: true-FP replay sampler, onset losses only;
- offset: independent normal V8.1 sampler, offset losses only.

The same trained offset stream is assembled with both onset arms at every epoch,
so control/replay offset weights are byte-identical by construction.  The final
deployable V8.4 model contains two complete causal streams and no shared
trainable variables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, index_guitarset
from causal_note.v81_model import build_v81_bag_model
from causal_note.v82b_replay import load_replay_points, summarize_replay
from causal_note.v84_model import assemble_v84_stream
from scripts.audit_anonymous_boundary_targets import _prepare_tracks
from scripts.train_boundaries import TrainingDataError, _configure_cpu_dependencies, split_tracks_by_group
from scripts.train_v81 import (
    SMOKE_TRAIN_EXAMPLES,
    SMOKE_VALIDATION_EXAMPLES,
    _assemble_numpy,
    _draw_examples,
    _epoch_path,
    _prepare_burst_tracks,
)
from scripts.train_v82b import _clean_replay_pool
from scripts.train_v82c_ab import (
    DEFAULT_MAX_REPLAY_PER_TRACK,
    DEFAULT_NEGATIVE_MARGIN,
    DEFAULT_REPLAY_FRACTION,
    DEFAULT_TRAIN_EXAMPLES,
    DEFAULT_VALIDATION_EXAMPLES,
    _apply_onset_replay_contract,
    _compile_model,
    _draw_replay_examples,
    _match_supervision_mass,
    _sha256,
    _supervised_counts,
    _weight_mass,
    _weights_digest,
)


def _mask_task(weights, kind: str) -> None:
    other = "offset" if kind == "onset" else "onset"
    for name, values in weights.items():
        if name.startswith(other + "_"):
            values[:] = 0.0


def _initialize_single_task_models(tf, source_path: Path, *, filters: int, learning_rate: float):
    source = tf.keras.models.load_model(source_path, compile=False)
    models = {}
    for name in ("onset_control", "onset_replay", "offset"):
        bag = build_v81_bag_model(filters=filters)
        stream = bag.get_layer(bag.stream_model_name)
        stream.set_weights(source.get_weights())
        if _weights_digest(stream) != _weights_digest(source):
            raise TrainingDataError(f"{name} failed exact V8.1 initialization")
        _compile_model(tf, bag, learning_rate)
        models[name] = bag
    return source, models


def _save_composites(tf, models, output_dir: Path, epoch: int, filters: int):
    onset_control = models["onset_control"].get_layer(models["onset_control"].stream_model_name)
    onset_replay = models["onset_replay"].get_layer(models["onset_replay"].stream_model_name)
    offset = models["offset"].get_layer(models["offset"].stream_model_name)

    control = assemble_v84_stream(onset_control, offset, filters=filters)
    replay = assemble_v84_stream(onset_replay, offset, filters=filters)
    control_offset = control.get_layer("v84_offset_stream")
    replay_offset = replay.get_layer("v84_offset_stream")
    if _weights_digest(control_offset) != _weights_digest(replay_offset):
        raise TrainingDataError("V8.4 A/B composites do not share identical offset weights")

    control_path = output_dir / f"control.epoch-{epoch:02d}.keras"
    replay_path = output_dir / f"replay.epoch-{epoch:02d}.keras"
    control.save(control_path)
    replay.save(replay_path)
    control.save(output_dir / "control.keras")
    replay.save(output_dir / "replay.keras")
    return control_path, replay_path, _weights_digest(offset)


def create_argument_parser():
    parser = argparse.ArgumentParser(description="V8.4 independent onset/offset controlled continuation A/B")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model" / "v84-ab")
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
    if not args.source_model.is_file() or not args.replay_audit.is_file():
        raise TrainingDataError("source model and replay audit must exist")

    indexed = tuple(track for track in index_guitarset(args.dataset_dir) if track.player_id in ALLOWED_PLAYERS)
    train_tracks, validation_tracks = split_tracks_by_group(indexed, validation_fraction=0.2, seed=args.seed)
    train = _prepare_burst_tracks(_prepare_tracks(train_tracks))
    validation = _prepare_burst_tracks(_prepare_tracks(validation_tracks))

    np, tf = _configure_cpu_dependencies(args.seed)
    source, models = _initialize_single_task_models(
        tf, args.source_model, filters=args.filters, learning_rate=args.learning_rate
    )
    receptive_field = int(models["offset"].receptive_field)
    maximum_horizon = int(models["offset"].maximum_horizon_samples)

    raw_replay = load_replay_points(args.replay_audit)
    replay_pool = _clean_replay_pool(
        train,
        raw_replay,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )

    validation_count = min(args.validation_examples, SMOKE_VALIDATION_EXAMPLES) if args.smoke else args.validation_examples
    validation_examples = _draw_examples(
        validation,
        count=validation_count,
        seed=args.seed + 991,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )
    validation_base = _assemble_numpy(
        np, validation, validation_examples,
        receptive_field=receptive_field,
        maximum_horizon=maximum_horizon,
        negative_margin=args.negative_margin,
    )
    validation_onset_weights = {name: values.copy() for name, values in validation_base[2].items()}
    validation_offset_weights = {name: values.copy() for name, values in validation_base[2].items()}
    _mask_task(validation_onset_weights, "onset")
    _mask_task(validation_offset_weights, "offset")

    epochs = 1 if args.smoke else args.epochs
    train_count = min(args.train_examples, SMOKE_TRAIN_EXAMPLES) if args.smoke else args.train_examples
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "experiment": "v84_fully_independent_task_pipelines_ab",
        "source": {
            "stream_model": str(args.source_model),
            "stream_model_file_sha256": _sha256(args.source_model),
            "stream_weights_sha256": _weights_digest(source),
            "replay_audit": str(args.replay_audit),
            "replay_audit_sha256": _sha256(args.replay_audit),
            "optimizer_state_restored": False,
        },
        "architecture": {
            "shared_trainable_variables": 0,
            "separate_complete_onset_stream": True,
            "separate_complete_offset_stream": True,
            "same_offset_stream_in_control_and_replay": True,
            "source_parameters_per_stream": int(source.count_params()),
            "deployed_parameters": int(source.count_params()) * 2,
        },
        "configuration": {
            "seed": args.seed,
            "epochs": epochs,
            "filters": args.filters,
            "learning_rate": args.learning_rate,
            "train_examples_per_task": train_count,
            "validation_examples": validation_count,
            "batch_size": 8,
            "steps_per_epoch_per_task": (train_count + 7) // 8,
            "replay_fraction": args.replay_fraction,
            "max_replay_per_track": args.max_replay_per_track,
            "onset_offset_samplers_independent": True,
            "onset_replay_can_update_offset": False,
            "onset_replay_can_select_offset_crops": False,
        },
        "clean_replay": summarize_replay(replay_pool),
        "epochs": [],
    }

    for epoch in range(1, epochs + 1):
        epoch_seed = args.seed + epoch * 10007
        onset_control_examples = _draw_examples(
            train, count=train_count, seed=epoch_seed,
            maximum_horizon=maximum_horizon, negative_margin=args.negative_margin,
        )
        onset_replay_examples, selected_replay = _draw_replay_examples(
            train, replay_pool, count=train_count, seed=epoch_seed,
            replay_fraction=args.replay_fraction,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
            max_replay_per_track=args.max_replay_per_track,
        )
        # Deliberately independent seed. Offset no longer inherits onset crop selection.
        offset_examples = _draw_examples(
            train, count=train_count, seed=epoch_seed + 7919,
            maximum_horizon=maximum_horizon, negative_margin=args.negative_margin,
        )

        onset_control_data = _assemble_numpy(
            np, train, onset_control_examples,
            receptive_field=receptive_field, maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        onset_replay_data = _assemble_numpy(
            np, train, onset_replay_examples,
            receptive_field=receptive_field, maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        offset_data = _assemble_numpy(
            np, train, offset_examples,
            receptive_field=receptive_field, maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        _apply_onset_replay_contract(onset_replay_data[1], onset_replay_data[2], onset_replay_examples)
        _mask_task(onset_control_data[2], "onset")
        _mask_task(onset_replay_data[2], "onset")
        _mask_task(offset_data[2], "offset")

        onset_mass_match = _match_supervision_mass(onset_control_data[2], onset_replay_data[2])
        offset_digest_before = _weights_digest(models["offset"].get_layer(models["offset"].stream_model_name))

        fit_seed = args.seed + epoch * 19001
        tf.keras.utils.set_random_seed(fit_seed)
        models["onset_control"].fit(
            onset_control_data[0], onset_control_data[1], sample_weight=onset_control_data[2],
            validation_data=(validation_base[0], validation_base[1], validation_onset_weights),
            initial_epoch=epoch - 1, epochs=epoch, batch_size=8, shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()], verbose=2,
        )
        tf.keras.utils.set_random_seed(fit_seed)
        models["onset_replay"].fit(
            onset_replay_data[0], onset_replay_data[1], sample_weight=onset_replay_data[2],
            validation_data=(validation_base[0], validation_base[1], validation_onset_weights),
            initial_epoch=epoch - 1, epochs=epoch, batch_size=8, shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()], verbose=2,
        )
        tf.keras.utils.set_random_seed(fit_seed + 31337)
        models["offset"].fit(
            offset_data[0], offset_data[1], sample_weight=offset_data[2],
            validation_data=(validation_base[0], validation_base[1], validation_offset_weights),
            initial_epoch=epoch - 1, epochs=epoch, batch_size=8, shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()], verbose=2,
        )

        control_path, replay_path, offset_digest = _save_composites(
            tf, models, args.output_dir, epoch, args.filters
        )
        manifest["epochs"].append({
            "epoch": epoch,
            "seed": epoch_seed,
            "onset_control_counts": _supervised_counts(np, onset_control_data[2]),
            "onset_replay_counts": _supervised_counts(np, onset_replay_data[2]),
            "offset_counts": _supervised_counts(np, offset_data[2]),
            "onset_control_mass": _weight_mass(onset_control_data[2]),
            "onset_replay_mass": _weight_mass(onset_replay_data[2]),
            "offset_mass": _weight_mass(offset_data[2]),
            "onset_mass_matching": onset_mass_match,
            "replay_summary": summarize_replay(selected_replay),
            "offset_weights_before_epoch_sha256": offset_digest_before,
            "offset_weights_after_epoch_sha256": offset_digest,
            "control_model": str(control_path),
            "replay_model": str(replay_path),
        })
        print(f"V8.4 epoch {epoch} replay:", summarize_replay(selected_replay))
        print(f"V8.4 epoch {epoch} offset digest:", offset_digest)
        print(f"saved V8.4 epoch {epoch}: control={control_path} replay={replay_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V8.4 manifest:", manifest_path)


if __name__ == "__main__":
    main()
