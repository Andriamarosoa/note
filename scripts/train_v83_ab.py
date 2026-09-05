"""V8.3 controlled split-task continuation: annotation control vs true-FP replay.

Both arms start from the exact same V8.1 epoch-03 stream function copied into
V8.3.  Shared dilations 1/2/4/8 stay common; later residual blocks and fusion
projections are duplicated into independent onset and offset towers.  Before
training, both V8.3 arms are numerically checked against the V8.1 source.

Sampling, optimizer, seeds, number of steps, validation set, onset replay
contract, and per-head supervision-mass matching are intentionally identical to
V8.2c so the experiment isolates the architectural split.
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
from causal_note.v82b_replay import load_replay_points, summarize_replay
from causal_note.v83_model import (
    DEFAULT_SPLIT_AFTER,
    build_v83_bag_model,
    initialize_v83_stream_from_v81,
)
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


def _functional_delta(tf, source, target) -> float:
    probe = tf.reshape(tf.linspace(-0.75, 0.75, 257), (1, 257, 1))
    source_outputs = source(probe, training=False)
    target_outputs = target(probe, training=False)
    deltas = []
    for name in (
        "onset_presence",
        "offset_presence",
        "onset_multiplicity",
        "offset_multiplicity",
    ):
        delta = tf.reduce_max(tf.abs(source_outputs[name] - target_outputs[name]))
        deltas.append(float(delta.numpy()))
    return max(deltas)


def _initialize_from_source(
    tf,
    source_model: Path,
    *,
    filters: int,
    split_after: int,
    learning_rate: float,
):
    source = tf.keras.models.load_model(source_model, compile=False)
    control = build_v83_bag_model(filters=filters, split_after=split_after)
    replay = build_v83_bag_model(filters=filters, split_after=split_after)
    control_stream = control.get_layer(control.stream_model_name)
    replay_stream = replay.get_layer(replay.stream_model_name)

    try:
        initialize_v83_stream_from_v81(source, control_stream, split_after=split_after)
        initialize_v83_stream_from_v81(source, replay_stream, split_after=split_after)
    except (KeyError, ValueError) as exc:
        raise TrainingDataError(
            "source V8.1 checkpoint is incompatible with requested V8.3 split"
        ) from exc

    control_delta = _functional_delta(tf, source, control_stream)
    replay_delta = _functional_delta(tf, source, replay_stream)
    if control_delta > 1e-6 or replay_delta > 1e-6:
        raise TrainingDataError(
            f"V8.3 initialization changed the V8.1 function: control={control_delta} replay={replay_delta}"
        )
    control_digest = _weights_digest(control_stream)
    replay_digest = _weights_digest(replay_stream)
    if control_digest != replay_digest:
        raise TrainingDataError("V8.3 A/B arms did not initialize identically")

    _compile_model(tf, control, learning_rate)
    _compile_model(tf, replay, learning_rate)
    return {
        "source_weights_sha256": _weights_digest(source),
        "v83_initial_weights_sha256": control_digest,
        "functional_max_abs_delta": max(control_delta, replay_delta),
        "source_parameters": int(source.count_params()),
        "v83_parameters": int(control_stream.count_params()),
    }, control, replay


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="V8.3 split onset/offset controlled continuation A/B."
    )
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model" / "v83-ab")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--split-after", type=int, default=DEFAULT_SPLIT_AFTER)
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
    if args.split_after <= 0 or args.split_after >= 10:
        raise TrainingDataError("split-after must be between 1 and 9 for the default V8.1 topology")
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
    initialization, control_model, replay_model = _initialize_from_source(
        tf,
        args.source_model,
        filters=args.filters,
        split_after=args.split_after,
        learning_rate=args.learning_rate,
    )
    receptive_field = int(control_model.receptive_field)
    maximum_horizon = int(control_model.maximum_horizon_samples)
    if receptive_field != int(replay_model.receptive_field):
        raise TrainingDataError("V8.3 A/B receptive fields differ")

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
        "experiment": "v83_split_task_controlled_continuation_ab",
        "source": {
            "stream_model": str(args.source_model),
            "stream_model_file_sha256": _sha256(args.source_model),
            "replay_audit": str(args.replay_audit),
            "replay_audit_sha256": _sha256(args.replay_audit),
            "optimizer_state_restored": False,
            **initialization,
        },
        "architecture": {
            "split_after": args.split_after,
            "shared_dilations": [1, 2, 4, 8][: args.split_after],
            "private_dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512][args.split_after :],
            "function_preserving_initialization": True,
            "private_onset_offset_towers": True,
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

    print("V8.3 source model sha256:", manifest["source"]["stream_model_file_sha256"])
    print("V8.3 source weights sha256:", initialization["source_weights_sha256"])
    print("V8.3 initial weights sha256:", initialization["v83_initial_weights_sha256"])
    print("V8.3 functional max abs delta:", initialization["functional_max_abs_delta"])
    print("V8.3 parameter counts:", initialization["source_parameters"], "->", initialization["v83_parameters"])
    print("V8.3 clean replay:", summarize_replay(replay_pool))

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
                raise TrainingDataError(f"V8.3 replay changed offset target {name}")
        for name, before in offset_weights_before.items():
            if not np.array_equal(before, replay_data[2][name]):
                raise TrainingDataError(f"V8.3 replay changed offset mask {name}")

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

        print(f"V8.3 epoch {epoch} control counts:", control_counts)
        print(f"V8.3 epoch {epoch} replay counts:", replay_counts)
        print(f"V8.3 epoch {epoch} mass match:", mass_report)
        print(f"V8.3 epoch {epoch} replay:", summarize_replay(selected_replay))

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
        print(f"saved V8.3 epoch {epoch}: control={control_epoch_path} replay={replay_epoch_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not control_output.is_file() or not replay_output.is_file():
        raise RuntimeError("V8.3 completed without both saved stream models")
    print("V8.3 manifest:", manifest_path)


if __name__ == "__main__":
    main()
