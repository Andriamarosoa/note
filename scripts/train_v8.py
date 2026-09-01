"""Train NOTE V8 directly from the audited exact-point populations.

V8 deliberately does not reuse V7's dense balanced-window objective. Presence
is trained with the Exp13 live-prior importance correction. Conditional
multiplicity is trained only on true boundaries, with additional rare-count
examples that carry zero presence weight so they cannot corrupt calibration.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import argparse
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, index_guitarset
from causal_note.v8_model import build_v8_point_model, calculate_receptive_field
from causal_note.v8_sampling import (
    V8PointExample,
    causal_context_bounds,
    hierarchical_targets,
    hierarchical_weights,
)
from scripts.audit_anonymous_boundary_targets import _prepare_tracks
from scripts.audit_exact_point_query_sampler import build_population
from scripts.audit_hard_negative_sampler import (
    _draw_fixed_candidate,
    analytical_candidate_report,
    candidate_source_counts,
)
from scripts.train_boundaries import (
    TrainingDataError,
    _configure_cpu_dependencies,
    decode_pcm16_mono_wav,
    split_tracks_by_group,
)


SMOKE_TRAIN_EXAMPLES = 96
SMOKE_VALIDATION_EXAMPLES = 48
SMOKE_FILTERS = 8


class _AudioCache:
    def __init__(self, capacity: int = 4) -> None:
        self._capacity = capacity
        self._items = OrderedDict()

    def get(self, track):
        key = (track.audio_zip, track.audio_member)
        cached = self._items.pop(key, None)
        if cached is None:
            cached = decode_pcm16_mono_wav(*key)
        self._items[key] = cached
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return cached


def _rare_targets(population):
    return tuple(
        target
        for stratum in ("onset_bearing", "offset_only")
        for target in population.positive_pools[stratum]
        if target.onset_count >= 2 or target.offset_count >= 2
    )


def _select_examples(
    population,
    *,
    split: str,
    h: int,
    seed: int,
    rare_extra: int,
):
    analytical = analytical_candidate_report(
        population,
        candidate_source_counts(h, split),
    )
    weights = analytical["importance_correction"]["weights_p_live_over_p_sample"]
    selected, _attempts = _draw_fixed_candidate(
        population,
        h=h,
        split=split,
        base_seed=seed,
    )
    examples = [
        V8PointExample(
            target.track_index,
            target.position,
            target.onset_count,
            target.offset_count,
            target.stratum,
            float(weights[target.stratum]),
            1.0,
        )
        for target in selected
    ]

    # Rare multiplicities are an independent conditional task. Their presence
    # weight is exactly zero, so this oversampling cannot change the learned
    # boundary/no-boundary prior.
    rare = _rare_targets(population)
    if rare and rare_extra:
        rng = random.Random(seed * 1009 + 97)
        for _ in range(rare_extra):
            target = rng.choice(rare)
            examples.append(
                V8PointExample(
                    target.track_index,
                    target.position,
                    target.onset_count,
                    target.offset_count,
                    "rare_multiplicity_extra",
                    0.0,
                    1.0,
                )
            )
    return tuple(examples)


def _deterministic_subset(examples, limit: int):
    """Select a stable spread over an ordered candidate list for smoke tests."""

    values = tuple(examples)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be an integer > 0")
    if len(values) <= limit:
        return values
    if limit == 1:
        return (values[len(values) // 2],)
    indices = tuple(
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    )
    if len(set(indices)) != limit:
        raise AssertionError("smoke subset indices unexpectedly collided")
    return tuple(values[index] for index in indices)


def _assemble_numpy(np, population, examples, receptive_field: int):
    cache = _AudioCache()
    count = len(examples)
    audio = np.zeros((count, receptive_field, 1), dtype=np.float32)
    targets = {
        "onset_presence": np.zeros((count, 1), dtype=np.float32),
        "offset_presence": np.zeros((count, 1), dtype=np.float32),
        "onset_multiplicity": np.zeros((count,), dtype=np.int32),
        "offset_multiplicity": np.zeros((count,), dtype=np.int32),
    }
    weights = {name: np.zeros((count,), dtype=np.float32) for name in targets}

    for row, example in enumerate(examples):
        item = population.tracks[example.track_index]
        decoded = cache.get(item.track)
        start, end, left_padding = causal_context_bounds(
            example.position,
            receptive_field,
        )
        integer = decoded.samples[start:end]
        expected = receptive_field - left_padding
        if len(integer) != expected:
            raise TrainingDataError("V8 point context length changed")
        audio[row, left_padding:, 0] = (
            np.asarray(integer, dtype=np.float32) / 32768.0
        )

        labels = hierarchical_targets(example)
        row_weights = hierarchical_weights(example)
        targets["onset_presence"][row, 0] = labels["onset_presence"]
        targets["offset_presence"][row, 0] = labels["offset_presence"]
        targets["onset_multiplicity"][row] = labels["onset_multiplicity"]
        targets["offset_multiplicity"][row] = labels["offset_multiplicity"]
        for name in weights:
            weights[name][row] = row_weights[name]

    return audio, targets, weights


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train V8 anonymous hierarchical exact-point detector."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "GuitarSet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model" / "causal-boundaries-v8.keras",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hard-negative-h",
        type=int,
        choices=(1, 2, 4),
        default=4,
        help="Exp13 distance-band intensity; V8 chooses h=4 for margin.",
    )
    parser.add_argument(
        "--rare-extra-train",
        type=int,
        default=128,
        help="extra conditional multiplicity examples per epoch",
    )
    parser.add_argument(
        "--rare-extra-validation",
        type=int,
        default=0,
        help="keep validation distribution untouched by default",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None):
    args = create_argument_parser().parse_args(argv)
    if args.epochs <= 0 or args.filters <= 0 or args.learning_rate <= 0:
        raise TrainingDataError("epochs, filters and learning rate must be positive")
    if args.rare_extra_train < 0 or args.rare_extra_validation < 0:
        raise TrainingDataError("rare-extra counts must be >= 0")

    indexed = tuple(
        track
        for track in index_guitarset(args.dataset_dir)
        if track.player_id in ALLOWED_PLAYERS
    )
    train_tracks, validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=0.2,
        seed=args.seed,
    )
    train_population = build_population(_prepare_tracks(train_tracks))
    validation_population = build_population(_prepare_tracks(validation_tracks))

    np, tf = _configure_cpu_dependencies(args.seed)
    receptive_field = calculate_receptive_field()
    model_filters = min(args.filters, SMOKE_FILTERS) if args.smoke else args.filters
    model = build_v8_point_model(filters=model_filters)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss={
            "onset_presence": tf.keras.losses.BinaryCrossentropy(),
            "offset_presence": tf.keras.losses.BinaryCrossentropy(),
            "onset_multiplicity": tf.keras.losses.SparseCategoricalCrossentropy(),
            "offset_multiplicity": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "onset_presence": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
            "offset_presence": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        },
    )

    validation_examples = _select_examples(
        validation_population,
        split="validation",
        h=args.hard_negative_h,
        seed=args.seed + 1,
        rare_extra=args.rare_extra_validation,
    )
    if args.smoke:
        validation_examples = _deterministic_subset(
            validation_examples,
            SMOKE_VALIDATION_EXAMPLES,
        )
    validation_data = _assemble_numpy(
        np,
        validation_population,
        validation_examples,
        receptive_field,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_point_path = args.output.parent / f"{args.output.stem}.point.best.keras"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 3

    epochs = 1 if args.smoke else args.epochs
    for epoch in range(epochs):
        train_examples = _select_examples(
            train_population,
            split="train",
            h=args.hard_negative_h,
            seed=args.seed + epoch * 10007,
            rare_extra=(
                min(args.rare_extra_train, 16)
                if args.smoke
                else args.rare_extra_train
            ),
        )
        if args.smoke:
            train_examples = _deterministic_subset(
                train_examples,
                SMOKE_TRAIN_EXAMPLES,
            )
            print(
                "V8 data smoke:",
                f"train_examples={len(train_examples)}",
                f"validation_examples={len(validation_examples)}",
                f"filters={model_filters}",
                f"receptive_field={receptive_field}",
            )
        train_data = _assemble_numpy(
            np,
            train_population,
            train_examples,
            receptive_field,
        )
        history = model.fit(
            train_data[0],
            train_data[1],
            sample_weight=train_data[2],
            validation_data=validation_data,
            initial_epoch=epoch,
            epochs=epoch + 1,
            batch_size=8,
            shuffle=True,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            verbose=2,
        )
        val_loss = float(history.history["val_loss"][-1])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save(best_point_path)
            model.get_layer(model.stream_model_name).save(args.output)
            print(
                f"new best V8 epoch {epoch + 1}: "
                f"val_loss={val_loss:.8f}"
            )
        else:
            epochs_without_improvement += 1
            if not args.smoke and epochs_without_improvement >= patience:
                print(
                    f"early stop after {patience} epochs without "
                    "validation improvement"
                )
                break

    if not args.output.is_file():
        raise RuntimeError("V8 training completed without a saved stream model")
    print(f"best V8 stream model: {args.output}")
    print(f"best V8 point model: {best_point_path}")


if __name__ == "__main__":
    main()
