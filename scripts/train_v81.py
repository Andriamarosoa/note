"""Train V8.1 from causal GuitarSet boundary bursts.

V8.1 is deliberately not an exact-sample classifier. A reference burst owns a
small causal response bag and the network may place one sparse pulse anywhere
inside that bag. The deployable stream network remains strictly causal.

Sampling is discriminative rather than prior-matched:
- 50% reference burst starts;
- 25% clean hard negatives whose response bag ends just before a burst;
- 25% clean background.

Ambiguous heads are masked instead of being mislabeled negative. Every epoch is
saved independently so model selection can be made from continuous-stream
metrics rather than validation loss alone.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, Sequence, Tuple

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
    BoundaryBurst,
    cluster_fixed_span,
    response_window_is_empty,
    training_context_bounds,
)
from scripts.audit_anonymous_boundary_targets import _prepare_tracks
from scripts.train_boundaries import (
    TrainingDataError,
    _configure_cpu_dependencies,
    decode_pcm16_mono_wav,
    split_tracks_by_group,
)


DEFAULT_TRAIN_EXAMPLES = 1600
DEFAULT_VALIDATION_EXAMPLES = 400
DEFAULT_NEGATIVE_MARGIN = 16
HARD_NEGATIVE_GAPS = (32, 128, 512)
SMOKE_TRAIN_EXAMPLES = 64
SMOKE_VALIDATION_EXAMPLES = 32
SMOKE_FILTERS = 8


@dataclass(frozen=True)
class BurstTrack:
    audit_track: object
    onset_positions: Tuple[int, ...]
    offset_positions: Tuple[int, ...]
    onset_bursts: Tuple[BoundaryBurst, ...]
    offset_bursts: Tuple[BoundaryBurst, ...]

    @property
    def frame_count(self) -> int:
        return int(self.audit_track.frame_count)


@dataclass(frozen=True)
class BagExample:
    track_index: int
    position: int
    stratum: str


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


def _boundary_positions(audit_track, kind: str) -> Tuple[int, ...]:
    values = []
    for notes in audit_track.slots:
        for note in notes:
            position = note.onset_sample if kind == "onset" else note.offset_sample
            if position < audit_track.frame_count:
                values.append(int(position))
    return tuple(sorted(values))


def _prepare_burst_tracks(audit_tracks: Sequence[object]) -> Tuple[BurstTrack, ...]:
    prepared = []
    for audit_track in audit_tracks:
        onset_positions = _boundary_positions(audit_track, "onset")
        offset_positions = _boundary_positions(audit_track, "offset")
        prepared.append(
            BurstTrack(
                audit_track=audit_track,
                onset_positions=onset_positions,
                offset_positions=offset_positions,
                onset_bursts=cluster_fixed_span(
                    onset_positions, DEFAULT_ONSET_HORIZON_SAMPLES
                ),
                offset_bursts=cluster_fixed_span(
                    offset_positions, DEFAULT_OFFSET_HORIZON_SAMPLES
                ),
            )
        )
    return tuple(prepared)


def _burst_maps(item: BurstTrack):
    return {
        "onset": {burst.start_sample: burst for burst in item.onset_bursts},
        "offset": {burst.start_sample: burst for burst in item.offset_bursts},
    }


def _positive_pool(tracks: Sequence[BurstTrack], maximum_horizon: int):
    result = []
    for track_index, item in enumerate(tracks):
        starts = {
            burst.start_sample for burst in item.onset_bursts
        } | {
            burst.start_sample for burst in item.offset_bursts
        }
        latest = item.frame_count - maximum_horizon - 1
        result.extend(
            BagExample(track_index, position, "positive_burst")
            for position in sorted(starts)
            if 0 <= position <= latest
        )
    if not result:
        raise TrainingDataError("V8.1 positive burst pool is empty")
    return tuple(result)


def _hard_negative_pool(tracks: Sequence[BurstTrack], maximum_horizon: int):
    result = []
    seen = set()
    for track_index, item in enumerate(tracks):
        latest = item.frame_count - maximum_horizon - 1
        for kind, horizon, bursts in (
            ("onset", DEFAULT_ONSET_HORIZON_SAMPLES, item.onset_bursts),
            ("offset", DEFAULT_OFFSET_HORIZON_SAMPLES, item.offset_bursts),
        ):
            for burst in bursts:
                for gap in HARD_NEGATIVE_GAPS:
                    position = burst.start_sample - horizon - gap
                    key = (track_index, position)
                    if 0 <= position <= latest and key not in seen:
                        seen.add(key)
                        result.append(
                            BagExample(track_index, position, f"pre_{kind}_gap_{gap}")
                        )
    if not result:
        raise TrainingDataError("V8.1 hard-negative pool is empty")
    return tuple(result)


def _head_supervision(
    item: BurstTrack,
    burst_maps,
    *,
    kind: str,
    position: int,
    negative_margin: int,
):
    if kind == "onset":
        positions = item.onset_positions
        horizon = DEFAULT_ONSET_HORIZON_SAMPLES
    else:
        positions = item.offset_positions
        horizon = DEFAULT_OFFSET_HORIZON_SAMPLES

    burst = burst_maps[kind].get(position)
    if burst is not None:
        return {
            "presence": 1.0,
            "mass": 1.0 / float(horizon + 1),
            "delay": 0.0,
            "count": burst.count_class,
            "presence_weight": 1.0,
            "mass_weight": 1.0,
            "delay_weight": 1.0,
            "count_weight": 1.0,
        }

    clean = response_window_is_empty(
        positions,
        position,
        horizon,
        margin_samples=negative_margin,
    )
    if clean:
        return {
            "presence": 0.0,
            "mass": 0.0,
            "delay": 0.0,
            "count": 0,
            "presence_weight": 1.0,
            "mass_weight": 1.0,
            "delay_weight": 0.0,
            "count_weight": 0.0,
        }

    # Another boundary occupies this response bag. This head is ambiguous at
    # the selected anchor and must not be trained as a negative.
    return {
        "presence": 0.0,
        "mass": 0.0,
        "delay": 0.0,
        "count": 0,
        "presence_weight": 0.0,
        "mass_weight": 0.0,
        "delay_weight": 0.0,
        "count_weight": 0.0,
    }


def _clean_background_example(
    tracks: Sequence[BurstTrack],
    rng: random.Random,
    maximum_horizon: int,
    negative_margin: int,
) -> BagExample:
    frame_weights = [max(1, item.frame_count - maximum_horizon) for item in tracks]
    for _ in range(20_000):
        track_index = rng.choices(range(len(tracks)), weights=frame_weights, k=1)[0]
        item = tracks[track_index]
        latest = item.frame_count - maximum_horizon - 1
        if latest < 0:
            continue
        position = rng.randint(0, latest)
        if response_window_is_empty(
            item.onset_positions,
            position,
            DEFAULT_ONSET_HORIZON_SAMPLES,
            margin_samples=negative_margin,
        ) and response_window_is_empty(
            item.offset_positions,
            position,
            DEFAULT_OFFSET_HORIZON_SAMPLES,
            margin_samples=negative_margin,
        ):
            return BagExample(track_index, position, "background")
    raise TrainingDataError("could not draw a clean V8.1 background example")


def _draw_examples(
    tracks: Sequence[BurstTrack],
    *,
    count: int,
    seed: int,
    maximum_horizon: int,
    negative_margin: int,
):
    if count < 4:
        raise TrainingDataError("V8.1 example count must be >= 4")
    rng = random.Random(seed)
    positives = _positive_pool(tracks, maximum_horizon)
    hard = _hard_negative_pool(tracks, maximum_horizon)

    positive_count = count // 2
    hard_count = count // 4
    background_count = count - positive_count - hard_count
    examples = [rng.choice(positives) for _ in range(positive_count)]
    examples.extend(rng.choice(hard) for _ in range(hard_count))
    examples.extend(
        _clean_background_example(
            tracks, rng, maximum_horizon, negative_margin
        )
        for _ in range(background_count)
    )
    rng.shuffle(examples)
    return tuple(examples)


def _assemble_numpy(
    np,
    tracks: Sequence[BurstTrack],
    examples: Sequence[BagExample],
    *,
    receptive_field: int,
    maximum_horizon: int,
    negative_margin: int,
):
    context_samples = receptive_field + maximum_horizon
    cache = _AudioCache()
    count = len(examples)
    audio = np.zeros((count, context_samples, 1), dtype=np.float32)
    targets: Dict[str, object] = {}
    weights: Dict[str, object] = {}
    for kind in ("onset", "offset"):
        targets[f"{kind}_bag_presence"] = np.zeros((count, 1), dtype=np.float32)
        targets[f"{kind}_mass"] = np.zeros((count, 1), dtype=np.float32)
        targets[f"{kind}_delay"] = np.zeros((count, 1), dtype=np.float32)
        targets[f"{kind}_count"] = np.zeros((count,), dtype=np.int32)
        for suffix in ("bag_presence", "mass", "delay", "count"):
            weights[f"{kind}_{suffix}"] = np.zeros((count,), dtype=np.float32)

    supervised_counts = {name: 0 for name in weights}
    for row, example in enumerate(examples):
        item = tracks[example.track_index]
        decoded = cache.get(item.audit_track.track)
        start, end, left_padding = training_context_bounds(
            example.position,
            receptive_field,
            maximum_horizon,
        )
        end = min(end, item.frame_count)
        integer = decoded.samples[start:end]
        write_start = left_padding
        write_end = write_start + len(integer)
        if write_end > context_samples:
            raise TrainingDataError("V8.1 causal context exceeded tensor length")
        audio[row, write_start:write_end, 0] = (
            np.asarray(integer, dtype=np.float32) / 32768.0
        )

        maps = _burst_maps(item)
        for kind in ("onset", "offset"):
            supervision = _head_supervision(
                item,
                maps,
                kind=kind,
                position=example.position,
                negative_margin=negative_margin,
            )
            targets[f"{kind}_bag_presence"][row, 0] = supervision["presence"]
            targets[f"{kind}_mass"][row, 0] = supervision["mass"]
            targets[f"{kind}_delay"][row, 0] = supervision["delay"]
            targets[f"{kind}_count"][row] = supervision["count"]
            for suffix in ("bag_presence", "mass", "delay", "count"):
                name = f"{kind}_{suffix}"
                weight = supervision[
                    "presence_weight" if suffix == "bag_presence" else f"{suffix}_weight"
                ]
                weights[name][row] = weight
                if weight:
                    supervised_counts[name] += 1

    return audio, targets, weights, supervised_counts


def _epoch_path(output: Path, epoch: int) -> Path:
    return output.with_name(f"{output.stem}.epoch-{epoch:02d}{output.suffix}")


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Train V8.1 causal burst detector.")
    parser.add_argument(
        "dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model" / "causal-boundaries-v81.keras",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument(
        "--validation-examples", type=int, default=DEFAULT_VALIDATION_EXAMPLES
    )
    parser.add_argument("--negative-margin", type=int, default=DEFAULT_NEGATIVE_MARGIN)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None):
    args = create_argument_parser().parse_args(argv)
    if args.epochs <= 0 or args.filters <= 0 or args.learning_rate <= 0:
        raise TrainingDataError("epochs, filters and learning rate must be positive")
    if args.train_examples <= 0 or args.validation_examples <= 0:
        raise TrainingDataError("example counts must be positive")
    if args.negative_margin < 0:
        raise TrainingDataError("negative margin must be >= 0")

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
            "onset_bag_presence": [
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
            "offset_bag_presence": [
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
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
    train_count = (
        min(args.train_examples, SMOKE_TRAIN_EXAMPLES)
        if args.smoke else args.train_examples
    )

    print(
        "V8.1 configuration:",
        f"receptive_field={receptive_field}",
        f"onset_horizon={DEFAULT_ONSET_HORIZON_SAMPLES}",
        f"offset_horizon={DEFAULT_OFFSET_HORIZON_SAMPLES}",
        f"context_samples={model.context_samples}",
        f"filters={filters}",
        f"train_examples={train_count}",
        f"validation_examples={validation_count}",
    )
    print("V8.1 validation supervision:", validation_data[3])

    for epoch in range(1, epochs + 1):
        train_examples = _draw_examples(
            train,
            count=train_count,
            seed=args.seed + epoch * 10007,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        train_data = _assemble_numpy(
            np,
            train,
            train_examples,
            receptive_field=receptive_field,
            maximum_horizon=maximum_horizon,
            negative_margin=args.negative_margin,
        )
        print(f"V8.1 epoch {epoch} supervision:", train_data[3])
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
        print(f"saved V8.1 stream epoch {epoch}: {epoch_path}")

    if not args.output.is_file():
        raise RuntimeError("V8.1 training completed without a saved stream model")
    print(f"latest V8.1 stream model: {args.output}")


if __name__ == "__main__":
    main()
