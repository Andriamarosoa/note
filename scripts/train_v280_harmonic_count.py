"""V28.0-B causal harmonic evidence count model.

This module builds the compact student proposed in the V28 research report.
It consumes only the causal cluster feature map from
``causal_note.v280_causal_cqt`` and emits an exact categorical distribution for
K=0..6.  Candidate identity/ranking is intentionally absent and remains frozen
outside this count expert.

Only the ``smoke`` command is exposed in V28.0-B.  It uses a deterministic
synthetic mini-batch, never indexes GuitarSet, and never opens an outer fold.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.v280_causal_cqt import (
    CausalCQTConfig,
    harmonic_offsets,
)


MODEL_KEY = "v280_chec_direct"
SEED = 28031
STRING_COUNT = 6
FRETS_PER_STRING = 20
CARDINALITY_CLASSES = STRING_COUNT + 1
STANDARD_TUNING_MIDI = (40, 45, 50, 55, 59, 64)
FRONT_CHANNELS = 8
TRUNK_CHANNELS = 32
TRUNK_BLOCKS = 3
STRING_HIDDEN = 16
GLOBAL_HIDDEN = 64
LOSS_WEIGHTS = {
    "cardinality": 1.00,
    "string_birth": 0.25,
    "string_fret_onset": 0.15,
    "pitch_onset": 0.10,
    "poibin_cardinality": 0.10,
}


class V280ModelError(ValueError):
    pass


def expected_parameter_count(*, harmonic: bool = True, max_harmonic_order: int = 8) -> int:
    """Analytical trainable-parameter count, usable before TensorFlow is installed."""
    if isinstance(max_harmonic_order, bool) or not isinstance(max_harmonic_order, int):
        raise V280ModelError("max_harmonic_order must be an integer")
    if max_harmonic_order < 2:
        raise V280ModelError("max_harmonic_order must be >= 2")

    def conv(kernel_t, kernel_f, in_channels, out_channels):
        return kernel_t * kernel_f * in_channels * out_channels + out_channels

    parameters = 0
    # Two frontend residual blocks, with a projection in the first.
    parameters += conv(1, 1, 3, FRONT_CHANNELS)
    parameters += conv(5, 5, 3, FRONT_CHANNELS)
    parameters += conv(5, 5, FRONT_CHANNELS, FRONT_CHANNELS)
    parameters += 2 * conv(5, 5, FRONT_CHANNELS, FRONT_CHANNELS)
    parameters += conv(1, 3, FRONT_CHANNELS, TRUNK_CHANNELS)
    harmonic_copies = 2 * (max_harmonic_order - 1)
    for _ in range(TRUNK_BLOCKS):
        if harmonic:
            parameters += conv(1, 1, TRUNK_CHANNELS * harmonic_copies, TRUNK_CHANNELS)
        parameters += 2 * conv(3, 3, TRUNK_CHANNELS, TRUNK_CHANNELS)
    # Pitch, string/fret, string, and global cardinality heads.
    parameters += TRUNK_CHANNELS + 1
    parameters += (TRUNK_CHANNELS + 2) * STRING_HIDDEN + STRING_HIDDEN
    parameters += STRING_HIDDEN + 1
    parameters += (STRING_HIDDEN + 2) * STRING_HIDDEN + STRING_HIDDEN
    parameters += STRING_HIDDEN + 1
    parameters += (2 * TRUNK_CHANNELS) * GLOBAL_HIDDEN + GLOBAL_HIDDEN
    parameters += GLOBAL_HIDDEN * CARDINALITY_CLASSES + CARDINALITY_CLASSES
    return int(parameters)


def guitar_pitch_count(config: CausalCQTConfig = CausalCQTConfig()) -> int:
    low = int(round(config.input_min_midi))
    high = int(round(config.output_max_midi))
    if not math.isclose(config.input_min_midi, low, abs_tol=1e-9):
        raise V280ModelError("input_min_midi must be an integer after frequency downsampling")
    if not math.isclose(config.output_max_midi, high, abs_tol=1e-9):
        raise V280ModelError("output_max_midi must be an integer after frequency downsampling")
    return high - low + 1


def string_fret_pitch_indices(config: CausalCQTConfig = CausalCQTConfig()) -> np.ndarray:
    """Map each physical string/fret position to the one-semitone pitch axis."""
    low = int(round(config.input_min_midi))
    pitches = np.asarray(STANDARD_TUNING_MIDI, dtype=np.int32)[:, None] + np.arange(
        FRETS_PER_STRING, dtype=np.int32
    )[None, :]
    indices = pitches - low
    if np.any(indices < 0) or np.any(indices >= guitar_pitch_count(config)):
        raise V280ModelError("configured output pitch range does not cover the guitar grid")
    return indices


def poisson_binomial_numpy(string_probability: np.ndarray) -> np.ndarray:
    """Exact Poisson-binomial distribution for six string-birth probabilities."""
    probability = np.asarray(string_probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != STRING_COUNT:
        raise V280ModelError("string probabilities must have shape (rows, 6)")
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise V280ModelError("string probabilities must be finite and in [0, 1]")
    result = np.zeros((len(probability), CARDINALITY_CLASSES), dtype=np.float64)
    result[:, 0] = 1.0
    for slot in range(STRING_COUNT):
        q = probability[:, slot : slot + 1]
        stay = result * (1.0 - q)
        born = np.concatenate((np.zeros((len(result), 1)), result[:, :-1]), axis=1) * q
        result = stay + born
    return result


def residual_cardinality_numpy(
    poisson_binomial: np.ndarray,
    residual_logits: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Combine interpretable string evidence with a global categorical residual."""
    base = np.asarray(poisson_binomial, dtype=np.float64)
    residual = np.asarray(residual_logits, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != CARDINALITY_CLASSES or residual.shape != base.shape:
        raise V280ModelError("Poisson-binomial and residual arrays must both have shape (rows, 7)")
    if not np.isfinite(base).all() or not np.isfinite(residual).all() or np.any(base < 0.0):
        raise V280ModelError("invalid cardinality inputs")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise V280ModelError("epsilon must be positive and finite")
    logits = np.log(np.maximum(base, epsilon)) + residual
    logits -= np.max(logits, axis=1, keepdims=True)
    unnormalized = np.exp(logits)
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)


def _poisson_binomial_tf(tf, string_probability):
    rows = tf.shape(string_probability)[0]
    result = tf.concat(
        [tf.ones((rows, 1), dtype=tf.float32), tf.zeros((rows, STRING_COUNT), dtype=tf.float32)],
        axis=1,
    )
    for slot in range(STRING_COUNT):
        q = tf.cast(string_probability[:, slot : slot + 1], tf.float32)
        stay = result * (1.0 - q)
        born = tf.concat([tf.zeros((rows, 1), dtype=tf.float32), result[:, :-1]], axis=1) * q
        result = stay + born
    return result


def _harmonic_shift_stack_tf(tf, values, offsets: np.ndarray):
    bins = values.shape[2]
    if bins is None:
        raise V280ModelError("harmonic convolution requires a static frequency dimension")
    copies = []
    for raw_offset in offsets:
        coordinates = tf.cast(tf.range(bins), tf.float32) + float(raw_offset)
        low = tf.cast(tf.floor(coordinates), tf.int32)
        high = low + 1
        alpha = coordinates - tf.cast(low, tf.float32)
        valid_low = tf.cast((low >= 0) & (low < bins), tf.float32)
        valid_high = tf.cast((high >= 0) & (high < bins), tf.float32)
        low_value = tf.gather(values, tf.clip_by_value(low, 0, bins - 1), axis=2)
        high_value = tf.gather(values, tf.clip_by_value(high, 0, bins - 1), axis=2)
        low_weight = tf.reshape((1.0 - alpha) * valid_low, (1, 1, bins, 1))
        high_weight = tf.reshape(alpha * valid_high, (1, 1, bins, 1))
        copies.append(low_value * low_weight + high_value * high_weight)
    return tf.concat(copies, axis=-1)


def _frequency_instance_norm(tf, keras, values, name: str):
    """Normalize each time/channel slice without mixing future frames."""
    def normalize(x):
        mean, variance = tf.nn.moments(tf.cast(x, tf.float32), axes=(2,), keepdims=True)
        return tf.cast((tf.cast(x, tf.float32) - mean) * tf.math.rsqrt(variance + 1e-5), x.dtype)

    return keras.layers.Lambda(normalize, name=name)(values)


def _causal_conv2d(keras, values, channels: int, kernel: tuple[int, int], name: str):
    time_kernel, frequency_kernel = kernel
    left_frequency = (frequency_kernel - 1) // 2
    right_frequency = frequency_kernel - 1 - left_frequency
    padded = keras.layers.ZeroPadding2D(
        padding=((time_kernel - 1, 0), (left_frequency, right_frequency)),
        name=name + "_causal_pad",
    )(values)
    return keras.layers.Conv2D(
        channels,
        kernel,
        padding="valid",
        use_bias=True,
        name=name,
    )(padded)


def _residual_block(tf, keras, values, channels: int, kernel: tuple[int, int], name: str):
    shortcut = values
    if values.shape[-1] != channels:
        shortcut = keras.layers.Conv2D(channels, (1, 1), padding="valid", name=name + "_projection")(
            shortcut
        )
    hidden = _causal_conv2d(keras, values, channels, kernel, name + "_conv1")
    hidden = _frequency_instance_norm(tf, keras, hidden, name + "_norm1")
    hidden = keras.layers.Activation("relu", name=name + "_relu1")(hidden)
    hidden = _causal_conv2d(keras, hidden, channels, kernel, name + "_conv2")
    hidden = _frequency_instance_norm(tf, keras, hidden, name + "_norm2")
    hidden = keras.layers.Add(name=name + "_add")([shortcut, hidden])
    return keras.layers.Activation("relu", name=name + "_relu2")(hidden)


def build_model(
    config: CausalCQTConfig = CausalCQTConfig(),
    *,
    seed: int = SEED,
    harmonic: bool = True,
):
    """Build and compile the compact V28-CHEC direct count expert."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build V28-CHEC") from exc

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    frequency_bins = len(config.center_frequencies_hz)
    pitch_bins = guitar_pitch_count(config)
    string_fret_indices = string_fret_pitch_indices(config)

    feature_map = keras.Input(
        (config.cluster_frames, frequency_bins, 3),
        name="v280_cqt_cluster",
    )
    x = _frequency_instance_norm(tf, keras, feature_map, "v280_input_frequency_norm")
    x = _residual_block(tf, keras, x, FRONT_CHANNELS, (5, 5), "v280_front_block1")
    x = _residual_block(tf, keras, x, FRONT_CHANNELS, (5, 5), "v280_front_block2")
    x = keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1)), name="v280_frequency_stride_pad")(x)
    x = keras.layers.Conv2D(
        TRUNK_CHANNELS,
        (1, 3),
        strides=(1, config.bins_per_semitone),
        padding="valid",
        name="v280_frequency_stride",
    )(x)
    x = _frequency_instance_norm(tf, keras, x, "v280_frequency_stride_norm")
    x = keras.layers.Activation("relu", name="v280_frequency_stride_relu")(x)

    offsets = harmonic_offsets(12, config.max_harmonic_order)
    for block in range(TRUNK_BLOCKS):
        prefix = f"v280_trunk{block + 1}"
        if harmonic:
            shifted = keras.layers.Lambda(
                lambda z, values=offsets: _harmonic_shift_stack_tf(tf, z, values),
                name=prefix + "_harmonic_shift",
            )(x)
            aggregate = keras.layers.Conv2D(
                TRUNK_CHANNELS, (1, 1), padding="valid", name=prefix + "_harmonic_aggregate"
            )(shifted)
            aggregate = _frequency_instance_norm(tf, keras, aggregate, prefix + "_harmonic_norm")
            x = keras.layers.Add(name=prefix + "_harmonic_add")([x, aggregate])
            x = keras.layers.Activation("relu", name=prefix + "_harmonic_relu")(x)
        x = _residual_block(tf, keras, x, TRUNK_CHANNELS, (3, 3), prefix + "_resnet")

    trunk = keras.layers.Lambda(lambda z: z, name="v280_trunk")(x)
    guitar = keras.layers.Lambda(
        lambda z: z[:, :, :pitch_bins, :], name="v280_guitar_pitch_crop"
    )(trunk)
    recent_frame_count = min(
        config.cluster_frames,
        int(math.ceil(config.cluster_post_samples / config.hop_length)) + 1,
    )
    recent = keras.layers.Lambda(
        lambda z: z[:, -recent_frame_count:, :, :], name="v280_recent_cluster_frames"
    )(guitar)
    pitch_features = keras.layers.Lambda(
        lambda z: tf.reduce_max(z, axis=1), name="v280_pitch_features"
    )(recent)

    pitch_logits = keras.layers.Dense(1, name="v280_pitch_onset_logit")(pitch_features)
    pitch_onset = keras.layers.Lambda(
        lambda z: tf.squeeze(tf.math.sigmoid(z), axis=-1), name="pitch_onset"
    )(pitch_logits)

    index_constant = tf.constant(string_fret_indices, dtype=tf.int32)
    string_fret_features = keras.layers.Lambda(
        lambda z: tf.gather(z, index_constant, axis=1), name="v280_string_fret_gather"
    )(pitch_features)
    coordinates = np.zeros((STRING_COUNT, FRETS_PER_STRING, 2), dtype=np.float32)
    coordinates[:, :, 0] = np.linspace(-1.0, 1.0, STRING_COUNT)[:, None]
    coordinates[:, :, 1] = np.linspace(-1.0, 1.0, FRETS_PER_STRING)[None, :]
    coordinate_constant = tf.constant(coordinates, dtype=tf.float32)
    position_coordinates = keras.layers.Lambda(
        lambda z: tf.tile(coordinate_constant[None, :, :, :], [tf.shape(z)[0], 1, 1, 1]),
        name="v280_string_fret_coordinates",
    )(string_fret_features)
    string_fret_hidden = keras.layers.Concatenate(name="v280_string_fret_context")(
        [string_fret_features, position_coordinates]
    )
    string_fret_hidden = keras.layers.Dense(
        STRING_HIDDEN, activation="relu", name="v280_string_fret_hidden"
    )(string_fret_hidden)
    string_fret_logit = keras.layers.Dense(1, name="v280_string_fret_logit")(string_fret_hidden)
    string_fret_onset = keras.layers.Lambda(
        lambda z: tf.squeeze(tf.math.sigmoid(z), axis=-1), name="string_fret_onset"
    )(string_fret_logit)

    string_features = keras.layers.Lambda(
        lambda z: tf.reduce_max(z, axis=2), name="v280_string_features"
    )(string_fret_hidden)
    string_map_max = keras.layers.Lambda(
        lambda z: tf.reduce_max(z, axis=2, keepdims=True), name="v280_string_map_max"
    )(string_fret_onset)
    string_map_mean = keras.layers.Lambda(
        lambda z: tf.reduce_mean(z, axis=2, keepdims=True), name="v280_string_map_mean"
    )(string_fret_onset)
    string_context = keras.layers.Concatenate(name="v280_string_context")(
        [string_features, string_map_max, string_map_mean]
    )
    string_hidden = keras.layers.Dense(STRING_HIDDEN, activation="relu", name="v280_string_hidden")(
        string_context
    )
    string_logit = keras.layers.Dense(1, name="v280_string_birth_logit")(string_hidden)
    string_birth = keras.layers.Lambda(
        lambda z: tf.squeeze(tf.math.sigmoid(z), axis=-1), name="string_birth"
    )(string_logit)

    poisson_binomial = keras.layers.Lambda(
        lambda z: _poisson_binomial_tf(tf, z), name="poibin_cardinality"
    )(string_birth)
    global_average = keras.layers.GlobalAveragePooling2D(name="v280_global_average")(trunk)
    global_maximum = keras.layers.GlobalMaxPooling2D(name="v280_global_maximum")(trunk)
    global_context = keras.layers.Concatenate(name="v280_global_context")(
        [global_average, global_maximum]
    )
    global_context = keras.layers.Dense(
        GLOBAL_HIDDEN, activation="relu", name="v280_global_hidden"
    )(global_context)
    residual_logits = keras.layers.Dense(
        CARDINALITY_CLASSES,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="v280_cardinality_residual_logits",
    )(global_context)
    cardinality = keras.layers.Lambda(
        lambda z: tf.nn.softmax(tf.math.log(tf.maximum(z[0], 1e-7)) + z[1], axis=-1),
        name="cardinality",
    )([poisson_binomial, residual_logits])

    model = keras.Model(
        feature_map,
        {
            "cardinality": cardinality,
            "string_birth": string_birth,
            "string_fret_onset": string_fret_onset,
            "pitch_onset": pitch_onset,
            "poibin_cardinality": poisson_binomial,
        },
        name=MODEL_KEY + ("_harmonic" if harmonic else "_ablation"),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss={
            "cardinality": "sparse_categorical_crossentropy",
            "string_birth": "binary_crossentropy",
            "string_fret_onset": "binary_crossentropy",
            "pitch_onset": "binary_crossentropy",
            "poibin_cardinality": "sparse_categorical_crossentropy",
        },
        loss_weights=LOSS_WEIGHTS,
    )
    expected = expected_parameter_count(
        harmonic=harmonic,
        max_harmonic_order=config.max_harmonic_order,
    )
    if model.count_params() != expected:
        raise V280ModelError(
            f"unexpected V28 parameter count: graph={model.count_params()} expected={expected}"
        )
    return model


def synthetic_targets(rows: int, config: CausalCQTConfig, seed: int):
    """Create internally consistent mini-batch labels for graph smoke tests."""
    if rows < CARDINALITY_CLASSES:
        raise V280ModelError("synthetic smoke needs at least seven rows")
    rng = np.random.default_rng(seed)
    k = np.arange(rows, dtype=np.int32) % CARDINALITY_CLASSES
    strings = np.zeros((rows, STRING_COUNT), dtype=np.float32)
    string_fret = np.zeros((rows, STRING_COUNT, FRETS_PER_STRING), dtype=np.float32)
    pitch = np.zeros((rows, guitar_pitch_count(config)), dtype=np.float32)
    pitch_indices = string_fret_pitch_indices(config)
    for row, count in enumerate(k):
        selected = rng.choice(STRING_COUNT, size=int(count), replace=False)
        for string in selected:
            fret = int(rng.integers(0, FRETS_PER_STRING))
            strings[row, string] = 1.0
            string_fret[row, string, fret] = 1.0
            pitch[row, pitch_indices[string, fret]] = 1.0
    return {
        "cardinality": k,
        "string_birth": strings,
        "string_fret_onset": string_fret,
        "pitch_onset": pitch,
        "poibin_cardinality": k,
    }


def smoke(output_dir: Path, steps: int, rows: int, seed: int) -> dict:
    """Run a bounded, dataset-free graph/gradient/mini-overfit smoke."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if not 1 <= steps <= 200:
        raise V280ModelError("steps must be in [1, 200]")
    if not CARDINALITY_CLASSES <= rows <= 128:
        raise V280ModelError("rows must be in [7, 128]")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for the V28 smoke") from exc

    config = CausalCQTConfig()
    rng = np.random.default_rng(seed)
    features = np.abs(
        rng.normal(
            0.0,
            0.3,
            (rows, config.cluster_frames, len(config.center_frequencies_hz), 3),
        )
    ).astype(np.float32)
    targets = synthetic_targets(rows, config, seed + 1)
    model = build_model(config, seed=seed, harmonic=True)
    prediction = model(features, training=False)
    for name in ("cardinality", "poibin_cardinality"):
        probability = np.asarray(prediction[name])
        if not np.isfinite(probability).all():
            raise V280ModelError(f"non-finite {name} prediction")
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    initial = model.test_on_batch(features, targets, return_dict=True)
    history = []
    for _ in range(steps):
        current = model.train_on_batch(features, targets, return_dict=True)
        history.append({key: float(value) for key, value in current.items()})
    final = model.test_on_batch(features, targets, return_dict=True)
    values = [*initial.values(), *final.values(), *(v for row in history for v in row.values())]
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise V280ModelError("non-finite V28 smoke loss")

    output_dir.mkdir(parents=True)
    weights = output_dir / "v280-chec-smoke.weights.h5"
    model.save_weights(weights)
    report = {
        "schema_version": 1,
        "experiment": "v280_chec_synthetic_smoke",
        "dataset_indexed": False,
        "outer_fold_opened": False,
        "historical_validation_or_locked12_evaluated": False,
        "external_checkpoint_loaded": False,
        "configuration_sha256": config.sha256,
        "seed": int(seed),
        "rows": int(rows),
        "steps": int(steps),
        "input_shape": list(features.shape),
        "output_shapes": {key: list(value.shape) for key, value in prediction.items()},
        "trainable_parameters": int(model.count_params()),
        "parameter_gate_under_300k": bool(model.count_params() < 300_000),
        "initial_losses": {key: float(value) for key, value in initial.items()},
        "final_losses": {key: float(value) for key, value in final.items()},
        "finite": True,
        "weights": weights.name,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    smoke_parser = subcommands.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", type=Path, required=True)
    smoke_parser.add_argument("--steps", type=int, default=32)
    smoke_parser.add_argument("--rows", type=int, default=14)
    smoke_parser.add_argument("--seed", type=int, default=SEED)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "smoke":
        smoke(args.output_dir, args.steps, args.rows, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
