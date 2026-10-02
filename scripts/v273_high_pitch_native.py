"""Matched HPR-v2 count model with a zero-initialized rescue residual branch."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v250_count_only import build_model
from scripts.train_v273_window_pair import weight_hash
from scripts.v273_spectral_normalization import (
    SEED, DROPOUT_SEED, EPOCHS, CHECKPOINTS, HISTORICAL_INITIAL_HASH,
    compile_metrics, layer_hashes)
from scripts.v273_window_experiment import epoch_order, require

ARMS = ("observed_only", "with_high_pitch")
LAUNCH = Path("analysis/v273-high-pitch-launch.json")
PROTOCOL = Path("analysis/v273-high-pitch-protocol.md")


def experiment_identity():
    cfg = json.loads(LAUNCH.read_text())
    require(
        cfg["arms"] == list(ARMS)
        and cfg["seed"] == SEED
        and cfg["dropout_seed"] == DROPOUT_SEED
        and cfg["epochs"] == EPOCHS
        and cfg["primary_epoch"] == 12
        and cfg["snapshots"] == list(CHECKPOINTS),
        "frozen training settings changed",
    )
    require(
        cfg["config_sha256"] == digest("analysis/v273-native-paired-config.json"),
        "partition changed",
    )
    require(
        cfg["feature_builder_sha256"] == digest("scripts/v273_high_pitch_map.py")
        and cfg["register_builder_sha256"] == digest("scripts/v273_high_pitch_registers.py"),
        "audited feature builder changed",
    )
    require(
        cfg.get("hpr_version") == 2
        and cfg.get("semitones") == 0
        and cfg.get("source_support_unchanged") is False
        and cfg.get("rescue_zero_initialized") is True,
        "HPR-v2 launch contract changed",
    )
    return dict(
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        launch_sha256=digest(LAUNCH),
        protocol_sha256=digest(PROTOCOL),
        config_sha256=cfg["config_sha256"],
        feature_builder_sha256=cfg["feature_builder_sha256"],
        register_builder_sha256=cfg["register_builder_sha256"],
    )


def treatment_maps(features, arm):
    """The main count path receives normal audio only in both arms."""
    require(arm in ARMS, "unknown arm")
    x = np.asarray(features, np.float32)
    require(
        x.ndim == 4 and x.shape[1:] == (31, 64, 4)
        and np.isfinite(x).all() and (x >= 0).all(),
        "invalid or outer feature maps",
    )
    return x[:, :, :, [0, 0, 2, 2]]


def rescue_maps(features):
    """Dedicated compressed-context channels, never concatenated into main evidence."""
    x = np.asarray(features, np.float32)
    require(
        x.ndim == 4 and x.shape[1:] == (31, 64, 4)
        and np.isfinite(x).all() and (x >= 0).all(),
        "invalid or outer feature maps",
    )
    return x[:, :, :, [1, 3]]


def build(arm):
    """Build matched models; only a fixed zero/one gate differs between arms."""
    import tensorflow as tf
    from tensorflow import keras

    require(arm in ARMS, "unknown arm")

    scaffold = build_model(
        "categorical",
        SEED,
        time_frames=31,
        spectral_normalization="fixed_scale",
        count_dropout_seed=DROPOUT_SEED,
        ownership_context=True,
        spectral_channels=4,
    )
    hidden = scaffold.get_layer("v240_cardinality_hidden2").output
    source_head = scaffold.get_layer("cardinality")

    main_head = keras.layers.Dense(
        7, activation=None, name="hprv2_main_logits"
    )
    main_logits = main_head(hidden)

    rescue_input = keras.Input((31, 64, 2), name="rescue_map")
    rescue = keras.layers.Rescaling(
        1.0/12.0, name="hprv2_rescue_fixed_scale"
    )(rescue_input)
    rescue = keras.layers.Conv2D(
        8, (3, 3), padding="same", activation="relu",
        name="hprv2_rescue_conv1",
    )(rescue)
    rescue = keras.layers.Conv2D(
        16, (3, 3), padding="same", activation="relu",
        name="hprv2_rescue_conv2",
    )(rescue)
    rescue_avg = keras.layers.GlobalAveragePooling2D(
        name="hprv2_rescue_global_average"
    )(rescue)
    rescue_max = keras.layers.GlobalMaxPooling2D(
        name="hprv2_rescue_global_max"
    )(rescue)
    rescue = keras.layers.Concatenate(
        name="hprv2_rescue_pool"
    )([rescue_avg, rescue_max])
    rescue = keras.layers.Dense(
        32, activation="relu", name="hprv2_rescue_dense"
    )(rescue)
    rescue_logits = keras.layers.Dense(
        7,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="hprv2_rescue_logits",
    )(rescue)

    gate_value = 1.0 if arm == "with_high_pitch" else 0.0
    gated = keras.layers.Lambda(
        lambda z, value=gate_value: z*value,
        name="hprv2_rescue_gate",
    )(rescue_logits)
    fused = keras.layers.Add(name="hprv2_fused_logits")(
        [main_logits, gated]
    )
    probability = keras.layers.Softmax(
        name="hprv2_cardinality"
    )(fused)

    model = keras.Model(
        list(scaffold.inputs) + [rescue_input],
        probability,
        name="v273_hprv2_" + arm,
    )

    # Exact initial parity with the historical four-channel control head.
    main_head.set_weights(source_head.get_weights())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="sparse_categorical_crossentropy",
    )
    return model


def inputs(cache, maps, geometry, ids, arm):
    owned = np.asarray(geometry[ids, :, 1:2], np.float32)
    require(np.isfinite(owned).all(), "outer/missing geometry")
    chosen = np.asarray(maps[ids], np.float32)
    return dict(
        candidate_set=np.asarray(cache["sequence"][ids], np.float32),
        candidate_mask=np.asarray(cache["mask"][ids], np.float32),
        cluster_stats=np.asarray(cache["stats"][ids], np.float32),
        spectral_map=treatment_maps(chosen, arm),
        ownership_map=owned,
        rescue_map=rescue_maps(chosen),
    )


def batches(cache, maps, geometry, ids, arm, *, labels=None, shuffle=False, batch_size=128):
    import tensorflow as tf

    base = np.asarray(ids, np.int64)
    require(len(base) and len(np.unique(base)) == len(base), "invalid indices")

    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch = 0
            self.order = epoch_order(base, SEED, 0) if shuffle else base.copy()

        def __len__(self):
            return (len(base)+batch_size-1)//batch_size

        def __getitem__(self, index):
            selected = self.order[index*batch_size:(index+1)*batch_size]
            x = inputs(cache, maps, geometry, selected, arm)
            return x if labels is None else (x, labels[selected])

        def on_epoch_end(self):
            self.epoch += 1
            if shuffle:
                self.order = epoch_order(base, SEED, self.epoch)

    return Batches()


def preflight(output):
    import tensorflow as tf

    require(tf.__version__ == "2.15.1", "pinned TensorFlow required")
    require(not output.exists(), "preflight output exists")
    tf.config.experimental.enable_op_determinism()
    identity = experiment_identity()

    historical = build_model("categorical", SEED, time_frames=31)
    require(
        weight_hash(historical) == HISTORICAL_INITIAL_HASH,
        "historical default changed",
    )
    del historical

    reference = build_model(
        "categorical",
        SEED,
        time_frames=31,
        spectral_normalization="fixed_scale",
        count_dropout_seed=DROPOUT_SEED,
        ownership_context=True,
        spectral_channels=4,
    )
    reference_shapes = {
        t.name.split(":")[0]: tuple(t.shape.as_list())
        for t in reference.inputs
    }
    rng = np.random.default_rng(90273)
    base_x = {
        key: rng.uniform(0, 1, (4, *shape[1:])).astype(np.float32)
        for key, shape in reference_shapes.items()
    }
    base_x["candidate_mask"].fill(1)
    reference_probability = reference(base_x, training=False).numpy()
    del reference

    rescue_value = rng.uniform(0, 1, (4, 31, 64, 2)).astype(np.float32)
    previous_initial = None
    previous_probability = None
    previous_dropout = None
    arms = {}

    for arm in ARMS:
        model = build(arm)
        initial = weight_hash(model)
        if previous_initial is None:
            previous_initial = initial
        require(initial == previous_initial, "paired initialization differs")

        shapes = {
            t.name.split(":")[0]: tuple(t.shape.as_list())
            for t in model.inputs
        }
        require(
            shapes["spectral_map"][1:] == (31, 64, 4)
            and shapes["rescue_map"][1:] == (31, 64, 2),
            "HPR-v2 streams not connected",
        )
        x = dict(base_x)
        x["rescue_map"] = rescue_value.copy()
        p = model(x, training=False).numpy()
        np.testing.assert_allclose(p, reference_probability, rtol=0, atol=1e-7)
        if previous_probability is None:
            previous_probability = p
        np.testing.assert_array_equal(p, previous_probability)

        rescue_head = model.get_layer("hprv2_rescue_logits")
        before_rescue = [w.copy() for w in rescue_head.get_weights()]
        require(
            all(np.count_nonzero(w) == 0 for w in before_rescue),
            "rescue residual is not zero initialized",
        )

        dropout = {
            name: model.get_layer(name)(
                tf.ones((2, 192)), training=True
            ).numpy()
            for name in ("cluster_dropout", "v240_cardinality_dropout")
        }
        if previous_dropout is None:
            previous_dropout = dropout
        for name in dropout:
            np.testing.assert_array_equal(dropout[name], previous_dropout[name])

        compile_metrics(model)
        require(int(model.optimizer.iterations.numpy()) == 0, "optimizer not fresh")
        log = model.train_on_batch(
            x, np.array([0, 1, 2, 3]), return_dict=True
        )
        require(
            all(np.isfinite(v) for v in log.values())
            and weight_hash(model) != initial,
            "synthetic update failed",
        )
        after_rescue = rescue_head.get_weights()
        rescue_changed = any(
            not np.array_equal(a, b)
            for a, b in zip(before_rescue, after_rescue)
        )
        if arm == "observed_only":
            require(not rescue_changed, "control rescue branch received gradients")
        else:
            require(rescue_changed, "treatment rescue branch did not learn")

        arms[arm] = dict(
            initial_sha256=initial,
            parameters=model.count_params(),
            initial_matches_four_channel_control=True,
            rescue_zero_initialized=True,
            rescue_changed_after_synthetic_update=rescue_changed,
            all_initial_dropout_masks_identical=True,
        )

    require(
        arms[ARMS[0]]["initial_sha256"] == arms[ARMS[1]]["initial_sha256"]
        and arms[ARMS[0]]["parameters"] == arms[ARMS[1]]["parameters"],
        "paired model contract differs",
    )

    from scripts.v273_high_pitch_map import time_frequency_map
    audio = np.random.default_rng(12).normal(0, .02, 24000).astype(np.float32)
    feature_times = {}
    for arm in ARMS:
        for _ in range(3):
            time_frequency_map(audio, 7000, include_pitch=(arm == "with_high_pitch"))
        elapsed = []
        for _ in range(30):
            start = time.perf_counter()
            time_frequency_map(audio, 7000, include_pitch=(arm == "with_high_pitch"))
            elapsed.append((time.perf_counter()-start)*1000)
        feature_times[arm] = dict(
            p50_ms=float(np.median(elapsed)),
            p95_ms=float(np.quantile(elapsed, .95)),
            repeats=30,
        )

    sample_model = build("with_high_pitch")
    sample_x = dict(base_x)
    sample_x["rescue_map"] = rescue_value.copy()
    single = {key: value[:1] for key, value in sample_x.items()}
    infer = tf.function(lambda values: sample_model(values, training=False))
    for _ in range(3):
        infer(single).numpy()
    inference_elapsed = []
    for _ in range(30):
        start = time.perf_counter()
        infer(single).numpy()
        inference_elapsed.append((time.perf_counter()-start)*1000)
    inference = dict(
        p50_ms=float(np.median(inference_elapsed)),
        p95_ms=float(np.quantile(inference_elapsed, .95)),
        repeats=30,
    )

    sample_model = build("with_high_pitch")
    sample_x = dict(base_x)
    sample_x["rescue_map"] = rescue_value.copy()
    single = {key: value[:1] for key, value in sample_x.items()}
    infer = tf.function(lambda values: sample_model(values, training=False))
    for _ in range(3):
        infer(single).numpy()
    inference_elapsed = []
    for _ in range(30):
        start = time.perf_counter()
        infer(single).numpy()
        inference_elapsed.append((time.perf_counter()-start)*1000)
    inference = dict(
        p50_ms=float(np.median(inference_elapsed)),
        p95_ms=float(np.quantile(inference_elapsed, .95)),
        repeats=30,
    )

    output.mkdir(parents=True)
    report = dict(
        status="passed",
        **identity,
        tensorflow=tf.__version__,
        numpy=np.__version__,
        arms=arms,
        same_initial_prediction_between_arms=True,
        initial_matches_four_channel_control=True,
        historical_default_unchanged=True,
        zero_initialized_rescue=True,
        control_rescue_gradient_blocked=True,
        epoch_budget=EPOCHS,
        checkpoints=list(CHECKPOINTS),
        feature_runtime=feature_times,
        inference_single_group=inference,
        latency_scope=(
            "HPR-v2 uses 2048 additional past samples but no additional future lookahead"
        ),
        script_sha256=digest(__file__),
        training_performed=False,
        output_corrector=False,
    )
    write_json(output/"report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    preflight(p.parse_args().output)
