"""Native acoustic-input extension for the count components of V27.3.

This module is integration code, not a completed GuitarSet experiment. It
uses the original V100 spectral map and the original V260/V272 count networks.
It does not consume V27.3 predictions and adds no output correction rule.
"""
from __future__ import annotations

import numpy as np

from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v260_count_weighting as v260
from scripts import train_v272_poly_conditional_count as v272


COMPONENTS = ("v260_uniform", "v260_weighted", "v272_uniform")
ARMS = ("control", "decay")
PRE_FRAMES = max(2, v100.PRE_SAMPLES // v100.FRAME_STEP - 1)
FEATURE_DIM = 2 * v100.SPECTRAL_BANDS
MIN_VALID_FRAMES = 6
POWER_FLOOR = 1e-6
RELATIVE_FLOOR = 0.02
MIN_POWER_DECAY_PER_SECOND = 0.5
MAX_POWER_DECAY_PER_SECOND = 120.0
MAX_LOG_POWER_RMSE = 0.30
SATURATION_LOG_POWER = 11.99


def native_decay_features(spectral):
    """Return 128 decay-novelty features per original 23 x 64 x 3 crop.

    V100 channel 0 is ln(1 + power / scalar). Scalar is constant within the
    crop, so regressing ln(expm1(channel_0)) estimates the *power* decay rate.
    Its first nine FFT windows end before cluster_start. Only these windows
    fit the line. Remaining windows provide evidence within the original
    +40 ms observation budget. No labels or cross-row statistics are used.
    """
    x = np.asarray(spectral, dtype=np.float64)
    expected = (v100.TIME_FRAMES, v100.SPECTRAL_BANDS, v100.SPECTRAL_CHANNELS)
    if x.ndim != 4 or x.shape[1:] != expected or len(x) == 0:
        raise ValueError(f"expected nonempty [batch,{expected}] native spectral maps")
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 12):
        raise ValueError("native spectral map must be finite and within [0,12]")
    log_power = x[..., 0]
    pre_log = log_power[:, :PRE_FRAMES]
    power = np.expm1(pre_log)
    floor = np.maximum(POWER_FLOOR, RELATIVE_FLOOR * power.max(axis=1, keepdims=True))
    valid = (power > floor) & (pre_log < SATURATION_LOG_POWER)
    y = np.log(np.maximum(power, POWER_FLOOR))
    t = ((np.arange(PRE_FRAMES) - PRE_FRAMES + 1)
         * (v100.FRAME_STEP / v100.SAMPLE_RATE))[None, :, None]
    count = valid.sum(axis=1)
    denominator = np.maximum(count, 1)
    mean_t = (t * valid).sum(axis=1) / denominator
    mean_y = (y * valid).sum(axis=1) / denominator
    centered_t = t - mean_t[:, None, :]
    variance = (centered_t**2 * valid).sum(axis=1)
    slope = ((centered_t * (y - mean_y[:, None, :]) * valid).sum(axis=1)
             / np.maximum(variance, 1e-12))
    intercept = mean_y - slope * mean_t
    fitted = intercept[:, None, :] + slope[:, None, :] * t
    rmse = np.sqrt(((y - fitted)**2 * valid).sum(axis=1) / denominator)
    reliable = (
        (count >= MIN_VALID_FRAMES)
        & (slope <= -MIN_POWER_DECAY_PER_SECOND)
        & (slope >= -MAX_POWER_DECAY_PER_SECOND)
        & (rmse <= MAX_LOG_POWER_RMSE)
        & np.all(pre_log < SATURATION_LOG_POWER, axis=1)
    )
    future_t = (np.arange(1, v100.TIME_FRAMES - PRE_FRAMES + 1)
                * (v100.FRAME_STEP / v100.SAMPLE_RATE))[None, :, None]
    predicted_ln_power = (
        intercept[:, None, :]
        + np.clip(slope, -MAX_POWER_DECAY_PER_SECOND, 0)[:, None, :] * future_t
    )
    predicted_log = np.log1p(np.exp(np.clip(predicted_ln_power, -40, 20)))
    novelty = np.maximum(log_power[:, PRE_FRAMES:] - predicted_log, 0)
    novelty *= reliable[:, None, :]
    features = np.concatenate([novelty.mean(axis=1), novelty.max(axis=1)], axis=1)
    return features.astype(np.float32), {
        "power_slope_per_second": slope,
        "log_power_rmse": rmse,
        "reliable": reliable,
    }


def native_inputs(cache, indices, arm):
    """Keep every original input; add acoustic evidence or its zero control."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    inputs = v102._inputs(cache, indices)
    if arm == "control":
        features = np.zeros((len(inputs["spectral_map"]), FEATURE_DIM), dtype=np.float32)
    else:
        features, _ = native_decay_features(inputs["spectral_map"])
    return {**inputs, "decay_evidence": features}


def build_count_model(component, seed):
    """Add a trainable acoustic branch before the original count hidden layers.

    Both arms use this same model and initializer. Only their acoustic input
    differs. The bias-free projection starts at zero, preserving every legacy
    parameter and initial prediction. For a zero input it remains inactive
    under the original Adam optimizer (no weight decay or regularization).

    Returns (extended, original) for parity checks. They share legacy layers;
    never train these two returned objects as independent comparison arms.
    Build a fresh pair with the same seed for each arm instead.
    """
    import tensorflow as tf

    if component not in COMPONENTS:
        raise ValueError(f"unknown count component {component!r}")
    if component.startswith("v260_"):
        original = v260.build_model(component.removeprefix("v260_"), seed)
        head_name = "cardinality"
    else:
        original = v272.build_model("uniform", seed)
        head_name = "v272_poly_cardinality"
    context = original.get_layer("v240_cardinality_context").output
    evidence = tf.keras.Input((FEATURE_DIM,), name="decay_evidence")
    added = tf.keras.layers.Dense(
        int(context.shape[-1]), use_bias=False, kernel_initializer="zeros",
        name="native_decay_projection",
    )(evidence)
    hidden = tf.keras.layers.Add(name="native_context_with_decay")([context, added])
    for name in ("v240_cardinality_hidden1", "v240_cardinality_dropout",
                 "v240_cardinality_hidden2", head_name):
        hidden = original.get_layer(name)(hidden)
    extended = tf.keras.Model(
        [*original.inputs, evidence], hidden, name=f"{component}_native_decay",
    )
    extended.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="sparse_categorical_crossentropy",
    )
    return extended, original
