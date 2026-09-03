"""V14.0 sequential CONTINUE/STOP decoder with acoustic explaining-away.

V13 showed a strong anonymous event representation but overcounted because six
independent presence heads could claim nearly the same TF/candidate evidence.
The frozen V13 failure audit found: (1) high per-query event/non-event AUC,
(2) false-positive queries with systematically weaker TF support, and (3) very
high candidate/time overlap on overcount rows.  V14 therefore changes the
*decision process*, not the grouping window or runtime information.

At step q=0..5 the decoder asks:
    "after q already-explained births, is there another birth?"
Each accepted soft event claims TF and candidate evidence; that claim is removed
from the residual evidence seen by later steps.  Continue heads are conditional:
q0 is trained on all rows, q1 only where K>=1, ..., q5 only where K>=5.  Thus the
last decision is directly K5-vs-K6 rather than being dominated by K0/K1 rows.

Runtime cardinality is prefix STOP decoding with a fixed 0.5 threshold:
    count = number of consecutive CONTINUE decisions before the first STOP.
There is no categorical K head and no threshold tuning.  Event time/candidate
heads are training auxiliaries/diagnostics; headline candidate realization keeps
the frozen V9+ ranking so this experiment isolates cardinality.

Evaluation is the same five-fold composition outer-clean protocol as V12/V13.
Historical validation/locked12 are never indexed or evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SLOT_COUNT
from scripts.train_boundaries import group_stem
from scripts.train_v90_structured_cluster_cardinality import MAX_CANDIDATES
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import TIME_FRAMES, _load_spectral_caches
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_oof_fold as oofmod
from scripts import train_v120_integrated_birth_source_time as v120
from scripts import train_v130_causal_event_set_decoder as v130

DEFAULT_SEED = 14041
STEPS = SLOT_COUNT
STOP_THRESHOLD = 0.5
MAX_META_EPOCHS = 35
QUERY_DIM = 96
EPS = 1e-6


class V140Error(RuntimeError):
    pass


def _input_by_name(model, name):
    for tensor in model.inputs:
        if tensor.name.split(":", 1)[0] == name:
            return tensor
    raise V140Error(f"model input not found: {name}")


def _build_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    base, _, token_shape = v102._build_model()
    candidate_context = base.get_layer("candidate_context").output
    tf_tokens = base.get_layer("tf_tokens").output
    candidate_set = _input_by_name(base, "candidate_set")
    candidate_mask = _input_by_name(base, "candidate_mask")

    cand = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="v140_candidate_norm")(candidate_set)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v140_candidate_hidden1")(cand)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v140_candidate_hidden2")(cand)
    cand_keys = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, use_bias=False), name="v140_candidate_keys")(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v140_tf_keys")(tf_tokens)

    # Residual evidence begins fully available. Candidate residual also respects
    # the runtime candidate mask. These tensors are updated differentiably.
    residual_tf = keras.layers.Lambda(lambda x: tf.ones_like(x[:, :, 0]), name="v140_initial_tf_residual")(tf_keys)
    residual_cand = keras.layers.Lambda(lambda m: tf.cast(m, tf.float32), name="v140_initial_candidate_residual")(candidate_mask)

    previous_state = keras.layers.Dense(QUERY_DIM, activation="relu", name="v140_initial_state")(candidate_context)
    survival = None
    continue_outputs = []
    time_outputs = []
    candidate_outputs = []
    tf_mass_outputs = []
    cand_mass_outputs = []
    residual_tf_outputs = []
    residual_cand_outputs = []

    token_freq = int(token_shape[1])

    for q in range(STEPS):
        step_context = keras.layers.Concatenate(name=f"v140_step_{q}_context")([candidate_context, previous_state])
        query = keras.layers.Dense(QUERY_DIM, activation="relu", name=f"v140_step_{q}_query")(step_context)

        tf_score = keras.layers.Lambda(
            lambda z: tf.einsum("btd,bd->bt", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"v140_step_{q}_tf_score",
        )([tf_keys, query])
        tf_affinity = keras.layers.Activation("sigmoid", name=f"v140_step_{q}_tf_affinity")(tf_score)
        tf_claimable = keras.layers.Multiply(name=f"v140_step_{q}_tf_claimable")([tf_affinity, residual_tf])
        tf_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v140_step_{q}_tf_distribution",
        )(tf_claimable)
        tf_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v140_step_{q}_tf_pool",
        )([tf_tokens, tf_dist])
        tf_mass = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v140_step_{q}_tf_mass",
        )([tf_claimable, residual_tf])

        cand_score = keras.layers.Lambda(
            lambda z: tf.einsum("bcd,bd->bc", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"v140_step_{q}_candidate_score",
        )([cand_keys, query])
        cand_affinity = keras.layers.Activation("sigmoid", name=f"v140_step_{q}_candidate_affinity")(cand_score)
        cand_claimable = keras.layers.Multiply(name=f"v140_step_{q}_candidate_claimable")([cand_affinity, residual_cand])
        cand_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v140_event_candidate_{q}",
        )(cand_claimable)
        cand_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v140_step_{q}_candidate_pool",
        )([cand, cand_dist])
        cand_mass = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v140_step_{q}_candidate_mass",
        )([cand_claimable, residual_cand])

        tf_grid = keras.layers.Reshape((TIME_FRAMES, token_freq), name=f"v140_step_{q}_tf_grid")(tf_claimable)
        time_mass = keras.layers.Lambda(lambda a: tf.reduce_sum(a, axis=2), name=f"v140_step_{q}_time_mass")(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS),
            name=f"v140_event_time_{q}",
        )(time_mass)

        residual_tf_fraction = keras.layers.Lambda(
            lambda r: tf.reduce_mean(r, axis=1, keepdims=True), name=f"v140_step_{q}_residual_tf_fraction"
        )(residual_tf)
        residual_cand_fraction = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v140_step_{q}_residual_candidate_fraction",
        )([residual_cand, candidate_mask])

        feature = keras.layers.Concatenate(name=f"v140_step_{q}_feature")([
            candidate_context, previous_state, query, tf_latent, cand_latent,
            tf_mass, cand_mass, residual_tf_fraction, residual_cand_fraction,
        ])
        feature = keras.layers.LayerNormalization(name=f"v140_step_{q}_feature_norm")(feature)
        hidden = keras.layers.Dense(
            128, activation="relu", kernel_regularizer=keras.regularizers.l2(1.5e-3),
            name=f"v140_step_{q}_hidden1",
        )(feature)
        hidden = keras.layers.Dropout(0.08, name=f"v140_step_{q}_dropout")(hidden)
        hidden = keras.layers.Dense(64, activation="relu", name=f"v140_step_{q}_hidden2")(hidden)
        cont = keras.layers.Dense(1, activation="sigmoid", name=f"v140_continue_{q}")(hidden)

        # survival_q = probability that event q exists under sequential CONTINUE.
        survival = cont if survival is None else keras.layers.Multiply(name=f"v140_survival_{q}")([survival, cont])

        # Explaining-away: if this event exists, claim its affinity-weighted TF
        # and candidate evidence. Later steps only see the residual. No labels or
        # hard decisions enter this path at runtime or training.
        tf_claim = keras.layers.Multiply(name=f"v140_step_{q}_tf_claim")([tf_affinity, survival])
        cand_claim = keras.layers.Multiply(name=f"v140_step_{q}_candidate_claim")([cand_affinity, survival])
        residual_tf = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v140_step_{q}_tf_residual_after",
        )([residual_tf, tf_claim])
        residual_cand = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v140_step_{q}_candidate_residual_after",
        )([residual_cand, cand_claim])

        previous_state = keras.layers.Dense(QUERY_DIM, activation="relu", name=f"v140_step_{q}_state")(hidden)
        continue_outputs.append(cont)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_dist)
        tf_mass_outputs.append(tf_mass)
        cand_mass_outputs.append(cand_mass)
        residual_tf_outputs.append(residual_tf_fraction)
        residual_cand_outputs.append(residual_cand_fraction)

    survival_vector = keras.layers.Concatenate(name="v140_survival_vector")([
        continue_outputs[0]
    ] + [base_surv for base_surv in []]) if False else None
    # Recompute cumulative survival explicitly from continue outputs for expected count.
    cumulative = []
    cur = None
    for q, cont in enumerate(continue_outputs):
        cur = cont if cur is None else keras.layers.Multiply(name=f"v140_expected_survival_{q}")([cur, cont])
        cumulative.append(cur)
    expected_count = keras.layers.Add(name="v140_expected_count")(cumulative)
    count_norm = keras.layers.Lambda(lambda x: x / float(STEPS), name="v140_count_norm")(expected_count)

    outputs = {}
    # Keep V10.2 physical heads only as low-weight training stabilizers.
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    for q in range(STEPS):
        outputs[f"v140_continue_{q}"] = continue_outputs[q]
        outputs[f"v140_event_time_{q}"] = time_outputs[q]
        outputs[f"v140_event_candidate_{q}"] = candidate_outputs[q]
        outputs[f"v140_tf_mass_{q}"] = tf_mass_outputs[q]
        outputs[f"v140_candidate_mass_{q}"] = cand_mass_outputs[q]
        outputs[f"v140_residual_tf_{q}"] = residual_tf_outputs[q]
        outputs[f"v140_residual_candidate_{q}"] = residual_cand_outputs[q]
    outputs["v140_count_norm"] = count_norm

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    for q in range(STEPS):
        loss[f"v140_continue_{q}"] = "binary_crossentropy"
        loss[f"v140_event_time_{q}"] = keras.losses.KLDivergence()
        loss[f"v140_event_candidate_{q}"] = "categorical_crossentropy"
        # Diagnostic outputs have zero loss but need no targets, so do not compile
        # losses for the mass/residual outputs.
    loss["v140_count_norm"] = "mse"

    loss_weights = {f"string_{slot}": 0.14 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.03 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.08 for slot in range(SLOT_COUNT)})
    for q in range(STEPS):
        loss_weights[f"v140_continue_{q}"] = 1.0
        loss_weights[f"v140_event_time_{q}"] = 0.25
        loss_weights[f"v140_event_candidate_{q}"] = 0.20
    loss_weights["v140_count_norm"] = 0.25

    model = keras.Model(base.inputs, outputs, name="v140_sequential_stop_explaining_away")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, token_shape


def _balanced_conditional_weights(target: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    reachable = np.asarray(reachable, dtype=bool).reshape(-1)
    out = np.zeros(len(target), dtype=np.float32)
    if not np.any(reachable):
        return out
    local = target[reachable]
    pos = int(np.sum(local > 0.5))
    neg = int(len(local) - pos)
    if pos == 0 or neg == 0:
        out[reachable] = 1.0
        return out
    wp = math.sqrt(len(local) / (2.0 * pos))
    wn = math.sqrt(len(local) / (2.0 * neg))
    weights = np.where(local > 0.5, np.clip(wp, 0.35, 8.0), np.clip(wn, 0.35, 8.0)).astype(np.float32)
    weights /= float(np.mean(weights))
    out[reachable] = weights
    return out


def _targets(cache, pitch_targets, string_time_targets, k, event_time, event_candidate):
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = np.asarray(cache["slot_targets"][:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"pitch_{slot}"] = np.asarray(pitch_targets[:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"time_{slot}"] = np.asarray(string_time_targets[:, slot, :], dtype=np.float32)
    for q in range(STEPS):
        out[f"v140_continue_{q}"] = (np.asarray(k) > q).astype(np.float32).reshape(-1, 1)
        out[f"v140_event_time_{q}"] = event_time[:, q, :]
        out[f"v140_event_candidate_{q}"] = event_candidate[:, q, :]
    out["v140_count_norm"] = (np.asarray(k, dtype=np.float32) / float(STEPS)).reshape(-1, 1)
    return out


def _sample_weights(cache, time_mask, k, event_valid):
    aux = v102._sample_weights(cache["slot_targets"], time_mask, k)
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = aux[f"string_{slot}"]
        out[f"pitch_{slot}"] = aux[f"pitch_{slot}"]
        out[f"time_{slot}"] = aux[f"time_{slot}"]
    kk = np.asarray(k, dtype=np.int32)
    for q in range(STEPS):
        reachable = kk >= q
        target = kk > q
        out[f"v140_continue_{q}"] = _balanced_conditional_weights(target, reachable)
        out[f"v140_event_time_{q}"] = np.asarray(event_valid[:, q], dtype=np.float32)
        out[f"v140_event_candidate_{q}"] = np.asarray(event_valid[:, q], dtype=np.float32)
    out["v140_count_norm"] = v102._count_weights(kk)
    return out


def _subset(mapping, idx):
    idx = np.asarray(idx, dtype=np.int64)
    return {key: np.asarray(value)[idx] for key, value in mapping.items()}


def _decode(model, inputs):
    raw = model.predict(inputs, batch_size=128, verbose=0)
    cont = np.stack([
        np.asarray(raw[f"v140_continue_{q}"], dtype=np.float64).reshape(-1)
        for q in range(STEPS)
    ], axis=1)
    time = np.stack([np.asarray(raw[f"v140_event_time_{q}"], dtype=np.float64) for q in range(STEPS)], axis=1)
    candidate = np.stack([np.asarray(raw[f"v140_event_candidate_{q}"], dtype=np.float64) for q in range(STEPS)], axis=1)
    tf_mass = np.stack([np.asarray(raw[f"v140_tf_mass_{q}"], dtype=np.float64).reshape(-1) for q in range(STEPS)], axis=1)
    cand_mass = np.stack([np.asarray(raw[f"v140_candidate_mass_{q}"], dtype=np.float64).reshape(-1) for q in range(STEPS)], axis=1)
    residual_tf = np.stack([np.asarray(raw[f"v140_residual_tf_{q}"], dtype=np.float64).reshape(-1) for q in range(STEPS)], axis=1)
    residual_cand = np.stack([np.asarray(raw[f"v140_residual_candidate_{q}"], dtype=np.float64).reshape(-1) for q in range(STEPS)], axis=1)
    pred = np.zeros(len(cont), dtype=np.int32)
    for row in range(len(cont)):
        count = 0
        for q in range(STEPS):
            if cont[row, q] < STOP_THRESHOLD:
                break
            count += 1
        pred[row] = count
    return pred, cont, time, candidate, tf_mass, cand_mass, residual_tf, residual_cand


def _load_v130(v130_dir: Path, fold: int, outer_idx: np.ndarray):
    paths = sorted(v130_dir.glob(f"**/predictions-fold-{fold}.npz"))
    if len(paths) != 1:
        raise V140Error(f"fold {fold}: expected exactly one frozen V13 prediction shard, got {len(paths)}")
    with np.load(paths[0], allow_pickle=False) as z:
        gi = np.asarray(z["global_index"], dtype=np.int64)
        pred = np.asarray(z["pred130"], dtype=np.int32)
    if not np.array_equal(gi, np.asarray(outer_idx, dtype=np.int64)):
        raise V140Error(f"fold {fold}: V13 outer rows mismatch")
    return pred


def _conditional_diagnostics(k, cont, tf_mass, cand_mass, pred):
    kk = np.asarray(k, dtype=np.int32)
    result = {"steps": {}, "mean_predicted_count": float(np.mean(pred)), "mean_true_count": float(np.mean(kk))}
    for q in range(STEPS):
        reachable = kk >= q
        y = kk > q
        p = cont[:, q]
        row = {
            "reachable": int(np.sum(reachable)),
            "positive": int(np.sum(reachable & y)),
            "negative": int(np.sum(reachable & ~y)),
            "mean_probability_positive": float(np.mean(p[reachable & y])) if np.any(reachable & y) else None,
            "mean_probability_negative": float(np.mean(p[reachable & ~y])) if np.any(reachable & ~y) else None,
            "tf_mass_positive": float(np.mean(tf_mass[reachable & y, q])) if np.any(reachable & y) else None,
            "tf_mass_negative": float(np.mean(tf_mass[reachable & ~y, q])) if np.any(reachable & ~y) else None,
            "candidate_mass_positive": float(np.mean(cand_mass[reachable & y, q])) if np.any(reachable & y) else None,
            "candidate_mass_negative": float(np.mean(cand_mass[reachable & ~y, q])) if np.any(reachable & ~y) else None,
        }
        result["steps"][str(q)] = row
    return result


def train_fold(args):
    if not 0 <= args.outer_fold < oofmod.FOLD_COUNT:
        raise V140Error("outer fold outside range")
    random.seed(args.seed + args.outer_fold)
    np.random.seed(args.seed + args.outer_fold)
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    tf.random.set_seed(args.seed + args.outer_fold)

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if train_members & validation_members:
        raise V140Error("train/validation leakage")
    if set(cache["track_members"]) != train_members:
        raise V140Error("spectral cache does not exactly cover train split")

    assignment, groups_per_fold, _, _ = oofmod._balanced_group_folds(cache, train_split)
    by_member = {t.annotation_member: t for t in train_split}
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    row_fold = np.asarray([assignment[group_stem(by_member[m])] for m in members], dtype=np.int16)
    outer_idx = np.flatnonzero(row_fold == args.outer_fold).astype(np.int64)
    remaining = sorted(set(range(oofmod.FOLD_COUNT)) - {args.outer_fold})
    meta_fold = remaining[0]
    meta_fit_idx = np.flatnonzero((row_fold != args.outer_fold) & (row_fold != meta_fold)).astype(np.int64)
    meta_val_idx = np.flatnonzero(row_fold == meta_fold).astype(np.int64)
    final_fit_idx = np.flatnonzero(row_fold != args.outer_fold).astype(np.int64)
    if np.intersect1d(final_fit_idx, outer_idx).size or np.intersect1d(meta_fit_idx, outer_idx).size or np.intersect1d(meta_val_idx, outer_idx).size:
        raise V140Error("outer fold leakage")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), STEPS)
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, string_time_targets, time_sample, supervision = v102._derive_supervision(
        members, candidate_samples, args.dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    _, event_time, event_candidate, event_valid, true_sample, event_diag = v130._ordered_event_supervision(
        cache, time_mask, string_time_targets, time_sample, k
    )
    all_targets = _targets(cache, pitch_targets, string_time_targets, k, event_time, event_candidate)
    all_weights = _sample_weights(cache, time_mask, k, event_valid)

    tf.keras.backend.clear_session()
    random.seed(args.seed + 100 + args.outer_fold)
    np.random.seed(args.seed + 100 + args.outer_fold)
    tf.random.set_seed(args.seed + 100 + args.outer_fold)
    probe, loss_weights, token_shape = _build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=2e-4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5),
    ]
    hist = probe.fit(
        v102._inputs(cache, meta_fit_idx), _subset(all_targets, meta_fit_idx),
        sample_weight=_subset(all_weights, meta_fit_idx),
        validation_data=(v102._inputs(cache, meta_val_idx), _subset(all_targets, meta_val_idx), _subset(all_weights, meta_val_idx)),
        epochs=MAX_META_EPOCHS, batch_size=64, shuffle=True, callbacks=callbacks, verbose=2,
    )
    selected_epochs = max(2, int(np.argmin(np.asarray(hist.history["val_loss"], dtype=np.float64)) + 1))
    meta_pred, meta_cont, *_ = _decode(probe, v102._inputs(cache, meta_val_idx))
    meta_metrics = v120._metrics(cache, train_split, meta_val_idx, meta_pred)

    tf.keras.backend.clear_session()
    random.seed(args.seed + 1000 + args.outer_fold)
    np.random.seed(args.seed + 1000 + args.outer_fold)
    tf.random.set_seed(args.seed + 1000 + args.outer_fold)
    model, _, _ = _build_model()
    model.fit(
        v102._inputs(cache, final_fit_idx), _subset(all_targets, final_fit_idx),
        sample_weight=_subset(all_weights, final_fit_idx), epochs=selected_epochs,
        batch_size=64, shuffle=True, verbose=2,
    )
    pred140, cont, event_time_pred, event_candidate_pred, tf_mass, cand_mass, residual_tf, residual_cand = _decode(
        model, v102._inputs(cache, outer_idx)
    )

    baseline = v120._load_baseline(args.baseline_eval_dir, args.outer_fold, outer_idx)
    pred101 = np.asarray(baseline["pred101_deploy"], dtype=np.int32)
    pred102 = np.asarray(baseline["pred102_deploy"], dtype=np.int32)
    pred104 = np.asarray(baseline["pred104_deploy"], dtype=np.int32)
    pred130 = _load_v130(args.v130_dir, args.outer_fold, outer_idx)

    full = {}
    for name, local in (("v101", pred101), ("v102", pred102), ("v104", pred104), ("v130", pred130), ("v140", pred140)):
        arr = np.full(len(k), -1, dtype=np.int32)
        arr[outer_idx] = local
        full[name] = arr

    players = np.asarray([v120._player(m) for m in members], dtype="U2")
    modes = np.asarray([v120._mode(m) for m in members], dtype="U8")
    groups = np.asarray([group_stem(by_member[m]) for m in members], dtype="U64")
    genres = np.asarray([v120._genre(g) for g in groups], dtype="U16")
    def outer_where(mask):
        return outer_idx[np.asarray(mask, dtype=bool)[outer_idx]]
    strata = {
        "aggregate": outer_idx,
        "comp": outer_where(modes == "comp"),
        "solo": outer_where(modes == "solo"),
        "player00": outer_where(players == "00"),
        "player00_comp": outer_where((players == "00") & (modes == "comp")),
        "player00_solo": outer_where((players == "00") & (modes == "solo")),
        "player00_rock_comp": outer_where((players == "00") & (modes == "comp") & (genres == "Rock")),
    }
    for player in sorted(set(players.tolist())):
        strata[f"player{player}"] = outer_where(players == player)
    strata_report = {name: v120._slice_report(cache, train_split, k, idx, full) for name, idx in strata.items()}

    ko = k[outer_idx]
    per_k = {}
    for value in range(STEPS + 1):
        local = ko == value
        row = {"clusters": int(np.sum(local))}
        for name, pred in (("v101", pred101), ("v102", pred102), ("v104", pred104), ("v130", pred130), ("v140", pred140)):
            if np.any(local):
                pp = pred[local]
                row[name] = {
                    "exact": float(np.mean(pp == value)),
                    "under_rate": float(np.mean(pp < value)),
                    "over_rate": float(np.mean(pp > value)),
                    "mae": float(np.mean(np.abs(pp - value))),
                }
            else:
                row[name] = {"exact": None, "under_rate": None, "over_rate": None, "mae": None}
        per_k[str(value)] = row

    report = {
        "schema_version": 1,
        "protocol": {
            "train_only_outer_clean": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "outer_fold_used_for_training": False,
            "outer_fold_used_for_epoch_selection": False,
            "runtime_inputs_use_annotations": False,
            "stop_threshold_tuned": False,
            "stop_threshold": STOP_THRESHOLD,
            "categorical_cardinality_head_exists": False,
            "cardinality_decode": "consecutive CONTINUE>=0.5 until first STOP",
            "conditional_continue_training": True,
            "explaining_away_uses_predictions_not_annotations": True,
            "grouping_window_ms_unchanged": 40,
            "offset_model_untouched": True,
        },
        "architecture": {
            "name": "V14.0 sequential STOP with acoustic explaining-away",
            "steps": STEPS,
            "step_outputs": ["continue", "birth_time_distribution", "candidate_distribution"],
            "tf_residual_explaining_away": True,
            "candidate_residual_explaining_away": True,
            "conditional_training_rule": "step q receives decision loss only on K>=q",
            "runtime_annotation_required": False,
            "headline_candidate_realization": "frozen V9+ ranking",
            "trainable_parameters": int(model.count_params()),
            "tf_token_shape": list(token_shape),
            "loss_weights": loss_weights,
        },
        "outer_fold": int(args.outer_fold),
        "outer_groups": groups_per_fold[args.outer_fold],
        "data": {
            "outer_clusters": int(len(outer_idx)), "outer_train_clusters": int(len(final_fit_idx)),
            "meta_fit_clusters": int(len(meta_fit_idx)), "meta_validation_clusters": int(len(meta_val_idx)),
            "meta_validation_fold": int(meta_fold), "selected_epochs": int(selected_epochs),
            "train_tracks": int(len(train_split)), "validation_tracks_not_evaluated": int(len(validation)),
        },
        "supervision": {
            "assigned_events": supervision.get("assigned_events"), "unassigned_events": supervision.get("unassigned_events"),
            "cluster_reconstruction": reconstruction, "anonymous_event_targets": event_diag,
        },
        "meta_validation": {
            "metrics": meta_metrics, "cardinality": v120._card(k[meta_val_idx], meta_pred),
            "mean_continue_by_step": np.mean(meta_cont, axis=0).tolist(),
        },
        "strata": strata_report,
        "per_true_k": per_k,
        "conditional_diagnostics": _conditional_diagnostics(ko, cont, tf_mass, cand_mass, pred140),
        "residual_diagnostics": {
            "mean_tf_residual_before_step": np.mean(residual_tf, axis=0).tolist(),
            "mean_candidate_residual_before_step": np.mean(residual_cand, axis=0).tolist(),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / f"predictions-fold-{args.outer_fold}.npz",
        outer_fold=np.full(len(outer_idx), args.outer_fold, dtype=np.int16),
        global_index=outer_idx, k=ko.astype(np.int16), member=members[outer_idx],
        pred101=pred101.astype(np.int16), pred102=pred102.astype(np.int16), pred104=pred104.astype(np.int16),
        pred130=pred130.astype(np.int16), pred140=pred140.astype(np.int16),
        continue_probability=cont.astype(np.float32), event_time=event_time_pred.astype(np.float32),
        event_candidate=event_candidate_pred.astype(np.float32), tf_mass=tf_mass.astype(np.float32),
        candidate_mass=cand_mass.astype(np.float32), residual_tf=residual_tf.astype(np.float32),
        residual_candidate=residual_cand.astype(np.float32),
    )
    model.save_weights(args.output_dir / f"v140-fold-{args.outer_fold}.weights.h5")
    print(json.dumps({
        "outer": args.outer_fold, "epochs": selected_epochs,
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v130_f1": report["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"],
        "v140_f1": report["strata"]["aggregate"]["v140"]["metrics"]["global"]["f1"],
        "mean_true_count": float(np.mean(ko)), "mean_v140_count": float(np.mean(pred140)),
        "k6_exact_v140": per_k["6"]["v140"]["exact"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path, required=True)
    p.add_argument("--v130-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
