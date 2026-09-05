"""V13.0 causal anonymous event-set decoder.

V12.0 confirmed that separating birth existence from positive multiplicity
recovers recall in the player00/comp region, but direct K=1..6 classification
remained unstable (especially across K2/K3/K6).  V13 removes the categorical
cardinality decision entirely.

The model predicts six anonymous event queries plus background.  Every query is
jointly grounded in the causal candidate set and the 23-frame spectral memory:
- competitive assignment of TF tokens across 6 event queries + background;
- competitive assignment of candidate tokens across 6 event queries + background;
- one presence probability per event query;
- one source-derived 23-frame birth-time distribution per event query;
- one candidate-assignment distribution per event query.

Training supervision is anonymous: true births inside a 40 ms group are sorted
by onset time only (stable tie order is training-only).  Query q is present iff
q < exact K.  No pitch/string identity is required by the event decoder.
V10.2 string/pitch/time heads remain only as low-weight training auxiliaries to
stabilize the acoustic source-time representation.

Runtime cardinality is a consequence, not a class:
    K = sum_q [P(event_q present) >= 0.5]
The fixed 0.5 threshold is not tuned.  Candidate realization for the headline
comparison remains the frozen ranking used by V9+ so the experiment isolates
whether event-set representation improves cardinality.  Query time/candidate
accuracy is reported diagnostically.

Evaluation is five-fold composition outer-clean.  For each outer fold, epoch
count is selected on one different inner fold, a fresh model is fit on all four
outer-train folds for exactly that epoch count, and the untouched outer fold is
evaluated once.  Historical validation/locked12 are never indexed/evaluated.
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
from scripts.train_v90_structured_cluster_cardinality import MAX_CANDIDATES, CLUSTER_WINDOW_SAMPLES
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import TIME_FRAMES, _load_spectral_caches
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_oof_fold as oofmod
from scripts import train_v120_integrated_birth_source_time as v120

DEFAULT_SEED = 13031
EVENT_QUERIES = SLOT_COUNT
BACKGROUND = EVENT_QUERIES
PRESENCE_THRESHOLD = 0.5
MAX_META_EPOCHS = 35
QUERY_DIM = 96


class V130Error(RuntimeError):
    pass


def _input_by_name(model, name):
    for tensor in model.inputs:
        if tensor.name.split(":", 1)[0] == name:
            return tensor
    raise V130Error(f"model input not found: {name}")


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

    # A fresh anonymous candidate memory.  The last two raw candidate features
    # already encode start-relative and center-relative causal time.
    cand = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="event_candidate_norm")(candidate_set)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="event_candidate_hidden1")(cand)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="event_candidate_hidden2")(cand)
    cand_keys = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, use_bias=False), name="event_candidate_keys")(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="event_tf_keys")(tf_tokens)

    queries = [
        keras.layers.Dense(QUERY_DIM, activation="relu", name=f"event_{q}_query")(candidate_context)
        for q in range(EVENT_QUERIES)
    ]

    tf_scores = [
        keras.layers.Lambda(
            lambda z: tf.einsum("btd,bd->bt", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"event_{q}_tf_score",
        )([tf_keys, queries[q]])
        for q in range(EVENT_QUERIES)
    ]
    tf_background = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="event_tf_background_score")(
        keras.layers.Dense(1, name="event_tf_background_dense")(tf_tokens)
    )
    tf_score_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=-1), name="event_tf_score_stack"
    )(tf_scores + [tf_background])
    tf_assignment = keras.layers.Softmax(axis=-1, name="event_tf_competition")(tf_score_stack)

    cand_scores = [
        keras.layers.Lambda(
            lambda z: tf.einsum("bcd,bd->bc", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"event_{q}_candidate_score",
        )([cand_keys, queries[q]])
        for q in range(EVENT_QUERIES)
    ]
    cand_background = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="event_candidate_background_score")(
        keras.layers.TimeDistributed(keras.layers.Dense(1), name="event_candidate_background_dense")(cand)
    )
    cand_score_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=-1), name="event_candidate_score_stack"
    )(cand_scores + [cand_background])
    cand_assignment = keras.layers.Softmax(axis=-1, name="event_candidate_competition")(cand_score_stack)

    token_freq = int(token_shape[1])
    tf_grid = keras.layers.Reshape(
        (TIME_FRAMES, token_freq, EVENT_QUERIES + 1), name="event_tf_assignment_grid"
    )(tf_assignment)

    presence_outputs = []
    time_outputs = []
    candidate_outputs = []
    for q in range(EVENT_QUERIES):
        spec_w = keras.layers.Lambda(lambda a, i=q: a[:, :, i], name=f"event_{q}_tf_weights")(tf_assignment)
        spec_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
            name=f"event_{q}_tf_pool",
        )([tf_tokens, spec_w])

        raw_cand_w = keras.layers.Lambda(lambda a, i=q: a[:, :, i], name=f"event_{q}_raw_candidate_weights")(cand_assignment)
        cand_w = keras.layers.Multiply(name=f"event_{q}_masked_candidate_weights")([raw_cand_w, candidate_mask])
        cand_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
            name=f"event_{q}_candidate_pool",
        )([cand, cand_w])
        cand_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + 1e-6),
            name=f"event_candidate_{q}",
        )(cand_w)

        time_mass = keras.layers.Lambda(
            lambda a, i=q: tf.reduce_sum(a[:, :, :, i], axis=2),
            name=f"event_{q}_time_mass",
        )(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + 1e-6),
            name=f"event_time_{q}",
        )(time_mass)

        spec_mass = keras.layers.Lambda(
            lambda w: tf.reduce_mean(w, axis=1, keepdims=True), name=f"event_{q}_tf_mass"
        )(spec_w)
        cand_mass = keras.layers.Lambda(
            lambda w: tf.reduce_sum(w, axis=1, keepdims=True)
            / (tf.reduce_sum(candidate_mask, axis=1, keepdims=True) + 1e-6),
            name=f"event_{q}_candidate_mass",
        )(cand_w)
        feature = keras.layers.Concatenate(name=f"event_{q}_feature")([
            candidate_context, queries[q], spec_latent, cand_latent, spec_mass, cand_mass
        ])
        feature = keras.layers.LayerNormalization(name=f"event_{q}_feature_norm")(feature)
        feature = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1.5e-3), name=f"event_{q}_hidden1")(feature)
        feature = keras.layers.Dropout(0.08, name=f"event_{q}_dropout")(feature)
        feature = keras.layers.Dense(64, activation="relu", name=f"event_{q}_hidden2")(feature)
        present = keras.layers.Dense(1, activation="sigmoid", name=f"event_present_{q}")(feature)

        presence_outputs.append(present)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_dist)

    presence_vector = keras.layers.Concatenate(name="event_presence_vector")(presence_outputs)
    count_norm = keras.layers.Lambda(
        lambda p: tf.reduce_sum(p, axis=1, keepdims=True) / float(EVENT_QUERIES),
        name="event_count_norm",
    )(presence_vector)

    outputs = {}
    # Low-weight physical auxiliaries from V10.2.  They never participate in decode.
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    for q in range(EVENT_QUERIES):
        outputs[f"event_present_{q}"] = presence_outputs[q]
        outputs[f"event_time_{q}"] = time_outputs[q]
        outputs[f"event_candidate_{q}"] = candidate_outputs[q]
    outputs["event_count_norm"] = count_norm

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    for q in range(EVENT_QUERIES):
        loss[f"event_present_{q}"] = "binary_crossentropy"
        loss[f"event_time_{q}"] = keras.losses.KLDivergence()
        loss[f"event_candidate_{q}"] = "categorical_crossentropy"
    loss["event_count_norm"] = "mse"

    loss_weights = {f"string_{slot}": 0.18 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.04 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    for q in range(EVENT_QUERIES):
        loss_weights[f"event_present_{q}"] = 1.0
        loss_weights[f"event_time_{q}"] = 0.30
        loss_weights[f"event_candidate_{q}"] = 0.25
    loss_weights["event_count_norm"] = 0.35

    model = keras.Model(base.inputs, outputs, name="v130_causal_anonymous_event_set")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, token_shape


def _balanced_binary_weights(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    pos = int(np.sum(target > 0.5))
    neg = int(len(target) - pos)
    out = np.ones(len(target), dtype=np.float32)
    if pos == 0 or neg == 0:
        return out
    wp = math.sqrt(len(target) / (2.0 * pos))
    wn = math.sqrt(len(target) / (2.0 * neg))
    out[target > 0.5] = float(np.clip(wp, 0.35, 8.0))
    out[target <= 0.5] = float(np.clip(wn, 0.35, 8.0))
    out /= float(np.mean(out))
    return out


def _ordered_event_supervision(cache, time_mask, time_targets, time_sample, k):
    n = len(k)
    present = np.zeros((n, EVENT_QUERIES), dtype=np.float32)
    event_time = np.zeros((n, EVENT_QUERIES, TIME_FRAMES), dtype=np.float32)
    event_candidate = np.zeros((n, EVENT_QUERIES, MAX_CANDIDATES), dtype=np.float32)
    event_valid = np.zeros((n, EVENT_QUERIES), dtype=np.float32)
    true_sample = np.full((n, EVENT_QUERIES), np.nan, dtype=np.float32)
    candidate_rel = np.asarray(cache["sequence"][:, :, -2], dtype=np.float32)
    candidate_mask = np.asarray(cache["mask"], dtype=np.float32)

    for row in range(n):
        count = int(np.clip(k[row], 0, EVENT_QUERIES))
        if count:
            present[row, :count] = 1.0
        slots = np.flatnonzero(np.asarray(time_mask[row]) > 0.5).tolist()
        slots.sort(key=lambda s: (float(time_sample[row, s]), int(s)))
        for q, slot in enumerate(slots[:EVENT_QUERIES]):
            sample = float(time_sample[row, slot])
            if not math.isfinite(sample):
                continue
            event_time[row, q] = np.asarray(time_targets[row, slot], dtype=np.float32)
            true_sample[row, q] = sample
            valid_candidates = np.flatnonzero(candidate_mask[row] > 0.5)
            if not len(valid_candidates):
                continue
            rel = float(np.clip(sample / max(1.0, float(CLUSTER_WINDOW_SAMPLES)), 0.0, 1.0))
            nearest = int(valid_candidates[np.argmin(np.abs(candidate_rel[row, valid_candidates] - rel))])
            event_candidate[row, q, nearest] = 1.0
            event_valid[row, q] = 1.0

    diag = {
        "exact_presence_rows": int(n),
        "timed_event_targets": int(np.sum(event_valid)),
        "expected_positive_queries": int(np.sum(present)),
        "timed_target_fraction": float(np.sum(event_valid) / max(1.0, float(np.sum(present)))),
    }
    return present, event_time, event_candidate, event_valid, true_sample, diag


def _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate):
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = np.asarray(cache["slot_targets"][:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"pitch_{slot}"] = np.asarray(pitch_targets[:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"time_{slot}"] = np.asarray(string_time_targets[:, slot, :], dtype=np.float32)
    for q in range(EVENT_QUERIES):
        out[f"event_present_{q}"] = event_present[:, q].reshape(-1, 1)
        out[f"event_time_{q}"] = event_time[:, q, :]
        out[f"event_candidate_{q}"] = event_candidate[:, q, :]
    out["event_count_norm"] = (np.asarray(k, dtype=np.float32) / float(EVENT_QUERIES)).reshape(-1, 1)
    return out


def _sample_weights(cache, time_mask, k, event_present, event_valid):
    aux = v102._sample_weights(cache["slot_targets"], time_mask, k)
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = aux[f"string_{slot}"]
        out[f"pitch_{slot}"] = aux[f"pitch_{slot}"]
        out[f"time_{slot}"] = aux[f"time_{slot}"]
    for q in range(EVENT_QUERIES):
        out[f"event_present_{q}"] = _balanced_binary_weights(event_present[:, q])
        out[f"event_time_{q}"] = np.asarray(event_valid[:, q], dtype=np.float32)
        out[f"event_candidate_{q}"] = np.asarray(event_valid[:, q], dtype=np.float32)
    out["event_count_norm"] = v102._count_weights(np.asarray(k, dtype=np.int32))
    return out


def _subset(mapping, idx):
    idx = np.asarray(idx, dtype=np.int64)
    return {key: np.asarray(value)[idx] for key, value in mapping.items()}


def _decode(model, inputs):
    raw = model.predict(inputs, batch_size=128, verbose=0)
    presence = np.stack([
        np.asarray(raw[f"event_present_{q}"], dtype=np.float64).reshape(-1)
        for q in range(EVENT_QUERIES)
    ], axis=1)
    time = np.stack([
        np.asarray(raw[f"event_time_{q}"], dtype=np.float64)
        for q in range(EVENT_QUERIES)
    ], axis=1)
    candidate = np.stack([
        np.asarray(raw[f"event_candidate_{q}"], dtype=np.float64)
        for q in range(EVENT_QUERIES)
    ], axis=1)
    active = presence >= PRESENCE_THRESHOLD
    pred = np.sum(active, axis=1).astype(np.int32)
    prefix_violation = np.zeros(len(pred), dtype=bool)
    for row in range(len(pred)):
        seen_zero = False
        for q in range(EVENT_QUERIES):
            if not active[row, q]:
                seen_zero = True
            elif seen_zero:
                prefix_violation[row] = True
                break
    return pred, presence, time, candidate, prefix_violation


def _event_diagnostics(time_pred, candidate_pred, event_valid, true_sample, candidate_target, idx):
    idx = np.asarray(idx, dtype=np.int64)
    valid = np.asarray(event_valid[idx]) > 0.5
    out = {"timed_targets": int(np.sum(valid))}
    if not np.any(valid):
        out.update({"time_mae_ms": None, "time_median_ms": None, "candidate_top1": None})
        return out
    expected = np.sum(
        np.asarray(time_pred[idx], dtype=np.float64) * v102.FRAME_CENTER_SAMPLES[None, None, :], axis=2
    )
    truth = np.asarray(true_sample[idx], dtype=np.float64)
    err_ms = np.abs(expected[valid] - truth[valid]) * 1000.0 / float(v102.SAMPLE_RATE)
    pred_c = np.argmax(np.asarray(candidate_pred[idx]), axis=2)
    true_c = np.argmax(np.asarray(candidate_target[idx]), axis=2)
    out.update({
        "time_mae_ms": float(np.mean(err_ms)),
        "time_median_ms": float(np.median(err_ms)),
        "time_p90_ms": float(np.percentile(err_ms, 90)),
        "candidate_top1": float(np.mean(pred_c[valid] == true_c[valid])),
    })
    return out


def train_fold(args):
    if not 0 <= args.outer_fold < oofmod.FOLD_COUNT:
        raise V130Error("outer fold outside range")
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
        raise V130Error("train/validation leakage")
    if set(cache["track_members"]) != train_members:
        raise V130Error("spectral cache does not exactly cover train split")

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
    if np.intersect1d(final_fit_idx, outer_idx).size:
        raise V130Error("outer fold leaked into final fit")
    if np.intersect1d(meta_fit_idx, outer_idx).size or np.intersect1d(meta_val_idx, outer_idx).size:
        raise V130Error("outer fold leaked into epoch selection")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), EVENT_QUERIES)
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, string_time_targets, time_sample, supervision = v102._derive_supervision(
        members, candidate_samples, args.dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    event_present, event_time, event_candidate, event_valid, true_sample, event_diag = _ordered_event_supervision(
        cache, time_mask, string_time_targets, time_sample, k
    )
    all_targets = _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate)
    all_weights = _sample_weights(cache, time_mask, k, event_present, event_valid)

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
        v102._inputs(cache, meta_fit_idx),
        _subset(all_targets, meta_fit_idx),
        sample_weight=_subset(all_weights, meta_fit_idx),
        validation_data=(
            v102._inputs(cache, meta_val_idx),
            _subset(all_targets, meta_val_idx),
            _subset(all_weights, meta_val_idx),
        ),
        epochs=MAX_META_EPOCHS,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    selected_epochs = max(2, int(np.argmin(np.asarray(hist.history["val_loss"], dtype=np.float64)) + 1))
    meta_pred, meta_presence, _, _, meta_prefix = _decode(probe, v102._inputs(cache, meta_val_idx))
    meta_metrics = v120._metrics(cache, train_split, meta_val_idx, meta_pred)

    tf.keras.backend.clear_session()
    random.seed(args.seed + 1000 + args.outer_fold)
    np.random.seed(args.seed + 1000 + args.outer_fold)
    tf.random.set_seed(args.seed + 1000 + args.outer_fold)
    model, _, _ = _build_model()
    model.fit(
        v102._inputs(cache, final_fit_idx),
        _subset(all_targets, final_fit_idx),
        sample_weight=_subset(all_weights, final_fit_idx),
        epochs=selected_epochs,
        batch_size=64,
        shuffle=True,
        verbose=2,
    )
    pred130_outer, presence_outer, event_time_outer, event_candidate_outer, prefix_outer = _decode(
        model, v102._inputs(cache, outer_idx)
    )

    baseline = v120._load_baseline(args.baseline_eval_dir, args.outer_fold, outer_idx)
    pred101_outer = np.asarray(baseline["pred101_deploy"], dtype=np.int32)
    pred102_outer = np.asarray(baseline["pred102_deploy"], dtype=np.int32)
    pred104_outer = np.asarray(baseline["pred104_deploy"], dtype=np.int32)

    full = {}
    for name, local in (
        ("v101", pred101_outer), ("v102", pred102_outer), ("v104", pred104_outer), ("v130", pred130_outer)
    ):
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

    strata_report = {
        name: v120._slice_report(cache, train_split, k, idx, full)
        for name, idx in strata.items()
    }

    ko = k[outer_idx]
    per_k = {}
    for value in range(EVENT_QUERIES + 1):
        local = ko == value
        row = {"clusters": int(np.sum(local))}
        for name, pred in (
            ("v101", pred101_outer), ("v102", pred102_outer), ("v104", pred104_outer), ("v130", pred130_outer)
        ):
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

    outer_event_diag = _event_diagnostics(
        event_time, event_candidate, event_valid, true_sample, event_candidate, outer_idx
    )
    report = {
        "schema_version": 1,
        "protocol": {
            "train_only_outer_clean": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "outer_fold_used_for_training": False,
            "outer_fold_used_for_epoch_selection": False,
            "runtime_inputs_use_annotations": False,
            "presence_threshold_tuned": False,
            "presence_threshold": PRESENCE_THRESHOLD,
            "categorical_cardinality_head_exists": False,
            "cardinality_decode": "sum(event_presence>=0.5)",
            "grouping_window_ms_unchanged": 40,
            "offset_model_untouched": True,
        },
        "architecture": {
            "name": "V13.0 causal anonymous event-set decoder",
            "event_queries": EVENT_QUERIES,
            "background_queries": 1,
            "tf_assignment_competitive_across_events": True,
            "candidate_assignment_competitive_across_events": True,
            "event_outputs": ["presence", "birth_time_distribution", "candidate_distribution"],
            "event_order_training_only": "ascending birth time; stable ties",
            "string_pitch_identity_required_at_runtime": False,
            "string_pitch_time_auxiliaries_training_only": True,
            "headline_candidate_realization": "frozen V9+ ranking; isolates event-derived cardinality",
            "trainable_parameters": int(model.count_params()),
            "tf_token_shape": list(token_shape),
            "loss_weights": loss_weights,
        },
        "outer_fold": int(args.outer_fold),
        "outer_groups": groups_per_fold[args.outer_fold],
        "data": {
            "outer_clusters": int(len(outer_idx)),
            "outer_train_clusters": int(len(final_fit_idx)),
            "meta_fit_clusters": int(len(meta_fit_idx)),
            "meta_validation_clusters": int(len(meta_val_idx)),
            "meta_validation_fold": int(meta_fold),
            "selected_epochs": int(selected_epochs),
            "train_tracks": int(len(train_split)),
            "validation_tracks_not_evaluated": int(len(validation)),
        },
        "supervision": {
            "assigned_events": supervision.get("assigned_events"),
            "unassigned_events": supervision.get("unassigned_events"),
            "slot_mask_agreement": supervision.get("slot_mask_agreement"),
            "active_slot_time_coverage": supervision.get("active_slot_time_coverage"),
            "cluster_reconstruction": reconstruction,
            "anonymous_event_targets": event_diag,
        },
        "meta_validation": {
            "metrics": meta_metrics,
            "cardinality": v120._card(k[meta_val_idx], meta_pred),
            "mean_presence_by_query": np.mean(meta_presence, axis=0).tolist(),
            "prefix_violation_rate": float(np.mean(meta_prefix)),
        },
        "strata": strata_report,
        "per_true_k": per_k,
        "event_diagnostics": outer_event_diag,
        "presence": {
            "mean_by_query": np.mean(presence_outer, axis=0).tolist(),
            "active_rate_by_query": np.mean(presence_outer >= PRESENCE_THRESHOLD, axis=0).tolist(),
            "prefix_violation_rate": float(np.mean(prefix_outer)),
            "mean_predicted_count": float(np.mean(pred130_outer)),
            "mean_true_count": float(np.mean(ko)),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / f"predictions-fold-{args.outer_fold}.npz",
        schema_version=np.asarray([1], dtype=np.int16),
        outer_fold=np.full(len(outer_idx), args.outer_fold, dtype=np.int16),
        global_index=outer_idx,
        k=ko.astype(np.int16),
        member=members[outer_idx],
        pred101=pred101_outer.astype(np.int16),
        pred102=pred102_outer.astype(np.int16),
        pred104=pred104_outer.astype(np.int16),
        pred130=pred130_outer.astype(np.int16),
        presence=presence_outer.astype(np.float32),
        event_time=event_time_outer.astype(np.float32),
        event_candidate=event_candidate_outer.astype(np.float32),
        prefix_violation=prefix_outer.astype(np.int8),
    )
    model.save_weights(args.output_dir / f"v130-fold-{args.outer_fold}.weights.h5")
    print(json.dumps({
        "outer": args.outer_fold,
        "epochs": selected_epochs,
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v130_f1": report["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"],
        "poly_exact_v104": report["strata"]["aggregate"]["v104"]["cardinality"]["poly_exact_accuracy"],
        "poly_exact_v130": report["strata"]["aggregate"]["v130"]["cardinality"]["poly_exact_accuracy"],
        "prefix_violation": report["presence"]["prefix_violation_rate"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
