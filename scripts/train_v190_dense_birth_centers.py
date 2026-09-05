"""V19 dense time-frequency birth-center proposal decoder.

The V19 audit rejected raw-candidate grouping (69.29% exact poly coverage) but
showed that true births are representable as distinct centers on the original
23x64 spectral grid on 99.38% of eligible poly rows and 100% of K4/K5/K6.
V19 therefore forms proposals upstream of the raw candidate set.

Kept from canonical V17.3: six exchangeable Bernoulli object decisions,
mass-preserving weights, exact 720 permutation matching, Poisson-binomial count
NLL=0.35, K=sum(p>=0.5), threshold 0.5 untuned, no categorical K head.
Treatment: a learned 23x64 birth-center distribution is supervised training-only;
its top-6 3x3 local maxima seed a fully shared decoder. Raw candidates remain
auxiliary realization memory and never define object identity. Locked12 remains
untouched.
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
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SLOT_COUNT
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v173_poibin_count_consistency as v173
from scripts import train_v176_shared_set_decoder as v176
from scripts import train_v180_evidence_seeded_competition as v180

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
BASE_ARM = "mass_permutation"
MODEL_KEY = "v190_dense_birth_centers"
PRED_KEY = "pred190_dense_birth_centers"
QUERY_DIM = 96
EPS = 1e-6
CENTER_MAP_WEIGHT = 0.25
CENTER_CELLS = v100.TIME_FRAMES * v100.SPECTRAL_BANDS
_LAST_CENTER_TARGETS = None
_LAST_CENTER_ELIGIBLE = None


class V190Error(RuntimeError):
    pass


def _input_by_name(model, name):
    return v180._input_by_name(model, name)


def _other_union(tf, assignment, q):
    return v180._other_union(tf, assignment, q)


def _nearest_log_band(hz):
    grid = np.geomspace(v100.MIN_HZ, v100.MAX_HZ, v100.SPECTRAL_BANDS).astype(np.float64)
    h = np.log(np.clip(np.asarray(hz, dtype=np.float64), 1e-9, None))[:, None]
    return np.argmin(np.abs(h - np.log(grid)[None, :]), axis=1).astype(np.int32)


def _birth_center_targets(cache, pitch_targets, string_time_targets, k):
    pitch = np.asarray(pitch_targets, dtype=np.float64)
    times = np.asarray(string_time_targets, dtype=np.float64)
    kk = np.asarray(k, dtype=np.int32)
    active = np.asarray(cache["slot_targets"], dtype=np.float32) > 0.5
    target = np.zeros((len(kk), CENTER_CELLS), dtype=np.float32)
    eligible = np.zeros(len(kk), dtype=np.float32)
    distinct = np.zeros(len(kk), dtype=np.int16)
    for r in range(len(kk)):
        kr = int(kk[r])
        if kr <= 0:
            continue
        slots = np.flatnonzero(active[r])
        if len(slots) != kr:
            continue
        frame = np.argmax(times[r, slots], axis=1).astype(np.int32)
        midi = pitch[r, slots] * float(v101.PITCH_SCALE)
        if not np.all(np.isfinite(midi)):
            continue
        hz = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
        band = _nearest_log_band(hz)
        flat = frame * int(v100.SPECTRAL_BANDS) + band
        np.add.at(target[r], flat, 1.0)
        total = float(np.sum(target[r]))
        if total <= 0:
            continue
        target[r] /= total
        eligible[r] = 1.0
        distinct[r] = int(len(np.unique(flat)))
    return target, eligible, distinct


def _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate):
    global _LAST_CENTER_TARGETS, _LAST_CENTER_ELIGIBLE
    out = v171._targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate)
    center, eligible, _ = _birth_center_targets(cache, pitch_targets, string_time_targets, k)
    out["birth_center_map"] = center
    _LAST_CENTER_TARGETS = center
    _LAST_CENTER_ELIGIBLE = eligible
    return out


def _sample_weights(cache, time_mask, k, event_present, event_valid):
    out = v171._sample_weights(cache, time_mask, k, event_present, event_valid)
    kk = np.asarray(k, dtype=np.int32)
    timed = np.sum(np.asarray(time_mask, dtype=np.float32) > 0.5, axis=1).astype(np.int32)
    out["birth_center_map"] = ((kk > 0) & (timed == kk)).astype(np.float32)
    return out


def _nms_topk_np(prob):
    x = np.asarray(prob, dtype=np.float64)
    grid = x.reshape((-1, v100.TIME_FRAMES, v100.SPECTRAL_BANDS))
    pad = np.pad(grid, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    neigh = [pad[:, dt:dt+v100.TIME_FRAMES, df:df+v100.SPECTRAL_BANDS] for dt in range(3) for df in range(3)]
    pooled = np.maximum.reduce(neigh)
    peak = grid >= (pooled - 1e-12)
    score = np.where(peak, grid, grid - 1e6).reshape((-1, CENTER_CELLS))
    part = np.argpartition(-score, kth=EVENT_QUERIES-1, axis=1)[:, :EVENT_QUERIES]
    return np.sort(part.astype(np.int32), axis=1)


def _center_diagnostics(prob, target, eligible, k):
    prob = np.asarray(prob, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=np.float32) > 0.5
    kk = np.asarray(k, dtype=np.int32)
    chosen = _nms_topk_np(prob)
    exact = np.zeros(len(kk), dtype=bool)
    frac = np.zeros(len(kk), dtype=np.float64)
    for r in range(len(kk)):
        if not eligible[r]:
            continue
        truth = np.flatnonzero(target[r] > 0.0)
        sel = set(int(x) for x in chosen[r])
        hits = sum(int(t) in sel for t in truth)
        exact[r] = bool(len(truth) and hits == len(truth))
        frac[r] = hits / float(max(1, len(truth)))
    pos = eligible & (kk > 0)
    poly = eligible & (kk >= 2)
    out = {
        "eligible_positive_rows": int(np.sum(pos)),
        "eligible_poly_rows": int(np.sum(poly)),
        "top6_exact_center_coverage_positive": float(np.mean(exact[pos])) if np.any(pos) else None,
        "top6_exact_center_coverage_poly": float(np.mean(exact[poly])) if np.any(poly) else None,
        "top6_mean_center_hit_fraction_poly": float(np.mean(frac[poly])) if np.any(poly) else None,
        "per_true_k": {},
    }
    for value in range(1, EVENT_QUERIES + 1):
        m = eligible & (kk == value)
        out["per_true_k"][str(value)] = {
            "rows": int(np.sum(m)),
            "top6_exact_center_coverage": float(np.mean(exact[m])) if np.any(m) else None,
            "top6_mean_center_hit_fraction": float(np.mean(frac[m])) if np.any(m) else None,
        }
    return out


def _build_model(spec):
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
    spectral = _input_by_name(base, "spectral_map")

    cand = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="v190_candidate_norm")(candidate_set)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v190_candidate_hidden1")(cand)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v190_candidate_hidden2")(cand)

    tc = np.linspace(-1.0, 1.0, v100.TIME_FRAMES, dtype=np.float32)[:, None]
    fc = np.linspace(-1.0, 1.0, v100.SPECTRAL_BANDS, dtype=np.float32)[None, :]
    coord = np.stack([np.broadcast_to(tc, (v100.TIME_FRAMES, v100.SPECTRAL_BANDS)), np.broadcast_to(fc, (v100.TIME_FRAMES, v100.SPECTRAL_BANDS))], axis=-1).astype(np.float32)
    coord_const = tf.constant(coord, dtype=tf.float32)
    coords = keras.layers.Lambda(lambda s: tf.tile(coord_const[None, :, :, :], [tf.shape(s)[0], 1, 1, 1]), name="v190_dense_coordinates")(spectral)
    dense = keras.layers.LayerNormalization(axis=-1, name="v190_dense_channel_norm")(spectral)
    dense = keras.layers.Concatenate(axis=-1, name="v190_dense_plus_coordinates")([dense, coords])
    dense = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="v190_dense_conv1")(dense)
    dense = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="v190_dense_conv2")(dense)
    dense_features = keras.layers.Conv2D(QUERY_DIM, (3, 3), padding="same", activation="relu", name="v190_dense_conv3")(dense)
    logits4 = keras.layers.Conv2D(1, (1, 1), padding="same", name="v190_birth_center_logits")(dense_features)
    logits = keras.layers.Reshape((CENTER_CELLS,), name="v190_birth_center_logits_flat")(logits4)
    center_map = keras.layers.Softmax(name="birth_center_map")(logits)

    pooled = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same", name="v190_center_local_pool")(logits4)
    peak4 = keras.layers.Lambda(lambda z: z[0] >= (z[1] - 1e-6), name="v190_center_peak_mask")([logits4, pooled])
    peak = keras.layers.Reshape((CENTER_CELLS,), name="v190_center_peak_mask_flat")(peak4)
    nms_score = keras.layers.Lambda(lambda z: tf.where(z[1], z[0], z[0] - tf.cast(1e6, z[0].dtype)), name="v190_center_nms_score")([logits, peak])
    raw_idx = keras.layers.Lambda(lambda x: tf.math.top_k(x, k=EVENT_QUERIES, sorted=False).indices, name="v190_center_top6_unsorted")(nms_score)
    anchor_idx = keras.layers.Lambda(lambda x: tf.sort(x, axis=-1), name="v190_center_anchor_indices")(raw_idx)
    dense_flat = keras.layers.Reshape((CENTER_CELLS, QUERY_DIM), name="v190_dense_feature_flat")(dense_features)
    anchor_seed = keras.layers.Lambda(lambda z: tf.gather(z[0], z[1], batch_dims=1), name="v190_dense_anchor_seed")([dense_flat, anchor_idx])

    gc = keras.layers.Dense(QUERY_DIM, activation="relu", name="v190_shared_global_context")(candidate_context)
    gc = keras.layers.Lambda(lambda x: tf.tile(x[:, None, :], [1, EVENT_QUERIES, 1]), name="v190_global_context_broadcast")(gc)
    proposals = keras.layers.LayerNormalization(name="v190_anchor_norm")(keras.layers.Add(name="v190_anchor_plus_context")([anchor_seed, gc]))
    cam = keras.layers.Lambda(lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, EVENT_QUERIES, 1]), name="v190_candidate_attention_mask")(candidate_mask)
    ca = keras.layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.05, name="v190_shared_candidate_cross_attention")(proposals, cand, attention_mask=cam)
    proposals = keras.layers.LayerNormalization(name="v190_candidate_cross_norm")(keras.layers.Add(name="v190_candidate_cross_residual")([proposals, ca]))
    ta = keras.layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.05, name="v190_shared_tf_cross_attention")(proposals, tf_tokens)
    proposals = keras.layers.LayerNormalization(name="v190_tf_cross_norm")(keras.layers.Add(name="v190_tf_cross_residual")([proposals, ta]))
    sa = keras.layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.05, name="v190_shared_proposal_self_attention")(proposals, proposals)
    proposals = keras.layers.LayerNormalization(name="v190_proposal_self_norm")(keras.layers.Add(name="v190_proposal_self_residual")([proposals, sa]))
    ff = keras.layers.Dense(192, activation="relu", name="v190_shared_proposal_ff1")(proposals)
    ff = keras.layers.Dropout(0.08, name="v190_shared_proposal_dropout")(ff)
    ff = keras.layers.Dense(QUERY_DIM, name="v190_shared_proposal_ff2")(ff)
    proposals = keras.layers.LayerNormalization(name="v190_proposal_ff_norm")(keras.layers.Add(name="v190_proposal_ff_residual")([proposals, ff]))

    cand_keys = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, use_bias=False), name="v190_candidate_keys")(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v190_tf_keys")(tf_tokens)
    score_q = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, use_bias=False), name="v190_shared_score_query")(proposals)
    tf_es = keras.layers.Lambda(lambda z: tf.einsum("btd,bqd->btq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)), name="v190_tf_event_scores")([tf_keys, score_q])
    tf_bg = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="v190_tf_background_score")(keras.layers.Dense(1, name="v190_tf_background_dense")(tf_tokens))
    tf_scores = keras.layers.Concatenate(axis=-1, name="v190_tf_score_stack")([tf_es, keras.layers.Lambda(lambda x: x[:, :, None], name="v190_tf_background_expand")(tf_bg)])
    tf_assign = keras.layers.Softmax(axis=-1, name="v190_tf_competition")(tf_scores)
    ce = keras.layers.Lambda(lambda z: tf.einsum("bcd,bqd->bcq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)), name="v190_candidate_event_scores")([cand_keys, score_q])
    cbg = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="v190_candidate_background_score")(keras.layers.TimeDistributed(keras.layers.Dense(1), name="v190_candidate_background_dense")(cand))
    cscore = keras.layers.Concatenate(axis=-1, name="v190_candidate_score_stack")([ce, keras.layers.Lambda(lambda x: x[:, :, None], name="v190_candidate_background_expand")(cbg)])
    cand_assign = keras.layers.Softmax(axis=-1, name="v190_candidate_competition")(cscore)

    token_freq = int(token_shape[1])
    local_features, time_outputs, candidate_outputs = [], [], []
    shared_norm = keras.layers.LayerNormalization(name="v190_shared_local_norm")
    shared_hidden = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1.5e-3), name="v190_shared_local_hidden")
    for q in range(EVENT_QUERIES):
        own_tf = keras.layers.Lambda(lambda a, i=q: a[:, :, i], name=f"v190_event_{q}_tf_weights")(tf_assign)
        tf_dist = keras.layers.Lambda(lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS), name=f"v190_event_{q}_tf_distribution")(own_tf)
        tf_latent = keras.layers.Lambda(lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1), name=f"v190_event_{q}_tf_pool")([tf_tokens, tf_dist])
        raw_c = keras.layers.Lambda(lambda a, i=q: a[:, :, i], name=f"v190_event_{q}_candidate_weights_raw")(cand_assign)
        own_c = keras.layers.Multiply(name=f"v190_event_{q}_candidate_weights")([raw_c, candidate_mask])
        cand_dist = keras.layers.Lambda(lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS), name=f"event_candidate_{q}")(own_c)
        cand_latent = keras.layers.Lambda(lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1), name=f"v190_event_{q}_candidate_pool")([cand, cand_dist])
        others = keras.layers.Lambda(lambda a, i=q: _other_union(tf, a, i), name=f"v190_event_{q}_other_candidate_coverage_raw")(cand_assign)
        others = keras.layers.Multiply(name=f"v190_event_{q}_other_candidate_coverage")([others, candidate_mask])
        overlap = keras.layers.Lambda(lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True) / (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS), name=f"v190_event_{q}_candidate_overlap")([own_c, others])
        cmass = keras.layers.Lambda(lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True) / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS), name=f"v190_event_{q}_candidate_mass")([own_c, candidate_mask])
        tmass = keras.layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True), name=f"v190_event_{q}_tf_mass")(own_tf)
        tf_grid = keras.layers.Reshape((v100.TIME_FRAMES, token_freq), name=f"v190_event_{q}_tf_grid")(own_tf)
        time_mass = keras.layers.Lambda(lambda a: tf.reduce_sum(a, axis=2), name=f"v190_event_{q}_time_mass")(tf_grid)
        time_dist = keras.layers.Lambda(lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS), name=f"event_time_{q}")(time_mass)
        pq = keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"v190_event_{q}_proposal")(proposals)
        local = shared_hidden(shared_norm(keras.layers.Concatenate(name=f"v190_event_{q}_local_feature")([candidate_context, pq, tf_latent, cand_latent, tmass, cmass, overlap])))
        local_features.append(local); time_outputs.append(time_dist); candidate_outputs.append(cand_dist)

    stack = keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="v190_proposal_stack")(local_features)
    ra = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.05, name="v190_global_reconciliation_attention")(stack, stack)
    rec = keras.layers.LayerNormalization(name="v190_global_reconciliation_norm")(keras.layers.Add(name="v190_global_reconciliation_residual")([stack, ra]))
    rff = keras.layers.Dense(128, activation="relu", name="v190_global_ff1")(rec)
    rff = keras.layers.Dropout(0.08, name="v190_global_dropout")(rff)
    rff = keras.layers.Dense(128, activation="relu", name="v190_global_ff2")(rff)
    rec = keras.layers.LayerNormalization(name="v190_global_ff_norm")(keras.layers.Add(name="v190_global_ff_residual")([rec, rff]))
    ph = keras.layers.TimeDistributed(keras.layers.Dense(64, activation="relu"), name="v190_shared_presence_hidden")(rec)
    pstack = keras.layers.TimeDistributed(keras.layers.Dense(1, activation="sigmoid"), name="v190_shared_presence")(ph)
    presence = [keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"event_present_{q}")(pstack) for q in range(EVENT_QUERIES)]
    count_norm = keras.layers.Lambda(lambda p: tf.reduce_sum(p, axis=1) / float(EVENT_QUERIES), name="event_count_norm")(pstack)

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    set_slots = []
    for q in range(EVENT_QUERIES):
        valid = keras.layers.Lambda(lambda x: tf.ones_like(x), name=f"v190_event_{q}_valid_placeholder")(presence[q])
        packed = keras.layers.Concatenate(name=f"v190_event_{q}_set_vector")([presence[q], valid, time_outputs[q], candidate_outputs[q]])
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = presence[q]; outputs[f"event_time_{q}"] = time_outputs[q]; outputs[f"event_candidate_{q}"] = candidate_outputs[q]
    outputs["event_set"] = keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="event_set")(set_slots)
    outputs["event_count_norm"] = count_norm
    outputs["birth_center_map"] = center_map

    loss = {f"string_{s}": "binary_crossentropy" for s in range(SLOT_COUNT)}
    loss.update({f"pitch_{s}": "mse" for s in range(SLOT_COUNT)})
    loss.update({f"time_{s}": keras.losses.KLDivergence() for s in range(SLOT_COUNT)})
    loss["event_set"] = v173._set_loss(spec); loss["event_count_norm"] = "mse"; loss["birth_center_map"] = "categorical_crossentropy"
    lw = {f"string_{s}": 0.18 for s in range(SLOT_COUNT)}
    lw.update({f"pitch_{s}": 0.04 for s in range(SLOT_COUNT)}); lw.update({f"time_{s}": 0.10 for s in range(SLOT_COUNT)})
    lw["event_set"] = 1.0; lw["event_count_norm"] = 0.0; lw["birth_center_map"] = CENTER_MAP_WEIGHT
    model = keras.Model(base.inputs, outputs, name="v190_dense_birth_center_decoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=lw)
    return model, lw, token_shape


def _postprocess(args, report, ctx, center_diag):
    report = v173._postprocess(args, report, ctx)
    v171._rename_report(report, v173.MODEL_KEY, MODEL_KEY)
    inherited = report.pop("v173")
    inherited.pop("final_model_presence_gradient_mass", None); inherited.pop("final_model_presence_gradient_mass_error", None)
    report["protocol"].update({
        "v190_dense_birth_centers": True,
        "v190_only_scientific_treatment": "learned 23x64 dense birth-center local maxima form proposals; candidates are auxiliary memory",
        "dense_center_grid": [int(v100.TIME_FRAMES), int(v100.SPECTRAL_BANDS)],
        "dense_center_supervision_training_only": True,
        "dense_center_loss": "categorical_crossentropy",
        "dense_center_loss_weight": CENTER_MAP_WEIGHT,
        "dense_center_loss_weight_tuned": False,
        "dense_center_anchor_selection": "top-6 3x3 local maxima then spatial sort",
        "fixed_anonymous_seed_count": 0, "learned_anonymous_seed_count": 0,
        "raw_candidate_is_object_identity": False,
        "v173_poisson_binomial_count_objective_unchanged": True, "v173_count_nll_weight": v173.COUNT_NLL_WEIGHT,
        "mass_preserving_exchangeable_weights_unchanged": True, "exact_720_truth_matching_unchanged": True,
        "runtime_count_decode_unchanged_from_v173": True, "runtime_presence_threshold": PRESENCE_THRESHOLD,
        "runtime_presence_threshold_tuned": False, "categorical_cardinality_head_exists": False,
        "v175_injective_ownership_used": False, "historical_validation_or_locked12_indexed_or_evaluated": False,
    })
    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z: data = {key: np.asarray(z[key]) for key in z.files}
    if "pred173_poibin" not in data: raise V190Error("missing inherited pred173_poibin")
    data[PRED_KEY] = data.pop("pred173_poibin")
    presence = np.asarray(data["presence"], dtype=np.float64); candidate = np.asarray(data["event_candidate"], dtype=np.float64)
    active = np.mean(presence >= PRESENCE_THRESHOLD, axis=0)
    occup = np.asarray([inherited["outer_match_occupancy"][str(q)]["matched_object_rate"] for q in range(EVENT_QUERIES)], dtype=np.float64)
    corr = float(np.corrcoef(active, occup)[0, 1]) if np.std(active) and np.std(occup) else 0.0
    outer = np.asarray(ctx["outer_idx"], dtype=np.int64); true_k = np.asarray(ctx["k"], dtype=np.int32)[outer]; hard_k = np.sum(presence >= PRESENCE_THRESHOLD, axis=1).astype(np.int32)
    dup = {}
    for value in range(2, EVENT_QUERIES+1):
        exact = (true_k == value) & (hard_k == value)
        dup[str(value)] = {"rows": int(np.sum(exact)), "raw_candidate_duplicate_argmax_exact_count": v176._duplicate_rate(candidate, presence, exact)}
    arch = {
        "outer_active_rate_by_proposal_position": active.tolist(), "outer_activity_gini": v176._gini(active),
        "outer_effective_active_slots": v176._effective_slots(active), "outer_matched_object_rate_by_proposal_position": occup.tolist(),
        "outer_active_occupancy_correlation": corr, "soft_presence_mass": float(np.mean(np.sum(presence, axis=1))),
        "raw_candidate_duplicates_exact_count_by_true_k": dup, "dense_center_diagnostics": center_diag,
    }
    report["v190"] = {**inherited, "model_key": MODEL_KEY, "architecture": arch}
    np.savez_compressed(npz_path, **data)
    old_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"; new_w = args.output_dir / f"v190-dense-birth-centers-fold-{args.outer_fold}.weights.h5"
    if old_w.exists(): old_w.replace(new_w)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def train_fold(args):
    global _LAST_CENTER_TARGETS, _LAST_CENTER_ELIGIBLE
    if args.seed != DEFAULT_SEED: raise V190Error(f"V19 requires seed {DEFAULT_SEED}")
    if args.arm != BASE_ARM: raise V190Error(f"V19 only supports {BASE_ARM}")
    ctx = v172._fold_context(args); specs = [ctx["meta_spec"], ctx["final_spec"]]; calls = {"count": 0}; built = []
    def builder():
        i = calls["count"]
        if i >= 2: raise V190Error("unexpected model build")
        calls["count"] += 1; t = _build_model(specs[i]); built.append(t[0]); return t
    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder; v130._targets = _targets; v130._sample_weights = _sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights
    if calls["count"] != 2 or _LAST_CENTER_TARGETS is None: raise V190Error("V19 build/target capture failed")
    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    raw = built[-1].predict(v102._inputs(ctx["cache"], outer), batch_size=128, verbose=0)
    cdiag = _center_diagnostics(np.asarray(raw["birth_center_map"]), np.asarray(_LAST_CENTER_TARGETS)[outer], np.asarray(_LAST_CENTER_ELIGIBLE)[outer], np.asarray(ctx["k"], dtype=np.int32)[outer])
    report = _postprocess(args, report, ctx, cdiag)
    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]; card = report["strata"]["aggregate"][MODEL_KEY]["cardinality"]; arch = report["v190"]["architecture"]
    print(json.dumps({"outer": args.outer_fold, "selected_epochs": report["data"]["selected_epochs"], "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"], "v190_f1": g["f1"], "pred_ref": g["prediction_reference_ratio"], "poly_exact": card.get("poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))), "center_top6_exact_poly": cdiag["top6_exact_center_coverage_poly"], "center_top6_hit_fraction_poly": cdiag["top6_mean_center_hit_fraction_poly"], "activity_gini": arch["outer_activity_gini"], "effective_active_slots": arch["outer_effective_active_slots"], "active_occupancy_correlation": arch["outer_active_occupancy_correlation"], "k2": report["per_true_k"]["2"][MODEL_KEY]["exact"], "k3": report["per_true_k"]["3"][MODEL_KEY]["exact"], "k4": report["per_true_k"]["4"][MODEL_KEY]["exact"], "k5": report["per_true_k"]["5"][MODEL_KEY]["exact"], "k6": report["per_true_k"]["6"][MODEL_KEY]["exact"]}, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser(); p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True); p.add_argument("--baseline-eval-dir", type=Path, required=True); p.add_argument("--outer-fold", type=int, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--arm", choices=[BASE_ARM], default=BASE_ARM); p.add_argument("--seed", type=int, default=DEFAULT_SEED); return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
