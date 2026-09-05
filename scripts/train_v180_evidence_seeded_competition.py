"""V18 evidence-seeded competitive proposal decoder.

V17.6 removed q-specific trainable heads but six fixed anonymous seeds still
re-specialized. V17.7 removed slots entirely, but its 48 independent candidate
Bernoullis exposed a formulation defect: V17.3 six-slot negative mass was
spread across a variable candidate field, starving unmatched candidates of
negative pressure (K6 had zero per-candidate BCE negative weight).

V18 keeps the scientifically sound six-object V17.3 objective, but removes
arbitrary proposal identity. Six proposal anchors are selected from evidence:
the six valid causal candidates with highest frozen V8.8 fused score. Their
encoded candidate features seed a fully shared decoder. The proposals then
compete jointly for all candidate and TF evidence plus background, so an anchor
is not hard ownership and can move to another candidate.

Treatment relative to canonical V17.3:
  * six proposals remain, so V17.3 mass-preserving weights are algebraically and
    optimization-wise defined on the same six Bernoulli decisions;
  * exact 720 permutation matching and exact Poisson-binomial K NLL (0.35) stay;
  * runtime K remains sum(p >= .5), untuned;
  * fixed/learned anonymous seed identities are absent;
  * proposal identity comes only from current-row causal evidence;
  * all proposal transforms and presence heads are shared;
  * candidates and TF tokens are competitively assigned across proposals plus
    a background channel before objectness is decided;
  * Locked12/historical validation are untouched.
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
from scripts.train_v100_spectral_string_slots import TIME_FRAMES
from scripts.train_v90_structured_cluster_cardinality import MAX_CANDIDATES
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v173_poibin_count_consistency as v173
from scripts import train_v176_shared_set_decoder as v176

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
BASE_ARM = "mass_permutation"
MODEL_KEY = "v180_evidence_seeded"
QUERY_DIM = 96
EPS = 1e-6
FUSED_FEATURE_FROM_END = 3  # sequence[..., -3] is frozen V8.8 fused score; -2/-1 are relative time.


class V180Error(RuntimeError):
    pass


def _input_by_name(model, name):
    for tensor in model.inputs:
        if tensor.name.split(":", 1)[0] == name:
            return tensor
    raise V180Error(f"model input not found: {name}")


def _other_union(tf, assignment, q: int):
    events = assignment[:, :, :EVENT_QUERIES]
    others = tf.concat([events[:, :, :q], events[:, :, q + 1 :]], axis=-1)
    return 1.0 - tf.reduce_prod(1.0 - tf.clip_by_value(others, 0.0, 1.0), axis=-1)


def _anchor_indices_np(sequence: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference NumPy top-6 evidence-anchor selection used by preflight/audit."""
    sequence = np.asarray(sequence, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    fused = sequence[:, :, -FUSED_FEATURE_FROM_END]
    score = np.where(mask > 0.5, fused, -1e9)
    order = np.argsort(-score, axis=1, kind="stable")[:, :EVENT_QUERIES]
    valid = np.take_along_axis(mask, order, axis=1)
    return order.astype(np.int32), valid.astype(np.float32)


def _build_model(spec: dict):
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

    # Shared candidate memory. The frozen fused score is still present in the
    # raw sequence and is used only for non-trainable evidence-anchor selection.
    cand = keras.layers.TimeDistributed(
        keras.layers.LayerNormalization(), name="v180_candidate_norm"
    )(candidate_set)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v180_candidate_hidden1"
    )(cand)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v180_candidate_hidden2"
    )(cand)

    fused = keras.layers.Lambda(
        lambda x: x[:, :, -FUSED_FEATURE_FROM_END], name="v180_frozen_fused_score"
    )(candidate_set)
    masked_fused = keras.layers.Lambda(
        lambda z: tf.where(z[1] > 0.5, z[0], tf.cast(-1e9, z[0].dtype)),
        name="v180_masked_fused_score",
    )([fused, candidate_mask])
    anchor_indices = keras.layers.Lambda(
        lambda x: tf.math.top_k(x, k=EVENT_QUERIES, sorted=True).indices,
        name="v180_evidence_anchor_indices",
    )(masked_fused)
    anchor_valid = keras.layers.Lambda(
        lambda z: tf.gather(z[0], z[1], batch_dims=1),
        name="v180_evidence_anchor_valid",
    )([candidate_mask, anchor_indices])
    anchor_seed = keras.layers.Lambda(
        lambda z: tf.gather(z[0], z[1], batch_dims=1),
        name="v180_evidence_anchor_seed",
    )([cand, anchor_indices])
    anchor_mask_e = keras.layers.Lambda(
        lambda m: tf.cast(m[:, :, None] > 0.5, tf.float32),
        name="v180_anchor_mask_expand",
    )(anchor_valid)
    anchor_seed = keras.layers.Multiply(name="v180_anchor_seed_masked")([anchor_seed, anchor_mask_e])

    global_context = keras.layers.Dense(
        QUERY_DIM, activation="relu", name="v180_shared_global_context"
    )(candidate_context)
    global_context = keras.layers.Lambda(
        lambda x: tf.tile(x[:, None, :], [1, EVENT_QUERIES, 1]),
        name="v180_global_context_broadcast",
    )(global_context)
    proposals = keras.layers.Add(name="v180_anchor_plus_context")([anchor_seed, global_context])
    proposals = keras.layers.Multiply(name="v180_anchor_context_masked")([proposals, anchor_mask_e])
    proposals = keras.layers.LayerNormalization(name="v180_anchor_norm")(proposals)

    candidate_attention_mask = keras.layers.Lambda(
        lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, EVENT_QUERIES, 1]),
        name="v180_candidate_attention_mask",
    )(candidate_mask)

    ca = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v180_shared_candidate_cross_attention"
    )(proposals, cand, attention_mask=candidate_attention_mask)
    proposals = keras.layers.Add(name="v180_candidate_cross_residual")([proposals, ca])
    proposals = keras.layers.LayerNormalization(name="v180_candidate_cross_norm")(proposals)
    proposals = keras.layers.Multiply(name="v180_candidate_cross_masked")([proposals, anchor_mask_e])

    ta = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v180_shared_tf_cross_attention"
    )(proposals, tf_tokens)
    proposals = keras.layers.Add(name="v180_tf_cross_residual")([proposals, ta])
    proposals = keras.layers.LayerNormalization(name="v180_tf_cross_norm")(proposals)
    proposals = keras.layers.Multiply(name="v180_tf_cross_masked")([proposals, anchor_mask_e])

    proposal_self_mask = keras.layers.Lambda(
        lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, EVENT_QUERIES, 1]),
        name="v180_proposal_self_mask",
    )(anchor_valid)
    sa = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v180_shared_proposal_self_attention"
    )(proposals, proposals, attention_mask=proposal_self_mask)
    proposals = keras.layers.Add(name="v180_proposal_self_residual")([proposals, sa])
    proposals = keras.layers.LayerNormalization(name="v180_proposal_self_norm")(proposals)
    proposals = keras.layers.Multiply(name="v180_proposal_self_masked")([proposals, anchor_mask_e])

    ff = keras.layers.Dense(192, activation="relu", name="v180_shared_proposal_ff1")(proposals)
    ff = keras.layers.Dropout(0.08, name="v180_shared_proposal_dropout")(ff)
    ff = keras.layers.Dense(QUERY_DIM, name="v180_shared_proposal_ff2")(ff)
    proposals = keras.layers.Add(name="v180_proposal_ff_residual")([proposals, ff])
    proposals = keras.layers.LayerNormalization(name="v180_proposal_ff_norm")(proposals)
    proposals = keras.layers.Multiply(name="v180_proposal_ff_masked")([proposals, anchor_mask_e])

    cand_keys = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v180_candidate_keys"
    )(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v180_tf_keys")(tf_tokens)
    score_queries = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v180_shared_score_query"
    )(proposals)

    tf_event_scores = keras.layers.Lambda(
        lambda z: tf.einsum("btd,bqd->btq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v180_tf_event_scores_raw",
    )([tf_keys, score_queries])
    tf_event_scores = keras.layers.Lambda(
        lambda z: tf.where(z[1][:, None, :] > 0.5, z[0], tf.cast(-1e9, z[0].dtype)),
        name="v180_tf_event_scores",
    )([tf_event_scores, anchor_valid])
    tf_background = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="v180_tf_background_score"
    )(keras.layers.Dense(1, name="v180_tf_background_dense")(tf_tokens))
    tf_scores = keras.layers.Concatenate(axis=-1, name="v180_tf_score_stack")([
        tf_event_scores,
        keras.layers.Lambda(lambda x: x[:, :, None], name="v180_tf_background_expand")(tf_background),
    ])
    tf_assignment = keras.layers.Softmax(axis=-1, name="v180_tf_competition")(tf_scores)

    cand_event_scores = keras.layers.Lambda(
        lambda z: tf.einsum("bcd,bqd->bcq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v180_candidate_event_scores_raw",
    )([cand_keys, score_queries])
    cand_event_scores = keras.layers.Lambda(
        lambda z: tf.where(z[1][:, None, :] > 0.5, z[0], tf.cast(-1e9, z[0].dtype)),
        name="v180_candidate_event_scores",
    )([cand_event_scores, anchor_valid])
    cand_background = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="v180_candidate_background_score"
    )(keras.layers.TimeDistributed(keras.layers.Dense(1), name="v180_candidate_background_dense")(cand))
    cand_scores = keras.layers.Concatenate(axis=-1, name="v180_candidate_score_stack")([
        cand_event_scores,
        keras.layers.Lambda(lambda x: x[:, :, None], name="v180_candidate_background_expand")(cand_background),
    ])
    cand_assignment = keras.layers.Softmax(axis=-1, name="v180_candidate_competition")(cand_scores)

    token_freq = int(token_shape[1])
    local_features, time_outputs, candidate_outputs = [], [], []
    shared_local_norm = keras.layers.LayerNormalization(name="v180_shared_local_norm")
    shared_local_hidden = keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1.5e-3),
        name="v180_shared_local_hidden",
    )

    for q in range(EVENT_QUERIES):
        own_tf = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v180_event_{q}_tf_weights"
        )(tf_assignment)
        tf_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v180_event_{q}_tf_distribution",
        )(own_tf)
        tf_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v180_event_{q}_tf_pool",
        )([tf_tokens, tf_dist])

        raw_cand = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v180_event_{q}_candidate_weights_raw"
        )(cand_assignment)
        own_cand = keras.layers.Multiply(name=f"v180_event_{q}_candidate_weights")(
            [raw_cand, candidate_mask]
        )
        cand_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"event_candidate_{q}",
        )(own_cand)
        cand_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v180_event_{q}_candidate_pool",
        )([cand, cand_dist])

        others_cand = keras.layers.Lambda(
            lambda a, i=q: _other_union(tf, a, i), name=f"v180_event_{q}_other_candidate_coverage_raw"
        )(cand_assignment)
        others_cand = keras.layers.Multiply(name=f"v180_event_{q}_other_candidate_coverage")(
            [others_cand, candidate_mask]
        )
        cand_overlap = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True)
            / (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v180_event_{q}_candidate_overlap",
        )([own_cand, others_cand])
        cand_mass = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v180_event_{q}_candidate_mass",
        )([own_cand, candidate_mask])
        tf_mass = keras.layers.Lambda(
            lambda w: tf.reduce_mean(w, axis=1, keepdims=True), name=f"v180_event_{q}_tf_mass"
        )(own_tf)

        tf_grid = keras.layers.Reshape(
            (TIME_FRAMES, token_freq), name=f"v180_event_{q}_tf_grid"
        )(own_tf)
        time_mass = keras.layers.Lambda(
            lambda a: tf.reduce_sum(a, axis=2), name=f"v180_event_{q}_time_mass"
        )(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS),
            name=f"event_time_{q}",
        )(time_mass)

        proposal_q = keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"v180_event_{q}_proposal"
        )(proposals)
        anchor_valid_q = keras.layers.Lambda(
            lambda x, i=q: x[:, i : i + 1], name=f"v180_event_{q}_anchor_valid"
        )(anchor_valid)
        local = keras.layers.Concatenate(name=f"v180_event_{q}_local_feature")([
            candidate_context,
            proposal_q,
            tf_latent,
            cand_latent,
            tf_mass,
            cand_mass,
            cand_overlap,
            anchor_valid_q,
        ])
        local = shared_local_norm(local)
        local = shared_local_hidden(local)
        local_features.append(local)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_dist)

    proposal_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="v180_proposal_stack"
    )(local_features)
    reconciled_att = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.05, name="v180_global_reconciliation_attention"
    )(proposal_stack, proposal_stack, attention_mask=proposal_self_mask)
    reconciled = keras.layers.Add(name="v180_global_reconciliation_residual")([
        proposal_stack, reconciled_att
    ])
    reconciled = keras.layers.LayerNormalization(name="v180_global_reconciliation_norm")(reconciled)
    reconciled = keras.layers.Multiply(name="v180_global_reconciliation_masked")([
        reconciled, anchor_mask_e
    ])
    rff = keras.layers.Dense(128, activation="relu", name="v180_global_ff1")(reconciled)
    rff = keras.layers.Dropout(0.08, name="v180_global_dropout")(rff)
    rff = keras.layers.Dense(128, activation="relu", name="v180_global_ff2")(rff)
    reconciled = keras.layers.Add(name="v180_global_ff_residual")([reconciled, rff])
    reconciled = keras.layers.LayerNormalization(name="v180_global_ff_norm")(reconciled)
    reconciled = keras.layers.Multiply(name="v180_global_ff_masked")([reconciled, anchor_mask_e])

    presence_hidden = keras.layers.TimeDistributed(
        keras.layers.Dense(64, activation="relu"), name="v180_shared_presence_hidden"
    )(reconciled)
    presence_raw = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation="sigmoid"), name="v180_shared_presence"
    )(presence_hidden)
    presence_stack = keras.layers.Multiply(name="v180_presence_anchor_masked")([
        presence_raw, anchor_mask_e
    ])

    presence_outputs = [
        keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"event_present_{q}")(presence_stack)
        for q in range(EVENT_QUERIES)
    ]
    count_norm = keras.layers.Lambda(
        lambda p: tf.reduce_sum(p, axis=1) / float(EVENT_QUERIES), name="event_count_norm"
    )(presence_stack)

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output

    set_slots = []
    for q in range(EVENT_QUERIES):
        present = presence_outputs[q]
        valid = keras.layers.Lambda(
            lambda x: tf.ones_like(x), name=f"v180_event_{q}_valid_placeholder"
        )(present)
        packed = keras.layers.Concatenate(name=f"v180_event_{q}_set_vector")([
            present, valid, time_outputs[q], candidate_outputs[q]
        ])
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = present
        outputs[f"event_time_{q}"] = time_outputs[q]
        outputs[f"event_candidate_{q}"] = candidate_outputs[q]
    outputs["event_set"] = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="event_set"
    )(set_slots)
    outputs["event_count_norm"] = count_norm

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["event_set"] = v173._set_loss(spec)
    loss["event_count_norm"] = "mse"
    lw = {f"string_{slot}": 0.18 for slot in range(SLOT_COUNT)}
    lw.update({f"pitch_{slot}": 0.04 for slot in range(SLOT_COUNT)})
    lw.update({f"time_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.0

    model = keras.Model(base.inputs, outputs, name="v180_evidence_seeded_competition")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=lw)
    return model, lw, token_shape


def _postprocess(args, report, ctx):
    report = v173._postprocess(args, report, ctx)
    v171._rename_report(report, v173.MODEL_KEY, MODEL_KEY)
    inherited = report.pop("v173")
    inherited.pop("final_model_presence_gradient_mass", None)
    inherited.pop("final_model_presence_gradient_mass_error", None)

    report["protocol"].update({
        "v180_evidence_seeded_competition": True,
        "v180_base_version": "V17.3 objective / V17.6 shared proposal mechanics",
        "v180_only_scientific_treatment": "fixed anonymous proposal identity -> top-6 frozen-fused evidence anchors",
        "fixed_anonymous_seed_count": 0,
        "learned_anonymous_seed_count": 0,
        "proposal_anchor_source": "top-6 valid raw candidates by frozen V8.8 fused score",
        "proposal_anchor_selection_trainable": False,
        "proposal_anchor_hard_ownership": False,
        "shared_candidate_cross_attention": True,
        "shared_tf_cross_attention": True,
        "competitive_candidate_assignment_with_background": True,
        "competitive_tf_assignment_with_background": True,
        "shared_local_feature_transform": True,
        "shared_presence_classifier": True,
        "v173_poisson_binomial_count_objective_unchanged": True,
        "v173_count_nll_weight": v173.COUNT_NLL_WEIGHT,
        "mass_preserving_exchangeable_weights_unchanged": True,
        "exact_720_truth_matching_unchanged": True,
        "runtime_count_decode_unchanged_from_v173": True,
        "runtime_presence_threshold": PRESENCE_THRESHOLD,
        "runtime_presence_threshold_tuned": False,
        "categorical_cardinality_head_exists": False,
        "v175_injective_ownership_used": False,
        "historical_validation_or_locked12_indexed_or_evaluated": False,
    })

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    old_pred = "pred173_poibin"
    if old_pred not in data:
        raise V180Error(f"missing inherited prediction key {old_pred}")
    data["pred180_evidence_seeded"] = data.pop(old_pred)

    presence = np.asarray(data["presence"], dtype=np.float64)
    candidate = np.asarray(data["event_candidate"], dtype=np.float64)
    active_rates = np.mean(presence >= PRESENCE_THRESHOLD, axis=0)
    occup = np.asarray([
        inherited["outer_match_occupancy"][str(q)]["matched_object_rate"]
        for q in range(EVENT_QUERIES)
    ], dtype=np.float64)
    corr = float(np.corrcoef(active_rates, occup)[0, 1]) if np.std(active_rates) and np.std(occup) else 0.0

    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    seq = np.asarray(ctx["cache"]["sequence"], dtype=np.float32)[outer]
    mask = np.asarray(ctx["cache"]["mask"], dtype=np.float32)[outer]
    anchor_idx, anchor_valid = _anchor_indices_np(seq, mask)
    valid_count = np.sum(mask > 0.5, axis=1).astype(np.int32)
    true_k = np.asarray(ctx["k"], dtype=np.int32)[outer]
    hard_k = np.sum(presence >= PRESENCE_THRESHOLD, axis=1).astype(np.int32)

    duplicate_by_k = {}
    for value in range(2, EVENT_QUERIES + 1):
        exact = (true_k == value) & (hard_k == value)
        duplicate_by_k[str(value)] = {
            "rows": int(np.sum(exact)),
            "raw_candidate_duplicate_argmax_exact_count": v176._duplicate_rate(candidate, presence, exact),
        }

    architecture = {
        "outer_active_rate_by_anchor_rank": active_rates.tolist(),
        "outer_activity_gini": v176._gini(active_rates),
        "outer_effective_active_slots": v176._effective_slots(active_rates),
        "outer_matched_object_rate_by_anchor_rank": occup.tolist(),
        "outer_active_occupancy_correlation": corr,
        "soft_presence_mass": float(np.mean(np.sum(presence, axis=1))),
        "mean_valid_candidate_count": float(np.mean(valid_count)),
        "mean_valid_anchor_count": float(np.mean(np.sum(anchor_valid > 0.5, axis=1))),
        "rows_with_fewer_than_6_valid_candidates": int(np.sum(valid_count < EVENT_QUERIES)),
        "rows_candidate_infeasible_c_lt_k": int(np.sum(valid_count < true_k)),
        "anchor_index_mean_by_rank": np.mean(anchor_idx, axis=0).tolist(),
        "raw_candidate_duplicates_exact_count_by_true_k": duplicate_by_k,
    }
    report["v180"] = {**inherited, "model_key": MODEL_KEY, "architecture": architecture}

    np.savez_compressed(npz_path, **data)
    old_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v180-evidence-seeded-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def train_fold(args):
    if args.seed != DEFAULT_SEED:
        raise V180Error(f"V18 requires seed {DEFAULT_SEED}, got {args.seed}")
    if args.arm != BASE_ARM:
        raise V180Error(f"V18 only supports base arm {BASE_ARM!r}")

    ctx = v172._fold_context(args)
    stage_specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V180Error(f"unexpected model build call {i + 1}")
        calls["count"] += 1
        return _build_model(stage_specs[i])

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = v171._targets
        v130._sample_weights = v171._sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights

    if calls["count"] != 2:
        raise V180Error(f"expected exactly 2 model builds, got {calls['count']}")
    report = _postprocess(args, report, ctx)

    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    card = report["strata"]["aggregate"][MODEL_KEY]["cardinality"]
    arch = report["v180"]["architecture"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v180_f1": g["f1"],
        "pred_ref": g["prediction_reference_ratio"],
        "poly_exact": card.get("poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))),
        "activity_gini": arch["outer_activity_gini"],
        "effective_active_slots": arch["outer_effective_active_slots"],
        "active_occupancy_correlation": arch["outer_active_occupancy_correlation"],
        "k2": report["per_true_k"]["2"][MODEL_KEY]["exact"],
        "k3": report["per_true_k"]["3"][MODEL_KEY]["exact"],
        "k4": report["per_true_k"]["4"][MODEL_KEY]["exact"],
        "k5": report["per_true_k"]["5"][MODEL_KEY]["exact"],
        "k6": report["per_true_k"]["6"][MODEL_KEY]["exact"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arm", choices=[BASE_ARM], default=BASE_ARM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
