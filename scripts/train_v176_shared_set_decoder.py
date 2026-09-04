"""V17.6 shared set-equivariant anonymous proposal decoder.

V17.3 showed that exact count consistency is useful but insufficient. V17.5 then
showed that candidate ownership is a real defect (raw duplicate ownership fell
materially and structured duplicates reached zero) while global F1 stayed
essentially unchanged. The next measured bottleneck is therefore proposal
formation itself: V16 still uses six independent query, local-hidden and
presence-head parameter sets, so an exchangeable loss is applied to a
non-exchangeable parameterization.

V17.6 changes that representation, starting from the canonical V17.3 objective:
  * six anonymous proposal seeds are fixed, deterministic and non-trainable;
  * one shared decoder cross-attends every seed to candidate and TF evidence;
  * candidate/TF competition and leave-one-out coverage retain the V16 logic;
  * local feature transformation is one shared layer reused by all proposals;
  * global proposal reconciliation remains permutation-equivariant self-attention;
  * presence is produced by one shared TimeDistributed head;
  * exact 720 truth matching, mass-preserving weights and V17.3 Poisson-binomial
    count NLL are unchanged;
  * runtime K remains sum(present >= 0.5), with no threshold tuning;
  * V17.5 injective runtime assignment is deliberately NOT included, so this is
    a controlled test of proposal generation rather than an ownership hybrid.

The fixed seed identities only break the otherwise exact symmetry. They contain
no trainable q-specific capacity: permuting seeds permutes proposal outputs while
all learned transforms remain shared.
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
from scripts.train_v100_spectral_string_slots import TIME_FRAMES
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v173_poibin_count_consistency as v173

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
BASE_ARM = "mass_permutation"
MODEL_KEY = "v176_shared_set"
QUERY_DIM = 96
EPS = 1e-6


class V176Error(RuntimeError):
    pass


def _input_by_name(model, name):
    for tensor in model.inputs:
        if tensor.name.split(":", 1)[0] == name:
            return tensor
    raise V176Error(f"model input not found: {name}")


def _fixed_seed_matrix_np() -> np.ndarray:
    """Six deterministic, normalized, non-trainable anonymous seed codes."""
    q = np.arange(1, EVENT_QUERIES + 1, dtype=np.float64)[:, None]
    d = np.arange(1, QUERY_DIM + 1, dtype=np.float64)[None, :]
    seed = np.sin(0.173 * q * d) + np.cos(0.113 * q * (d + 1.0))
    seed -= np.mean(seed, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(seed * seed, axis=1, keepdims=True))
    seed /= np.maximum(norm, 1e-12)
    return seed.astype(np.float32)


def _other_union(tf, assignment, q: int):
    events = assignment[:, :, :EVENT_QUERIES]
    left = events[:, :, :q]
    right = events[:, :, q + 1 :]
    others = tf.concat([left, right], axis=-1)
    return 1.0 - tf.reduce_prod(1.0 - tf.clip_by_value(others, 0.0, 1.0), axis=-1)


def _build_model(spec: dict):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    # Reuse the same V10.2 evidence backbone and auxiliary outputs that V16 uses.
    base, _, token_shape = v102._build_model()
    candidate_context = base.get_layer("candidate_context").output
    tf_tokens = base.get_layer("tf_tokens").output
    candidate_set = _input_by_name(base, "candidate_set")
    candidate_mask = _input_by_name(base, "candidate_mask")

    # Shared candidate encoder, matching V16's dimensionality.
    cand = keras.layers.TimeDistributed(
        keras.layers.LayerNormalization(), name="v176_candidate_norm"
    )(candidate_set)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v176_candidate_hidden1"
    )(cand)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v176_candidate_hidden2"
    )(cand)

    # Fixed anonymous seeds + one shared global context projection.
    seed_const = tf.constant(_fixed_seed_matrix_np(), dtype=tf.float32)
    fixed_seeds = keras.layers.Lambda(
        lambda x: tf.tile(seed_const[None, :, :], [tf.shape(x)[0], 1, 1]),
        name="v176_fixed_anonymous_seeds",
    )(candidate_context)
    seed_context = keras.layers.Dense(
        QUERY_DIM, activation="relu", name="v176_shared_seed_context"
    )(candidate_context)
    seed_context = keras.layers.Lambda(
        lambda x: tf.tile(x[:, None, :], [1, EVENT_QUERIES, 1]),
        name="v176_seed_context_broadcast",
    )(seed_context)
    proposals = keras.layers.Add(name="v176_seed_plus_context")([fixed_seeds, seed_context])
    proposals = keras.layers.LayerNormalization(name="v176_seed_norm")(proposals)

    candidate_attention_mask = keras.layers.Lambda(
        lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, EVENT_QUERIES, 1]),
        name="v176_candidate_attention_mask",
    )(candidate_mask)

    # All proposal decoder parameters are shared across the six proposal rows.
    ca = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v176_shared_candidate_cross_attention"
    )(proposals, cand, attention_mask=candidate_attention_mask)
    proposals = keras.layers.Add(name="v176_candidate_cross_residual")([proposals, ca])
    proposals = keras.layers.LayerNormalization(name="v176_candidate_cross_norm")(proposals)

    ta = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v176_shared_tf_cross_attention"
    )(proposals, tf_tokens)
    proposals = keras.layers.Add(name="v176_tf_cross_residual")([proposals, ta])
    proposals = keras.layers.LayerNormalization(name="v176_tf_cross_norm")(proposals)

    sa = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v176_shared_seed_self_attention"
    )(proposals, proposals)
    proposals = keras.layers.Add(name="v176_seed_self_residual")([proposals, sa])
    proposals = keras.layers.LayerNormalization(name="v176_seed_self_norm")(proposals)

    ff = keras.layers.Dense(192, activation="relu", name="v176_shared_seed_ff1")(proposals)
    ff = keras.layers.Dropout(0.08, name="v176_shared_seed_dropout")(ff)
    ff = keras.layers.Dense(QUERY_DIM, name="v176_shared_seed_ff2")(ff)
    proposals = keras.layers.Add(name="v176_seed_ff_residual")([proposals, ff])
    proposals = keras.layers.LayerNormalization(name="v176_seed_ff_norm")(proposals)

    # Shared evidence scoring. Competition is per TF/candidate item across all
    # anonymous proposals plus one background channel, as in V16.
    cand_keys = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v176_candidate_keys"
    )(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v176_tf_keys")(tf_tokens)
    score_queries = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v176_shared_score_query"
    )(proposals)

    tf_event_scores = keras.layers.Lambda(
        lambda z: tf.einsum("btd,bqd->btq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v176_tf_event_scores",
    )([tf_keys, score_queries])
    tf_background = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="v176_tf_background_score"
    )(keras.layers.Dense(1, name="v176_tf_background_dense")(tf_tokens))
    tf_score_stack = keras.layers.Concatenate(axis=-1, name="v176_tf_score_stack")([
        tf_event_scores,
        keras.layers.Lambda(lambda x: x[:, :, None], name="v176_tf_background_expand")(tf_background),
    ])
    tf_assignment = keras.layers.Softmax(axis=-1, name="v176_tf_competition")(tf_score_stack)

    cand_event_scores = keras.layers.Lambda(
        lambda z: tf.einsum("bcd,bqd->bcq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v176_candidate_event_scores",
    )([cand_keys, score_queries])
    cand_background = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="v176_candidate_background_score"
    )(
        keras.layers.TimeDistributed(
            keras.layers.Dense(1), name="v176_candidate_background_dense"
        )(cand)
    )
    cand_score_stack = keras.layers.Concatenate(axis=-1, name="v176_candidate_score_stack")([
        cand_event_scores,
        keras.layers.Lambda(
            lambda x: x[:, :, None], name="v176_candidate_background_expand"
        )(cand_background),
    ])
    cand_assignment = keras.layers.Softmax(axis=-1, name="v176_candidate_competition")(
        cand_score_stack
    )

    token_freq = int(token_shape[1])
    local_features = []
    time_outputs = []
    candidate_outputs = []

    # These layers are intentionally instantiated ONCE then reused for every q.
    shared_local_norm = keras.layers.LayerNormalization(name="v176_shared_local_norm")
    shared_local_hidden = keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1.5e-3),
        name="v176_shared_local_hidden",
    )

    for q in range(EVENT_QUERIES):
        own_tf = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v176_event_{q}_tf_full_weights"
        )(tf_assignment)
        others_tf = keras.layers.Lambda(
            lambda a, i=q: _other_union(tf, a, i),
            name=f"v176_event_{q}_tf_others_coverage",
        )(tf_assignment)
        novel_tf = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v176_event_{q}_tf_novel_weights",
        )([own_tf, others_tf])
        tf_full_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_tf_full_distribution",
        )(own_tf)
        tf_novel_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_tf_novel_distribution",
        )(novel_tf)
        tf_full_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v176_event_{q}_tf_full_pool",
        )([tf_tokens, tf_full_dist])
        tf_novel_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v176_event_{q}_tf_novel_pool",
        )([tf_tokens, tf_novel_dist])

        raw_cand = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v176_event_{q}_candidate_full_weights_raw"
        )(cand_assignment)
        own_cand = keras.layers.Multiply(name=f"v176_event_{q}_candidate_full_weights")(
            [raw_cand, candidate_mask]
        )
        others_cand_raw = keras.layers.Lambda(
            lambda a, i=q: _other_union(tf, a, i),
            name=f"v176_event_{q}_candidate_others_coverage_raw",
        )(cand_assignment)
        others_cand = keras.layers.Multiply(
            name=f"v176_event_{q}_candidate_others_coverage"
        )([others_cand_raw, candidate_mask])
        novel_cand = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v176_event_{q}_candidate_novel_weights",
        )([own_cand, others_cand])
        cand_full_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_candidate_full_distribution",
        )(own_cand)
        cand_novel_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"event_candidate_{q}",
        )(novel_cand)
        cand_full_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v176_event_{q}_candidate_full_pool",
        )([cand, cand_full_dist])
        cand_novel_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v176_event_{q}_candidate_novel_pool",
        )([cand, cand_novel_dist])

        tf_novelty = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_tf_novelty_fraction",
        )([novel_tf, own_tf])
        tf_overlap = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True)
            / (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_tf_overlap_fraction",
        )([own_tf, others_tf])
        cand_novelty = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_candidate_novelty_fraction",
        )([novel_cand, own_cand])
        cand_overlap = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True)
            / (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_candidate_overlap_fraction",
        )([own_cand, others_cand])
        tf_other_coverage = keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
            name=f"v176_event_{q}_tf_other_coverage_fraction",
        )(others_tf)
        cand_other_coverage = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v176_event_{q}_candidate_other_coverage_fraction",
        )([others_cand, candidate_mask])

        tf_grid = keras.layers.Reshape(
            (TIME_FRAMES, token_freq), name=f"v176_event_{q}_tf_novel_grid"
        )(novel_tf)
        time_mass = keras.layers.Lambda(
            lambda a: tf.reduce_sum(a, axis=2), name=f"v176_event_{q}_time_mass"
        )(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS),
            name=f"event_time_{q}",
        )(time_mass)

        proposal_q = keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"v176_event_{q}_proposal_feature"
        )(proposals)
        local = keras.layers.Concatenate(name=f"v176_event_{q}_local_feature")([
            candidate_context,
            proposal_q,
            tf_full_latent,
            tf_novel_latent,
            cand_full_latent,
            cand_novel_latent,
            tf_novelty,
            cand_novelty,
            tf_overlap,
            cand_overlap,
            tf_other_coverage,
            cand_other_coverage,
        ])
        local = shared_local_norm(local)
        local = shared_local_hidden(local)
        local_features.append(local)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_novel_dist)

    proposal_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="v176_proposal_stack"
    )(local_features)
    attended = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.05, name="v176_global_reconciliation_attention"
    )(proposal_stack, proposal_stack)
    reconciled = keras.layers.Add(name="v176_global_reconciliation_residual")(
        [proposal_stack, attended]
    )
    reconciled = keras.layers.LayerNormalization(name="v176_global_reconciliation_norm")(reconciled)
    rff = keras.layers.Dense(128, activation="relu", name="v176_global_ff1")(reconciled)
    rff = keras.layers.Dropout(0.08, name="v176_global_dropout")(rff)
    rff = keras.layers.Dense(128, activation="relu", name="v176_global_ff2")(rff)
    reconciled = keras.layers.Add(name="v176_global_ff_residual")([reconciled, rff])
    reconciled = keras.layers.LayerNormalization(name="v176_global_ff_norm")(reconciled)

    # One objectness classifier shared over proposal rows.
    presence_hidden = keras.layers.TimeDistributed(
        keras.layers.Dense(64, activation="relu"), name="v176_shared_presence_hidden"
    )(reconciled)
    presence_stack = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation="sigmoid"), name="v176_shared_presence"
    )(presence_hidden)

    presence_outputs = [
        keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"event_present_{q}"
        )(presence_stack)
        for q in range(EVENT_QUERIES)
    ]
    count_norm = keras.layers.Lambda(
        lambda p: tf.reduce_sum(p, axis=1) / float(EVENT_QUERIES),
        name="event_count_norm",
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
            lambda x: tf.ones_like(x), name=f"v176_event_{q}_valid_placeholder"
        )(present)
        packed = keras.layers.Concatenate(name=f"v176_event_{q}_set_vector")([
            present,
            valid,
            time_outputs[q],
            candidate_outputs[q],
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

    model = keras.Model(base.inputs, outputs, name="v176_shared_set_decoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=lw,
    )
    return model, lw, token_shape


def _gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if not np.any(x):
        return 0.0
    n = len(x)
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x)) - (n + 1.0) / n)


def _effective_slots(values) -> float:
    x = np.asarray(values, dtype=np.float64)
    den = float(np.sum(x * x))
    return float(np.sum(x) ** 2 / den) if den else 0.0


def _duplicate_rate(candidate: np.ndarray, presence: np.ndarray, mask: np.ndarray) -> Optional[float]:
    rows = []
    for c, p, keep in zip(candidate, presence, mask):
        active = np.flatnonzero(p >= PRESENCE_THRESHOLD)
        if len(active) < 2 or not keep:
            continue
        ids = np.argmax(c[active], axis=1)
        rows.append(len(np.unique(ids)) < len(ids))
    return float(np.mean(rows)) if rows else None


def _postprocess(args, report, ctx):
    # Reuse canonical V17.3 reporting/matching diagnostics first, then correct
    # protocol metadata that necessarily differs because the proposal graph changed.
    report = v173._postprocess(args, report, ctx)
    v171._rename_report(report, v173.MODEL_KEY, MODEL_KEY)

    inherited = report.pop("v173")
    # V17.2's gradient diagnostic rebuilds the V16 graph, so it is not a valid
    # V17.6 final-model diagnostic. Remove it rather than silently mislabel it.
    inherited.pop("final_model_presence_gradient_mass", None)
    inherited.pop("final_model_presence_gradient_mass_error", None)

    report["protocol"].update({
        "v176_shared_set_decoder": True,
        "v176_base_version": "V17.3",
        "v176_only_architecture_change": "q-specific proposal/local/presence parameterization -> shared set-equivariant decoder",
        "v16_proposal_graph_unchanged": False,
        "v173_poisson_binomial_count_objective_unchanged": True,
        "v173_count_nll_weight": v173.COUNT_NLL_WEIGHT,
        "mass_preserving_exchangeable_weights_unchanged": True,
        "exact_720_truth_matching_unchanged": True,
        "runtime_count_decode_unchanged_from_v173": True,
        "runtime_presence_threshold": PRESENCE_THRESHOLD,
        "runtime_presence_threshold_tuned": False,
        "categorical_cardinality_head_exists": False,
        "fixed_anonymous_seed_count": EVENT_QUERIES,
        "anonymous_seeds_trainable": False,
        "query_specific_trainable_event_query_layers": False,
        "query_specific_trainable_local_hidden_layers": False,
        "query_specific_trainable_presence_heads": False,
        "shared_candidate_cross_attention": True,
        "shared_tf_cross_attention": True,
        "shared_local_feature_transform": True,
        "shared_presence_classifier": True,
        "v175_injective_ownership_used": False,
        "historical_validation_or_locked12_indexed_or_evaluated": False,
    })

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}

    old_pred = "pred173_poibin"
    if old_pred not in data:
        raise V176Error(f"missing inherited prediction key {old_pred}")
    data["pred176_shared_set"] = data.pop(old_pred)

    presence = np.asarray(data["presence"], dtype=np.float64)
    candidate = np.asarray(data["event_candidate"], dtype=np.float64)
    active_rates = np.mean(presence >= PRESENCE_THRESHOLD, axis=0)
    occup = np.asarray(
        [inherited["outer_match_occupancy"][str(q)]["matched_object_rate"] for q in range(EVENT_QUERIES)],
        dtype=np.float64,
    )
    corr = float(np.corrcoef(active_rates, occup)[0, 1]) if np.std(active_rates) and np.std(occup) else 0.0

    true_k = np.asarray(ctx["k"], dtype=np.int32)[np.asarray(ctx["outer_idx"], dtype=np.int64)]
    hard_k = np.sum(presence >= PRESENCE_THRESHOLD, axis=1).astype(np.int32)
    duplicate_by_k = {}
    for value in range(2, EVENT_QUERIES + 1):
        exact = (true_k == value) & (hard_k == value)
        duplicate_by_k[str(value)] = {
            "rows": int(np.sum(exact)),
            "raw_candidate_duplicate_argmax_exact_count": _duplicate_rate(
                candidate, presence, exact
            ),
        }

    architecture = {
        "fixed_seed_matrix": _fixed_seed_matrix_np().tolist(),
        "outer_active_rate_by_query": active_rates.tolist(),
        "outer_activity_gini": _gini(active_rates),
        "outer_effective_active_slots": _effective_slots(active_rates),
        "outer_matched_object_rate_by_query": occup.tolist(),
        "outer_active_occupancy_correlation": corr,
        "soft_presence_mass": float(np.mean(np.sum(presence, axis=1))),
        "raw_candidate_duplicates_exact_count_by_true_k": duplicate_by_k,
    }
    report["v176"] = {
        **inherited,
        "model_key": MODEL_KEY,
        "architecture": architecture,
    }

    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v176-shared-set-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)

    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def train_fold(args):
    if args.seed != DEFAULT_SEED:
        raise V176Error(f"V17.6 requires seed {DEFAULT_SEED}, got {args.seed}")
    if args.arm != BASE_ARM:
        raise V176Error(f"V17.6 only supports base arm {BASE_ARM!r}")

    ctx = v172._fold_context(args)
    stage_specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V176Error(f"unexpected model build call {i + 1}")
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
        raise V176Error(f"expected exactly 2 model builds, got {calls['count']}")

    report = _postprocess(args, report, ctx)
    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    card = report["strata"]["aggregate"][MODEL_KEY]["cardinality"]
    arch = report["v176"]["architecture"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v176_f1": g["f1"],
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
