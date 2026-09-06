"""V22.1 jointly-trained exchangeable dense proposal decoder.

V21 showed that V19 conv3 contains useful local birth information while the
1x1 center head does not separate hard alternatives. V22 then proved that
replacing only that scorer is destructive because the old decoder is tightly
co-adapted to the old anchors. V22.1 therefore removes the discrete proposal
interface entirely.

Scientific treatment relative to V19:
- keep the 23x64 conv evidence branch;
- remove birth-center softmax supervision, local-max NMS and top-k gather;
- form six proposals with shared Slot Attention directly over all 1472 conv3
  cells, so event-set gradients reach proposal formation;
- slots use one shared learned initialization distribution and iid noise during
  fitting (fixed deterministic iid draw at evaluation), with shared attention,
  GRU and MLP parameters; there are no q-specific proposal parameters.

Everything downstream stays aligned with V19/V17.3: shared candidate/TF
cross-attention, proposal reconciliation, exact 720 permutation set matching,
mass-preserving presence weights, exact Poisson-binomial count NLL=0.35 and
runtime K=sum(p>=0.5). Threshold 0.5 is not tuned. Locked12 is untouched.
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
MODEL_KEY = "v221_joint_slot_attention"
PRED_KEY = "pred221_joint_slot_attention"
QUERY_DIM = 96
CENTER_CELLS = v100.TIME_FRAMES * v100.SPECTRAL_BANDS
SLOT_ITERS = 3
SLOT_MLP_DIM = 192
EPS = 1e-6


class V221Error(RuntimeError):
    pass


def _input_by_name(model, name):
    return v180._input_by_name(model, name)


def _other_union(tf, assignment, q):
    return v180._other_union(tf, assignment, q)


def _gini(x):
    return v176._gini(np.asarray(x, dtype=np.float64))


def _build_model(spec):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    class ExchangeableSlotAttention(keras.layers.Layer):
        """Shared Slot Attention with no slot-specific trainable parameters."""

        def __init__(self, num_slots, dim, iters=3, mlp_dim=192, **kwargs):
            super().__init__(**kwargs)
            self.num_slots = int(num_slots)
            self.dim = int(dim)
            self.iters = int(iters)
            self.mlp_dim = int(mlp_dim)
            self.norm_inputs = keras.layers.LayerNormalization(name="input_norm")
            self.norm_slots = keras.layers.LayerNormalization(name="slot_norm")
            self.norm_mlp = keras.layers.LayerNormalization(name="mlp_norm")
            self.to_k = keras.layers.Dense(dim, use_bias=False, name="key")
            self.to_v = keras.layers.Dense(dim, use_bias=False, name="value")
            self.to_q = keras.layers.Dense(dim, use_bias=False, name="query")
            self.gru = keras.layers.GRUCell(dim, name="shared_gru")
            self.mlp1 = keras.layers.Dense(mlp_dim, activation="relu", name="shared_mlp1")
            self.mlp2 = keras.layers.Dense(dim, name="shared_mlp2")

        def build(self, input_shape):
            self.slot_mu = self.add_weight(
                name="shared_slot_mu", shape=(1, 1, self.dim), initializer="zeros", trainable=True
            )
            self.slot_logsigma = self.add_weight(
                name="shared_slot_logsigma", shape=(1, 1, self.dim),
                initializer=keras.initializers.Constant(-1.5), trainable=True
            )
            rng = np.random.default_rng(DEFAULT_SEED + 221)
            fixed = rng.standard_normal((1, self.num_slots, self.dim)).astype(np.float32)
            self.fixed_eval_noise = self.add_weight(
                name="fixed_eval_iid_noise", shape=fixed.shape,
                initializer=keras.initializers.Constant(fixed), trainable=False
            )
            super().build(input_shape)

        def call(self, inputs, training=None):
            x = self.norm_inputs(inputs)
            k = self.to_k(x)
            v = self.to_v(x)
            batch = tf.shape(inputs)[0]
            sigma = tf.nn.softplus(self.slot_logsigma) + 1e-4
            if training is None:
                training = keras.backend.learning_phase()

            def train_noise():
                return tf.random.normal((batch, self.num_slots, self.dim), dtype=inputs.dtype)

            def eval_noise():
                return tf.tile(tf.cast(self.fixed_eval_noise, inputs.dtype), [batch, 1, 1])

            if isinstance(training, bool):
                noise = train_noise() if training else eval_noise()
            else:
                noise = tf.cond(tf.cast(training, tf.bool), train_noise, eval_noise)
            slots = tf.cast(self.slot_mu, inputs.dtype) + tf.cast(sigma, inputs.dtype) * noise
            scale = tf.cast(self.dim, inputs.dtype) ** -0.5
            last_attn = None
            for _ in range(self.iters):
                prev = slots
                q = self.to_q(self.norm_slots(slots))
                logits = tf.einsum("bnd,bsd->bns", k, q) * scale
                # Competition is across slots for every evidence cell.
                attn = tf.nn.softmax(logits, axis=-1) + tf.cast(EPS, logits.dtype)
                weights = attn / (tf.reduce_sum(attn, axis=1, keepdims=True) + tf.cast(EPS, logits.dtype))
                updates = tf.einsum("bns,bnd->bsd", weights, v)
                flat_u = tf.reshape(updates, (-1, self.dim))
                flat_s = tf.reshape(prev, (-1, self.dim))
                flat_out, _ = self.gru(flat_u, [flat_s])
                slots = tf.reshape(flat_out, (batch, self.num_slots, self.dim))
                slots = slots + self.mlp2(self.mlp1(self.norm_mlp(slots)))
                last_attn = attn
            return slots, last_attn

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "num_slots": self.num_slots, "dim": self.dim,
                "iters": self.iters, "mlp_dim": self.mlp_dim,
            })
            return cfg

    base, _, token_shape = v102._build_model()
    candidate_context = base.get_layer("candidate_context").output
    tf_tokens = base.get_layer("tf_tokens").output
    candidate_set = _input_by_name(base, "candidate_set")
    candidate_mask = _input_by_name(base, "candidate_mask")
    spectral = _input_by_name(base, "spectral_map")

    cand = keras.layers.TimeDistributed(
        keras.layers.LayerNormalization(), name="v221_candidate_norm"
    )(candidate_set)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v221_candidate_hidden1"
    )(cand)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v221_candidate_hidden2"
    )(cand)

    # Keep V19's fine 23x64 convolutional evidence representation.
    tc = np.linspace(-1.0, 1.0, v100.TIME_FRAMES, dtype=np.float32)[:, None]
    fc = np.linspace(-1.0, 1.0, v100.SPECTRAL_BANDS, dtype=np.float32)[None, :]
    coord = np.stack(
        [
            np.broadcast_to(tc, (v100.TIME_FRAMES, v100.SPECTRAL_BANDS)),
            np.broadcast_to(fc, (v100.TIME_FRAMES, v100.SPECTRAL_BANDS)),
        ],
        axis=-1,
    ).astype(np.float32)
    coord_const = tf.constant(coord, dtype=tf.float32)
    coords = keras.layers.Lambda(
        lambda s: tf.tile(coord_const[None, :, :, :], [tf.shape(s)[0], 1, 1, 1]),
        name="v221_dense_coordinates",
    )(spectral)
    dense = keras.layers.LayerNormalization(axis=-1, name="v221_dense_channel_norm")(spectral)
    dense = keras.layers.Concatenate(axis=-1, name="v221_dense_plus_coordinates")([dense, coords])
    dense = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="v221_dense_conv1")(dense)
    dense = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="v221_dense_conv2")(dense)
    dense_features = keras.layers.Conv2D(
        QUERY_DIM, (3, 3), padding="same", activation="relu", name="v221_dense_conv3"
    )(dense)
    dense_flat = keras.layers.Reshape(
        (CENTER_CELLS, QUERY_DIM), name="v221_dense_feature_flat"
    )(dense_features)

    slot_layer = ExchangeableSlotAttention(
        EVENT_QUERIES, QUERY_DIM, iters=SLOT_ITERS, mlp_dim=SLOT_MLP_DIM,
        name="v221_slot_attention",
    )
    proposal_slots, slot_attention = slot_layer(dense_flat)
    # Keep the attention tensor attached to the graph for diagnostics while all
    # downstream computation uses the slots themselves.
    _ = keras.layers.Lambda(lambda a: tf.reduce_mean(a, axis=1), name="v221_slot_attention_mean_mass")(slot_attention)

    gc = keras.layers.Dense(QUERY_DIM, activation="relu", name="v221_shared_global_context")(candidate_context)
    gc = keras.layers.Lambda(
        lambda x: tf.tile(x[:, None, :], [1, EVENT_QUERIES, 1]),
        name="v221_global_context_broadcast",
    )(gc)
    proposals = keras.layers.LayerNormalization(name="v221_slot_plus_context_norm")(
        keras.layers.Add(name="v221_slot_plus_context")([proposal_slots, gc])
    )

    cam = keras.layers.Lambda(
        lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, EVENT_QUERIES, 1]),
        name="v221_candidate_attention_mask",
    )(candidate_mask)
    ca = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v221_shared_candidate_cross_attention"
    )(proposals, cand, attention_mask=cam)
    proposals = keras.layers.LayerNormalization(name="v221_candidate_cross_norm")(
        keras.layers.Add(name="v221_candidate_cross_residual")([proposals, ca])
    )
    ta = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v221_shared_tf_cross_attention"
    )(proposals, tf_tokens)
    proposals = keras.layers.LayerNormalization(name="v221_tf_cross_norm")(
        keras.layers.Add(name="v221_tf_cross_residual")([proposals, ta])
    )
    sa = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v221_shared_proposal_self_attention"
    )(proposals, proposals)
    proposals = keras.layers.LayerNormalization(name="v221_proposal_self_norm")(
        keras.layers.Add(name="v221_proposal_self_residual")([proposals, sa])
    )
    ff = keras.layers.Dense(192, activation="relu", name="v221_shared_proposal_ff1")(proposals)
    ff = keras.layers.Dropout(0.08, name="v221_shared_proposal_dropout")(ff)
    ff = keras.layers.Dense(QUERY_DIM, name="v221_shared_proposal_ff2")(ff)
    proposals = keras.layers.LayerNormalization(name="v221_proposal_ff_norm")(
        keras.layers.Add(name="v221_proposal_ff_residual")([proposals, ff])
    )

    cand_keys = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v221_candidate_keys"
    )(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v221_tf_keys")(tf_tokens)
    score_q = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v221_shared_score_query"
    )(proposals)
    tf_es = keras.layers.Lambda(
        lambda z: tf.einsum("btd,bqd->btq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v221_tf_event_scores",
    )([tf_keys, score_q])
    tf_bg = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="v221_tf_background_score")(
        keras.layers.Dense(1, name="v221_tf_background_dense")(tf_tokens)
    )
    tf_scores = keras.layers.Concatenate(axis=-1, name="v221_tf_score_stack")([
        tf_es,
        keras.layers.Lambda(lambda x: x[:, :, None], name="v221_tf_background_expand")(tf_bg),
    ])
    tf_assign = keras.layers.Softmax(axis=-1, name="v221_tf_competition")(tf_scores)

    ce = keras.layers.Lambda(
        lambda z: tf.einsum("bcd,bqd->bcq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v221_candidate_event_scores",
    )([cand_keys, score_q])
    cbg = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="v221_candidate_background_score")(
        keras.layers.TimeDistributed(keras.layers.Dense(1), name="v221_candidate_background_dense")(cand)
    )
    cscore = keras.layers.Concatenate(axis=-1, name="v221_candidate_score_stack")([
        ce,
        keras.layers.Lambda(lambda x: x[:, :, None], name="v221_candidate_background_expand")(cbg),
    ])
    cand_assign = keras.layers.Softmax(axis=-1, name="v221_candidate_competition")(cscore)

    token_freq = int(token_shape[1])
    local_features, time_outputs, candidate_outputs = [], [], []
    shared_norm = keras.layers.LayerNormalization(name="v221_shared_local_norm")
    shared_hidden = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.l2(1.5e-3),
        name="v221_shared_local_hidden",
    )
    for q in range(EVENT_QUERIES):
        own_tf = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v221_event_{q}_tf_weights"
        )(tf_assign)
        tf_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v221_event_{q}_tf_distribution",
        )(own_tf)
        tf_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v221_event_{q}_tf_pool",
        )([tf_tokens, tf_dist])
        raw_c = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v221_event_{q}_candidate_weights_raw"
        )(cand_assign)
        own_c = keras.layers.Multiply(name=f"v221_event_{q}_candidate_weights")([raw_c, candidate_mask])
        cand_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"event_candidate_{q}",
        )(own_c)
        cand_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v221_event_{q}_candidate_pool",
        )([cand, cand_dist])
        others = keras.layers.Lambda(
            lambda a, i=q: _other_union(tf, a, i),
            name=f"v221_event_{q}_other_candidate_coverage_raw",
        )(cand_assign)
        others = keras.layers.Multiply(
            name=f"v221_event_{q}_other_candidate_coverage"
        )([others, candidate_mask])
        overlap = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True)
            / (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v221_event_{q}_candidate_overlap",
        )([own_c, others])
        cmass = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v221_event_{q}_candidate_mass",
        )([own_c, candidate_mask])
        tmass = keras.layers.Lambda(
            lambda w: tf.reduce_mean(w, axis=1, keepdims=True),
            name=f"v221_event_{q}_tf_mass",
        )(own_tf)
        tf_grid = keras.layers.Reshape(
            (v100.TIME_FRAMES, token_freq), name=f"v221_event_{q}_tf_grid"
        )(own_tf)
        time_mass = keras.layers.Lambda(
            lambda a: tf.reduce_sum(a, axis=2), name=f"v221_event_{q}_time_mass"
        )(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS),
            name=f"event_time_{q}",
        )(time_mass)
        pq = keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"v221_event_{q}_proposal"
        )(proposals)
        local = shared_hidden(
            shared_norm(
                keras.layers.Concatenate(name=f"v221_event_{q}_local_feature")([
                    candidate_context, pq, tf_latent, cand_latent, tmass, cmass, overlap,
                ])
            )
        )
        local_features.append(local)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_dist)

    stack = keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="v221_proposal_stack")(
        local_features
    )
    ra = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.05, name="v221_global_reconciliation_attention"
    )(stack, stack)
    rec = keras.layers.LayerNormalization(name="v221_global_reconciliation_norm")(
        keras.layers.Add(name="v221_global_reconciliation_residual")([stack, ra])
    )
    rff = keras.layers.Dense(128, activation="relu", name="v221_global_ff1")(rec)
    rff = keras.layers.Dropout(0.08, name="v221_global_dropout")(rff)
    rff = keras.layers.Dense(128, activation="relu", name="v221_global_ff2")(rff)
    rec = keras.layers.LayerNormalization(name="v221_global_ff_norm")(
        keras.layers.Add(name="v221_global_ff_residual")([rec, rff])
    )
    ph = keras.layers.TimeDistributed(
        keras.layers.Dense(64, activation="relu"), name="v221_shared_presence_hidden"
    )(rec)
    pstack = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation="sigmoid"), name="v221_shared_presence"
    )(ph)
    presence = [
        keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"event_present_{q}")(pstack)
        for q in range(EVENT_QUERIES)
    ]
    count_norm = keras.layers.Lambda(
        lambda p: tf.reduce_sum(p, axis=1) / float(EVENT_QUERIES),
        name="event_count_norm",
    )(pstack)

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    set_slots = []
    for q in range(EVENT_QUERIES):
        valid = keras.layers.Lambda(
            lambda x: tf.ones_like(x), name=f"v221_event_{q}_valid_placeholder"
        )(presence[q])
        packed = keras.layers.Concatenate(name=f"v221_event_{q}_set_vector")([
            presence[q], valid, time_outputs[q], candidate_outputs[q]
        ])
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = presence[q]
        outputs[f"event_time_{q}"] = time_outputs[q]
        outputs[f"event_candidate_{q}"] = candidate_outputs[q]
    outputs["event_set"] = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="event_set"
    )(set_slots)
    outputs["event_count_norm"] = count_norm

    loss = {f"string_{s}": "binary_crossentropy" for s in range(SLOT_COUNT)}
    loss.update({f"pitch_{s}": "mse" for s in range(SLOT_COUNT)})
    loss.update({f"time_{s}": keras.losses.KLDivergence() for s in range(SLOT_COUNT)})
    loss["event_set"] = v173._set_loss(spec)
    loss["event_count_norm"] = "mse"
    lw = {f"string_{s}": 0.18 for s in range(SLOT_COUNT)}
    lw.update({f"pitch_{s}": 0.04 for s in range(SLOT_COUNT)})
    lw.update({f"time_{s}": 0.10 for s in range(SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.0

    model = keras.Model(base.inputs, outputs, name="v221_joint_slot_attention_decoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=lw,
    )
    return model, lw, token_shape


def _slot_diagnostics(model, inputs, max_rows=2048):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    n = min(max_rows, len(next(iter(inputs.values()))))
    sub_inputs = {k: np.asarray(v)[:n] for k, v in inputs.items()}
    layer = model.get_layer("v221_slot_attention")
    probe = keras.Model(model.inputs, layer.output)
    slots, attn = probe.predict(sub_inputs, batch_size=64, verbose=0)
    slots = np.asarray(slots, dtype=np.float64)
    attn = np.asarray(attn, dtype=np.float64)
    norms = np.linalg.norm(slots, axis=2, keepdims=True) + 1e-12
    unit = slots / norms
    cos = np.einsum("bsd,btd->bst", unit, unit)
    tri = np.triu_indices(EVENT_QUERIES, 1)
    pair_cos = cos[:, tri[0], tri[1]]
    hard = np.argmax(attn, axis=2)
    share = np.stack([(hard == q).mean(axis=1) for q in range(EVENT_QUERIES)], axis=1)
    p = np.clip(attn, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=2) / math.log(float(EVENT_QUERIES))
    return {
        "rows_sampled": int(n),
        "mean_pairwise_slot_cosine": float(np.mean(pair_cos)),
        "p90_pairwise_slot_cosine": float(np.percentile(pair_cos, 90)),
        "mean_cell_assignment_entropy": float(np.mean(ent)),
        "mean_hard_cell_share_by_slot": np.mean(share, axis=0).tolist(),
        "hard_cell_share_gini": _gini(np.mean(share, axis=0)),
    }


def _postprocess(args, report, ctx, slot_diag):
    report = v173._postprocess(args, report, ctx)
    v171._rename_report(report, v173.MODEL_KEY, MODEL_KEY)
    inherited = report.pop("v173")
    inherited.pop("final_model_presence_gradient_mass", None)
    inherited.pop("final_model_presence_gradient_mass_error", None)
    report["protocol"].update({
        "v221_joint_slot_attention": True,
        "v221_only_scientific_treatment_from_v190": (
            "replace dense center head + 3x3 NMS + top6 gather with jointly-trained shared Slot Attention over all 1472 conv3 cells; remove center-map auxiliary loss"
        ),
        "dense_conv_grid": [int(v100.TIME_FRAMES), int(v100.SPECTRAL_BANDS)],
        "dense_conv_parameterization_retained_from_v190": True,
        "discrete_topk_or_nms_in_proposal_path": False,
        "birth_center_auxiliary_head_exists": False,
        "proposal_formation_receives_event_set_gradient": True,
        "slot_attention_iterations": SLOT_ITERS,
        "slot_attention_competition_axis": "slots_per_dense_cell",
        "slot_specific_trainable_parameters": False,
        "slot_initialization": "iid shared learned mean/scale; stochastic fit, fixed iid evaluation draw",
        "fixed_anonymous_seed_count": 0,
        "learned_anonymous_seed_count": 0,
        "raw_candidate_is_object_identity": False,
        "v173_poisson_binomial_count_objective_unchanged": True,
        "v173_count_nll_weight": v173.COUNT_NLL_WEIGHT,
        "mass_preserving_exchangeable_weights_unchanged": True,
        "exact_720_truth_matching_unchanged": True,
        "runtime_count_decode_unchanged_from_v173": True,
        "runtime_presence_threshold": PRESENCE_THRESHOLD,
        "runtime_presence_threshold_tuned": False,
        "categorical_cardinality_head_exists": False,
        "historical_validation_or_locked12_indexed_or_evaluated": False,
    })

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if "pred173_poibin" not in data:
        raise V221Error("missing inherited pred173_poibin")
    data[PRED_KEY] = data.pop("pred173_poibin")
    presence = np.asarray(data["presence"], dtype=np.float64)
    candidate = np.asarray(data["event_candidate"], dtype=np.float64)
    active = np.mean(presence >= PRESENCE_THRESHOLD, axis=0)
    occup = np.asarray(
        [inherited["outer_match_occupancy"][str(q)]["matched_object_rate"] for q in range(EVENT_QUERIES)],
        dtype=np.float64,
    )
    corr = float(np.corrcoef(active, occup)[0, 1]) if np.std(active) and np.std(occup) else 0.0
    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    true_k = np.asarray(ctx["k"], dtype=np.int32)[outer]
    hard_k = np.sum(presence >= PRESENCE_THRESHOLD, axis=1).astype(np.int32)
    dup = {}
    for value in range(2, EVENT_QUERIES + 1):
        exact = (true_k == value) & (hard_k == value)
        dup[str(value)] = {
            "rows": int(np.sum(exact)),
            "raw_candidate_duplicate_argmax_exact_count": v176._duplicate_rate(
                candidate, presence, exact
            ),
        }
    arch = {
        "outer_active_rate_by_proposal_position": active.tolist(),
        "outer_activity_gini": v176._gini(active),
        "outer_effective_active_slots": v176._effective_slots(active),
        "outer_matched_object_rate_by_proposal_position": occup.tolist(),
        "outer_active_occupancy_correlation": corr,
        "soft_presence_mass": float(np.mean(np.sum(presence, axis=1))),
        "raw_candidate_duplicates_exact_count_by_true_k": dup,
        "slot_attention_diagnostics": slot_diag,
    }
    report["v221"] = {**inherited, "model_key": MODEL_KEY, "architecture": arch}
    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v221-joint-slot-attention-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def train_fold(args):
    if args.seed != DEFAULT_SEED:
        raise V221Error(f"V22.1 requires seed {DEFAULT_SEED}")
    if args.arm != BASE_ARM:
        raise V221Error(f"V22.1 only supports {BASE_ARM}")

    ctx = v172._fold_context(args)
    specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}
    built = []

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V221Error("unexpected model build")
        calls["count"] += 1
        t = _build_model(specs[i])
        built.append(t[0])
        return t

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = v171._targets
        v130._sample_weights = v171._sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights

    if calls["count"] != 2:
        raise V221Error(f"expected two model builds, got {calls['count']}")
    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    outer_inputs = v102._inputs(ctx["cache"], outer)
    slot_diag = _slot_diagnostics(built[-1], outer_inputs)
    report = _postprocess(args, report, ctx, slot_diag)
    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    card = report["strata"]["aggregate"][MODEL_KEY]["cardinality"]
    arch = report["v221"]["architecture"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v221_f1": g["f1"],
        "pred_ref": g["prediction_reference_ratio"],
        "poly_exact": card.get("poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))),
        "activity_gini": arch["outer_activity_gini"],
        "effective_active_slots": arch["outer_effective_active_slots"],
        "slot_pair_cosine": slot_diag["mean_pairwise_slot_cosine"],
        "slot_assignment_entropy": slot_diag["mean_cell_assignment_entropy"],
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
