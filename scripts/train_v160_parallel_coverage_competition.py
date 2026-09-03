"""V16.0 parallel birth proposals with global coverage competition.

V13 recovered difficult/high-K births with parallel event queries but duplicated
shared evidence. V14/V15 controlled duplication with a sequential STOP chain,
which improved calibration but made late births depend on all previous decisions
and collapsed K6 to zero.

V16 removes the sequential chain entirely:
  * six event proposals are computed in parallel from full TF/candidate evidence;
  * TF and candidate evidence remain competitively assigned across proposals;
  * every proposal receives leave-one-out coverage from the other five proposals;
  * full and exclusive/novel evidence are pooled separately;
  * proposal features are reconciled jointly with self-attention before presence;
  * runtime K is the number of final proposal presences >= 0.5.

There is no categorical K head, no CONTINUE/STOP, no tuned threshold and no
runtime annotation input. The audited V13 outer-clean harness is reused so only
the architecture changes.
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
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
QUERY_DIM = 96
EPS = 1e-6


class V160Error(RuntimeError):
    pass


def _input_by_name(model, name):
    for tensor in model.inputs:
        if tensor.name.split(":", 1)[0] == name:
            return tensor
    raise V160Error(f"model input not found: {name}")


def _other_union(tf, assignment, q: int):
    events = assignment[:, :, :EVENT_QUERIES]
    left = events[:, :, :q]
    right = events[:, :, q + 1 :]
    others = tf.concat([left, right], axis=-1)
    return 1.0 - tf.reduce_prod(1.0 - tf.clip_by_value(others, 0.0, 1.0), axis=-1)


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

    cand = keras.layers.TimeDistributed(
        keras.layers.LayerNormalization(), name="v160_candidate_norm"
    )(candidate_set)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v160_candidate_hidden1"
    )(cand)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v160_candidate_hidden2"
    )(cand)
    cand_keys = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v160_candidate_keys"
    )(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v160_tf_keys")(tf_tokens)

    queries = [
        keras.layers.Dense(QUERY_DIM, activation="relu", name=f"v160_event_{q}_query")(candidate_context)
        for q in range(EVENT_QUERIES)
    ]

    tf_scores = [
        keras.layers.Lambda(
            lambda z: tf.einsum("btd,bd->bt", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"v160_event_{q}_tf_score",
        )([tf_keys, queries[q]])
        for q in range(EVENT_QUERIES)
    ]
    tf_background = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="v160_tf_background_score"
    )(keras.layers.Dense(1, name="v160_tf_background_dense")(tf_tokens))
    tf_score_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=-1), name="v160_tf_score_stack"
    )(tf_scores + [tf_background])
    tf_assignment = keras.layers.Softmax(axis=-1, name="v160_tf_competition")(tf_score_stack)

    cand_scores = [
        keras.layers.Lambda(
            lambda z: tf.einsum("bcd,bd->bc", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"v160_event_{q}_candidate_score",
        )([cand_keys, queries[q]])
        for q in range(EVENT_QUERIES)
    ]
    cand_background = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="v160_candidate_background_score"
    )(
        keras.layers.TimeDistributed(
            keras.layers.Dense(1), name="v160_candidate_background_dense"
        )(cand)
    )
    cand_score_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=-1), name="v160_candidate_score_stack"
    )(cand_scores + [cand_background])
    cand_assignment = keras.layers.Softmax(axis=-1, name="v160_candidate_competition")(
        cand_score_stack
    )

    token_freq = int(token_shape[1])
    local_features = []
    time_outputs = []
    candidate_outputs = []

    for q in range(EVENT_QUERIES):
        own_tf = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v160_event_{q}_tf_full_weights"
        )(tf_assignment)
        others_tf = keras.layers.Lambda(
            lambda a, i=q: _other_union(tf, a, i),
            name=f"v160_event_{q}_tf_others_coverage",
        )(tf_assignment)
        novel_tf = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v160_event_{q}_tf_novel_weights",
        )([own_tf, others_tf])

        tf_full_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_tf_full_distribution",
        )(own_tf)
        tf_novel_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_tf_novel_distribution",
        )(novel_tf)
        tf_full_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v160_event_{q}_tf_full_pool",
        )([tf_tokens, tf_full_dist])
        tf_novel_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v160_event_{q}_tf_novel_pool",
        )([tf_tokens, tf_novel_dist])

        raw_cand = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"v160_event_{q}_candidate_full_weights_raw"
        )(cand_assignment)
        own_cand = keras.layers.Multiply(name=f"v160_event_{q}_candidate_full_weights")(
            [raw_cand, candidate_mask]
        )
        others_cand_raw = keras.layers.Lambda(
            lambda a, i=q: _other_union(tf, a, i),
            name=f"v160_event_{q}_candidate_others_coverage_raw",
        )(cand_assignment)
        others_cand = keras.layers.Multiply(
            name=f"v160_event_{q}_candidate_others_coverage"
        )([others_cand_raw, candidate_mask])
        novel_cand = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v160_event_{q}_candidate_novel_weights",
        )([own_cand, others_cand])

        cand_full_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_candidate_full_distribution",
        )(own_cand)
        cand_novel_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"event_candidate_{q}",
        )(novel_cand)
        cand_full_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v160_event_{q}_candidate_full_pool",
        )([cand, cand_full_dist])
        cand_novel_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v160_event_{q}_candidate_novel_pool",
        )([cand, cand_novel_dist])

        tf_novelty = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_tf_novelty_fraction",
        )([novel_tf, own_tf])
        tf_overlap = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True)
            / (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_tf_overlap_fraction",
        )([own_tf, others_tf])
        cand_novelty = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_candidate_novelty_fraction",
        )([novel_cand, own_cand])
        cand_overlap = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True)
            / (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_candidate_overlap_fraction",
        )([own_cand, others_cand])
        tf_other_coverage = keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
            name=f"v160_event_{q}_tf_other_coverage_fraction",
        )(others_tf)
        cand_other_coverage = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v160_event_{q}_candidate_other_coverage_fraction",
        )([others_cand, candidate_mask])

        tf_grid = keras.layers.Reshape(
            (TIME_FRAMES, token_freq), name=f"v160_event_{q}_tf_novel_grid"
        )(novel_tf)
        time_mass = keras.layers.Lambda(
            lambda a: tf.reduce_sum(a, axis=2), name=f"v160_event_{q}_time_mass"
        )(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS),
            name=f"event_time_{q}",
        )(time_mass)

        local = keras.layers.Concatenate(name=f"v160_event_{q}_local_feature")([
            candidate_context,
            queries[q],
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
        local = keras.layers.LayerNormalization(name=f"v160_event_{q}_local_norm")(local)
        local = keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1.5e-3),
            name=f"v160_event_{q}_local_hidden",
        )(local)
        local_features.append(local)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_novel_dist)

    proposal_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="v160_proposal_stack"
    )(local_features)
    attended = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.05, name="v160_global_reconciliation_attention"
    )(proposal_stack, proposal_stack)
    reconciled = keras.layers.Add(name="v160_global_reconciliation_residual")(
        [proposal_stack, attended]
    )
    reconciled = keras.layers.LayerNormalization(name="v160_global_reconciliation_norm")(reconciled)
    ff = keras.layers.Dense(128, activation="relu", name="v160_global_ff1")(reconciled)
    ff = keras.layers.Dropout(0.08, name="v160_global_dropout")(ff)
    ff = keras.layers.Dense(128, activation="relu", name="v160_global_ff2")(ff)
    reconciled = keras.layers.Add(name="v160_global_ff_residual")([reconciled, ff])
    reconciled = keras.layers.LayerNormalization(name="v160_global_ff_norm")(reconciled)

    presence_outputs = []
    for q in range(EVENT_QUERIES):
        q_feature = keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"v160_event_{q}_reconciled_feature"
        )(reconciled)
        q_feature = keras.layers.Dense(64, activation="relu", name=f"v160_event_{q}_presence_hidden")(
            q_feature
        )
        present = keras.layers.Dense(1, activation="sigmoid", name=f"event_present_{q}")(
            q_feature
        )
        presence_outputs.append(present)

    presence_vector = keras.layers.Concatenate(name="v160_event_presence_vector")(
        presence_outputs
    )
    count_norm = keras.layers.Lambda(
        lambda p: tf.reduce_sum(p, axis=1, keepdims=True) / float(EVENT_QUERIES),
        name="event_count_norm",
    )(presence_vector)

    outputs = {}
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

    model = keras.Model(base.inputs, outputs, name="v160_parallel_coverage_competition")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=loss_weights,
    )
    return model, loss_weights, token_shape


def _postprocess(output_dir: Path, fold: int, report: dict) -> dict:
    for row in report.get("strata", {}).values():
        if row and "v130" in row:
            row["v160"] = row.pop("v130")
    for row in report.get("per_true_k", {}).values():
        if row and "v130" in row:
            row["v160"] = row.pop("v130")

    arch0 = dict(report.get("architecture", {}))
    report["protocol"].update({
        "parallel_event_decoder": True,
        "sequential_stop_decoder": False,
        "global_coverage_competition": True,
        "leave_one_out_coverage": True,
        "global_query_reconciliation": True,
        "full_evidence_visible_to_every_proposal": True,
    })
    report["architecture"] = {
        "name": "V16.0 parallel birth proposals + global coverage competition",
        "event_queries": EVENT_QUERIES,
        "background_queries": 1,
        "parallel_proposals": True,
        "tf_assignment_competitive_across_events": True,
        "candidate_assignment_competitive_across_events": True,
        "leave_one_out_tf_coverage": True,
        "leave_one_out_candidate_coverage": True,
        "full_and_novel_evidence_pooled_separately": True,
        "global_self_attention_reconciliation": True,
        "sequential_state_or_stop": False,
        "event_order_training_only": "ascending birth time; stable ties",
        "headline_candidate_realization": "frozen V9+ ranking; isolates event-derived cardinality",
        "trainable_parameters": arch0.get("trainable_parameters"),
        "tf_token_shape": arch0.get("tf_token_shape"),
        "loss_weights": arch0.get("loss_weights"),
    }

    rp = output_dir / f"report-fold-{fold}.json"
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    npz_path = output_dir / f"predictions-fold-{fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    data["pred160"] = data.pop("pred130")
    np.savez_compressed(npz_path, **data)

    old_w = output_dir / f"v130-fold-{fold}.weights.h5"
    new_w = output_dir / f"v160-fold-{fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    return report


def train_fold(args):
    original = v130._build_model
    try:
        v130._build_model = _build_model
        report = v130.train_fold(args)
    finally:
        v130._build_model = original
    report = _postprocess(args.output_dir, args.outer_fold, report)
    print(json.dumps({
        "outer": args.outer_fold,
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v160_f1": report["strata"]["aggregate"]["v160"]["metrics"]["global"]["f1"],
        "poly_exact_v160": report["strata"]["aggregate"]["v160"]["cardinality"]["poly_exact_accuracy"],
        "k5_exact_v160": report["per_true_k"]["5"]["v160"]["exact"],
        "k6_exact_v160": report["per_true_k"]["6"]["v160"]["exact"],
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
