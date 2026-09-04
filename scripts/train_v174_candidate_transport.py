"""V17.4 presence-mass capacity transport for candidate ownership.

Starts from V17.3 and changes one architectural mechanism only: the six
candidate distributions are coupled through a differentiable, permutation-
equivariant capacity transport before the exact set loss and runtime candidate
selection.  Row mass is each query's (stop-gradient) presence probability, so
inactive proposals consume little ownership capacity while active proposals
compete for distinct candidates.

Preserved from V17.3:
- V16 trainable backbone and six proposal/objectness branches;
- fit-fold-only mass-preserving exchangeable presence weights;
- exact 6! truth assignment;
- exact Poisson-binomial count NLL at weight 0.35;
- K=sum(presence>=0.5), threshold 0.5, no categorical K head;
- seed/folds/candidate evidence/Locked12 protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Optional, Sequence

import numpy as np

from causal_note.guitarset import SLOT_COUNT
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v160_parallel_coverage_competition as v160
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v173_poibin_count_consistency as v173

DEFAULT_SEED = v173.DEFAULT_SEED
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = v173.PRESENCE_THRESHOLD
BASE_ARM = v173.BASE_ARM
MODEL_KEY = "v174_transport"
TRANSPORT_ITERATIONS = 32
TRANSPORT_EPS = 1e-6
MIN_ROW_MASS = 1e-4


class V174Error(RuntimeError):
    pass


def _capacity_transport_np(candidate, presence, mask, iterations=TRANSPORT_ITERATIONS):
    """Reference NumPy implementation of the presence-mass capacity projection."""
    c = np.maximum(np.asarray(candidate, dtype=np.float64), 0.0)
    p = np.clip(np.asarray(presence, dtype=np.float64), MIN_ROW_MASS, 1.0)
    m = (np.asarray(mask, dtype=np.float64) > 0.5).astype(np.float64)
    if c.ndim != 3 or c.shape[1] != EVENT_QUERIES:
        raise V174Error(f"candidate must be [B,{EVENT_QUERIES},C], got {c.shape}")
    if p.shape != c.shape[:2] or m.shape != (c.shape[0], c.shape[2]):
        raise V174Error(f"presence/mask shape mismatch: c={c.shape} p={p.shape} m={m.shape}")

    valid_count = np.sum(m, axis=1, keepdims=True)
    fallback = m[:, None, :] / np.maximum(valid_count[:, None, :], 1.0)
    cond = (c + TRANSPORT_EPS) * m[:, None, :]
    row = np.sum(cond, axis=2, keepdims=True)
    cond = np.where(row > TRANSPORT_EPS, cond / np.maximum(row, TRANSPORT_EPS), fallback)

    target = p[:, :, None]
    mass = cond * target
    total_mass = np.sum(target, axis=1)  # [B,1]
    capacity = np.maximum(1.0, total_mass / np.maximum(valid_count, 1.0))

    for _ in range(int(iterations)):
        row = np.sum(mass, axis=2, keepdims=True)
        mass = mass * (target / np.maximum(row, TRANSPORT_EPS))
        col = np.sum(mass, axis=1)
        scale = np.minimum(1.0, capacity / np.maximum(col, TRANSPORT_EPS))
        mass = mass * scale[:, None, :]

    conditional = mass / np.maximum(target, TRANSPORT_EPS)
    conditional *= m[:, None, :]
    return conditional, mass, capacity


def _capacity_transport_tf(tf, candidate, presence, mask):
    c = tf.maximum(tf.cast(candidate, tf.float32), 0.0)
    # Stop candidate-detail gradients from changing objectness/count economics.
    p = tf.stop_gradient(
        tf.clip_by_value(tf.cast(presence, tf.float32), MIN_ROW_MASS, 1.0)
    )
    m = tf.cast(tf.cast(mask, tf.float32) > 0.5, tf.float32)
    valid_count = tf.reduce_sum(m, axis=1, keepdims=True)
    fallback = m[:, None, :] / tf.maximum(valid_count[:, None, :], 1.0)
    cond = (c + tf.constant(TRANSPORT_EPS, tf.float32)) * m[:, None, :]
    row = tf.reduce_sum(cond, axis=2, keepdims=True)
    cond = tf.where(
        row > TRANSPORT_EPS,
        cond / tf.maximum(row, TRANSPORT_EPS),
        fallback,
    )

    target = p[:, :, None]
    mass = cond * target
    total_mass = tf.reduce_sum(target, axis=1)
    capacity = tf.maximum(1.0, total_mass / tf.maximum(valid_count, 1.0))
    for _ in range(TRANSPORT_ITERATIONS):
        row = tf.reduce_sum(mass, axis=2, keepdims=True)
        mass = mass * (target / tf.maximum(row, TRANSPORT_EPS))
        col = tf.reduce_sum(mass, axis=1)
        scale = tf.minimum(1.0, capacity / tf.maximum(col, TRANSPORT_EPS))
        mass = mass * scale[:, None, :]

    conditional = mass / tf.maximum(target, TRANSPORT_EPS)
    return conditional * m[:, None, :]


def _build_model(spec: dict):
    """Reuse V16/V17.3 trainable graph and replace candidate output by transport."""
    import tensorflow as tf
    from tensorflow import keras

    base, _, token_shape = v160._build_model()
    candidate_mask = v160._input_by_name(base, "candidate_mask")
    presence = [base.get_layer(f"event_present_{q}").output for q in range(EVENT_QUERIES)]
    time = [base.get_layer(f"event_time_{q}").output for q in range(EVENT_QUERIES)]
    raw_candidate = [base.get_layer(f"event_candidate_{q}").output for q in range(EVENT_QUERIES)]

    presence_vector = keras.layers.Concatenate(name="v174_presence_vector")(presence)
    candidate_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="v174_raw_candidate_stack"
    )(raw_candidate)
    transported = keras.layers.Lambda(
        lambda z: _capacity_transport_tf(tf, z[0], z[1], z[2]),
        name="v174_presence_mass_candidate_transport",
    )([candidate_stack, presence_vector, candidate_mask])
    candidate = [
        keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"v174_event_candidate_{q}"
        )(transported)
        for q in range(EVENT_QUERIES)
    ]

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output

    set_slots = []
    for q in range(EVENT_QUERIES):
        valid = keras.layers.Lambda(
            lambda x: tf.ones_like(x), name=f"v174_event_{q}_valid_placeholder"
        )(presence[q])
        packed = keras.layers.Concatenate(name=f"v174_event_{q}_set_vector")(
            [presence[q], valid, time[q], candidate[q]]
        )
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = presence[q]
        outputs[f"event_time_{q}"] = time[q]
        outputs[f"event_candidate_{q}"] = candidate[q]

    outputs["event_set"] = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="event_set"
    )(set_slots)
    outputs["event_count_norm"] = base.get_layer("event_count_norm").output

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
    model = keras.Model(base.inputs, outputs, name="v174_candidate_transport")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=lw)
    return model, lw, token_shape


def _duplicate_argmax_rate(dist: np.ndarray, active: np.ndarray):
    vals = []
    for r in range(len(active)):
        ids = np.flatnonzero(active[r])
        if len(ids) < 2:
            continue
        arg = np.argmax(dist[r, ids], axis=1)
        vals.append(len(np.unique(arg)) < len(arg))
    return float(np.mean(vals)) if vals else None


def _postprocess(args, report, ctx):
    # Reuse all V17.3 outer-clean diagnostics and protocol checks first.
    report = v173._postprocess(args, report, ctx)
    v171._rename_report(report, v173.MODEL_KEY, MODEL_KEY)

    report["protocol"].update(
        {
            "v174_presence_mass_candidate_transport": True,
            "v174_only_architecture_change_from_v173": "candidate ownership capacity transport",
            "v16_proposal_graph_unchanged": False,
            "v16_trainable_backbone_unchanged": True,
            "v173_objectness_count_objective_unchanged": True,
            "runtime_graph_unchanged_from_v172_c": False,
            "runtime_decode_unchanged_from_v172_c": False,
            "runtime_count_decode_unchanged_from_v173": True,
            "runtime_candidate_ownership_changed": True,
            "runtime_presence_threshold": PRESENCE_THRESHOLD,
            "runtime_presence_threshold_tuned": False,
            "candidate_transport_iterations": TRANSPORT_ITERATIONS,
            "candidate_transport_presence_stop_gradient": True,
            "candidate_transport_column_capacity": "max(1, sum_presence / valid_candidate_count)",
            "categorical_cardinality_head_exists": False,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
        }
    )

    inherited = report.pop("v173")
    report["v174"] = {
        **inherited,
        "model_key": MODEL_KEY,
        "base_version": "V17.3",
        "candidate_transport": {
            "type": "presence_mass_capacity_projection",
            "iterations": TRANSPORT_ITERATIONS,
            "epsilon": TRANSPORT_EPS,
            "minimum_row_mass": MIN_ROW_MASS,
            "presence_gradient_stopped": True,
            "trainable_parameters_added": 0,
        },
    }

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    old_pred = "pred173_poibin"
    if old_pred not in data:
        raise V174Error(f"missing postprocessed prediction key {old_pred}")
    data["pred174_transport"] = data.pop(old_pred)
    np.savez_compressed(npz_path, **data)

    k = np.asarray(data["k"], dtype=np.int32)
    presence = np.asarray(data["presence"], dtype=np.float64)
    candidate = np.asarray(data["event_candidate"], dtype=np.float64)
    hard = np.sum(presence >= PRESENCE_THRESHOLD, axis=1).astype(np.int32)
    report["v174"]["candidate_duplicate_argmax_by_true_k"] = {}
    for value in range(2, EVENT_QUERIES + 1):
        m = k == value
        exact = m & (hard == value)
        report["v174"]["candidate_duplicate_argmax_by_true_k"][str(value)] = {
            "all_rows": _duplicate_argmax_rate(candidate[m], presence[m] >= PRESENCE_THRESHOLD),
            "exact_count_rows": _duplicate_argmax_rate(candidate[exact], presence[exact] >= PRESENCE_THRESHOLD),
            "exact_count_clusters": int(np.sum(exact)),
        }

    old_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v174-transport-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)

    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def train_fold(args):
    if args.seed != DEFAULT_SEED:
        raise V174Error(f"V17.4 requires seed {DEFAULT_SEED}, got {args.seed}")
    if args.arm != BASE_ARM:
        raise V174Error(f"V17.4 only supports base arm {BASE_ARM!r}")

    ctx = v172._fold_context(args)
    stage_specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V174Error(f"unexpected model build call {i + 1}")
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
        raise V174Error(f"expected exactly 2 model builds, got {calls['count']}")
    report = _postprocess(args, report, ctx)
    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    d = report["v174"]["candidate_duplicate_argmax_by_true_k"]
    print(
        json.dumps(
            {
                "outer": args.outer_fold,
                "selected_epochs": report["data"]["selected_epochs"],
                "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
                "v174_f1": g["f1"],
                "pred_ref": g["prediction_reference_ratio"],
                "k2": report["per_true_k"]["2"][MODEL_KEY]["exact"],
                "k3": report["per_true_k"]["3"][MODEL_KEY]["exact"],
                "k5": report["per_true_k"]["5"][MODEL_KEY]["exact"],
                "k6": report["per_true_k"]["6"][MODEL_KEY]["exact"],
                "duplicate_exact_k2": d["2"]["exact_count_rows"],
                "duplicate_exact_k5": d["5"]["exact_count_rows"],
                "duplicate_exact_k6": d["6"]["exact_count_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=Path("data/GuitarSet"))
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
