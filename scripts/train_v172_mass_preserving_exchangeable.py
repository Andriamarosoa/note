"""V17.2 three-arm causal decomposition with mass-preserving exchangeable weights.

Arms:
  ranked_ordered:
    V16-style fit-fold-only sqrt-balanced query weights + fixed ordered assignment.
  mass_ordered:
    compress the ranked query weights into true-K-conditioned object/no-object
    weights that preserve total V16-style presence coefficient mass per row,
    while removing query identity from weighting; fixed ordered assignment.
  mass_permutation:
    identical mass-preserving exchangeable weights, exact 6! assignment.

The V16 proposal graph, seed, outer-clean folds, candidate ranking and threshold
remain unchanged. No categorical cardinality head and no threshold tuning.
"""
from __future__ import annotations

import argparse
import json
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
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v160_parallel_coverage_competition as v160
from scripts import train_v171_controlled_assignment_ab as v171

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
ARMS = ("ranked_ordered", "mass_ordered", "mass_permutation")
SET_PRESENCE_WEIGHT = v171.SET_PRESENCE_WEIGHT
SET_TIME_WEIGHT = v171.SET_TIME_WEIGHT
SET_CANDIDATE_WEIGHT = v171.SET_CANDIDATE_WEIGHT
SET_VALID_OFFSET = v171.SET_VALID_OFFSET
SET_TIME_OFFSET = v171.SET_TIME_OFFSET
SET_CANDIDATE_OFFSET = v171.SET_CANDIDATE_OFFSET
PERMUTATIONS = v171.PERMUTATIONS
PERMUTATION_MATRICES = v171.PERMUTATION_MATRICES
IDENTITY_ASSIGNMENT = v171.IDENTITY_ASSIGNMENT


class V172Error(RuntimeError):
    pass


def _ranked_sqrt_weights(event_present: np.ndarray, idx: np.ndarray) -> dict:
    present = np.asarray(event_present, dtype=np.float32)[np.asarray(idx, dtype=np.int64)]
    pos, neg, prevalence, rows = [], [], [], []
    for q in range(EVENT_QUERIES):
        target = present[:, q].reshape(-1)
        sw = v130._balanced_binary_weights(target)
        pm = target > 0.5
        nm = ~pm
        wp = float(np.mean(sw[pm])) if np.any(pm) else 1.0
        wn = float(np.mean(sw[nm])) if np.any(nm) else 1.0
        pos.append(wp)
        neg.append(wn)
        prevalence.append(float(np.mean(pm)))
        rows.append({
            "query": q,
            "positive_fraction": float(np.mean(pm)),
            "positive_weight": wp,
            "negative_weight": wn,
            "positive_negative_ratio": float(wp / wn) if wn else None,
        })
    return {
        "positive": pos,
        "negative": neg,
        "prevalence": prevalence,
        "queries": rows,
    }


def _mass_tables(ranked: dict) -> dict:
    pos = np.asarray(ranked["positive"], dtype=np.float64)
    neg = np.asarray(ranked["negative"], dtype=np.float64)
    obj, null, proof = [], [], {}
    for k in range(EVENT_QUERIES + 1):
        ow = float(np.mean(pos[:k])) if k else 0.0
        nw = float(np.mean(neg[k:])) if k < EVENT_QUERIES else 0.0
        original = float(np.sum(pos[:k]) + np.sum(neg[k:]))
        compressed = float(k * ow + (EVENT_QUERIES - k) * nw)
        if abs(original - compressed) > 1e-9:
            raise V172Error(f"K={k}: mass preservation failed {original} != {compressed}")
        obj.append(ow)
        null.append(nw)
        proof[str(k)] = {
            "original_total_presence_coefficient_mass": original,
            "compressed_total_presence_coefficient_mass": compressed,
            "object_weight": ow,
            "no_object_weight": nw,
        }
    return {"object_by_k": obj, "no_object_by_k": null, "proof_by_k": proof}


def _weight_spec(event_present: np.ndarray, idx: np.ndarray) -> dict:
    ranked = _ranked_sqrt_weights(event_present, idx)
    mass = _mass_tables(ranked)
    return {"ranked": ranked, "mass": mass}


def _fold_context(args):
    ctx = v171._fold_context(args)
    ctx["meta_spec"] = _weight_spec(ctx["event_present"], ctx["meta_fit_idx"])
    ctx["final_spec"] = _weight_spec(ctx["event_present"], ctx["final_fit_idx"])
    return ctx


def _class_weight_tensor(tf, arm: str, truth_present, spec: dict):
    y = truth_present[:, None, :]
    if arm == "ranked_ordered":
        pos = tf.constant(np.asarray(spec["ranked"]["positive"], dtype=np.float32)[None, :, None])
        neg = tf.constant(np.asarray(spec["ranked"]["negative"], dtype=np.float32)[None, :, None])
        return y * pos + (1.0 - y) * neg
    k = tf.cast(tf.reduce_sum(truth_present, axis=1), tf.int32)
    obj_table = tf.constant(np.asarray(spec["mass"]["object_by_k"], dtype=np.float32))
    null_table = tf.constant(np.asarray(spec["mass"]["no_object_by_k"], dtype=np.float32))
    obj = tf.gather(obj_table, k)[:, None, None]
    null = tf.gather(null_table, k)[:, None, None]
    return y * obj + (1.0 - y) * null


def _set_loss(arm: str, spec: dict):
    import tensorflow as tf
    from tensorflow import keras
    if arm not in ARMS:
        raise V172Error(f"unsupported arm {arm!r}")
    perm = tf.constant(PERMUTATION_MATRICES, dtype=tf.float32)
    identity = tf.constant(IDENTITY_ASSIGNMENT, dtype=tf.float32)

    class MassPreservingSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name=f"v172_{arm}_event_set_loss")

        def call(self, y_true, y_pred):
            yt = tf.cast(y_true, tf.float32)
            yp = tf.cast(y_pred, tf.float32)
            truth_present = yt[:, :, 0]
            truth_valid = yt[:, :, SET_VALID_OFFSET]
            truth_time = yt[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET]
            truth_candidate = yt[:, :, SET_CANDIDATE_OFFSET:]
            pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
            pred_time = tf.clip_by_value(yp[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET], 1e-7, 1.0)
            pred_candidate = tf.clip_by_value(yp[:, :, SET_CANDIDATE_OFFSET:], 1e-7, 1.0)
            y = truth_present[:, None, :]
            p = pred_present[:, :, None]
            cw = _class_weight_tensor(tf, arm, truth_present, spec)
            presence_cost = -cw * (y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
            time_cost = -tf.einsum("btd,bqd->bqt", truth_time, tf.math.log(pred_time))
            candidate_cost = -tf.einsum("btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate))
            detail = (truth_present * truth_valid)[:, None, :]
            pair = (
                SET_PRESENCE_WEIGHT * presence_cost
                + SET_TIME_WEIGHT * detail * time_cost
                + SET_CANDIDATE_WEIGHT * detail * candidate_cost
            )
            if arm in ("ranked_ordered", "mass_ordered"):
                return tf.einsum("bqt,qt->b", pair, identity)
            scores = tf.einsum("bqt,rqt->br", pair, perm)
            return tf.reduce_min(scores, axis=1)

    return MassPreservingSetLoss()


def _build_model(arm: str, spec: dict):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    base, _, token_shape = v160._build_model()
    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    set_slots = []
    for q in range(EVENT_QUERIES):
        present = base.get_layer(f"event_present_{q}").output
        time = base.get_layer(f"event_time_{q}").output
        candidate = base.get_layer(f"event_candidate_{q}").output
        valid = keras.layers.Lambda(
            lambda x: tf.ones_like(x), name=f"v172_event_{q}_valid_placeholder"
        )(present)
        packed = keras.layers.Concatenate(name=f"v172_event_{q}_set_vector")(
            [present, valid, time, candidate]
        )
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = present
        outputs[f"event_time_{q}"] = time
        outputs[f"event_candidate_{q}"] = candidate
    outputs["event_set"] = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="event_set"
    )(set_slots)
    outputs["event_count_norm"] = base.get_layer("event_count_norm").output

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["event_set"] = _set_loss(arm, spec)
    loss["event_count_norm"] = "mse"

    lw = {f"string_{slot}": 0.18 for slot in range(SLOT_COUNT)}
    lw.update({f"pitch_{slot}": 0.04 for slot in range(SLOT_COUNT)})
    lw.update({f"time_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.35
    model = keras.Model(base.inputs, outputs, name=f"v172_{arm}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=lw)
    return model, lw, token_shape


def _numpy_class_weights(arm: str, truth_present: np.ndarray, spec: dict) -> np.ndarray:
    y = np.asarray(truth_present, dtype=np.float64)[:, None, :]
    if arm == "ranked_ordered":
        pos = np.asarray(spec["ranked"]["positive"], dtype=np.float64)[None, :, None]
        neg = np.asarray(spec["ranked"]["negative"], dtype=np.float64)[None, :, None]
        return y * pos + (1.0 - y) * neg
    k = np.rint(np.sum(truth_present, axis=1)).astype(np.int32)
    obj = np.asarray(spec["mass"]["object_by_k"], dtype=np.float64)[k][:, None, None]
    null = np.asarray(spec["mass"]["no_object_by_k"], dtype=np.float64)[k][:, None, None]
    return y * obj + (1.0 - y) * null


def _numpy_pair_cost(arm, yt, presence, time, candidate, spec):
    truth_present = np.asarray(yt[:, :, 0], dtype=np.float64)
    truth_valid = np.asarray(yt[:, :, SET_VALID_OFFSET], dtype=np.float64)
    truth_time = np.asarray(yt[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET], dtype=np.float64)
    truth_candidate = np.asarray(yt[:, :, SET_CANDIDATE_OFFSET:], dtype=np.float64)
    p = np.clip(np.asarray(presence, dtype=np.float64), 1e-6, 1.0 - 1e-6)[:, :, None]
    y = truth_present[:, None, :]
    cw = _numpy_class_weights(arm, truth_present, spec)
    presence_cost = -cw * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    pt = np.clip(np.asarray(time, dtype=np.float64), 1e-7, 1.0)
    pc = np.clip(np.asarray(candidate, dtype=np.float64), 1e-7, 1.0)
    time_cost = -np.einsum("btd,bqd->bqt", truth_time, np.log(pt))
    candidate_cost = -np.einsum("btd,bqd->bqt", truth_candidate, np.log(pc))
    detail = (truth_present * truth_valid)[:, None, :]
    return (
        SET_PRESENCE_WEIGHT * presence_cost
        + SET_TIME_WEIGHT * detail * time_cost
        + SET_CANDIDATE_WEIGHT * detail * candidate_cost
    )


def _assignment_indices(arm: str, pair_cost: np.ndarray) -> np.ndarray:
    n = pair_cost.shape[0]
    if arm in ("ranked_ordered", "mass_ordered"):
        return np.tile(np.arange(EVENT_QUERIES, dtype=np.int32), (n, 1))
    scores = np.einsum("bqt,rqt->br", pair_cost, PERMUTATION_MATRICES)
    return PERMUTATIONS[np.argmin(scores, axis=1)]


def _match_occupancy(arm, yt, presence, time, candidate, spec):
    pair = _numpy_pair_cost(arm, yt, presence, time, candidate, spec)
    assigned = _assignment_indices(arm, pair)
    truth_present = np.asarray(yt[:, :, 0], dtype=np.float64)
    truth_valid = np.asarray(yt[:, :, SET_VALID_OFFSET], dtype=np.float64)
    out = {}
    for q in range(EVENT_QUERIES):
        t = assigned[:, q]
        obj = truth_present[np.arange(len(t)), t] > 0.5
        valid = truth_valid[np.arange(len(t)), t] > 0.5
        out[str(q)] = {
            "matched_object_rate": float(np.mean(obj)),
            "matched_no_object_rate": float(np.mean(~obj)),
            "matched_valid_detail_rate": float(np.mean(valid)),
            "truth_slot_distribution": {
                str(s): int(np.sum(t == s)) for s in range(EVENT_QUERIES)
            },
        }
    return out


def _gradient_mass(model, cache, idx, yt, arm, spec):
    import tensorflow as tf
    idx = np.asarray(idx, dtype=np.int64)[: min(256, len(idx))]
    yt = np.asarray(yt, dtype=np.float32)[: len(idx)]
    x = v102._inputs(cache, idx)

    with tf.GradientTape(persistent=True) as tape:
        yp = tf.cast(model(x, training=False)["event_set"], tf.float32)
        truth = tf.constant(yt, dtype=tf.float32)
        truth_present = truth[:, :, 0]
        pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
        truth_valid = truth[:, :, SET_VALID_OFFSET]
        truth_time = truth[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET]
        truth_candidate = truth[:, :, SET_CANDIDATE_OFFSET:]
        pred_time = tf.clip_by_value(yp[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET], 1e-7, 1.0)
        pred_candidate = tf.clip_by_value(yp[:, :, SET_CANDIDATE_OFFSET:], 1e-7, 1.0)
        y = truth_present[:, None, :]
        p = pred_present[:, :, None]
        cw = _class_weight_tensor(tf, arm, truth_present, spec)
        presence_cost = -cw * (y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        time_cost = -tf.einsum("btd,bqd->bqt", truth_time, tf.math.log(pred_time))
        candidate_cost = -tf.einsum("btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate))
        detail = (truth_present * truth_valid)[:, None, :]
        pair = (
            SET_PRESENCE_WEIGHT * presence_cost
            + SET_TIME_WEIGHT * detail * time_cost
            + SET_CANDIDATE_WEIGHT * detail * candidate_cost
        )
        if arm in ("ranked_ordered", "mass_ordered"):
            assignment = tf.tile(tf.range(EVENT_QUERIES)[None, :], [tf.shape(yp)[0], 1])
        else:
            scores = tf.einsum(
                "bqt,rqt->br", pair, tf.constant(PERMUTATION_MATRICES, dtype=tf.float32)
            )
            best = tf.argmin(scores, axis=1, output_type=tf.int32)
            assignment = tf.gather(tf.constant(PERMUTATIONS, dtype=tf.int32), best)

        batch = tf.range(tf.shape(assignment)[0], dtype=tf.int32)[:, None]
        gather = tf.stack([tf.tile(batch, [1, EVENT_QUERIES]), assignment], axis=-1)
        matched_y = tf.gather_nd(truth_present, gather)
        q_idx = tf.tile(tf.range(EVENT_QUERIES, dtype=tf.int32)[None, :], [tf.shape(assignment)[0], 1])
        pair_idx = tf.stack([tf.tile(batch, [1, EVENT_QUERIES]), q_idx, assignment], axis=-1)
        matched_w = tf.gather_nd(cw, pair_idx)
        pos_loss = tf.reduce_mean(
            tf.reduce_sum(-matched_w * matched_y * tf.math.log(pred_present), axis=1)
        )
        neg_loss = tf.reduce_mean(
            tf.reduce_sum(-matched_w * (1.0 - matched_y) * tf.math.log(1.0 - pred_present), axis=1)
        )

    gp = [g for g in tape.gradient(pos_loss, model.trainable_variables) if g is not None]
    gn = [g for g in tape.gradient(neg_loss, model.trainable_variables) if g is not None]
    del tape
    return {
        "rows": int(len(idx)),
        "positive_presence_loss": float(pos_loss.numpy()),
        "negative_presence_loss": float(neg_loss.numpy()),
        "positive_gradient_global_norm": float(tf.linalg.global_norm(gp).numpy()) if gp else 0.0,
        "negative_gradient_global_norm": float(tf.linalg.global_norm(gn).numpy()) if gn else 0.0,
    }


def _postprocess(args, report, ctx):
    arm = args.arm
    key = f"v172_{arm}"
    v171._rename_report(report, "v130", key)
    report["protocol"].update({
        "v172_mass_preserving_decomposition": True,
        "assignment_arm": arm,
        "v16_proposal_graph_unchanged": True,
        "shared_seed": int(args.seed),
        "presence_threshold": PRESENCE_THRESHOLD,
        "presence_threshold_tuned": False,
        "categorical_cardinality_head_exists": False,
        "fit_fold_only_sqrt_presence_weighting": True,
        "linear_equal_mass_weighting": False,
        "ranked_query_weighting": arm == "ranked_ordered",
        "cardinality_conditioned_exchangeable_weighting": arm.startswith("mass_"),
        "permutation_invariant_set_matching": arm == "mass_permutation",
        "set_matching_permutations": int(len(PERMUTATIONS)) if arm == "mass_permutation" else 1,
        "historical_validation_or_locked12_indexed_or_evaluated": False,
    })
    report["v172"] = {
        "arm": arm,
        "meta_fit_weight_spec": ctx["meta_spec"],
        "final_fit_weight_spec": ctx["final_spec"],
        "matching_cost_weights": {
            "presence": SET_PRESENCE_WEIGHT,
            "time": SET_TIME_WEIGHT,
            "candidate": SET_CANDIDATE_WEIGHT,
        },
    }

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}

    presence = np.asarray(data["presence"], dtype=np.float64)
    report["v172"]["outer_presence"] = {
        str(q): {
            "mean": float(np.mean(presence[:, q])),
            "active_rate_at_0p5": float(np.mean(presence[:, q] >= PRESENCE_THRESHOLD)),
        }
        for q in range(EVENT_QUERIES)
    }

    cache = ctx["cache"]
    members = ctx["members"]
    candidate_samples, _ = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, string_time_targets, time_sample, _ = v102._derive_supervision(
        members,
        candidate_samples,
        args.dataset_dir,
        expected_slot_targets=cache["slot_targets"],
    )
    event_present, event_time, event_candidate, event_valid, _, _ = v130._ordered_event_supervision(
        cache, time_mask, string_time_targets, time_sample, ctx["k"]
    )
    all_targets = v171._targets(
        cache,
        pitch_targets,
        string_time_targets,
        ctx["k"],
        event_present,
        event_time,
        event_candidate,
    )
    outer_idx = ctx["outer_idx"]
    yt_outer = np.asarray(all_targets["event_set"])[outer_idx]
    report["v172"]["outer_match_occupancy"] = _match_occupancy(
        arm,
        yt_outer,
        data["presence"],
        data["event_time"],
        data["event_candidate"],
        ctx["final_spec"],
    )

    old_w = args.output_dir / f"v130-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            random.seed(args.seed + 1000 + args.outer_fold)
            np.random.seed(args.seed + 1000 + args.outer_fold)
            tf.random.set_seed(args.seed + 1000 + args.outer_fold)
            gm, _, _ = _build_model(arm, ctx["final_spec"])
            gm.load_weights(old_w)
            fit_sample = ctx["final_fit_idx"][: min(256, len(ctx["final_fit_idx"]))]
            yt_fit = np.asarray(all_targets["event_set"])[fit_sample]
            report["v172"]["final_model_presence_gradient_mass"] = _gradient_mass(
                gm, cache, fit_sample, yt_fit, arm, ctx["final_spec"]
            )
        except Exception as exc:
            report["v172"]["final_model_presence_gradient_mass_error"] = repr(exc)

    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    data[f"pred172_{arm}"] = data.pop("pred130")
    np.savez_compressed(npz_path, **data)
    new_w = args.output_dir / f"v172-{arm}-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    return report


def train_fold(args):
    if args.arm not in ARMS:
        raise V172Error(f"arm must be one of {ARMS}")
    if args.seed != DEFAULT_SEED:
        raise V172Error(f"V17.2 requires seed {DEFAULT_SEED}, got {args.seed}")

    ctx = _fold_context(args)
    stage_specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V172Error(f"unexpected model build call {i + 1}")
        calls["count"] += 1
        return _build_model(args.arm, stage_specs[i])

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = v171._targets
        v130._sample_weights = v171._sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights

    if calls["count"] != 2:
        raise V172Error(f"expected exactly 2 model builds, got {calls['count']}")
    report = _postprocess(args, report, ctx)
    key = f"v172_{args.arm}"
    card = report["strata"]["aggregate"][key]["cardinality"]
    print(json.dumps({
        "arm": args.arm,
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "controlled_f1": report["strata"]["aggregate"][key]["metrics"]["global"]["f1"],
        "pred_ref": report["strata"]["aggregate"][key]["metrics"]["global"]["prediction_reference_ratio"],
        "poly_exact": card.get("poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))),
        "k5": report["per_true_k"]["5"][key]["exact"],
        "k6": report["per_true_k"]["6"][key]["exact"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
