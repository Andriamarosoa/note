"""V17.1 controlled A/B: ordered assignment vs exact permutation matching.

Both arms use the exact V16 proposal graph, seed 16061, fixed 0.5 runtime
threshold, the same fold split and the same fit-fold-only symmetric object /
no-object weighting. The only treatment is event-target assignment:
  ordered: identity q -> time-sorted truth slot
  permutation: exact minimum over all 6! one-to-one assignments
"""
from __future__ import annotations

import argparse
import itertools
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
from scripts.train_boundaries import group_stem
from scripts.train_v100_spectral_string_slots import TIME_FRAMES, _load_spectral_caches
from scripts.train_v90_structured_cluster_cardinality import MAX_CANDIDATES
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_oof_fold as oofmod
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v160_parallel_coverage_competition as v160

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
SET_PRESENCE_WEIGHT = 1.0
SET_TIME_WEIGHT = 0.30
SET_CANDIDATE_WEIGHT = 0.25
SET_VALID_OFFSET = 1
SET_TIME_OFFSET = 2
SET_CANDIDATE_OFFSET = SET_TIME_OFFSET + TIME_FRAMES
SET_WIDTH = 2 + TIME_FRAMES + MAX_CANDIDATES
ARMS = ("ordered", "permutation")
PERMUTATIONS = np.asarray(list(itertools.permutations(range(EVENT_QUERIES))), dtype=np.int32)


class V171Error(RuntimeError):
    pass


def _permutation_matrices():
    mats = np.zeros((len(PERMUTATIONS), EVENT_QUERIES, EVENT_QUERIES), dtype=np.float32)
    for r, perm in enumerate(PERMUTATIONS):
        for q, truth in enumerate(perm):
            mats[r, q, truth] = 1.0
    return mats


PERMUTATION_MATRICES = _permutation_matrices()
IDENTITY_ASSIGNMENT = np.eye(EVENT_QUERIES, dtype=np.float32)


def _symmetric_class_weights(event_present: np.ndarray, idx: np.ndarray) -> dict:
    labels = np.asarray(event_present, dtype=np.float32)[np.asarray(idx, dtype=np.int64)].reshape(-1)
    pos = int(np.sum(labels > 0.5))
    neg = int(len(labels) - pos)
    if pos <= 0 or neg <= 0:
        raise V171Error(f"cannot balance presence classes: pos={pos} neg={neg}")
    pos_weight = float(len(labels) / (2.0 * pos))
    neg_weight = float(len(labels) / (2.0 * neg))
    return {
        "labels": int(len(labels)),
        "positive_labels": pos,
        "negative_labels": neg,
        "positive_fraction": float(pos / len(labels)),
        "positive_weight": pos_weight,
        "negative_weight": neg_weight,
        "positive_coefficient_mass": float(pos * pos_weight),
        "negative_coefficient_mass": float(neg * neg_weight),
    }


def _fold_context(args):
    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
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
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), EVENT_QUERIES)
    event_present = (np.arange(EVENT_QUERIES, dtype=np.int32)[None, :] < k[:, None]).astype(np.float32)
    return {
        "cache": cache,
        "train_split": train_split,
        "validation": validation,
        "groups_per_fold": groups_per_fold,
        "members": members,
        "outer_idx": outer_idx,
        "meta_fit_idx": meta_fit_idx,
        "meta_val_idx": meta_val_idx,
        "final_fit_idx": final_fit_idx,
        "k": k,
        "event_present": event_present,
        "meta_weights": _symmetric_class_weights(event_present, meta_fit_idx),
        "final_weights": _symmetric_class_weights(event_present, final_fit_idx),
    }


def _set_loss(arm: str, positive_weight: float, negative_weight: float):
    import tensorflow as tf
    from tensorflow import keras
    if arm not in ARMS:
        raise V171Error(f"unsupported arm {arm!r}")
    perm = tf.constant(PERMUTATION_MATRICES, dtype=tf.float32)
    identity = tf.constant(IDENTITY_ASSIGNMENT, dtype=tf.float32)
    wp = tf.constant(float(positive_weight), dtype=tf.float32)
    wn = tf.constant(float(negative_weight), dtype=tf.float32)

    class ControlledAssignmentSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name=f"v171_{arm}_controlled_event_set_loss")

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
            class_weight = y * wp + (1.0 - y) * wn
            presence_cost = -class_weight * (y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
            time_cost = -tf.einsum("btd,bqd->bqt", truth_time, tf.math.log(pred_time))
            candidate_cost = -tf.einsum("btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate))
            detail = (truth_present * truth_valid)[:, None, :]
            pair_cost = SET_PRESENCE_WEIGHT * presence_cost + SET_TIME_WEIGHT * detail * time_cost + SET_CANDIDATE_WEIGHT * detail * candidate_cost
            if arm == "ordered":
                return tf.einsum("bqt,qt->b", pair_cost, identity)
            permutation_cost = tf.einsum("bqt,rqt->br", pair_cost, perm)
            return tf.reduce_min(permutation_cost, axis=1)
    return ControlledAssignmentSetLoss()


def _build_controlled_model(arm: str, weights: dict):
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
        valid_placeholder = keras.layers.Lambda(lambda x: tf.ones_like(x), name=f"v171_event_{q}_valid_placeholder")(present)
        packed = keras.layers.Concatenate(name=f"v171_event_{q}_set_vector")([present, valid_placeholder, time, candidate])
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = present
        outputs[f"event_time_{q}"] = time
        outputs[f"event_candidate_{q}"] = candidate
    outputs["event_set"] = keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="event_set")(set_slots)
    outputs["event_count_norm"] = base.get_layer("event_count_norm").output
    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["event_set"] = _set_loss(arm, weights["positive_weight"], weights["negative_weight"])
    loss["event_count_norm"] = "mse"
    loss_weights = {f"string_{slot}": 0.18 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.04 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    loss_weights["event_set"] = 1.0
    loss_weights["event_count_norm"] = 0.35
    model = keras.Model(base.inputs, outputs, name=f"v171_{arm}_controlled_assignment")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, token_shape


def _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate):
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = np.asarray(cache["slot_targets"][:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"pitch_{slot}"] = np.asarray(pitch_targets[:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"time_{slot}"] = np.asarray(string_time_targets[:, slot, :], dtype=np.float32)
    present = np.asarray(event_present, dtype=np.float32)
    valid = (np.sum(np.asarray(event_candidate, dtype=np.float32), axis=2) > 0.5).astype(np.float32)
    packed = np.concatenate([present[:, :, None], valid[:, :, None], np.asarray(event_time, dtype=np.float32), np.asarray(event_candidate, dtype=np.float32)], axis=2)
    if packed.shape[2] != SET_WIDTH:
        raise V171Error(f"unexpected event-set width {packed.shape}")
    out["event_set"] = packed
    out["event_count_norm"] = (np.asarray(k, dtype=np.float32) / float(EVENT_QUERIES)).reshape(-1, 1)
    return out


def _sample_weights(cache, time_mask, k, event_present, event_valid):
    base = v102._sample_weights(cache["slot_targets"], time_mask, k)
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = base[f"string_{slot}"]
        out[f"pitch_{slot}"] = base[f"pitch_{slot}"]
        out[f"time_{slot}"] = base[f"time_{slot}"]
    out["event_set"] = np.ones(len(k), dtype=np.float32)
    out["event_count_norm"] = v102._count_weights(np.asarray(k, dtype=np.int32))
    return out


def _numpy_pair_cost(yt, presence, time, candidate, weights):
    truth_present = np.asarray(yt[:, :, 0], dtype=np.float64)
    truth_valid = np.asarray(yt[:, :, SET_VALID_OFFSET], dtype=np.float64)
    truth_time = np.asarray(yt[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET], dtype=np.float64)
    truth_candidate = np.asarray(yt[:, :, SET_CANDIDATE_OFFSET:], dtype=np.float64)
    p = np.clip(np.asarray(presence, dtype=np.float64), 1e-6, 1.0 - 1e-6)[:, :, None]
    y = truth_present[:, None, :]
    cw = y * float(weights["positive_weight"]) + (1.0 - y) * float(weights["negative_weight"])
    presence_cost = -cw * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    pt = np.clip(np.asarray(time, dtype=np.float64), 1e-7, 1.0)
    pc = np.clip(np.asarray(candidate, dtype=np.float64), 1e-7, 1.0)
    time_cost = -np.einsum("btd,bqd->bqt", truth_time, np.log(pt))
    candidate_cost = -np.einsum("btd,bqd->bqt", truth_candidate, np.log(pc))
    detail = (truth_present * truth_valid)[:, None, :]
    return SET_PRESENCE_WEIGHT * presence_cost + SET_TIME_WEIGHT * detail * time_cost + SET_CANDIDATE_WEIGHT * detail * candidate_cost


def _assignment_indices(arm, pair_cost):
    n = pair_cost.shape[0]
    if arm == "ordered":
        return np.tile(np.arange(EVENT_QUERIES, dtype=np.int32), (n, 1))
    scores = np.einsum("bqt,rqt->br", pair_cost, PERMUTATION_MATRICES)
    best = np.argmin(scores, axis=1)
    return PERMUTATIONS[best]


def _match_occupancy(arm, yt, presence, time, candidate, weights):
    pair = _numpy_pair_cost(yt, presence, time, candidate, weights)
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
            "truth_slot_distribution": {str(s): int(np.sum(t == s)) for s in range(EVENT_QUERIES)},
        }
    return out


def _gradient_mass(model, cache, idx, yt, arm, weights):
    import tensorflow as tf
    idx = np.asarray(idx, dtype=np.int64)[: min(256, len(idx))]
    yt = np.asarray(yt, dtype=np.float32)[: len(idx)]
    x = v102._inputs(cache, idx)
    with tf.GradientTape(persistent=True) as tape:
        yp = tf.cast(model(x, training=False)["event_set"], tf.float32)
        truth = tf.constant(yt, dtype=tf.float32)
        truth_present = truth[:, :, 0]
        truth_valid = truth[:, :, SET_VALID_OFFSET]
        truth_time = truth[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET]
        truth_candidate = truth[:, :, SET_CANDIDATE_OFFSET:]
        pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
        pred_time = tf.clip_by_value(yp[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET], 1e-7, 1.0)
        pred_candidate = tf.clip_by_value(yp[:, :, SET_CANDIDATE_OFFSET:], 1e-7, 1.0)
        y = truth_present[:, None, :]
        p = pred_present[:, :, None]
        wp = tf.constant(float(weights["positive_weight"]), dtype=tf.float32)
        wn = tf.constant(float(weights["negative_weight"]), dtype=tf.float32)
        class_weight = y * wp + (1.0 - y) * wn
        presence_cost = -class_weight * (y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        time_cost = -tf.einsum("btd,bqd->bqt", truth_time, tf.math.log(pred_time))
        candidate_cost = -tf.einsum("btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate))
        detail = (truth_present * truth_valid)[:, None, :]
        pair_cost = SET_PRESENCE_WEIGHT * presence_cost + SET_TIME_WEIGHT * detail * time_cost + SET_CANDIDATE_WEIGHT * detail * candidate_cost
        if arm == "ordered":
            assignment = tf.tile(tf.range(EVENT_QUERIES)[None, :], [tf.shape(yp)[0], 1])
        else:
            scores = tf.einsum("bqt,rqt->br", pair_cost, tf.constant(PERMUTATION_MATRICES, dtype=tf.float32))
            best = tf.argmin(scores, axis=1, output_type=tf.int32)
            assignment = tf.gather(tf.constant(PERMUTATIONS, dtype=tf.int32), best)
        batch = tf.range(tf.shape(assignment)[0], dtype=tf.int32)[:, None]
        gather = tf.stack([tf.tile(batch, [1, EVENT_QUERIES]), assignment], axis=-1)
        matched_y = tf.gather_nd(truth_present, gather)
        pos = matched_y
        neg = 1.0 - matched_y
        pos_loss = tf.reduce_mean(tf.reduce_sum(-wp * pos * tf.math.log(pred_present), axis=1))
        neg_loss = tf.reduce_mean(tf.reduce_sum(-wn * neg * tf.math.log(1.0 - pred_present), axis=1))
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


def _rename_report(report: dict, src: str, dst: str):
    for row in report.get("strata", {}).values():
        if row and src in row:
            row[dst] = row.pop(src)
    for row in report.get("per_true_k", {}).values():
        if row and src in row:
            row[dst] = row.pop(src)


def _postprocess(args, report, ctx):
    arm = args.arm
    key = f"v171_{arm}"
    _rename_report(report, "v130", key)
    report["protocol"].update({
        "controlled_assignment_ab": True,
        "assignment_arm": arm,
        "only_assignment_differs_between_arms": True,
        "v16_proposal_graph_unchanged": True,
        "shared_seed": int(args.seed),
        "presence_threshold": PRESENCE_THRESHOLD,
        "presence_threshold_tuned": False,
        "categorical_cardinality_head_exists": False,
        "fit_fold_only_presence_weighting": True,
        "symmetric_object_no_object_weighting": True,
        "ordered_query_presence_targets": arm == "ordered",
        "permutation_invariant_set_matching": arm == "permutation",
        "set_matching_permutations": int(len(PERMUTATIONS)) if arm == "permutation" else 1,
    })
    report["controlled_ab"] = {
        "arm": arm,
        "seed": int(args.seed),
        "meta_fit_presence_weights": ctx["meta_weights"],
        "final_fit_presence_weights": ctx["final_weights"],
        "matching_cost_weights": {"presence": SET_PRESENCE_WEIGHT, "time": SET_TIME_WEIGHT, "candidate": SET_CANDIDATE_WEIGHT},
    }
    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    cache = ctx["cache"]
    members = ctx["members"]
    candidate_samples, _ = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, string_time_targets, time_sample, _ = v102._derive_supervision(members, candidate_samples, args.dataset_dir, expected_slot_targets=cache["slot_targets"])
    event_present, event_time, event_candidate, event_valid, _, _ = v130._ordered_event_supervision(cache, time_mask, string_time_targets, time_sample, ctx["k"])
    all_targets = _targets(cache, pitch_targets, string_time_targets, ctx["k"], event_present, event_time, event_candidate)
    outer_idx = ctx["outer_idx"]
    yt_outer = np.asarray(all_targets["event_set"])[outer_idx]
    report["controlled_ab"]["outer_match_occupancy"] = _match_occupancy(arm, yt_outer, data["presence"], data["event_time"], data["event_candidate"], ctx["final_weights"])
    old_w = args.output_dir / f"v130-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            random.seed(args.seed + 1000 + args.outer_fold)
            np.random.seed(args.seed + 1000 + args.outer_fold)
            tf.random.set_seed(args.seed + 1000 + args.outer_fold)
            gm, _, _ = _build_controlled_model(arm, ctx["final_weights"])
            gm.load_weights(old_w)
            fit_sample = ctx["final_fit_idx"][: min(256, len(ctx["final_fit_idx"]))]
            yt_fit = np.asarray(all_targets["event_set"])[fit_sample]
            report["controlled_ab"]["final_model_presence_gradient_mass"] = _gradient_mass(gm, cache, fit_sample, yt_fit, arm, ctx["final_weights"])
        except Exception as exc:
            report["controlled_ab"]["final_model_presence_gradient_mass_error"] = repr(exc)
    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    data[f"pred171_{arm}"] = data.pop("pred130")
    np.savez_compressed(npz_path, **data)
    new_w = args.output_dir / f"v171-{arm}-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    return report


def train_fold(args):
    if args.arm not in ARMS:
        raise V171Error(f"arm must be one of {ARMS}")
    if args.seed != DEFAULT_SEED:
        raise V171Error(f"controlled A/B requires seed {DEFAULT_SEED}, got {args.seed}")
    ctx = _fold_context(args)
    stage_weights = [ctx["meta_weights"], ctx["final_weights"]]
    build_calls = {"count": 0}
    def builder():
        i = build_calls["count"]
        if i >= 2:
            raise V171Error(f"unexpected model build call {i + 1}")
        build_calls["count"] += 1
        return _build_controlled_model(args.arm, stage_weights[i])
    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = _targets
        v130._sample_weights = _sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights
    if build_calls["count"] != 2:
        raise V171Error(f"expected exactly 2 model builds, got {build_calls['count']}")
    report = _postprocess(args, report, ctx)
    key = f"v171_{args.arm}"
    card = report["strata"]["aggregate"][key]["cardinality"]
    print(json.dumps({
        "arm": args.arm,
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "controlled_f1": report["strata"]["aggregate"][key]["metrics"]["global"]["f1"],
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
