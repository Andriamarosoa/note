"""V17.0 permutation-invariant event-set matching on the V16 proposal graph.

V16 proved that removing sequential STOP can recover high cardinality (especially
K5), but the deep audit showed that its supervision is still ordinal: q0..q5
are trained as K>=1..K>=6. q5 therefore has only 104 positive rows, even though
its AUC is ~0.97, and calibration varies wildly across outer folds. The same
audit also found frequent duplicate proposal claims while 97%+ of V16
undercounts already have enough acoustic candidates.

V17 changes one thing only: assignment of event supervision. The six V16
proposals, coverage competition, self-attention, frozen candidate ranking,
outer-clean protocol and fixed 0.5 runtime threshold are preserved. Ordered
per-query event losses are replaced by one exact permutation-invariant set loss.
For each row, all 6! assignments between predicted proposals and the six truth
slots (births plus no-object slots) are scored; the minimum-cost one-to-one
assignment provides the gradient. This removes q-index cardinality semantics and
penalizes duplicate proposals without adding a categorical K head.
"""
from __future__ import annotations

import argparse
import itertools
import json
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
from scripts import train_v160_parallel_coverage_competition as v160

DEFAULT_SEED = 17071
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
SET_PRESENCE_WEIGHT = 1.0
SET_TIME_WEIGHT = 0.30
SET_CANDIDATE_WEIGHT = 0.25
SET_VALID_OFFSET = 1
SET_TIME_OFFSET = 2
SET_CANDIDATE_OFFSET = SET_TIME_OFFSET + TIME_FRAMES
SET_WIDTH = 2 + TIME_FRAMES + MAX_CANDIDATES
PERMUTATIONS = np.asarray(list(itertools.permutations(range(EVENT_QUERIES))), dtype=np.int32)


class V170Error(RuntimeError):
    pass


def _permutation_matrices():
    mats = np.zeros((len(PERMUTATIONS), EVENT_QUERIES, EVENT_QUERIES), dtype=np.float32)
    for r, perm in enumerate(PERMUTATIONS):
        for q, truth in enumerate(perm):
            mats[r, q, truth] = 1.0
    return mats


PERMUTATION_MATRICES = _permutation_matrices()


def _set_loss():
    import tensorflow as tf
    from tensorflow import keras

    perm = tf.constant(PERMUTATION_MATRICES, dtype=tf.float32)

    class ExactPermutationSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="exact_permutation_event_set_loss")

        def call(self, y_true, y_pred):
            yt = tf.cast(y_true, tf.float32)
            yp = tf.cast(y_pred, tf.float32)
            truth_present = yt[:, :, 0]
            truth_valid = yt[:, :, SET_VALID_OFFSET]
            truth_time = yt[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET]
            truth_candidate = yt[:, :, SET_CANDIDATE_OFFSET:]

            pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
            pred_time = tf.clip_by_value(
                yp[:, :, SET_TIME_OFFSET:SET_CANDIDATE_OFFSET], 1e-7, 1.0
            )
            pred_candidate = tf.clip_by_value(
                yp[:, :, SET_CANDIDATE_OFFSET:], 1e-7, 1.0
            )

            y = truth_present[:, None, :]
            p = pred_present[:, :, None]
            presence_cost = -(
                y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p)
            )
            time_cost = -tf.einsum(
                "btd,bqd->bqt", truth_time, tf.math.log(pred_time)
            )
            candidate_cost = -tf.einsum(
                "btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate)
            )
            detail = (truth_present * truth_valid)[:, None, :]
            pair_cost = (
                SET_PRESENCE_WEIGHT * presence_cost
                + SET_TIME_WEIGHT * detail * time_cost
                + SET_CANDIDATE_WEIGHT * detail * candidate_cost
            )
            # pair_cost[b,q,t]. Each permutation matrix selects exactly one
            # truth slot t for every predicted query q and vice versa.
            permutation_cost = tf.einsum("bqt,rqt->br", pair_cost, perm)
            return tf.reduce_min(permutation_cost, axis=1)

    return ExactPermutationSetLoss()


def _build_model():
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
        valid_placeholder = keras.layers.Lambda(
            lambda x: tf.ones_like(x), name=f"v170_event_{q}_valid_placeholder"
        )(present)
        packed = keras.layers.Concatenate(name=f"v170_event_{q}_set_vector")(
            [present, valid_placeholder, time, candidate]
        )
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = present
        outputs[f"event_time_{q}"] = time
        outputs[f"event_candidate_{q}"] = candidate

    event_set = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="event_set"
    )(set_slots)
    outputs["event_set"] = event_set
    outputs["event_count_norm"] = base.get_layer("event_count_norm").output

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["event_set"] = _set_loss()
    # Same differentiable cardinality consequence as V16; this is derived from
    # sum(presence), not a categorical K head.
    loss["event_count_norm"] = "mse"

    loss_weights = {f"string_{slot}": 0.18 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.04 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    loss_weights["event_set"] = 1.0
    loss_weights["event_count_norm"] = 0.35

    model = keras.Model(base.inputs, outputs, name="v170_permutation_set_matching")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=loss_weights,
    )
    return model, loss_weights, token_shape


def _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate):
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = np.asarray(cache["slot_targets"][:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"pitch_{slot}"] = np.asarray(pitch_targets[:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"time_{slot}"] = np.asarray(string_time_targets[:, slot, :], dtype=np.float32)

    present = np.asarray(event_present, dtype=np.float32)
    valid = (np.sum(np.asarray(event_candidate, dtype=np.float32), axis=2) > 0.5).astype(np.float32)
    packed = np.concatenate(
        [
            present[:, :, None],
            valid[:, :, None],
            np.asarray(event_time, dtype=np.float32),
            np.asarray(event_candidate, dtype=np.float32),
        ],
        axis=2,
    )
    if packed.shape[2] != SET_WIDTH:
        raise V170Error(f"unexpected event-set width {packed.shape}")
    out["event_set"] = packed
    out["event_count_norm"] = (
        np.asarray(k, dtype=np.float32) / float(EVENT_QUERIES)
    ).reshape(-1, 1)
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


def _rename_report(report: dict, src: str, dst: str):
    for row in report.get("strata", {}).values():
        if row and src in row:
            row[dst] = row.pop(src)
    for row in report.get("per_true_k", {}).values():
        if row and src in row:
            row[dst] = row.pop(src)


def _postprocess(output_dir: Path, fold: int, report: dict):
    _rename_report(report, "v130", "v170")
    original_arch = dict(report.get("architecture", {}))
    report["protocol"].update({
        "parallel_event_decoder": True,
        "sequential_stop_decoder": False,
        "global_coverage_competition": True,
        "global_query_reconciliation": True,
        "permutation_invariant_set_matching": True,
        "ordered_query_presence_targets": False,
        "set_matching_permutations": int(len(PERMUTATIONS)),
        "presence_threshold_tuned": False,
        "presence_threshold": PRESENCE_THRESHOLD,
        "categorical_cardinality_head_exists": False,
    })
    report["architecture"] = {
        "name": "V17.0 V16 proposals + exact permutation-invariant set matching",
        "event_queries": EVENT_QUERIES,
        "parallel_proposals": True,
        "v16_proposal_graph_unchanged": True,
        "global_self_attention_reconciliation": True,
        "truth_order_has_loss_semantics": False,
        "one_to_one_matching": True,
        "matching_method": "exact min over 6! permutations",
        "matching_cost_weights": {
            "presence": SET_PRESENCE_WEIGHT,
            "time": SET_TIME_WEIGHT,
            "candidate": SET_CANDIDATE_WEIGHT,
        },
        "event_count_norm_weight": 0.35,
        "headline_candidate_realization": "frozen V9+ ranking; isolates cardinality/set supervision",
        "trainable_parameters": int(_build_model()[0].count_params()) if False else original_arch.get("trainable_parameters"),
    }
    rp = output_dir / f"report-fold-{fold}.json"
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    npz_path = output_dir / f"predictions-fold-{fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    data["pred170"] = data.pop("pred130")
    np.savez_compressed(npz_path, **data)

    old_w = output_dir / f"v130-fold-{fold}.weights.h5"
    new_w = output_dir / f"v170-fold-{fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    return report


def train_fold(args):
    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = _build_model
        v130._targets = _targets
        v130._sample_weights = _sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights
    report = _postprocess(args.output_dir, args.outer_fold, report)
    card = report["strata"]["aggregate"]["v170"]["cardinality"]
    print(json.dumps({
        "outer": args.outer_fold,
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v170_f1": report["strata"]["aggregate"]["v170"]["metrics"]["global"]["f1"],
        "poly_exact_v170": card.get("poly_accuracy", card.get("poly_exact_accuracy")),
        "k5_exact_v170": report["per_true_k"]["5"]["v170"]["exact"],
        "k6_exact_v170": report["per_true_k"]["6"]["v170"]["exact"],
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
