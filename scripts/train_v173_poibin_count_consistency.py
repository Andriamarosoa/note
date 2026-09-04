"""V17.3 exact Poisson-binomial cardinality consistency.

Starts from V17.2 mass_permutation and changes one training objective only:
- keep the V16 proposal graph unchanged;
- keep fit-fold-only mass-preserving exchangeable presence weights unchanged;
- keep exact 6! permutation matching unchanged;
- keep runtime K=sum(present>=0.5) unchanged;
- replace the weak event_count_norm MSE contribution with an exact
  Poisson-binomial negative log-likelihood for the true number of active
  exchangeable queries.

There is no categorical K head and no threshold tuning. The Poisson-binomial
term is assignment invariant and adds no runtime layer or latency.
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
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
BASE_ARM = "mass_permutation"
MODEL_KEY = "v173_poibin"
COUNT_NLL_WEIGHT = 0.35


class V173Error(RuntimeError):
    pass


def _poibin_distribution_np(p: np.ndarray) -> np.ndarray:
    """Exact Poisson-binomial P(N=k), k=0..6, for independent query Bernoullis."""
    x = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != EVENT_QUERIES:
        raise V173Error(f"expected [B,{EVENT_QUERIES}] probabilities, got {x.shape}")
    dist = np.zeros((len(x), EVENT_QUERIES + 1), dtype=np.float64)
    dist[:, 0] = 1.0
    for q in range(EVENT_QUERIES):
        pq = x[:, q : q + 1]
        shifted = np.concatenate([np.zeros((len(x), 1), dtype=np.float64), dist[:, :-1]], axis=1)
        dist = dist * (1.0 - pq) + shifted * pq
    return dist


def _poibin_distribution_tf(tf, p):
    p = tf.clip_by_value(tf.cast(p, tf.float32), 1e-6, 1.0 - 1e-6)
    batch = tf.shape(p)[0]
    dist = tf.concat(
        [tf.ones((batch, 1), dtype=tf.float32), tf.zeros((batch, EVENT_QUERIES), dtype=tf.float32)],
        axis=1,
    )
    for q in range(EVENT_QUERIES):
        pq = p[:, q : q + 1]
        shifted = tf.concat([tf.zeros((batch, 1), dtype=tf.float32), dist[:, :-1]], axis=1)
        dist = dist * (1.0 - pq) + shifted * pq
    return dist


def _set_loss(spec: dict):
    import tensorflow as tf
    from tensorflow import keras

    perm = tf.constant(v172.PERMUTATION_MATRICES, dtype=tf.float32)

    class PoissonBinomialSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="v173_poibin_event_set_loss")

        def call(self, y_true, y_pred):
            yt = tf.cast(y_true, tf.float32)
            yp = tf.cast(y_pred, tf.float32)
            truth_present = yt[:, :, 0]
            truth_valid = yt[:, :, v171.SET_VALID_OFFSET]
            truth_time = yt[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET]
            truth_candidate = yt[:, :, v171.SET_CANDIDATE_OFFSET :]

            pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
            pred_time = tf.clip_by_value(
                yp[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET], 1e-7, 1.0
            )
            pred_candidate = tf.clip_by_value(yp[:, :, v171.SET_CANDIDATE_OFFSET :], 1e-7, 1.0)

            # V17.2-C matching term, unchanged.
            y = truth_present[:, None, :]
            p = pred_present[:, :, None]
            cw = v172._class_weight_tensor(tf, BASE_ARM, truth_present, spec)
            presence_cost = -cw * (
                y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p)
            )
            time_cost = -tf.einsum("btd,bqd->bqt", truth_time, tf.math.log(pred_time))
            candidate_cost = -tf.einsum(
                "btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate)
            )
            detail = (truth_present * truth_valid)[:, None, :]
            pair = (
                v172.SET_PRESENCE_WEIGHT * presence_cost
                + v172.SET_TIME_WEIGHT * detail * time_cost
                + v172.SET_CANDIDATE_WEIGHT * detail * candidate_cost
            )
            scores = tf.einsum("bqt,rqt->br", pair, perm)
            matching_loss = tf.reduce_min(scores, axis=1)

            # Exact assignment-invariant cardinality likelihood.
            count_dist = _poibin_distribution_tf(tf, pred_present)
            true_k = tf.cast(tf.reduce_sum(truth_present, axis=1), tf.int32)
            batch = tf.range(tf.shape(true_k)[0], dtype=tf.int32)
            true_prob = tf.gather_nd(count_dist, tf.stack([batch, true_k], axis=1))
            count_nll = -tf.math.log(tf.clip_by_value(true_prob, 1e-7, 1.0))
            return matching_loss + tf.constant(COUNT_NLL_WEIGHT, tf.float32) * count_nll

    return PoissonBinomialSetLoss()


def _build_model(spec: dict):
    """Reuse the V17.2-C graph exactly, then recompile with the V17.3 objective."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    model, old_lw, token_shape = v172._build_model(BASE_ARM, spec)

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["event_set"] = _set_loss(spec)
    # Keep this existing graph output for graph identity / diagnostics, but its
    # weak MSE supervision is disabled to isolate the exact count objective.
    loss["event_count_norm"] = "mse"

    lw = dict(old_lw)
    lw["event_count_norm"] = 0.0
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=lw,
    )
    return model, lw, token_shape


def _postprocess(args, report, ctx):
    # First let the proven V17.2 postprocessor calculate exactly the same outer
    # diagnostics, occupancy and protocol invariants for mass_permutation.
    report = v172._postprocess(args, report, ctx)
    old_key = "v172_mass_permutation"
    v171._rename_report(report, old_key, MODEL_KEY)

    report["protocol"].update(
        {
            "v173_poibin_count_consistency": True,
            "v173_base_arm": BASE_ARM,
            "v173_only_training_objective_change_from_v172_c": (
                "event_count_norm MSE contribution -> exact Poisson-binomial cardinality NLL"
            ),
            "runtime_graph_unchanged_from_v172_c": True,
            "runtime_decode_unchanged_from_v172_c": True,
            "runtime_presence_threshold": PRESENCE_THRESHOLD,
            "runtime_presence_threshold_tuned": False,
            "categorical_cardinality_head_exists": False,
            "poisson_binomial_cardinality_consistency": True,
            "poisson_binomial_cardinality_nll_weight": COUNT_NLL_WEIGHT,
            "event_count_norm_mse_loss_weight": 0.0,
            "count_objective_assignment_invariant": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
        }
    )

    inherited = report.pop("v172")
    report["v173"] = {
        **inherited,
        "base_arm": BASE_ARM,
        "model_key": MODEL_KEY,
        "count_objective": {
            "type": "exact_poisson_binomial_negative_log_likelihood",
            "weight": COUNT_NLL_WEIGHT,
            "old_event_count_norm_mse_weight": 0.0,
            "runtime_decode_changed": False,
            "runtime_graph_changed": False,
        },
    }

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    old_pred = "pred172_mass_permutation"
    if old_pred not in data:
        raise V173Error(f"missing postprocessed prediction key {old_pred}")
    data["pred173_poibin"] = data.pop(old_pred)
    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v172-mass_permutation-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)

    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def train_fold(args):
    if args.seed != DEFAULT_SEED:
        raise V173Error(f"V17.3 requires seed {DEFAULT_SEED}, got {args.seed}")
    if args.arm != BASE_ARM:
        raise V173Error(f"V17.3 only supports base arm {BASE_ARM!r}")

    ctx = v172._fold_context(args)
    stage_specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V173Error(f"unexpected model build call {i + 1}")
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
        raise V173Error(f"expected exactly 2 model builds, got {calls['count']}")

    report = _postprocess(args, report, ctx)
    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    card = report["strata"]["aggregate"][MODEL_KEY]["cardinality"]
    print(
        json.dumps(
            {
                "outer": args.outer_fold,
                "selected_epochs": report["data"]["selected_epochs"],
                "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
                "v173_f1": g["f1"],
                "pred_ref": g["prediction_reference_ratio"],
                "poly_exact": card.get(
                    "poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))
                ),
                "k2": report["per_true_k"]["2"][MODEL_KEY]["exact"],
                "k3": report["per_true_k"]["3"][MODEL_KEY]["exact"],
                "k5": report["per_true_k"]["5"][MODEL_KEY]["exact"],
                "k6": report["per_true_k"]["6"][MODEL_KEY]["exact"],
            },
            indent=2,
            sort_keys=True,
        )
    )
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
