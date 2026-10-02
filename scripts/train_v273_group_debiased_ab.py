"""Matched A/B: native Exact-K baseline vs removal of biased V8.8 count interface.

Both arms use the frozen 31-frame uniform-input bundle, identical partitions,
seed, epoch order, optimizer, batch size and eight-epoch budget.

Treatment group_debiased zeros only:
  * per-candidate V8.8 cluster_router + local_cardinality[0:4]
  * group stats derived from those outputs: router_mean,
    local_count_mean_norm, local_count_weighted_norm, local_nonzero_max

All spectral inputs, x88 features, isolated_birth, cluster_birth, fused_birth,
candidate timing/geometry and candidate-count/group-width stats are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v90_structured_cluster_cardinality import FROZEN_CANDIDATE_DIM
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, batch_inputs, epoch_order, array_hash, require

ARMS = ("baseline", "group_debiased")
FRAMES = 31
BATCH_SIZE = 128


def weight_hash(model):
    h = hashlib.sha256()
    for a in model.get_weights():
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def build_model(arm: str, seed: int):
    import tensorflow as tf
    if arm not in ARMS:
        raise ValueError(arm)

    base = v260.build_model("uniform", seed, time_frames=FRAMES)
    if arm == "baseline":
        return base

    if FROZEN_CANDIDATE_DIM != V88_FEATURE_DIM + 8:
        raise RuntimeError("unexpected frozen V8.8 feature layout")

    by_name = {x.name.split(":")[0]: x for x in base.inputs}
    candidate_set = by_name["candidate_set"]
    candidate_mask = by_name["candidate_mask"]
    cluster_stats = by_name["cluster_stats"]
    spectral_map = by_name["spectral_map"]

    left = candidate_set[:, :, :V88_FEATURE_DIM]
    zero_count = tf.zeros_like(candidate_set[:, :, V88_FEATURE_DIM:V88_FEATURE_DIM + 5])
    right = candidate_set[:, :, V88_FEATURE_DIM + 5:]
    debiased_candidates = tf.keras.layers.Lambda(
        lambda z: tf.concat(z, axis=-1),
        name="group_debiased_candidate_interface",
    )([left, zero_count, right])

    keep_stats = cluster_stats[:, :4]
    zero_stats = tf.zeros_like(cluster_stats[:, 4:8])
    debiased_stats = tf.keras.layers.Lambda(
        lambda z: tf.concat(z, axis=-1),
        name="group_debiased_stats_interface",
    )([keep_stats, zero_stats])

    out = base({
        "candidate_set": debiased_candidates,
        "candidate_mask": candidate_mask,
        "cluster_stats": debiased_stats,
        "spectral_map": spectral_map,
    })
    model = tf.keras.Model(inputs=base.inputs, outputs=out, name="v273_group_debiased")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="sparse_categorical_crossentropy",
    )
    return model


def batches(cache, indices, *, seed=0, shuffle=False, k=None, batch_size=BATCH_SIZE):
    import tensorflow as tf
    base = np.asarray(indices, np.int64)
    require(len(base) > 0 and len(np.unique(base)) == len(base), "invalid batch population")

    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch = 0
            self.order = epoch_order(base, seed, 0) if shuffle else base.copy()

        def __len__(self):
            return (len(base) + batch_size - 1) // batch_size

        def __getitem__(self, batch):
            ids = self.order[batch * batch_size:(batch + 1) * batch_size]
            x = batch_inputs(cache, ids, FRAMES)
            if k is None:
                return x
            return x, k[ids]

        def on_epoch_end(self):
            self.epoch += 1
            if shuffle:
                self.order = epoch_order(base, seed, self.epoch)

    return Batches()


def metrics(k, pred):
    k = np.asarray(k)
    pred = np.asarray(pred)
    poly = k >= 2
    result = {
        "rows": int(len(k)),
        "exact": float(np.mean(pred == k)),
        "poly_rows": int(poly.sum()),
        "poly_exact": float(np.mean(pred[poly] == k[poly])) if poly.any() else None,
        "under": int(np.sum(pred < k)),
        "over": int(np.sum(pred > k)),
        "confusion": np.bincount(k * 7 + pred, minlength=49).reshape(7, 7).tolist(),
        "by_k": {},
    }
    for value in range(7):
        take = k == value
        result["by_k"][str(value)] = {
            "rows": int(take.sum()),
            "exact": float(np.mean(pred[take] == value)) if take.any() else None,
            "under": int(np.sum(pred[take] < value)),
            "over": int(np.sum(pred[take] > value)),
        }
    return result


def transitions(k, a, b):
    k = np.asarray(k)
    a = np.asarray(a)
    b = np.asarray(b)
    return {
        "corrected": int(np.sum((a != k) & (b == k))),
        "regressed": int(np.sum((a == k) & (b != k))),
        "under_to_correct": int(np.sum((a < k) & (b == k))),
        "over_to_correct": int(np.sum((a > k) & (b == k))),
        "correct_to_under": int(np.sum((a == k) & (b < k))),
        "correct_to_over": int(np.sum((a == k) & (b > k))),
        "net_correct": int(np.sum(b == k) - np.sum(a == k)),
    }


def train(args):
    import tensorflow as tf
    require(tf.__version__ == "2.15.1", "pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(), "refusing to overwrite output")

    cfg = load_config(args.config)
    budget = cfg["folds"][3]["epochs"]["v260_uniform"]
    require(budget == 8, "frozen eight-epoch budget changed")

    cache, parts, manifest = load_bundle(args.bundle, args.config)
    k = np.minimum(cache["exact"].astype(np.int32), 6)
    fit = parts["final_fit"]
    outer = parts["outer"]
    require(not np.intersect1d(fit, outer).size, "outer leakage")
    require(len(outer) == 15279, "outer population changed")

    args.output.mkdir(parents=True)
    seed = v260.SEED + 1003
    order_hash = hashlib.sha256()
    for epoch in range(budget):
        order_hash.update(epoch_order(fit, seed, epoch).tobytes())

    initial = {}
    for arm in ARMS:
        model = build_model(arm, seed)
        initial[arm] = {
            "weight_hash": weight_hash(model),
            "trainable_params": int(np.sum([np.prod(v.shape) for v in model.trainable_variables])),
        }
        del model
        tf.keras.backend.clear_session()

    require(initial["baseline"]["weight_hash"] == initial["group_debiased"]["weight_hash"],
            "arms do not start from identical trainable weights")
    require(initial["baseline"]["trainable_params"] == initial["group_debiased"]["trainable_params"],
            "arm parameter counts differ")

    protocol = {
        "experiment": "v273_group_debiased_ab",
        "outer_fold": 3,
        "frames": FRAMES,
        "weighting": "uniform",
        "epochs": budget,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "epoch_order_sha256": order_hash.hexdigest(),
        "bundle_sha256": digest(args.bundle / "bundle.json"),
        "config_sha256": digest(args.config),
        "arms": list(ARMS),
        "only_treatment": (
            "zero V8.8 cluster_router + local_cardinality in per-candidate features and "
            "their four derived group stats; preserve all remaining candidate/spectral evidence"
        ),
        "identical_initial_trainable_weights": True,
        "identical_trainable_parameter_count": True,
        "class_weighting": "uniform",
        "outer_used_for_selection": False,
        "endpoint": "fixed epoch 8",
        "automatic_promotion": False,
    }
    write_json(args.output / "protocol.json", protocol)

    report = {"status": "training", "protocol": protocol, "initial": initial, "arms": {}}
    predictions = {
        "global_index": np.asarray(outer, np.int64),
        "member": np.asarray(cache["members"][outer]),
        "cluster_start_samples": np.asarray(cache["cluster_start_samples"][outer]),
        "k": k[outer],
    }

    for arm in ARMS:
        tf.keras.backend.clear_session()
        model = build_model(arm, seed)
        require(weight_hash(model) == initial[arm]["weight_hash"], "initialization drift")

        observed = []
        history_rows = []
        train_seq = batches(cache, fit, seed=seed, shuffle=True, k=k)

        class AuditEpoch(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                expected = epoch_order(fit, seed, epoch)
                np.testing.assert_array_equal(train_seq.order, expected)
                observed.append(array_hash(train_seq.order))

            def on_epoch_end(self, epoch, logs=None):
                row = {"epoch": int(epoch) + 1}
                row.update({x: float(y) for x, y in (logs or {}).items()})
                history_rows.append(row)
                write_json(args.output / f"{arm}-progress.json", {
                    "status": "training",
                    "arm": arm,
                    "completed_epoch": int(epoch) + 1,
                    "expected_epochs": budget,
                    "history": history_rows,
                })

        hist = model.fit(
            train_seq,
            epochs=budget,
            shuffle=False,
            workers=0,
            max_queue_size=1,
            verbose=2,
            callbacks=[tf.keras.callbacks.TerminateOnNaN(), AuditEpoch()],
        )
        require(len(history_rows) == budget, "incomplete training")
        require(all(np.isfinite(v).all() for v in hist.history.values()), "nonfinite training")

        probability = np.asarray(
            model.predict(batches(cache, outer), workers=0, max_queue_size=1, verbose=0),
            np.float32,
        )
        require(probability.shape == (len(outer), 7), "wrong prediction shape")
        require(np.isfinite(probability).all() and np.allclose(probability.sum(1), 1.0, atol=1e-5),
                "invalid probabilities")
        pred = probability.argmax(1).astype(np.int32)
        m = metrics(k[outer], pred)

        model.save_weights(args.output / f"{arm}.weights.h5")
        report["arms"][arm] = {
            "metrics": m,
            "history": history_rows,
            "observed_epoch_order_sha256": observed,
            "final_weights_sha256": digest(args.output / f"{arm}.weights.h5"),
        }
        predictions[f"{arm}_probability"] = probability
        predictions[f"{arm}_predicted"] = pred
        write_json(args.output / "report.json", report)
        np.savez_compressed(args.output / "predictions.npz", **predictions)

        del model, train_seq, hist
        tf.keras.backend.clear_session()

    a = predictions["baseline_predicted"]
    b = predictions["group_debiased_predicted"]
    report["status"] = "completed"
    report["paired"] = {
        "all": transitions(k[outer], a, b),
        "poly": transitions(k[outer][k[outer] >= 2], a[k[outer] >= 2], b[k[outer] >= 2]),
    }
    write_json(args.output / "report.json", report)

    lines = [
        "# V27.3 Group-Debiased Exact-K A/B",
        "",
        "Même bundle, seed, ordre des batches, 31 frames, uniform weighting et 8 epochs.",
        "",
        "| Mesure | Baseline | Group-Debiased | Delta |",
        "|---|---:|---:|---:|",
    ]
    bm = report["arms"]["baseline"]["metrics"]
    dm = report["arms"]["group_debiased"]["metrics"]
    lines += [
        f"| exact global | {100*bm['exact']:.3f}% | {100*dm['exact']:.3f}% | {100*(dm['exact']-bm['exact']):+.3f} pt |",
        f"| exact poly | {100*bm['poly_exact']:.3f}% | {100*dm['poly_exact']:.3f}% | {100*(dm['poly_exact']-bm['poly_exact']):+.3f} pt |",
        f"| under | {bm['under']} | {dm['under']} | {dm['under']-bm['under']:+d} |",
        f"| over | {bm['over']} | {dm['over']} | {dm['over']-bm['over']:+d} |",
    ]
    for value in (2, 3, 4):
        x = bm["by_k"][str(value)]["exact"]
        y = dm["by_k"][str(value)]["exact"]
        lines.append(f"| K{value} exact | {100*x:.3f}% | {100*y:.3f}% | {100*(y-x):+.3f} pt |")
    p = report["paired"]["all"]
    lines += [
        "",
        f"Paired: corrigés **{p['corrected']}**, régressions **{p['regressed']}**, net **{p['net_correct']:+d}**.",
        "",
        "Aucune promotion automatique.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    train(p.parse_args())
