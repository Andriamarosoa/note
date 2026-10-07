"""Matched native count-network experiment: normalized versus raw extra evidence.

Each arm starts at the same fold-specific archived uniform checkpoint, extends
the first count CNN with three zero-initialized input channels, and repeats the
historical eight-epoch freeze_local_combo fine-tune on its three FIT folds.
The control repeats normalized channels; the treatment adds raw channels / 12.
No extra audio, notes, YourMT3 predictions, or held-out labels enter training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from scripts.train_v273_group_gate_ab import build_model, weight_hash
from scripts.train_v273_loss_weighting_ab import class_weight_table
from scripts.audit_v273_candidate_hidden1_multifold_worker import (
    SEED, EPOCHS, FREEZE_LOCAL, INTERNAL_FOLDS, WEIGHT_ARM,
    nested_base, compile_model, train_model, predict, fold_vector,
)
from scripts.summarize_v273_reference_hidden1_pairwise_confirmation import discover_outer
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, batch_inputs, require, array_hash
from scripts.yourmt3_exactk_common import metrics, digest, write_json

ARMS = ("normalized_duplicate", "raw")
CONV = "v240_dense_conv1"


def extend_anchor(anchor, arm):
    """Copy every weight; only Conv1 grows from 5 to 8 input channels."""
    require(arm in ARMS, "unknown evidence arm")
    model = build_model("learned_gate", SEED, spectral_evidence=arm)
    source, target = nested_base(anchor), nested_base(model)
    for layer in target.layers:
        if not layer.weights:
            continue
        weights = source.get_layer(layer.name).get_weights()
        if layer.name == CONV:
            kernel, bias = weights
            require(kernel.shape == (3, 3, 5, 32), "anchor CNN layout drift")
            extended = np.zeros((3, 3, 8, 32), dtype=kernel.dtype)
            extended[:, :, :5, :] = kernel
            weights = [extended, bias]
        layer.set_weights(weights)
    for layer in model.layers:
        if layer is not target and layer.weights:
            layer.set_weights(anchor.get_layer(layer.name).get_weights())
    require(model.count_params() == anchor.count_params() + 864, "unexpected added capacity")
    return model


def freeze_and_compile(model):
    base = nested_base(model)
    frozen = {}
    for layer in base.layers:
        layer.trainable = layer.name not in FREEZE_LOCAL
        if layer.name in FREEZE_LOCAL:
            frozen[layer.name] = [w.copy() for w in layer.get_weights()]
    compile_model(model)
    return frozen


def paired(y, before, after):
    out = {}
    for name, mask in [("global", np.ones(len(y), bool)), ("low", y <= 1), ("poly", y >= 2)]:
        out[name] = {
            "corrected": int(np.sum(mask & (before != y) & (after == y))),
            "regressed": int(np.sum(mask & (before == y) & (after != y))),
            "net": int(np.sum(mask & (after == y)) - np.sum(mask & (before == y))),
        }
    return out


def train(a):
    import tensorflow as tf
    require(tf.__version__ == "2.15.1", "pinned TensorFlow required")
    require(a.fold in INTERNAL_FOLDS, "forbidden validation fold")
    require(not a.output.exists(), "refusing to overwrite")
    tf.config.experimental.enable_op_determinism()
    a.output.mkdir(parents=True)
    cfg = load_config(a.config)
    cache, _, manifest = load_bundle(a.bundle, a.config)
    folds = fold_vector(cache, cfg)
    names = np.asarray(cache["members"]).astype(str)
    require(all(m[:2] != "05" for m in names), "player05 present")
    fit = np.flatnonzero(np.isin(folds, [f for f in INTERNAL_FOLDS if f != a.fold]))
    val = np.flatnonzero(folds == a.fold)
    require(not np.intersect1d(fit, val).size and not np.any(folds[fit] == 3), "fold leakage")
    k = np.minimum(cache["exact"].astype(np.int32), 6)
    y = k[val]
    _, archived, uw, fw = discover_outer(a.fold_root, a.fold)
    require(archived["protocol"]["fit_folds"] == [f for f in INTERNAL_FOLDS if f != a.fold],
            "checkpoint training partition drift")
    require(digest(uw) == archived["hashes"]["uniform_weights"], "uniform checkpoint changed")
    require(digest(fw) == archived["hashes"]["freeze_weights"], "reference checkpoint changed")

    reference = build_model("learned_gate", SEED)
    reference.load_weights(fw)
    _, base_pred = predict(reference, cache, val)
    base_metrics = metrics(y, base_pred)
    expected = archived["reference"]["freeze_local_combo"]
    for cls in range(7):
        require(abs(base_metrics["by_k"][str(cls)]["exact"] - expected["by_k"][str(cls)]["exact"]) < 1e-12,
                f"reference K{cls} replay drift")
    del reference
    anchor = build_model("learned_gate", SEED)
    anchor.load_weights(uw)
    anchor_hash = weight_hash(anchor)
    sample = batch_inputs(cache, fit[:128], 31)
    anchor_sample = np.asarray(anchor(sample, training=False))
    table = class_weight_table(WEIGHT_ARM, k[fit])
    protocol = {
        "experiment": "v273_raw_spectral_native_matched_ab", "validation_fold": a.fold,
        "fit_folds": [f for f in INTERNAL_FOLDS if f != a.fold], "fold3_used": False,
        "player05_used": False, "arms": list(ARMS), "epochs": EPOCHS, "seed": SEED,
        "weighting": WEIGHT_ARM, "sample_weight_table": table.tolist(),
        "freeze": list(FREEZE_LOCAL), "same_anchor": True, "same_parameter_count": True,
        "same_epoch_order": True, "reset_training_rng_each_arm": True,
        "window_samples": 4096, "sample_rate": 44100, "additional_audio_samples": 0,
        "target": "K0-K6 only", "note_identification": False,
        "validation_used_for_epoch_or_arm_selection": False, "automatic_promotion": False,
        "validation_scope": "exploratory repeated internal folds; no new independent test",
    }
    report = {"status": "training", "protocol": protocol, "reference": base_metrics,
              "rows": {"fit": len(fit), "validation": len(val)}, "arms": {},
              "sources": {"uniform_sha256": digest(uw), "freeze_sha256": digest(fw),
                          "bundle_manifest_sha256": digest(a.bundle / "bundle.json"),
                          "config_sha256": digest(a.config), "fit_indices_sha256": array_hash(fit),
                          "validation_indices_sha256": array_hash(val), "anchor_hash": anchor_hash}}
    write_json(a.output / "protocol.json", protocol)
    saved = {"global_index": val, "member": names[val], "k": y,
             "baseline": base_pred, "fold": np.full(len(val), a.fold, np.int32)}
    first_hash = None
    first_order = None
    control_pred = None
    for arm in ARMS:
        model = extend_anchor(anchor, arm)
        initial = weight_hash(model)
        if first_hash is None:
            first_hash = initial
        require(initial == first_hash, "matched arms start from different weights")
        prob0 = np.asarray(model(sample, training=False))
        np.testing.assert_allclose(prob0, anchor_sample, atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(prob0.argmax(1), anchor_sample.argmax(1))
        frozen = freeze_and_compile(model)
        # Builders consume random numbers; start both training paths with the
        # same RNG state independently of graph construction/weight loading.
        tf.keras.utils.set_random_seed(SEED)
        history, order = train_model(model, cache, fit, k, table, a.output, arm)
        if first_order is None:
            first_order = order
        require(order == first_order, "training order differs")
        for name, before in frozen.items():
            after = nested_base(model).get_layer(name).get_weights()
            require(all(np.array_equal(x, z) for x, z in zip(before, after)), "frozen layer changed")
        prob, pred = predict(model, cache, val)
        output = a.output / f"{arm}.weights.h5"
        model.save_weights(output)
        kernel = nested_base(model).get_layer(CONV).get_weights()[0]
        rec = {"metrics": metrics(y, pred), "vs_reference": paired(y, base_pred, pred),
               "initial_hash": initial, "parameters": model.count_params(), "history": history,
               "epoch_order": order, "weights_sha256": digest(output),
               "extra_channel_kernel_norm": float(np.linalg.norm(kernel[:, :, 5:, :]))}
        if control_pred is None:
            control_pred = pred
        else:
            rec["vs_normalized_duplicate"] = paired(y, control_pred, pred)
        report["arms"][arm] = rec
        saved[arm + "_predicted"] = pred
        saved[arm + "_probability"] = prob
        write_json(a.output / "report.json", report)
        np.savez_compressed(a.output / "predictions.npz", **saved)
        print(json.dumps({"fold": a.fold, "arm": arm, "metrics": rec["metrics"],
                          "vs_reference": rec["vs_reference"]}), flush=True)
        del model
    report["status"] = "completed"
    write_json(a.output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "fold-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fold", type=int, choices=INTERNAL_FOLDS, required=True)
    train(p.parse_args())
