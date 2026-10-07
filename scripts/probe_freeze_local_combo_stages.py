"""Cross-fold ablation of freeze_local_combo into encoder/pre-decoder/full stages.

Definitions for this audit:
- encoder representation: v240_cardinality_context
- pre-decoder representation: v240_cardinality_hidden2 (96-D)
- full decoder: original 7-way cardinality softmax

For each held-out fold in {0,1,2,4}, the exact fold-specific
freeze_local_combo checkpoint was trained on the other three folds. We extract
its representations on those fit folds and on the held-out fold, then train the
same fixed linear probe used in the YourMT3+ representation audit. No fold-3
or player05 data is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_v273_candidate_hidden1_multifold_worker import (
    INTERNAL_FOLDS,
    SEED,
    fold_vector,
    nested_base,
)
from scripts.summarize_v273_reference_hidden1_pairwise_confirmation import discover_outer
from scripts.train_v273_group_gate_ab import build_model
from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import batch_inputs, load_bundle, require
from scripts.yourmt3_exactk_common import metrics, paired, write_json

FRAMES = 31
PROBE_SEED = 17064


def fit_probe(train_x, train_y, test_x):
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_x = scaler.fit_transform(np.asarray(train_x, np.float32))
    test_x = scaler.transform(np.asarray(test_x, np.float32))
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-4,
        shuffle=True,
        random_state=PROBE_SEED,
        average=True,
        n_jobs=1,
    )
    clf.fit(train_x, train_y)
    return clf.predict(test_x).astype(np.int32)


def gated_base_inputs(model, x):
    """Replay the learned V27.3 trust gate before entering the nested count model."""
    import tensorflow as tf

    gate_model = tf.keras.Model(
        model.inputs,
        model.get_layer("v273_gate_value").output,
        name="freeze_gate_probe",
    )
    gate = np.asarray(gate_model.predict(x, batch_size=256, verbose=0), np.float32)
    require(gate.shape == (len(x["cluster_stats"]), 1), f"gate shape drift: {gate.shape}")

    cand = np.asarray(x["candidate_set"], np.float32).copy()
    stats = np.asarray(x["cluster_stats"], np.float32).copy()
    require(cand.shape[-1] >= V88_FEATURE_DIM + 5, "candidate pseudo-count layout drift")
    require(stats.shape[-1] >= 8, "stats pseudo-count layout drift")

    cand[:, :, V88_FEATURE_DIM:V88_FEATURE_DIM + 5] *= gate[:, None, :]
    stats[:, 4:8] *= gate

    return {
        "candidate_set": cand,
        "candidate_mask": np.asarray(x["candidate_mask"], np.float32),
        "cluster_stats": stats,
        "spectral_map": np.asarray(x["spectral_map"], np.float32),
    }


def stage_outputs(model, cache, ids):
    """Return encoder context, pre-decoder hidden2, and original probabilities."""
    import tensorflow as tf

    ids = np.asarray(ids, np.int64)
    x = batch_inputs(cache, ids, FRAMES)
    base_x = gated_base_inputs(model, x)
    base = nested_base(model)

    stage_model = tf.keras.Model(
        base.inputs,
        [
            base.get_layer("v240_cardinality_context").output,
            base.get_layer("v240_cardinality_hidden2").output,
            base.get_layer("cardinality").output,
        ],
        name="freeze_stage_probe",
    )
    encoder, predecoder, probability = stage_model.predict(
        base_x, batch_size=256, verbose=0
    )
    encoder = np.asarray(encoder, np.float32)
    predecoder = np.asarray(predecoder, np.float32)
    probability = np.asarray(probability, np.float32)

    require(encoder.ndim == 2 and len(encoder) == len(ids), "encoder output drift")
    require(
        predecoder.shape == (len(ids), 96),
        f"pre-decoder output drift: {predecoder.shape}",
    )
    require(
        probability.shape == (len(ids), 7)
        and np.all(np.isfinite(probability))
        and np.allclose(probability.sum(1), 1.0, atol=1e-5),
        "cardinality probability drift",
    )
    require(np.all(np.isfinite(encoder)) and np.all(np.isfinite(predecoder)), "non-finite representation")
    return encoder, predecoder, probability


def main(a):
    import tensorflow as tf

    require(tf.__version__ == "2.15.1", "pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(a.val_fold in INTERNAL_FOLDS, "invalid held-out fold")
    require(not a.output.exists(), "refusing overwrite")
    a.output.mkdir(parents=True)

    cfg = load_config(a.config)
    cache, _, _ = load_bundle(a.bundle, a.config)
    folds = fold_vector(cache, cfg)
    names = np.asarray(cache["members"]).astype(str)
    require("05" not in {x[:2] for x in names}, "player05 present")

    fit_folds = [f for f in INTERNAL_FOLDS if f != a.val_fold]
    fit = np.flatnonzero(np.isin(folds, fit_folds))
    val = np.flatnonzero(folds == a.val_fold)
    outer3 = np.flatnonzero(folds == 3)
    require(
        len(fit) and len(val) and len(outer3)
        and not np.intersect1d(fit, val).size
        and not np.intersect1d(fit, outer3).size
        and not np.intersect1d(val, outer3).size,
        "partition leakage",
    )
    k = np.minimum(np.asarray(cache["exact"], np.int32), 6)

    _, saved, _, freeze_weights = discover_outer(a.fold_root, a.val_fold)
    require(
        saved["protocol"]["fit_folds"] == fit_folds,
        "checkpoint fit-fold identity drift",
    )

    model = build_model("learned_gate", SEED)
    model.load_weights(freeze_weights)

    enc_fit, pre_fit, _ = stage_outputs(model, cache, fit)
    enc_val, pre_val, prob_val = stage_outputs(model, cache, val)
    full_pred = prob_val.argmax(1).astype(np.int32)
    y_fit, y_val = k[fit], k[val]

    reference = metrics(y_val, full_pred)
    expected = saved["reference"]["freeze_local_combo"]
    require(
        abs(reference["exact"] - expected["exact"]) < 1e-12,
        "freeze_local_combo replay drift",
    )
    for cls in range(7):
        require(
            abs(reference["by_k"][str(cls)]["exact"] - expected["by_k"][str(cls)]["exact"]) < 1e-12,
            f"freeze K{cls} replay drift",
        )

    enc_pred = fit_probe(enc_fit, y_fit, enc_val)
    pre_pred = fit_probe(pre_fit, y_fit, pre_val)

    report = {
        "status": "completed",
        "protocol": {
            "experiment": "freeze_local_combo_stage_ablation",
            "validation_fold": a.val_fold,
            "fit_folds": fit_folds,
            "fold3_used": False,
            "player05_used": False,
            "checkpoint_training": "exact archived fold-specific freeze_local_combo",
            "encoder_cut": "v240_cardinality_context",
            "predecoder_cut": "v240_cardinality_hidden2",
            "full_decoder": "original Dense(7, softmax) cardinality head",
            "probe": "StandardScaler + fixed SGDClassifier(log_loss, balanced, alpha=1e-4, average=True)",
            "probe_seed": PROBE_SEED,
            "held_out_tuning": False,
            "automatic_promotion": False,
        },
        "rows": {"fit": int(len(fit)), "validation": int(len(val)), "fold3_excluded": int(len(outer3))},
        "dimensions": {
            "encoder": int(enc_val.shape[1]),
            "predecoder": int(pre_val.shape[1]),
            "decoder_classes": 7,
        },
        "freeze_local_combo": reference,
        "encoder_linear_probe": {
            "metrics": metrics(y_val, enc_pred),
            "paired_vs_full": paired(y_val, full_pred, enc_pred),
        },
        "predecoder_linear_probe": {
            "metrics": metrics(y_val, pre_pred),
            "paired_vs_full": paired(y_val, full_pred, pre_pred),
        },
    }
    write_json(a.output / "report.json", report)
    np.savez_compressed(
        a.output / "predictions.npz",
        global_index=val,
        k=y_val,
        freeze_local_combo=full_pred,
        encoder_linear_probe=enc_pred,
        predecoder_linear_probe=pre_pred,
    )

    lines = [
        f"# freeze_local_combo stage ablation — held-out fold {a.val_fold}",
        "",
        f"Fit folds: {fit_folds}; fold 3 and player05 excluded.",
        "",
        "| Stage | Global Exact-K | Poly K2-K6 | Poly under | Poly over |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, rec in [
        ("encoder linear probe", report["encoder_linear_probe"]["metrics"]),
        ("pre-decoder linear probe", report["predecoder_linear_probe"]["metrics"]),
        ("freeze_local_combo full", reference),
    ]:
        p = rec["poly"]
        lines.append(
            f"| {label} | {100*rec['exact']:.4f}% | {100*p['exact']:.4f}% | {p['under']} | {p['over']} |"
        )
    lines += [
        "",
        f"Encoder dimension: {enc_val.shape[1]}. Pre-decoder dimension: {pre_val.shape[1]}.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "fold-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--val-fold", type=int, choices=INTERNAL_FOLDS, required=True)
    main(p.parse_args())
