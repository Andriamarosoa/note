"""V23.2: soft transport supervision with unchanged exact-K runtime transport.

V23.1 proved that categorical K can drive an exact-K runtime plan, but its
coverage loss was numerically undefined whenever hard K=0 because Keras
categorical crossentropy renormalized an all-zero prediction. V23.2 changes
only the training path: coverage supervision uses the same transport scores
with soft categorical survival mass (plus a tiny fixed numerical floor), while
runtime event activation and runtime transport remain hard exact-K.

Locked12 is never indexed or evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v190_dense_birth_centers as v190
from scripts import train_v231_cardinality_transport as v231

ROOT = v231.ROOT
DEFAULT_SEED = v231.DEFAULT_SEED
BASE_ARM = v231.BASE_ARM
EVENT_QUERIES = v231.EVENT_QUERIES
CARDINALITY_CLASSES = v231.CARDINALITY_CLASSES
CENTER_CELLS = v231.CENTER_CELLS
MODEL_KEY = "v232_soft_train_hard_runtime_transport"
PRED_KEY = "pred232_soft_train_hard_runtime_transport"
SOFT_MASS_FLOOR = 1e-4


class V232Error(RuntimeError):
    pass


def _runtime_coverage_from_plan(plan):
    z = np.asarray(plan, dtype=np.float32)
    raw = np.sum(z, axis=1)
    denom = np.sum(raw, axis=1, keepdims=True)
    out = np.zeros_like(raw)
    np.divide(raw, denom, out=out, where=denom > 0.0)
    return out


def _build_model(spec):
    import tensorflow as tf
    from tensorflow import keras

    base, _, token_shape = v231._build_model(spec)
    outputs = dict(base.output)

    # Runtime plan stays exactly the V23.1 hard-forward exact-K plan.
    # Training coverage is a second plan over the same learned scores, using
    # categorical survival probabilities so it is never the all-zero vector.
    scores = base.get_layer("v231_transport_scores").output
    soft_active = base.get_layer("v231_soft_survival").output
    soft_mass = keras.layers.Lambda(
        lambda a: tf.maximum(tf.cast(a, tf.float32), tf.constant(SOFT_MASS_FLOOR, tf.float32)),
        name="v232_soft_training_row_mass",
    )(soft_active)
    soft_plan = keras.layers.Lambda(
        lambda z: v231._capped_transport_tf(tf, z[0], z[1]),
        name="v232_soft_training_transport_plan",
    )([scores, soft_mass])
    soft_raw = keras.layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=1), name="v232_soft_training_coverage_raw"
    )(soft_plan)
    soft_coverage = keras.layers.Lambda(
        lambda x: x / (tf.reduce_sum(x, axis=1, keepdims=True) + v231.EPS),
        name="v232_soft_training_coverage",
    )(soft_raw)
    outputs["transport_coverage"] = soft_coverage

    loss = {f"string_{s}": "binary_crossentropy" for s in range(v231.SLOT_COUNT)}
    loss.update({f"pitch_{s}": "mse" for s in range(v231.SLOT_COUNT)})
    loss.update({f"time_{s}": keras.losses.KLDivergence() for s in range(v231.SLOT_COUNT)})
    loss["event_set"] = v231._set_loss()
    loss["event_count_norm"] = "mse"
    loss["birth_center_map"] = "categorical_crossentropy"
    loss["transport_coverage"] = "categorical_crossentropy"
    loss["cardinality"] = "categorical_crossentropy"

    lw = {f"string_{s}": 0.18 for s in range(v231.SLOT_COUNT)}
    lw.update({f"pitch_{s}": 0.04 for s in range(v231.SLOT_COUNT)})
    lw.update({f"time_{s}": 0.10 for s in range(v231.SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.0
    lw["birth_center_map"] = v231.CENTER_MAP_WEIGHT
    lw["transport_coverage"] = v231.TRANSPORT_COVERAGE_WEIGHT
    lw["cardinality"] = v231.CARDINALITY_WEIGHT

    model = keras.Model(base.inputs, outputs, name="v232_soft_train_hard_runtime_transport")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=lw)
    return model, lw, token_shape


def _transport_mass_diag(row_mass, card_prob):
    mass = np.asarray(row_mass, dtype=np.float64)
    pred_k = np.argmax(np.asarray(card_prob), axis=1).astype(np.int32)
    bits = (mass >= 0.5).astype(np.int8)
    decoded = np.sum(bits, axis=1).astype(np.int32)
    # Only 0->1 is illegal. The normal exact-K prefix transition 1->0 is valid.
    violations = np.any(np.diff(bits, axis=1) > 0, axis=1)
    return {
        "rows": int(len(mass)),
        "exact_k_mass_decode_rate": float(np.mean(decoded == pred_k)),
        "max_absolute_total_mass_error": float(np.max(np.abs(np.sum(mass, axis=1) - pred_k))),
        "prefix_violation_rate": float(np.mean(violations)),
        "mean_total_mass": float(np.mean(np.sum(mass, axis=1))),
        "mean_predicted_k": float(np.mean(pred_k)),
    }


def _postprocess(args, report, ctx, raw, center_diag, soft_coverage_diag, runtime_coverage_diag):
    report = v231._postprocess(args, report, ctx, raw, center_diag, soft_coverage_diag)
    v171._rename_report(report, v231.MODEL_KEY, MODEL_KEY)

    block = report.pop("v231")
    block["model_key"] = MODEL_KEY
    arch = block["architecture"]
    arch["transport_mass"] = _transport_mass_diag(raw["transport_row_mass"], raw["cardinality"])
    arch["transport_training_coverage_diagnostics"] = arch.pop("transport_coverage_diagnostics")
    arch["transport_runtime_coverage_diagnostics"] = runtime_coverage_diag
    arch["soft_training_row_mass_floor"] = SOFT_MASS_FLOOR
    report["v232"] = block

    p = report["protocol"]
    p["v231_cardinality_conditioned_transport"] = False
    p["v232_soft_train_hard_runtime_transport"] = True
    p["v232_only_scientific_treatment"] = (
        "coverage loss uses soft categorical-survival transport; runtime categorical K, "
        "binary q<K activation, and exact-K hard-forward transport are unchanged"
    )
    p["training_transport_uses_soft_survival"] = True
    p["training_transport_min_row_mass_floor"] = SOFT_MASS_FLOOR
    p["training_transport_min_row_mass_floor_tuned"] = False
    p["runtime_transport_remains_exact_k"] = True
    p["historical_validation_or_locked12_indexed_or_evaluated"] = False

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if v231.PRED_KEY not in data:
        raise V232Error(f"missing inherited prediction key {v231.PRED_KEY}")
    data[PRED_KEY] = data.pop(v231.PRED_KEY)
    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v231-cardinality-transport-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v232-soft-train-hard-runtime-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)

    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def train_fold(args):
    import tensorflow as tf

    if args.seed != DEFAULT_SEED:
        raise V232Error(f"V23.2 requires seed {DEFAULT_SEED}")
    if args.arm != BASE_ARM:
        raise V232Error(f"V23.2 only supports {BASE_ARM}")

    ctx = v172._fold_context(args)
    specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}
    built = []

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V232Error("unexpected model build")
        calls["count"] += 1
        model = _build_model(specs[i])
        built.append(model[0])
        return model

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = v231._targets
        v130._sample_weights = v231._sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights

    if calls["count"] != 2 or v231._LAST_CENTER_TARGETS is None:
        raise V232Error("V23.2 build/target capture failed")

    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    inputs = v102._inputs(ctx["cache"], outer)
    raw = built[-1].predict(inputs, batch_size=128, verbose=0)
    runtime_plan_probe = tf.keras.Model(
        built[-1].inputs, built[-1].get_layer("v231_transport_plan").output
    )
    runtime_plan = runtime_plan_probe.predict(inputs, batch_size=128, verbose=0)
    raw["transport_runtime_coverage"] = _runtime_coverage_from_plan(runtime_plan)

    target = np.asarray(v231._LAST_CENTER_TARGETS)[outer]
    eligible = np.asarray(v231._LAST_CENTER_ELIGIBLE)[outer]
    kk = np.asarray(ctx["k"], dtype=np.int32)[outer]
    center_diag = v190._center_diagnostics(np.asarray(raw["birth_center_map"]), target, eligible, kk)
    soft_diag = v190._center_diagnostics(np.asarray(raw["transport_coverage"]), target, eligible, kk)
    runtime_diag = v190._center_diagnostics(np.asarray(raw["transport_runtime_coverage"]), target, eligible, kk)
    report = _postprocess(args, report, ctx, raw, center_diag, soft_diag, runtime_diag)

    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    a = report["v232"]["architecture"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v232_f1": g["f1"],
        "pred_ref": g["prediction_reference_ratio"],
        "categorical_k_exact": a["categorical_cardinality"]["exact"],
        "categorical_k_poly_exact": a["categorical_cardinality"]["poly_exact"],
        "exact_k_transport_mass": a["transport_mass"]["exact_k_mass_decode_rate"],
        "soft_coverage_top6_exact_poly": a["transport_training_coverage_diagnostics"]["top6_exact_center_coverage_poly"],
        "runtime_coverage_top6_exact_poly": a["transport_runtime_coverage_diagnostics"]["top6_exact_center_coverage_poly"],
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
