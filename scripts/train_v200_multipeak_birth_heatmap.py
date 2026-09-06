"""V20 independent multi-peak birth heatmap.

Post-V19 audit proved that the 23x64 representation is geometrically capable
of separating 99.38% of eligible poly birth sets, and V19 improved F1 versus
V17.3, but its learned centers were not the source of that gain. V19 used one
1472-way softmax with a target normalized across K true centers: all births had
to share one unit of probability mass, while discrete top-k coordinates were
not differentiable from the downstream event-set loss.

V20 changes only the dense-center formulation:
- keep the entire V19 dense conv/proposal/set graph;
- keep the same top-6 3x3 local-max proposal extraction;
- replace categorical softmax center output by 1472 independent sigmoids;
- replace row-normalized center targets by independent multi-hot centers;
- use a standard focal binary heatmap loss, normalized per positive center;
- retain the V19 center auxiliary coefficient 0.25 unchanged/untuned.

Canonical V17.3 set objective remains unchanged: six Bernoulli event decisions,
mass-preserving exchangeable weights, exact 720 permutation matching,
Poisson-binomial count NLL=.35, runtime K=sum(p>=.5), threshold .5 untuned.
Locked12/historical validation remain untouched.
"""
from __future__ import annotations

import argparse
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
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v173_poibin_count_consistency as v173
from scripts import train_v190_dense_birth_centers as v190

DEFAULT_SEED = v190.DEFAULT_SEED
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = v190.PRESENCE_THRESHOLD
BASE_ARM = v190.BASE_ARM
MODEL_KEY = "v200_multipeak_birth_heatmap"
PRED_KEY = "pred200_multipeak_birth_heatmap"
CENTER_MAP_WEIGHT = v190.CENTER_MAP_WEIGHT
CENTER_CELLS = v190.CENTER_CELLS
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0
_LAST_CENTER_TARGETS = None
_LAST_CENTER_ELIGIBLE = None


class V200Error(RuntimeError):
    pass


def _multipeak_targets(cache, pitch_targets, string_time_targets, k):
    """Same exact V19 center coordinates, but independent multi-hot occupancy."""
    normalized, eligible, distinct = v190._birth_center_targets(
        cache, pitch_targets, string_time_targets, k
    )
    return (np.asarray(normalized) > 0.0).astype(np.float32), eligible, distinct


def _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate):
    global _LAST_CENTER_TARGETS, _LAST_CENTER_ELIGIBLE
    out = v171._targets(
        cache, pitch_targets, string_time_targets, k,
        event_present, event_time, event_candidate,
    )
    center, eligible, _ = _multipeak_targets(cache, pitch_targets, string_time_targets, k)
    out["birth_center_heatmap"] = center
    _LAST_CENTER_TARGETS = center
    _LAST_CENTER_ELIGIBLE = eligible
    return out


def _sample_weights(cache, time_mask, k, event_present, event_valid):
    out = v171._sample_weights(cache, time_mask, k, event_present, event_valid)
    # Preserve V19 eligibility/sample-weight semantics exactly.
    kk = np.asarray(k, dtype=np.int32)
    timed = np.sum(np.asarray(time_mask, dtype=np.float32) > 0.5, axis=1).astype(np.int32)
    out["birth_center_heatmap"] = ((kk > 0) & (timed == kk)).astype(np.float32)
    return out


def _focal_heatmap_loss():
    import tensorflow as tf
    from tensorflow import keras

    class MultiPeakFocalLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="v200_multipeak_focal_heatmap_loss")

        def call(self, y_true, y_pred):
            y = tf.cast(y_true, tf.float32)
            p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
            pos = tf.cast(y > 0.5, tf.float32)
            neg = 1.0 - pos
            pos_loss = -tf.pow(1.0 - p, FOCAL_ALPHA) * tf.math.log(p) * pos
            neg_loss = -tf.pow(p, FOCAL_ALPHA) * tf.math.log(1.0 - p) * neg
            positives = tf.reduce_sum(pos, axis=1)
            denom = tf.maximum(positives, 1.0)
            return (tf.reduce_sum(pos_loss, axis=1) + tf.reduce_sum(neg_loss, axis=1)) / denom

    return MultiPeakFocalLoss()


def _build_model(spec):
    """Reuse every V19 parameterized layer; change only center output/objective."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    base, _, token_shape = v190._build_model(spec)
    # Reconstruct the same dict of V19 outputs except the categorical softmax.
    outputs = {name: tensor for name, tensor in zip(base.output_names, base.outputs)}
    outputs.pop("birth_center_map", None)
    logits = base.get_layer("v190_birth_center_logits_flat").output
    outputs["birth_center_heatmap"] = keras.layers.Activation(
        "sigmoid", name="birth_center_heatmap"
    )(logits)

    loss = {f"string_{s}": "binary_crossentropy" for s in range(SLOT_COUNT)}
    loss.update({f"pitch_{s}": "mse" for s in range(SLOT_COUNT)})
    loss.update({f"time_{s}": keras.losses.KLDivergence() for s in range(SLOT_COUNT)})
    loss["event_set"] = v173._set_loss(spec)
    loss["event_count_norm"] = "mse"
    loss["birth_center_heatmap"] = _focal_heatmap_loss()

    lw = {f"string_{s}": 0.18 for s in range(SLOT_COUNT)}
    lw.update({f"pitch_{s}": 0.04 for s in range(SLOT_COUNT)})
    lw.update({f"time_{s}": 0.10 for s in range(SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.0
    lw["birth_center_heatmap"] = CENTER_MAP_WEIGHT

    model = keras.Model(base.inputs, outputs, name="v200_multipeak_birth_heatmap_decoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=lw,
    )
    return model, lw, token_shape


def _postprocess(args, report, ctx, center_diag):
    # Reuse V19 postprocessing first so all canonical outer diagnostics remain identical.
    report = v190._postprocess(args, report, ctx, center_diag)
    v171._rename_report(report, v190.MODEL_KEY, MODEL_KEY)
    inherited = report.pop("v190")
    inherited["model_key"] = MODEL_KEY
    inherited["architecture"]["dense_center_diagnostics"] = center_diag
    report["v200"] = inherited

    report["protocol"].update({
        "v190_dense_birth_centers": False,
        "v200_multipeak_birth_heatmap": True,
        "v200_base_version": "V19 dense conv/proposal/set graph",
        "v200_only_scientific_treatment": (
            "1472-way categorical softmax + row-normalized center target -> "
            "independent sigmoid multi-hot centers + focal binary heatmap loss"
        ),
        "dense_center_loss": "independent_binary_focal",
        "dense_center_loss_weight": CENTER_MAP_WEIGHT,
        "dense_center_loss_weight_tuned": False,
        "dense_center_target_normalized_across_births": False,
        "dense_center_independent_cell_probabilities": True,
        "dense_center_focal_alpha": FOCAL_ALPHA,
        "dense_center_focal_beta": FOCAL_BETA,
        "dense_center_anchor_selection_unchanged_from_v190": True,
        "dense_conv_parameterization_unchanged_from_v190": True,
        "v173_poisson_binomial_count_objective_unchanged": True,
        "v173_count_nll_weight": v173.COUNT_NLL_WEIGHT,
        "mass_preserving_exchangeable_weights_unchanged": True,
        "exact_720_truth_matching_unchanged": True,
        "runtime_presence_threshold": PRESENCE_THRESHOLD,
        "runtime_presence_threshold_tuned": False,
        "historical_validation_or_locked12_indexed_or_evaluated": False,
    })

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if v190.PRED_KEY not in data:
        raise V200Error(f"missing inherited {v190.PRED_KEY}")
    data[PRED_KEY] = data.pop(v190.PRED_KEY)
    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v190-dense-birth-centers-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v200-multipeak-birth-heatmap-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def train_fold(args):
    global _LAST_CENTER_TARGETS, _LAST_CENTER_ELIGIBLE
    if args.seed != DEFAULT_SEED:
        raise V200Error(f"V20 requires seed {DEFAULT_SEED}")
    if args.arm != BASE_ARM:
        raise V200Error(f"V20 only supports {BASE_ARM}")

    ctx = v172._fold_context(args)
    specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}
    built = []

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V200Error("unexpected model build")
        calls["count"] += 1
        t = _build_model(specs[i])
        built.append(t[0])
        return t

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = _targets
        v130._sample_weights = _sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights

    if calls["count"] != 2 or _LAST_CENTER_TARGETS is None:
        raise V200Error("V20 build/target capture failed")
    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    raw = built[-1].predict(v102._inputs(ctx["cache"], outer), batch_size=128, verbose=0)
    cdiag = v190._center_diagnostics(
        np.asarray(raw["birth_center_heatmap"]),
        np.asarray(_LAST_CENTER_TARGETS)[outer],
        np.asarray(_LAST_CENTER_ELIGIBLE)[outer],
        np.asarray(ctx["k"], dtype=np.int32)[outer],
    )
    report = _postprocess(args, report, ctx, cdiag)

    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    card = report["strata"]["aggregate"][MODEL_KEY]["cardinality"]
    arch = report["v200"]["architecture"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v200_f1": g["f1"],
        "pred_ref": g["prediction_reference_ratio"],
        "poly_exact": card.get("poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))),
        "center_top6_exact_poly": cdiag["top6_exact_center_coverage_poly"],
        "center_top6_hit_fraction_poly": cdiag["top6_mean_center_hit_fraction_poly"],
        "activity_gini": arch["outer_activity_gini"],
        "effective_active_slots": arch["outer_effective_active_slots"],
        "active_occupancy_correlation": arch["outer_active_occupancy_correlation"],
        "k2": report["per_true_k"]["2"][MODEL_KEY]["exact"],
        "k3": report["per_true_k"]["3"][MODEL_KEY]["exact"],
        "k4": report["per_true_k"]["4"][MODEL_KEY]["exact"],
        "k5": report["per_true_k"]["5"][MODEL_KEY]["exact"],
        "k6": report["per_true_k"]["6"][MODEL_KEY]["exact"],
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
