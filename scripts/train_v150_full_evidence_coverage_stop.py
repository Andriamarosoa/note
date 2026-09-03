"""V15.0 full-evidence sequential STOP with explicit coverage/novelty memory.

V13 showed that event queries can recover difficult/high-K births, but independent
presence heads duplicate the same evidence. V14 fixed the overcount with a
sequential CONTINUE/STOP process and destructive explaining-away, but later steps
lost acoustic information and K6 collapsed to zero.

V15 keeps V14's conditional sequential decision rule while changing only the
memory mechanism:
  * every step sees the complete TF and candidate evidence;
  * previous accepted soft events accumulate a differentiable coverage map;
  * each step receives both full-evidence and novelty-only pooled features;
  * coverage is updated as a probabilistic union, never by erasing the source;
  * runtime count is consecutive CONTINUE>=0.5 until first STOP.

No categorical K head, no threshold tuning, no runtime annotations. Evaluation is
identical five-fold composition outer-clean train-only protocol. locked12 is never
indexed or evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
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
from scripts.train_v100_spectral_string_slots import TIME_FRAMES
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v140_sequential_stop_explaining_away as v140

DEFAULT_SEED = 15051
STEPS = SLOT_COUNT
STOP_THRESHOLD = 0.5
QUERY_DIM = 96
EPS = 1e-6


class V150Error(RuntimeError):
    pass


def _input_by_name(model, name):
    for tensor in model.inputs:
        if tensor.name.split(":", 1)[0] == name:
            return tensor
    raise V150Error(f"model input not found: {name}")


def _build_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    base, _, token_shape = v102._build_model()
    candidate_context = base.get_layer("candidate_context").output
    tf_tokens = base.get_layer("tf_tokens").output
    candidate_set = _input_by_name(base, "candidate_set")
    candidate_mask = _input_by_name(base, "candidate_mask")

    cand = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="v150_candidate_norm")(candidate_set)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v150_candidate_hidden1")(cand)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v150_candidate_hidden2")(cand)
    cand_keys = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, use_bias=False), name="v150_candidate_keys")(cand)
    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v150_tf_keys")(tf_tokens)

    # Coverage starts empty. Unlike V14, the original evidence tensors are never
    # multiplied by a residual mask. Coverage is memory, not destructive state.
    coverage_tf = keras.layers.Lambda(lambda x: tf.zeros_like(x[:, :, 0]), name="v150_initial_tf_coverage")(tf_keys)
    coverage_cand = keras.layers.Lambda(lambda m: tf.zeros_like(tf.cast(m, tf.float32)), name="v150_initial_candidate_coverage")(candidate_mask)
    previous_state = keras.layers.Dense(QUERY_DIM, activation="relu", name="v150_initial_state")(candidate_context)

    survival = None
    continue_outputs = []
    time_outputs = []
    candidate_outputs = []
    novelty_tf_outputs = []
    novelty_cand_outputs = []
    coverage_tf_outputs = []
    coverage_cand_outputs = []
    overlap_tf_outputs = []
    overlap_cand_outputs = []

    token_freq = int(token_shape[1])

    for q in range(STEPS):
        coverage_tf_fraction = keras.layers.Lambda(
            lambda c: tf.reduce_mean(c, axis=1, keepdims=True),
            name=f"v150_step_{q}_coverage_tf_fraction",
        )(coverage_tf)
        coverage_cand_fraction = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_coverage_candidate_fraction",
        )([coverage_cand, candidate_mask])

        step_context = keras.layers.Concatenate(name=f"v150_step_{q}_context")([
            candidate_context, previous_state, coverage_tf_fraction, coverage_cand_fraction
        ])
        query = keras.layers.Dense(QUERY_DIM, activation="relu", name=f"v150_step_{q}_query")(step_context)

        # Full TF evidence remains visible at every step.
        tf_score = keras.layers.Lambda(
            lambda z: tf.einsum("btd,bd->bt", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"v150_step_{q}_tf_score",
        )([tf_keys, query])
        tf_affinity = keras.layers.Activation("sigmoid", name=f"v150_step_{q}_tf_affinity")(tf_score)
        tf_full_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_tf_full_distribution",
        )(tf_affinity)
        tf_full_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v150_step_{q}_tf_full_pool",
        )([tf_tokens, tf_full_dist])

        tf_novel = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v150_step_{q}_tf_novel",
        )([tf_affinity, coverage_tf])
        tf_novel_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_tf_novel_distribution",
        )(tf_novel)
        tf_novel_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v150_step_{q}_tf_novel_pool",
        )([tf_tokens, tf_novel_dist])
        tf_novelty_fraction = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_tf_novelty_fraction",
        )([tf_novel, tf_affinity])
        tf_overlap_fraction = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_tf_overlap_fraction",
        )([tf_affinity, coverage_tf])

        # Same full/novel decomposition for candidate evidence.
        cand_score = keras.layers.Lambda(
            lambda z: tf.einsum("bcd,bd->bc", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
            name=f"v150_step_{q}_candidate_score",
        )([cand_keys, query])
        cand_affinity_raw = keras.layers.Activation("sigmoid", name=f"v150_step_{q}_candidate_affinity_raw")(cand_score)
        cand_affinity = keras.layers.Multiply(name=f"v150_step_{q}_candidate_affinity")([cand_affinity_raw, candidate_mask])
        cand_full_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_candidate_full_distribution",
        )(cand_affinity)
        cand_full_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v150_step_{q}_candidate_full_pool",
        )([cand, cand_full_dist])

        cand_novel = keras.layers.Lambda(
            lambda z: z[0] * (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v150_step_{q}_candidate_novel",
        )([cand_affinity, coverage_cand])
        cand_novel_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS),
            name=f"v150_event_candidate_{q}",
        )(cand_novel)
        cand_novel_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1),
            name=f"v150_step_{q}_candidate_novel_pool",
        )([cand, cand_novel_dist])
        cand_novelty_fraction = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[1], axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_candidate_novelty_fraction",
        )([cand_novel, cand_affinity])
        cand_overlap_fraction = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1, keepdims=True) /
                      (tf.reduce_sum(z[0], axis=1, keepdims=True) + EPS),
            name=f"v150_step_{q}_candidate_overlap_fraction",
        )([cand_affinity, coverage_cand])

        # Birth-time auxiliary uses novel TF evidence, while full TF latent still
        # reaches the CONTINUE head and can rescue late/high-K births.
        tf_grid = keras.layers.Reshape((TIME_FRAMES, token_freq), name=f"v150_step_{q}_tf_grid")(tf_novel)
        time_mass = keras.layers.Lambda(lambda a: tf.reduce_sum(a, axis=2), name=f"v150_step_{q}_time_mass")(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS),
            name=f"v150_event_time_{q}",
        )(time_mass)

        feature = keras.layers.Concatenate(name=f"v150_step_{q}_feature")([
            candidate_context, previous_state, query,
            tf_full_latent, tf_novel_latent, cand_full_latent, cand_novel_latent,
            tf_novelty_fraction, cand_novelty_fraction,
            tf_overlap_fraction, cand_overlap_fraction,
            coverage_tf_fraction, coverage_cand_fraction,
        ])
        feature = keras.layers.LayerNormalization(name=f"v150_step_{q}_feature_norm")(feature)
        hidden = keras.layers.Dense(
            128, activation="relu", kernel_regularizer=keras.regularizers.l2(1.5e-3),
            name=f"v150_step_{q}_hidden1",
        )(feature)
        hidden = keras.layers.Dropout(0.08, name=f"v150_step_{q}_dropout")(hidden)
        hidden = keras.layers.Dense(64, activation="relu", name=f"v150_step_{q}_hidden2")(hidden)

        # Keep compatibility keys for V14's already-audited training harness.
        cont = keras.layers.Dense(1, activation="sigmoid", name=f"v140_continue_{q}")(hidden)
        survival = cont if survival is None else keras.layers.Multiply(name=f"v150_survival_{q}")([survival, cont])

        # Probabilistic union coverage. The source itself stays intact.
        tf_claim = keras.layers.Multiply(name=f"v150_step_{q}_tf_claim")([tf_affinity, survival])
        cand_claim = keras.layers.Multiply(name=f"v150_step_{q}_candidate_claim")([cand_affinity, survival])
        coverage_tf = keras.layers.Lambda(
            lambda z: 1.0 - (1.0 - tf.clip_by_value(z[0], 0.0, 1.0)) *
                              (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v150_step_{q}_tf_coverage_after",
        )([coverage_tf, tf_claim])
        coverage_cand = keras.layers.Lambda(
            lambda z: 1.0 - (1.0 - tf.clip_by_value(z[0], 0.0, 1.0)) *
                              (1.0 - tf.clip_by_value(z[1], 0.0, 1.0)),
            name=f"v150_step_{q}_candidate_coverage_after",
        )([coverage_cand, cand_claim])

        previous_state = keras.layers.Dense(QUERY_DIM, activation="relu", name=f"v150_step_{q}_state")(hidden)
        continue_outputs.append(cont)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_novel_dist)
        novelty_tf_outputs.append(tf_novelty_fraction)
        novelty_cand_outputs.append(cand_novelty_fraction)
        coverage_tf_outputs.append(coverage_tf_fraction)
        coverage_cand_outputs.append(coverage_cand_fraction)
        overlap_tf_outputs.append(tf_overlap_fraction)
        overlap_cand_outputs.append(cand_overlap_fraction)

    cumulative = []
    cur = None
    for q, cont in enumerate(continue_outputs):
        cur = cont if cur is None else keras.layers.Multiply(name=f"v150_expected_survival_{q}")([cur, cont])
        cumulative.append(cur)
    expected_count = keras.layers.Add(name="v150_expected_count")(cumulative)
    count_norm = keras.layers.Lambda(lambda x: x / float(STEPS), name="v140_count_norm")(expected_count)

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    for q in range(STEPS):
        outputs[f"v140_continue_{q}"] = continue_outputs[q]
        outputs[f"v140_event_time_{q}"] = time_outputs[q]
        outputs[f"v140_event_candidate_{q}"] = candidate_outputs[q]
        # V14 harness calls these 'mass'/'residual'; in V15 they carry novelty
        # and pre-step coverage fractions and are renamed in the final report.
        outputs[f"v140_tf_mass_{q}"] = novelty_tf_outputs[q]
        outputs[f"v140_candidate_mass_{q}"] = novelty_cand_outputs[q]
        outputs[f"v140_residual_tf_{q}"] = coverage_tf_outputs[q]
        outputs[f"v140_residual_candidate_{q}"] = coverage_cand_outputs[q]
    outputs["v140_count_norm"] = count_norm

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    for q in range(STEPS):
        loss[f"v140_continue_{q}"] = "binary_crossentropy"
        loss[f"v140_event_time_{q}"] = keras.losses.KLDivergence()
        loss[f"v140_event_candidate_{q}"] = "categorical_crossentropy"
    loss["v140_count_norm"] = "mse"

    # Identical weights to V14: isolate memory mechanism rather than retuning.
    loss_weights = {f"string_{slot}": 0.14 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.03 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.08 for slot in range(SLOT_COUNT)})
    for q in range(STEPS):
        loss_weights[f"v140_continue_{q}"] = 1.0
        loss_weights[f"v140_event_time_{q}"] = 0.25
        loss_weights[f"v140_event_candidate_{q}"] = 0.20
    loss_weights["v140_count_norm"] = 0.25

    model = keras.Model(base.inputs, outputs, name="v150_full_evidence_coverage_stop")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, token_shape


def _postprocess(output_dir: Path, fold: int, report: dict) -> dict:
    # The V14 harness evaluated the V15 model under the compatibility key v140.
    # Rename only after all metrics are computed.
    for row in report.get("strata", {}).values():
        if row and "v140" in row:
            row["v150"] = row.pop("v140")
    for row in report.get("per_true_k", {}).values():
        if "v140" in row:
            row["v150"] = row.pop("v140")

    p = report["protocol"]
    p.pop("explaining_away_uses_predictions_not_annotations", None)
    p.update({
        "coverage_memory_uses_predictions_not_annotations": True,
        "destructive_explaining_away": False,
        "full_evidence_visible_every_step": True,
        "novelty_memory": True,
    })
    report["architecture"] = {
        **report["architecture"],
        "name": "V15.0 full-evidence sequential STOP with coverage/novelty memory",
        "tf_residual_explaining_away": False,
        "candidate_residual_explaining_away": False,
        "full_tf_evidence_each_step": True,
        "full_candidate_evidence_each_step": True,
        "coverage_update": "probabilistic union of prediction-weighted claims",
        "novelty_features": True,
        "conditional_training_rule": "step q receives decision loss only on K>=q",
    }
    old = report.pop("residual_diagnostics", None)
    if old:
        report["coverage_diagnostics"] = {
            "mean_tf_coverage_before_step": old.get("mean_tf_residual_before_step"),
            "mean_candidate_coverage_before_step": old.get("mean_candidate_residual_before_step"),
        }
    report["conditional_diagnostics"]["diagnostic_mass_semantics"] = "novelty fraction relative to full affinity"

    rp = output_dir / f"report-fold-{fold}.json"
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    npz_path = output_dir / f"predictions-fold-{fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    data["pred150"] = data.pop("pred140")
    data["coverage_tf"] = data.pop("residual_tf")
    data["coverage_candidate"] = data.pop("residual_candidate")
    np.savez_compressed(npz_path, **data)

    old_w = output_dir / f"v140-fold-{fold}.weights.h5"
    new_w = output_dir / f"v150-fold-{fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    return report


def train_fold(args):
    # Reuse the already-audited V14 outer-clean harness and change only the graph.
    original = v140._build_model
    try:
        v140._build_model = _build_model
        report = v140.train_fold(args)
    finally:
        v140._build_model = original
    report = _postprocess(args.output_dir, args.outer_fold, report)
    print(json.dumps({
        "outer": args.outer_fold,
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v130_f1": report["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"],
        "v150_f1": report["strata"]["aggregate"]["v150"]["metrics"]["global"]["f1"],
        "k5_exact_v150": report["per_true_k"]["5"]["v150"]["exact"],
        "k6_exact_v150": report["per_true_k"]["6"]["v150"]["exact"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path, required=True)
    p.add_argument("--v130-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
