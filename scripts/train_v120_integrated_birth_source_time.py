"""V12.0 integrated birth -> positive multiplicity source-time model.

V11.1 established, with a nested outer-clean audit, that separating birth
existence from positive multiplicity recovers recall in the player00/comp/Rock
failure region, but a small fusion head did not improve global F1 and lost high-K
multiplicity.  V12.0 moves the factorization into the acoustic source-time model
itself instead of stacking another fusion on frozen experts.

Architecture:
- V10.2 causal competitive six-string + background source-time backbone;
- explicit binary birth head P(K>0);
- positive-only six-class multiplicity head Q(K=1..6 | birth);
- positive-only Poisson-binomial source-count auxiliary (K0 normalized away);
- positive-only tail auxiliaries P(K>=2)..P(K>=6), including direct K6 pressure;
- multiplicity context includes source assignment mass and per-string birth-time
  expectation/entropy so micro-strum geometry can inform K;
- K0 has zero weight in every multiplicity/tail/source-count objective.

Runtime decode is genuinely hierarchical and fixed a priori:
    P_birth < 0.5  -> K=0
    P_birth >= 0.5 -> K=1+argmax Q(K|birth)

The script supports one outer composition fold at a time.  Epoch count is chosen
only on one inner fold from the remaining four, then a fresh model is trained on
all four outer-train folds for exactly that many epochs and evaluated once on the
untouched outer fold.  Historical validation/locked12 are never indexed or
used for model selection or evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import ALLOWED_PLAYERS, SLOT_COUNT
from scripts.train_boundaries import group_stem
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import TIME_FRAMES, _load_spectral_caches
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_class_conditional_fusion as v104
from scripts import train_v104_oof_fold as oofmod

DEFAULT_SEED = 12031
BIRTH_THRESHOLD = 0.5
MAX_META_EPOCHS = 35
TAIL_STAGES = tuple(range(2, SLOT_COUNT + 1))


class V120Error(RuntimeError):
    pass


def _build_model():
    """Build a joint acoustic model; no V10.1/V10.2 prediction is an input."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    base, _, token_shape = v102._build_model()

    count_latent = base.get_layer("ordinal_aux_hidden2").output
    string_birth = base.get_layer("string_birth_vector").output
    pb_all = base.get_layer("structured_count").output
    pb_positive = keras.layers.Lambda(
        lambda p: p[:, 1:] / (tf.reduce_sum(p[:, 1:], axis=1, keepdims=True) + 1e-7),
        name="pb_positive_count",
    )(pb_all)

    assignment = base.get_layer("competitive_source_assignment").output
    source_mass = keras.layers.Lambda(
        lambda a: tf.reduce_mean(a, axis=1), name="source_assignment_mass"
    )(assignment)

    time_evidence = []
    time_axis = tf.constant(np.linspace(-1.0, 1.0, TIME_FRAMES, dtype=np.float32))
    for slot in range(SLOT_COUNT):
        t = base.get_layer(f"time_{slot}").output
        expected = keras.layers.Lambda(
            lambda q: tf.reduce_sum(q * time_axis[None, :], axis=1, keepdims=True),
            name=f"string_{slot}_time_expected",
        )(t)
        entropy = keras.layers.Lambda(
            lambda q: -tf.reduce_sum(q * tf.math.log(tf.clip_by_value(q, 1e-7, 1.0)), axis=1, keepdims=True)
            / math.log(float(TIME_FRAMES)),
            name=f"string_{slot}_time_entropy",
        )(t)
        time_evidence.extend((expected, entropy))

    context = keras.layers.Concatenate(name="integrated_count_context")([
        count_latent,
        string_birth,
        pb_positive,
        source_mass,
        *time_evidence,
    ])
    context = keras.layers.LayerNormalization(name="integrated_count_norm")(context)
    shared = keras.layers.Dense(160, activation="relu", kernel_regularizer=keras.regularizers.l2(2e-3), name="integrated_shared_1")(context)
    shared = keras.layers.Dropout(0.10, name="integrated_shared_dropout")(shared)
    shared = keras.layers.Dense(96, activation="relu", kernel_regularizer=keras.regularizers.l2(2e-3), name="integrated_shared_2")(shared)

    birth_hidden = keras.layers.Dense(64, activation="relu", name="birth_hidden")(shared)
    birth = keras.layers.Dense(1, activation="sigmoid", name="birth")(birth_hidden)

    mult_hidden = keras.layers.Concatenate(name="multiplicity_source_context")([
        shared, string_birth, pb_positive, source_mass, *time_evidence
    ])
    mult_hidden = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(2e-3), name="multiplicity_hidden_1")(mult_hidden)
    mult_hidden = keras.layers.Dropout(0.08, name="multiplicity_dropout")(mult_hidden)
    mult_hidden = keras.layers.Dense(80, activation="relu", name="multiplicity_hidden_2")(mult_hidden)
    multiplicity = keras.layers.Dense(SLOT_COUNT, activation="softmax", name="multiplicity")(mult_hidden)
    tail = {
        f"ge{stage}_positive": keras.layers.Dense(1, activation="sigmoid", name=f"ge{stage}_positive")(mult_hidden)
        for stage in TAIL_STAGES
    }

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    outputs["birth"] = birth
    outputs["multiplicity"] = multiplicity
    outputs["pb_positive_count"] = pb_positive
    outputs.update(tail)

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["birth"] = "binary_crossentropy"
    loss["multiplicity"] = "categorical_crossentropy"
    loss["pb_positive_count"] = "categorical_crossentropy"
    loss.update({f"ge{stage}_positive": "binary_crossentropy" for stage in TAIL_STAGES})

    loss_weights = {f"string_{slot}": 0.45 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.30 for slot in range(SLOT_COUNT)})
    loss_weights["birth"] = 1.50
    loss_weights["multiplicity"] = 1.50
    loss_weights["pb_positive_count"] = 0.60
    loss_weights.update({f"ge{stage}_positive": 0.15 * math.sqrt(stage - 1) for stage in TAIL_STAGES})

    model = keras.Model(base.inputs, outputs, name="v120_integrated_birth_source_time")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, token_shape


def _positive_class_weights(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.int32)
    active = k > 0
    out = np.zeros(len(k), dtype=np.float32)
    if not np.any(active):
        return out
    counts = np.asarray([np.sum(k == value) for value in range(1, SLOT_COUNT + 1)], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    raw = np.sqrt(float(np.sum(active)) / (SLOT_COUNT * counts))
    raw = np.clip(raw, 0.70, 4.50)
    raw /= np.sum(raw * counts) / np.sum(counts)
    out[active] = raw[k[active] - 1].astype(np.float32)
    out[active] /= float(np.mean(out[active]))
    return out


def _birth_weights(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.int32)
    pos = int(np.sum(k > 0))
    neg = int(np.sum(k == 0))
    if pos == 0 or neg == 0:
        raise V120Error("birth loss requires positive and negative rows")
    pos_weight = min(5.0, math.sqrt(neg / pos))
    out = np.ones(len(k), dtype=np.float32)
    out[k > 0] = float(pos_weight)
    out /= float(np.mean(out))
    return out


def _tail_weights(k: np.ndarray, stage: int) -> np.ndarray:
    k = np.asarray(k, dtype=np.int32)
    active = k > 0
    out = np.zeros(len(k), dtype=np.float32)
    if not np.any(active):
        return out
    target = k >= stage
    pos = int(np.sum(active & target))
    neg = int(np.sum(active & ~target))
    if pos == 0 or neg == 0:
        out[active] = 1.0
        return out
    wp = min(6.0, math.sqrt(neg / pos))
    out[active & ~target] = 1.0
    out[active & target] = float(wp)
    out[active] /= float(np.mean(out[active]))
    return out


def _targets(slot_targets, pitch_targets, time_targets, k):
    k = np.minimum(np.asarray(k, dtype=np.int32), SLOT_COUNT)
    mult_class = np.clip(k - 1, 0, SLOT_COUNT - 1)
    mult = np.eye(SLOT_COUNT, dtype=np.float32)[mult_class]
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = np.asarray(slot_targets[:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"pitch_{slot}"] = np.asarray(pitch_targets[:, slot], dtype=np.float32).reshape(-1, 1)
        out[f"time_{slot}"] = np.asarray(time_targets[:, slot, :], dtype=np.float32)
    out["birth"] = (k > 0).astype(np.float32).reshape(-1, 1)
    out["multiplicity"] = mult
    out["pb_positive_count"] = mult
    out.update({f"ge{stage}_positive": (k >= stage).astype(np.float32).reshape(-1, 1) for stage in TAIL_STAGES})
    return out


def _sample_weights(slot_targets, time_mask, k):
    base = v102._sample_weights(slot_targets, time_mask, k)
    out = {}
    for slot in range(SLOT_COUNT):
        out[f"string_{slot}"] = base[f"string_{slot}"]
        out[f"pitch_{slot}"] = base[f"pitch_{slot}"]
        out[f"time_{slot}"] = base[f"time_{slot}"]
    out["birth"] = _birth_weights(k)
    positive = _positive_class_weights(k)
    out["multiplicity"] = positive
    out["pb_positive_count"] = positive
    out.update({f"ge{stage}_positive": _tail_weights(k, stage) for stage in TAIL_STAGES})
    return out


def _decode(model, inputs):
    raw = model.predict(inputs, batch_size=128, verbose=0)
    birth = np.asarray(raw["birth"], dtype=np.float64).reshape(-1)
    multiplicity = np.asarray(raw["multiplicity"], dtype=np.float64)
    pb_positive = np.asarray(raw["pb_positive_count"], dtype=np.float64)
    pred = np.zeros(len(birth), dtype=np.int32)
    positive = birth >= BIRTH_THRESHOLD
    pred[positive] = 1 + np.argmax(multiplicity[positive], axis=1).astype(np.int32)
    tails = np.stack([
        np.asarray(raw[f"ge{stage}_positive"], dtype=np.float64).reshape(-1)
        for stage in TAIL_STAGES
    ], axis=1)
    return pred, birth, multiplicity, pb_positive, tails


def _metrics(cache, train_split, idx, pred):
    return v104._metrics_for_indices(
        cache, train_split, np.asarray(idx, dtype=np.int64), np.asarray(pred, dtype=np.int32)
    )


def _card(k, pred):
    return v102._cardinality_report(np.asarray(k, dtype=np.int32), np.asarray(pred, dtype=np.int32))


def _player(member: str):
    p = str(member).split("_", 1)[0]
    if p not in ALLOWED_PLAYERS:
        raise V120Error(f"unexpected player {p}")
    return p


def _mode(member: str):
    s = str(member)
    if s.endswith("_comp.jams"):
        return "comp"
    if s.endswith("_solo.jams"):
        return "solo"
    return "other"


def _genre(group: str):
    m = re.match(r"^([A-Za-z]+)", str(group))
    return m.group(1) if m else "unknown"


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def _load_baseline(eval_dir: Path, outer: int, outer_idx: np.ndarray):
    paths = sorted(eval_dir.glob(f"**/v104-nested-eval-{outer}.npz"))
    if len(paths) != 1:
        raise V120Error(f"outer {outer}: expected one nested V10.4 eval shard, found {len(paths)}")
    data = _load_npz(paths[0])
    if not np.array_equal(np.asarray(data["global_index"], dtype=np.int64), outer_idx):
        raise V120Error(f"outer {outer}: baseline coverage mismatch")
    return data


def _slice_report(cache, train_split, k, idx, predictions):
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return None
    out = {"clusters": int(len(idx))}
    for name, pred in predictions.items():
        out[name] = {
            "metrics": _metrics(cache, train_split, idx, np.asarray(pred, dtype=np.int32)[idx]),
            "cardinality": _card(k[idx], np.asarray(pred, dtype=np.int32)[idx]),
        }
    return out


def train_fold(args):
    random.seed(args.seed + args.outer_fold)
    np.random.seed(args.seed + args.outer_fold)
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    tf.random.set_seed(args.seed + args.outer_fold)

    if not 0 <= args.outer_fold < oofmod.FOLD_COUNT:
        raise V120Error("outer fold outside range")

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if train_members & validation_members:
        raise V120Error("train/validation leakage")
    if set(cache["track_members"]) != train_members:
        raise V120Error("spectral cache does not exactly cover train split")

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
    if np.intersect1d(final_fit_idx, outer_idx).size:
        raise V120Error("outer fold leaked into final fit")
    if np.intersect1d(meta_fit_idx, outer_idx).size or np.intersect1d(meta_val_idx, outer_idx).size:
        raise V120Error("outer fold leaked into epoch selection")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, time_targets, _, supervision = v102._derive_supervision(
        members,
        candidate_samples,
        args.dataset_dir,
        expected_slot_targets=cache["slot_targets"],
    )

    tf.keras.backend.clear_session()
    random.seed(args.seed + 100 + args.outer_fold)
    np.random.seed(args.seed + 100 + args.outer_fold)
    tf.random.set_seed(args.seed + 100 + args.outer_fold)
    probe, loss_weights, token_shape = _build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=2e-4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5),
    ]
    hist = probe.fit(
        v102._inputs(cache, meta_fit_idx),
        _targets(cache["slot_targets"][meta_fit_idx], pitch_targets[meta_fit_idx], time_targets[meta_fit_idx], k[meta_fit_idx]),
        sample_weight=_sample_weights(cache["slot_targets"][meta_fit_idx], time_mask[meta_fit_idx], k[meta_fit_idx]),
        validation_data=(
            v102._inputs(cache, meta_val_idx),
            _targets(cache["slot_targets"][meta_val_idx], pitch_targets[meta_val_idx], time_targets[meta_val_idx], k[meta_val_idx]),
            _sample_weights(cache["slot_targets"][meta_val_idx], time_mask[meta_val_idx], k[meta_val_idx]),
        ),
        epochs=MAX_META_EPOCHS,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    selected_epochs = max(2, int(np.argmin(np.asarray(hist.history["val_loss"], dtype=np.float64)) + 1))
    meta_pred, meta_birth, _, _, _ = _decode(probe, v102._inputs(cache, meta_val_idx))
    meta_metrics = _metrics(cache, train_split, meta_val_idx, meta_pred)

    tf.keras.backend.clear_session()
    random.seed(args.seed + 1000 + args.outer_fold)
    np.random.seed(args.seed + 1000 + args.outer_fold)
    tf.random.set_seed(args.seed + 1000 + args.outer_fold)
    model, _, _ = _build_model()
    model.fit(
        v102._inputs(cache, final_fit_idx),
        _targets(cache["slot_targets"][final_fit_idx], pitch_targets[final_fit_idx], time_targets[final_fit_idx], k[final_fit_idx]),
        sample_weight=_sample_weights(cache["slot_targets"][final_fit_idx], time_mask[final_fit_idx], k[final_fit_idx]),
        epochs=selected_epochs,
        batch_size=64,
        shuffle=True,
        verbose=2,
    )
    pred120_outer, birth_outer, mult_outer, pb_outer, tails_outer = _decode(model, v102._inputs(cache, outer_idx))

    baseline = _load_baseline(args.baseline_eval_dir, args.outer_fold, outer_idx)
    pred101_outer = np.asarray(baseline["pred101_deploy"], dtype=np.int32)
    pred102_outer = np.asarray(baseline["pred102_deploy"], dtype=np.int32)
    pred104_outer = np.asarray(baseline["pred104_deploy"], dtype=np.int32)

    pred120 = np.full(len(k), -1, dtype=np.int32)
    pred101 = np.full(len(k), -1, dtype=np.int32)
    pred102 = np.full(len(k), -1, dtype=np.int32)
    pred104 = np.full(len(k), -1, dtype=np.int32)
    pred120[outer_idx] = pred120_outer
    pred101[outer_idx] = pred101_outer
    pred102[outer_idx] = pred102_outer
    pred104[outer_idx] = pred104_outer

    players = np.asarray([_player(m) for m in members], dtype="U2")
    modes = np.asarray([_mode(m) for m in members], dtype="U8")
    groups = np.asarray([group_stem(by_member[m]) for m in members], dtype="U64")
    genres = np.asarray([_genre(g) for g in groups], dtype="U16")

    def outer_where(mask):
        return outer_idx[np.asarray(mask, dtype=bool)[outer_idx]]

    strata = {
        "aggregate": outer_idx,
        "comp": outer_where(modes == "comp"),
        "solo": outer_where(modes == "solo"),
        "player00": outer_where(players == "00"),
        "player00_comp": outer_where((players == "00") & (modes == "comp")),
        "player00_solo": outer_where((players == "00") & (modes == "solo")),
        "player00_rock_comp": outer_where((players == "00") & (modes == "comp") & (genres == "Rock")),
    }
    for player in ALLOWED_PLAYERS:
        strata[f"player{player}"] = outer_where(players == player)

    predictions = {"v101": pred101, "v102": pred102, "v104": pred104, "v120": pred120}
    strata_report = {
        name: _slice_report(cache, train_split, k, idx, predictions)
        for name, idx in strata.items()
    }

    per_k = {}
    ko = k[outer_idx]
    for value in range(SLOT_COUNT + 1):
        local = ko == value
        row = {"clusters": int(np.sum(local))}
        for name, pred in (
            ("v101", pred101_outer), ("v102", pred102_outer),
            ("v104", pred104_outer), ("v120", pred120_outer),
        ):
            if np.any(local):
                pp = pred[local]
                row[name] = {
                    "exact": float(np.mean(pp == value)),
                    "under_rate": float(np.mean(pp < value)),
                    "over_rate": float(np.mean(pp > value)),
                    "mae": float(np.mean(np.abs(pp - value))),
                }
            else:
                row[name] = {"exact": None, "under_rate": None, "over_rate": None, "mae": None}
        per_k[str(value)] = row

    report = {
        "schema_version": 1,
        "protocol": {
            "train_only_nested_outer_holdout": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "outer_fold_used_for_training": False,
            "outer_fold_used_for_epoch_selection": False,
            "runtime_inputs_use_annotations": False,
            "birth_threshold_tuned": False,
            "birth_threshold": BIRTH_THRESHOLD,
            "multiplicity_loss_receives_k0_weight": False,
            "tail_losses_receive_k0_weight": False,
            "positive_pb_loss_receives_k0_weight": False,
            "grouping_window_ms_unchanged": 40,
            "offset_model_untouched": True,
        },
        "architecture": {
            "name": "V12.0 integrated hierarchical birth/source-time multiplicity",
            "acoustic_backbone": "V10.2 competitive six-string+background source-time",
            "expert_predictions_as_runtime_inputs": False,
            "count_factorization": "P(K>0) then Q(K=1..6|birth)",
            "positive_source_count_auxiliary": "Poisson-binomial normalized over K=1..6",
            "high_k_tail_auxiliaries": [f"P(K>={stage}|birth)" for stage in TAIL_STAGES],
            "multiplicity_uses_source_mass": True,
            "multiplicity_uses_source_time_expectation_entropy": True,
            "decode": "birth>=0.5 then 1+argmax multiplicity, else 0",
            "trainable_parameters": int(model.count_params()),
            "tf_token_shape": list(token_shape),
            "loss_weights": loss_weights,
        },
        "outer_fold": int(args.outer_fold),
        "outer_groups": groups_per_fold[args.outer_fold],
        "data": {
            "outer_clusters": int(len(outer_idx)),
            "outer_train_clusters": int(len(final_fit_idx)),
            "meta_fit_clusters": int(len(meta_fit_idx)),
            "meta_validation_clusters": int(len(meta_val_idx)),
            "meta_validation_fold": int(meta_fold),
            "selected_epochs": int(selected_epochs),
            "train_tracks": int(len(train_split)),
            "validation_tracks_not_evaluated": int(len(validation)),
        },
        "supervision": {
            "assigned_events": supervision.get("assigned_events"),
            "unassigned_events": supervision.get("unassigned_events"),
            "slot_mask_agreement": supervision.get("slot_mask_agreement"),
            "active_slot_time_coverage": supervision.get("active_slot_time_coverage"),
            "cluster_reconstruction": reconstruction,
        },
        "meta_validation": {
            "metrics": meta_metrics,
            "cardinality": _card(k[meta_val_idx], meta_pred),
            "mean_birth_probability": float(np.mean(meta_birth)),
        },
        "strata": strata_report,
        "per_true_k": per_k,
        "birth": {
            "mean_probability": float(np.mean(birth_outer)),
            "positive_rate_at_050": float(np.mean(birth_outer >= BIRTH_THRESHOLD)),
            "true_birth_rate": float(np.mean(ko > 0)),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output_dir / f"predictions-fold-{args.outer_fold}.npz",
        schema_version=np.asarray([1], dtype=np.int16),
        outer_fold=np.full(len(outer_idx), args.outer_fold, dtype=np.int16),
        global_index=outer_idx,
        k=ko.astype(np.int16),
        member=members[outer_idx],
        pred101=pred101_outer.astype(np.int16),
        pred102=pred102_outer.astype(np.int16),
        pred104=pred104_outer.astype(np.int16),
        pred120=pred120_outer.astype(np.int16),
        birth=birth_outer.astype(np.float32),
        multiplicity=mult_outer.astype(np.float32),
        pb_positive=pb_outer.astype(np.float32),
        tails=tails_outer.astype(np.float32),
    )
    model.save_weights(args.output_dir / f"v120-fold-{args.outer_fold}.weights.h5")
    print(json.dumps({
        "outer": args.outer_fold,
        "epochs": selected_epochs,
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v120_f1": report["strata"]["aggregate"]["v120"]["metrics"]["global"]["f1"],
        "player00_rock_comp_v104": (
            report["strata"]["player00_rock_comp"]["v104"]["metrics"]["global"]["f1"]
            if report["strata"]["player00_rock_comp"] else None
        ),
        "player00_rock_comp_v120": (
            report["strata"]["player00_rock_comp"]["v120"]["metrics"]["global"]["f1"]
            if report["strata"]["player00_rock_comp"] else None
        ),
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
