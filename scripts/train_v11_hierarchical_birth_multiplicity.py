"""Strict-OOF V11 pilot: hierarchical birth existence + positive multiplicity.

Motivation comes from the train-only player00 failure-mode audit: V10.4 improves
aggregate exact K largely through K=0 while losing K=1/K=2 birth recall in some
compositions.  V11 removes that objective coupling structurally:

    b = P(K > 0)
    q = P(K = 1..6 | K > 0)
    P(K=0) = 1-b
    P(K=k) = b*q[k-1], k=1..6

K=0 examples have zero weight in the multiplicity loss, so silence can never
improve the multiplicity objective.  This is a train-only strict cross-fit
pilot.  Historical validation and locked12 are never indexed or evaluated.

For each held composition fold, one of the four remaining folds is used only as
an inner meta-validation fold.  The other three select a birth-loss weighting
variant and epoch count.  The selected configuration is then retrained on all
four non-held folds for exactly that many epochs and evaluated once on the held
fold.  Thus every reported V11 prediction is outer-fold clean.
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
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_class_conditional_fusion as v104

DEFAULT_SEED = 11101
MAX_META_EPOCHS = 60
BASELINE_V104_EPOCHS = 22  # frozen historical train-only V10.4 selection
BIRTH_VARIANTS = ("plain", "sqrt_positive")


def _build_model(feature_dim: int):
    import tensorflow as tf
    from tensorflow import keras

    feat = keras.Input((feature_dim,), name="fusion_features")
    anchor = keras.Input((SLOT_COUNT + 1,), name="v101_anchor")
    p2 = keras.Input((SLOT_COUNT + 1,), name="v102_count")

    log_anchor = keras.layers.Lambda(
        lambda x: tf.math.log(tf.clip_by_value(x, 1e-7, 1.0)), name="log_anchor"
    )(anchor)
    log_p2 = keras.layers.Lambda(
        lambda x: tf.math.log(tf.clip_by_value(x, 1e-7, 1.0)), name="log_p2"
    )(p2)
    delta = keras.layers.Subtract(name="expert_delta")([p2, anchor])
    x = keras.layers.Concatenate(name="hier_input")([feat, log_anchor, log_p2, delta])
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(3e-3), name="shared_1")(x)
    x = keras.layers.Dropout(0.08, name="shared_dropout")(x)
    x = keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(3e-3), name="shared_2")(x)

    birth = keras.layers.Dense(1, activation="sigmoid", name="birth")(x)
    mult = keras.layers.Dense(SLOT_COUNT, activation="softmax", name="multiplicity")(x)
    model = keras.Model(
        {"fusion_features": feat, "v101_anchor": anchor, "v102_count": p2},
        {"birth": birth, "multiplicity": mult},
        name="v11_hierarchical_birth_multiplicity",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(7e-4),
        loss={"birth": "binary_crossentropy", "multiplicity": "categorical_crossentropy"},
        loss_weights={"birth": 1.0, "multiplicity": 1.0},
    )
    return model


def _inputs(features, anchor, p102):
    return {
        "fusion_features": np.asarray(features, dtype=np.float32),
        "v101_anchor": np.asarray(anchor, dtype=np.float32),
        "v102_count": np.asarray(p102, dtype=np.float32),
    }


def _targets(k):
    k = np.asarray(k, dtype=np.int32)
    birth = (k > 0).astype(np.float32)[:, None]
    mult_class = np.clip(k - 1, 0, SLOT_COUNT - 1)
    mult = np.eye(SLOT_COUNT, dtype=np.float32)[mult_class]
    return {"birth": birth, "multiplicity": mult}


def _weight_spec(k_fit, variant: str):
    k_fit = np.asarray(k_fit, dtype=np.int32)
    pos = int(np.sum(k_fit > 0))
    neg = int(np.sum(k_fit == 0))
    if pos == 0 or neg == 0:
        raise RuntimeError("birth weighting requires both classes")
    if variant == "plain":
        birth_pos = 1.0
    elif variant == "sqrt_positive":
        birth_pos = float(np.sqrt(neg / pos))
    else:
        raise RuntimeError(f"unknown birth variant {variant}")

    # Positive multiplicity weights are estimated only from K>0. K0 has zero
    # multiplicity weight and therefore cannot affect this objective.
    counts = np.asarray([np.sum(k_fit == kk) for kk in range(1, SLOT_COUNT + 1)], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    raw = np.sqrt(np.sum(counts) / (SLOT_COUNT * counts))
    raw /= np.sum(raw * counts) / np.sum(counts)
    raw = np.clip(raw, 0.70, 2.50)
    return birth_pos, raw.astype(np.float32)


def _sample_weights(k, birth_pos, mult_by_class):
    k = np.asarray(k, dtype=np.int32)
    wb = np.ones(len(k), dtype=np.float32)
    wb[k > 0] = float(birth_pos)
    wb /= np.mean(wb)
    wm = np.zeros(len(k), dtype=np.float32)
    pos = k > 0
    if np.any(pos):
        wm[pos] = np.asarray(mult_by_class, dtype=np.float32)[k[pos] - 1]
        wm[pos] /= np.mean(wm[pos])
    return {"birth": wb, "multiplicity": wm}


def _probabilities(model, features, anchor, p102):
    out = model.predict(_inputs(features, anchor, p102), batch_size=256, verbose=0)
    b = np.asarray(out["birth"], dtype=np.float64).reshape(-1)
    q = np.asarray(out["multiplicity"], dtype=np.float64)
    p = np.empty((len(b), SLOT_COUNT + 1), dtype=np.float64)
    p[:, 0] = 1.0 - b
    p[:, 1:] = b[:, None] * q
    p /= np.sum(p, axis=1, keepdims=True)
    return p


def _card(k, pred):
    return v102._cardinality_report(np.asarray(k, dtype=np.int32), np.asarray(pred, dtype=np.int32))


def _metrics(cache, train_split, idx, pred):
    return v104._metrics_for_indices(
        cache, train_split, np.asarray(idx, dtype=np.int64), np.asarray(pred, dtype=np.int32)
    )


def _player(member: str):
    p = str(member).split("_", 1)[0]
    if p not in ALLOWED_PLAYERS:
        raise RuntimeError(f"unexpected player {p}")
    return p


def _mode(member: str):
    m = str(member)
    if m.endswith("_comp.jams"):
        return "comp"
    if m.endswith("_solo.jams"):
        return "solo"
    return "other"


def _genre(group: str):
    m = re.match(r"^([A-Za-z]+)", str(group))
    return m.group(1) if m else "unknown"


def _train_meta_candidate(cache, train_split, k, features, anchor, p102, fold, held, variant, seed):
    import tensorflow as tf
    from tensorflow import keras

    remaining = sorted(set(range(v104.FOLD_COUNT)) - {held})
    meta_val_fold = remaining[0]
    fit = np.flatnonzero((fold != held) & (fold != meta_val_fold)).astype(np.int64)
    val = np.flatnonzero(fold == meta_val_fold).astype(np.int64)
    x_fit, mean, std = v104._standardize_fit(features[fit])
    x_val = v104._standardize_apply(features[val], mean, std)
    birth_pos, mult_w = _weight_spec(k[fit], variant)

    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = _build_model(features.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, min_delta=2e-4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5),
    ]
    hist = model.fit(
        _inputs(x_fit, anchor[fit], p102[fit]),
        _targets(k[fit]),
        sample_weight=_sample_weights(k[fit], birth_pos, mult_w),
        validation_data=(
            _inputs(x_val, anchor[val], p102[val]),
            _targets(k[val]),
            _sample_weights(k[val], birth_pos, mult_w),
        ),
        epochs=MAX_META_EPOCHS,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    selected_epochs = max(2, int(np.argmin(np.asarray(hist.history["val_loss"], dtype=np.float64)) + 1))
    pv = _probabilities(model, x_val, anchor[val], p102[val])
    pred = np.argmax(pv, axis=1).astype(np.int32)
    f1 = float(_metrics(cache, train_split, val, pred)["global"]["f1"])
    card = _card(k[val], pred)
    return {
        "variant": variant,
        "meta_validation_fold": int(meta_val_fold),
        "meta_fit_clusters": int(len(fit)),
        "meta_validation_clusters": int(len(val)),
        "selected_epochs": int(selected_epochs),
        "meta_validation_f1": f1,
        "meta_validation_cardinality": card,
        "birth_positive_weight": float(birth_pos),
        "multiplicity_class_weights": [float(x) for x in mult_w],
    }


def _fit_outer(features, anchor, p102, k, fold, held, selection, seed):
    import tensorflow as tf

    fit = np.flatnonzero(fold != held).astype(np.int64)
    val = np.flatnonzero(fold == held).astype(np.int64)
    x_fit, mean, std = v104._standardize_fit(features[fit])
    x_val = v104._standardize_apply(features[val], mean, std)
    birth_pos, mult_w = _weight_spec(k[fit], selection["variant"])
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = _build_model(features.shape[1])
    model.fit(
        _inputs(x_fit, anchor[fit], p102[fit]),
        _targets(k[fit]),
        sample_weight=_sample_weights(k[fit], birth_pos, mult_w),
        epochs=int(selection["selected_epochs"]),
        batch_size=128,
        shuffle=True,
        verbose=0,
    )
    p = _probabilities(model, x_val, anchor[val], p102[val])
    return val, np.argmax(p, axis=1).astype(np.int32), p


def _fit_v104_baseline(features, anchor, p102, k, fold, held, seed):
    import tensorflow as tf

    fit = np.flatnonzero(fold != held).astype(np.int64)
    val = np.flatnonzero(fold == held).astype(np.int64)
    x_fit, mean, std = v104._standardize_fit(features[fit])
    x_val = v104._standardize_apply(features[val], mean, std)
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model, _, _ = v104._build_fusion(features.shape[1])
    model.fit(
        v104._inputs(x_fit, anchor[fit], p102[fit]),
        np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[fit]],
        sample_weight=v104._mild_count_weights(k[fit]),
        epochs=BASELINE_V104_EPOCHS,
        batch_size=128,
        shuffle=True,
        verbose=0,
    )
    p = np.asarray(model.predict(v104._inputs(x_val, anchor[val], p102[val]), batch_size=256, verbose=0))
    return val, np.argmax(p, axis=1).astype(np.int32)


def _slice(cache, train_split, k, idx, preds):
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return None
    out = {"clusters": int(len(idx))}
    for name, pred_all in preds.items():
        out[name] = {
            "metrics": _metrics(cache, train_split, idx, np.asarray(pred_all)[idx]),
            "cardinality": _card(k[idx], np.asarray(pred_all)[idx]),
        }
    return out


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if set(cache["track_members"]) != train_members:
        raise RuntimeError("cache/train split mismatch")
    if train_members & validation_members:
        raise RuntimeError("train/validation leakage")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    oof, oof_paths, group_folds = v104._load_oof(args.oof_dir, len(k))
    if not np.array_equal(k, np.asarray(oof["k"], dtype=np.int32)):
        raise RuntimeError("OOF labels differ from frozen cache")
    fold = np.asarray(oof["fold"], dtype=np.int16)
    features = np.asarray(oof["features"], dtype=np.float32)
    anchor = np.asarray(oof["anchor"], dtype=np.float32)
    p102 = np.asarray(oof["p102"], dtype=np.float32)

    pred11 = np.full(len(k), -1, dtype=np.int32)
    pred104 = np.full(len(k), -1, dtype=np.int32)
    fold_reports = {}
    for held in range(v104.FOLD_COUNT):
        candidates = []
        for vi, variant in enumerate(BIRTH_VARIANTS):
            candidates.append(_train_meta_candidate(
                cache, train_split, k, features, anchor, p102, fold, held, variant,
                args.seed + held * 100 + vi,
            ))
        # Selection metric is the actual downstream anonymous onset F1, not count CE.
        selection = max(candidates, key=lambda x: (x["meta_validation_f1"], -x["selected_epochs"]))
        val, pv11, _ = _fit_outer(
            features, anchor, p102, k, fold, held, selection, args.seed + 1000 + held
        )
        val104, pv104 = _fit_v104_baseline(
            features, anchor, p102, k, fold, held, args.seed + 2000 + held
        )
        if not np.array_equal(val, val104):
            raise RuntimeError("V11/V10.4 held index mismatch")
        pred11[val] = pv11
        pred104[val] = pv104
        fold_reports[str(held)] = {
            "clusters": int(len(val)),
            "meta_candidates": candidates,
            "selected": selection,
            "v101": {"metrics": _metrics(cache, train_split, val, np.asarray(oof["pred101"])[val]), "cardinality": _card(k[val], np.asarray(oof["pred101"])[val])},
            "v102": {"metrics": _metrics(cache, train_split, val, np.asarray(oof["pred102"])[val]), "cardinality": _card(k[val], np.asarray(oof["pred102"])[val])},
            "v104": {"metrics": _metrics(cache, train_split, val, pv104), "cardinality": _card(k[val], pv104)},
            "v11": {"metrics": _metrics(cache, train_split, val, pv11), "cardinality": _card(k[val], pv11)},
        }
        print(json.dumps({
            "held": held,
            "variant": selection["variant"],
            "epochs": selection["selected_epochs"],
            "meta_f1": selection["meta_validation_f1"],
            "v104_f1": fold_reports[str(held)]["v104"]["metrics"]["global"]["f1"],
            "v11_f1": fold_reports[str(held)]["v11"]["metrics"]["global"]["f1"],
        }, sort_keys=True))

    if np.any(pred11 < 0) or np.any(pred104 < 0):
        raise RuntimeError("crossfit predictions incomplete")

    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    players = np.asarray([_player(x) for x in members], dtype="U2")
    modes = np.asarray([_mode(x) for x in members], dtype="U8")
    by_member = {t.annotation_member: t for t in train_split}
    groups = np.asarray([group_stem(by_member[m]) for m in members], dtype="U64")
    genres = np.asarray([_genre(g) for g in groups], dtype="U16")

    preds = {
        "v101": np.asarray(oof["pred101"], dtype=np.int32),
        "v102": np.asarray(oof["pred102"], dtype=np.int32),
        "v104": pred104,
        "v11": pred11,
    }
    all_idx = np.arange(len(k), dtype=np.int64)
    strata = {
        "aggregate": all_idx,
        "player00": np.flatnonzero(players == "00").astype(np.int64),
        "player00_comp": np.flatnonzero((players == "00") & (modes == "comp")).astype(np.int64),
        "player00_solo": np.flatnonzero((players == "00") & (modes == "solo")).astype(np.int64),
        "player00_rock": np.flatnonzero((players == "00") & (genres == "Rock")).astype(np.int64),
        "player00_rock_comp": np.flatnonzero((players == "00") & (genres == "Rock") & (modes == "comp")).astype(np.int64),
    }
    for player in ALLOWED_PLAYERS:
        strata[f"player{player}"] = np.flatnonzero(players == player).astype(np.int64)

    stratum_report = {name: _slice(cache, train_split, k, idx, preds) for name, idx in strata.items()}
    per_k = {}
    for kk in range(SLOT_COUNT + 1):
        idx = np.flatnonzero(k == kk).astype(np.int64)
        row = {"clusters": int(len(idx))}
        for name, p in preds.items():
            pp = p[idx]
            row[name] = {
                "exact": float(np.mean(pp == kk)),
                "under_rate": float(np.mean(pp < kk)),
                "over_rate": float(np.mean(pp > kk)),
                "mean_error": float(np.mean(pp - kk)),
            }
        per_k[str(kk)] = row

    result = {
        "schema_version": 1,
        "protocol": {
            "train_only_strict_crossfit": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "held_fold_used_for_hyperparameter_or_epoch_selection": False,
            "multiplicity_loss_receives_k0_weight": False,
            "selection_metric": "inner-fold anonymous onset F1",
            "grouping_window_ms_unchanged": 40,
            "offset_model_untouched": True,
        },
        "architecture": {
            "factorization": "P(K=0)=1-P_birth; P(K=k)=P_birth*Q(k|birth), k=1..6",
            "feature_dim": int(features.shape[1]),
            "birth_variants": list(BIRTH_VARIANTS),
            "v104_baseline_epochs": BASELINE_V104_EPOCHS,
        },
        "data": {
            "clusters": int(len(k)),
            "train_tracks": int(len(train_split)),
            "validation_tracks_not_evaluated": int(len(validation)),
            "composition_groups": int(len(group_folds)),
            "oof_shards": [str(x) for x in oof_paths],
        },
        "folds": fold_reports,
        "strata": stratum_report,
        "per_true_k": per_k,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    def sf(s, model):
        return result["strata"][s][model]["metrics"]["global"]["f1"]
    print(json.dumps({
        "aggregate": {m: sf("aggregate", m) for m in preds},
        "player00": {m: sf("player00", m) for m in preds},
        "player00_rock_comp": {m: sf("player00_rock_comp", m) for m in preds},
        "v11_minus_v104": {
            "aggregate": sf("aggregate", "v11") - sf("aggregate", "v104"),
            "player00": sf("player00", "v11") - sf("player00", "v104"),
            "player00_rock_comp": sf("player00_rock_comp", "v11") - sf("player00_rock_comp", "v104"),
        },
    }, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--oof-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
