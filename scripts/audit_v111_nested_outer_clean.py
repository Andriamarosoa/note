"""Nested outer-clean audit for V11.1 hierarchical birth/multiplicity fusion.

This corrects an end-to-end leakage limitation in the earlier global-OOF V11
pilot.  For each outer composition fold, the fusion is trained only on the four
inner-OOF shards created by the completed V10.4 nested deployment audit.  Every
expert that produced those inner features excluded the outer fold entirely.
The frozen deployment experts for that outer fold are then used exactly once for
outer evaluation.

Historical validation and locked12 are never indexed or evaluated.
"""
from __future__ import annotations

import argparse
import json
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
from scripts import train_v104_oof_fold as oofmod
from scripts import train_v11_hierarchical_birth_multiplicity as v11

BIRTH_THRESHOLD = 0.5


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _concat(paths):
    parts = [_load_npz(p) for p in paths]
    keys = {"global_index", "k", "member", "features", "anchor", "p102", "pred101", "pred102", "held_fold"}
    for p in parts:
        if not keys.issubset(p):
            raise RuntimeError(f"nested shard missing keys: {sorted(keys - set(p))}")
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    order = np.argsort(merged["global_index"], kind="stable")
    return {k: v[order] for k, v in merged.items()}


def _decode(model, x, anchor, p102):
    out = model.predict(v11._inputs(x, anchor, p102), batch_size=256, verbose=0)
    birth = np.asarray(out["birth"], dtype=np.float64).reshape(-1)
    q = np.asarray(out["multiplicity"], dtype=np.float64)
    if q.shape != (len(birth), SLOT_COUNT):
        raise RuntimeError(f"unexpected multiplicity shape {q.shape}")
    pred = np.zeros(len(birth), dtype=np.int32)
    pos = birth >= BIRTH_THRESHOLD
    pred[pos] = 1 + np.argmax(q[pos], axis=1).astype(np.int32)
    return pred, birth, q


def _metrics(cache, train_split, global_idx, pred):
    return v104._metrics_for_indices(
        cache, train_split, np.asarray(global_idx, dtype=np.int64), np.asarray(pred, dtype=np.int32)
    )


def _card(k, pred):
    return v102._cardinality_report(np.asarray(k, dtype=np.int32), np.asarray(pred, dtype=np.int32))


def _train_candidate(cache, train_split, inner, fit_local, val_local, variant, seed):
    import tensorflow as tf
    from tensorflow import keras

    features = np.asarray(inner["features"], dtype=np.float32)
    anchor = np.asarray(inner["anchor"], dtype=np.float32)
    p102 = np.asarray(inner["p102"], dtype=np.float32)
    k = np.asarray(inner["k"], dtype=np.int32)
    global_idx = np.asarray(inner["global_index"], dtype=np.int64)

    x_fit, mean, std = v104._standardize_fit(features[fit_local])
    x_val = v104._standardize_apply(features[val_local], mean, std)
    birth_pos, mult_w = v11._weight_spec(k[fit_local], variant)

    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = v11._build_model(features.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, min_delta=2e-4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5
        ),
    ]
    hist = model.fit(
        v11._inputs(x_fit, anchor[fit_local], p102[fit_local]),
        v11._targets(k[fit_local]),
        sample_weight=v11._sample_weights(k[fit_local], birth_pos, mult_w),
        validation_data=(
            v11._inputs(x_val, anchor[val_local], p102[val_local]),
            v11._targets(k[val_local]),
            v11._sample_weights(k[val_local], birth_pos, mult_w),
        ),
        epochs=v11.MAX_META_EPOCHS,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    epochs = max(2, int(np.argmin(np.asarray(hist.history["val_loss"], dtype=np.float64)) + 1))
    pred, birth, _ = _decode(model, x_val, anchor[val_local], p102[val_local])
    m = _metrics(cache, train_split, global_idx[val_local], pred)
    return {
        "variant": variant,
        "selected_epochs": int(epochs),
        "meta_validation_f1": float(m["global"]["f1"]),
        "meta_validation_precision": float(m["global"]["precision"]),
        "meta_validation_recall": float(m["global"]["recall"]),
        "meta_validation_cardinality": _card(k[val_local], pred),
        "birth_positive_weight": float(birth_pos),
        "mean_birth_probability": float(np.mean(birth)),
    }


def _fit_final(inner, selection, deploy, seed):
    import tensorflow as tf

    features = np.asarray(inner["features"], dtype=np.float32)
    anchor = np.asarray(inner["anchor"], dtype=np.float32)
    p102 = np.asarray(inner["p102"], dtype=np.float32)
    k = np.asarray(inner["k"], dtype=np.int32)
    x_all, mean, std = v104._standardize_fit(features)
    birth_pos, mult_w = v11._weight_spec(k, selection["variant"])

    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = v11._build_model(features.shape[1])
    model.fit(
        v11._inputs(x_all, anchor, p102),
        v11._targets(k),
        sample_weight=v11._sample_weights(k, birth_pos, mult_w),
        epochs=int(selection["selected_epochs"]),
        batch_size=128,
        shuffle=True,
        verbose=0,
    )
    x_outer = v104._standardize_apply(np.asarray(deploy["features"], dtype=np.float32), mean, std)
    return _decode(model, x_outer, deploy["anchor"], deploy["p102"])


def _player(member: str):
    p = str(member).split("_", 1)[0]
    if p not in ALLOWED_PLAYERS:
        raise RuntimeError(f"unexpected player {p}")
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


def audit(args):
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

    assignment, groups_per_fold, _, _ = oofmod._balanced_group_folds(cache, train_split)
    by_member = {t.annotation_member: t for t in train_split}
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    row_fold = np.asarray(
        [assignment[group_stem(by_member[m])] for m in members], dtype=np.int16
    )
    k_all = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)

    pred111 = np.full(len(k_all), -1, dtype=np.int32)
    pred104 = np.full(len(k_all), -1, dtype=np.int32)
    pred101 = np.full(len(k_all), -1, dtype=np.int32)
    pred102 = np.full(len(k_all), -1, dtype=np.int32)
    birth_all = np.full(len(k_all), np.nan, dtype=np.float64)
    fold_reports = {}

    for outer in range(oofmod.FOLD_COUNT):
        outer_idx = np.flatnonzero(row_fold == outer).astype(np.int64)
        outer_train_idx = np.flatnonzero(row_fold != outer).astype(np.int64)
        inner_paths = sorted(args.inner_dir.glob(f"**/v104-nested-outer-{outer}-inner-*.npz"))
        if len(inner_paths) != 4:
            raise RuntimeError(f"outer {outer}: expected 4 inner shards, found {len(inner_paths)}")
        inner = _concat(inner_paths)
        if not np.array_equal(np.asarray(inner["global_index"], dtype=np.int64), outer_train_idx):
            raise RuntimeError(f"outer {outer}: inner OOF coverage mismatch")
        if not np.array_equal(np.asarray(inner["k"], dtype=np.int32), k_all[outer_train_idx]):
            raise RuntimeError(f"outer {outer}: inner labels mismatch")

        held = np.asarray(inner["held_fold"], dtype=np.int16)
        inner_ids = sorted(set(held.tolist()))
        expected = sorted(set(range(oofmod.FOLD_COUNT)) - {outer})
        if inner_ids != expected:
            raise RuntimeError(f"outer {outer}: inner IDs {inner_ids} != {expected}")
        meta_fold = inner_ids[0]
        meta_val = np.flatnonzero(held == meta_fold).astype(np.int64)
        meta_fit = np.flatnonzero(held != meta_fold).astype(np.int64)

        candidates = []
        for vi, variant in enumerate(v11.BIRTH_VARIANTS):
            candidates.append(_train_candidate(
                cache, train_split, inner, meta_fit, meta_val, variant,
                args.seed + outer * 100 + vi,
            ))
        selection = max(candidates, key=lambda x: (x["meta_validation_f1"], -x["selected_epochs"]))

        deploy_paths = sorted(args.outer_dir.glob(f"**/v104-nested-outer-{outer}-deploy.npz"))
        if len(deploy_paths) != 1:
            raise RuntimeError(f"outer {outer}: expected one deployment shard, found {len(deploy_paths)}")
        deploy = _load_npz(deploy_paths[0])
        if not np.array_equal(np.asarray(deploy["global_index"], dtype=np.int64), outer_idx):
            raise RuntimeError(f"outer {outer}: deployment coverage mismatch")
        if not np.array_equal(np.asarray(deploy["k"], dtype=np.int32), k_all[outer_idx]):
            raise RuntimeError(f"outer {outer}: deployment labels mismatch")

        p111, birth, _ = _fit_final(inner, selection, deploy, args.seed + 1000 + outer)
        pred111[outer_idx] = p111
        birth_all[outer_idx] = birth
        pred101[outer_idx] = np.asarray(deploy["pred101"], dtype=np.int32)
        pred102[outer_idx] = np.asarray(deploy["pred102"], dtype=np.int32)

        eval_paths = sorted(args.eval_dir.glob(f"**/v104-nested-eval-{outer}.npz"))
        if len(eval_paths) != 1:
            raise RuntimeError(f"outer {outer}: expected one V10.4 eval shard, found {len(eval_paths)}")
        ev = _load_npz(eval_paths[0])
        if not np.array_equal(np.asarray(ev["global_index"], dtype=np.int64), outer_idx):
            raise RuntimeError(f"outer {outer}: V10.4 eval coverage mismatch")
        pred104[outer_idx] = np.asarray(ev["pred104_deploy"], dtype=np.int32)

        ko = k_all[outer_idx]
        fold_reports[str(outer)] = {
            "outer_groups": groups_per_fold[outer],
            "outer_clusters": int(len(outer_idx)),
            "meta_validation_inner_fold": int(meta_fold),
            "meta_candidates": candidates,
            "selected": selection,
            "v101": {"metrics": _metrics(cache, train_split, outer_idx, pred101[outer_idx]), "cardinality": _card(ko, pred101[outer_idx])},
            "v102": {"metrics": _metrics(cache, train_split, outer_idx, pred102[outer_idx]), "cardinality": _card(ko, pred102[outer_idx])},
            "v104": {"metrics": _metrics(cache, train_split, outer_idx, pred104[outer_idx]), "cardinality": _card(ko, pred104[outer_idx])},
            "v111": {"metrics": _metrics(cache, train_split, outer_idx, p111), "cardinality": _card(ko, p111)},
            "birth": {
                "mean_probability": float(np.mean(birth)),
                "positive_rate_at_050": float(np.mean(birth >= BIRTH_THRESHOLD)),
            },
        }
        print(json.dumps({
            "outer": outer,
            "variant": selection["variant"],
            "epochs": selection["selected_epochs"],
            "v104_f1": fold_reports[str(outer)]["v104"]["metrics"]["global"]["f1"],
            "v111_f1": fold_reports[str(outer)]["v111"]["metrics"]["global"]["f1"],
        }, sort_keys=True))

    if np.any(pred111 < 0) or np.any(pred104 < 0) or np.any(pred101 < 0) or np.any(pred102 < 0):
        raise RuntimeError("nested predictions incomplete")
    if np.any(~np.isfinite(birth_all)):
        raise RuntimeError("birth probabilities incomplete")

    players = np.asarray([_player(m) for m in members], dtype="U2")
    modes = np.asarray([_mode(m) for m in members], dtype="U8")
    groups = np.asarray([group_stem(by_member[m]) for m in members], dtype="U64")
    genres = np.asarray([_genre(g) for g in groups], dtype="U16")
    preds = {"v101": pred101, "v102": pred102, "v104": pred104, "v111": pred111}

    def slice_report(idx):
        idx = np.asarray(idx, dtype=np.int64)
        out = {"clusters": int(len(idx))}
        for name, p in preds.items():
            out[name] = {
                "metrics": _metrics(cache, train_split, idx, p[idx]),
                "cardinality": _card(k_all[idx], p[idx]),
            }
        out["birth"] = {
            "mean_probability": float(np.mean(birth_all[idx])),
            "positive_rate_at_050": float(np.mean(birth_all[idx] >= BIRTH_THRESHOLD)),
            "true_birth_rate": float(np.mean(k_all[idx] > 0)),
        }
        return out

    all_idx = np.arange(len(k_all), dtype=np.int64)
    strata_idx = {
        "aggregate": all_idx,
        "player00": np.flatnonzero(players == "00").astype(np.int64),
        "player00_comp": np.flatnonzero((players == "00") & (modes == "comp")).astype(np.int64),
        "player00_solo": np.flatnonzero((players == "00") & (modes == "solo")).astype(np.int64),
        "player00_rock": np.flatnonzero((players == "00") & (genres == "Rock")).astype(np.int64),
        "player00_rock_comp": np.flatnonzero((players == "00") & (genres == "Rock") & (modes == "comp")).astype(np.int64),
    }
    for player in ALLOWED_PLAYERS:
        strata_idx[f"player{player}"] = np.flatnonzero(players == player).astype(np.int64)
    strata = {name: slice_report(idx) for name, idx in strata_idx.items()}

    per_k = {}
    for kk in range(SLOT_COUNT + 1):
        idx = np.flatnonzero(k_all == kk).astype(np.int64)
        row = {"clusters": int(len(idx))}
        for name, p in preds.items():
            pp = p[idx]
            row[name] = {
                "exact": float(np.mean(pp == kk)),
                "under_rate": float(np.mean(pp < kk)),
                "over_rate": float(np.mean(pp > kk)),
                "mean_error": float(np.mean(pp - kk)),
            }
        row["v111_birth_mean"] = float(np.mean(birth_all[idx]))
        row["v111_birth_positive_rate_050"] = float(np.mean(birth_all[idx] >= BIRTH_THRESHOLD))
        per_k[str(kk)] = row

    result = {
        "schema_version": 1,
        "protocol": {
            "train_only_nested_outer_holdout": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "outer_fold_seen_by_any_inner_expert_training": False,
            "outer_fold_seen_by_v111_training": False,
            "outer_fold_seen_by_v111_variant_or_epoch_selection": False,
            "deployment_experts_train_on_all_four_outer_train_folds": True,
            "multiplicity_loss_receives_k0_weight": False,
            "birth_threshold": BIRTH_THRESHOLD,
            "birth_threshold_tuned": False,
            "grouping_window_ms_unchanged": 40,
            "offset_model_untouched": True,
        },
        "architecture": {
            "decoder": "if P_birth < 0.5: K=0; else K=1+argmax Q(K|birth)",
            "birth_variants": list(v11.BIRTH_VARIANTS),
        },
        "data": {
            "clusters": int(len(k_all)),
            "train_tracks": int(len(train_split)),
            "validation_tracks_not_evaluated": int(len(validation)),
        },
        "folds": fold_reports,
        "strata": strata,
        "per_true_k": per_k,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=all_idx,
        k=k_all.astype(np.int16),
        member=members,
        pred101=pred101.astype(np.int16),
        pred102=pred102.astype(np.int16),
        pred104=pred104.astype(np.int16),
        pred111=pred111.astype(np.int16),
        birth111=birth_all.astype(np.float32),
    )
    print(json.dumps({
        "aggregate": {m: strata["aggregate"][m]["metrics"]["global"]["f1"] for m in preds},
        "player00": {m: strata["player00"][m]["metrics"]["global"]["f1"] for m in preds},
        "player00_rock_comp": {m: strata["player00_rock_comp"][m]["metrics"]["global"]["f1"] for m in preds},
    }, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--inner-dir", type=Path, required=True)
    p.add_argument("--outer-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=11131)
    return p


def main(argv: Optional[Sequence[str]] = None):
    audit(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
