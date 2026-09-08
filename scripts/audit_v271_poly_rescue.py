"""V27.1 nested polyphonic rescue on top of the frozen V27 low-K fusion.

The V27 reference already delegates every V10.4 K<=1 row to the V26-uniform
count head. V27.1 therefore uses the other V26 arm (class-weighted) only on
rows that remain K<=1 after low_k_fusion. The weighted arm may lift such a row
to K>=2 when its exact predicted-class probability clears a threshold selected
inside the outer-training partition.

One threshold is selected independently for each outer fold. The untouched
outer labels are never used during threshold selection. The selection order is
fixed before outer evaluation: maximize inner polyphonic exact-K, then inner
global exact-K, then prefer fewer rescues / the higher threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from causal_note.guitarset import SLOT_COUNT
from scripts import audit_v270_class_conditional_fusion as v270
from scripts import train_v104_class_conditional_fusion as v104
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v260_count_weighting as v260

FOLD_COUNT = 5
V104_NESTED_RUN = v270.V104_NESTED_RUN
V104_NESTED_SHA = v270.V104_NESTED_SHA
V111_AUDIT_RUN = v270.V111_AUDIT_RUN
V111_AUDIT_SHA = v270.V111_AUDIT_SHA
V260_RUN = v270.V260_RUN
V260_SHA = v270.V260_SHA
V104_CALIBRATION_SEED = 10521
NO_RESCUE_THRESHOLD = 1.000001
EXPECTED_ROWS = 76768
EXPECTED_LOW_K_EVENT_COUNTS = (33409, 5696, 10811)
EXPECTED_LOW_K_EXACT_PERCENT = 82.4953
EXPECTED_LOW_K_POLY_EXACT_PERCENT = 37.5705


class V271Error(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one(paths, label: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise V271Error(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _validate_probability(probability, label: str) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != SLOT_COUNT + 1:
        raise V271Error(f"invalid {label} shape {p.shape}")
    if not np.isfinite(p).all() or np.any(p < -1e-7):
        raise V271Error(f"invalid {label} values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise V271Error(f"{label} is not normalized")
    return p.astype(np.float32)


def exact_class_confidence(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return argmax K and P(K=argmax), the confidence used by V27.1."""
    p = _validate_probability(probability, "V26-weighted probability")
    pred = np.argmax(p, axis=1).astype(np.int32)
    conf = p[np.arange(len(p)), pred].astype(np.float64)
    return pred, conf


def apply_poly_rescue(anchor, low_k, weighted_probability, threshold: float):
    """Apply the frozen V27.1 rule without consulting labels."""
    anchor = v270._prediction(anchor, "V10.4 anchor")
    low_k = v270._prediction(low_k, "V27 low_k_fusion")
    if anchor.shape != low_k.shape:
        raise V271Error("anchor/low-k shape mismatch")
    weighted_pred, confidence = exact_class_confidence(weighted_probability)
    if weighted_pred.shape != anchor.shape:
        raise V271Error("weighted/anchor shape mismatch")
    if not np.isfinite(threshold):
        raise V271Error("threshold must be finite")
    eligible = (anchor <= 1) & (low_k <= 1) & (weighted_pred >= 2)
    activated = eligible & (confidence >= float(threshold))
    rescue = low_k.copy()
    rescue[activated] = weighted_pred[activated]
    return rescue.astype(np.int32), activated, weighted_pred, confidence


def _threshold_candidates(confidence: np.ndarray, eligible: np.ndarray) -> list[float]:
    values = np.asarray(confidence, dtype=np.float64)[np.asarray(eligible, dtype=bool)]
    return [NO_RESCUE_THRESHOLD, *sorted(set(float(x) for x in values), reverse=True)]


def select_threshold(k, anchor, low_k, weighted_probability) -> dict:
    """Select one threshold using inner labels only and a predeclared ordering."""
    k = v270._prediction(k, "inner truth")
    anchor = v270._prediction(anchor, "inner V10.4")
    low_k = v270._prediction(low_k, "inner low_k_fusion")
    weighted_pred, confidence = exact_class_confidence(weighted_probability)
    if not (k.shape == anchor.shape == low_k.shape == weighted_pred.shape):
        raise V271Error("inner calibration shape mismatch")
    eligible = (anchor <= 1) & (low_k <= 1) & (weighted_pred >= 2)
    baseline = v270.cardinality_report(k, low_k)
    best = None
    evaluated = 0
    for threshold in _threshold_candidates(confidence, eligible):
        pred, active, _, _ = apply_poly_rescue(anchor, low_k, weighted_probability, threshold)
        card = v270.cardinality_report(k, pred)
        activated = int(active.sum())
        key = (
            int(card["poly_correct"]),
            int(card["correct"]),
            -activated,
            float(threshold),
        )
        evaluated += 1
        if best is None or key > best[0]:
            best = (key, float(threshold), pred, active, card)
    assert best is not None
    _, threshold, pred, active, card = best
    transition = v270.transition_report(k, low_k, pred)
    return {
        "threshold": threshold,
        "eligible_rows": int(eligible.sum()),
        "activated_rows": int(active.sum()),
        "candidate_thresholds_evaluated": evaluated,
        "baseline": baseline,
        "selected": card,
        "transition_from_low_k_fusion": transition,
        "score": "P_v260_weighted(K=argmax)",
        "selection_order": [
            "maximize poly_correct",
            "maximize global correct",
            "minimize activated rescues",
            "prefer higher threshold",
        ],
    }


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def _load_inner_v104(inner_dir: Path, outer_fold: int, final_idx, meta_val_idx, k, members):
    paths = sorted(inner_dir.glob(f"**/v104-nested-outer-{outer_fold}-inner-*.npz"))
    if len(paths) != 4:
        raise V271Error(f"outer {outer_fold}: expected four V10.4 inner shards, found {len(paths)}")
    required = {"global_index", "k", "member", "features", "anchor", "p102", "held_fold"}
    parts = []
    for path in paths:
        part = _load_npz(path)
        if not required.issubset(part):
            raise V271Error(f"inner V10.4 schema mismatch: {path}")
        parts.append({key: part[key] for key in required})
    merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in required}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    idx = np.asarray(merged["global_index"], dtype=np.int64)
    final_idx = np.asarray(final_idx, dtype=np.int64)
    meta_val_idx = np.asarray(meta_val_idx, dtype=np.int64)
    if not np.array_equal(idx, final_idx):
        raise V271Error(f"outer {outer_fold}: inner V10.4 coverage mismatch")
    if not np.array_equal(np.asarray(merged["k"], dtype=np.int32), np.asarray(k)[final_idx]):
        raise V271Error(f"outer {outer_fold}: inner V10.4 labels mismatch")
    if not np.array_equal(np.asarray(merged["member"]).astype(str), np.asarray(members)[final_idx].astype(str)):
        raise V271Error(f"outer {outer_fold}: inner V10.4 members mismatch")
    held = np.asarray(merged["held_fold"], dtype=np.int16)
    remaining = sorted(set(range(FOLD_COUNT)) - {outer_fold})
    if sorted(set(held.tolist())) != remaining:
        raise V271Error(f"outer {outer_fold}: inner held-fold IDs mismatch")
    meta_fold = remaining[0]
    if not np.array_equal(idx[held == meta_fold], meta_val_idx):
        raise V271Error(f"outer {outer_fold}: V10.4/V26 meta-validation fold mismatch")
    return merged, paths, meta_fold


def _inner_v104_probability(inner: dict, meta_fold: int, seed: int = V104_CALIBRATION_SEED):
    import tensorflow as tf
    from tensorflow import keras

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    held = np.asarray(inner["held_fold"], dtype=np.int16)
    fit = np.flatnonzero(held != meta_fold).astype(np.int64)
    val = np.flatnonzero(held == meta_fold).astype(np.int64)
    features = np.asarray(inner["features"], dtype=np.float32)
    anchor = _validate_probability(inner["anchor"], "V10.1 anchor")
    p102 = _validate_probability(inner["p102"], "V10.2 probability")
    k = np.asarray(inner["k"], dtype=np.int32)
    x_fit, mean, std = v104._standardize_fit(features[fit])
    x_val = v104._standardize_apply(features[val], mean, std)
    model, _, _ = v104._build_fusion(features.shape[1])
    history = model.fit(
        v104._inputs(x_fit, anchor[fit], p102[fit]),
        np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[fit]],
        sample_weight=v104._mild_count_weights(k[fit]),
        validation_data=(
            v104._inputs(x_val, anchor[val], p102[val]),
            np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[val]],
            v104._mild_count_weights(k[val]),
        ),
        epochs=v104.MAX_META_EPOCHS,
        batch_size=128,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, min_delta=2e-4, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5
            ),
            keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    )
    loss = np.asarray(history.history.get("val_loss", []), dtype=np.float64)
    if len(loss) == 0 or not np.isfinite(loss).all():
        raise V271Error("invalid inner V10.4 validation history")
    selected_epochs = int(np.argmin(loss)) + 1
    probability = np.asarray(
        model.predict(v104._inputs(x_val, anchor[val], p102[val]), batch_size=256, verbose=0)
    )
    return _validate_probability(probability, "inner V10.4 probability"), selected_epochs


def _inner_v260_probability(ctx: dict, arm: str, outer_fold: int, expected_epochs: int):
    import tensorflow as tf

    _, fit, val, _ = v260.validate_partitions(ctx)
    k = np.asarray(ctx["k"], dtype=np.int32)
    table = v260.arm_weights(arm, k[fit])
    model = v260.build_model(arm, v260.SEED + 100 + outer_fold)
    history = model.fit(
        v260.v102._inputs(ctx["cache"], fit),
        k[fit],
        sample_weight=table[k[fit]],
        validation_data=(v260.v102._inputs(ctx["cache"], val), k[val], table[k[val]]),
        epochs=v260.MAX_EPOCHS,
        batch_size=128,
        shuffle=True,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, min_delta=0, restore_best_weights=True
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
    )
    loss = np.asarray(history.history.get("val_loss", []), dtype=np.float64)
    if len(loss) == 0 or not np.isfinite(loss).all():
        raise V271Error(f"invalid inner V26 {arm} validation history")
    selected_epochs = int(np.argmin(loss)) + 1
    if selected_epochs != int(expected_epochs):
        raise V271Error(
            f"V26 {arm} inner replay epoch mismatch: {selected_epochs} != {expected_epochs}"
        )
    probability = np.asarray(model.predict(v260.v102._inputs(ctx["cache"], val), batch_size=128, verbose=0))
    return _validate_probability(probability, f"inner V26 {arm} probability"), selected_epochs


def _load_v260_fold(source_dir: Path, fold: int, expected_outer, k, members):
    prediction_path = _one(source_dir.glob(f"**/predictions-fold-{fold}.npz"), f"V26 fold {fold} predictions")
    report_path = _one(source_dir.glob(f"**/report-fold-{fold}.json"), f"V26 fold {fold} report")
    report = json.loads(report_path.read_text())
    if report.get("fold") != fold or report.get("protocol", {}).get("experiment") != "v260_count_weighting_ab":
        raise V271Error(f"V26 fold {fold} report mismatch")
    with np.load(prediction_path, allow_pickle=False) as z:
        required = {"global_index", "k", "member", "uniform_probability", "weighted_probability"}
        if not required.issubset(z.files):
            raise V271Error(f"V26 fold {fold} prediction schema mismatch")
        idx = np.asarray(z["global_index"], dtype=np.int64)
        truth = np.asarray(z["k"], dtype=np.int32)
        saved_members = np.asarray(z["member"]).astype(str)
        uniform = _validate_probability(z["uniform_probability"], "outer V26-uniform probability")
        weighted = _validate_probability(z["weighted_probability"], "outer V26-weighted probability")
    expected_outer = np.asarray(expected_outer, dtype=np.int64)
    if not np.array_equal(idx, expected_outer):
        raise V271Error(f"V26 fold {fold} outer index mismatch")
    if not np.array_equal(truth, np.asarray(k)[expected_outer]):
        raise V271Error(f"V26 fold {fold} labels mismatch")
    if not np.array_equal(saved_members, np.asarray(members)[expected_outer].astype(str)):
        raise V271Error(f"V26 fold {fold} members mismatch")
    return uniform, weighted, report, prediction_path, report_path


def _transition_by_anchor(k, anchor, low_k, rescue, active) -> dict:
    k = np.asarray(k, dtype=np.int32)
    anchor = np.asarray(anchor, dtype=np.int32)
    low_k = np.asarray(low_k, dtype=np.int32)
    rescue = np.asarray(rescue, dtype=np.int32)
    active = np.asarray(active, dtype=bool)
    out = {}
    for aa in (0, 1):
        mask = anchor == aa
        changed = mask & active
        out[str(aa)] = {
            "rows": int(mask.sum()),
            "activated": int(changed.sum()),
            "corrected": int(np.sum(changed & (low_k != k) & (rescue == k))),
            "regressed": int(np.sum(changed & (low_k == k) & (rescue != k))),
            "poly_corrected": int(np.sum(changed & (k >= 2) & (rescue == k))),
        }
    return out


def _event_counts(metric: dict) -> tuple[int, int, int]:
    return tuple(int(metric[key]) for key in ("true_positive", "false_positive", "false_negative"))


def run_fold(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(FOLD_COUNT):
        raise V271Error("invalid outer fold")

    ctx = v171._fold_context(args)
    outer, fit, val, final = v260.validate_partitions(ctx)
    cache = ctx["cache"]
    k = np.asarray(ctx["k"], dtype=np.int32)
    members = np.asarray(ctx["members"]).astype(str)
    train_members = {track.annotation_member for track in ctx["train_split"]}
    validation_members = {track.annotation_member for track in ctx["validation"]}
    if set(cache["track_members"]) != train_members or train_members & validation_members:
        raise V271Error("dataset/cache split mismatch")

    inner, inner_paths, meta_fold = _load_inner_v104(
        args.v104_inner_dir, args.outer_fold, final, val, k, members
    )
    inner_v104_probability, v104_epochs = _inner_v104_probability(inner, meta_fold)
    inner_anchor = np.argmax(inner_v104_probability, axis=1).astype(np.int32)

    uniform_outer, weighted_outer, v260_report, v260_prediction_path, v260_report_path = _load_v260_fold(
        args.v260_dir, args.outer_fold, outer, k, members
    )
    inner_weighted, weighted_epochs = _inner_v260_probability(
        ctx, "weighted", args.outer_fold, v260_report["arms"]["weighted"]["epochs"]
    )
    inner_uniform, uniform_epochs = _inner_v260_probability(
        ctx, "uniform", args.outer_fold, v260_report["arms"]["uniform"]["epochs"]
    )
    if len(inner_anchor) != len(val) or len(inner_uniform) != len(val) or len(inner_weighted) != len(val):
        raise V271Error("inner calibration row count mismatch")
    inner_uniform_pred = np.argmax(inner_uniform, axis=1).astype(np.int32)
    inner_low = v270.fuse_counts(inner_anchor, inner_uniform_pred, "low_k_fusion")
    calibration = select_threshold(k[val], inner_anchor, inner_low, inner_weighted)
    threshold = float(calibration["threshold"])

    v111, v111_report, v111_prediction_path, v111_report_path = v270._load_v111(args.v111_dir)
    expected_index = np.arange(len(k), dtype=np.int64)
    if not np.array_equal(np.asarray(v111["global_index"], dtype=np.int64), expected_index):
        raise V271Error("V11.1 source coverage mismatch")
    if not np.array_equal(np.asarray(v111["k"], dtype=np.int32), k):
        raise V271Error("V11.1 labels mismatch")
    if not np.array_equal(np.asarray(v111["member"]).astype(str), members):
        raise V271Error("V11.1 members mismatch")
    anchor_outer = v270._prediction(v111["pred104"][outer], "outer V10.4")
    uniform_outer_pred = np.argmax(uniform_outer, axis=1).astype(np.int32)
    low_outer = v270.fuse_counts(anchor_outer, uniform_outer_pred, "low_k_fusion")
    rescue_outer, active_outer, weighted_outer_pred, confidence_outer = apply_poly_rescue(
        anchor_outer, low_outer, weighted_outer, threshold
    )
    kout = k[outer]
    low_card = v270.cardinality_report(kout, low_outer)
    rescue_card = v270.cardinality_report(kout, rescue_outer)
    if rescue_card["poly_correct"] < low_card["poly_correct"]:
        raise V271Error("polyphonic structural guard violated")
    if np.any(active_outer & (low_outer >= 2)):
        raise V271Error("rescue changed an already-polyphonic low_k_fusion prediction")

    low_event = v104._metrics_for_indices(cache, ctx["train_split"], outer, low_outer)["global"]
    rescue_event = v104._metrics_for_indices(cache, ctx["train_split"], outer, rescue_outer)["global"]
    transition = v270.transition_report(kout, low_outer, rescue_outer)
    by_anchor = _transition_by_anchor(kout, anchor_outer, low_outer, rescue_outer, active_outer)

    report = {
        "schema_version": 1,
        "protocol": {
            "experiment": "v271_poly_rescue",
            "reference": "v270 low_k_fusion",
            "primary_metric": "polyphonic exact-K",
            "secondary_metrics": ["global exact-K", "event F1 at 50 ms", "TP", "FP", "FN"],
            "frozen_outer_v104_anchor": True,
            "frozen_outer_v260_uniform_reference_specialist": True,
            "frozen_outer_v260_weighted_rescue_specialist": True,
            "threshold_calibrated_inner_only": True,
            "outer_labels_used_for_threshold_selection": False,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "confidence_score": "P_v260_weighted(K=argmax)",
            "eligibility": "V104_K<=1 and low_k_fusion_K<=1 and V260_weighted_K>=2",
            "selection_order": calibration["selection_order"],
            "event_f1_used_for_threshold_selection": False,
            "candidate_identity_and_ranking": "frozen V10.4 cache/runtime ranking",
            "polyphonic_guard": "only low_k_fusion predictions <=1 can change",
            "automatic_promotion_rule": (
                "promote only if aggregate polyphonic exact-K is strictly above V27 low_k_fusion; "
                "global exact-K and F1 are reported secondary metrics, not vetoes"
            ),
        },
        "fold": int(args.outer_fold),
        "data": {
            "outer_rows": int(len(outer)),
            "inner_fit_rows": int(len(fit)),
            "inner_validation_rows": int(len(val)),
            "inner_meta_fold": int(meta_fold),
        },
        "inner_calibration": {
            **calibration,
            "v104_probe_selected_epochs": int(v104_epochs),
            "v260_weighted_replayed_epochs": int(weighted_epochs),
            "v260_uniform_replayed_epochs": int(uniform_epochs),
        },
        "outer": {
            "threshold": threshold,
            "low_k_fusion": low_card,
            "poly_rescue": rescue_card,
            "event_low_k_fusion": low_event,
            "event_poly_rescue": rescue_event,
            "transition_from_low_k_fusion": transition,
            "transition_by_v104_anchor": by_anchor,
            "activated_rows": int(active_outer.sum()),
        },
        "sources": {
            "v104_nested_run": V104_NESTED_RUN,
            "v104_nested_sha": V104_NESTED_SHA,
            "v111_audit_run": V111_AUDIT_RUN,
            "v111_audit_sha": V111_AUDIT_SHA,
            "v260_run": V260_RUN,
            "v260_sha": V260_SHA,
            "files": [
                *({"path": str(path), "sha256": digest(path)} for path in inner_paths),
                {"path": str(v111_prediction_path), "sha256": digest(v111_prediction_path)},
                {"path": str(v111_report_path), "sha256": digest(v111_report_path)},
                {"path": str(v260_prediction_path), "sha256": digest(v260_prediction_path)},
                {"path": str(v260_report_path), "sha256": digest(v260_report_path)},
            ],
        },
    }

    args.output_dir.mkdir(parents=True)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output_dir / f"predictions-fold-{args.outer_fold}.npz",
        outer_fold=np.full(len(outer), args.outer_fold, dtype=np.int16),
        global_index=outer,
        member=members[outer],
        k=kout.astype(np.int16),
        pred104=anchor_outer.astype(np.int16),
        pred260_uniform=uniform_outer_pred.astype(np.int16),
        pred260_weighted=weighted_outer_pred.astype(np.int16),
        pred270_low_k_fusion=low_outer.astype(np.int16),
        pred271_poly_rescue=rescue_outer.astype(np.int16),
        weighted_exact_confidence=confidence_outer.astype(np.float32),
        rescue_activated=active_outer.astype(np.int8),
        selected_threshold=np.asarray([threshold], dtype=np.float64),
    )
    print(json.dumps({
        "fold": args.outer_fold,
        "threshold": threshold,
        "inner_activated": calibration["activated_rows"],
        "outer_activated": int(active_outer.sum()),
        "low_poly_exact": low_card["poly_exact"],
        "rescue_poly_exact": rescue_card["poly_exact"],
        "low_exact": low_card["exact"],
        "rescue_exact": rescue_card["exact"],
        "low_f1": low_event["f1"],
        "rescue_f1": rescue_event["f1"],
    }, indent=2, sort_keys=True))
    return report


def _aggregate_event(metrics: list[dict]) -> dict:
    tp = sum(int(m["true_positive"]) for m in metrics)
    fp = sum(int(m["false_positive"]) for m in metrics)
    fn = sum(int(m["false_negative"]) for m in metrics)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize(args):
    reports = []
    arrays = []
    for fold in range(FOLD_COUNT):
        report_path = _one(args.input_dir.glob(f"**/report-fold-{fold}.json"), f"V27.1 fold {fold} report")
        pred_path = _one(args.input_dir.glob(f"**/predictions-fold-{fold}.npz"), f"V27.1 fold {fold} predictions")
        report = json.loads(report_path.read_text())
        if report.get("fold") != fold or report.get("protocol", {}).get("experiment") != "v271_poly_rescue":
            raise V271Error(f"invalid V27.1 fold report {fold}")
        arr = _load_npz(pred_path)
        if not np.all(np.asarray(arr["outer_fold"]) == fold):
            raise V271Error(f"fold marker mismatch {fold}")
        reports.append(report)
        arrays.append(arr)
    merged = {key: np.concatenate([a[key] for a in arrays], axis=0) for key in arrays[0] if key != "selected_threshold"}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if not np.array_equal(np.asarray(merged["global_index"], dtype=np.int64), np.arange(EXPECTED_ROWS, dtype=np.int64)):
        raise V271Error("outer predictions do not provide exact one-time coverage")
    k = np.asarray(merged["k"], dtype=np.int32)
    low = np.asarray(merged["pred270_low_k_fusion"], dtype=np.int32)
    rescue = np.asarray(merged["pred271_poly_rescue"], dtype=np.int32)
    low_card = v270.cardinality_report(k, low)
    rescue_card = v270.cardinality_report(k, rescue)
    if round(100.0 * low_card["exact"], 4) != EXPECTED_LOW_K_EXACT_PERCENT:
        raise V271Error("V27 low_k_fusion exact-K baseline failed to reproduce")
    if round(100.0 * low_card["poly_exact"], 4) != EXPECTED_LOW_K_POLY_EXACT_PERCENT:
        raise V271Error("V27 low_k_fusion poly exact-K baseline failed to reproduce")
    if rescue_card["poly_correct"] < low_card["poly_correct"]:
        raise V271Error("aggregate polyphonic structural guard violated")
    for report in reports:
        if report["outer"]["poly_rescue"]["poly_correct"] < report["outer"]["low_k_fusion"]["poly_correct"]:
            raise V271Error(f"fold {report['fold']} polyphonic structural guard violated")

    low_event = _aggregate_event([r["outer"]["event_low_k_fusion"] for r in reports])
    rescue_event = _aggregate_event([r["outer"]["event_poly_rescue"] for r in reports])
    if _event_counts(low_event) != EXPECTED_LOW_K_EVENT_COUNTS:
        raise V271Error(
            f"V27 low_k_fusion event baseline failed to reproduce: {_event_counts(low_event)}"
        )
    delta = {
        "poly_exact_k_percentage_points_vs_low_k_fusion": 100.0 * (
            rescue_card["poly_exact"] - low_card["poly_exact"]
        ),
        "exact_k_percentage_points_vs_low_k_fusion": 100.0 * (
            rescue_card["exact"] - low_card["exact"]
        ),
        "event_f1_percentage_points_vs_low_k_fusion": 100.0 * (
            rescue_event["f1"] - low_event["f1"]
        ),
    }
    promote = rescue_card["poly_correct"] > low_card["poly_correct"]
    result = {
        "schema_version": 1,
        "protocol": {
            "experiment": "v271_poly_rescue",
            "primary_metric": "polyphonic exact-K",
            "thresholds_selected_inner_only": True,
            "outer_fold_used_once_after_threshold_freeze": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "promotion_rule_predeclared": True,
        },
        "reference": {
            "name": "v270 low_k_fusion",
            "cardinality": low_card,
            "event_50ms": low_event,
        },
        "poly_rescue": {
            "cardinality": rescue_card,
            "event_50ms": rescue_event,
        },
        "delta": delta,
        "threshold_by_fold": {str(r["fold"]): r["outer"]["threshold"] for r in reports},
        "per_fold": {
            str(r["fold"]): {
                "threshold": r["outer"]["threshold"],
                "inner_calibration": r["inner_calibration"],
                "low_k_fusion": r["outer"]["low_k_fusion"],
                "poly_rescue": r["outer"]["poly_rescue"],
                "transition": r["outer"]["transition_from_low_k_fusion"],
                "transition_by_v104_anchor": r["outer"]["transition_by_v104_anchor"],
            }
            for r in reports
        },
        "decision": {
            "promote_poly_rescue": bool(promote),
            "reason": (
                "aggregate polyphonic exact-K strictly improved over V27 low_k_fusion"
                if promote else
                "aggregate polyphonic exact-K did not strictly improve over V27 low_k_fusion"
            ),
        },
        "sources": {
            "v104_nested_run": V104_NESTED_RUN,
            "v104_nested_sha": V104_NESTED_SHA,
            "v111_audit_run": V111_AUDIT_RUN,
            "v111_audit_sha": V111_AUDIT_SHA,
            "v260_run": V260_RUN,
            "v260_sha": V260_SHA,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", **merged)
    print(json.dumps({
        "reference_poly_exact_k": low_card["poly_exact"],
        "poly_rescue_exact_k": rescue_card["poly_exact"],
        "delta_poly_pp": delta["poly_exact_k_percentage_points_vs_low_k_fusion"],
        "reference_exact_k": low_card["exact"],
        "poly_rescue_global_exact_k": rescue_card["exact"],
        "delta_exact_pp": delta["exact_k_percentage_points_vs_low_k_fusion"],
        "reference_f1": low_event["f1"],
        "poly_rescue_f1": rescue_event["f1"],
        "delta_f1_pp": delta["event_f1_percentage_points_vs_low_k_fusion"],
        "promote": promote,
    }, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--dataset-dir", type=Path, required=True)
    f.add_argument("--cache-dir", type=Path, required=True)
    f.add_argument("--v104-inner-dir", type=Path, required=True)
    f.add_argument("--v111-dir", type=Path, required=True)
    f.add_argument("--v260-dir", type=Path, required=True)
    f.add_argument("--outer-fold", type=int, required=True)
    f.add_argument("--output-dir", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--input-dir", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == "fold":
        run_fold(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
