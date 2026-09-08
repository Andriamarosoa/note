"""V27.2 conditional polyphonic count reclassification.

V27.1 remains frozen on rows it predicts K<=1.  Rows already predicted
polyphonic by V27.1 are reclassified by a fresh five-way head trained only on
true K>=2 rows.  The experiment therefore targets the dominant remaining
2<->3<->4 confusions without adding another low-K rescue threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import audit_v270_class_conditional_fusion as v270
from scripts import train_v104_class_conditional_fusion as v104
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v240_categorical_k_candidate_subset as v240
from scripts import train_v260_count_weighting as v260
from scripts import train_v102_source_time_assignment as v102

FOLD_COUNT = 5
ARMS = ("uniform", "weighted")
SEED = 17261
MAX_EPOCHS = 30
V271_RUN = 34270082405
V271_SHA = "aed78122c5d85079c84b597e78331ba080f8b4b3"
EXPECTED_ROWS = 76768
EXPECTED_POLY_ROWS = 9401
EXPECTED_V271_POLY_CORRECT = 4003
EXPECTED_V271_GLOBAL_CORRECT = 62556
EXPECTED_V271_EVENT_COUNTS = (34708, 8116, 9512)


class V272Error(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one(paths, label: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise V272Error(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _prediction(x, label: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.int32)
    if a.ndim != 1 or np.any(a < 0) or np.any(a > 6):
        raise V272Error(f"invalid {label} prediction")
    return a


def _validate_poly_probability(probability, label: str = "poly probability") -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5:
        raise V272Error(f"invalid {label} shape {p.shape}")
    if not np.isfinite(p).all() or np.any(p < -1e-7):
        raise V272Error(f"invalid {label} values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise V272Error(f"{label} is not normalized")
    return p.astype(np.float32)


def poly_class_weights(fit_k, arm: str) -> np.ndarray:
    """Return weights for K=2..6, normalized on the supplied poly fit rows."""
    k = np.asarray(fit_k, dtype=np.int32)
    if k.ndim != 1 or len(k) == 0 or np.any(k < 2) or np.any(k > 6):
        raise V272Error("poly class weights require non-empty K=2..6 labels")
    if arm == "uniform":
        return np.ones(5, dtype=np.float32)
    if arm != "weighted":
        raise V272Error(f"invalid arm {arm!r}")
    cls = k - 2
    counts = np.bincount(cls, minlength=5)
    if np.any(counts == 0):
        raise V272Error(f"missing conditional class in fit partition: {counts.tolist()}")
    table = np.sqrt(len(k) / (5.0 * counts.astype(np.float64)))
    table = np.clip(table, 0.35, 4.0)
    table /= np.mean(table[cls])
    return table.astype(np.float32)


def build_model(arm: str, seed: int):
    import tensorflow as tf

    if arm not in ARMS:
        raise V272Error(f"invalid arm {arm!r}")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    scaffold, _, _ = v240._build_model({})
    hidden = scaffold.get_layer("v240_cardinality_hidden2").output
    out = tf.keras.layers.Dense(5, activation="softmax", name="v272_poly_cardinality")(hidden)
    model = tf.keras.Model(scaffold.inputs, out, name=f"v272_poly_{arm}")
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss="sparse_categorical_crossentropy")
    return model


def apply_poly_reclassification(base, poly_probability):
    """Replace only V27.1 K>=2 rows by the conditional K=2..6 argmax."""
    base = _prediction(base, "V27.1")
    p = _validate_poly_probability(poly_probability)
    if len(base) != len(p):
        raise V272Error("base/poly probability row mismatch")
    specialist = np.argmax(p, axis=1).astype(np.int32) + 2
    active = base >= 2
    out = base.copy()
    out[active] = specialist[active]
    if np.any(out[~active] != base[~active]) or np.any(out[active] < 2):
        raise V272Error("conditional reclassification invariant violated")
    return out.astype(np.int32), active, specialist


def select_arm(inner: dict) -> str:
    """Frozen arm ordering: exact rows, lower NLL, then uniform tie-break."""
    if set(inner) != set(ARMS):
        raise V272Error("arm-selection inputs incomplete")
    scored = []
    for arm in ARMS:
        row = inner[arm]
        exact = int(row["correct"])
        nll = float(row["selected_val_loss"])
        if exact < 0 or not np.isfinite(nll):
            raise V272Error("invalid inner arm score")
        tie = 1 if arm == "uniform" else 0
        scored.append(((exact, -nll, tie), arm))
    return max(scored)[1]


def _load_v271_fold(source_dir: Path, fold: int, outer, k, members):
    report_path = _one(source_dir.glob(f"**/report-fold-{fold}.json"), f"V27.1 fold {fold} report")
    pred_path = _one(source_dir.glob(f"**/predictions-fold-{fold}.npz"), f"V27.1 fold {fold} predictions")
    report = json.loads(report_path.read_text())
    if report.get("fold") != fold or report.get("protocol", {}).get("experiment") != "v271_poly_rescue":
        raise V272Error("invalid V27.1 source report")
    with np.load(pred_path, allow_pickle=False) as z:
        required = {"outer_fold", "global_index", "member", "k", "pred271_poly_rescue"}
        if not required.issubset(z.files):
            raise V272Error("invalid V27.1 source prediction schema")
        idx = np.asarray(z["global_index"], dtype=np.int64)
        truth = np.asarray(z["k"], dtype=np.int32)
        saved_members = np.asarray(z["member"]).astype(str)
        base = _prediction(z["pred271_poly_rescue"], "source V27.1")
        marker = np.asarray(z["outer_fold"], dtype=np.int16)
    outer = np.asarray(outer, dtype=np.int64)
    if not np.array_equal(idx, outer) or not np.all(marker == fold):
        raise V272Error("V27.1 outer index/fold mismatch")
    if not np.array_equal(truth, np.asarray(k)[outer]):
        raise V272Error("V27.1 label mismatch")
    if not np.array_equal(saved_members, np.asarray(members)[outer].astype(str)):
        raise V272Error("V27.1 member mismatch")
    return base, report, pred_path, report_path


def _event_counts(metric: dict) -> tuple[int, int, int]:
    return tuple(int(metric[x]) for x in ("true_positive", "false_positive", "false_negative"))


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


def train_fold(args):
    import tensorflow as tf

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.outer_fold not in range(FOLD_COUNT):
        raise V272Error("invalid outer fold")

    ctx = v171._fold_context(args)
    outer, fit, val, final = v260.validate_partitions(ctx)
    cache = ctx["cache"]
    k = np.asarray(ctx["k"], dtype=np.int32)
    members = np.asarray(ctx["members"]).astype(str)

    train_members = {t.annotation_member for t in ctx["train_split"]}
    validation_members = {t.annotation_member for t in ctx["validation"]}
    if set(cache["members"]) != train_members or train_members & validation_members:
        raise V272Error("dataset/cache split mismatch")

    base_outer, source_report, source_pred_path, source_report_path = _load_v271_fold(
        args.v271_dir, args.outer_fold, outer, k, members
    )
    kout = k[outer]
    source_card = v270.cardinality_report(kout, base_outer)
    expected_source_card = source_report["outer"]["poly_rescue"]
    if source_card["correct"] != int(expected_source_card["correct"]) or source_card["poly_correct"] != int(expected_source_card["poly_correct"]):
        raise V272Error("V27.1 source cardinality failed to reproduce")

    fit_poly = fit[k[fit] >= 2]
    val_poly = val[k[val] >= 2]
    final_poly = final[k[final] >= 2]
    if min(len(fit_poly), len(val_poly), len(final_poly)) <= 0:
        raise V272Error("empty polyphonic nested partition")

    arms = {}
    for arm in ARMS:
        fit_table = poly_class_weights(k[fit_poly], arm)
        model = build_model(arm, SEED + 100 + 10 * args.outer_fold + ARMS.index(arm))
        history = model.fit(
            v102._inputs(cache, fit_poly),
            k[fit_poly] - 2,
            sample_weight=fit_table[k[fit_poly] - 2],
            validation_data=(
                v102._inputs(cache, val_poly),
                k[val_poly] - 2,
                fit_table[k[val_poly] - 2],
            ),
            epochs=MAX_EPOCHS,
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
        val_loss = np.asarray(history.history.get("val_loss", []), dtype=np.float64)
        if len(val_loss) == 0 or not np.isfinite(val_loss).all():
            raise V272Error(f"invalid {arm} validation history")
        epochs = int(np.argmin(val_loss)) + 1
        probability = _validate_poly_probability(
            model.predict(v102._inputs(cache, val_poly), batch_size=128, verbose=0),
            f"inner {arm} probability",
        )
        pred = np.argmax(probability, axis=1).astype(np.int32) + 2
        correct = int(np.sum(pred == k[val_poly]))
        arms[arm] = {
            "epochs": epochs,
            "correct": correct,
            "rows": int(len(val_poly)),
            "exact": float(correct / len(val_poly)),
            "selected_val_loss": float(val_loss[epochs - 1]),
            "fit_class_weights_k2_to_k6": fit_table.tolist(),
            "history": history.history,
        }
        del model

    selected = select_arm(arms)
    selected_epochs = int(arms[selected]["epochs"])
    final_table = poly_class_weights(k[final_poly], selected)
    model = build_model(selected, SEED + 1000 + args.outer_fold)
    final_history = model.fit(
        v102._inputs(cache, final_poly),
        k[final_poly] - 2,
        sample_weight=final_table[k[final_poly] - 2],
        epochs=selected_epochs,
        batch_size=128,
        shuffle=True,
        verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )
    loss = np.asarray(final_history.history.get("loss", []), dtype=np.float64)
    if len(loss) != selected_epochs or not np.isfinite(loss).all():
        raise V272Error("invalid final refit history")

    outer_probability = _validate_poly_probability(
        model.predict(v102._inputs(cache, outer), batch_size=128, verbose=0),
        "outer conditional probability",
    )
    hybrid, active, specialist = apply_poly_reclassification(base_outer, outer_probability)
    if np.any(hybrid[~active] != base_outer[~active]) or np.any(active & (base_outer < 2)):
        raise V272Error("runtime mask invariant violated")

    base_card = v270.cardinality_report(kout, base_outer)
    hybrid_card = v270.cardinality_report(kout, hybrid)
    base_correct_nonpoly = int(np.sum((kout < 2) & (base_outer == kout)))
    hybrid_correct_nonpoly = int(np.sum((kout < 2) & (hybrid == kout)))
    if base_correct_nonpoly != hybrid_correct_nonpoly:
        raise V272Error("nonpoly correct-row invariant violated")
    if (hybrid_card["correct"] - base_card["correct"]) != (hybrid_card["poly_correct"] - base_card["poly_correct"]):
        raise V272Error("global/poly correct delta invariant violated")

    base_event = v104._metrics_for_indices(cache, ctx["train_split"], outer, base_outer)["global"]
    expected_event = source_report["outer"]["event_poly_rescue"]
    if _event_counts(base_event) != _event_counts(expected_event):
        raise V272Error("V27.1 source event metric failed to reproduce")
    hybrid_event = v104._metrics_for_indices(cache, ctx["train_split"], outer, hybrid)["global"]

    transition = v270.transition_report(kout, base_outer, hybrid)
    report = {
        "schema_version": 1,
        "protocol": {
            "experiment": "v272_poly_conditional_count",
            "reference": "v271 poly_rescue",
            "primary_metric": "polyphonic exact-K",
            "training_rows": "true K>=2 only",
            "runtime_rule": "keep V27.1 when base<=1; replace by conditional K=2..6 head when base>=2",
            "inner_arm_selection": [
                "maximize conditional exact rows",
                "minimize selected validation NLL",
                "prefer uniform on exact tie",
            ],
            "outer_labels_used_for_arm_or_epoch_selection": False,
            "threshold_tuning": False,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "automatic_promotion_rule": "aggregate outer poly exact-K strictly greater than V27.1",
        },
        "fold": int(args.outer_fold),
        "data": {
            "outer_rows": int(len(outer)),
            "fit_poly_rows": int(len(fit_poly)),
            "val_poly_rows": int(len(val_poly)),
            "final_fit_poly_rows": int(len(final_poly)),
            "active_outer_rows": int(active.sum()),
        },
        "inner": {
            "arms": arms,
            "selected_arm": selected,
            "selected_epochs": selected_epochs,
            "final_class_weights_k2_to_k6": final_table.tolist(),
        },
        "outer": {
            "v271": base_card,
            "v272": hybrid_card,
            "event_v271": base_event,
            "event_v272": hybrid_event,
            "transition_from_v271": transition,
        },
        "sources": {
            "v271_run": V271_RUN,
            "v271_sha": V271_SHA,
            "v271_prediction_sha256": digest(source_pred_path),
            "v271_report_sha256": digest(source_report_path),
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
        pred271_poly_rescue=base_outer.astype(np.int16),
        pred272_poly_conditional=hybrid.astype(np.int16),
        specialist_pred=specialist.astype(np.int16),
        specialist_probability=outer_probability.astype(np.float32),
        active_reclassification=active.astype(np.int8),
    )
    model.save_weights(args.output_dir / "selected.weights.h5")
    print(json.dumps({
        "fold": args.outer_fold,
        "selected_arm": selected,
        "selected_epochs": selected_epochs,
        "active_outer_rows": int(active.sum()),
        "v271_poly_exact": base_card["poly_exact"],
        "v272_poly_exact": hybrid_card["poly_exact"],
        "v271_exact": base_card["exact"],
        "v272_exact": hybrid_card["exact"],
        "v271_f1": base_event["f1"],
        "v272_f1": hybrid_event["f1"],
    }, indent=2, sort_keys=True))
    return report


def summarize(args):
    reports = []
    arrays = []
    for fold in range(FOLD_COUNT):
        report_path = _one(args.input_dir.glob(f"**/report-fold-{fold}.json"), f"V27.2 fold {fold} report")
        pred_path = _one(args.input_dir.glob(f"**/predictions-fold-{fold}.npz"), f"V27.2 fold {fold} predictions")
        report = json.loads(report_path.read_text())
        if report.get("fold") != fold or report.get("protocol", {}).get("experiment") != "v272_poly_conditional_count":
            raise V272Error(f"invalid V27.2 report fold {fold}")
        with np.load(pred_path, allow_pickle=False) as z:
            arr = {key: np.asarray(z[key]) for key in z.files}
        if not np.all(np.asarray(arr["outer_fold"]) == fold):
            raise V272Error(f"fold marker mismatch {fold}")
        reports.append(report)
        arrays.append(arr)

    merged = {key: np.concatenate([a[key] for a in arrays], axis=0) for key in arrays[0]}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if not np.array_equal(np.asarray(merged["global_index"], dtype=np.int64), np.arange(EXPECTED_ROWS, dtype=np.int64)):
        raise V272Error("outer predictions do not provide exact one-time coverage")

    k = np.asarray(merged["k"], dtype=np.int32)
    base = _prediction(merged["pred271_poly_rescue"], "aggregate V27.1")
    hybrid = _prediction(merged["pred272_poly_conditional"], "aggregate V27.2")
    base_card = v270.cardinality_report(k, base)
    hybrid_card = v270.cardinality_report(k, hybrid)
    if base_card["rows"] != EXPECTED_ROWS or base_card["poly_rows"] != EXPECTED_POLY_ROWS:
        raise V272Error("reference row counts mismatch")
    if base_card["poly_correct"] != EXPECTED_V271_POLY_CORRECT or base_card["correct"] != EXPECTED_V271_GLOBAL_CORRECT:
        raise V272Error("V27.1 aggregate cardinality baseline failed to reproduce")

    base_event = _aggregate_event([r["outer"]["event_v271"] for r in reports])
    hybrid_event = _aggregate_event([r["outer"]["event_v272"] for r in reports])
    if _event_counts(base_event) != EXPECTED_V271_EVENT_COUNTS:
        raise V272Error("V27.1 aggregate event baseline failed to reproduce")

    delta_poly = 100.0 * (hybrid_card["poly_exact"] - base_card["poly_exact"])
    delta_exact = 100.0 * (hybrid_card["exact"] - base_card["exact"])
    delta_f1 = 100.0 * (hybrid_event["f1"] - base_event["f1"])
    if (hybrid_card["correct"] - base_card["correct"]) != (hybrid_card["poly_correct"] - base_card["poly_correct"]):
        raise V272Error("aggregate global/poly correct delta invariant violated")
    promote = hybrid_card["poly_correct"] > base_card["poly_correct"]

    result = {
        "schema_version": 1,
        "protocol": {
            "experiment": "v272_poly_conditional_count",
            "primary_metric": "polyphonic exact-K",
            "outer_fold_used_once_after_inner_arm_and_epoch_selection": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "promotion_rule_predeclared": True,
        },
        "reference": {"name": "v271 poly_rescue", "cardinality": base_card, "event_50ms": base_event},
        "poly_conditional": {"cardinality": hybrid_card, "event_50ms": hybrid_event},
        "delta": {
            "poly_exact_k_percentage_points_vs_v271": delta_poly,
            "exact_k_percentage_points_vs_v271": delta_exact,
            "event_f1_percentage_points_vs_v271": delta_f1,
        },
        "per_fold": {
            str(r["fold"]): {
                "selected_arm": r["inner"]["selected_arm"],
                "selected_epochs": r["inner"]["selected_epochs"],
                "inner_arms": r["inner"]["arms"],
                "v271": r["outer"]["v271"],
                "v272": r["outer"]["v272"],
                "transition": r["outer"]["transition_from_v271"],
            }
            for r in reports
        },
        "decision": {
            "promote_poly_conditional": bool(promote),
            "reason": (
                "aggregate polyphonic exact-K strictly improved over V27.1"
                if promote else
                "aggregate polyphonic exact-K did not strictly improve over V27.1"
            ),
        },
        "source": {"v271_run": V271_RUN, "v271_sha": V271_SHA},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", **merged)
    print(json.dumps({
        "reference_poly_exact_k": base_card["poly_exact"],
        "poly_conditional_exact_k": hybrid_card["poly_exact"],
        "delta_poly_pp": delta_poly,
        "reference_exact_k": base_card["exact"],
        "poly_conditional_global_exact_k": hybrid_card["exact"],
        "delta_exact_pp": delta_exact,
        "reference_f1": base_event["f1"],
        "poly_conditional_f1": hybrid_event["f1"],
        "delta_f1_pp": delta_f1,
        "promote": promote,
    }, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fold")
    for name in ("dataset-dir", "cache-dir", "v271-dir", "output-dir"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--outer-fold", type=int, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--input-dir", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "fold":
        train_fold(args)
    else:
        summarize(args)
