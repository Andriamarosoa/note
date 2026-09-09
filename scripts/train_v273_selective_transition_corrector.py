"""V27.3 nested selective correction of adjacent polyphonic transitions.

V27.1 remains the frozen runtime reference.  The failed V27.2 conditional
head is used only as a proposal signal.  Four directed transitions are
eligible: 2->3, 3->2, 3->4 and 4->3.  One probability-margin threshold per
direction is selected on the inner validation partition, including an
explicit no-override option, before the untouched outer fold is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import audit_v270_class_conditional_fusion as v270
from scripts import audit_v271_poly_rescue as v271
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_class_conditional_fusion as v104
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v260_count_weighting as v260
from scripts import train_v272_poly_conditional_count as v272


FOLD_COUNT = 5
TRANSITIONS = ((2, 3), (3, 2), (3, 4), (4, 3))
NO_OVERRIDE_MARGIN = 1.000001
V104_NESTED_RUN = 33647694565
V104_NESTED_SHA = "345dea1e92f6281583c58590a0b1286ce4df459b"
V260_RUN = 34201830555
V260_SHA = "5f41972a6e2f081d6f2ad652beae31dfff9f0076"
V272_RUN = 34287254337
V272_SHA = "d11f73079a9eb12b2ebce3c8d1df8209b0bd6bdf"
EXPECTED_ROWS = 76768
EXPECTED_POLY_ROWS = 9401
EXPECTED_V271_CORRECT = 62556
EXPECTED_V271_POLY_CORRECT = 4003
EXPECTED_V271_EVENT_COUNTS = (34708, 8116, 9512)


class V273Error(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one(paths, label: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise V273Error(f"expected one {label}, found {len(paths)}")
    return paths[0]


def transition_name(source: int, target: int) -> str:
    pair = (int(source), int(target))
    if pair not in TRANSITIONS:
        raise V273Error(f"unsupported transition {pair}")
    return f"{pair[0]}_to_{pair[1]}"


def _validate_thresholds(thresholds: dict) -> dict[str, float]:
    expected = {transition_name(*pair) for pair in TRANSITIONS}
    if set(thresholds) != expected:
        raise V273Error("transition threshold set mismatch")
    result = {key: float(value) for key, value in thresholds.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise V273Error("transition thresholds must be finite")
    return result


def conditional_proposal(base, probability):
    """Return V27.2 proposal and P(proposal)-P(V27.1 count)."""
    base = v272._prediction(base, "V27.1 base")
    p = v272._validate_poly_probability(probability, "V27.2 conditional probability")
    if len(base) != len(p):
        raise V273Error("base/conditional probability row mismatch")
    proposal = np.argmax(p, axis=1).astype(np.int32) + 2
    margin = np.zeros(len(base), dtype=np.float64)
    poly_base = base >= 2
    rows = np.flatnonzero(poly_base)
    margin[rows] = (
        p[rows, proposal[rows] - 2].astype(np.float64)
        - p[rows, base[rows] - 2].astype(np.float64)
    )
    if np.any(margin[poly_base] < -1e-7):
        raise V273Error("conditional argmax has a negative pairwise margin")
    return proposal, margin


def apply_transition_corrector(base, probability, thresholds: dict):
    """Apply frozen directed thresholds without consulting labels."""
    base = v272._prediction(base, "V27.1 base")
    thresholds = _validate_thresholds(thresholds)
    proposal, margin = conditional_proposal(base, probability)
    corrected = base.copy()
    active = np.zeros(len(base), dtype=bool)
    direction = np.full(len(base), -1, dtype=np.int16)
    for direction_id, (source, target) in enumerate(TRANSITIONS):
        threshold = thresholds[transition_name(source, target)]
        mask = (base == source) & (proposal == target) & (margin >= threshold)
        corrected[mask] = target
        active |= mask
        direction[mask] = direction_id
    if np.any(corrected[base <= 1] != base[base <= 1]):
        raise V273Error("corrector changed a non-polyphonic V27.1 prediction")
    changed_pairs = set(zip(base[active].tolist(), corrected[active].tolist()))
    if not changed_pairs.issubset(set(TRANSITIONS)):
        raise V273Error(f"corrector emitted unsupported transitions: {changed_pairs}")
    return corrected.astype(np.int32), active, proposal, margin, direction


def _threshold_candidates(margin, eligible) -> list[float]:
    values = np.asarray(margin, dtype=np.float64)[np.asarray(eligible, dtype=bool)]
    return [NO_OVERRIDE_MARGIN, *sorted(set(float(value) for value in values), reverse=True)]


def select_transition_thresholds(k, base, probability) -> dict:
    """Select one threshold per directed transition using inner labels only."""
    k = v272._prediction(k, "inner truth")
    base = v272._prediction(base, "inner V27.1")
    proposal, margin = conditional_proposal(base, probability)
    if not (k.shape == base.shape == proposal.shape):
        raise V273Error("inner transition calibration shape mismatch")
    thresholds = {}
    directions = {}
    for source, target in TRANSITIONS:
        name = transition_name(source, target)
        eligible = (base == source) & (proposal == target)
        best = None
        evaluated = 0
        for threshold in _threshold_candidates(margin, eligible):
            active = eligible & (margin >= threshold)
            corrected = int(np.sum(active & (k == target)))
            regressed = int(np.sum(active & (k == source)))
            neutral = int(active.sum()) - corrected - regressed
            net = corrected - regressed
            key = (net, -int(active.sum()), float(threshold))
            evaluated += 1
            if best is None or key > best[0]:
                best = (key, float(threshold), int(active.sum()), corrected, regressed, neutral)
        assert best is not None
        _, threshold, activated, corrected, regressed, neutral = best
        thresholds[name] = threshold
        directions[name] = {
            "eligible_rows": int(eligible.sum()),
            "candidate_thresholds_evaluated": evaluated,
            "threshold": threshold,
            "activated_rows": activated,
            "corrected_rows": corrected,
            "regressed_rows": regressed,
            "neutral_rows": neutral,
            "net_correct_rows": corrected - regressed,
        }
    selected, active, _, _, _ = apply_transition_corrector(base, probability, thresholds)
    return {
        "score": "P_v272(proposed_K) - P_v272(V271_K)",
        "transitions": [transition_name(*pair) for pair in TRANSITIONS],
        "selection_order": [
            "maximize net exact rows independently per directed transition",
            "minimize activated rows",
            "prefer higher threshold",
        ],
        "thresholds": thresholds,
        "directions": directions,
        "activated_rows": int(active.sum()),
        "baseline": v270.cardinality_report(k, base),
        "selected": v270.cardinality_report(k, selected),
        "transition_from_v271": v270.transition_report(k, base, selected),
    }


def _load_v272_fold(source_dir: Path, fold: int, outer, k, members):
    report_path = _one(
        source_dir.glob(f"**/report-fold-{fold}.json"), f"V27.2 fold {fold} report"
    )
    prediction_path = _one(
        source_dir.glob(f"**/predictions-fold-{fold}.npz"),
        f"V27.2 fold {fold} predictions",
    )
    report = json.loads(report_path.read_text())
    if report.get("fold") != fold:
        raise V273Error("V27.2 source fold mismatch")
    if report.get("protocol", {}).get("experiment") != "v272_poly_conditional_count":
        raise V273Error("V27.2 source protocol mismatch")
    if report.get("inner", {}).get("selected_arm") != "uniform":
        raise V273Error("V27.2 source did not select the expected uniform arm")
    with np.load(prediction_path, allow_pickle=False) as z:
        required = {
            "outer_fold",
            "global_index",
            "member",
            "k",
            "pred271_poly_rescue",
            "pred272_poly_conditional",
            "specialist_pred",
            "specialist_probability",
        }
        if not required.issubset(z.files):
            raise V273Error("V27.2 source prediction schema mismatch")
        marker = np.asarray(z["outer_fold"], dtype=np.int16)
        index = np.asarray(z["global_index"], dtype=np.int64)
        truth = np.asarray(z["k"], dtype=np.int32)
        saved_members = np.asarray(z["member"]).astype(str)
        base = v272._prediction(z["pred271_poly_rescue"], "source V27.1")
        saved_v272 = v272._prediction(z["pred272_poly_conditional"], "source V27.2")
        saved_specialist = v272._prediction(z["specialist_pred"], "source V27.2 specialist")
        probability = v272._validate_poly_probability(
            z["specialist_probability"], "source V27.2 probability"
        )
    outer = np.asarray(outer, dtype=np.int64)
    if not np.array_equal(index, outer) or not np.all(marker == fold):
        raise V273Error("V27.2 outer index/fold mismatch")
    if not np.array_equal(truth, np.asarray(k)[outer]):
        raise V273Error("V27.2 labels mismatch")
    if not np.array_equal(saved_members, np.asarray(members)[outer].astype(str)):
        raise V273Error("V27.2 members mismatch")
    replay_v272, _, specialist = v272.apply_poly_reclassification(base, probability)
    if not np.array_equal(saved_specialist, specialist) or not np.array_equal(saved_v272, replay_v272):
        raise V273Error("V27.2 source predictions do not replay")
    return base, probability, report, prediction_path, report_path


def _fit_fixed_inner_v260_probability(
    ctx: dict, arm: str, outer_fold: int, fixed_epochs: int
):
    """Refit a V26 probe for its archived inner-only epoch budget.

    Re-running early stopping can move the apparent best epoch across CPU
    runners because hardware and numerical-kernel differences perturb the
    optimization trajectory.
    The archived epoch budget is already an inner-only choice, so refitting
    exactly that many epochs is the stable replay contract needed here.
    """
    import tensorflow as tf

    _, fit, val, _ = v260.validate_partitions(ctx)
    k = np.asarray(ctx["k"], dtype=np.int32)
    fixed_epochs = int(fixed_epochs)
    if fixed_epochs <= 0 or fixed_epochs > v260.MAX_EPOCHS:
        raise V273Error(f"invalid frozen V26 {arm} epoch budget: {fixed_epochs}")
    table = v260.arm_weights(arm, k[fit])
    model = v260.build_model(arm, v260.SEED + 100 + outer_fold)
    history = model.fit(
        v260.v102._inputs(ctx["cache"], fit),
        k[fit],
        sample_weight=table[k[fit]],
        validation_data=(
            v260.v102._inputs(ctx["cache"], val),
            k[val],
            table[k[val]],
        ),
        epochs=fixed_epochs,
        batch_size=128,
        shuffle=True,
        verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )
    loss = np.asarray(history.history.get("loss", []), dtype=np.float64)
    val_loss = np.asarray(history.history.get("val_loss", []), dtype=np.float64)
    if (
        len(loss) != fixed_epochs
        or len(val_loss) != fixed_epochs
        or not np.isfinite(loss).all()
        or not np.isfinite(val_loss).all()
    ):
        raise V273Error(f"invalid fixed-epoch inner V26 {arm} history")
    probability = np.asarray(
        model.predict(v260.v102._inputs(ctx["cache"], val), batch_size=128, verbose=0)
    )
    return v271._validate_probability(
        probability, f"inner V26 {arm} probability"
    ), {
        "fixed_source_epochs": fixed_epochs,
        "observed_best_epoch_within_budget": int(np.argmin(val_loss)) + 1,
        "final_val_loss": float(val_loss[-1]),
    }


def _replay_inner_v271(ctx: dict, args, v260_report: dict):
    outer, fit, val, final = v260.validate_partitions(ctx)
    k = np.asarray(ctx["k"], dtype=np.int32)
    members = np.asarray(ctx["members"]).astype(str)
    inner, inner_paths, meta_fold = v271._load_inner_v104(
        args.v104_inner_dir, args.outer_fold, final, val, k, members
    )
    v104_probability, v104_epochs = v271._inner_v104_probability(inner, meta_fold)
    anchor = np.argmax(v104_probability, axis=1).astype(np.int32)
    weighted, weighted_replay = _fit_fixed_inner_v260_probability(
        ctx,
        "weighted",
        args.outer_fold,
        v260_report["arms"]["weighted"]["epochs"],
    )
    uniform, uniform_replay = _fit_fixed_inner_v260_probability(
        ctx,
        "uniform",
        args.outer_fold,
        v260_report["arms"]["uniform"]["epochs"],
    )
    if not (len(anchor) == len(uniform) == len(weighted) == len(val)):
        raise V273Error("inner V27.1 replay row mismatch")
    low = v270.fuse_counts(anchor, np.argmax(uniform, axis=1), "low_k_fusion")
    calibration = v271.select_threshold(k[val], anchor, low, weighted)
    base, _, _, _ = v271.apply_poly_rescue(
        anchor, low, weighted, calibration["threshold"]
    )
    return base, {
        "meta_fold": int(meta_fold),
        "v104_selected_epochs": int(v104_epochs),
        "v260_weighted_fixed_replay": weighted_replay,
        "v260_uniform_fixed_replay": uniform_replay,
        "v271_threshold": float(calibration["threshold"]),
    }, inner_paths


def _replay_inner_v272(ctx: dict, outer_fold: int, source_report: dict):
    import tensorflow as tf

    _, fit, val, _ = v260.validate_partitions(ctx)
    k = np.asarray(ctx["k"], dtype=np.int32)
    fit_poly = fit[k[fit] >= 2]
    val_poly = val[k[val] >= 2]
    expected_epochs = int(source_report["inner"]["arms"]["uniform"]["epochs"])
    table = v272.poly_class_weights(k[fit_poly], "uniform")
    model = v272.build_model("uniform", v272.SEED + 100 + 10 * outer_fold)
    history = model.fit(
        v102._inputs(ctx["cache"], fit_poly),
        k[fit_poly] - 2,
        sample_weight=table[k[fit_poly] - 2],
        validation_data=(
            v102._inputs(ctx["cache"], val_poly),
            k[val_poly] - 2,
            table[k[val_poly] - 2],
        ),
        epochs=expected_epochs,
        batch_size=128,
        shuffle=True,
        verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )
    val_loss = np.asarray(history.history.get("val_loss", []), dtype=np.float64)
    if len(val_loss) != expected_epochs or not np.isfinite(val_loss).all():
        raise V273Error("invalid inner V27.2 validation history")
    observed_best_epoch = int(np.argmin(val_loss)) + 1
    probability = v272._validate_poly_probability(
        model.predict(v102._inputs(ctx["cache"], val), batch_size=128, verbose=0),
        "inner V27.2 probability",
    )
    poly_prediction = np.argmax(probability[k[val] >= 2], axis=1).astype(np.int32) + 2
    observed_correct = int(np.sum(poly_prediction == k[val_poly]))
    source_correct = int(source_report["inner"]["arms"]["uniform"]["correct"])
    return probability, {
        "uniform_fixed_source_epochs": expected_epochs,
        "uniform_observed_best_epoch_within_budget": observed_best_epoch,
        "uniform_poly_rows": int(len(val_poly)),
        "uniform_poly_correct": observed_correct,
        "uniform_source_archived_poly_correct": source_correct,
        "uniform_final_val_loss": float(val_loss[-1]),
    }


def _event_counts(metric: dict) -> tuple[int, int, int]:
    return tuple(
        int(metric[key]) for key in ("true_positive", "false_positive", "false_negative")
    )


def _aggregate_event(metrics: list[dict]) -> dict:
    tp = sum(int(metric["true_positive"]) for metric in metrics)
    fp = sum(int(metric["false_positive"]) for metric in metrics)
    fn = sum(int(metric["false_negative"]) for metric in metrics)
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
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.outer_fold not in range(FOLD_COUNT):
        raise V273Error("invalid outer fold")

    ctx = v171._fold_context(args)
    outer, _, val, _ = v260.validate_partitions(ctx)
    cache = ctx["cache"]
    k = np.asarray(ctx["k"], dtype=np.int32)
    members = np.asarray(ctx["members"]).astype(str)
    train_members = {track.annotation_member for track in ctx["train_split"]}
    validation_members = {track.annotation_member for track in ctx["validation"]}
    if set(cache["members"]) != train_members or train_members & validation_members:
        raise V273Error("dataset/cache split mismatch")

    base_outer, probability_outer, source_report, source_pred, source_json = _load_v272_fold(
        args.v272_dir, args.outer_fold, outer, k, members
    )
    _, _, v260_report, v260_pred, v260_json = v271._load_v260_fold(
        args.v260_dir, args.outer_fold, outer, k, members
    )
    base_inner, v271_replay, inner_paths = _replay_inner_v271(ctx, args, v260_report)
    probability_inner, v272_replay = _replay_inner_v272(
        ctx, args.outer_fold, source_report
    )
    calibration = select_transition_thresholds(k[val], base_inner, probability_inner)
    corrected_outer, active, proposal, margin, direction = apply_transition_corrector(
        base_outer, probability_outer, calibration["thresholds"]
    )

    kout = k[outer]
    base_card = v270.cardinality_report(kout, base_outer)
    corrected_card = v270.cardinality_report(kout, corrected_outer)
    source_base_card = source_report["outer"]["v271"]
    if (
        base_card["correct"] != int(source_base_card["correct"])
        or base_card["poly_correct"] != int(source_base_card["poly_correct"])
    ):
        raise V273Error("V27.1 source cardinality failed to reproduce")
    if np.any(corrected_outer[base_outer <= 1] != base_outer[base_outer <= 1]):
        raise V273Error("outer corrector changed V27.1 K<=1")
    base_nonpoly = int(np.sum((kout < 2) & (base_outer == kout)))
    corrected_nonpoly = int(np.sum((kout < 2) & (corrected_outer == kout)))
    if base_nonpoly != corrected_nonpoly:
        raise V273Error("outer nonpoly correct-row invariant violated")
    if (corrected_card["correct"] - base_card["correct"]) != (
        corrected_card["poly_correct"] - base_card["poly_correct"]
    ):
        raise V273Error("outer global/poly correct delta invariant violated")

    base_event = v104._metrics_for_indices(
        cache, ctx["train_split"], outer, base_outer
    )["global"]
    expected_event = source_report["outer"]["event_v271"]
    if _event_counts(base_event) != _event_counts(expected_event):
        raise V273Error("V27.1 source event metric failed to reproduce")
    corrected_event = v104._metrics_for_indices(
        cache, ctx["train_split"], outer, corrected_outer
    )["global"]

    direction_report = {}
    for direction_id, pair in enumerate(TRANSITIONS):
        name = transition_name(*pair)
        mask = active & (direction == direction_id)
        direction_report[name] = {
            "threshold": float(calibration["thresholds"][name]),
            "activated_rows": int(mask.sum()),
            "corrected_rows": int(np.sum(mask & (corrected_outer == kout) & (base_outer != kout))),
            "regressed_rows": int(np.sum(mask & (base_outer == kout) & (corrected_outer != kout))),
        }
        direction_report[name]["net_correct_rows"] = (
            direction_report[name]["corrected_rows"]
            - direction_report[name]["regressed_rows"]
        )

    report = {
        "schema_version": 1,
        "protocol": {
            "experiment": "v273_selective_transition_corrector",
            "reference": "v271 poly_rescue",
            "proposal_source": "v272 uniform conditional poly count head",
            "primary_metric": "polyphonic exact-K",
            "eligible_transitions": [transition_name(*pair) for pair in TRANSITIONS],
            "confidence_score": "P_v272(proposed_K) - P_v272(V271_K)",
            "thresholds_selected_inner_only": True,
            "inner_epoch_budgets": "frozen from archived inner-only V26/V27.2 reports",
            "hardware_sensitive_early_stopping_replay_required": False,
            "outer_labels_used_for_threshold_selection": False,
            "event_f1_used_for_threshold_selection": False,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "automatic_promotion_rule": "aggregate outer poly exact-K strictly greater than V27.1",
            "superseded_run": 34293193376,
        },
        "fold": int(args.outer_fold),
        "data": {
            "outer_rows": int(len(outer)),
            "inner_validation_rows": int(len(val)),
            "active_outer_rows": int(active.sum()),
        },
        "inner": {
            "v271_replay": v271_replay,
            "v272_replay": v272_replay,
            "transition_calibration": calibration,
        },
        "outer": {
            "v271": base_card,
            "v273": corrected_card,
            "event_v271": base_event,
            "event_v273": corrected_event,
            "transition_from_v271": v270.transition_report(kout, base_outer, corrected_outer),
            "by_direction": direction_report,
        },
        "sources": {
            "v104_nested_run": V104_NESTED_RUN,
            "v104_nested_sha": V104_NESTED_SHA,
            "v260_run": V260_RUN,
            "v260_sha": V260_SHA,
            "v272_run": V272_RUN,
            "v272_sha": V272_SHA,
            "files": [
                *({"path": str(path), "sha256": digest(path)} for path in inner_paths),
                {"path": str(v260_pred), "sha256": digest(v260_pred)},
                {"path": str(v260_json), "sha256": digest(v260_json)},
                {"path": str(source_pred), "sha256": digest(source_pred)},
                {"path": str(source_json), "sha256": digest(source_json)},
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
        pred271_poly_rescue=base_outer.astype(np.int16),
        pred273_selective_transition=corrected_outer.astype(np.int16),
        specialist_pred=proposal.astype(np.int16),
        specialist_probability=probability_outer.astype(np.float32),
        specialist_margin=margin.astype(np.float32),
        correction_activated=active.astype(np.int8),
        correction_direction=direction.astype(np.int16),
        selected_thresholds=np.asarray(
            [calibration["thresholds"][transition_name(*pair)] for pair in TRANSITIONS],
            dtype=np.float64,
        ),
    )
    print(
        json.dumps(
            {
                "fold": args.outer_fold,
                "thresholds": calibration["thresholds"],
                "active_outer_rows": int(active.sum()),
                "v271_poly_exact": base_card["poly_exact"],
                "v273_poly_exact": corrected_card["poly_exact"],
                "v271_exact": base_card["exact"],
                "v273_exact": corrected_card["exact"],
                "v271_f1": base_event["f1"],
                "v273_f1": corrected_event["f1"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def summarize(args):
    reports = []
    arrays = []
    for fold in range(FOLD_COUNT):
        report_path = _one(
            args.input_dir.glob(f"**/report-fold-{fold}.json"), f"V27.3 fold {fold} report"
        )
        prediction_path = _one(
            args.input_dir.glob(f"**/predictions-fold-{fold}.npz"),
            f"V27.3 fold {fold} predictions",
        )
        report = json.loads(report_path.read_text())
        if report.get("fold") != fold:
            raise V273Error(f"V27.3 fold marker mismatch: {fold}")
        if report.get("protocol", {}).get("experiment") != "v273_selective_transition_corrector":
            raise V273Error(f"V27.3 protocol mismatch: {fold}")
        with np.load(prediction_path, allow_pickle=False) as z:
            arr = {key: np.asarray(z[key]) for key in z.files if key != "selected_thresholds"}
        if not np.all(arr["outer_fold"] == fold):
            raise V273Error(f"V27.3 prediction fold mismatch: {fold}")
        reports.append(report)
        arrays.append(arr)

    merged = {key: np.concatenate([arr[key] for arr in arrays], axis=0) for key in arrays[0]}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if not np.array_equal(merged["global_index"], np.arange(EXPECTED_ROWS)):
        raise V273Error("outer predictions do not provide exact one-time coverage")
    k = v272._prediction(merged["k"], "aggregate truth")
    base = v272._prediction(merged["pred271_poly_rescue"], "aggregate V27.1")
    corrected = v272._prediction(
        merged["pred273_selective_transition"], "aggregate V27.3"
    )
    base_card = v270.cardinality_report(k, base)
    corrected_card = v270.cardinality_report(k, corrected)
    if base_card["rows"] != EXPECTED_ROWS or base_card["poly_rows"] != EXPECTED_POLY_ROWS:
        raise V273Error("V27.1 aggregate row counts mismatch")
    if base_card["correct"] != EXPECTED_V271_CORRECT or base_card["poly_correct"] != EXPECTED_V271_POLY_CORRECT:
        raise V273Error("V27.1 aggregate cardinality baseline failed to reproduce")
    if (corrected_card["correct"] - base_card["correct"]) != (
        corrected_card["poly_correct"] - base_card["poly_correct"]
    ):
        raise V273Error("aggregate global/poly correct delta invariant violated")

    base_event = _aggregate_event([report["outer"]["event_v271"] for report in reports])
    corrected_event = _aggregate_event([report["outer"]["event_v273"] for report in reports])
    if _event_counts(base_event) != EXPECTED_V271_EVENT_COUNTS:
        raise V273Error("V27.1 aggregate event baseline failed to reproduce")
    delta = {
        "poly_exact_k_percentage_points_vs_v271": 100.0
        * (corrected_card["poly_exact"] - base_card["poly_exact"]),
        "exact_k_percentage_points_vs_v271": 100.0
        * (corrected_card["exact"] - base_card["exact"]),
        "event_f1_percentage_points_vs_v271": 100.0
        * (corrected_event["f1"] - base_event["f1"]),
    }
    promote = corrected_card["poly_correct"] > base_card["poly_correct"]
    result = {
        "schema_version": 1,
        "protocol": {
            "experiment": "v273_selective_transition_corrector",
            "primary_metric": "polyphonic exact-K",
            "thresholds_selected_inner_only": True,
            "outer_fold_used_once_after_threshold_freeze": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "promotion_rule_predeclared": True,
        },
        "reference": {"name": "v271 poly_rescue", "cardinality": base_card, "event_50ms": base_event},
        "selective_transition": {"cardinality": corrected_card, "event_50ms": corrected_event},
        "delta": delta,
        "thresholds_by_fold": {
            str(report["fold"]): report["inner"]["transition_calibration"]["thresholds"]
            for report in reports
        },
        "per_fold": {
            str(report["fold"]): {
                "v271": report["outer"]["v271"],
                "v273": report["outer"]["v273"],
                "transition": report["outer"]["transition_from_v271"],
                "by_direction": report["outer"]["by_direction"],
                "inner_calibration": report["inner"]["transition_calibration"],
            }
            for report in reports
        },
        "decision": {
            "promote_selective_transition": bool(promote),
            "reason": (
                "aggregate polyphonic exact-K strictly improved over V27.1"
                if promote
                else "aggregate polyphonic exact-K did not strictly improve over V27.1"
            ),
        },
        "source": {"v272_run": V272_RUN, "v272_sha": V272_SHA},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(args.output_dir / "predictions.npz", **merged)
    print(
        json.dumps(
            {
                "reference_poly_exact_k": base_card["poly_exact"],
                "selective_transition_poly_exact_k": corrected_card["poly_exact"],
                "delta_poly_pp": delta["poly_exact_k_percentage_points_vs_v271"],
                "reference_exact_k": base_card["exact"],
                "selective_transition_exact_k": corrected_card["exact"],
                "delta_exact_pp": delta["exact_k_percentage_points_vs_v271"],
                "reference_f1": base_event["f1"],
                "selective_transition_f1": corrected_event["f1"],
                "delta_f1_pp": delta["event_f1_percentage_points_vs_v271"],
                "promote": promote,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    fold = sub.add_parser("fold")
    for name in (
        "dataset-dir",
        "cache-dir",
        "v104-inner-dir",
        "v260-dir",
        "v272-dir",
        "output-dir",
    ):
        fold.add_argument("--" + name, type=Path, required=True)
    fold.add_argument("--outer-fold", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--input-dir", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "fold":
        train_fold(args)
    else:
        summarize(args)
