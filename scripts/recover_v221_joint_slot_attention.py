"""Reporting/runtime recovery for the canonical V22.1 run.

This file does not change the V22.1 scientific treatment.

Two recovery modes are supported:

* final-fit: folds 0/1 only.  Their canonical inner/meta runs completed and
  selected 16/12 epochs respectively, but GitHub's 300 minute job timeout
  killed the final fit.  We reuse those canonical selected epoch counts and
  execute only the deterministic final fit + untouched outer evaluation.
* postprocess: folds 2/3/4 only.  Their final fits and outer evaluations
  completed and raw V13-compatible report/prediction/weight files were written;
  the job then died in a legacy terminal print KeyError.  We load those files
  and execute only the V17.2 -> V17.3 -> V22.1 reporting postprocessors.

Locked12/historical validation are never indexed or evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v221_joint_slot_attention as v221

CANONICAL_RUN = 34057761196
SELECTED_EPOCHS = {0: 16, 1: 12}
COMPLETED_RAW_FOLDS = {2, 3, 4}


class RecoveryError(RuntimeError):
    pass


def _args(ns) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=Path(ns.dataset_dir),
        cache_dir=Path(ns.cache_dir),
        baseline_eval_dir=Path(ns.baseline_eval_dir) if ns.baseline_eval_dir else None,
        outer_fold=int(ns.outer_fold),
        output_dir=Path(ns.output_dir),
        arm=v221.BASE_ARM,
        seed=v221.DEFAULT_SEED,
    )


def _seed_final(args):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    random.seed(args.seed + 1000 + args.outer_fold)
    np.random.seed(args.seed + 1000 + args.outer_fold)
    tf.random.set_seed(args.seed + 1000 + args.outer_fold)


def _load_raw_report(args):
    path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    if not path.exists():
        raise RecoveryError(f"missing raw report: {path}")
    return json.loads(path.read_text())


def _postprocess_saved(args):
    """Finish V22.1 reporting from a completed raw V13-compatible artifact."""
    import tensorflow as tf

    if args.outer_fold not in COMPLETED_RAW_FOLDS:
        raise RecoveryError("postprocess mode is restricted to canonical completed folds 2/3/4")
    raw_weight = args.output_dir / f"v130-fold-{args.outer_fold}.weights.h5"
    raw_pred = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    if not raw_weight.exists() or not raw_pred.exists():
        raise RecoveryError("completed raw fold is missing weight/prediction files")

    report = _load_raw_report(args)
    ctx = v172._fold_context(args)
    _seed_final(args)
    model, _, _ = v221._build_model(ctx["final_spec"])
    model.load_weights(raw_weight)
    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    slot_diag = v221._slot_diagnostics(model, v102._inputs(ctx["cache"], outer))
    report.setdefault("protocol", {})["v221_recovery"] = {
        "mode": "postprocess_only",
        "canonical_source_run": CANONICAL_RUN,
        "model_retrained": False,
        "outer_predictions_recomputed": False,
        "scientific_configuration_changed": False,
    }
    report = v221._postprocess(args, report, ctx, slot_diag)
    tf.keras.backend.clear_session()
    return report


def _final_fit_only(args):
    """Reuse canonical selected epochs, skip meta fitting, rerun only final fit."""
    import tensorflow as tf
    from tensorflow import keras

    fold = args.outer_fold
    if fold not in SELECTED_EPOCHS:
        raise RecoveryError("final-fit recovery is restricted to canonical timed-out folds 0/1")
    if args.baseline_eval_dir is None:
        raise RecoveryError("final-fit mode requires --baseline-eval-dir")
    selected = int(SELECTED_EPOCHS[fold])
    ctx = v172._fold_context(args)
    specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"build": 0, "fit": 0}
    built = []

    def builder():
        i = calls["build"]
        if i >= 2:
            raise RecoveryError("unexpected model build during final-fit recovery")
        calls["build"] += 1
        item = v221._build_model(specs[i])
        built.append(item[0])
        return item

    original_fit = keras.Model.fit

    def recovery_fit(self, *fit_args, **fit_kwargs):
        i = calls["fit"]
        calls["fit"] += 1
        if i == 0:
            # The canonical meta phase already ran in run 34057761196.
            # v130 only consumes val_loss here to derive selected_epochs.
            # A strictly decreasing synthetic trace reproduces exactly the
            # canonical selected count without fitting or selecting anything.
            return SimpleNamespace(history={"val_loss": list(np.arange(selected, 0, -1, dtype=np.float64))})
        if i != 1:
            raise RecoveryError("unexpected extra fit call")
        epochs = int(fit_kwargs.get("epochs", -1))
        if epochs != selected:
            raise RecoveryError(f"selected epoch mismatch: expected {selected}, got {epochs}")
        return original_fit(self, *fit_args, **fit_kwargs)

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = v171._targets
        v130._sample_weights = v171._sample_weights
        keras.Model.fit = recovery_fit
        try:
            report = v130.train_fold(args)
        except KeyError as exc:
            # Canonical V22.1 exposed a reporting-only legacy key failure after
            # report/predictions/weights had already been persisted.
            if exc.args != ("poly_exact_accuracy",):
                raise
            report = _load_raw_report(args)
    finally:
        keras.Model.fit = original_fit
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights

    if calls != {"build": 2, "fit": 2}:
        raise RecoveryError(f"unexpected recovery call counts: {calls}")
    if int(report["data"]["selected_epochs"]) != selected:
        raise RecoveryError("raw report does not preserve canonical selected epochs")

    # Do not publish diagnostics from the intentionally skipped meta model.
    report["meta_validation"] = {
        "metrics": None,
        "cardinality": None,
        "canonical_selected_epochs": selected,
        "source_run": CANONICAL_RUN,
        "recovery_note": "canonical meta selection reused; meta model was not refit during recovery",
    }
    report.setdefault("protocol", {})["v221_recovery"] = {
        "mode": "final_fit_only",
        "canonical_source_run": CANONICAL_RUN,
        "canonical_selected_epochs": selected,
        "meta_selection_rerun": False,
        "final_fit_retrained_from_canonical_seed": True,
        "outer_fold_used_for_training": False,
        "scientific_configuration_changed": False,
    }

    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    slot_diag = v221._slot_diagnostics(built[-1], v102._inputs(ctx["cache"], outer))
    report = v221._postprocess(args, report, ctx, slot_diag)
    tf.keras.backend.clear_session()
    return report


def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["final-fit", "postprocess"], required=True)
    ns = p.parse_args(argv)
    args = _args(ns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = _final_fit_only(args) if ns.mode == "final-fit" else _postprocess_saved(args)
    g = report["strata"]["aggregate"][v221.MODEL_KEY]["metrics"]["global"]
    card = report["strata"]["aggregate"][v221.MODEL_KEY]["cardinality"]
    print(json.dumps({
        "fold": args.outer_fold,
        "mode": ns.mode,
        "selected_epochs": report["data"]["selected_epochs"],
        "f1": g["f1"],
        "precision": g["precision"],
        "recall": g["recall"],
        "pred_ref": g["prediction_reference_ratio"],
        "poly_exact": card.get("poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))),
        "locked12_indexed_or_evaluated": report["protocol"].get("historical_validation_or_locked12_indexed_or_evaluated"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
