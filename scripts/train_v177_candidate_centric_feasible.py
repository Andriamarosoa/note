"""V17.7 feasibility correction for the pure candidate-centric decoder.

The first V17.7 run exposed a real representational boundary that the synthetic
preflight missed: 180/76,768 outer-clean rows have fewer valid candidate tokens
than true births (C < K).  A one-to-one candidate-object set cannot represent
those rows.  The original exact-DP loss therefore reached its INF sentinel and
made a 0.2345% structural minority dominate optimization.

This correction keeps the V17.7 architecture unchanged: no anonymous seeds,
queries, dustbins, thresholds, categorical count head or extra parameters are
introduced.  It changes training treatment only for structurally infeasible
rows: the candidate-set objective is zero when C < K.  Those rows remain in the
outer-clean evaluation and therefore still count as model errors.  All feasible
rows retain the original V17.7 loss exactly, including V17.3 mass-preserving
weights and the 0.35 Poisson-binomial count NLL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import train_v177_candidate_centric as v177

DEFAULT_SEED = v177.DEFAULT_SEED
PRESENCE_THRESHOLD = v177.PRESENCE_THRESHOLD
BASE_ARM = v177.BASE_ARM
MODEL_KEY = v177.MODEL_KEY
MAX_CANDIDATES = v177.MAX_CANDIDATES
EVENT_QUERIES = v177.EVENT_QUERIES
COUNT_NLL_WEIGHT = v177.COUNT_NLL_WEIGHT


class V177FeasibilityError(RuntimeError):
    pass


def _candidate_feasibility_mask_np(candidate_mask, k):
    candidate_mask = np.asarray(candidate_mask, dtype=np.float32)
    k = np.asarray(k, dtype=np.int32).reshape(-1)
    valid_count = np.sum(candidate_mask > 0.5, axis=1).astype(np.int32)
    if len(valid_count) != len(k):
        raise V177FeasibilityError("candidate mask / K row mismatch")
    return valid_count >= k, valid_count


def _candidate_set_loss_feasible(spec: dict):
    """Original V17.7 set loss, exactly masked only where C < K."""
    import tensorflow as tf
    from tensorflow import keras

    base = v177._candidate_set_loss(spec)

    class FeasibleCandidateCentricSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="v177_candidate_centric_feasible_set_loss")

        def call(self, y_true, y_pred):
            yt = tf.cast(y_true, tf.float32)
            yp = tf.cast(y_pred, tf.float32)
            raw = base.call(yt, yp)
            true_k = tf.reduce_sum(yt[:, :, 0], axis=1)
            valid_count = tf.reduce_sum(
                tf.cast(yp[:, :, v177.v171.SET_VALID_OFFSET] > 0.5, tf.float32), axis=1
            )
            feasible = valid_count >= true_k
            return tf.where(feasible, raw, tf.zeros_like(raw))

    return FeasibleCandidateCentricSetLoss()


def _correct_postprocess_report(report: dict, ctx: dict) -> dict:
    outer_idx = np.asarray(ctx["outer_idx"], dtype=np.int64)
    k = np.asarray(ctx["k"], dtype=np.int32)[outer_idx]
    candidate_mask = np.asarray(ctx["cache"]["mask"], dtype=np.float32)[outer_idx]
    feasible, valid_count = _candidate_feasibility_mask_np(candidate_mask, k)
    infeasible = ~feasible

    p = report["protocol"]
    p.update({
        "v177_candidate_feasibility_correction": True,
        "candidate_set_loss_scope": "only rows with valid_candidate_count >= true_k",
        "candidate_infeasible_rows_event_set_weight": 0.0,
        "candidate_infeasible_rows_kept_in_outer_evaluation": True,
        "candidate_infeasible_rows_removed_from_dataset": False,
        "candidate_infeasible_definition": "valid_candidate_count < true_k",
        "v173_mass_coefficient_preservation_scope": "all event-set-supervised feasible rows",
        "v177_architecture_changed_by_feasibility_correction": False,
        "v177_parameters_added_by_feasibility_correction": 0,
    })

    by_k = {}
    for value in range(EVENT_QUERIES + 1):
        rows = k == value
        if not np.any(rows):
            continue
        frows = rows & feasible
        irows = rows & infeasible
        probes = [
            v177._candidate_mass_weights_np(value, int(c), ctx["final_spec"])
            for c in np.unique(valid_count[frows])
        ] if np.any(frows) else []
        max_error = max(
            abs(x["new_total_presence_coefficient_mass"] - x["old_total_presence_coefficient_mass"])
            for x in probes
        ) if probes else 0.0
        by_k[str(value)] = {
            "rows": int(np.sum(rows)),
            "feasible_rows": int(np.sum(frows)),
            "infeasible_rows": int(np.sum(irows)),
            "infeasible_rate": float(np.mean(infeasible[rows])),
            "feasible_valid_candidate_count_min": int(np.min(valid_count[frows])) if np.any(frows) else None,
            "feasible_valid_candidate_count_max": int(np.max(valid_count[frows])) if np.any(frows) else None,
            "max_abs_mass_error_on_feasible_rows": float(max_error),
        }

    arch = report["v177"]["architecture"]
    arch["candidate_feasibility"] = {
        "outer_rows": int(len(k)),
        "feasible_rows": int(np.sum(feasible)),
        "infeasible_rows": int(np.sum(infeasible)),
        "infeasible_rate": float(np.mean(infeasible)),
        "shortfall_histogram": {
            str(int(s)): int(np.sum((k - valid_count)[infeasible] == s))
            for s in sorted(np.unique((k - valid_count)[infeasible]).tolist())
        } if np.any(infeasible) else {},
        "by_true_k": by_k,
        "event_set_loss_masked_only_on_infeasible_rows": True,
        "outer_evaluation_includes_infeasible_rows": True,
    }
    # Replace the original all-row algebra audit, whose error is expected on C<K.
    arch["mass_preservation_by_true_k"] = by_k

    return report


def train_fold(args):
    original_loss_factory = v177._candidate_set_loss
    original_postprocess = v177._postprocess

    def corrected_postprocess(a, report, ctx, capture):
        out = original_postprocess(a, report, ctx, capture)
        out = _correct_postprocess_report(out, ctx)
        path = a.output_dir / f"report-fold-{a.outer_fold}.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        return out

    try:
        v177._candidate_set_loss = _candidate_set_loss_feasible
        v177._postprocess = corrected_postprocess
        return v177.train_fold(args)
    finally:
        v177._candidate_set_loss = original_loss_factory
        v177._postprocess = original_postprocess


def parser():
    return v177.parser()


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
