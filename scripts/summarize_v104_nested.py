"""Aggregate five deployment-matched nested V10.4 outer-fold audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MODELS = ("v101_deployment", "v102_deployment", "v104_probe_ensemble", "v104_deployment")
ARRANGEMENTS = ("global", "solo", "comp")


def _sum_metrics(items):
    tp = sum(int(x["true_positive"]) for x in items)
    fp = sum(int(x["false_positive"]) for x in items)
    fn = sum(int(x["false_negative"]) for x in items)
    pred = sum(int(x["prediction_count"]) for x in items)
    ref = sum(int(x["reference_count"]) for x in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "prediction_count": pred,
        "reference_count": ref,
        "prediction_reference_ratio": pred / ref if ref else 0.0,
    }


def _sum_card(items):
    conf = np.sum([np.asarray(x["confusion_true_rows_pred_columns"], dtype=np.int64) for x in items], axis=0)
    total = int(np.sum(conf))
    exact = int(np.trace(conf))
    classes = np.arange(conf.shape[0], dtype=np.int64)
    mae_num = 0
    for t in range(conf.shape[0]):
        mae_num += int(np.sum(conf[t] * np.abs(classes - t)))
    birth_total = int(np.sum(conf[1:]))
    birth_exact = int(np.trace(conf[1:, 1:]))
    poly_total = int(np.sum(conf[2:]))
    poly_exact = sum(int(conf[k, k]) for k in range(2, conf.shape[0]))
    return {
        "accuracy": exact / total if total else 0.0,
        "birth_cluster_accuracy": birth_exact / birth_total if birth_total else 0.0,
        "poly_cluster_accuracy": poly_exact / poly_total if poly_total else 0.0,
        "mean_absolute_class_error": mae_num / total if total else 0.0,
        "confusion_true_rows_pred_columns": conf.tolist(),
    }


def summarize(args):
    reports = []
    arrays = []
    for outer in range(5):
        rp = sorted(args.input_dir.glob(f"**/v104-nested-eval-{outer}.json"))
        npz = sorted(args.input_dir.glob(f"**/v104-nested-eval-{outer}.npz"))
        if len(rp) != 1 or len(npz) != 1:
            raise RuntimeError(f"missing/duplicate eval artifact for outer {outer}: {len(rp)}/{len(npz)}")
        reports.append(json.loads(rp[0].read_text()))
        with np.load(npz[0], allow_pickle=False) as z:
            arrays.append({k: np.asarray(z[k]) for k in z.files})

    idx = np.concatenate([a["global_index"] for a in arrays]).astype(np.int64)
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    if not np.array_equal(idx, np.arange(len(idx), dtype=np.int64)):
        raise RuntimeError("five outer folds are not exact one-time full train coverage")

    aggregate_metrics = {}
    aggregate_cardinality = {}
    for model in MODELS:
        aggregate_metrics[model] = {}
        for arr in ARRANGEMENTS:
            aggregate_metrics[model][arr] = _sum_metrics(
                [r["outer_metrics"][model][arr] for r in reports]
            )
        aggregate_cardinality[model] = _sum_card(
            [r["outer_cardinality"][model] for r in reports]
        )

    probe_pred = np.concatenate([a["pred104_probe_ensemble"] for a in arrays])[order]
    deploy_pred = np.concatenate([a["pred104_deploy"] for a in arrays])[order]
    agreement = float(np.mean(probe_pred == deploy_pred))

    by_player = {}
    players = sorted({p for r in reports for p in r.get("by_player", {})})
    for player in players:
        present = [r["by_player"][player] for r in reports if player in r.get("by_player", {})]
        by_player[player] = {
            "clusters": int(sum(x["clusters"] for x in present)),
            "probe_ensemble_metrics": {
                arr: _sum_metrics([x["probe_ensemble_metrics"][arr] for x in present])
                for arr in ARRANGEMENTS
            },
            "deployment_metrics": {
                arr: _sum_metrics([x["deployment_metrics"][arr] for x in present])
                for arr in ARRANGEMENTS
            },
            "probe_ensemble_cardinality": _sum_card(
                [x["probe_ensemble_cardinality"] for x in present]
            ),
            "deployment_cardinality": _sum_card(
                [x["deployment_cardinality"] for x in present]
            ),
        }

    per_outer = {
        str(r["outer_fold"]): {
            "groups": r["outer_groups"],
            "selected_epochs": r["data"]["selected_fusion_epochs"],
            "v104_probe_f1": r["outer_metrics"]["v104_probe_ensemble"]["global"]["f1"],
            "v104_deploy_f1": r["outer_metrics"]["v104_deployment"]["global"]["f1"],
            "probe_minus_deploy_f1": r["retraining_effect_same_outer_rows"]["probe_ensemble_minus_deploy_global_f1"],
            "probe_deploy_agreement": r["retraining_effect_same_outer_rows"]["probe_ensemble_vs_deploy_decoded_agreement"],
            "v101_probe_to_deploy_tv": r["retraining_effect_same_outer_rows"]["mean_v101_probe_to_deploy_tv"],
            "v102_probe_to_deploy_tv": r["retraining_effect_same_outer_rows"]["mean_v102_probe_to_deploy_tv"],
        }
        for r in reports
    }

    result = {
        "schema_version": 1,
        "protocol": {
            "train_only_nested_outer_holdout": True,
            "historical_validation_or_locked_evaluated": False,
            "outer_fold_count": 5,
            "exact_one_time_outer_coverage": True,
            "deployment_matched_expert_retraining": True,
        },
        "data": {"clusters": int(len(idx))},
        "aggregate_metrics": aggregate_metrics,
        "aggregate_cardinality": aggregate_cardinality,
        "retraining_effect": {
            "probe_ensemble_vs_deploy_decoded_agreement": agreement,
            "probe_minus_deploy_global_f1": (
                aggregate_metrics["v104_probe_ensemble"]["global"]["f1"]
                - aggregate_metrics["v104_deployment"]["global"]["f1"]
            ),
            "outer_folds_where_probe_beats_deploy": int(sum(
                1 for x in per_outer.values() if x["probe_minus_deploy_f1"] > 0
            )),
            "outer_folds_where_deploy_beats_probe": int(sum(
                1 for x in per_outer.values() if x["probe_minus_deploy_f1"] < 0
            )),
        },
        "per_outer": per_outer,
        "by_player": by_player,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "v101": aggregate_metrics["v101_deployment"]["global"]["f1"],
        "v102": aggregate_metrics["v102_deployment"]["global"]["f1"],
        "v104_probe": aggregate_metrics["v104_probe_ensemble"]["global"]["f1"],
        "v104_deploy": aggregate_metrics["v104_deployment"]["global"]["f1"],
        "probe_minus_deploy": result["retraining_effect"]["probe_minus_deploy_global_f1"],
        "agreement": agreement,
        "probe_wins": result["retraining_effect"]["outer_folds_where_probe_beats_deploy"],
        "deploy_wins": result["retraining_effect"]["outer_folds_where_deploy_beats_probe"],
    }, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    summarize(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
