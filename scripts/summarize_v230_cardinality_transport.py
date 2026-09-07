"""Aggregate the pre-V23 cardinality/transport audit over the five outer-clean folds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

V104_F1 = 0.8038423365943571
TOLS = ("exact", "dt1df1", "dt1df2", "dt2df4")
BUDGETS = ("K", "2K", "4K", "8K", "48")
MODELS = ("v19", "conv_global_probe", "dense_score_probe", "combined_probe", "true_k_oracle_existing_candidate_ranking")


def metric_sum(rows):
    tp = sum(int(r["true_positive"]) for r in rows); fp = sum(int(r["false_positive"]) for r in rows); fn = sum(int(r["false_negative"]) for r in rows)
    pred = sum(int(r["prediction_count"]) for r in rows); ref = sum(int(r["reference_count"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"f1": f1, "precision": p, "recall": rc, "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "prediction_count": pred, "reference_count": ref, "prediction_reference_ratio": pred / ref if ref else None}


def card(k, pred):
    k = np.asarray(k, np.int32); pred = np.asarray(pred, np.int32); poly = k >= 2
    per = {}
    vals = []
    for value in range(7):
        m = k == value; ex = float(np.mean(pred[m] == value)) if np.any(m) else None
        if ex is not None: vals.append(ex)
        per[str(value)] = {"rows": int(np.sum(m)), "exact": ex, "mean_predicted_k": float(np.mean(pred[m])) if np.any(m) else None}
    return {"exact": float(np.mean(pred == k)), "balanced_exact": float(np.mean(vals)),
            "poly_exact": float(np.mean(pred[poly] == k[poly])) if np.any(poly) else None,
            "mae": float(np.mean(np.abs(pred-k))), "mean_predicted_k": float(np.mean(pred)), "mean_true_k": float(np.mean(k)), "per_true_k": per}


def load(root):
    reports = []; parts = []
    for fold in range(5):
        rps = list(Path(root).glob(f"**/report-fold-{fold}.json"))
        if len(rps) != 1: raise RuntimeError(f"fold {fold}: expected one report, got {len(rps)}")
        r = json.loads(rps[0].read_text()); p = r["protocol"]
        assert p["pre_v23_cardinality_transport_audit"] is True and p["audit_only_no_v23_model"] is True
        assert p["probe_training_rows_final_fit_only"] is True and p["outer_rows_untouched_until_probe_fit"] is True
        assert p["ridge_sweep"] is False and p["runtime_threshold_sweep"] is False
        assert p["locked12_indexed_or_evaluated"] is False
        reports.append(r)
        npzs = list(rps[0].parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1: raise RuntimeError(f"fold {fold}: missing predictions")
        with np.load(npzs[0], allow_pickle=False) as z: parts.append({k: np.asarray(z[k]) for k in z.files})
    keys = set(parts[0])
    for p in parts[1:]: keys &= set(p)
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.argsort(merged["global_index"], kind="stable"); merged = {k:v[order] for k,v in merged.items()}
    if len(merged["global_index"]) != 76768 or len(np.unique(merged["global_index"])) != 76768: raise RuntimeError("invalid outer-clean coverage")
    return reports, merged


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input-dir", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    reports, m = load(args.input_dir); k = np.asarray(m["k"], np.int32)
    preds = {"v19": np.asarray(m["pred_v19"], np.int32), "conv_global_probe": np.asarray(m["pred_conv"], np.int32),
             "dense_score_probe": np.asarray(m["pred_score"], np.int32), "combined_probe": np.asarray(m["pred_combined"], np.int32),
             "true_k_oracle_existing_candidate_ranking": k.copy()}
    aggregate_metrics = {}
    for name in MODELS:
        aggregate_metrics[name] = metric_sum([r["end_to_end_count_realization"][name]["global"] for r in reports])
    cards = {name: card(k, pred) for name, pred in preds.items()}
    eligible = np.asarray(m["eligible"], np.int8) > 0; distinct = np.asarray(m["distinct"], np.int32)
    rep = eligible & (k > 0) & (distinct == k); rep_poly = rep & (k >= 2); eligible_poly = eligible & (k >= 2)
    coverage = {}
    for tol in TOLS:
        coverage[tol] = {}
        for budget in BUDGETS:
            a = np.asarray(m[f"match_{tol}_{budget}"], np.float64)
            row = {"representable_positive_rows": int(np.sum(rep)), "representable_poly_rows": int(np.sum(rep_poly)),
                   "full_recovery_positive": float(np.mean(a[rep] >= 1-1e-7)), "mean_recovered_fraction_positive": float(np.mean(a[rep])),
                   "full_recovery_poly": float(np.mean(a[rep_poly] >= 1-1e-7)), "mean_recovered_fraction_poly": float(np.mean(a[rep_poly])),
                   "per_true_k": {}}
            for value in range(1,7):
                mk = rep & (k == value); row["per_true_k"][str(value)] = {"rows": int(np.sum(mk)),
                    "full_recovery": float(np.mean(a[mk] >= 1-1e-7)) if np.any(mk) else None,
                    "mean_recovered_fraction": float(np.mean(a[mk])) if np.any(mk) else None}
            coverage[tol][budget] = row
    def depth(name):
        a = np.asarray(m[name], np.int32); solved = rep_poly & (a >= 0); vals = a[solved]
        return {"representable_poly_rows": int(np.sum(rep_poly)), "recoverable_within_top48_rate": float(np.mean(a[rep_poly] >= 0)),
                "median_min_rank_depth_when_recoverable": float(np.median(vals)) if len(vals) else None,
                "p90_min_rank_depth_when_recoverable": float(np.percentile(vals,90)) if len(vals) else None}
    hard_examples = np.asarray([r["local_scorer"]["outer_auc_examples"] for r in reports], np.float64)
    old_auc = np.asarray([r["local_scorer"]["old_v19_hard_negative_auc"] for r in reports], np.float64)
    new_auc = np.asarray([r["local_scorer"]["shared_conv_ridge_hard_negative_auc"] for r in reports], np.float64)
    topk = coverage["dt1df2"]["K"]["full_recovery_poly"]; top48 = coverage["dt1df2"]["48"]["full_recovery_poly"]
    result = {
      "schema_version":1,
      "protocol":{"pre_v23_cardinality_transport_audit":True,"outer_clean_rows":76768,"five_outer_folds":True,"locked12_indexed_or_evaluated":False,
                  "no_v23_model_trained":True,"no_threshold_or_hyperparameter_sweep":True,"rank_budgets_diagnostic_only":list(BUDGETS)},
      "representation":{"eligible_positive_rows":int(np.sum(eligible & (k>0))),"eligible_poly_rows":int(np.sum(eligible_poly)),
                        "distinct_representable_positive_rows":int(np.sum(rep)),"distinct_representable_poly_rows":int(np.sum(rep_poly)),
                        "distinct_center_rate_eligible_poly":float(np.mean(distinct[eligible_poly] == k[eligible_poly]))},
      "local_scorer":{"weighted_old_v19_hard_auc":float(np.average(old_auc,weights=hard_examples)),
                      "weighted_shared_conv_ridge_hard_auc":float(np.average(new_auc,weights=hard_examples)),"outer_auc_examples":int(np.sum(hard_examples))},
      "cardinality":cards,"end_to_end_count_realization":aggregate_metrics,
      "transport":{"coverage":coverage,"min_rank_depth_dt1df2":depth("min_depth_dt1df2"),"min_rank_depth_dt2df4":depth("min_depth_dt2df4"),
                   "topk_redundancy_representable_poly":{"mean_close_pair_fraction_dt1df2":float(np.mean(np.asarray(m["close_pair_fraction"])[rep_poly])),
                                                         "rows_with_any_close_pair_dt1df2":float(np.mean(np.asarray(m["any_close_pair"])[rep_poly] > 0))}},
      "comparison":{"protected_v104_f1":V104_F1,"v19_f1":aggregate_metrics["v19"]["f1"],
                    "combined_count_probe_f1":aggregate_metrics["combined_probe"]["f1"],
                    "true_k_oracle_f1":aggregate_metrics["true_k_oracle_existing_candidate_ranking"]["f1"],
                    "combined_minus_v19_f1_pp":100*(aggregate_metrics["combined_probe"]["f1"]-aggregate_metrics["v19"]["f1"]),
                    "true_k_oracle_minus_v19_f1_pp":100*(aggregate_metrics["true_k_oracle_existing_candidate_ranking"]["f1"]-aggregate_metrics["v19"]["f1"]),
                    "true_k_oracle_minus_v104_f1_pp":100*(aggregate_metrics["true_k_oracle_existing_candidate_ranking"]["f1"]-V104_F1),
                    "poly_dt1df2_topk_full":topk,"poly_dt1df2_top48_full":top48,"poly_dt1df2_transport_headroom_pp":100*(top48-topk)},
      "folds":{str(r["outer_fold"]):{"v19_f1":r["end_to_end_count_realization"]["v19"]["global"]["f1"],
                                    "combined_probe_f1":r["end_to_end_count_realization"]["combined_probe"]["global"]["f1"],
                                    "true_k_oracle_f1":r["end_to_end_count_realization"]["true_k_oracle_existing_candidate_ranking"]["global"]["f1"],
                                    "hard_auc":r["local_scorer"]["shared_conv_ridge_hard_negative_auc"],
                                    "top48_dt1df2_poly":r["transport"]["coverage"]["dt1df2"]["48"]["full_recovery_poly"]} for r in reports}
    }
    (args.output_dir/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(args.output_dir/"predictions.npz", **m)
    print(json.dumps(result["comparison"],indent=2,sort_keys=True)); print(json.dumps(result["local_scorer"],indent=2,sort_keys=True))

if __name__ == "__main__": main()
