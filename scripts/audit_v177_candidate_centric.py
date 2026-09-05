from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

V104_F1 = 0.8038423365943571
V173_F1 = 0.7920712806963364


def pooled_metric(reports, path):
    tp = fp = fn = 0
    for r in reports:
        obj = r
        for k in path:
            obj = obj[k]
        tp += int(obj["true_positive"])
        fp += int(obj["false_positive"])
        fn += int(obj["false_negative"])
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
        "prediction_count": tp + fp,
        "reference_count": tp + fn,
        "prediction_reference_ratio": (tp + fp) / (tp + fn),
    }


def full_poibin(p):
    p = np.asarray(p, dtype=np.float64)
    n, m = p.shape
    dist = np.zeros((n, m + 1), dtype=np.float64)
    dist[:, 0] = 1.0
    for j in range(m):
        pj = p[:, j]
        old = dist.copy()
        dist[:, 0] = old[:, 0] * (1 - pj)
        dist[:, 1 : j + 2] = old[:, 1 : j + 2] * (1 - pj[:, None]) + old[:, : j + 1] * pj[:, None]
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_paths = sorted(args.input_dir.glob("**/report-fold-*.json"))
    if len(report_paths) != 5:
        raise SystemExit(f"expected 5 reports, got {len(report_paths)}")

    reports = []
    arrays = []
    for rp in report_paths:
        match = re.search(r"report-fold-(\d+)\.json$", rp.name)
        fold_id = int(match.group(1))
        report = json.loads(rp.read_text())
        reports.append(report)
        with np.load(rp.parent / f"predictions-fold-{fold_id}.npz", allow_pickle=False) as z:
            arrays.append(
                {
                    "fold": np.full(len(z["k"]), fold_id, dtype=np.int16),
                    "k": z["k"].astype(np.int16),
                    "objectness": z["candidate_objectness"].astype(np.float32),
                    "valid_count": z["candidate_valid_count"].astype(np.int16),
                }
            )

    reports = sorted(reports, key=lambda r: int(r["outer_fold"]))
    fold = np.concatenate([x["fold"] for x in arrays])
    k = np.concatenate([x["k"] for x in arrays]).astype(int)
    objectness = np.concatenate([x["objectness"] for x in arrays]).astype(np.float64)
    valid = np.concatenate([x["valid_count"] for x in arrays]).astype(int)
    if len(k) != 76768:
        raise SystemExit(f"outer rows {len(k)} != 76768")

    hard = (objectness >= 0.5).sum(axis=1)
    runtime = np.minimum(hard, 6)
    soft = objectness.sum(axis=1)
    poibin = full_poibin(objectness)
    map_count = np.minimum(poibin.argmax(axis=1), 6)
    true_prob = poibin[np.arange(len(k)), k]
    count_nll = -np.log(np.clip(true_prob, 1e-7, 1.0))

    direct = pooled_metric(reports, ["strata", "aggregate", "v177_candidate_centric", "metrics", "global"])
    count_only = pooled_metric(reports, ["v177", "count_only_strata", "aggregate", "global"])

    per_k = {}
    weighting = {}
    for true_k in range(7):
        mask = k == true_k
        per_k[str(true_k)] = {
            "rows": int(mask.sum()),
            "valid_candidate_count_mean": float(valid[mask].mean()),
            "soft_object_count_mean": float(soft[mask].mean()),
            "hard_object_count_mean": float(hard[mask].mean()),
            "hard_fraction_of_valid_mean": float(np.mean(hard[mask] / np.maximum(valid[mask], 1))),
            "runtime_exact_k": float(np.mean(runtime[mask] == true_k)),
            "poibin_map_exact_k": float(np.mean(map_count[mask] == true_k)),
            "active_gt6_rate": float(np.mean(hard[mask] > 6)),
            "corr_valid_count_soft_count": float(np.corrcoef(valid[mask], soft[mask])[0, 1]) if np.std(valid[mask]) > 0 else None,
            "corr_valid_count_hard_count": float(np.corrcoef(valid[mask], hard[mask])[0, 1]) if np.std(valid[mask]) > 0 else None,
            "count_nll_mean": float(count_nll[mask].mean()),
            "c_lt_k_rows": int(np.sum(valid[mask] < true_k)),
            "c_eq_k_rows": int(np.sum(valid[mask] == true_k)),
        }

        row_positive = []
        row_negative = []
        for fold_id in range(5):
            fm = mask & (fold == fold_id)
            if not np.any(fm):
                continue
            mass = reports[fold_id]["v177"]["final_fit_weight_spec"]["mass"]
            positive = float(mass["object_by_k"][true_k])
            old_negative = float(mass["no_object_by_k"][true_k])
            negative_total = (6 - true_k) * old_negative
            c = valid[fm]
            negative = np.where(c > true_k, negative_total / np.maximum(c - true_k, 1), 0.0)
            row_positive.extend([positive] * int(fm.sum()))
            row_negative.extend(negative.tolist())

        positive = np.asarray(row_positive, dtype=np.float64)
        negative = np.asarray(row_negative, dtype=np.float64)
        finite = (negative > 0) & (positive > 0)
        weighting[str(true_k)] = {
            "positive_weight_mean": float(positive.mean()) if len(positive) else 0.0,
            "negative_per_candidate_median": float(np.median(negative)) if len(negative) else 0.0,
            "negative_per_candidate_p25": float(np.quantile(negative, 0.25)) if len(negative) else 0.0,
            "negative_per_candidate_p75": float(np.quantile(negative, 0.75)) if len(negative) else 0.0,
            "positive_to_negative_ratio_median": float(np.median(positive[finite] / negative[finite])) if np.any(finite) else None,
            "zero_negative_weight_rate": float(np.mean(negative == 0)) if len(negative) else 0.0,
        }

    confusion = np.zeros((7, 7), dtype=np.int64)
    for true_k, pred_k in zip(k, runtime):
        confusion[true_k, pred_k] += 1
    confusion_rate = (confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)).tolist()

    identity = {
        "candidate_top1_by_fold": [float(r["event_diagnostics"]["candidate_top1"]) for r in reports],
        "time_mae_ms_by_fold": [float(r["event_diagnostics"]["time_mae_ms"]) for r in reports],
        "time_mae_ms_mean": float(np.mean([r["event_diagnostics"]["time_mae_ms"] for r in reports])),
    }

    infeasible = int(np.sum(valid < k))
    c_eq_k_non6 = int(np.sum((valid == k) & (k < 6) & (k > 0)))

    report = {
        "protocol": {
            "source_run": 33893583081,
            "outer_clean_rows": int(len(k)),
            "threshold": 0.5,
            "no_retraining": True,
            "locked12_touched": False,
        },
        "headline": {
            "v177_direct": direct,
            "v177_count_only": count_only,
            "direct_minus_count_only_f1_pp": 100 * (direct["f1"] - count_only["f1"]),
            "v177_minus_v173_f1_pp": 100 * (direct["f1"] - V173_F1),
            "v177_minus_v104_f1_pp": 100 * (direct["f1"] - V104_F1),
        },
        "cardinality": {
            "true_mean": float(k.mean()),
            "soft_mean": float(soft.mean()),
            "hard_mean_uncapped": float(hard.mean()),
            "runtime_mean_capped": float(runtime.mean()),
            "soft_excess_vs_truth_pct": float(100 * (soft.mean() / k.mean() - 1)),
            "runtime_exact_k": float(np.mean(runtime == k)),
            "poibin_map_exact_k": float(np.mean(map_count == k)),
            "poly_runtime_exact_k": float(np.mean(runtime[k >= 2] == k[k >= 2])),
            "per_true_k": per_k,
            "runtime_confusion_rate": confusion_rate,
        },
        "weighting": weighting,
        "representation": {
            "candidate_infeasible_c_lt_k_rows": infeasible,
            "candidate_infeasible_rate": float(infeasible / len(k)),
            "c_eq_k_lt6_rows_where_legacy_null_mass_has_no_carrier": c_eq_k_non6,
            "candidate_count_to_soft_count_correlation_by_k": {
                str(x): per_k[str(x)]["corr_valid_count_soft_count"] for x in range(1, 7)
            },
            "candidate_count_to_hard_count_correlation_by_k": {
                str(x): per_k[str(x)]["corr_valid_count_hard_count"] for x in range(1, 7)
            },
        },
        "realization": identity,
        "findings": [
            "V17.7 collapse is systematic and direct candidate realization adds almost nothing over count-only.",
            "Candidate identity/time realization is effectively solved once an object is selected.",
            "Legacy six-slot mass preservation is not optimization-equivalent in a variable 1..48 candidate hypothesis field.",
            "Per-candidate negative pressure collapses as K and valid candidate count rise; at K=6 it is exactly zero.",
            "Predicted object count becomes strongly dependent on the number of available candidates, especially K>=3.",
            "The Poisson-binomial MAP does not rescue cardinality; the issue is learned objectness, not only threshold decoding.",
            "V17.7 therefore does not cleanly falsify candidate-centric decoding; it falsifies independent candidate Bernoullis with transplanted six-slot null-mass weighting.",
        ],
    }

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# V17.7 post-audit",
        "",
        f"- direct F1: {100 * direct['f1']:.6f}%",
        f"- count-only F1: {100 * count_only['f1']:.6f}%",
        f"- direct-count delta: {100 * (direct['f1'] - count_only['f1']):+.6f} pp",
        f"- vs V17.3: {100 * (direct['f1'] - V173_F1):+.6f} pp",
        f"- vs V10.4: {100 * (direct['f1'] - V104_F1):+.6f} pp",
        f"- truth mean K: {k.mean():.6f}",
        f"- soft object mass: {soft.mean():.6f} ({100 * (soft.mean() / k.mean() - 1):+.2f}% vs truth)",
        f"- C<K rows: {infeasible}",
        f"- C==K<6 positive rows with no carrier for legacy null mass: {c_eq_k_non6}",
        "",
        "## Per-K",
        "",
        "| K | valid mean | soft | hard | exact | corr(C,soft) | median neg w | median pos/neg |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for true_k in range(7):
        q = per_k[str(true_k)]
        w = weighting[str(true_k)]
        ratio = w["positive_to_negative_ratio_median"]
        lines.append(
            f"| {true_k} | {q['valid_candidate_count_mean']:.3f} | {q['soft_object_count_mean']:.3f} | "
            f"{q['hard_object_count_mean']:.3f} | {100 * q['runtime_exact_k']:.2f}% | "
            f"{q['corr_valid_count_soft_count'] if q['corr_valid_count_soft_count'] is not None else 0:.3f} | "
            f"{w['negative_per_candidate_median']:.4f} | {ratio if ratio is not None else float('inf'):.2f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        "V17.7 is rejected as a model, but the candidate-centric hypothesis is not cleanly rejected.",
        "The dominant confound is the six-slot null-mass transfer into a much larger variable candidate field, which creates negative-gradient starvation and candidate-count-dependent objectness.",
        "A next architecture should introduce competition/normalization at the event-proposal group level rather than independent Bernoulli objectness for raw candidates.",
    ]
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(report["headline"], indent=2))
    print("soft_mean", soft.mean(), "truth", k.mean(), "C<K", infeasible, "C==K<6", c_eq_k_non6)
    for true_k in range(7):
        print("K", true_k, per_k[str(true_k)], weighting[str(true_k)])


if __name__ == "__main__":
    main()
