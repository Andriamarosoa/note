"""Aggregate progressive semitone disappearance trajectories across folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

EXPERIMENT = "v273_semitone_disappearance_fold"
THRESHOLDS = tuple(range(1, 26))
DIRECTIONS = ("le", "ge")


def discover(root):
    found = {}
    for path in root.rglob("report.json"):
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        if report.get("experiment") != EXPERIMENT:
            continue
        fold = int(report["validation_fold"])
        require(fold in FOLDS and fold not in found, "duplicate/bad fold report")
        traj = path.parent / "trajectories.npz"
        require(traj.exists(), "missing trajectories")
        found[fold] = (report, traj)
    require(set(found) == set(FOLDS), f"missing fold reports: {set(FOLDS)-set(found)}")
    return found


def load_rows(path, fold):
    with np.load(path, allow_pickle=False) as z:
        require(set(("row_id","true_k","predicted","probability","steps","first_k2_step",
                     "first_non3_step","first_below3_step","reentered_k3",
                     "monotone_nonincreasing")) <= set(z.files), "trajectory schema drift")
        rows = []
        for i in range(len(z["row_id"])):
            rows.append({
                "row_id": int(z["row_id"][i]),
                "fold": int(fold),
                "true_k": int(z["true_k"][i]),
                "first_k2_step": int(z["first_k2_step"][i]),
                "first_non3_step": int(z["first_non3_step"][i]),
                "first_below3_step": int(z["first_below3_step"][i]),
                "reentered_k3": bool(z["reentered_k3"][i]),
                "monotone_nonincreasing": bool(z["monotone_nonincreasing"][i]),
                "trajectory": np.asarray(z["predicted"][i], np.int32),
            })
        return rows


def apply_rule(feature, direction, threshold):
    feature = np.asarray(feature)
    if direction == "le":
        return feature <= threshold
    if direction == "ge":
        return feature >= threshold
    raise ValueError(direction)


def accounting(rows, direction=None, threshold=None):
    y = np.asarray([r["true_k"] for r in rows], np.int32)
    if direction is None:
        take = np.zeros(len(rows), bool)
    else:
        f = np.asarray([r["first_k2_step"] for r in rows], np.int32)
        take = apply_rule(f, direction, threshold)
    corrections = int(np.sum(take & (y == 2)))
    regressions = int(np.sum(take & (y == 3)))
    return {
        "rows": len(rows),
        "applied": int(take.sum()),
        "corrections": corrections,
        "regressions": regressions,
        "global_net": corrections - regressions,
    }


def select_rule(rows):
    candidates = []
    for direction in DIRECTIONS:
        for threshold in THRESHOLDS:
            q = accounting(rows, direction, threshold)
            candidates.append({
                "direction": direction,
                "threshold": threshold,
                **q,
            })
    order = {"le": 0, "ge": 1}
    best = min(
        candidates,
        key=lambda q: (
            -q["global_net"],
            q["regressions"],
            q["applied"],
            q["threshold"],
            order[q["direction"]],
        ),
    )
    if best["global_net"] <= 0:
        return None, candidates
    return best, candidates


def auc_feature(rows, key):
    y = np.asarray([r["true_k"] == 2 for r in rows], np.int32)
    x = np.asarray([r[key] for r in rows], np.float64)
    if len(np.unique(y)) < 2:
        return None
    raw = float(roc_auc_score(y, x))
    return {
        "raw": raw,
        "oriented": max(raw, 1.0 - raw),
        "direction": "later_implies_K2" if raw >= .5 else "earlier_implies_K2",
    }


def class_stats(rows, key, true_k):
    x = np.asarray([r[key] for r in rows if r["true_k"] == true_k], np.float64)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, .25)),
        "q75": float(np.quantile(x, .75)),
        "censored25": int(np.sum(x == 25)),
    }


def survival(rows, true_k):
    selected = [r for r in rows if r["true_k"] == true_k]
    return {
        str(step): {
            "rows": len(selected),
            "still_k3": int(sum(int(r["trajectory"][step]) == 3 for r in selected)),
            "at_k2": int(sum(int(r["trajectory"][step]) == 2 for r in selected)),
        }
        for step in range(25)
    }


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    found = discover(a.input_root)
    rows = []
    source = {}
    for fold in FOLDS:
        report, path = found[fold]
        require(report["plus0_prediction_parity"], "+0 parity failed upstream")
        require(report["plus0_spectral_max_abs_error"] <= 1e-3, "+0 map replay failed upstream")
        fold_rows = load_rows(path, fold)
        require(len(fold_rows) == report["rows"], "row count mismatch")
        rows += fold_rows
        source[str(fold)] = {
            "report_sha256": sha256_file(path.parent / "report.json"),
            "trajectories_sha256": sha256_file(path),
            "rows": len(fold_rows),
            "auc": report["first_k2_oriented_auc"],
            "direction": report["first_k2_direction"],
        }

    require(len(rows) == 488, "frozen cohort total changed")
    require(len({r["row_id"] for r in rows}) == len(rows), "duplicate cross-fold row")

    rotations = []
    for fold in FOLDS:
        fit = [r for r in rows if r["fold"] != fold]
        val = [r for r in rows if r["fold"] == fold]
        selected, candidates = select_rule(fit)
        if selected is None:
            val_result = accounting(val)
        else:
            val_result = accounting(val, selected["direction"], selected["threshold"])
        rotations.append({
            "fold": fold,
            "fit_rows": len(fit),
            "val_rows": len(val),
            "selected_rule": None if selected is None else {
                "direction": selected["direction"],
                "threshold": selected["threshold"],
                "fit_net": selected["global_net"],
                "fit_corrections": selected["corrections"],
                "fit_regressions": selected["regressions"],
                "fit_applied": selected["applied"],
            },
            "val_result": val_result,
            "val_auc_first_k2": auc_feature(val, "first_k2_step"),
            "fit_top5_rules": sorted(
                candidates,
                key=lambda q: (-q["global_net"], q["regressions"], q["applied"],
                               q["threshold"], 0 if q["direction"] == "le" else 1)
            )[:5],
        })

    total = {
        key: int(sum(r["val_result"][key] for r in rotations))
        for key in ("rows", "applied", "corrections", "regressions", "global_net")
    }

    y = np.asarray([r["true_k"] for r in rows])
    report = {
        "status": "completed",
        "experiment": "v273_semitone_disappearance_summary",
        "rows": len(rows),
        "K2_rows": int(np.sum(y == 2)),
        "K3_rows": int(np.sum(y == 3)),
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "feature": "first_k2_step; 25 means no K2 through +24",
        "global_auc": {
            key: auc_feature(rows, key)
            for key in ("first_k2_step", "first_non3_step", "first_below3_step")
        },
        "global_class_stats": {
            key: {
                "K2": class_stats(rows, key, 2),
                "K3": class_stats(rows, key, 3),
            }
            for key in ("first_k2_step", "first_non3_step", "first_below3_step")
        },
        "global_reentered_k3_rate": {
            str(k): float(np.mean([r["reentered_k3"] for r in rows if r["true_k"] == k]))
            for k in (2, 3)
        },
        "global_monotone_nonincreasing_rate": {
            str(k): float(np.mean([r["monotone_nonincreasing"] for r in rows if r["true_k"] == k]))
            for k in (2, 3)
        },
        "survival": {str(k): survival(rows, k) for k in (2, 3)},
        "rotations": rotations,
        "selected_policy_total": total,
        "correct_all_reference": {
            "corrections": int(np.sum(y == 2)),
            "regressions": int(np.sum(y == 3)),
            "global_net": int(np.sum(y == 2) - np.sum(y == 3)),
        },
        "source_folds": source,
        "automatic_promotion": False,
        "limitations": [
            "Only frozen K2/K3 B_low/base-K3 cases are included; no other-K action accounting.",
            "Only spectral maps are transposed; temporal candidate features remain fixed.",
            "Internal folds have been previously inspected.",
        ],
    }

    a.output.mkdir(parents=True)
    write_json(a.output / "report.json", report)

    f = report["global_class_stats"]["first_k2_step"]
    auc = report["global_auc"]["first_k2_step"]
    lines = [
        "# Progressive semitone disappearance — multi-fold summary",
        "",
        f"Frozen cohort: **{len(rows)} rows** (K2={report['K2_rows']}, K3={report['K3_rows']}).",
        "",
        "| class | median first K2 | q25 | q75 | no K2 through +24 | reentry K3 | monotone K |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in (2, 3):
        q = f["K"+str(k)]
        lines.append(
            f"| K{k} | {q['median']:.2f} | {q['q25']:.2f} | {q['q75']:.2f} | "
            f"{q['censored25']} | {report['global_reentered_k3_rate'][str(k)]:.3f} | "
            f"{report['global_monotone_nonincreasing_rate'][str(k)]:.3f} |"
        )
    lines += [
        "",
        f"Global first-K2 oriented AUC: **{auc['oriented']:.3f}** ({auc['direction']}).",
        "",
        "| VAL fold | FIT rule | FIT net | corrections | regressions | VAL net |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rotations:
        s = r["selected_rule"]
        if s is None:
            rule, fitnet = "abstain", 0
        else:
            op = "<=" if s["direction"] == "le" else ">="
            rule = f"first_k2 {op} {s['threshold']}"
            fitnet = s["fit_net"]
        q = r["val_result"]
        lines.append(
            f"| {r['fold']} | {rule} | {fitnet:+d} | "
            f"{q['corrections']} | {q['regressions']} | {q['global_net']:+d} |"
        )
    lines += [
        "",
        f"FIT-selected policy total: **{total['corrections']} corrections / "
        f"{total['regressions']} regressions = net {total['global_net']:+d}**.",
        f"Correct-all reference on this cohort: **{report['correct_all_reference']['global_net']:+d}**.",
        "",
        "Fold 3 excluded. No automatic promotion.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
