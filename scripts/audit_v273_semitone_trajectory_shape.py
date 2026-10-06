"""Cross-fold classifier on full K trajectories from semitone pitch shift."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

PROTOCOL = Path("analysis/v273-semitone-trajectory-shape-protocol.md")
SOURCE_EXPERIMENT = "v273_semitone_disappearance_fold"
FEATURES = (
    "fraction_k2",
    "fraction_k3",
    "fraction_below3",
    "transitions_3_to_2",
    "transitions_2_to_3",
    "transitions_total",
    "longest_k2_run",
    "longest_k3_run_after_exit",
    "total_variation",
    "unique_k_count",
    "state_entropy",
    "final_k",
    "first_k2_step",
    "first_non3_step",
    "first_below3_step",
    "reentered_k3",
    "monotone_nonincreasing",
)


def discover(root):
    found = {}
    for p in root.rglob("report.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("experiment") != SOURCE_EXPERIMENT:
            continue
        fold = int(r["validation_fold"])
        require(fold in FOLDS and fold not in found, "duplicate or invalid source fold")
        traj = p.parent / "trajectories.npz"
        require(traj.exists(), "missing trajectories.npz")
        found[fold] = (r, traj)
    require(set(found) == set(FOLDS), f"missing source folds: {set(FOLDS)-set(found)}")
    return found


def longest_run(values, target):
    best = 0
    cur = 0
    for v in values:
        if int(v) == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def entropy(values):
    x = np.asarray(values, np.int32)
    counts = np.bincount(x, minlength=7).astype(np.float64)
    p = counts[counts > 0] / len(x)
    return float(-np.sum(p * np.log2(p))) if len(p) else 0.0


def feature_row(ktraj, first_k2, first_non3, first_below3, reentered, monotone):
    ktraj = np.asarray(ktraj, np.int32)
    require(ktraj.shape == (25,) and int(ktraj[0]) == 3, "trajectory shape/start drift")
    tail = ktraj[1:]
    diffs = np.diff(ktraj)
    first_exit = int(first_non3)
    after_exit = ktraj[min(first_exit + 1, len(ktraj)):] if first_exit < 25 else np.empty(0, np.int32)
    vals = {
        "fraction_k2": float(np.mean(tail == 2)),
        "fraction_k3": float(np.mean(tail == 3)),
        "fraction_below3": float(np.mean(tail < 3)),
        "transitions_3_to_2": float(np.sum((ktraj[:-1] == 3) & (ktraj[1:] == 2))),
        "transitions_2_to_3": float(np.sum((ktraj[:-1] == 2) & (ktraj[1:] == 3))),
        "transitions_total": float(np.sum(diffs != 0)),
        "longest_k2_run": float(longest_run(tail, 2)),
        "longest_k3_run_after_exit": float(longest_run(after_exit, 3)),
        "total_variation": float(np.sum(np.abs(diffs))),
        "unique_k_count": float(len(np.unique(ktraj))),
        "state_entropy": entropy(tail),
        "final_k": float(ktraj[-1]),
        "first_k2_step": float(first_k2),
        "first_non3_step": float(first_non3),
        "first_below3_step": float(first_below3),
        "reentered_k3": float(bool(reentered)),
        "monotone_nonincreasing": float(bool(monotone)),
    }
    return np.asarray([vals[name] for name in FEATURES], np.float64), vals


def load_rows(found):
    rows = []
    for fold in FOLDS:
        report, path = found[fold]
        require(report["plus0_prediction_parity"], "source +0 parity failed")
        with np.load(path, allow_pickle=False) as z:
            require(len(z["row_id"]) == report["rows"], "source row mismatch")
            for i in range(len(z["row_id"])):
                vec, named = feature_row(
                    z["predicted"][i],
                    int(z["first_k2_step"][i]),
                    int(z["first_non3_step"][i]),
                    int(z["first_below3_step"][i]),
                    bool(z["reentered_k3"][i]),
                    bool(z["monotone_nonincreasing"][i]),
                )
                rows.append({
                    "row_id": int(z["row_id"][i]),
                    "fold": fold,
                    "true_k": int(z["true_k"][i]),
                    "X": vec,
                    "features": named,
                })
    require(len(rows) == 488 and len({r["row_id"] for r in rows}) == 488, "cohort drift")
    require(all(r["true_k"] in (2, 3) for r in rows), "non K2/K3 row")
    return rows


def model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=39431,
        )),
    ])


def stats(values):
    x = np.asarray(values, np.float64)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, .25)),
        "q75": float(np.quantile(x, .75)),
        "std": float(x.std()),
    }


def univariate(rows):
    y = np.asarray([r["true_k"] == 2 for r in rows], np.int32)
    X = np.stack([r["X"] for r in rows])
    out = {}
    for j, name in enumerate(FEATURES):
        raw = float(roc_auc_score(y, X[:, j]))
        out[name] = {
            "K2": stats(X[y == 1, j]),
            "K3": stats(X[y == 0, j]),
            "raw_auc": raw,
            "oriented_auc": max(raw, 1.0 - raw),
            "direction": "K2_higher" if raw >= .5 else "K2_lower",
        }
    return out


def evaluate_rotation(rows, val_fold):
    fit = [r for r in rows if r["fold"] != val_fold]
    val = [r for r in rows if r["fold"] == val_fold]
    Xf = np.stack([r["X"] for r in fit])
    yf = np.asarray([r["true_k"] == 2 for r in fit], np.int32)
    Xv = np.stack([r["X"] for r in val])
    yv = np.asarray([r["true_k"] == 2 for r in val], np.int32)

    m = model()
    m.fit(Xf, yf)
    p = m.predict_proba(Xv)[:, 1]
    action = p >= .5
    corrections = int(np.sum(action & (yv == 1)))
    regressions = int(np.sum(action & (yv == 0)))
    auc = float(roc_auc_score(yv, p))

    scale = m.named_steps["scale"]
    lr = m.named_steps["lr"]
    corr = np.corrcoef(scale.transform(Xf), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "fold": val_fold,
        "fit_rows": len(fit),
        "val_rows": len(val),
        "val_K2": int(yv.sum()),
        "val_K3": int((yv == 0).sum()),
        "val_auc": auc,
        "applied": int(action.sum()),
        "corrections": corrections,
        "regressions": regressions,
        "global_net": corrections - regressions,
        "coefficients_standardized": {
            name: float(value) for name, value in zip(FEATURES, lr.coef_[0])
        },
        "intercept": float(lr.intercept_[0]),
        "fit_correlation": corr.tolist(),
        "fit_univariate": univariate(fit),
        "val_univariate": univariate(val),
    }


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")
    found = discover(a.input_root)
    rows = load_rows(found)

    rotations = [evaluate_rotation(rows, fold) for fold in FOLDS]
    total = {
        key: int(sum(r[key] for r in rotations))
        for key in ("val_rows", "applied", "corrections", "regressions", "global_net")
    }
    mean_auc = float(np.mean([r["val_auc"] for r in rotations]))
    min_auc = float(np.min([r["val_auc"] for r in rotations]))
    nonnegative_folds = int(sum(r["global_net"] >= 0 for r in rotations))
    positive_folds = int(sum(r["global_net"] > 0 for r in rotations))
    useful = bool(total["global_net"] > 0 and nonnegative_folds >= 3 and mean_auc > .55)

    report = {
        "status": "completed",
        "experiment": "v273_semitone_trajectory_shape",
        "features": list(FEATURES),
        "classifier": "StandardScaler + LogisticRegression C=1 balanced lbfgs threshold=.5",
        "rows": len(rows),
        "K2_rows": int(sum(r["true_k"] == 2 for r in rows)),
        "K3_rows": int(sum(r["true_k"] == 3 for r in rows)),
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "rotations": rotations,
        "total": total,
        "mean_val_auc": mean_auc,
        "min_val_auc": min_auc,
        "nonnegative_folds": nonnegative_folds,
        "positive_folds": positive_folds,
        "global_univariate": univariate(rows),
        "decision": {
            "criteria_met": useful,
            "requires_net_positive": total["global_net"] > 0,
            "requires_at_least_3_nonnegative_folds": nonnegative_folds >= 3,
            "requires_mean_auc_gt_055": mean_auc > .55,
            "automatic_promotion": False,
        },
        "source_sha256": {
            "protocol": sha256_file(PROTOCOL),
            "script": sha256_file(__file__),
            **{
                f"fold_{fold}_trajectories": sha256_file(found[fold][1])
                for fold in FOLDS
            },
        },
        "limitations": [
            "Trajectory features derive from pitch-shifted spectral maps while temporal candidate inputs remain frozen.",
            "Internal folds were already inspected in prior development.",
            "No feature subset selection was attempted; coefficients are descriptive.",
        ],
    }

    a.output.mkdir(parents=True)
    write_json(a.output / "report.json", report)

    lines = [
        "# Full semitone trajectory-shape audit",
        "",
        "| VAL fold | AUC | actions | corrections | regressions | net |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rotations:
        lines.append(
            f"| {r['fold']} | {r['val_auc']:.3f} | {r['applied']} | "
            f"{r['corrections']} | {r['regressions']} | {r['global_net']:+d} |"
        )
    lines += [
        "",
        f"Total: **{total['corrections']} corrections / {total['regressions']} regressions "
        f"= net {total['global_net']:+d}**.",
        f"Mean VAL AUC: **{mean_auc:.3f}**; min fold AUC: **{min_auc:.3f}**.",
        f"Non-negative folds: **{nonnegative_folds}/4**; positive folds: **{positive_folds}/4**.",
        f"Pre-registered usefulness criteria met: **{useful}**.",
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
