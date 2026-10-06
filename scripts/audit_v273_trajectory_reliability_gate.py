"""Nested reliability gate for semitone-trajectory K3->K2 actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.v273_residual_audit import FOLDS, require, write_json
from scripts.audit_v273_semitone_trajectory_shape import FEATURES as TRAJ_FEATURES, feature_row

SOURCE_TRAJ_EXPERIMENT = "v273_semitone_disappearance_fold"
GATE_FEATURES_A = tuple(TRAJ_FEATURES) + ("p_traj_k2",)
GATE_FEATURES_B = GATE_FEATURES_A + (
    "best_pair_residual_ratio",
    "best_triplet_residual_ratio",
    "median_triplet_f0",
    "base_probability_k2",
    "base_probability_k3",
    "base_margin_k3_minus_k2",
)


def clf(seed):
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        )),
    ])


def discover_traj(root):
    out = {}
    for p in root.rglob("report.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("experiment") != SOURCE_TRAJ_EXPERIMENT:
            continue
        f = int(r["validation_fold"])
        require(f in FOLDS and f not in out, "duplicate/bad trajectory fold")
        npz = p.parent / "trajectories.npz"
        require(npz.exists(), "missing trajectory npz")
        out[f] = npz
    require(set(out) == set(FOLDS), "missing trajectory folds")
    return out


def load_trajectory_rows(root):
    found = discover_traj(root)
    rows = {}
    for fold, path in found.items():
        with np.load(path, allow_pickle=False) as z:
            for i in range(len(z["row_id"])):
                vec, named = feature_row(
                    z["predicted"][i],
                    int(z["first_k2_step"][i]),
                    int(z["first_non3_step"][i]),
                    int(z["first_below3_step"][i]),
                    bool(z["reentered_k3"][i]),
                    bool(z["monotone_nonincreasing"][i]),
                )
                rid = int(z["row_id"][i])
                require(rid not in rows, "duplicate trajectory row")
                rows[rid] = {
                    "row_id": rid,
                    "fold": fold,
                    "true_k": int(z["true_k"][i]),
                    "traj": vec,
                    "traj_named": named,
                }
    require(len(rows) == 488, "trajectory cohort drift")
    return rows


def discover_residual(root):
    out = {}
    for p in root.rglob("replay.npz"):
        fold = None
        with np.load(p, allow_pickle=False) as z:
            vals = np.unique(z["val_fold"])
            if len(vals) == 1:
                fold = int(vals[0])
        if fold in FOLDS:
            require(fold not in out, "duplicate residual fold")
            out[fold] = p
    require(set(out) == set(FOLDS), f"missing residual folds {set(FOLDS)-set(out)}")
    return out


def attach_residual(rows, root):
    found = discover_residual(root)
    attached = 0
    for fold, path in found.items():
        with np.load(path, allow_pickle=False) as z:
            full_id_to_index = {int(v): i for i, v in enumerate(z["val_ids"])}
            for j, rid0 in enumerate(z["val_action_ids"]):
                rid = int(rid0)
                if rid not in rows:
                    continue
                require(rows[rid]["fold"] == fold, "fold mismatch")
                require(bool(z["val_valid"][j]), "frozen cohort contains invalid residual")
                k = int(z["val_true_k"][full_id_to_index[rid]])
                require(k == rows[rid]["true_k"], "true-k mismatch")
                p = np.asarray(z["val_base_probability"][full_id_to_index[rid]], np.float64)
                x = np.asarray(z["val_X"][j], np.float64)
                rows[rid]["acoustic"] = np.asarray([
                    x[0], x[1], float(z["val_f0"][j]),
                    p[2], p[3], p[3] - p[2],
                ], np.float64)
                attached += 1
    require(attached == 488 and all("acoustic" in r for r in rows.values()),
            f"residual attach mismatch {attached}")
    return rows


def ordered(rows):
    return [rows[k] for k in sorted(rows)]


def fit_level1(train_rows):
    X = np.stack([r["traj"] for r in train_rows])
    y = np.asarray([r["true_k"] == 2 for r in train_rows], np.int32)
    require(len(np.unique(y)) == 2, "level1 binary collapse")
    m = clf(39431)
    m.fit(X, y)
    return m


def level1_predict(model, rows):
    X = np.stack([r["traj"] for r in rows])
    p = model.predict_proba(X)[:, 1]
    return p, p >= .5


def gate_matrix(rows, ptraj, arm):
    base = np.column_stack([np.stack([r["traj"] for r in rows]), np.asarray(ptraj)])
    if arm == "trajectory_only":
        return base
    if arm == "trajectory_plus_acoustic":
        return np.column_stack([base, np.stack([r["acoustic"] for r in rows])])
    raise ValueError(arm)


def collect_oof_actions(all_rows, outer_fold, arm):
    fit_folds = [f for f in FOLDS if f != outer_fold]
    gx, gy = [], []
    meta = []
    for held in fit_folds:
        train = [r for r in all_rows if r["fold"] in fit_folds and r["fold"] != held]
        held_rows = [r for r in all_rows if r["fold"] == held]
        m = fit_level1(train)
        p, action = level1_predict(m, held_rows)
        Xg = gate_matrix(held_rows, p, arm)
        for i, take in enumerate(action):
            if not take:
                continue
            gx.append(Xg[i])
            gy.append(1 if held_rows[i]["true_k"] == 2 else 0)
            meta.append({
                "row_id": held_rows[i]["row_id"],
                "fold": held,
                "correct_action": bool(gy[-1]),
            })
    X = np.asarray(gx, np.float64)
    y = np.asarray(gy, np.int32)
    return X, y, meta


def accounting(rows, action):
    y = np.asarray([r["true_k"] for r in rows], np.int32)
    action = np.asarray(action, bool)
    corrections = int(np.sum(action & (y == 2)))
    regressions = int(np.sum(action & (y == 3)))
    return {
        "rows": len(rows),
        "actions": int(action.sum()),
        "corrections": corrections,
        "regressions": regressions,
        "net": corrections - regressions,
    }


def evaluate_fold(all_rows, outer_fold):
    fit = [r for r in all_rows if r["fold"] != outer_fold]
    val = [r for r in all_rows if r["fold"] == outer_fold]

    level1 = fit_level1(fit)
    pval, raw_action = level1_predict(level1, val)
    raw = accounting(val, raw_action)

    result = {
        "fold": outer_fold,
        "raw": raw,
        "arms": {},
    }

    for arm in ("trajectory_only", "trajectory_plus_acoustic"):
        Xg, yg, meta = collect_oof_actions(all_rows, outer_fold, arm)
        entry = {
            "oof_action_rows": int(len(yg)),
            "oof_correct": int(np.sum(yg == 1)),
            "oof_regression": int(np.sum(yg == 0)),
            "status": "completed",
        }
        if len(yg) < 24 or len(np.unique(yg)) < 2:
            entry["status"] = "abstain_insufficient_oof"
            gated_action = np.zeros(len(val), bool)
            entry["val_reliability_auc"] = None
            entry["result"] = accounting(val, gated_action)
            entry["blocked_raw_actions"] = raw["actions"]
            result["arms"][arm] = entry
            continue

        gate = clf(39531)
        gate.fit(Xg, yg)

        Xv = gate_matrix(val, pval, arm)
        grel = gate.predict_proba(Xv)[:, 1]
        gated_action = raw_action & (grel >= .5)
        val_action_mask = raw_action
        val_rel_auc = None
        if np.sum(val_action_mask) and len(np.unique(
            np.asarray([r["true_k"] == 2 for r in val], np.int32)[val_action_mask]
        )) == 2:
            yy = np.asarray([r["true_k"] == 2 for r in val], np.int32)[val_action_mask]
            val_rel_auc = float(roc_auc_score(yy, grel[val_action_mask]))

        entry.update({
            "val_reliability_auc": val_rel_auc,
            "result": accounting(val, gated_action),
            "blocked_raw_actions": int(np.sum(raw_action & ~gated_action)),
            "gate_coefficients_standardized": {
                name: float(v) for name, v in zip(
                    GATE_FEATURES_A if arm == "trajectory_only" else GATE_FEATURES_B,
                    gate.named_steps["lr"].coef_[0]
                )
            },
        })
        result["arms"][arm] = entry

    return result


def aggregate(folds, arm):
    keys = ("actions", "corrections", "regressions", "net")
    return {
        k: int(sum(f["arms"][arm]["result"][k] for f in folds))
        for k in keys
    }


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    rows = load_trajectory_rows(a.trajectory_root)
    attach_residual(rows, a.residual_root)
    all_rows = ordered(rows)

    folds = [evaluate_fold(all_rows, f) for f in FOLDS]
    raw_total = {
        k: int(sum(f["raw"][k] for f in folds))
        for k in ("actions", "corrections", "regressions", "net")
    }
    a_total = aggregate(folds, "trajectory_only")
    b_total = aggregate(folds, "trajectory_plus_acoustic")

    b_nonneg = sum(f["arms"]["trajectory_plus_acoustic"]["result"]["net"] >= 0 for f in folds)
    b_worst = min(f["arms"]["trajectory_plus_acoustic"]["result"]["net"] for f in folds)
    kept_fraction = b_total["corrections"] / max(1, raw_total["corrections"])
    criteria = {
        "improves_over_raw_net": b_total["net"] > raw_total["net"],
        "net_positive": b_total["net"] > 0,
        "at_least_3_nonnegative_folds": b_nonneg >= 3,
        "worst_fold_not_below_minus5": b_worst >= -5,
        "keeps_at_least_25pct_raw_corrections": kept_fraction >= .25,
    }
    criteria["all_met"] = all(criteria.values())

    report = {
        "status": "completed",
        "experiment": "v273_nested_trajectory_reliability_gate",
        "rows": len(all_rows),
        "folds": folds,
        "outer_fold_3_used": False,
        "raw_total": raw_total,
        "trajectory_only_total": a_total,
        "trajectory_plus_acoustic_total": b_total,
        "combined_nonnegative_folds": b_nonneg,
        "combined_worst_fold_net": b_worst,
        "combined_kept_raw_corrections_fraction": kept_fraction,
        "criteria": criteria,
        "feature_sets": {
            "trajectory_only": list(GATE_FEATURES_A),
            "trajectory_plus_acoustic": list(GATE_FEATURES_B),
        },
        "automatic_promotion": False,
    }

    a.output.mkdir(parents=True)
    write_json(a.output / "report.json", report)

    lines = [
        "# Nested trajectory reliability gate",
        "",
        "| fold | raw net | traj-gate net | combined-gate net | combined corr | combined reg | rel AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        ca = f["arms"]["trajectory_plus_acoustic"]
        auc = ca["val_reliability_auc"]
        lines.append(
            f"| {f['fold']} | {f['raw']['net']:+d} | "
            f"{f['arms']['trajectory_only']['result']['net']:+d} | "
            f"{ca['result']['net']:+d} | {ca['result']['corrections']} | "
            f"{ca['result']['regressions']} | {'n/a' if auc is None else f'{auc:.3f}'} |"
        )
    lines += [
        "",
        f"Raw trajectory total: **{raw_total['corrections']} / {raw_total['regressions']} = {raw_total['net']:+d}**.",
        f"Trajectory-only gate: **{a_total['corrections']} / {a_total['regressions']} = {a_total['net']:+d}**.",
        f"Trajectory + acoustic gate: **{b_total['corrections']} / {b_total['regressions']} = {b_total['net']:+d}**.",
        f"Pre-registered progression criteria met: **{criteria['all_met']}**.",
        "",
        "Fold 3 excluded. No automatic promotion.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-root", type=Path, required=True)
    p.add_argument("--residual-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
