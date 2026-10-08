"""Fold-2 generalization audit; read-only replay of frozen 488-case flow extracts.

Exploratory evidence ONLY: this cohort was previously inspected. Never promote a
corrector from these scores. A future independent holdout is still required.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.summarize_v273_energy_flux_combined import (
    FOLDS, account, choose, clf, evaluate, require,
)

GROUPS = {
    "signed_flux": ("flux__", "absolute__"),
    "birth_death_flow": ("logflow__", "relative__", "weighted_relative__"),
    "flux_birth_124": ("flux__", "absolute__", "logflow__", "relative__", "weighted_relative__"),
    "previous_combined_291": (
        "energy__", "flux__", "geometry__", "interaction__", "logflow__",
        "relative__", "weighted_relative__", "residual__", "relative_residual__",
        "weighted_residual__", "passive__",
    ),
    "all_307": (
        "energy__", "flux__", "geometry__", "interaction__", "logflow__",
        "relative__", "weighted_relative__", "residual__", "relative_residual__",
        "weighted_residual__", "passive__", "absolute__",
    ),
}

def load(root, cases):
    rows = []
    found = set()
    for path in sorted(root.rglob("report.json")):
        meta = json.loads(path.read_text())
        if meta.get("experiment") != "v273_energy_flux_combined_extract":
            continue
        f = int(meta["fold"])
        require(f in FOLDS and f not in found, "duplicate or alien fold")
        found.add(f)
        chunk = [json.loads(line) for line in (path.parent / "rows.jsonl").read_text().splitlines() if line.strip()]
        require(len(chunk) == meta["rows"], "row count drift")
        require(all(r["fold"] == f and r["true_k"] in (2, 3) for r in chunk), "fold/target drift")
        rows.extend(chunk)
    require(found == set(FOLDS) and len(rows) == 488, "cohort missing")
    rows.sort(key=lambda r: (r["fold"], r["row_id"]))
    row_ids = [int(r["row_id"]) for r in rows]
    require(len(set(row_ids)) == 488, "duplicate case")
    y = np.array([r["true_k"] == 2 for r in rows], int)
    folds = np.array([r["fold"] for r in rows], int)
    require(int(y.sum()) == 216, "class count drift")
    names = sorted(rows[0]["features"])
    require(len(names) == 307 and all(sorted(r["features"]) == names for r in rows), "feature schema drift")
    source = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
    ids = {int(r["row_id"]): r for r in source}
    require(len(source) == len(ids) == 488 and set(ids) == set(row_ids), "frozen source mismatch")
    tracks = {f: set() for f in FOLDS}
    players = {f: Counter() for f in FOLDS}
    event_keys = set()
    for r in rows:
        ref = ids[int(r["row_id"])]
        require(int(ref["fold"]) == r["fold"] and int(ref["true_K"]) == r["true_k"], "metadata/label mismatch")
        name = str(ref["recording_id"])
        tracks[r["fold"]].add(name)
        players[r["fold"]][name.split("_")[0]] += 1
        event_keys.add((name, int(ref["start_sample"])))
    overlap = {f"{a}-{b}": len(tracks[a] & tracks[b]) for i, a in enumerate(FOLDS) for b in FOLDS[i+1:]}
    require(not any(overlap.values()), "RECORDING LEAKAGE")
    return rows, names, y, folds, {
        "recordings_per_fold": {str(k): len(v) for k,v in tracks.items()},
        "cross_fold_recording_overlap": overlap,
        "unique_recording_onsets": len(event_keys),
        "players_per_fold": {str(k): dict(v) for k,v in players.items()},
    }

def fold2_features(X, names, y, folds):
    fit = folds != 2
    val = ~fit
    scale = np.std(X[fit], axis=0) + 1e-12
    diff_fit = (np.mean(X[fit & (y == 1)], axis=0) -
                np.mean(X[fit & (y == 0)], axis=0)) / scale
    diff_val = (np.mean(X[val & (y == 1)], axis=0) -
                np.mean(X[val & (y == 0)], axis=0)) / scale
    significant = np.abs(diff_fit) >= .15
    flipped = (diff_fit * diff_val < 0) & significant
    ranked = np.argsort(-np.abs(diff_fit))[:25]
    model = clf().fit(X[fit], y[fit])
    coef = model.named_steps["logisticregression"].coef_[0]
    topcoef = np.argsort(-np.abs(coef))[:20]
    return {
        "fold2_k2_prior": float(np.mean(y[val])),
        "other_fold_k2_prior": float(np.mean(y[fit])),
        "train_separating_features": int(significant.sum()),
        "fold2_direction_reversals": int(flipped.sum()),
        "reversal_fraction": float(flipped.sum()/max(int(significant.sum()),1)),
        "top_train_separation_features": [
            dict(name=names[j], train_delta=float(diff_fit[j]), fold2_delta=float(diff_val[j]),
                 reversed=bool(diff_fit[j] * diff_val[j] < 0)) for j in ranked
        ],
        "top_trained_lr_weights": [
            dict(name=names[j], coefficient=float(coef[j]),
                 train_delta=float(diff_fit[j]), fold2_delta=float(diff_val[j])) for j in topcoef
        ],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "refusing overwrite")
    rows, names, y, folds, metadata = load(a.input_root, a.cases)
    families = {}
    for name, prefixes in GROUPS.items():
        cols = [f for f in names if f.startswith(prefixes)]
        X = np.asarray([[r["features"][f] for f in cols] for r in rows], float)
        require(cols and np.isfinite(X).all(), "invalid features")
        result = evaluate(X, y, folds)
        f2 = next(rot for rot in result["rotations"] if rot["val_fold"] == 2)
        fit = folds != 2
        val = folds == 2
        scores = clf().fit(X[fit], y[fit]).predict_proba(X[val])[:,1]
        oracle = choose(y[val], scores)["net"]  # Hindsight only, NEVER an operating threshold.
        families[name] = dict(dimensions=len(cols), **result,
                              macro_fold_auc=float(np.mean([r["val_auc"] for r in result["rotations"]])),
                              fold2_policy=f2["val_policy"], fold2_auc=f2["val_auc"],
                              fold2_oracle_net_diagnostic=int(oracle))
        if name == "flux_birth_124":
            diagnostic = fold2_features(X, cols, y, folds)
    report = dict(status="completed", cohort="frozen_488_K2_K3",
                  exploratory=True, independent_holdout=False,
                  fold3_used=False, automatic_promotion=False,
                  features_total=len(names), recordings=metadata,
                  fold2_feature_diagnostics=diagnostic, families=families)
    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines = [
        "# Flux + birth/death: corrected families and fold 2",
        "",
        "Exploratory replay on **previously inspected** cases. Not an independent holdout.",
        "",
        "| family | dims | OOF AUC | macro AUC | corrections | regressions | net | fold2 AUC | fold2 net | fold2 oracle (diagnostic) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name,q in families.items():
        s=q["selected_policy_total"]
        lines.append(f"| {name} | {q['dimensions']} | {q['oof_auc']:.4f} | {q['macro_fold_auc']:.4f} | {s['corrections']} | {s['regressions']} | {s['net']:+d} | {q['fold2_auc']:.4f} | {q['fold2_policy']['net']:+d} | {q['fold2_oracle_net_diagnostic']} |")
    d=diagnostic
    lines += ["", "## Fold 2 feature shift",
        f"K2 prevalence: fold2 {d['fold2_k2_prior']:.4f}; other folds {d['other_fold_k2_prior']:.4f}",
        f"Train-separating features: {d['train_separating_features']}; reversed on fold2: {d['fold2_direction_reversals']} ({d['reversal_fraction']:.1%})",
        "", "| feature | train K2-K3 (standardized) | fold2 K2-K3 | reversed |", "|---|---:|---:|---|"]
    for d in diagnostic["top_train_separation_features"][:20]:
        lines.append(f"| {d['name']} | {d['train_delta']:+.4f} | {d['fold2_delta']:+.4f} | {d['reversed']} |")
    lines += ["", "## Source metadata", json.dumps(metadata,indent=2,sort_keys=True), "",
              "No correction promoted; fold 3 excluded. Hindsight oracle is not a valid deployable result."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__ == "__main__":
    main()
