"""Experimental seven-class rescue for frozen baseline K0/K1/K5.

Requires NEW acoustic rows for every baseline K0/K1/K5 event, not merely
the 7,493 historical K2/K3/K4 examples. No changes to baseline K2/K3/K4.
Outer-fold training and inner-fold abstention tuning never see test labels.
Development folds are historically exposed; no independent claim or promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.summarize_v273_harmonic_global import load_cohort, get_matrix
from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require

SOURCES = (0, 1, 5)
CLASSES = np.arange(7)
EXPERIMENT = "v273_harmonic_global_candidate_extract"


def read_scope(root, y, base, ids, fold, member, start):
    """Demand exactly the full label-blind selected baseline cohort."""
    lookup = {int(v): i for i, v in enumerate(ids)}
    wanted = set(np.flatnonzero(np.isin(base, SOURCES)).tolist())
    seen = set()
    rows = []
    reports = set()
    for file in sorted(root.rglob("report.json")):
        report = json.loads(file.read_text())
        if report.get("experiment") != EXPERIMENT:
            continue
        require(tuple(report.get("candidate_baselines", ())) == SOURCES,
                "wrong baseline scope, refusing historic K2/K3/K4 features")
        require(report.get("primary_only") is True, "scope rescue needs primary-only rows")
        f = int(report["fold"])
        require(f in FOLDS and f not in reports, "fold repeated or forbidden")
        reports.add(f)
        actual = 0
        with (file.parent / "rows.jsonl").open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                ident = int(row["global_index"])
                require(ident in lookup, "unknown global ID")
                ix = lookup[ident]
                require(ix in wanted and ix not in seen, "wrong/duplicate scope ID")
                require(int(fold[ix]) == f == int(row["fold"]), "cross-fold mismatch")
                require(row["recording_id"] == member[ix] and
                        int(row["start_sample"]) == int(start[ix]), "recording/time mismatch")
                require(int(row["baseline_k"]) == int(base[ix]) and
                        int(row["true_k"]) == int(y[ix]), "cohort integrity mismatch")
                require("features" in row and "detuned" not in row, "feature-mode mismatch")
                seen.add(ix)
                rows.append((ix, row))
                actual += 1
        require(actual == int(report["eligible_rows"]), "row count mismatch")
    require(reports == set(FOLDS), "all four fold artifacts required")
    require(seen == wanted, "missing or extra scope examples")
    rows.sort(key=lambda item: item[0])
    positions = np.array([i for i, _ in rows], int)
    samples = [r for _, r in rows]
    keys = sorted(samples[0]["features"])
    require(all(sorted(r["features"]) == keys for r in samples), "feature schema mismatch")
    return positions, samples, keys


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2500,
                                            random_state=27402, solver="lbfgs"))


def probabilities(train_x, train_y, test_x):
    classifier = model().fit(train_x, train_y)
    scores = classifier.predict_proba(test_x)
    full = np.zeros((len(test_x), 7), float)
    full[:, classifier[-1].classes_.astype(int)] = scores
    require(np.isfinite(full).all() and np.all(full >= 0) and
            np.allclose(full.sum(axis=1), 1), "probabilities invalid")
    return full


def proposals(scores, base):
    """Class, relative margin; the unchanged frozen action is always allowed."""
    p = np.argmax(scores, axis=1)
    delta = scores[np.arange(len(p)), p] - scores[np.arange(len(p)), base]
    return p, delta


def choose_threshold(truth, base, proposal, margin):
    """Fast exact O(n log n) sweep; never used on the outer test fold."""
    eligible = (proposal != base) & np.isfinite(margin)
    if not eligible.any():
        return dict(threshold=float("inf"), net=0, actions=0)
    order = np.argsort(-margin[eligible], kind="stable")
    weights = (proposal[eligible] == truth[eligible]).astype(int) - (
        base[eligible] == truth[eligible]).astype(int)
    scores = margin[eligible][order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    gain = cumulative[ends]
    actions = ends + 1
    # Abstention wins ties at zero; nonzero net ties prefer fewer actions.
    valid = np.flatnonzero(gain > 0)
    if len(valid) == 0:
        return dict(threshold=float("inf"), net=0, actions=0)
    max_gain = int(gain[valid].max())
    pick = int(valid[np.flatnonzero(gain[valid] == max_gain)[0]])
    return dict(threshold=float(scores[ends[pick]]), net=max_gain,
                actions=int(actions[pick]))


def tune_outer(train_x, train_y, train_base, train_fold):
    """Crossfit inside the outer training folds; separate gates per source."""
    oof = np.full((len(train_y), 7), np.nan)
    for inner in sorted(set(train_fold)):
        held = train_fold == inner
        fit = ~held
        require(fit.any() and held.any(), "inner split")
        oof[held] = probabilities(train_x[fit], train_y[fit], train_x[held])
    require(np.isfinite(oof).all(), "incomplete inner OOF")
    p, margin = proposals(oof, train_base)
    return {k: choose_threshold(train_y[train_base == k],
                                train_base[train_base == k],
                                p[train_base == k], margin[train_base == k])
            for k in SOURCES}


def run(cohort, input_root, output):
    require(not output.exists(), "refusing overwrite")
    y, b, ids, fold, member, start = load_cohort(cohort)
    pos, rows, names = read_scope(input_root, y, b, ids, fold, member, start)
    require(len(pos) == 51816, "expected historic K0/K1/K5 cohort missing")
    x = np.column_stack([get_matrix(rows, names, "features", tuple()),
                         np.eye(7)[b[pos]]])
    yp, bp, fp = y[pos], b[pos], fold[pos]
    predicted = b.copy()
    fold_reports = {}
    for outer in FOLDS:
        fit = fp != outer
        held = fp == outer
        require(fit.any() and held.any(), "outer split empty")
        thresholds = tune_outer(x[fit], yp[fit], bp[fit], fp[fit])
        probability = probabilities(x[fit], yp[fit], x[held])
        offer, delta = proposals(probability, bp[held])
        accept = np.zeros(int(held.sum()), bool)
        for k in SOURCES:
            select = bp[held] == k
            accept[select] = (offer[select] != k) & (
                delta[select] >= thresholds[k]["threshold"])
        selected_ids = pos[held]
        predicted[selected_ids] = np.where(accept, offer, bp[held])
        summary = paired(y[selected_ids], b[selected_ids],
                         predicted[selected_ids])["global"]
        fold_reports[str(outer)] = dict(
            n=int(held.sum()), actions=int(accept.sum()), paired=summary,
            thresholds={str(k): thresholds[k] for k in SOURCES})
    unchanged = np.ones(len(b), bool)
    unchanged[pos] = False
    require(np.array_equal(predicted[unchanged], b[unchanged]),
            "legacy K2/K3/K4 or other rows changed")
    require(len(pos) + unchanged.sum() == len(b), "coverage mismatch")
    report = dict(status="completed", automatic_promotion=False,
                  validation="outer recording-fold OOF with train-only nested thresholds; historically exposed folds",
                  source_baselines=list(SOURCES), rows=int(len(y)),
                  eligible=int(len(pos)), actions=int(np.sum(predicted != b)),
                  baseline=metrics(y, b), candidate=metrics(y, predicted),
                  paired=paired(y, b, predicted), folds=fold_reports,
                  limitations=["No independent never-used cohort",
                               "Corrector relies on 160ms post-onset features; not online causal",
                               "Does not modify the previous K2/K3/K4 network"])
    output.mkdir(parents=True)
    np.savez_compressed(output/"predictions.npz", global_index=ids,
                        true_K=y, frozen_baseline_K=b, fold=fold,
                        eligible_global_index=ids[pos], predicted_K=predicted)
    (output/"report.json").write_text(json.dumps(report, sort_keys=True,
                                                 indent=2, allow_nan=False) + "\n")
    g = report["paired"]["global"]
    (output/"report.md").write_text(
        "# Baseline K0/K1/K5 rescue (experimental)\n\n"
        + "No independent validation or promotion.\n\n"
        + "Corrections: " + str(g["corrections"]) + "; regressions: "
        + str(g["regressions"]) + "; net: " + str(g["net"]) + "\n"
        + "\nExact-K global: " + str(100*report["candidate"]["exact"])
        + "%; poly: " + str(100*report["candidate"]["poly"]["exact"]) + "%\n")
    print(json.dumps({"paired": g, "folds": fold_reports}, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.cohort, args.input_root, args.output)


if __name__ == "__main__":
    main()
