"""Preregistered three-arm experiment: legacy, repaired inputs, coherent decision.

Same native 59,309 events and frozen baseline. The four exposed folds are
experimental evaluation only; no unseen-player generalization is claimed.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from scripts.audit_v273_selector_design import aligned_positions, load_feature_rows
from scripts.evaluate_v273_audit_aware_selection import inference_inputs
from scripts.evaluate_v273_dynamic_heads import oof_experts, outer_experts
from scripts.learn_v273_audit_aware_selection import append_audit_aware_descriptors
from scripts.learn_v273_transition_combo_risk import (
    compute_direct_options, attach_oof_subset_audits, train_model,
)
from scripts.learn_v273_coherent_selector import train_coherent
from scripts.v273_selector_contract import ProducerPool, assemble_outer, matrices, require, FOLDS
from scripts.yourmt3_exactk_common import metrics, paired, digest


def legacy_outer(x, raw, y, base, folds, outer):
    tr = np.flatnonzero(folds != outer); val = np.flatnonzero(folds == outer)
    fit = oof_experts({k: v[tr] for k, v in x.items()}, y[tr], base[tr], folds[tr])
    test = outer_experts({k: dict(train=v[tr], test=v[val]) for k, v in x.items()}, y[tr], base[val])
    a, am = compute_direct_options(fit, base[tr])
    b, bm = compute_direct_options(test, base[val])
    af, bf, _ = attach_oof_subset_audits(a, am, base[tr], y[tr], folds[tr], b, bm, base[val])
    scale = StandardScaler().fit(raw[tr])
    ca = np.clip(scale.transform(raw[tr]), -6, 6)
    cb = np.clip(scale.transform(raw[val]), -6, 6)
    aa, bb, _ = append_audit_aware_descriptors(af, bf, am, bm, base[tr], base[val],
                                             y[tr], folds[tr], ca, cb, "both")
    return (inference_inputs(fit, base[tr], ca, aa, am),
            inference_inputs(test, base[val], cb, bb, bm), tr, val)


def render(report):
    ref, out, pair = report["freeze_reference"], report["candidate"], report["paired"]
    rows = ["# V27.3 — selector repair: "+report["arm"], "",
        "Expérience sur folds 0/1/2/4 déjà exposés ; aucune validation indépendante ni promotion.", "",
        "| Système | Exact-K global | Exact-K poly | Corrections | Régressions | Net |",
        "|---|---:|---:|---:|---:|---:|",
        f"| freeze_local_combo | {100*ref['exact']:.4f}% | {100*ref['poly']['exact']:.4f}% | — | — | — |",
        f"| {report['arm']} | {100*out['exact']:.4f}% | {100*out['poly']['exact']:.4f}% | "
        f"{pair['global']['corrections']} | {pair['global']['regressions']} | {pair['global']['net']:+d} |",
        "", "| Vrai K | Référence | Variante | Corrections | Régressions | Net |",
        "|---|---:|---:|---:|---:|---:|"]
    for k in range(7):
        a, b, p = ref["by_k"][str(k)], out["by_k"][str(k)], pair["by_k"][str(k)]
        rows.append(f"| K{k} | {100*a['exact']:.4f}% | {100*b['exact']:.4f}% | "
                    f"{p['corrections']} | {p['regressions']} | {p['net']:+d} |")
    rows += ["", "| Fold | Corrections | Régressions | Net |", "|---|---:|---:|---:|"]
    for f in FOLDS:
        p = report["folds"][str(f)]["paired"]["global"]
        rows.append(f"| {f} | {p['corrections']} | {p['regressions']} | {p['net']:+d} |")
    rows += ["", "## Configuration", "", "```json",
             json.dumps(report["configuration"], indent=2, sort_keys=True), "```", "",
             "## Limites", ""] + ["- "+s for s in report["limitations"]]
    return "\n".join(rows)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--arm", choices=("legacy", "contracts", "coherent"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite experiment")
    with np.load(args.reference, allow_pickle=False) as z:
        y, base, ids, folds, eligible_ids, archived = (z[k] for k in
            ("true_K", "frozen_baseline_K", "global_index", "fold", "eligible_global_index", "predicted_K"))
    positions = aligned_positions(ids, eligible_ids)
    require(len(y) == 59309 and len(positions) == 7493, "cohort size drift")
    require(int(np.sum(y == base)) == 48454 and int(np.sum(y >= 2)) == 7385 and
            int(np.sum((y == base) & (y >= 2))) == 2530, "baseline drift")
    require(set(folds) == set(FOLDS), "fold drift")
    rows, names = load_feature_rows(args.features, ids, y, base, folds, positions)
    yc, bc, fc = y[positions], base[positions], folds[positions]
    recordings = {}
    for row in rows:
        name = row["recording_id"]
        require(name[:2] in ("00", "01", "02", "03", "04"), "player05 encountered")
        if name in recordings:
            require(recordings[name] == row["fold"], "recording split across folds")
        recordings[name] = row["fold"]
    x, raw, context_names = matrices(rows)
    require(len(context_names) == 43, "declared feature families changed")
    legacy_names = [n for n in names if n.startswith(("spectral__", "birth__", "persistence__", "damping__"))][:26]
    legacy_raw = np.array([[r["features"][n] for n in legacy_names] for r in rows], float)
    pool = ProducerPool(x, yc, bc, fc, eligible_ids)
    prediction = base.copy()
    cp = np.full((len(positions), 5), np.nan, np.float32)
    rp = np.full_like(cp, np.nan)
    kp = np.full(len(positions), np.nan, np.float32)
    posterior = np.full((len(positions), 7), np.nan, np.float32)
    weights = np.full((len(positions), 5, 64), np.nan, np.float32)
    fold_reports = {}
    provenance = []
    params = []
    args.output.mkdir(parents=True)
    for outer in FOLDS:
        print(json.dumps(dict(arm=args.arm, outer_fold=outer, stage="build_inputs")), flush=True)
        if args.arm == "legacy":
            train, held, tr, val = legacy_outer(x, legacy_raw, yc, bc, fc, outer)
        else:
            train, held, tr, val, manifest = assemble_outer(pool, raw, outer)
            provenance.append(manifest)
        print(json.dumps(dict(arm=args.arm, outer_fold=outer, stage="train", epochs=30)), flush=True)
        if args.arm == "coherent":
            proposed, out, history, model = train_coherent(train, yc[tr], bc[tr], held)
            posterior[val] = out["class_probability"].numpy()
        else:
            action, out, history, model = train_model(train, yc[tr], bc[tr], held, return_model=True)
            proposed = np.where(action == 7, bc[val], action)
        require(np.isin(proposed, (2, 3, 4, 5, 6)).all(), "unsupported output destination")
        prediction[positions[val]] = proposed
        cp[val] = out["candidate_correct"].numpy()
        rp[val] = out["candidate_regress"].numpy()
        kp[val] = out["keep_correct"].numpy()
        weights[val] = out["subset_attention"].numpy()
        params.append(int(model.count_params()))
        model.save_weights(str(args.output/f"fold-{outer}.weights.h5"))
        event = positions[val]
        paired_fold = paired(y[event], base[event], prediction[event])
        fold_reports[str(outer)] = dict(eligible_rows=len(val), paired=paired_fold,
            training=history, parameters=int(model.count_params()))
        (args.output/f"fold-{outer}-result.json").write_text(json.dumps(
            fold_reports[str(outer)], indent=2, sort_keys=True, allow_nan=False)+"\n")
        print(json.dumps(dict(arm=args.arm, outer_fold=outer, stage="evaluated",
                              paired=paired_fold["global"], history=history)), flush=True)
        del train, held, out, model
        gc.collect()
    require(all(np.isfinite(a).all() for a in (cp, rp, kp, weights)), "missing fold predictions")
    if args.arm == "coherent":
        require(np.isfinite(posterior).all(), "missing class probabilities")
        require(np.max(np.ptp(rp, axis=1)) == 0, "inconsistent baseline correctness")
    eligible_mask = np.zeros(len(y), bool); eligible_mask[positions] = True
    require(np.array_equal(prediction[~eligible_mask], base[~eligible_mask]), "eligibility drift")
    replica = int(np.sum(prediction != archived))
    if args.arm == "legacy":
        require(replica == 0, "legacy does not exactly reproduce archived both selector")
    baseline = metrics(y, base); candidate = metrics(y, prediction)
    comparison = paired(y, base, prediction)
    require(candidate["correct"] == baseline["correct"]+comparison["global"]["net"], "metric mismatch")
    report = dict(status="completed", arm=args.arm, promotion=False,
        independent_validation=False, reference_sha256=digest(args.reference),
        freeze_reference=baseline, candidate=candidate, paired=comparison, folds=fold_reports,
        changes_from_archived_selector=replica,
        configuration=dict(epochs=30, seed=27402, learning_rate=.002, batch_size=192,
            width=32, parameter_counts=params, eligible_rows=7493, native_rows=59309,
            context_names=legacy_names if args.arm == "legacy" else context_names,
            context_dimensions=26 if args.arm == "legacy" else 43,
            subset_descriptor_dimensions=21, identical_epoch_budget=True,
            audit_producers_exclude_receiver=args.arm != "legacy",
            one_probability_of_baseline_correctness=args.arm == "coherent",
            objective="unweighted seven-class log loss" if args.arm == "coherent" else "legacy weighted action CE and three BCE losses",
            H0="fixed smoothed 0.84 posterior" if args.arm == "legacy" else "deterministic baseline-action encoding; correctness learned",
            output_destinations=[2, 3, 4, 5, 6], folds=list(FOLDS), fold3_used=False, player05_used=False),
        provenance=provenance,
        expert_producers=list(pool.manifests.values()) if args.arm != "legacy" else [],
        limitations=[
            "Development-exposed folds, not independent validation; no reference promotion.",
            "Only baseline-predicted K2/K3/K4 events have acoustic features; 1,918 poly errors remain outside the correction scope.",
            "K0/K1 are modeled as true classes by the coherent loss but remain forbidden output destinations for this experiment.",
            "H0 encodes the known fallback action; the original frozen network posterior is not available in these artifacts.",
            "The contracts arm bundles context, composition, fallback metadata and nested-producer repairs; it does not isolate each one.",
            "The coherent arm changes both the probability parameterization and loss; parameter counts are reported, not claimed equal.",
            "All sound observables remain fixed harmonic-template summaries with up to +160ms lookahead.",
            "No threshold or architecture choice is retuned using the outcomes of this run.",
        ])
    (args.output/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    (args.output/"report.md").write_text(render(report))
    data = dict(global_index=ids, true_K=y, frozen_baseline_K=base, predicted_K=prediction,
        fold=folds, eligible_global_index=eligible_ids,
        candidate_correction_regression_risk=np.stack([cp, rp], axis=-1),
        keep_probability=kp, selected_subset_probabilities=weights)
    if args.arm == "coherent":
        data["class_probability"] = posterior
    np.savez_compressed(args.output/"predictions.npz", **data)
    print(render(report), flush=True)


if __name__ == "__main__":
    main()
