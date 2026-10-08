"""Forensic audit of the frozen coherent selector, not a new trained predictor.

Descriptive comparisons condition on the same initial class and decision.
Input interventions are fixed before execution, label-blind, and deliberately
diagnostic: they may break feature dependence and do not establish causality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.audit_v273_selector_design import aligned_positions, load_feature_rows
from scripts.verify_v273_selector_repair_artifacts import metrics, named_pairs, require

SEEDS = (27403, 27404, 27405)
BINS = (0., .05, .10, .20, .40, 1.000001)
HEADS = ("H0", "spectral", "lifecycle", "harmonic_full", "fundamentals", "sources", "Cxy")
PERMUTATIONS = ("context_all", "spectral", "birth", "damping", "persistence",
                "direct_votes", "audit_rates", "keep_metadata", "identity_control")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    require(bool(rows), "empty CSV")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def describe(v):
    v = np.asarray(v)
    if not len(v):
        return dict(n=0, mean=None, q25=None, median=None, q75=None)
    return dict(n=len(v), mean=float(v.mean()), q25=float(np.quantile(v, .25)),
                median=float(np.median(v)), q75=float(np.quantile(v, .75)))


def derangement(base, ids, seed):
    """Within-source-class donors, deterministic and independent of labels."""
    donor = np.arange(len(base))
    rng = np.random.default_rng(seed)
    for k in sorted(set(base)):
        idx = np.flatnonzero(base == k)
        idx = idx[np.argsort(ids[idx], kind="stable")]
        require(len(idx) >= 2, "too few same-source donors")
        shift = int(rng.integers(1, len(idx)))
        donor[idx] = np.roll(idx, shift)
    require(np.all(donor != np.arange(len(base))) and np.array_equal(base[donor], base), "donor scope")
    return donor


def perturb(held, name, donor, context_names):
    """Keep base, legal mask, and untouched input blocks identical."""
    out = dict(held)
    if name == "count_reference_size":
        a = held["subset_features"].copy()
        # Three outer-train reference folds at evaluation, two per train receiver.
        # This prespecified 2/3 sensitivity check changes only the two count slots.
        for j in (6, 17):
            a[..., j] = np.log1p(np.expm1(a[..., j])*(2/3))
        out["subset_features"] = a
    elif name in ("context_all", "spectral", "birth", "damping", "persistence"):
        cols = list(range(len(context_names))) if name == "context_all" else [
            i for i, n in enumerate(context_names) if n.startswith(name+"__")]
        require(bool(cols), "missing context family")
        out["context"] = held["context"].copy()
        out["context"][:, cols] = held["context"][donor][:, cols]
    elif name == "keep_metadata":
        out["keep_heads"] = held["keep_heads"][donor]
    else:
        cols = {"direct_votes": list(range(6)), "audit_rates": [7, 8, 9, 18, 19, 20],
                "identity_control": list(range(10, 17))}[name]
        a = held["subset_features"].copy()
        a[..., cols] = held["subset_features"][donor][..., cols]
        out["subset_features"] = a
    return out


def decode(q, base):
    best = 2+np.argmax(q[:, 2:], axis=1)
    row = np.arange(len(base))
    return np.where(q[row, best] > q[row, base], best, base)


def flow(y, b, p, q, mask):
    row = np.arange(len(y)); changed = mask & (p != b)
    correction = changed & (p == y) & (b != y)
    regression = changed & (b == y) & (p != y)
    advantage = q[row, p]-q[row, b]
    n = int(changed.sum())
    return dict(rows=n, corrections=int(correction.sum()), regressions=int(regression.sum()),
        neutral=n-int(correction.sum())-int(regression.sum()),
        net=int(correction.sum())-int(regression.sum()),
        probability_expected_net=float(advantage[changed].sum()),
        probability_expected_corrections=float(q[row[changed], p[changed]].sum()),
        probability_expected_regressions=float(q[row[changed], b[changed]].sum()),
        margin_corrections=describe(advantage[correction]), margin_regressions=describe(advantage[regression]))


def feature_contrasts(rows, y, b, p, folds):
    names = sorted(rows[0]["features"])
    x = np.array([[r["features"][k] for k in names] for r in rows])
    table = []
    for source in (2, 3, 4):
        for target in range(2, 7):
            selected = (b == source) & (p == target) & (p != b)
            cor = selected & (y == target); reg = selected & (y == source)
            if min(cor.sum(), reg.sum()) < 20:
                continue
            for j, name in enumerate(names):
                c, r = x[cor, j], x[reg, j]
                sd = np.sqrt((np.var(c)+np.var(r))/2)
                d = float((r.mean()-c.mean())/sd) if sd > 1e-12 else 0.
                take = cor | reg
                auc = float(roc_auc_score(reg[take], x[take, j]))
                fold_d = []
                for f in sorted(set(folds)):
                    fc, fr = cor & (folds == f), reg & (folds == f)
                    if min(fc.sum(), fr.sum()) < 5:
                        continue
                    fstd = np.sqrt((np.var(x[fc, j])+np.var(x[fr, j]))/2)
                    fold_d.append(float((x[fr, j].mean()-x[fc, j].mean())/fstd) if fstd > 1e-12 else 0.)
                table.append(dict(source=source, target=target, feature=name,
                    corrections=int(cor.sum()), regressions=int(reg.sum()),
                    correction_median=float(np.median(c)), regression_median=float(np.median(r)),
                    standardized_reg_minus_cor=d, descriptive_auc_high_value_regression=auc,
                    eligible_folds=len(fold_d), sign_agreement_folds=sum(np.sign(v) == np.sign(d) for v in fold_d),
                    fold_effects=json.dumps(fold_d)))
    return table


def audit_replay(args, rows, native, pos, report):
    import tensorflow as tf
    from scripts.learn_v273_coherent_selector import CoherentCombinationArbiter
    from scripts.v273_selector_contract import ProducerPool, matrices, assemble_outer
    y, b, folds = (native[k][pos] for k in ("true_K", "frozen_baseline_K", "fold"))
    ids = native["eligible_global_index"]
    reference_q = native["class_probability"]
    x, raw, names = matrices(rows)
    pool = ProducerPool(x, y, b, folds, ids)
    specs = [(name, seed) for name in PERMUTATIONS for seed in SEEDS]
    specs += [("count_reference_size", 0)]
    probabilities = {f"{name}:{seed}": np.full_like(reference_q, np.nan) for name, seed in specs}
    experts = np.full((len(y), 6, 5), np.nan, np.float32)
    selected = np.full((len(y), 5, 21), np.nan, np.float32)
    shift_rows = []
    checks = []

    def infer(model, inputs, details=False):
        chunks = []
        for start in range(0, len(inputs["baseline"]), 192):
            out = model({k: v[start:start+192] for k, v in inputs.items()}, training=False)
            keys = ("class_probability", "subset_attention", "selected_subset_features") if details else ("class_probability",)
            chunks.append({k: out[k].numpy() for k in keys})
        return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    for outer in (0, 1, 2, 4):
        print(json.dumps(dict(stage="reconstruct", outer=outer)), flush=True)
        train, held, tr, val, provenance = assemble_outer(pool, raw, outer)
        model = CoherentCombinationArbiter(width=32)
        model({k: v[:1] for k, v in held.items()})
        model.load_weights(str(args.coherent/f"fold-{outer}.weights.h5"))
        original = infer(model, held, True)
        dq = float(np.max(np.abs(original["class_probability"]-reference_q[val])))
        da = float(np.max(np.abs(original["subset_attention"]-native["selected_subset_probabilities"][val])))
        require(dq < 2e-6 and da < 2e-6, "reconstructed probabilities/attention differ")
        require(np.array_equal(decode(original["class_probability"], b[val]), native["predicted_K"][pos[val]]), "frozen decoder replay differs")
        require(provenance == next(a for a in report["provenance"] if a["outer_fold"] == outer), "reconstructed provenance differs")
        checks.append(dict(fold=outer, rows=len(val), max_probability_error=dq, max_attention_error=da, prediction_mismatches=0))
        experts[val] = pool.predict(set(folds[tr]), val, {outer})
        selected[val] = original["selected_subset_features"]
        # Reference count channels have different producer sample sizes at train/eval.
        for source in (2, 3, 4):
            for target in range(2, 7):
                if target == source:
                    continue
                for name, column in (("global_count", 6), ("context_count", 17)):
                    train_values = train["subset_features"][b[tr] == source, target-2, :, column]
                    valid = train["subset_mask"][b[tr] == source, target-2]
                    train_max = float(train_values[valid].max())
                    sel = (b[val] == source) & (native["predicted_K"][pos[val]] == target)
                    values = selected[val[sel], target-2, column]
                    shift_rows.append(dict(fold=outer, source=source, target=target, channel=name,
                        selected_rows=int(sel.sum()), train_max=train_max,
                        held_selected_mean=float(values.mean()) if len(values) else None,
                        held_selected_above_train_max=int(np.sum(values > train_max+1e-6))))
        for name, seed in specs:
            donor = derangement(b[val], ids[val], seed) if seed else None
            altered = perturb(held, name, donor, names)
            q = infer(model, altered)["class_probability"]
            probabilities[f"{name}:{seed}"][val] = q
            if name == "identity_control":
                require(np.array_equal(altered["subset_features"], held["subset_features"]), "identity permutation altered descriptors")
                require(np.array_equal(q, original["class_probability"]), "negative control differs")
        print(json.dumps(dict(stage="interventions_completed", outer=outer, variants=len(specs))), flush=True)
        tf.keras.backend.clear_session()
    require(all(np.isfinite(a).all() for a in (experts, selected, *probabilities.values())), "incomplete replay")
    return probabilities, experts, selected, checks, shift_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coherent", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite diagnostic")
    root = Path(__file__).resolve().parents[1]
    verification = json.loads((root/"analysis/evidence/v273-selector-repair/verification.json").read_text())
    for arm in ("coherent", "legacy"):
        require(sha(getattr(args, arm)/"predictions.npz") == verification["arms"][arm]["predictions_sha256"], "input NPZ changed")
    with np.load(args.coherent/"predictions.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    with np.load(args.legacy/"predictions.npz", allow_pickle=False) as z:
        legacy = z["predicted_K"]
    ids, yn, bn, pn, fn = (data[k] for k in ("global_index", "true_K", "frozen_baseline_K", "predicted_K", "fold"))
    pos = aligned_positions(ids, data["eligible_global_index"])
    rows, names = load_feature_rows(args.features, ids, yn, bn, fn, pos)
    y, b, p, folds, old = yn[pos], bn[pos], pn[pos], fn[pos], legacy[pos]
    q = data["class_probability"]; index = np.arange(len(y))
    require(np.array_equal(decode(q, b), p), "archived decoder mismatch")
    report = json.loads((args.coherent/"report.json").read_text())
    require(metrics(yn, pn) == report["candidate"] and named_pairs(yn, bn, pn) == report["paired"], "archived score mismatch")
    transition = []
    for source in (2, 3, 4):
        for target in range(2, 7):
            if source == target:
                continue
            take = (b == source) & (p == target)
            transition.append(dict(source=source, target=target, **flow(y, b, p, q, take),
                by_fold={str(f): flow(y, b, p, q, take & (folds == f)) for f in (0, 1, 2, 4)}))
    margin = q[index, p]-q[index, b]
    bins = []
    for source in (None, 2, 3, 4):
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            take = (p != b) & (margin >= lo) & (margin < hi)
            if source is not None:
                take &= b == source
            bins.append(dict(source=source, lower=lo, upper=hi, **flow(y, b, p, q, take)))
    regression = (y == b) & (p != b) & np.isin(y, (3, 4))
    require(int(np.sum(regression & (y == 3))) == 405 and int(np.sum(regression & (y == 4))) == 189, "K3/K4 regression count")
    correction = (y != b) & (p == y)
    events = []
    attention = data["selected_subset_probabilities"]
    masks = np.array([[1]+[int(bool(i & (1 << j))) for j in range(6)] for i in range(64)])
    for i, row in enumerate(rows):
        w = attention[i, p[i]-2]
        n = 64 if (b[i], p[i]) in {(2, 3), (3, 2), (3, 4), (4, 3)} else 32
        entropy = float(-np.sum(w[w > 0]*np.log(w[w > 0]))) if p[i] != b[i] else 0.
        event = dict(global_index=int(ids[pos[i]]), fold=int(folds[i]), recording_id=row["recording_id"],
            start_sample=row["start_sample"], start_seconds=row["start_sample"]/44100.,
            true_k=int(y[i]), baseline_k=int(b[i]), predicted_k=int(p[i]), legacy_k=int(old[i]),
            outcome="correction" if correction[i] else "regression" if y[i] == b[i] and p[i] != b[i] else "neutral_change" if p[i] != b[i] else "keep",
            expected_advantage=float(margin[i]), predicted_probability=float(q[i, p[i]]),
            baseline_probability=float(q[i, b[i]]), low_class_probability=float(q[i, :2].sum()),
            normalized_attention_entropy=entropy/np.log(n),
            effective_subset_fraction=float(np.exp(entropy)/n) if p[i] != b[i] else 0.,
            most_weighted_subset=int(w.argmax()) if p[i] != b[i] else -1,
            most_weighted_subset_weight=float(w.max()))
        event.update({f"probability_k{k}": float(q[i, k]) for k in range(7)})
        event.update({f"attention_inclusion_{name}": float(w @ masks[:, j]) for j, name in enumerate(HEADS)})
        events.append(event)
    feature_table = feature_contrasts(rows, y, b, p, folds)
    out = dict(status="completed", input_run_id=37766719862,
        input_sha256={arm: sha(getattr(args, arm)/"predictions.npz") for arm in ("coherent", "legacy")},
        original_metrics=report["candidate"], original_paired=report["paired"],
        native_rows=len(yn), eligible_rows=len(y), independent_validation=False, promotion=False,
        transitions=transition, margin_bins=bins, all_changes=flow(y, b, p, q, np.ones(len(y), bool)),
        regression_overlap={str(k): dict(total=int(np.sum(regression & (y == k))),
            also_legacy_regression=int(np.sum(regression & (y == k) & (old != y))),
            newly_broken=int(np.sum(regression & (y == k) & (old == y)))) for k in (3, 4)},
        intervention_spec=dict(permutations=list(PERMUTATIONS), seeds=list(SEEDS),
            grouping="held-out fold and initial class; cyclic non-self donors sorted by native ID",
            count_scaling="only log1p-count channels 6/17: log1p(expm1(x)*2/3), from two versus three reference folds",
            labels_used_to_construct_interventions=False, trained_weights_changed=False),
        limitations=["Post-hoc diagnosis on exposed development folds, not independent validation.",
            "Feature contrasts condition on observed decisions; neither AUC nor effect sizes are held-out predictor scores.",
            "Permutation interventions can break joint feature dependence; they measure frozen-model sensitivity, not acoustic causation.",
            "Count scaling is one fixed sensitivity check, not a trained or promoted repair.",
            "Attention weights are descriptive and are not causal head importance.",
            "No novel audio, recording-independent confidence interval, or new threshold search."])
    args.output.mkdir(parents=True)
    if args.replay:
        altered, experts, selected, checks, count_shifts = audit_replay(args, rows, data, pos, report)
        out["replay_checks"] = checks
        out["interventions"] = {}
        out["count_shifts"] = count_shifts
        for name, alt_q in altered.items():
            alt_p = decode(alt_q, b)
            full = bn.copy(); full[pos] = alt_p
            out["interventions"][name] = dict(paired_vs_freeze=named_pairs(yn, bn, full),
                paired_vs_coherent=named_pairs(yn, pn, full),
                original_corrections_lost=int(np.sum(correction & (alt_p != y))),
                original_k34_regressions_restored=int(np.sum(regression & (alt_p == y))),
                by_fold={str(f): named_pairs(y[folds == f], p[folds == f], alt_p[folds == f])["global"] for f in (0, 1, 2, 4)})
        for i, event in enumerate(events):
            d = experts[i, 1:, p[i]-2]-experts[i, 1:, b[i]-2]
            event.update(expert_mean_advantage=float(d.mean()), experts_supporting_decision=int(np.sum(d > 0)))
            for j, name in enumerate(HEADS[1:6], 1):
                event[f"expert_{name}_baseline_probability"] = float(experts[i, j, b[i]-2])
                event[f"expert_{name}_decision_probability"] = float(experts[i, j, p[i]-2])
            for j in range(21):
                event[f"selected_descriptor_{j}"] = float(selected[i, p[i]-2, j])
        np.savez_compressed(args.output/"diagnostic-replays.npz", eligible_global_index=data["eligible_global_index"],
            expert_probability=experts, selected_descriptors=selected, **{k.replace(":", "__"): v for k, v in altered.items()})
    write_csv(args.output/"event-audit.csv", events)
    case_rows = [dict(**events[i], **rows[i]["features"]) for i in np.flatnonzero(regression)]
    write_csv(args.output/"k3k4-regressions.csv", case_rows)
    write_csv(args.output/"feature-contrasts.csv", feature_table)
    out["event_count"] = len(events)
    out["regression_case_count"] = len(case_rows)
    out["feature_contrast_count"] = len(feature_table)
    (args.output/"report.json").write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(status="completed", regression_cases=len(case_rows), expected_net=out["all_changes"]["probability_expected_net"], actual_net=out["all_changes"]["net"], replay=args.replay)), flush=True)


if __name__ == "__main__":
    main()
