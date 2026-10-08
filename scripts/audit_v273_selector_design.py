"""Reproduce selector decisions and audit their contract without retraining.

Inputs are immutable artifacts from run 37758789956 (baseline, both) and
37740493178 (four feature folds). This is a forensic replay on exposed data,
not an independent validation or a replacement predictor.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

KS = np.arange(2, 7)
EPS = 1e-6
REVIEW_REF = "aa4ca4b7729a00e19dcfd1a071d6afac09933dc3"
RUN_REF = "2507836c4ee590fb944fb8d2df0ac401ae0c7704"
AUDITED_FILES = (
    "scripts/audit_v273_kconditional_127_subsets.py",
    "scripts/evaluate_v273_audit_aware_selection.py",
    "scripts/evaluate_v273_dynamic_heads.py",
    "scripts/evaluate_v273_neural_history_mix.py",
    "scripts/extract_v273_harmonic_global.py",
    "scripts/extract_v273_harmonic_trajectory.py",
    "scripts/learn_v273_audit_aware_selection.py",
    "scripts/learn_v273_transition_combo_risk.py",
    "scripts/summarize_v273_harmonic_global.py",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def aligned_positions(ids, selected):
    require(len(np.unique(ids)) == len(ids), "duplicate native ID")
    require(len(np.unique(selected)) == len(selected), "duplicate eligible ID")
    lookup = {int(v): i for i, v in enumerate(ids)}
    require(all(int(v) in lookup for v in selected), "unknown eligible ID")
    return np.array([lookup[int(v)] for v in selected], dtype=int)


def legacy_decode(correct, regress, keep_preferred, baseline):
    """Exact archived score formula; reproduction is checked on every row."""
    scores = np.column_stack([
        np.log(np.maximum(correct, EPS)) - np.log(np.maximum(regress, EPS)),
        np.log(np.maximum(keep_preferred, EPS))
        - np.log(np.maximum(1. - keep_preferred, EPS)),
    ])
    scores[np.arange(len(baseline)), baseline - 2] = -1e9
    action = scores.argmax(axis=1)
    return np.where(action == 5, baseline, np.r_[KS, 7][action]), scores


def expected_net_replay(correct, regress, baseline):
    """Single prespecified diagnostic: expected Exact-K gain versus KEEP=0.

    No label input, fitted threshold, or search across policies. These heads
    were trained jointly with the OLD decoder, so this is not a trained fix.
    Exact ties preserve the baseline.
    """
    gain = correct - regress
    gain = gain.copy()
    gain[np.arange(len(baseline)), baseline - 2] = -np.inf
    candidate = gain.argmax(axis=1)
    return np.where(gain[np.arange(len(baseline)), candidate] > 0,
                    KS[candidate], baseline)


def metrics(y, baseline, prediction):
    def group(mask):
        fixes = int(np.sum(mask & (baseline != y) & (prediction == y)))
        regressions = int(np.sum(mask & (baseline == y) & (prediction != y)))
        return dict(rows=int(mask.sum()), correct=int(np.sum(mask & (prediction == y))),
                    exact=float(np.mean(prediction[mask] == y[mask])) if mask.any() else None,
                    changes=int(np.sum(mask & (prediction != baseline))),
                    corrections=fixes, regressions=regressions, net=fixes-regressions)
    return dict(global_metrics=group(np.ones(len(y), bool)), poly=group(y >= 2),
                by_k={str(k): group(y == k) for k in range(7)})


def source_namespace(repo, path, constants=(), functions=(), initial=None):
    """Execute only reviewed pure production functions, without ML imports."""
    tree = ast.parse((repo/path).read_text())
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in functions) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
              t.id in constants for t in n.targets))]
    ns = dict(np=np, require=require)
    ns.update(initial or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(repo/path), "exec"), ns)
    return ns


def source_contracts(repo, feature_names):
    # Run the ACTUAL context-selection expression on the real artifact schema.
    evaluator = repo/"scripts/evaluate_v273_audit_aware_selection.py"
    tree = ast.parse(evaluator.read_text())
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "ctx_names" for t in n.targets)]
    require(len(assignments) == 1, "context assignment changed; review source")
    chosen = eval(compile(ast.Expression(assignments[0].value), str(evaluator), "eval"),
                  {"names": feature_names})
    dynamic = source_namespace(repo, "scripts/evaluate_v273_dynamic_heads.py",
                               ("BASE_PROB",), ("fixed_head",))
    heads = source_namespace(repo, "scripts/evaluate_v273_neural_history_mix.py",
             ("CORRECTIVE", "FIXES", "ACTION_HEADS", "EXPERT_NAMES", "HEADS", "H"),
             ("build_candidate_logits",), {"KEEP": 7})
    vote = source_namespace(repo, "scripts/audit_v273_kconditional_127_subsets.py",
                            ("CORRECTORS",), ("votes",))
    options = source_namespace(repo, "scripts/learn_v273_transition_combo_risk.py",
                ("K_RANGE", "SRC_RANGE", "CORRECTION", "COMBOS", "COMBO_BINARY"),
                ("compute_direct_options", "targets"), {"votes": vote["votes"]})
    b = np.array([2, 3, 4])
    rng = np.random.default_rng(27402)
    p = rng.uniform(.1, 1., (3, 6, 5)).astype(np.float32)
    p /= p.sum(axis=-1, keepdims=True)
    p[:, 0] = dynamic["fixed_head"](b)
    opts, valid = options["compute_direct_options"](p, b)
    mismatches = 0
    checked = 0
    for i, source in enumerate(b):
        for j, target in enumerate(KS):
            expected = options["COMBO_BINARY"][:, 6] * float((source, target) in options["CORRECTION"])
            checked += int(valid[i, j].sum())
            mismatches += int(np.sum(valid[i, j] & (opts[i, j, :, 5] != expected)))
    head_logits, active, _, _ = heads["build_candidate_logits"](p, b)
    return dict(
        schema_features=len(feature_names),
        schema_families=dict(Counter(n.split("__")[0] for n in feature_names)),
        direct_context_names=chosen,
        direct_context_families=dict(Counter(n.split("__")[0] for n in chosen)),
        spectral_features_directly_visible=sum(n.startswith("spectral__") for n in chosen),
        correction_presence_descriptor=dict(checked_slots=checked, incorrect_slots=mismatches,
            scope="one structural representative per baseline K2/K3/K4; not error counts"),
        baseline_posterior=dynamic["fixed_head"](b).tolist(),
        active_keep_adapter_values=np.exp(head_logits[:, 10:, 7])[active[:, 10:]].tolist(),
        low_k_keep_target=options["targets"](np.array([0, 1]), np.array([2, 3]))[3].tolist(),
    )


def load_feature_rows(root, ids, y, b, folds, pos):
    by_id = {}
    for file in sorted(root.rglob("rows.jsonl")):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            ident = int(row["global_index"])
            require(ident not in by_id, "duplicate feature ID")
            by_id[ident] = row
    require(set(by_id) == set(map(int, ids[pos])), "feature coverage mismatch")
    rows = [by_id[int(ids[i])] for i in pos]
    names = sorted(rows[0]["features"])
    for i, row in zip(pos, rows):
        require(row["true_k"] == y[i] and row["baseline_k"] == b[i] and
                row["fold"] == folds[i], "feature/label/fold alignment mismatch")
        for field in ("features", "fundamental", "detuned", "scrambled"):
            require(sorted(row[field]) == names, "feature schema mismatch")
    return rows, names


def producer_dependencies(y, folds, ids):
    """Reconstruct actual logistic producer train IDs from its split/mask code.

    Checks the audit-reference producer, not only the receiving row's own
    OOF prediction. Existence of an inner dependency is not outer-test leakage.
    """
    records = []
    for outer in sorted(np.unique(folds)):
        for receiver in sorted(set(folds) - {outer}):
            for reference in sorted(set(folds) - {outer, receiver}):
                train = (folds != outer) & (folds != reference) & (y >= 2)
                overlap = ids[train & (folds == receiver)]
                records.append(dict(outer_fold=int(outer), receiver_fold=int(receiver),
                    reference_prediction_fold=int(reference),
                    specialist_train_rows=int(train.sum()), receiver_ids_in_producer_train=int(len(overlap)),
                    overlap_id_sha256=hashlib.sha256(overlap.astype("<i8").tobytes()).hexdigest(),
                    outer_ids_in_producer_train=int(np.sum(train & (folds == outer)))))
    return records


def write_csv(path, rows):
    require(bool(rows), "empty evidence table")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_variant(root, name, feature_root, output, repo):
    path = root/name/"predictions-and-selected-combinations.npz"
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    ids, y, b, p, folds = (data[k] for k in
        ("global_index", "true_K", "frozen_baseline_K", "predicted_K", "fold"))
    pos = aligned_positions(ids, data["eligible_global_index"])
    require(len(y) == 59309 and len(pos) == 7493, "native size drift")
    require(int(np.sum(y == b)) == 48454 and int(np.sum(y >= 2)) == 7385
            and int(np.sum((y == b) & (y >= 2))) == 2530, "reference drift")
    require(set(folds) == {0, 1, 2, 4}, "fold drift")
    require(np.isin(b[pos], (2, 3, 4)).all(), "eligibility changed")
    require(np.isin(p[pos], KS).all(), "unsupported archived prediction")
    eligible = np.zeros(len(y), bool); eligible[pos] = True
    require(np.array_equal(p[~eligible], b[~eligible]), "changed outside eligibility")
    risk = data["candidate_correction_regression_risk"]
    cp, rp, kp = risk[:, :, 0], risk[:, :, 1], data["keep_probability"]
    require(all(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in (cp, rp, kp)),
            "invalid probabilities")
    recovered, scores = legacy_decode(cp, rp, kp, b[pos])
    require(np.array_equal(recovered, p[pos]), "legacy replay differs from archive")
    replay = b.copy(); replay[pos] = expected_net_replay(cp, rp, b[pos])
    rows, names = load_feature_rows(feature_root, ids, y, b, folds, pos)
    # KEEP probabilities use the policy-preferred label (including wrong K0/K1).
    changed = p[pos] != b[pos]
    target_index = p[pos] - 2
    selected_cp = cp[np.arange(len(pos)), target_index]
    selected_rp = rp[np.arange(len(pos)), target_index]
    negative = changed & (selected_cp < selected_rp)
    k3_reg = (y[pos] == 3) & (b[pos] == 3) & (p[pos] != 3)
    audit_rows = []
    for local, native in enumerate(pos):
        row = rows[local]
        record = dict(global_index=int(ids[native]), fold=int(folds[native]),
            recording_id=row["recording_id"], start_sample=int(row["start_sample"]),
            true_K=int(y[native]), frozen_K=int(b[native]), predicted_K=int(p[native]),
            expected_net_replay_K=int(replay[native]),
            baseline_correct=int(b[native] == y[native]),
            correction=int(b[native] != y[native] and p[native] == y[native]),
            regression=int(b[native] == y[native] and p[native] != y[native]),
            selected_negative_estimated_net=int(negative[local]),
            keep_preferred_probability=float(kp[local]), keep_logit=float(scores[local, 5]))
        for j, k in enumerate(KS):
            record.update({f"K{k}_correct_prob": float(cp[local, j]),
                           f"K{k}_regress_prob": float(rp[local, j]),
                           f"K{k}_action_score": float(scores[local, j])})
        record.update(row["features"])
        audit_rows.append(record)
    write_csv(output/f"{name}-event-matrix.csv", audit_rows)
    write_csv(output/f"{name}-k3-regressions.csv", [r for r, mask in zip(audit_rows, k3_reg) if mask])
    summary_fields = ("global_index", "fold", "recording_id", "start_sample", "true_K",
        "frozen_K", "predicted_K", "expected_net_replay_K", "selected_negative_estimated_net",
        "keep_preferred_probability", "keep_logit")
    write_csv(output/f"{name}-k3-regressions-summary.csv",
              [{k: r[k] for k in summary_fields} for r, mask in zip(audit_rows, k3_reg) if mask])
    masked_rp = np.ma.array(rp, mask=(KS[None, :] == b[pos, None]))
    spread = np.asarray(masked_rp.max(axis=1) - masked_rp.min(axis=1))
    result = dict(artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reproduction_disagreements=int(np.sum(recovered != p[pos])),
        frozen_reference=metrics(y, b, b), original_selector=metrics(y, b, p),
        expected_net_replay=metrics(y, b, replay),
        replay_decision_changes=int(np.sum(replay != p)),
        negative_estimated_net_selected=dict(rows=int(negative.sum()),
            corrections=int(np.sum(negative & (p[pos] == y[pos]) & (b[pos] != y[pos]))),
            regressions=int(np.sum(negative & (p[pos] != y[pos]) & (b[pos] == y[pos])))),
        k3_regressions=int(k3_reg.sum()),
        same_event_regression_risk_spread=dict(median=float(np.median(spread)),
            p95=float(np.quantile(spread, .95)), above_point_one=int(np.sum(spread > .1))),
        coverage=dict(eligible_true_poly=int(np.sum(eligible & (y >= 2))),
            eligible_true_low_k=int(np.sum(eligible & (y < 2))),
            poly_errors_outside_eligibility=int(np.sum((y >= 2) & ~eligible & (y != b))),
            true_poly_predicted_K0_K1=int(np.sum((y >= 2) & (b < 2))),
            perfect_eligible_poly_upper_bound=int(np.sum((y >= 2) & (eligible | (b == y)))),
            poly_rows=int(np.sum(y >= 2))),
        by_fold={str(f):dict(original=metrics(y[folds == f], b[folds == f], p[folds == f]),
                     replay=metrics(y[folds == f], b[folds == f], replay[folds == f])) for f in (0, 1, 2, 4)})
    return result, names, producer_dependencies(y[pos], folds[pos], ids[pos])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite audit")
    args.output.mkdir(parents=True)
    results = {}
    for mode in ("baseline", "both"):
        result, names, provenance = audit_variant(args.artifacts, mode, args.features,
                                                   args.output, args.source_root)
        results[mode] = result
    # Constructed calibrated-posterior counterexample, explicitly not real audio.
    correct = np.array([[.30, .40, .20, .06, .04]])
    regress = np.full((1, 5), .40)
    b = np.array([3]); keep = np.array([.40])
    old, _ = legacy_decode(correct, regress, keep, b)
    alternate = expected_net_replay(correct, regress, b)
    report = dict(status="completed", source_ref=REVIEW_REF, evaluated_run_ref=RUN_REF,
        evaluated_run=37758789956, independent_validation=False, promotion=False,
        replay_only=True, variants=results, source_contract=source_contracts(args.source_root, names),
        inner_audit_producer_dependencies=provenance,
        calibrated_posterior_counterexample=dict(truth_probabilities=correct[0].tolist(),
            baseline_K=3, old_prediction=int(old[0]), expected_net_prediction=int(alternate[0]),
            interpretation="synthetic probability distribution, not measured native performance"),
        source_sha256={name:hashlib.sha256((args.source_root/name).read_bytes()).hexdigest()
                       for name in AUDITED_FILES},
        limitations=["No weights retrained and no fresh holdout evaluated.",
            "A decoder intervention is confounded by heads trained jointly with the old score and weighted loss.",
            "Inner producer dependence is reconstructed from actual split logic and artifact IDs; no claim of outer-test-label leakage.",
            "Missing direct spectral context does not imply spectral evidence is absent from specialist outputs.",
            "Theoretical eligibility ceiling uses labels for diagnosis only, never prediction.",
            "Matrices include observable features and archived risks, but not individual fundamental candidates, head posteriors, or audio clips absent from these artifacts."])
    (args.output/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"source_contract":report["source_contract"],
                      "variants":{m:{"reproduction":r["reproduction_disagreements"],
                        "original":r["original_selector"]["global_metrics"],
                        "replay":r["expected_net_replay"]["global_metrics"],
                        "coverage":r["coverage"]} for m,r in results.items()}},indent=2))


if __name__ == "__main__":
    main()
