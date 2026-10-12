"""S274: label-blind two-active-fold S46 gate, plus segregated DEV-only audit.

Motivation: S219 accepts a specialist proposal from a *single* compatible
held-fold group if other groups abstain. Test a predefined stricter
cross-fold-support rule without using any IDMT XML labels and without
modifying original S46/S69/S80 assets. GuitarSet training overlap means
ANY accuracy printed here is development-only, never official promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.audit_v273_s69_s46_vote_guard import POLICIES

S268_FIELDS = {
    "row_id", "variant_ids", "original_freeze_K", "S18_parent_K",
    "original_19_branch_folds", "S46_193_policy_K",
    "per_fold_specialist_K", "active_fold_count",
    "parent_incompatible_specialists", "contradictory_specialists",
}
S261_FIELDS = {
    "row_id", "freeze_K", "original_fold_ids", "original_S18_by_fold_K",
    "S18_piece_neutral_consensus_K", "S18_quorum_of_3_supported",
    "S18_vote_strength", "S18_abstention_to_freeze",
}
PROPOSAL_FIELDS = {
    "row_id", "variant_ids", "S18_parent_K", "original_S46_K",
    "S274_twofold_S46_K", "eligible_fold_count",
    "support_for_original_proposal",
}
S269_FIELDS = {
    "row_id", "S18_parent_K",
    "balanced_s46_K", "balanced_protected_K",
    "polymax_s46_K", "polymax_protected_K",
    "balanced_fold_veto", "balanced_veto_three_folds",
    "polymax_fold_veto", "polymax_veto_three_folds",
}
LABEL_FIELDS = {"row_id", "true_K", "original_fold", "original_freeze_K"}
BAD = {"true_K", "truth", "label", "labels", "target", "midi", "y", "accuracy", "regression"}


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def load(path, fields, *, label_free=False):
    with np.load(path, allow_pickle=False) as z:
        need(set(z.files) == fields, "Input schema mismatch: " + str(path))
        if label_free:
            need(not BAD.intersection(z.files), "Labels forbidden in gate inputs")
        return {k: z[k].copy() for k in z.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def twofold_gate(original, per_fold, fold_parent, parent, reported_active):
    """NO LABEL INPUTS. Demand support from >=2 compatible original folds."""
    original = np.asarray(original, np.int8)
    per_fold = np.asarray(per_fold, np.int8)
    fold_parent = np.asarray(fold_parent, np.int8)
    parent = np.asarray(parent, np.int8)
    reported_active = np.asarray(reported_active)
    n, p = original.shape
    need(p == 193 and per_fold.shape == (4, n, p)
         and fold_parent.shape == (4, n)
         and parent.shape == (n,)
         and reported_active.shape == (n, p),
         "original S46/S18 shapes drifted")
    need(np.isin(original, range(7)).all()
         and np.isin(parent, range(7)).all()
         and np.isin(per_fold, range(-1, 7)).all(),
         "illegal K class in source proposal")
    eligible = (per_fold >= 0) & (fold_parent[:, :, None] == parent[None, :, None])
    active = eligible.sum(axis=0)
    need(np.array_equal(active, reported_active),
         "S219 active-fold support provenance inconsistent")
    supports = np.sum(eligible & (per_fold == original[None, :, :]), axis=0)
    changes = original != parent[:, None]
    need(np.all(supports[changes] >= 1),
         "S219 original changed without any compatible active fold")
    need(np.array_equal(original[:, 0], parent),
         "first S46 policy must be the unchanged S18 parent")
    # Conservative ablation: only cancel S219 accepted actions, never
    # create a different intervention or break the original policy bank.
    new = np.where(changes & (supports >= 2), original, parent[:, None])
    need(np.all((new == original) | (new == parent[:, None])),
         "gate may only retain or cancel an existing intervention")
    return new.astype(np.int8), supports.astype(np.uint8), active.astype(np.uint8)


def selftest():
    n = 3
    parent = np.array([2, 3, 1], np.int8)
    original = np.broadcast_to(parent[:, None], (n, 193)).copy()
    pf = np.full((4, n, 193), -1, np.int8)
    fp = np.broadcast_to(parent[None, :], (4, n)).copy()
    original[0, 1] = 3
    pf[0, 0, 1] = 3  # One active fold: should become parent 2.
    original[1, 1] = 2
    pf[1, 1, 1] = pf[2, 1, 1] = 2  # Two folds: keep 2.
    original[2, 1] = 2
    pf[2, 2, 1] = pf[3, 2, 1] = 2
    fp[3, 2] = 4  # Wrong-parent fold cannot contribute to quorum.
    active = ((pf >= 0) & (fp[:, :, None] == parent[None, :, None])).sum(0)
    new, supports, counted = twofold_gate(original, pf, fp, parent, active)
    need(new[:, 1].tolist() == [2, 2, 1]
         and supports[:, 1].tolist() == [1, 2, 1]
         and np.array_equal(counted, active),
         "S274 one/twofold and parent-compatibility tests failed")
    # The gate must be invariant to original fold input order.
    z, _, _ = twofold_gate(original, pf[::-1], fp[::-1], parent, active)
    need(np.array_equal(new, z), "S274 fold order changes outcome")
    print("S274 selftest PASS: one fold suppressed; two folds retained; wrong parent ignored")


def propose(s268_path, s261_path, output):
    need(not output.exists(), "cannot overwrite frozen label-blind proposal")
    s = load(s268_path, S268_FIELDS, label_free=True)
    parent = load(s261_path, S261_FIELDS, label_free=True)
    ids = s["row_id"]
    n = len(ids)
    need(n == 59309 and np.array_equal(ids, parent["row_id"])
         and np.array_equal(s["original_19_branch_folds"],
                            np.array([0]*5 + [1]*4 + [2]*5 + [4]*5, np.int8)),
         "S274 requires exact 19 S267 branches / 59309 original events")
    need(np.array_equal(s["S18_parent_K"], parent["S18_piece_neutral_consensus_K"])
         and np.array_equal(s["original_freeze_K"], parent["freeze_K"]),
         "S18 parent changed since S268")
    original = s["S46_193_policy_K"]
    updated, supporters, active = twofold_gate(
        original, s["per_fold_specialist_K"],
        parent["original_S18_by_fold_K"],
        s["S18_parent_K"], s["active_fold_count"])
    changed = original != s["S18_parent_K"][:, None]
    retained = updated != s["S18_parent_K"][:, None]
    need(np.all(~retained | changed), "twofold gate creates an intervention")
    output.mkdir(parents=True)
    path = output / "S274_label_free_S46_twofold_GuitarSet_59309.npz"
    np.savez_compressed(path,
        row_id=ids, variant_ids=s["variant_ids"],
        S18_parent_K=s["S18_parent_K"],
        original_S46_K=original, S274_twofold_S46_K=updated,
        eligible_fold_count=active,
        support_for_original_proposal=supporters)
    report = dict(
        status="S274_TWO_ACTIVE_FOLD_S46_LABEL_FREE_ABLATION",
        native_events=n, S46_policies=193,
        original_source_S268_SHA256=digest(s268_path),
        original_source_S261_SHA256=digest(s261_path),
        onefold_interventions_suppressed=int(np.count_nonzero(changed & (supporters == 1))),
        two_or_more_fold_interventions_retained=int(np.count_nonzero(retained)),
        originally_active_intervention_cells=int(np.count_nonzero(changed)),
        affected_events=int(np.count_nonzero(np.any(updated != original, axis=1))),
        per_policy_changed_cell_counts={
            "original": int(np.count_nonzero(changed)),
            "candidate": int(np.count_nonzero(retained))},
        no_true_K_or_midi_read=True, IDMT_XML_never_used=True,
        original_weights_unchanged=True,
        new_gate_score_not_measured=True,
        S80_end_to_end_not_replayed=True, S80_promoted=False,
        immutable_candidate_SHA256=digest(path))
    (output / "label_free_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def score(y, base, previous, candidate, mask):
    n = int(mask.sum())
    need(n > 0, "empty scoring group")
    yy = y[mask]
    bb = base[mask]
    pp = previous[mask]
    cc = candidate[mask]
    def ledger(pred):
        changed = pred != bb
        fixes = int(np.count_nonzero(changed & (bb != yy) & (pred == yy)))
        regressions = int(np.count_nonzero(changed & (bb == yy) & (pred != yy)))
        return dict(correct=int(np.count_nonzero(pred == yy)),
                    changes_vs_S18=int(np.count_nonzero(changed)),
                    corrections_vs_S18=fixes, regressions_vs_S18=regressions,
                    net_vs_S18=fixes-regressions)
    return dict(n=n, S18=ledger(bb), S46_S219=ledger(pp),
                S46_S274_twofold=ledger(cc),
                original_to_candidate_fixes=int(np.count_nonzero(
                    (pp != yy) & (cc == yy))),
                original_to_candidate_regressions=int(np.count_nonzero(
                    (pp == yy) & (cc != yy))))


def audit(s274_path, s269_path, labels_path, output):
    need(not output.exists(), "cannot overwrite an audit")
    p = load(s274_path, PROPOSAL_FIELDS, label_free=True)
    s69 = load(s269_path, S269_FIELDS, label_free=True)
    labeled = load(labels_path, LABEL_FIELDS)
    ids = p["row_id"]
    need(len(ids) == 59309
         and np.array_equal(ids, s69["row_id"])
         and np.array_equal(ids, labeled["row_id"]),
         "frozen labels must join exact canonical native event IDs")
    need(np.array_equal(p["S18_parent_K"], s69["S18_parent_K"])
         and np.array_equal(p["S18_parent_K"].shape, (59309,)),
         "source parent shape disagreement")
    y = labeled["true_K"]
    need(np.isin(y, range(7)).all()
         and int(np.count_nonzero(y >= 2)) == 7385,
         "original GuitarSet true_K reference invariants")
    names = list(map(str, p["variant_ids"]))
    result = {}
    for label, (key, _) in POLICIES.items():
        need(key in names, "original pre-registered S69 source policy missing")
        k = names.index(key)
        orig = p["original_S46_K"][:, k]
        new = p["S274_twofold_S46_K"][:, k]
        base = p["S18_parent_K"]
        need(np.array_equal(orig, s69[label + "_s46_K"]),
             "S269 original S46 key must match S274 before gate")
        per_k = {f"K{k0}": score(y, base, orig, new, y == k0)
                 for k0 in range(7) if np.any(y == k0)}
        result[label] = dict(
            original_policy_key=key,
            original_changes=int(np.count_nonzero(orig != base)),
            candidate_changes=int(np.count_nonzero(new != base)),
            changes_abstained=int(np.count_nonzero(orig != new)),
            all_events=score(y, base, orig, new, np.ones(len(y), bool)),
            true_poly_K2to6=score(y, base, orig, new, y >= 2),
            by_true_K=per_k)
    output.mkdir(parents=True)
    report = dict(
        status="S274_S46_ONLY_GUITARSET_DEVELOPMENT_ABLATION_WITH_TRAINING_OVERLAP",
        comparison_scope="S46 selected policies BEFORE S69; not full S80",
        native_events=len(y), native_true_poly=int(np.count_nonzero(y >= 2)),
        source_S274_SHA256=digest(s274_path), source_S269_SHA256=digest(s269_path),
        original_true_K_SHA256=digest(labels_path),
        policies=result,
        labels_opened_only_after_label_free_candidate_written=True,
        original_GuitarSet_training_overlap_in_three_folds=True,
        no_song_disjoint_evaluation=True,
        IDMT_XML_labels_never_used_for_gate_or_tuning=True,
        no_parameter_search_or_model_training=True,
        independent_promotion_gate_satisfied=False, S80_promoted=False)
    (output / "development_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({label: {
        "original_changes": data["original_changes"],
        "candidate_changes": data["candidate_changes"],
        "poly": data["true_poly_K2to6"],
        "all": data["all_events"]
    } for label, data in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("propose", "audit"))
    for k in ("s268", "s261", "s274", "s269", "labels", "out"):
        parser.add_argument("--" + k, type=Path)
    args = parser.parse_args()
    selftest()
    if args.self_test and not args.mode:
        raise SystemExit(0)
    if args.mode == "propose":
        need(all(getattr(args, k) is not None for k in ("s268", "s261", "out")),
             "propose requires exact S268/S261 and a separate output")
        propose(args.s268, args.s261, args.out)
    elif args.mode == "audit":
        need(all(getattr(args, k) is not None for k in ("s274", "s269", "labels", "out")),
             "audit requires frozen S274 candidate, S269, and separately loaded truth")
        audit(args.s274, args.s269, args.labels, args.out)
    else:
        raise SystemExit("choose --mode or --self-test")
