"""Recompute the fixed three-arm result without TensorFlow or evaluator imports.

Checks native predictions, probability/decision consistency and exported producer
ID manifests. It does not establish unseen-data generalization or calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARTIFACTS = {
    "legacy": (11545671181, "e447a18990c534cf4b5cd405b1bba430fe5c046de75552faa0327110f71a3022"),
    "contracts": (11544644114, "1581c57d5b35db3fdbf0cf9ad21c188da15458aa0d48894cf455d27187c0c309"),
    "coherent": (11544713559, "58717a0298c5dd899622326660f10a0b597236ecc2bb70210caebc81d13c9b91"),
}
FOLDS = {0, 1, 2, 4}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_digest(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(y, p):
    n = len(y)
    c = int(np.count_nonzero(y == p))
    return dict(rows=n, correct=c, exact=c/n if n else None,
                under=int(np.count_nonzero(p < y)), over=int(np.count_nonzero(p > y)))


def metrics(y, p):
    out = counts(y, p)
    out["poly"] = counts(y[y >= 2], p[y >= 2])
    out["by_k"] = {str(k): counts(y[y == k], p[y == k]) for k in range(7)}
    out["confusion_true_by_predicted"] = np.bincount(7*y+p, minlength=49).reshape(7, 7).tolist()
    return out


def pair_counts(y, b, p):
    fixed = int(np.count_nonzero((b != y) & (p == y)))
    broken = int(np.count_nonzero((b == y) & (p != y)))
    return dict(corrections=fixed, regressions=broken, net=fixed-broken,
                changed=int(np.count_nonzero(p != b)))


def paired(y, b, p):
    return dict(global_=pair_counts(y, b, p),
                poly=pair_counts(y[y >= 2], b[y >= 2], p[y >= 2]),
                by_k={str(k): pair_counts(y[y == k], b[y == k], p[y == k]) for k in range(7)})


def named_pairs(y, b, p):
    out = paired(y, b, p)
    out["global"] = out.pop("global_")
    return out


def verify_provenance(report, ids, y, folds):
    producers = {}
    for producer in report["expert_producers"]:
        key = tuple(producer["train_folds"])
        require(key == tuple(sorted(set(key))) and set(key) < FOLDS, "producer folds")
        fit = np.isin(folds, key) & (y >= 2)
        require(np.array_equal(producer["train_global_ids"], ids[fit]), "producer IDs")
        require(producer["train_rows"] == int(fit.sum()), "producer count")
        require(producer["train_id_sha256"] == ids_digest(ids[fit]), "producer digest")
        require(producer["true_k_counts"] == np.bincount(y[fit], minlength=7).tolist(), "producer labels")
        require(key not in producers, "duplicate producer")
        producers[key] = producer
    require(len(producers) == 14, "expected all fourteen training fold subsets")
    paths = 0
    outers = set()

    def check_path(path, excluded):
        predicted = path["reference_prediction_fold"]
        require(set(path["excluded_folds"]) == excluded, "producer exclusions")
        require(set(path["train_folds"]) == FOLDS-excluded-{predicted}, "producer fit scope")
        producer = producers[tuple(path["train_folds"])]
        require(path["train_id_sha256"] == producer["train_id_sha256"], "path producer digest")
        forbidden_ids = set(ids[np.isin(folds, list(excluded | {predicted}))])
        require(not forbidden_ids.intersection(producer["train_global_ids"]), "forbidden training ID")
        require(path["forbidden_train_overlap"] == 0, "reported path overlap")

    for outer in report["provenance"]:
        o = outer["outer_fold"]
        require(o in FOLDS and o not in outers, "outer fold")
        outers.add(o)
        require(outer["outer_train_ids_sha256"] == ids_digest(ids[folds != o]), "outer train IDs")
        require(outer["outer_test_ids_sha256"] == ids_digest(ids[folds == o]), "outer test IDs")
        require(outer["outer_test_train_overlap"] == 0, "outer overlap")
        require(outer["audit_context_scalers_and_clusters_exclude_receiver"] is True, "audit preprocessing scope")
        require({p["reference_prediction_fold"] for p in outer["outer_train_oof_producers"]} == FOLDS-{o}, "outer OOF coverage")
        for path in outer["outer_train_oof_producers"]:
            check_path(path, {o})
        receivers = set()
        for audit in outer["inner_audits"]:
            receiver = audit["receiver_fold"]
            require(receiver != o and receiver not in receivers and receiver in FOLDS, "receiver fold")
            receivers.add(receiver)
            refs = FOLDS-{o, receiver}
            reference = np.isin(folds, list(refs))
            require(audit["outer_fold"] == o and set(audit["reference_folds"]) == refs, "audit folds")
            require(audit["audit_and_regime_train_rows"] == int(reference.sum()), "audit fit count")
            require(audit["audit_and_regime_train_id_sha256"] == ids_digest(ids[reference]), "audit fit IDs")
            require(audit["receiver_id_sha256"] == ids_digest(ids[folds == receiver]), "audit receiver IDs")
            require(audit["forbidden_train_overlap"] == 0, "audit overlap")
            require({p["reference_prediction_fold"] for p in audit["reference_expert_producers"]} == refs, "audit reference coverage")
            for path in audit["reference_expert_producers"]:
                check_path(path, {o, receiver})
                paths += 1
        require(receivers == FOLDS-{o}, "audit receiver coverage")
    require(outers == FOLDS and paths == 24, "all outer folds and nested paths")
    return dict(distinct_producer_training_sets=len(producers), nested_reference_paths=paths,
                checked_against_native_ids=True, forbidden_id_overlap=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True, help="Contains arm.zip and arm/{report.json,predictions.npz}")
    parser.add_argument("--reference", type=Path, required=True, help="Archived both predictions NPZ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as z:
        reference = {k: z[k] for k in z.files}
    ids, y, b, folds, eligible = (reference[k] for k in
        ("global_index", "true_K", "frozen_baseline_K", "fold", "eligible_global_index"))
    require(len(ids) == 59309 and len(set(ids)) == 59309 and len(eligible) == 7493, "native cohort")
    position = {int(v): i for i, v in enumerate(ids)}
    ep = np.array([position[int(v)] for v in eligible])
    active = np.zeros(len(ids), bool); active[ep] = True
    require(np.array_equal(active, np.isin(b, [2, 3, 4])), "eligible scope")
    require(set(folds) == FOLDS and int(np.sum(y >= 2)) == 7385, "fold/poly cohort")
    require(int(np.sum(y == b)) == 48454 and int(np.sum((y == b) & (y >= 2))) == 2530, "frozen scores")
    summary = dict(status="verified", run_id=37766719862,
        source_commit="3774441791695c23bb81b4039892d65e7faa77ae",
        native_rows=len(y), eligible_rows=len(ep), freeze_reference=metrics(y, b),
        independent_validation=False, promotion=False, arms={})
    predictions = {}
    manifests = {}
    for arm, (artifact_id, sha) in ARTIFACTS.items():
        require(digest(args.artifacts/(arm+".zip")) == sha, "artifact archive digest")
        directory = args.artifacts/arm
        report = json.loads((directory/"report.json").read_text())
        with np.load(directory/"predictions.npz", allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        for name in ("global_index", "true_K", "frozen_baseline_K", "fold", "eligible_global_index"):
            require(np.array_equal(data[name], reference[name]), arm+" native alignment "+name)
        require(report["reference_sha256"] == digest(args.reference), "source NPZ digest")
        require(report["arm"] == arm and report["status"] == "completed", "reported arm/status")
        require(report["promotion"] is False and report["independent_validation"] is False, "reported limits")
        p = data["predicted_K"]
        require(np.array_equal(p[~active], b[~active]), "change outside eligible events")
        require(np.isin(p[active], [2, 3, 4, 5, 6]).all(), "forbidden output destination")
        measured = metrics(y, p); pair = named_pairs(y, b, p)
        require(report["freeze_reference"] == summary["freeze_reference"], "reported baseline")
        require(report["candidate"] == measured and report["paired"] == pair, "reported metric mismatch")
        for f in sorted(FOLDS):
            take = active & (folds == f)
            require(report["folds"][str(f)]["paired"] == named_pairs(y[take], b[take], p[take]), "reported fold metric")
            require(report["folds"][str(f)]["eligible_rows"] == int(take.sum()), "reported fold count")
        difference = int(np.count_nonzero(p != reference["predicted_K"]))
        require(difference == report["changes_from_archived_selector"], "legacy distance")
        entry = dict(artifact_id=artifact_id, archive_sha256=sha,
            predictions_sha256=digest(directory/"predictions.npz"),
            candidate=measured, paired=pair, changes_from_archived_selector=difference,
            fold_nets={f: report["folds"][f]["paired"]["global"]["net"] for f in report["folds"]},
            parameters=report["configuration"]["parameter_counts"])
        if arm == "legacy":
            require(difference == 0, "legacy reproduction")
        else:
            entry["provenance_verification"] = verify_provenance(report, eligible, y[ep], folds[ep])
            manifests[arm] = (report["provenance"], report["expert_producers"])
        attention = data["selected_subset_probabilities"]
        require(attention.shape == (7493, 5, 64) and np.isfinite(attention).all() and (attention >= 0).all(), "attention array")
        for source in (2, 3, 4):
            at = b[ep] == source
            for target in range(2, 7):
                w = attention[at, target-2]
                if target == source:
                    require(np.count_nonzero(w) == 0, "baseline attention mask")
                else:
                    require(np.allclose(w.sum(1), 1, atol=1e-6), "attention normalization")
                    if (source, target) not in {(2, 3), (3, 2), (3, 4), (4, 3)}:
                        require(np.count_nonzero(w[:, 32:]) == 0, "32 subset mask")
        if arm == "coherent":
            q = data["class_probability"]
            require(q.shape == (7493, 7) and np.isfinite(q).all() and (q >= 0).all(), "probability array")
            require(np.allclose(q.sum(1), 1, atol=1e-6), "probability normalization")
            row = np.arange(len(ep)); pb = q[row, b[ep]]
            risk = data["candidate_correction_regression_risk"]
            require(np.array_equal(risk[:, :, 0], q[:, 2:]), "candidate posterior")
            require(np.array_equal(risk[:, :, 1], np.broadcast_to(pb[:, None], (len(ep), 5))), "single regression risk")
            require(np.array_equal(data["keep_probability"], pb), "KEEP correctness")
            best = 2+q[:, 2:].argmax(1)
            decoded = np.where(q[row, best] > pb, best, b[ep])
            require(np.array_equal(decoded, p[ep]), "expected gain decoder")
            entry["coherence_verification"] = dict(probability_rows=len(ep), decoder_mismatches=0,
                max_probability_sum_error=float(np.max(np.abs(q.sum(1)-1))),
                regression_risk_spread=float(np.max(np.ptp(risk[:, :, 1], axis=1))),
                empirical_calibration_established=False)
        summary["arms"][arm] = entry
        predictions[arm] = p
    require(manifests["contracts"] == manifests["coherent"], "repaired arms provenance differs")
    summary["coherent_versus_contracts"] = named_pairs(y, predictions["contracts"], predictions["coherent"])
    summary["coherent_versus_legacy"] = named_pairs(y, predictions["legacy"], predictions["coherent"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": summary["status"], "nets": {a: s["paired"]["global"]["net"] for a, s in summary["arms"].items()}}))


if __name__ == "__main__":
    main()
