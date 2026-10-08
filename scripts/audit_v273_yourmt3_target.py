"""Compare the preserved catalogue with the pinned YourMT3+ Exact-K run.

No training or policy selection for deployment. Oracle scores use labels only
to establish finite-catalogue ceilings, never as deployable predictions.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.yourmt3_exactk_common import (
    CHECKPOINT_SHA256, FOLDS, SPACE_REVISION, digest, metrics, paired, require,
)

UPSTREAM_SHA256 = "d47cbe59f66139dbbbb0c38b1dc2a1eb050d7ea25062f1dbba096971cb56eb89"


def load_npz(source):
    with np.load(source, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def align(source_ids, target_ids):
    require(len(np.unique(source_ids)) == len(source_ids), "duplicate source IDs")
    require(len(np.unique(target_ids)) == len(target_ids), "duplicate target IDs")
    require(np.array_equal(np.sort(source_ids), np.sort(target_ids)), "cohort mismatch")
    order = np.argsort(source_ids)
    return order[np.searchsorted(source_ids[order], target_ids)]


def audit(root, output):
    evidence = root / "analysis/evidence"
    loops = evidence / "v273-regression-loops"
    archive = evidence / "v273-yourmt3-target/upstream/yourmt3-exactk-summary.zip"
    require(digest(archive) == UPSTREAM_SHA256, "YourMT3+ archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        upstream_report = json.loads(z.read("report.json"))
        upstream = load_npz(io.BytesIO(z.read("predictions.npz")))
    require(upstream_report["status"] == "completed", "incomplete YourMT3+ run")
    require(tuple(upstream_report["folds"]) == FOLDS, "YourMT3+ fold drift")
    for fold in FOLDS:
        r = upstream_report["by_fold"][str(fold)]
        require(r["status"] == "completed" and not r["smoke_subset"], "partial fold")
        require(r["protocol"]["checkpoint_sha256"] == CHECKPOINT_SHA256,
                "YourMT3+ checkpoint drift")
        require(r["protocol"]["space_revision"] == SPACE_REVISION,
                "YourMT3+ source drift")

    inputs_path = loops / "prepared/inputs.npz"
    d = load_npz(inputs_path)
    ids, y, base, fold = (d[k] for k in
                          ("native_global_index", "native_truth", "native_baseline", "native_fold"))
    require(len(y) == 59309 and set(np.unique(fold)) == set(FOLDS), "native cohort drift")
    order = align(upstream["global_index"], ids)
    for key, target in (("k", y), ("baseline", base), ("fold", fold)):
        require(np.array_equal(upstream[key][order], target), f"YourMT3+ {key} mismatch")
    competitor = upstream["predicted"][order]
    require(metrics(y, competitor) == upstream_report["yourmt3_plus"], "YourMT3+ score drift")
    require(metrics(y, base) == upstream_report["freeze_local_combo"], "reference score drift")
    require(paired(y, base, competitor) == upstream_report["paired"], "paired score drift")
    for f in FOLDS:
        mask = fold == f
        expected = upstream_report["by_fold"][str(f)]
        require(metrics(y[mask], competitor[mask]) == expected["yourmt3_plus"],
                "YourMT3+ fold score drift")
        require(metrics(y[mask], base[mask]) == expected["freeze_local_combo"],
                "reference fold score drift")
        require(paired(y[mask], base[mask], competitor[mask]) == expected["paired"],
                "paired fold score drift")

    memories = [loops / "all-candidate-decisions.npz",
                loops / "posthoc-audit/appended-candidates.npz"]
    names, predictions = [], []
    for path in memories:
        memory = load_npz(path)
        order = align(memory["global_index"], ids)
        for key, target in (("true_K", y), ("frozen_baseline_K", base), ("fold", fold)):
            require(np.array_equal(memory[key][order], target), f"memory {key} mismatch")
        names.extend(memory["variant_ids"].tolist())
        predictions.append(memory["predictions"][order])
    require(len(names) == len(set(names)) == 211, "candidate inventory drift")
    predictions = np.column_stack(predictions)
    require(predictions.shape == (len(y), len(names)), "candidate shape mismatch")
    require(np.all((predictions >= 0) & (predictions <= 6)), "invalid candidate class")
    scores = np.sum(predictions == y[:, None], axis=0)
    best_index = int(np.argmax(scores))
    best = predictions[:, best_index]
    competitor_metrics = metrics(y, competitor)
    best_metrics = metrics(y, best)

    pos, proposal = d["native_position"], d["proposal"]
    require(proposal.shape == (7493, 255), "proposal inventory drift")
    require(np.array_equal(ids[pos], d["global_index"]), "proposal ID mismatch")
    require(np.array_equal(y[pos], d["truth"]), "proposal truth mismatch")
    require(np.array_equal(base[pos], d["baseline"]), "proposal baseline mismatch")
    require(np.array_equal(fold[pos], d["fold"]), "proposal fold mismatch")
    require(np.all((proposal >= 2) & (proposal <= 6)), "proposal class drift")
    require(np.array_equal(np.flatnonzero(np.isin(base, [2, 3, 4])), np.sort(pos)),
            "eligibility drift")

    # Reachability is the exact finite action set: KEEP plus every whole group.
    # No single-member veto, no estimated probability threshold, no fitting.
    actions = np.zeros((len(y), 7), bool)
    actions[np.arange(len(y)), base] = True
    actions[np.repeat(pos, proposal.shape[1]), proposal.ravel()] = True
    current_reachable = actions[np.arange(len(y)), y]
    historical_reachable = np.any(predictions == y[:, None], axis=1) | (base == y)
    combined_reachable = current_reachable | historical_reachable
    eligible = np.zeros(len(y), bool)
    eligible[pos] = True
    oracle_masks = {"current_255_plus_keep": current_reachable,
                    "historical_211_plus_keep": historical_reachable,
                    "current_and_historical_plus_keep": combined_reachable}
    ceilings = {}
    for name, mask in oracle_masks.items():
        oracle = np.where(mask, y, base)
        require(np.array_equal(oracle == y, mask), "invalid oracle construction")
        m = metrics(y, oracle)
        ceilings[name] = {
            "oracle_not_a_model": True, "metrics": m,
            "additional_correct_vs_freeze": int(np.sum(mask & (base != y))),
            "global_gap_to_yourmt3_correct": competitor_metrics["correct"] - m["correct"],
            "minimum_extra_correct_to_exceed_global": max(0, competitor_metrics["correct"] + 1 - m["correct"]),
            "can_exceed_yourmt3_global": m["correct"] > competitor_metrics["correct"],
            "can_exceed_yourmt3_poly": m["poly"]["correct"] > competitor_metrics["poly"]["correct"],
        }

    regions = {}
    masks = {"ineligible": ~eligible,
             "eligible_truth_absent": eligible & ~current_reachable,
             "eligible_truth_available": eligible & current_reachable}
    require(np.all(np.sum(list(masks.values()), axis=0) == 1), "regions do not partition cohort")
    for name, mask in masks.items():
        regions[name] = {
            "rows": int(mask.sum()),
            "current_correct": int(np.sum(mask & (best == y))),
            "yourmt3_correct": int(np.sum(mask & (competitor == y))),
            "yourmt3_only_correct": int(np.sum(mask & (competitor == y) & (best != y))),
            "current_only_correct": int(np.sum(mask & (best == y) & (competitor != y))),
            "net_advantage_yourmt3": int(np.sum(mask & (competitor == y)) - np.sum(mask & (best == y))),
        }

    report = {
        "status": "completed", "training_performed": False,
        "automatic_promotion": False, "goal_met": False,
        "descriptive_target_met": best_metrics["correct"] > competitor_metrics["correct"]
        and best_metrics["poly"]["correct"] > competitor_metrics["poly"]["correct"],
        "goal": "exceed_pinned_yourmt3_global_and_poly_then_validate_on_unseen_data",
        "comparison": "paired_descriptive_on_59309_previously_exposed_native_rows",
        "yourmt3_pretraining_overlap_excluded": False,
        "equal_latency_comparison": False,
        "independent_validation": False,
        "folds": list(FOLDS), "policies": len(names),
        "freeze_local_combo": metrics(y, base),
        "yourmt3_plus": competitor_metrics,
        "current_best": {"id": names[best_index], "metrics": best_metrics,
                         "selection": "descriptive_maximum_of_211_exposed_policies_not_new_validation",
                         "versus_yourmt3": paired(y, competitor, best),
                         "versus_freeze": paired(y, base, best)},
        "remaining_current_gap": {
            "global_correct": competitor_metrics["correct"] - best_metrics["correct"],
            "poly_correct": competitor_metrics["poly"]["correct"] - best_metrics["poly"]["correct"],
            "global_points": 100 * (competitor_metrics["exact"] - best_metrics["exact"]),
            "poly_points": 100 * (competitor_metrics["poly"]["exact"] - best_metrics["poly"]["exact"]),
        },
        "ceilings": ceilings, "regions": regions,
        "historical_extra_reachable_errors_beyond_current": int(np.sum(historical_reachable & ~current_reachable)),
        "by_fold": {str(f): {"current": metrics(y[fold == f], best[fold == f]),
                             "yourmt3": metrics(y[fold == f], competitor[fold == f]),
                             "paired_current_vs_yourmt3": paired(y[fold == f], competitor[fold == f], best[fold == f])}
                    for f in FOLDS},
        "inputs_sha256": {str(p.relative_to(root)): digest(p)
                          for p in [archive, inputs_path, *memories]},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    np.savez_compressed(output / "row-evidence.npz", global_index=ids, true_K=y,
                        baseline_K=base, fold=fold, current_best_K=best, yourmt3_K=competitor,
                        eligible=eligible, current_action_set=actions,
                        current_reachable=current_reachable,
                        historical_reachable=historical_reachable,
                        combined_reachable=combined_reachable)
    with (output / "all-211-versus-yourmt3.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant_id", "global_correct", "global_exact",
                               "poly_correct", "poly_exact", "current_only_correct",
                               "yourmt3_only_correct", "net_vs_yourmt3", "exceeds_global_and_poly"])
        writer.writeheader()
        for j, name in enumerate(names):
            m = metrics(y, predictions[:, j])
            p = paired(y, competitor, predictions[:, j])["global"]
            writer.writerow({"variant_id": name, "global_correct": m["correct"],
                             "global_exact": m["exact"], "poly_correct": m["poly"]["correct"],
                             "poly_exact": m["poly"]["exact"], "current_only_correct": p["corrections"],
                             "yourmt3_only_correct": p["regressions"], "net_vs_yourmt3": p["net"],
                             "exceeds_global_and_poly": m["correct"] > competitor_metrics["correct"]
                             and m["poly"]["correct"] > competitor_metrics["poly"]["correct"]})
    print(json.dumps({"goal_met": report["goal_met"], "current_best": names[best_index],
                      "remaining_gap": report["remaining_current_gap"],
                      "oracle_global_correct": {k: v["metrics"]["correct"] for k, v in ceilings.items()}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output)
