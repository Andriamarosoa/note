"""Fixed 4-signal consensus follow-up on dual-engine-confirmed x4 disappearances.

Post-selection development audit. The four signals were chosen from the preceding
cross-fold signal audit because each had stable fold-oriented univariate AUC.

For every held-out fold:
- thresholds are computed ONLY on the other three folds as the midpoint between
  the true-K2 and true-K3 medians;
- directions are fixed a priori from the prior mechanistic audit;
- action requires 3 of 4 votes.

Also reports 2/4, 4/4 and leave-one-feature-out majority diagnostics.
No fold-3 data and no promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOLDS = (0, 1, 2, 4)

FEATURES = (
    ("attackf0__nov_pre_norm_range", "le"),
    ("attackf0__nov_onset_contrast_raw_range", "le"),
    ("attackf0__nov_post1_norm_median", "ge"),
    ("attackf0__nov_positive_retained_fraction_range", "le"),
)


def require(c, m):
    if not c:
        raise RuntimeError(m)


def discover(root):
    rows = []
    for p in root.rglob("rows.jsonl"):
        rr = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        rows.extend(rr)
    require(len(rows) == 76, f"cohort drift: {len(rows)}")
    require(len({int(r["row_id"]) for r in rows}) == 76, "duplicate row")
    return sorted(rows, key=lambda r: (int(r["fold"]), int(r["row_id"])))


def midpoint_threshold(rows, feature):
    k2 = np.asarray([r["features"][feature] for r in rows if int(r["true_k"]) == 2], float)
    k3 = np.asarray([r["features"][feature] for r in rows if int(r["true_k"]) == 3], float)
    require(len(k2) and len(k3), "class collapse")
    return float((np.median(k2) + np.median(k3)) / 2.0)


def vote(value, threshold, direction):
    return value <= threshold if direction == "le" else value >= threshold


def evaluate_rule(rows, feature_specs, votes_required):
    rotations = []
    for f in FOLDS:
        fit = [r for r in rows if int(r["fold"]) != f]
        val = [r for r in rows if int(r["fold"]) == f]
        thresholds = {
            name: midpoint_threshold(fit, name) for name, _ in feature_specs
        }

        actions = []
        for r in val:
            votes = sum(
                vote(float(r["features"][name]), thresholds[name], direction)
                for name, direction in feature_specs
            )
            actions.append(votes >= votes_required)

        y = np.asarray([int(r["true_k"]) for r in val], int)
        take = np.asarray(actions, bool)
        corr = int(np.sum(take & (y == 2)))
        reg = int(np.sum(take & (y == 3)))
        rotations.append({
            "fold": int(f),
            "val_rows": len(val),
            "val_K2": int(np.sum(y == 2)),
            "val_K3": int(np.sum(y == 3)),
            "thresholds": thresholds,
            "actions": int(take.sum()),
            "corrections": corr,
            "regressions": reg,
            "net": corr - reg,
        })

    total = {
        key: int(sum(r[key] for r in rotations))
        for key in ("actions", "corrections", "regressions", "net")
    }
    return {"rotations": rotations, "total": total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    require(not a.output.exists(), "refusing overwrite")
    rows = discover(a.input_root)
    k2 = sum(int(r["true_k"]) == 2 for r in rows)
    k3 = sum(int(r["true_k"]) == 3 for r in rows)
    require((k2, k3) == (40, 36), "truth accounting drift")

    rules = {
        "vote_2_of_4": evaluate_rule(rows, FEATURES, 2),
        "vote_3_of_4": evaluate_rule(rows, FEATURES, 3),
        "vote_4_of_4": evaluate_rule(rows, FEATURES, 4),
    }

    lofo = {}
    for drop, _ in FEATURES:
        specs = tuple(x for x in FEATURES if x[0] != drop)
        lofo[drop] = evaluate_rule(rows, specs, 2)  # majority of remaining 3

    candidate = rules["vote_3_of_4"]
    folds_positive = sum(r["net"] > 0 for r in candidate["rotations"])
    folds_nonnegative = sum(r["net"] >= 0 for r in candidate["rotations"])
    min_fold_net = min(r["net"] for r in candidate["rotations"])

    report = {
        "status": "completed",
        "experiment": "v273_confirmed_x4_stable_quartet_consensus",
        "post_selection_followup": True,
        "independent_confirmation": False,
        "rows": 76,
        "K2": 40,
        "K3": 36,
        "outer_fold_3_used": False,
        "baseline_take_all": {
            "actions": 76,
            "corrections": 40,
            "regressions": 36,
            "net": 4,
        },
        "features": [
            {"name": name, "direction_for_K2": direction}
            for name, direction in FEATURES
        ],
        "threshold_method": "midpoint of FIT-only K2 and K3 medians",
        "rules": rules,
        "candidate_rule": "vote_3_of_4",
        "candidate_stability": {
            "positive_folds": folds_positive,
            "nonnegative_folds": folds_nonnegative,
            "min_fold_net": int(min_fold_net),
        },
        "leave_one_feature_out_majority": lofo,
        "automatic_promotion": False,
        "limitations": [
            "Feature identities were selected after inspection of the same internal folds.",
            "This is therefore a post-selection development audit, not independent confirmation.",
        ],
    }

    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Confirmed x4 stable-quartet consensus",
        "",
        "Post-selection development audit on 76 dual-engine-confirmed disappearances.",
        "",
        "Baseline take-all: **40 corrections / 36 regressions = +4**.",
        "",
        "| rule | actions | corrections | regressions | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("vote_2_of_4", "vote_3_of_4", "vote_4_of_4"):
        q = rules[name]["total"]
        lines.append(
            f"| {name} | {q['actions']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |"
        )

    lines += [
        "",
        "## 3-of-4 held-out folds",
        "",
        "| fold | actions | corrections | regressions | net |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in candidate["rotations"]:
        lines.append(
            f"| {r['fold']} | {r['actions']} | {r['corrections']} | {r['regressions']} | {r['net']:+d} |"
        )

    lines += [
        "",
        "## Leave-one-feature-out majority",
        "",
        "| removed feature | corrections | regressions | net |",
        "|---|---:|---:|---:|",
    ]
    for feat, q in lofo.items():
        t = q["total"]
        lines.append(f"| {feat} | {t['corrections']} | {t['regressions']} | {t['net']:+d} |")

    lines += [
        "",
        f"3-of-4 stability: **{folds_positive}/4 positive folds**, "
        f"**{folds_nonnegative}/4 non-negative**, minimum fold net **{min_fold_net:+d}**.",
        "",
        "No promotion: the same folds were used to discover these four signals.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
