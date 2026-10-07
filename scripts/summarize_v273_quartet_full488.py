"""Validate stable quartet beyond the dual-engine-common subset.

Thresholds are learned only from 'both' rows in FIT folds, then applied unchanged
to all four x4 categories in the held-out fold:
- both
- librosa_only
- rubberband_only
- neither

This tests whether the physical attack-homogeneity signal generalizes to rows
that were not used to discover or tune the quartet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOLDS = (0, 1, 2, 4)
CATEGORIES = ("both", "librosa_only", "rubberband_only", "neither")
FEATURES = (
    ("nov_pre_norm_range", "le"),
    ("nov_onset_contrast_raw_range", "le"),
    ("nov_post1_norm_median", "ge"),
    ("nov_positive_retained_fraction_range", "le"),
)


def require(c, m):
    if not c:
        raise RuntimeError(m)


def discover(root):
    rows = []
    reports = {}
    for rp in root.rglob("report.json"):
        try:
            rep = json.loads(rp.read_text())
        except Exception:
            continue
        if rep.get("experiment") != "v273_quartet_full488_extract":
            continue
        f = int(rep["fold"])
        require(f in FOLDS and f not in reports, "duplicate/bad fold")
        jp = rp.parent / "rows.jsonl"
        rr = [json.loads(x) for x in jp.read_text().splitlines() if x.strip()]
        require(len(rr) == int(rep["rows"]), "row count mismatch")
        rows.extend(rr)
        reports[f] = rep
    require(set(reports) == set(FOLDS), f"missing folds {set(FOLDS)-set(reports)}")
    require(len(rows) == 488, f"cohort drift {len(rows)}")
    require(len({int(r["row_id"]) for r in rows}) == 488, "duplicate rows")
    return sorted(rows, key=lambda r: (int(r["fold"]), int(r["row_id"])))


def threshold(fit_common, feature):
    k2 = np.asarray([r["features"][feature] for r in fit_common if int(r["true_k"]) == 2], float)
    k3 = np.asarray([r["features"][feature] for r in fit_common if int(r["true_k"]) == 3], float)
    require(len(k2) and len(k3), "common FIT class collapse")
    return float((np.median(k2) + np.median(k3)) / 2.0)


def row_action(row, th):
    votes = 0
    for feat, direction in FEATURES:
        x = float(row["features"][feat])
        votes += x <= th[feat] if direction == "le" else x >= th[feat]
    return votes >= 3


def accounting(rows, th):
    action = np.asarray([row_action(r, th) for r in rows], bool)
    y = np.asarray([int(r["true_k"]) for r in rows], int)
    corr = int(np.sum(action & (y == 2)))
    reg = int(np.sum(action & (y == 3)))
    return {
        "rows": len(rows),
        "K2": int(np.sum(y == 2)),
        "K3": int(np.sum(y == 3)),
        "actions": int(action.sum()),
        "corrections": corr,
        "regressions": reg,
        "net": corr - reg,
        "precision_K2_among_actions": float(corr / action.sum()) if action.any() else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    require(not a.output.exists(), "refusing overwrite")
    rows = discover(a.input_root)

    global_counts = {
        c: {
            "rows": sum(r["category"] == c for r in rows),
            "K2": sum(r["category"] == c and int(r["true_k"]) == 2 for r in rows),
            "K3": sum(r["category"] == c and int(r["true_k"]) == 3 for r in rows),
        }
        for c in CATEGORIES
    }
    require(global_counts["both"] == {"rows": 76, "K2": 40, "K3": 36}, "both cohort drift")
    require(global_counts["librosa_only"]["rows"] == 23, "librosa-only drift")
    require(global_counts["rubberband_only"]["rows"] == 30, "rubber-only drift")
    require(global_counts["neither"]["rows"] == 359, "neither drift")

    rotations = []
    category_totals = {
        c: {"rows": 0, "K2": 0, "K3": 0, "actions": 0, "corrections": 0, "regressions": 0, "net": 0}
        for c in CATEGORIES
    }
    all_total = {"rows": 0, "K2": 0, "K3": 0, "actions": 0, "corrections": 0, "regressions": 0, "net": 0}

    for f in FOLDS:
        fit_common = [r for r in rows if int(r["fold"]) != f and r["category"] == "both"]
        val = [r for r in rows if int(r["fold"]) == f]
        th = {feat: threshold(fit_common, feat) for feat, _ in FEATURES}

        cats = {}
        for c in CATEGORIES:
            rr = [r for r in val if r["category"] == c]
            q = accounting(rr, th)
            cats[c] = q
            for key in category_totals[c]:
                category_totals[c][key] += int(q[key]) if key != "precision_K2_among_actions" else 0

        qa = accounting(val, th)
        for key in all_total:
            all_total[key] += int(qa[key])

        rotations.append({
            "fold": int(f),
            "thresholds_from_common_fit_only": th,
            "categories": cats,
            "all_488_fold": qa,
        })

    # Recompute category precision after aggregate accounting.
    for c, q in category_totals.items():
        q["precision_K2_among_actions"] = (
            float(q["corrections"] / q["actions"]) if q["actions"] else None
        )
    all_total["precision_K2_among_actions"] = (
        float(all_total["corrections"] / all_total["actions"]) if all_total["actions"] else None
    )

    common_only = category_totals["both"]
    outside = {
        key: int(sum(category_totals[c][key] for c in ("librosa_only", "rubberband_only", "neither")))
        for key in ("rows", "K2", "K3", "actions", "corrections", "regressions", "net")
    }
    outside["precision_K2_among_actions"] = (
        float(outside["corrections"] / outside["actions"]) if outside["actions"] else None
    )

    report = {
        "status": "completed",
        "experiment": "v273_quartet_full488_validation",
        "post_selection_followup": True,
        "independent_row_subset_validation": True,
        "rows": 488,
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "threshold_training_population": "dual-engine-common rows from FIT folds only",
        "candidate_rule": "3 of 4 fixed attack-homogeneity votes",
        "global_category_counts": global_counts,
        "category_totals": category_totals,
        "outside_common_total": outside,
        "all_rows_total": all_total,
        "rotations": rotations,
        "automatic_promotion": False,
        "limitations": [
            "Feature identities came from the prior common-76 analysis.",
            "The non-common 412 rows are a new row subset for this quartet, but share the same recordings/folds.",
        ],
    }

    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def pp(v):
        return "n/a" if v is None else f"{100*v:.1f}%"

    lines = [
        "# Stable quartet beyond the common x4 subset",
        "",
        "Thresholds are trained **only on dual-engine-common FIT rows**.",
        "",
        "| category | rows | K2/K3 | actions | corrections | regressions | net | K2 precision among actions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CATEGORIES:
        q = category_totals[c]
        lines.append(
            f"| {c} | {q['rows']} | {q['K2']}/{q['K3']} | {q['actions']} | "
            f"{q['corrections']} | {q['regressions']} | {q['net']:+d} | "
            f"{pp(q['precision_K2_among_actions'])} |"
        )

    lines += [
        "",
        f"Outside the 76 common rows: **{outside['corrections']} corrections / "
        f"{outside['regressions']} regressions = {outside['net']:+d}** "
        f"over {outside['actions']} actions.",
        f"All 488 rows: **{all_total['corrections']} corrections / "
        f"{all_total['regressions']} regressions = {all_total['net']:+d}** "
        f"over {all_total['actions']} actions.",
        "",
        "## Held-out folds — all categories combined",
        "",
        "| fold | actions | corrections | regressions | net |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rotations:
        q = r["all_488_fold"]
        lines.append(
            f"| {r['fold']} | {q['actions']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |"
        )

    lines += [
        "",
        "No promotion; this is a post-selection validation on a broader row population.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
