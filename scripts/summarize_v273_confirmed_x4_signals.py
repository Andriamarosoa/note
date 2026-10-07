"""Cross-fold evaluation of signals on dual-engine-confirmed x4 disappearances."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDS = (0, 1, 2, 4)
EXPERIMENT = "v273_confirmed_x4_signal_extract"


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
        if rep.get("experiment") != EXPERIMENT:
            continue
        f = int(rep["fold"])
        require(f in FOLDS and f not in reports, "duplicate/bad fold")
        jp = rp.parent / "rows.jsonl"
        require(jp.exists(), "missing rows.jsonl")
        rr = [json.loads(x) for x in jp.read_text().splitlines() if x.strip()]
        require(len(rr) == int(rep["rows"]), "row count drift")
        rows.extend(rr)
        reports[f] = rep
    require(set(reports) == set(FOLDS), f"missing folds {set(FOLDS)-set(reports)}")
    return rows, reports


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=27376,
        ),
    )


def auc_or_none(y, score):
    y = np.asarray(y, np.int32)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def threshold_candidates(score):
    s = np.asarray(score, np.float64)
    vals = np.unique(s)
    out = [float(np.nextafter(vals.max(), np.inf))]  # abstain
    out += [float(v) for v in vals]
    if len(vals) > 1:
        out += [float((a + b) / 2.0) for a, b in zip(vals[:-1], vals[1:])]
    return sorted(set(out))


def accounting(y, score, threshold):
    y = np.asarray(y, np.int32)
    take = np.asarray(score, np.float64) >= float(threshold)
    corr = int(np.sum(take & (y == 1)))
    reg = int(np.sum(take & (y == 0)))
    return {
        "actions": int(take.sum()),
        "corrections": corr,
        "regressions": reg,
        "net": corr - reg,
    }


def select_threshold(y, score):
    candidates = []
    for t in threshold_candidates(score):
        q = accounting(y, score, t)
        candidates.append({"threshold": float(t), **q})
    best = min(
        candidates,
        key=lambda q: (-q["net"], q["regressions"], q["actions"], -q["threshold"]),
    )
    if best["net"] <= 0:
        # Explicit abstention.
        t = float(np.nextafter(np.max(score), np.inf))
        return {"threshold": t, **accounting(y, score, t)}, candidates
    return best, candidates


def inner_oof_scores(X, y, folds, fit_mask):
    fit_folds = sorted(set(int(x) for x in folds[fit_mask]))
    out = np.full(len(y), np.nan, np.float64)
    for inner_val in fit_folds:
        tr = fit_mask & (folds != inner_val)
        va = fit_mask & (folds == inner_val)
        require(tr.any() and va.any(), "empty nested split")
        require(len(np.unique(y[tr])) == 2, f"nested class collapse train excluding fold {inner_val}")
        m = classifier()
        m.fit(X[tr], y[tr])
        out[va] = m.predict_proba(X[va])[:, 1]
    require(np.isfinite(out[fit_mask]).all(), "nested OOF score missing")
    return out


def evaluate_family(X, y, folds):
    oof = np.full(len(y), np.nan, np.float64)
    rotations = []
    for val_fold in FOLDS:
        fit = folds != val_fold
        val = folds == val_fold
        require(fit.any() and val.any(), "empty rotation")
        require(len(np.unique(y[fit])) == 2, "fit class collapse")
        nested = inner_oof_scores(X, y, folds, fit)
        selected, _ = select_threshold(y[fit], nested[fit])

        m = classifier()
        m.fit(X[fit], y[fit])
        p = m.predict_proba(X[val])[:, 1]
        oof[val] = p
        q = accounting(y[val], p, selected["threshold"])
        rotations.append({
            "val_fold": int(val_fold),
            "fit_rows": int(fit.sum()),
            "val_rows": int(val.sum()),
            "val_K2": int(np.sum(y[val] == 1)),
            "val_K3": int(np.sum(y[val] == 0)),
            "val_auc": auc_or_none(y[val], p),
            "selected_threshold": float(selected["threshold"]),
            "nested_fit_selection": {
                "actions": int(selected["actions"]),
                "corrections": int(selected["corrections"]),
                "regressions": int(selected["regressions"]),
                "net": int(selected["net"]),
            },
            "val_policy": q,
        })

    require(np.isfinite(oof).all(), "OOF prediction missing")
    total = {
        key: int(sum(r["val_policy"][key] for r in rotations))
        for key in ("actions", "corrections", "regressions", "net")
    }
    return {
        "oof_auc": auc_or_none(y, oof),
        "per_fold_auc": {str(r["val_fold"]): r["val_auc"] for r in rotations},
        "rotations": rotations,
        "selected_policy_total": total,
    }


def univariate(rows, names, y, folds):
    X = np.asarray([[float(r["features"][n]) for n in names] for r in rows], np.float64)
    out = []
    for j, name in enumerate(names):
        vals = []
        per = {}
        for f in FOLDS:
            fit = folds != f
            val = folds == f
            if len(np.unique(y[val])) < 2:
                per[str(f)] = None
                continue
            med2 = np.median(X[fit & (y == 1), j])
            med3 = np.median(X[fit & (y == 0), j])
            orient = 1.0 if med2 >= med3 else -1.0
            a = float(roc_auc_score(y[val], orient * X[val, j]))
            vals.append(a)
            per[str(f)] = a
        if vals:
            out.append({
                "feature": name,
                "mean_oof_auc": float(np.mean(vals)),
                "min_oof_auc": float(np.min(vals)),
                "max_oof_auc": float(np.max(vals)),
                "per_fold": per,
            })
    out.sort(key=lambda q: (q["mean_oof_auc"], q["min_oof_auc"]), reverse=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    require(not a.output.exists(), "refusing overwrite")
    rows, source = discover(a.input_root)
    require(len(rows) == 76, f"confirmed common cohort drift: {len(rows)}")
    require(len({int(r["row_id"]) for r in rows}) == 76, "duplicate row")

    rows = sorted(rows, key=lambda r: (int(r["fold"]), int(r["row_id"])))
    y = np.asarray([1 if int(r["true_k"]) == 2 else 0 for r in rows], np.int32)
    folds = np.asarray([int(r["fold"]) for r in rows], np.int32)
    require(int(np.sum(y == 1)) == 40 and int(np.sum(y == 0)) == 36, "truth accounting drift")

    names = sorted(rows[0]["features"])
    require(all(sorted(r["features"]) == names for r in rows), "feature schema drift")

    prefixes = {
        "transient": ("transient__",),
        "attack_f0_exclusive": ("attackf0__",),
        "harmonic_reconstruction": ("harmonic__",),
        "spectral_morphology": ("morph__",),
        "trajectory": ("trajectory__",),
        "x4_confidence": ("confidence__",),
        "physical_audio_combined": ("transient__", "attackf0__", "harmonic__", "morph__"),
        "all_combined": ("transient__", "attackf0__", "harmonic__", "morph__", "trajectory__", "confidence__"),
    }

    families = {}
    family_names = {}
    for family, prefs in prefixes.items():
        fn = [n for n in names if any(n.startswith(p) for p in prefs)]
        require(fn, f"empty family {family}")
        family_names[family] = fn
        X = np.asarray([[float(r["features"][n]) for n in fn] for r in rows], np.float64)
        require(np.isfinite(X).all(), f"nonfinite family {family}")
        families[family] = {
            "dimensions": len(fn),
            **evaluate_family(X, y, folds),
        }

    uni = univariate(rows, names, y, folds)

    baseline = {
        "actions": 76,
        "corrections": 40,
        "regressions": 36,
        "net": 4,
    }

    best_family = max(
        families,
        key=lambda k: (
            families[k]["selected_policy_total"]["net"],
            families[k]["oof_auc"] if families[k]["oof_auc"] is not None else -1,
        ),
    )

    report = {
        "status": "completed",
        "experiment": "v273_confirmed_x4_signal_crossfold",
        "rows": 76,
        "K2": 40,
        "K3": 36,
        "folds": list(FOLDS),
        "outer_fold_3_used": False,
        "baseline_take_all_confirmed_x4": baseline,
        "families": families,
        "family_features": family_names,
        "best_family_by_val_net_then_auc": best_family,
        "univariate_oof": uni,
        "source_extract_reports": {str(k): v for k, v in source.items()},
        "automatic_promotion": False,
        "notes": [
            "All feature extraction is label-independent.",
            "Each outer rotation trains on three folds and tests the held-out fold.",
            "The action threshold is selected only from nested OOF predictions inside the three FIT folds.",
        ],
    }

    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Confirmed x4 disappearance — signal audit",
        "",
        "Cohort: **76 dual-engine-confirmed K3→K2 disappearances** "
        "(40 true K2 / 36 true K3).",
        "",
        "Take-all baseline: **40 corrections / 36 regressions = +4**.",
        "",
        "| signal family | dims | OOF AUC | selected actions | corrections | regressions | VAL net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, q in families.items():
        t = q["selected_policy_total"]
        auc = "n/a" if q["oof_auc"] is None else f"{q['oof_auc']:.3f}"
        lines.append(
            f"| {family} | {q['dimensions']} | {auc} | {t['actions']} | "
            f"{t['corrections']} | {t['regressions']} | {t['net']:+d} |"
        )

    lines += [
        "",
        f"Best family by held-out net then AUC: **{best_family}**.",
        "",
        "## Best individual signals",
        "",
        "| feature | mean fold-oriented AUC | minimum fold AUC |",
        "|---|---:|---:|",
    ]
    for q in uni[:15]:
        lines.append(
            f"| {q['feature']} | {q['mean_oof_auc']:.3f} | {q['min_oof_auc']:.3f} |"
        )

    lines += [
        "",
        "Thresholds are selected inside FIT only; fold 3 is excluded.",
        "Diagnostic/development result only; no automatic correction is promoted.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
