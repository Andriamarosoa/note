"""Cross-fold test of pickup-energy + signed-flow features on the frozen 488 rows."""
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
EXPERIMENT = "v273_energy_flow_extract"


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
            random_state=27391,
        ),
    )


def auc_or_none(y, s):
    y = np.asarray(y, np.int32)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def threshold_candidates(score):
    vals = np.unique(np.asarray(score, np.float64))
    out = [float(np.nextafter(vals.max(), np.inf))]
    out += [float(v) for v in vals]
    if len(vals) > 1:
        out += [float((a + b) / 2.0) for a, b in zip(vals[:-1], vals[1:])]
    return sorted(set(out))


def accounting(y, score, threshold):
    # Positive class = true K2.  The frozen 488-row cohort has baseline K3,
    # therefore "take" means propose K3 -> K2.
    take = np.asarray(score, np.float64) >= float(threshold)
    y = np.asarray(y, np.int32)
    corr = int(np.sum(take & (y == 1)))
    reg = int(np.sum(take & (y == 0)))
    return {
        "actions": int(np.sum(take)),
        "corrections": corr,
        "regressions": reg,
        "net": corr - reg,
    }


def select_threshold(y, score):
    cand = []
    for t in threshold_candidates(score):
        cand.append({"threshold": float(t), **accounting(y, score, t)})
    best = min(cand, key=lambda q: (-q["net"], q["regressions"], q["actions"], -q["threshold"]))
    if best["net"] <= 0:
        t = float(np.nextafter(np.max(score), np.inf))
        return {"threshold": t, **accounting(y, score, t)}
    return best


def inner_oof(X, y, folds, fit_mask):
    out = np.full(len(y), np.nan, np.float64)
    for vf in sorted(set(int(v) for v in folds[fit_mask])):
        tr = fit_mask & (folds != vf)
        va = fit_mask & (folds == vf)
        require(tr.any() and va.any() and len(np.unique(y[tr])) == 2, "inner split failure")
        m = classifier()
        m.fit(X[tr], y[tr])
        out[va] = m.predict_proba(X[va])[:, 1]
    require(np.isfinite(out[fit_mask]).all(), "inner OOF missing")
    return out


def evaluate(X, y, folds):
    oof = np.full(len(y), np.nan, np.float64)
    rotations = []
    for vf in FOLDS:
        fit = folds != vf
        val = folds == vf
        require(fit.any() and val.any() and len(np.unique(y[fit])) == 2, "outer split failure")
        nested = inner_oof(X, y, folds, fit)
        selected = select_threshold(y[fit], nested[fit])
        m = classifier()
        m.fit(X[fit], y[fit])
        p = m.predict_proba(X[val])[:, 1]
        oof[val] = p
        rotations.append({
            "val_fold": int(vf),
            "val_rows": int(np.sum(val)),
            "val_K2": int(np.sum(y[val] == 1)),
            "val_K3": int(np.sum(y[val] == 0)),
            "val_auc": auc_or_none(y[val], p),
            "selected_threshold": float(selected["threshold"]),
            "nested_fit_policy": {k: int(selected[k]) for k in ("actions","corrections","regressions","net")},
            "val_policy": accounting(y[val], p, selected["threshold"]),
        })
    require(np.isfinite(oof).all(), "outer OOF missing")
    total = {
        k: int(sum(r["val_policy"][k] for r in rotations))
        for k in ("actions", "corrections", "regressions", "net")
    }
    return {
        "oof_auc": auc_or_none(y, oof),
        "per_fold_auc": {str(r["val_fold"]): r["val_auc"] for r in rotations},
        "selected_policy_total": total,
        "rotations": rotations,
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
            m2 = np.median(X[fit & (y == 1), j])
            m3 = np.median(X[fit & (y == 0), j])
            orient = 1.0 if m2 >= m3 else -1.0
            a = float(roc_auc_score(y[val], orient * X[val, j]))
            vals.append(a)
            per[str(f)] = a
        if vals:
            out.append({
                "feature": name,
                "mean_fold_oriented_auc": float(np.mean(vals)),
                "min_fold_auc": float(np.min(vals)),
                "max_fold_auc": float(np.max(vals)),
                "per_fold": per,
            })
    out.sort(key=lambda q: (q["mean_fold_oriented_auc"], q["min_fold_auc"]), reverse=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    require(not a.output.exists(), "refusing overwrite")
    rows, reports = discover(a.input_root)
    require(len(rows) == 488, f"cohort drift: {len(rows)}")
    require(len({int(r["row_id"]) for r in rows}) == 488, "duplicate row")

    rows = sorted(rows, key=lambda r: (int(r["fold"]), int(r["row_id"])))
    y = np.asarray([1 if int(r["true_k"]) == 2 else 0 for r in rows], np.int32)
    folds = np.asarray([int(r["fold"]) for r in rows], np.int32)
    require(set(folds.tolist()) == set(FOLDS), "fold drift")
    require(int(np.sum(y == 1)) == 216 and int(np.sum(y == 0)) == 272,
            f"truth accounting drift K2={np.sum(y==1)} K3={np.sum(y==0)}")

    names = sorted(rows[0]["features"])
    require(all(sorted(r["features"]) == names for r in rows), "schema drift")

    groups = {
        "energy_only": ("energy__",),
        "flux_only": ("flux__",),
        "flow_geometry_only": ("geometry__",),
        "energy_flux": ("energy__", "flux__"),
        "energy_flux_geometry": ("energy__", "flux__", "geometry__"),
        "interaction_only": ("interaction__",),
        "full_energy_flow": ("energy__", "flux__", "geometry__", "interaction__"),
    }

    families = {}
    family_features = {}
    for family, prefs in groups.items():
        fn = [n for n in names if any(n.startswith(p) for p in prefs)]
        require(fn, f"empty family {family}")
        X = np.asarray([[float(r["features"][n]) for n in fn] for r in rows], np.float64)
        require(np.isfinite(X).all(), f"nonfinite {family}")
        family_features[family] = fn
        families[family] = {"dimensions": len(fn), **evaluate(X, y, folds)}

    uni = univariate(rows, names, y, folds)
    best = max(
        families,
        key=lambda k: (
            families[k]["oof_auc"] if families[k]["oof_auc"] is not None else -1.0,
            families[k]["selected_policy_total"]["net"],
        ),
    )

    # Baseline K3 on this cohort: all 272 true K3 are correct, all 216 K2 are wrong.
    baseline = {
        "rows": 488,
        "correct": 272,
        "exact": 272 / 488,
        "under": 0,
        "over": 216,
    }

    report = {
        "status": "completed",
        "experiment": "v273_energy_flow_crossfold",
        "rows": 488,
        "K2": 216,
        "K3": 272,
        "folds": list(FOLDS),
        "fold3_used": False,
        "baseline_all_predicted_K3": baseline,
        "families": families,
        "family_features": family_features,
        "best_family_by_oof_auc_then_net": best,
        "univariate": uni,
        "source_reports": {str(k): v for k, v in reports.items()},
        "automatic_promotion": False,
        "interpretation_guard": (
            "Features are pickup-domain energy/flow proxies. A positive result supports "
            "a dynamic energy-state signal; it does not establish literal fluid-energy flux."
        ),
        "notes": [
            "Feature extraction is label-independent.",
            "Each outer rotation is evaluated on one untouched fold.",
            "K3->K2 action thresholds are selected only from nested OOF predictions inside FIT folds.",
            "No fold-3 rows are used.",
        ],
    }

    a.output.mkdir(parents=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# V27.3 energy-flow audit",
        "",
        "Frozen difficult cohort: **488 baseline-K3 rows = 216 true K2 / 272 true K3**.",
        "",
        "These are mono guitar-pickup energy/flow proxies, not direct acoustic intensity.",
        "",
        "| family | dims | OOF AUC | actions | corrections | regressions | net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, q in families.items():
        t = q["selected_policy_total"]
        auc = "n/a" if q["oof_auc"] is None else f"{q['oof_auc']:.4f}"
        lines.append(
            f"| {family} | {q['dimensions']} | {auc} | {t['actions']} | "
            f"{t['corrections']} | {t['regressions']} | {t['net']:+d} |"
        )

    lines += [
        "",
        f"Best family by OOF AUC then net: **{best}**.",
        "",
        "## Best individual signals",
        "",
        "| feature | mean fold-oriented AUC | minimum fold AUC |",
        "|---|---:|---:|",
    ]
    for q in uni[:20]:
        lines.append(
            f"| {q['feature']} | {q['mean_fold_oriented_auc']:.4f} | {q['min_fold_auc']:.4f} |"
        )
    lines += [
        "",
        "Held-out folds only for reported AUC/policy; fold 3 excluded.",
        "Development audit only; no correction is automatically promoted.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
