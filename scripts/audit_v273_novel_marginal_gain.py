"""FIT-only Exact-K audit of a novel marginal third-component gain feature.

The reconstruction is frozen to archived pool64_t2 frequencies/residuals.
Only one new inference feature is added. No annotations and no fold 3.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_decay_guard import accounting
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_attack_novelty import harmonic_mask
from scripts.v273_residual_audit import FOLDS, fit_classifier, model_state, require, sha256_file, write_json

PROTOCOL = Path("analysis/v273-novel-marginal-gain-protocol.md")
NAMES = ("control_residual2", "plus_novel3")
EPS = 1e-12


def fit_small_nnls(D, x):
    """Exact active-subset NNLS for at most three columns."""
    k = D.shape[1]
    require(1 <= k <= 3, "small NNLS supports 1..3 columns")
    x2 = float(x @ x)
    best = (x2, np.zeros(k), ())
    for n in range(1, k + 1):
        for active in itertools.combinations(range(k), n):
            ids = list(active)
            A = D[:, ids]
            G = A.T @ A
            b = A.T @ x
            a = np.linalg.solve(G + 1e-10 * np.eye(n), b)
            if np.any(a < 0):
                continue
            pred = A @ a
            cost = float(np.sum((x - pred) ** 2))
            if cost < best[0] - 1e-15:
                full = np.zeros(k)
                full[ids] = a
                best = (cost, full, tuple(active))
    return best


def marginal_feature(freq, x, triplet_f0):
    D = np.column_stack([h.template(freq, float(f)) for f in triplet_f0])
    j3, a3, _ = fit_small_nnls(D, x)
    y3 = D @ a3

    best_pair = None
    for pair in ((0, 1), (0, 2), (1, 2)):
        D2 = D[:, pair]
        j2, a2, _ = fit_small_nnls(D2, x)
        if best_pair is None or (j2, pair) < (best_pair[0], best_pair[1]):
            best_pair = (j2, pair, a2, D2 @ a2)

    j2, pair, _, y2 = best_pair
    missing = ({0, 1, 2} - set(pair)).pop()
    own = harmonic_mask(freq, float(triplet_f0[missing]))
    peer = np.zeros(len(freq), dtype=bool)
    for idx in pair:
        peer |= harmonic_mask(freq, float(triplet_f0[idx]))
    unique = own & ~peer

    delta = y3 - y2
    delta2 = float(delta @ delta)
    unique_delta2 = float(delta[unique] @ delta[unique])
    unique_fraction = unique_delta2 / (delta2 + EPS)
    x2 = float(x @ x)
    marginal_gain = max(float(j2 - j3), 0.0) / (x2 + EPS)
    novel = marginal_gain * unique_fraction
    return {
        "novel_marginal_gain": float(novel),
        "marginal_gain": float(marginal_gain),
        "delta_unique_fraction": float(unique_fraction),
        "best_internal_pair": list(pair),
        "marginal_component": int(missing),
        "j2_ratio": float(j2 / (x2 + EPS)),
        "j3_ratio": float(j3 / (x2 + EPS)),
    }


def model_probability(Xfit, yfit, Xval, states=None):
    train = np.isin(yfit, (2, 3))
    require(len(np.unique(yfit[train])) == 2, "one-class FIT")
    model = fit_classifier(Xfit[train], (yfit[train] == 2).astype(int))
    if states is not None:
        states.append(model_state(model))
    return model.predict_proba(Xval)[:, 1]


def select_on_fit(X2, X3, y, fold):
    models = (X2, X3)
    reports, predictions = [], []
    for X in models:
        p = np.empty(len(y), dtype=np.float64)
        for heldout in np.unique(fold):
            valid = fold == heldout
            p[valid] = model_probability(X[~valid], y[~valid], X[valid])
        reports.append(accounting(y, p))
        predictions.append(p)
    chosen = min(
        range(len(NAMES)),
        key=lambda i: (-reports[i]["global_net"], reports[i]["regressions"],
                       reports[i]["applied"], i),
    )
    return (chosen if reports[chosen]["global_net"] > 0 else None), reports, np.asarray(predictions)


def safe_auc(y, score):
    mask = np.isin(y, (2, 3))
    yy = (y[mask] == 2).astype(int)
    if len(np.unique(yy)) < 2:
        return None
    return float(roc_auc_score(yy, score[mask]))


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")
    inputs, records = action_inputs(a.exports)

    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open("rb") as stream:
            require(hashlib.file_digest(stream, "md5").hexdigest() == expected,
                    "dataset changed: " + name)

    with np.load(a.capacity / "features.npz", allow_pickle=False) as z:
        capacity = dict(z)
    ids = capacity["row_id"]
    require(set(map(int, ids)) == set(records), "capacity/control row identity changed")
    control = capacity["features"][:, 3, :2]
    triplet_f0 = capacity["triplet_f0"][:, 3]
    by_id_index = {int(row): i for i, row in enumerate(ids)}

    wanted = {member for member, _ in records.values()}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset)
              if t.annotation_member in wanted}
    require(set(tracks) == wanted, "missing audio")
    by_member = {}
    for row, (member, start) in records.items():
        by_member.setdefault(member, []).append((row, start))

    computed = {}
    for member in sorted(wanted):
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768.0
        for row, start in by_member[member]:
            freq, x = h.transition_spectrum(samples, start)
            idx = by_id_index[int(row)]
            computed[int(row)] = marginal_feature(freq, x, triplet_f0[idx])
        print(json.dumps({"recording": member, "rows": len(computed)}), flush=True)

    require(len(computed) == len(ids), "incomplete feature extraction")
    novel = np.array([computed[int(row)]["novel_marginal_gain"] for row in ids])
    details = np.array([
        [computed[int(row)]["marginal_gain"],
         computed[int(row)]["delta_unique_fraction"],
         computed[int(row)]["j2_ratio"],
         computed[int(row)]["j3_ratio"]]
        for row in ids
    ])

    # Frozen control: archived residuals are reused without recomputation.
    features3 = np.column_stack([control, novel])
    a.output.mkdir(parents=True)
    np.savez_compressed(
        a.output / "features.npz",
        row_id=ids,
        control_residuals=control,
        novel_marginal_gain=novel,
        diagnostics=details,
        diagnostic_names=np.asarray([
            "marginal_gain", "delta_unique_fraction", "internal_pair_residual_ratio",
            "triplet_replay_residual_ratio"
        ]),
    )
    (a.output / "rows.jsonl").write_text(
        "".join(json.dumps({"row_id": int(row), **computed[int(row)]}, sort_keys=True) + "\n"
                for row in ids),
        encoding="utf-8",
    )

    reports, scores = [], {}
    aucs = {}
    for outer, arr in inputs.items():
        data = {}
        for split in ("fit", "val"):
            mask = arr[split + "_b_low"] & (arr[split + "_base_k"] == 3)
            valid = arr[split + "_valid"]
            row_ids = arr[split + "_ids"][mask][valid]
            ix = np.array([by_id_index[int(i)] for i in row_ids])
            data[split] = {
                "X2": control[ix],
                "X3": features3[ix],
                "novel": novel[ix],
                "y": arr[split + "_true_k"][mask][valid],
                "fold": arr[split + "_fold"][mask][valid],
                "ids": row_ids,
            }

        fit, val = data["fit"], data["val"]
        selected, cv, inner = select_on_fit(fit["X2"], fit["X3"], fit["y"], fit["fold"])
        states = []
        vp2 = model_probability(fit["X2"], fit["y"], val["X2"], states)
        vp3 = model_probability(fit["X3"], fit["y"], val["X3"], states)
        write_json(a.output / f"models-fold-{outer}.json", dict(zip(NAMES, states)))
        vp = np.asarray([vp2, vp3])
        chosen = np.zeros(len(val["y"])) if selected is None else vp[selected]

        aucs[str(outer)] = {
            "novel_univariate_val_k2_vs_k3": safe_auc(val["y"], val["novel"]),
            "control_model_val_k2_vs_k3": safe_auc(val["y"], vp2),
            "plus_novel_model_val_k2_vs_k3": safe_auc(val["y"], vp3),
        }
        reports.append({
            "fold": outer,
            "selected": None if selected is None else NAMES[selected],
            "fit_cv": dict(zip(NAMES, cv)),
            "val_fixed": {
                NAMES[0]: accounting(val["y"], vp2),
                NAMES[1]: accounting(val["y"], vp3),
            },
            "val_selected": accounting(val["y"], chosen),
            "auc": aucs[str(outer)],
        })

        for split in ("fit", "val"):
            for key in ("ids", "y", "fold"):
                scores[f"fold_{outer}_{split}_{key}"] = data[split][key]
            scores[f"fold_{outer}_{split}_novel"] = data[split]["novel"]
        scores[f"fold_{outer}_inner_probability"] = inner
        scores[f"fold_{outer}_val_probability"] = vp

    np.savez_compressed(a.output / "scores.npz", **scores)

    def total(rows):
        return {k: sum(r[k] for r in rows) for k in
                ("rows", "applied", "corrections", "regressions",
                 "other_k_actions", "global_net")}

    report = {
        "status": "completed",
        "experiment": "v273_novel_marginal_gain",
        "folds": reports,
        "aucs": aucs,
        "variants": list(NAMES),
        "outer_fold_3_used": False,
        "annotation_frequencies_used": False,
        "neural_training": False,
        "threshold_search": False,
        "automatic_promotion": False,
        "unique_audio_rows": len(ids),
        "recordings": len(wanted),
        "feature_definition": "max(J2-J3,0)/||x||^2 * ||(y3-y2)[unique_third_bins]||^2/||y3-y2||^2",
        "total_selected": total([r["val_selected"] for r in reports]),
        "total_fixed": {
            name: total([r["val_fixed"][name] for r in reports])
            for name in NAMES
        },
        "selection": "FIT-only rotation; net, fewer regressions, fewer actions, model order; abstain if net <= 0",
        "source_run": 37356100423,
        "source_data_md5": DATA_MD5,
        "source_manifests_sha256": {
            str(f): sha256_file(a.exports / f"fold-{f}" / "manifest.json") for f in FOLDS
        },
        "source_sha256": {
            "protocol": sha256_file(PROTOCOL),
            "script": sha256_file(__file__),
            "capacity_features": sha256_file(a.capacity / "features.npz"),
        },
        "rows_sha256": sha256_file(a.output / "rows.jsonl"),
        "runtime": {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p) for p in
               ("numpy", "scipy", "scikit-learn")},
        },
        "limitations": [
            "Internal folds have already been inspected; this is exploratory research.",
            "The triplet frequencies and two residual controls are frozen from pool64_t2.",
            "The new feature uses no annotations but is motivated by prior annotation-conditioned diagnostics.",
            "Normal residual-audio path only; no compressed-path conclusion.",
        ],
    }

    # Control must reproduce the already archived pool64_t2 fixed accounting.
    previous = json.loads((a.capacity / "report.json").read_text())
    require(report["total_fixed"]["control_residual2"] ==
            previous["total_fixed"]["pool64_t2"],
            "control Exact-K accounting changed")

    write_json(a.output / "report.json", report)

    lines = [
        "# Novel marginal third-component gain",
        "",
        "| model | corrections | regressions | other-K actions | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        q = report["total_fixed"][name]
        lines.append(
            f"| {name} | {q['corrections']} | {q['regressions']} | "
            f"{q['other_k_actions']} | {q['global_net']:+d} |"
        )
    q = report["total_selected"]
    lines += [
        "",
        f"FIT-selected policy: **{q['corrections']} corrections / "
        f"{q['regressions']} regressions, net {q['global_net']:+d}**.",
        "",
        "Selections: " + ", ".join(
            f"{r['fold']}={r['selected'] or 'abstain'}" for r in reports
        ) + ".",
        "",
        "Univariate novel-feature AUC K2-vs-K3 by VAL fold: " +
        ", ".join(
            f"{f}={aucs[str(f)]['novel_univariate_val_k2_vs_k3']:.3f}"
            for f in FOLDS
        ) + ".",
        "",
        "Fold 3 excluded. No promotion. Normal-audio path only.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exports", "dataset", "capacity", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
