"""FIT-only Exact-K audit of an exclusivity-aware reconstruction rerank.

No annotation frequencies enter inference. The control must reproduce the
archived pool64_t2 arm exactly. Fold 3 is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_candidate_ranking import trace_nms
from scripts.audit_v273_harmonic_decay_guard import accounting, extraction_tables, probability
from scripts.audit_v273_hann_shape_guard import combinations, _pair_cost, _single_cost
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5, match_frequencies
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_attack_novelty import harmonic_mask
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

PROTOCOL = Path("analysis/v273-exclusive-rerank-protocol.md")
NAMES = ("control_pool64_t2", "exclusive_rerank64")
TOP_N = 64
EPS = 1e-12


def all_costs(D, x, k):
    """All active-subset NNLS costs, matching best_fast semantics."""
    require(k in (2, 3), "unsupported k")
    G, b, x2 = D.T @ D, D.T @ x, float(x @ x)
    combo = combinations(D.shape[1], k)
    if k == 2:
        cost = _pair_cost(G, b, x2, combo)
    else:
        GG = G[combo[:, :, None], combo[:, None, :]]
        bb = b[combo]
        coefficient = np.linalg.solve(GG + 1e-10 * np.eye(3), bb[..., None])[..., 0]
        cost = x2 - 2 * np.einsum("ij,ij->i", coefficient, bb)
        cost += np.einsum("ij,ijk,ik->i", coefficient, GG, coefficient)
        cost = np.where(np.all(coefficient >= 0, axis=1), cost, x2)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            cost = np.minimum(cost, _pair_cost(G, b, x2, combo[:, [i, j]]))
        for i in range(3):
            cost = np.minimum(cost, _single_cost(G, b, x2, combo[:, i]))
    return np.maximum(cost, 0.0), combo, x2


def combo_min_unique(freq, x, f0s):
    masks = [harmonic_mask(freq, float(f)) for f in f0s]
    fractions = []
    for i, own in enumerate(masks):
        peer = np.zeros(len(freq), dtype=bool)
        for j, mask in enumerate(masks):
            if i != j:
                peer |= mask
        total = float(x[own].sum())
        unique = float(x[own & ~peer].sum())
        fractions.append(unique / (total + EPS))
    return float(min(fractions)), fractions


def rerank(freq, x, pool_ids, costs, combo, x2, top_n=TOP_N):
    normalized = costs / (x2 + EPS)
    order = np.argsort(normalized, kind="stable")
    top = order[:min(top_n, len(order))]
    best = None
    diagnostics = []
    for rank, idx in enumerate(top):
        ids = combo[idx]
        f0s = h.F0_GRID[pool_ids[ids]]
        minimum, fractions = combo_min_unique(freq, x, f0s)
        score = minimum / (float(normalized[idx]) + EPS)
        key = (score, -float(normalized[idx]), -rank)
        diagnostics.append((int(idx), float(normalized[idx]), minimum, score))
        if best is None or key > best[0]:
            best = (key, int(idx), minimum, fractions, score, rank)
    require(best is not None, "empty rerank")
    _, idx, minimum, fractions, score, rank = best
    return {
        "combo_index": idx,
        "combo": combo[idx],
        "normalized_cost": float(normalized[idx]),
        "min_unique": float(minimum),
        "unique_fractions": [float(v) for v in fractions],
        "rerank_score": float(score),
        "nnls_rank_within_top": int(rank),
    }


def pool64(freq, x, tables):
    neighbors, active, _ = tables
    sampled = x[neighbors]
    peaks = sampled.max(axis=2) * active
    salience = np.zeros(len(h.F0_GRID))
    for j in range(h.MAX_HARMONICS):
        salience += peaks[:, j] / math.sqrt(j + 1)
    _, selected, _ = trace_nms(salience)
    require(len(selected) >= 64, "insufficient NMS pool")
    return selected[:64]


def extract_row(freq, x, tables):
    pool_ids = pool64(freq, x, tables)
    D = tables[2][2.0][:, pool_ids]
    values = np.zeros((2, 3), dtype=np.float64)
    pair_f0 = np.zeros((2, 2), dtype=np.float64)
    triplet_f0 = np.zeros((2, 3), dtype=np.float64)
    exclusivity = np.zeros((2, 2), dtype=np.float64)

    for k, target in ((2, pair_f0), (3, triplet_f0)):
        costs, combo, x2 = all_costs(D, x, k)
        normalized = costs / (x2 + EPS)
        control_idx = int(np.argmin(normalized))
        control_combo = combo[control_idx]
        control_f0 = h.F0_GRID[pool_ids[control_combo]]
        control_min, _ = combo_min_unique(freq, x, control_f0)

        rr = rerank(freq, x, pool_ids, costs, combo, x2)
        rerank_combo = rr["combo"]
        rerank_f0 = h.F0_GRID[pool_ids[rerank_combo]]

        target[0] = control_f0
        target[1] = rerank_f0
        exclusivity[0, k - 2] = control_min
        exclusivity[1, k - 2] = rr["min_unique"]

        values[0, k - 2] = float(normalized[control_idx])
        values[1, k - 2] = rr["normalized_cost"]

    values[:, 2] = np.median(triplet_f0, axis=1)
    return values, pair_f0, triplet_f0, exclusivity


def select_on_fit(X, y, fold):
    require(set(np.unique(fold)) <= set(FOLDS), "outer fold")
    reports, predictions = [], []
    for i in range(len(NAMES)):
        p = np.empty(len(y), dtype=np.float64)
        for heldout in np.unique(fold):
            valid = fold == heldout
            p[valid] = probability(X[~valid, i, :2], y[~valid], X[valid, i, :2])
        reports.append(accounting(y, p))
        predictions.append(p)
    chosen = min(
        range(len(NAMES)),
        key=lambda i: (-reports[i]["global_net"], reports[i]["regressions"],
                       reports[i]["applied"], i),
    )
    return (chosen if reports[chosen]["global_net"] > 0 else None), reports, np.asarray(predictions)


def diagnostic_matches(cases_path, computed):
    if cases_path is None:
        return None
    rows = [json.loads(s) for s in cases_path.read_text().splitlines()]
    require(len(rows) == 488, "diagnostic cohort changed")
    out = {}
    for name, arm in zip(NAMES, range(2)):
        by_k = {}
        for k in (2, 3):
            rr = [r for r in rows if r["true_K"] == k]
            pair_complete = 0
            trip_complete = 0
            pair_matches = []
            trip_matches = []
            for r in rr:
                expected = [n["frequency_hz"] for n in r["owned_notes"]]
                _, pair_f0, triplet_f0, _ = computed[int(r["row_id"])]
                pm = match_frequencies(expected, pair_f0[arm])[0]
                tm = match_frequencies(expected, triplet_f0[arm])[0]
                pair_matches.append(pm)
                trip_matches.append(tm)
                pair_complete += int(pm == min(k, 2))
                trip_complete += int(tm == k)
            by_k[str(k)] = {
                "rows": len(rr),
                "pair_complete": pair_complete,
                "triplet_complete": trip_complete,
                "mean_pair_matches": float(np.mean(pair_matches)),
                "mean_triplet_matches": float(np.mean(trip_matches)),
            }
        out[name] = by_k
    return out


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")
    inputs, records = action_inputs(a.exports)
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open("rb") as stream:
            require(hashlib.file_digest(stream, "md5").hexdigest() == expected,
                    "dataset changed: " + name)

    wanted = {member for member, _ in records.values()}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset)
              if t.annotation_member in wanted}
    require(set(tracks) == wanted, "missing tracks")
    by_member = {}
    for row, (member, start) in records.items():
        by_member.setdefault(member, []).append((row, start))

    computed, tables, started = {}, None, time.monotonic()
    for member in sorted(wanted):
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768.0
        for row, start in by_member[member]:
            freq, x = h.transition_spectrum(samples, start)
            if tables is None:
                tables = extraction_tables(freq)
            computed[row] = extract_row(freq, x, tables)
        print(json.dumps({"recording": member, "rows": len(computed),
                          "seconds": time.monotonic() - started}), flush=True)

    ids = np.array(sorted(computed))
    values = np.array([computed[int(i)][0] for i in ids])
    pairs = np.array([computed[int(i)][1] for i in ids])
    triplets = np.array([computed[int(i)][2] for i in ids])
    exclusivity = np.array([computed[int(i)][3] for i in ids])

    with np.load(a.capacity / "features.npz", allow_pickle=False) as previous:
        np.testing.assert_array_equal(ids, previous["row_id"])
        control_residual_error = float(
            np.max(np.abs(values[:, 0, :2] - previous["features"][:, 3, :2]))
        )
        control_frequency_error = float(max(
            np.max(np.abs(pairs[:, 0] - previous["pair_f0"][:, 3])),
            np.max(np.abs(triplets[:, 0] - previous["triplet_f0"][:, 3])),
        ))
        require(control_residual_error < 1e-10 and control_frequency_error == 0,
                "pool64_t2 control changed")

    a.output.mkdir(parents=True)
    np.savez_compressed(
        a.output / "features.npz",
        row_id=ids,
        features=values,
        pair_f0=pairs,
        triplet_f0=triplets,
        exclusivity=exclusivity,
        variants=np.asarray(NAMES),
    )

    reports, scores = [], {}
    for fold, arr in inputs.items():
        data = {}
        for split in ("fit", "val"):
            mask = arr[split + "_b_low"] & (arr[split + "_base_k"] == 3)
            valid = arr[split + "_valid"]
            row_ids = arr[split + "_ids"][mask][valid]
            data[split] = {
                "X": np.array([computed[int(i)][0] for i in row_ids]),
                "y": arr[split + "_true_k"][mask][valid],
                "fold": arr[split + "_fold"][mask][valid],
                "ids": row_ids,
            }
        fit, val = data["fit"], data["val"]
        selected, cv, inner = select_on_fit(fit["X"], fit["y"], fit["fold"])
        states = []
        val_probability = np.array([
            probability(fit["X"][:, i, :2], fit["y"], val["X"][:, i, :2], states)
            for i in range(len(NAMES))
        ])
        write_json(a.output / f"models-fold-{fold}.json", dict(zip(NAMES, states)))
        for split in ("fit", "val"):
            for key in ("ids", "y", "fold"):
                scores[f"fold_{fold}_{split}_{key}"] = data[split][key]
        scores[f"fold_{fold}_inner_probability"] = inner
        scores[f"fold_{fold}_val_probability"] = val_probability
        chosen_p = np.zeros(len(val["y"])) if selected is None else val_probability[selected]
        reports.append({
            "fold": fold,
            "selected": None if selected is None else NAMES[selected],
            "fit_cv": dict(zip(NAMES, cv)),
            "val_fixed": {
                name: accounting(val["y"], p)
                for name, p in zip(NAMES, val_probability)
            },
            "val_selected": accounting(val["y"], chosen_p),
        })

    np.savez_compressed(a.output / "scores.npz", **scores)

    def total(rows):
        return {k: sum(r[k] for r in rows) for k in
                ("rows", "applied", "corrections", "regressions",
                 "other_k_actions", "global_net")}

    report = {
        "status": "completed",
        "folds": reports,
        "variants": list(NAMES),
        "outer_fold_3_used": False,
        "annotation_frequencies_used_for_inference": False,
        "neural_training": False,
        "automatic_promotion": False,
        "unique_audio_rows": len(ids),
        "recordings": len(wanted),
        "top_n_nnls_combinations": TOP_N,
        "rerank_score": "min_internal_unique_attack_fraction / (normalized_nnls_cost + 1e-12)",
        "control_max_abs_residual_error": control_residual_error,
        "control_max_abs_frequency_error": control_frequency_error,
        "total_selected": total([r["val_selected"] for r in reports]),
        "total_fixed": {
            name: total([r["val_fixed"][name] for r in reports])
            for name in NAMES
        },
        "selection": "FIT-only rotation; net, fewer regressions, fewer actions, arm order; abstain if net <= 0",
        "diagnostic_frequency_matches": diagnostic_matches(a.cases, computed),
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
        "runtime": {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p) for p in
               ("numpy", "scipy", "scikit-learn")},
        },
        "limitations": [
            "Internal folds have been inspected previously; this is not untouched final validation.",
            "Exclusive support is computed only among each proposed combination, without annotations.",
            "Reranking is restricted to the 64 lowest-NNLS combinations by preregistered protocol.",
            "Normal residual-audio path only; no compressed-path conclusion.",
        ],
    }

    previous = json.loads((a.capacity / "report.json").read_text())
    require(report["total_fixed"]["control_pool64_t2"] ==
            previous["total_fixed"]["pool64_t2"],
            "control Exact-K accounting changed")

    write_json(a.output / "report.json", report)

    lines = [
        "# Exclusive-support reconstruction rerank",
        "",
        "| arm | corrections | regressions | other-K actions | net |",
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
        "Selections by outer internal fold: " +
        ", ".join(f"{r['fold']}={r['selected'] or 'abstain'}" for r in reports) + ".",
        "",
        "Fold 3 excluded. No automatic promotion. Normal-audio path only.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exports", "dataset", "capacity", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cases", type=Path)
    run(p.parse_args())
