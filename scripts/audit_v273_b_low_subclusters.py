"""Unsupervised subclustering audit of B_low + base-K3 action rows.

Clusters are formed on FIT structural/inference features only. True K is used
only after clustering to choose FIT correction clusters and to evaluate VAL.
Fold 3 is never loaded.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.v273_residual_audit import FOLDS, require, sha256_file, verify_export, write_json

PROTOCOL = Path("analysis/v273-b-low-subclustering-protocol.md")
K_GRID = (2, 3, 4, 5, 6)
MIN_CLUSTER_FRACTION = 0.05
PCA_MAX_COMPONENTS = 16
SEED_BASE = 39170
STABILITY_RUNS = 8


def action_view(arr, split):
    action = arr[split + "_b_low"] & (arr[split + "_base_k"] == 3)
    X = np.asarray(arr[split + "_structural"][action], np.float64)
    y = np.asarray(arr[split + "_true_k"][action], np.int32)
    ids = np.asarray(arr[split + "_ids"][action], np.int64)
    fold = np.asarray(arr[split + "_fold"][action], np.int32)
    recording = np.asarray(arr[split + "_recording"][action]).astype(str)
    require(len(X) == len(y) == len(ids) == len(fold) == len(recording),
            "action alignment mismatch")
    require(X.ndim == 2 and X.shape[1] == 37, "structural feature schema changed")
    require(np.isfinite(X).all(), "nonfinite structural feature")
    return {"X": X, "y": y, "ids": ids, "fold": fold, "recording": recording}


def fit_projection(X):
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    full = PCA(svd_solver="full", random_state=SEED_BASE)
    full.fit(Z)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    needed = int(np.searchsorted(cumulative, 0.90) + 1)
    n_components = max(2, min(PCA_MAX_COMPONENTS, needed, Z.shape[1], len(Z) - 1))
    pca = PCA(n_components=n_components, svd_solver="full", random_state=SEED_BASE)
    P = pca.fit_transform(Z)
    return scaler, pca, P, {
        "components_for_90pct_uncapped": needed,
        "components_used": n_components,
        "explained_variance_used": float(pca.explained_variance_ratio_.sum()),
        "capped_at_16": bool(needed > PCA_MAX_COMPONENTS),
    }


def cluster_quality(P, k, seed):
    km = KMeans(n_clusters=k, n_init=100, random_state=seed)
    labels = km.fit_predict(P)
    counts = np.bincount(labels, minlength=k)
    fractions = counts / len(labels)
    admissible = bool(np.min(fractions) >= MIN_CLUSTER_FRACTION)
    sil = float(silhouette_score(P, labels)) if len(np.unique(labels)) > 1 else -1.0

    stability = []
    for offset in range(STABILITY_RUNS):
        alt = KMeans(n_clusters=k, n_init=20, random_state=seed + 100 + offset)
        alt_labels = alt.fit_predict(P)
        stability.append(float(adjusted_rand_score(labels, alt_labels)))
    return km, labels, {
        "k": k,
        "admissible": admissible,
        "silhouette": sil,
        "cluster_counts": counts.astype(int).tolist(),
        "cluster_fractions": fractions.tolist(),
        "stability_ari_mean": float(np.mean(stability)),
        "stability_ari_min": float(np.min(stability)),
        "stability_ari_runs": stability,
    }


def select_k(P, seed):
    candidates = []
    models = {}
    labels_by_k = {}
    for k in K_GRID:
        require(len(P) > k, "too few FIT rows for K grid")
        km, labels, quality = cluster_quality(P, k, seed + k)
        candidates.append(quality)
        models[k] = km
        labels_by_k[k] = labels
    admissible = [q for q in candidates if q["admissible"]]
    require(admissible, "no admissible K")
    chosen = max(admissible, key=lambda q: (q["silhouette"], -q["k"]))
    k = int(chosen["k"])
    return k, models[k], labels_by_k[k], candidates


def counts_for_cluster(y, labels, cluster):
    m = labels == cluster
    return {
        "rows": int(m.sum()),
        "K2": int(np.sum(m & (y == 2))),
        "K3": int(np.sum(m & (y == 3))),
        "other_K": int(np.sum(m & ~np.isin(y, (2, 3)))),
        "fit_or_val_net_if_3to2": int(np.sum(m & (y == 2)) - np.sum(m & (y == 3))),
        "k2_fraction_among_k23": (
            float(np.sum(m & (y == 2)) / max(1, np.sum(m & np.isin(y, (2, 3)))))
        ),
    }


def policy_from_fit(y_fit, labels_fit, k):
    active = []
    table = {}
    for cluster in range(k):
        entry = counts_for_cluster(y_fit, labels_fit, cluster)
        entry["active_3to2"] = entry["fit_or_val_net_if_3to2"] > 0
        if entry["active_3to2"]:
            active.append(cluster)
        table[str(cluster)] = entry
    return active, table


def evaluate_policy(y, labels, active, k):
    apply = np.isin(labels, np.asarray(active, dtype=int)) if active else np.zeros(len(y), bool)
    corrections = int(np.sum(apply & (y == 2)))
    regressions = int(np.sum(apply & (y == 3)))
    table = {}
    oracle = 0
    for cluster in range(k):
        entry = counts_for_cluster(y, labels, cluster)
        oracle += max(entry["fit_or_val_net_if_3to2"], 0)
        entry["selected_by_fit"] = cluster in active
        table[str(cluster)] = entry
    return {
        "rows": len(y),
        "applied": int(apply.sum()),
        "corrections": corrections,
        "regressions": regressions,
        "other_k_actions": int(np.sum(apply & ~np.isin(y, (2, 3)))),
        "global_net": corrections - regressions,
        "oracle_val_cluster_net": int(oracle),
        "all_rows_3to2_net": int(np.sum(y == 2) - np.sum(y == 3)),
        "clusters": table,
    }


def centroid_block_summary(centroids_scaled, pca, scaler):
    # Return centroids back in original 37-dimensional structural space and
    # summarize the known feature blocks rather than naming unknown stats.
    original = scaler.inverse_transform(pca.inverse_transform(centroids_scaled))
    blocks = {}
    for i, row in enumerate(original):
        blocks[str(i)] = {
            "stats_0_7_mean": float(np.mean(row[0:8])),
            "candidate_count": float(row[8]),
            "v88_tail_mean_8_mean": float(np.mean(row[9:17])),
            "v88_tail_max_8_mean": float(np.mean(row[17:25])),
            "spectral_summary_5_mean": float(np.mean(row[25:30])),
            "base_probability_argmax": int(np.argmax(row[30:37])),
            "base_probability_k2": float(row[32]),
            "base_probability_k3": float(row[33]),
        }
    return blocks


def run_fold(root, outer_fold):
    verify_export(root)
    with np.load(root / "replay.npz", allow_pickle=False) as z:
        arr = dict(z)

    fit = action_view(arr, "fit")
    val = action_view(arr, "val")
    require(set(np.unique(fit["fold"])) <= set(FOLDS) - {outer_fold}, "FIT fold leak")
    require(np.all(val["fold"] == outer_fold), "wrong VAL fold")
    require(not np.intersect1d(fit["ids"], val["ids"]).size, "FIT/VAL row overlap")
    require(not np.intersect1d(fit["recording"], val["recording"]).size, "FIT/VAL recording overlap")

    scaler, pca, Pf, projection = fit_projection(fit["X"])
    Pv = pca.transform(scaler.transform(val["X"]))

    chosen_k, km, lf, candidates = select_k(Pf, SEED_BASE + outer_fold * 1000)
    lv = km.predict(Pv)

    active, fit_clusters = policy_from_fit(fit["y"], lf, chosen_k)
    val_result = evaluate_policy(val["y"], lv, active, chosen_k)

    return {
        "fold": outer_fold,
        "fit_rows": len(fit["y"]),
        "val_rows": len(val["y"]),
        "projection": projection,
        "k_candidates": candidates,
        "selected_k": chosen_k,
        "active_clusters_from_fit": active,
        "fit_clusters": fit_clusters,
        "val_policy": val_result,
        "centroid_blocks": centroid_block_summary(km.cluster_centers_, pca, scaler),
    }


def sum_metrics(rows, key):
    fields = ("rows", "applied", "corrections", "regressions", "other_k_actions",
              "global_net", "oracle_val_cluster_net", "all_rows_3to2_net")
    return {f: int(sum(r[key][f] for r in rows)) for f in fields}


def run(a):
    require(not a.output.exists(), "refusing overwrite")
    require(PROTOCOL.exists(), "missing preregistered protocol")
    reports = []
    for fold in FOLDS:
        reports.append(run_fold(a.exports / f"fold-{fold}", fold))

    selected_policy = sum_metrics(reports, "val_policy")
    selected_ks = [r["selected_k"] for r in reports]

    report = {
        "status": "completed",
        "experiment": "v273_B_low_baseK3_subclustering",
        "folds": reports,
        "validation_folds": list(FOLDS),
        "outer_fold_3_used": False,
        "cluster_labels_used_for_geometry": False,
        "true_k_used_for_cluster_geometry": False,
        "cluster_features": "37 exported inference-time structural features only",
        "k_grid": list(K_GRID),
        "k_selection": "FIT-only silhouette; reject any K with cluster <5%; tie -> smaller K",
        "stability": f"{STABILITY_RUNS} additional KMeans initializations, ARI diagnostic only",
        "policy": "activate 3->2 for FIT clusters with #K2-#K3 > 0; transfer unchanged to VAL",
        "selected_k_by_fold": selected_ks,
        "selected_policy_total": selected_policy,
        "oracle_val_total": int(sum(r["val_policy"]["oracle_val_cluster_net"] for r in reports)),
        "all_rows_3to2_total_net": int(sum(r["val_policy"]["all_rows_3to2_net"] for r in reports)),
        "automatic_promotion": False,
        "source_sha256": {
            "protocol": sha256_file(PROTOCOL),
            "script": sha256_file(__file__),
            **{f"manifest_fold_{f}": sha256_file(a.exports / f"fold-{f}" / "manifest.json")
               for f in FOLDS},
        },
        "runtime": {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p)
               for p in ("numpy", "scipy", "scikit-learn")},
        },
        "limitations": [
            "Subclusters are internal and fold-specific; cluster numeric IDs are not matched across rotations.",
            "True K selects which already-formed FIT clusters receive 3->2, so the correction policy is supervised after unsupervised geometry.",
            "Oracle VAL is diagnostic only and cannot be used for model selection.",
            "Internal folds were already inspected in earlier research; this is not untouched final validation.",
        ],
    }

    a.output.mkdir(parents=True)
    write_json(a.output / "report.json", report)

    lines = [
        "# B_low + base-K3 subclustering audit",
        "",
        "| VAL fold | selected K | silhouette | stability ARI | active FIT clusters | corrections | regressions | net | oracle cluster net | all-rows net |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        chosen = next(q for q in r["k_candidates"] if q["k"] == r["selected_k"])
        q = r["val_policy"]
        lines.append(
            f"| {r['fold']} | {r['selected_k']} | {chosen['silhouette']:.3f} | "
            f"{chosen['stability_ari_mean']:.3f} | "
            f"{','.join(map(str, r['active_clusters_from_fit'])) or 'none'} | "
            f"{q['corrections']} | {q['regressions']} | {q['global_net']:+d} | "
            f"{q['oracle_val_cluster_net']:+d} | {q['all_rows_3to2_net']:+d} |"
        )
    q = selected_policy
    lines += [
        "",
        f"FIT-selected cluster policy total: **{q['corrections']} corrections / "
        f"{q['regressions']} regressions = net {q['global_net']:+d}**.",
        f"Fixed-partition VAL oracle total: **{report['oracle_val_total']:+d}**.",
        f"Correct-all-B_low/baseK3 reference net: **{report['all_rows_3to2_total_net']:+d}**.",
        "",
        "Fold 3 excluded. No automatic promotion.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
