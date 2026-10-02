"""Cluster pre-HPR Exact-K failures for true K=2,3,4.

This is a diagnostic only. Clustering never receives true_k, predicted_k, or
error direction. Those labels are used only after clustering to interpret the
result. Sources are the frozen fold-3 low-K and acoustic residual audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 27334

ACOUSTIC_FEATURES = (
    "rms_dbfs",
    "pre_rms_dbfs",
    "post_rms_dbfs",
    "late_rms_dbfs",
    "post_minus_pre_db",
    "peak_absolute",
    "spectral_flatness",
    "eligible_births",
    "contested_births",
    "max_simultaneous_notes",
    "native_positive_peak",
    "native_flux_peak",
)

CONTEXT_FEATURES = (
    "foreign_births_31",
    "foreign_births_added_tail",
    "own_births_added_tail",
    "carried_notes_at_start",
    "overlapping_notes_31",
    "offsets_31",
    "foreign_assigned_births_31",
    "unassigned_births_31",
    "retained_candidates",
    "full_candidates",
    "candidate_count_log",
    "group_width",
    "proposal_mean",
    "proposal_max",
    "router_mean",
    "local_count_mean_norm",
    "local_count_weighted_norm",
    "local_birth_max",
)

COUNT_LIKE = {
    "eligible_births",
    "contested_births",
    "max_simultaneous_notes",
    "foreign_births_31",
    "foreign_births_added_tail",
    "own_births_added_tail",
    "carried_notes_at_start",
    "overlapping_notes_31",
    "offsets_31",
    "foreign_assigned_births_31",
    "unassigned_births_31",
    "retained_candidates",
    "full_candidates",
}


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def safe_float(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y)
    if not finite.all():
        replacement = np.median(y[finite]) if finite.any() else 0.0
        y = np.where(finite, y, replacement)
    return y


def feature_matrix(acoustic: dict[str, np.ndarray], context: dict[str, np.ndarray]):
    cols, names = [], []
    for name in ACOUSTIC_FEATURES:
        if name not in acoustic:
            continue
        x = safe_float(acoustic[name])
        if name in COUNT_LIKE:
            x = np.log1p(np.maximum(x, 0))
        cols.append(x)
        names.append(name)
    for name in CONTEXT_FEATURES:
        if name not in context:
            continue
        x = safe_float(context[name])
        if name in COUNT_LIKE:
            x = np.log1p(np.maximum(x, 0))
        cols.append(x)
        names.append(name)
    X = np.column_stack(cols)
    variance = X.var(axis=0)
    keep = variance > 1e-12
    return X[:, keep], [n for n, k in zip(names, keep) if k]


def transition_counts(true_k: np.ndarray, pred: np.ndarray, rows: np.ndarray):
    c = Counter(f"{int(a)}→{int(b)}" for a, b in zip(true_k[rows], pred[rows]))
    return [{"transition": k, "rows": int(v)} for k, v in c.most_common()]


def weighted_correct_reference(
    Z: np.ndarray,
    true_k: np.ndarray,
    pred: np.ndarray,
    cluster_rows: np.ndarray,
) -> np.ndarray:
    values, counts = np.unique(true_k[cluster_rows], return_counts=True)
    weights = counts / counts.sum()
    ref = np.zeros(Z.shape[1], dtype=np.float64)
    used = 0.0
    for k, w in zip(values, weights):
        correct = (true_k == k) & (pred == true_k)
        if correct.any():
            ref += w * Z[correct].mean(axis=0)
            used += w
    if used == 0:
        return np.zeros(Z.shape[1])
    return ref / used


def consistent_features_by_k(
    Z: np.ndarray,
    true_k: np.ndarray,
    pred: np.ndarray,
    cluster_rows: np.ndarray,
    names: list[str],
):
    per_k = {}
    for k in (2, 3, 4):
        rows = cluster_rows[true_k[cluster_rows] == k]
        correct = np.flatnonzero((true_k == k) & (pred == k))
        if len(rows) < 5 or len(correct) < 5:
            continue
        per_k[k] = Z[rows].mean(axis=0) - Z[correct].mean(axis=0)
    if len(per_k) < 2:
        return []
    out = []
    mat = np.stack(list(per_k.values()))
    for j, name in enumerate(names):
        v = mat[:, j]
        same_sign = np.all(v > 0) or np.all(v < 0)
        if same_sign and np.min(np.abs(v)) >= 0.15:
            out.append(
                {
                    "feature": name,
                    "direction": "higher" if np.all(v > 0) else "lower",
                    "min_abs_delta_z": float(np.min(np.abs(v))),
                    "mean_delta_z": float(v.mean()),
                    "by_true_k": {
                        str(k): float(per_k[k][j]) for k in sorted(per_k)
                    },
                }
            )
    return sorted(out, key=lambda r: abs(r["mean_delta_z"]), reverse=True)[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-k", type=Path, required=True)
    ap.add_argument("--acoustic", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    context = load_npz(find_one(args.low_k, "row-context.npz"))
    acoustic = load_npz(find_one(args.acoustic, "rows.npz"))

    for key in ("global_index", "k"):
        if key not in context:
            raise RuntimeError(f"{key} missing from low-k source")
    for key in ("global_index", "k", "predicted"):
        if key not in acoustic:
            raise RuntimeError(f"{key} missing from acoustic source")

    np.testing.assert_array_equal(context["global_index"], acoustic["global_index"])
    np.testing.assert_array_equal(context["k"], acoustic["k"])

    true_k = acoustic["k"].astype(np.int32)
    pred = acoustic["predicted"].astype(np.int32)
    global_index = acoustic["global_index"].astype(np.int64)
    member = acoustic["member"].astype(str)
    start = acoustic["start"].astype(np.int64)

    X_all, names = feature_matrix(acoustic, context)
    if len(X_all) != len(true_k):
        raise RuntimeError("feature population mismatch")

    target = np.isin(true_k, (2, 3, 4))
    failures = target & (pred != true_k)
    correct = target & (pred == true_k)
    rows = np.flatnonzero(failures)
    if len(rows) < 100:
        raise RuntimeError(f"too few failures: {len(rows)}")

    scaler = StandardScaler()
    scaler.fit(X_all[target])
    Z_all = scaler.transform(X_all)
    Z = Z_all[rows]

    candidates = []
    best = None
    max_k = min(8, max(2, len(rows) // 80))
    for n_clusters in range(2, max_k + 1):
        model = KMeans(
            n_clusters=n_clusters,
            n_init=50,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(Z)
        counts = np.bincount(labels, minlength=n_clusters)
        score = float(silhouette_score(Z, labels))
        record = {
            "k": n_clusters,
            "silhouette": score,
            "min_cluster_rows": int(counts.min()),
            "max_cluster_rows": int(counts.max()),
        }
        candidates.append(record)
        admissible = counts.min() >= max(15, int(0.02 * len(rows)))
        if admissible and (best is None or score > best[0]):
            best = (score, model, labels)
    if best is None:
        # KMeans itself cannot create empty clusters here; use best silhouette
        n = max(candidates, key=lambda x: x["silhouette"])["k"]
        model = KMeans(n_clusters=n, n_init=50, random_state=RANDOM_STATE)
        labels = model.fit_predict(Z)
        best = (float(silhouette_score(Z, labels)), model, labels)

    _, model, labels = best
    n_clusters = model.n_clusters

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    xy = pca.fit_transform(Z)

    clusters = []
    for cid in range(n_clusters):
        local = np.flatnonzero(labels == cid)
        cr = rows[local]
        ref = weighted_correct_reference(Z_all, true_k, pred, cr)
        mean_z = Z_all[cr].mean(axis=0)
        delta = mean_z - ref
        order = np.argsort(-np.abs(delta))
        top = [
            {
                "feature": names[j],
                "delta_vs_sameK_correct_z": float(delta[j]),
                "cluster_mean_z": float(mean_z[j]),
            }
            for j in order[:8]
        ]

        k_counts = Counter(map(int, true_k[cr]))
        direction = Counter(
            "under" if pred[i] < true_k[i] else "over" for i in cr
        )
        clusters.append(
            {
                "cluster": int(cid),
                "rows": int(len(cr)),
                "share_of_failures": float(len(cr) / len(rows)),
                "true_k": {str(k): int(v) for k, v in sorted(k_counts.items())},
                "error_direction": dict(direction),
                "transitions": transition_counts(true_k, pred, cr)[:8],
                "top_signatures_vs_sameK_correct": top,
                "features_consistent_across_true_k": consistent_features_by_k(
                    Z_all, true_k, pred, cr, names
                ),
            }
        )

    # Global common failure signature, controlled for the K2/K3/K4 composition.
    ref_all = weighted_correct_reference(Z_all, true_k, pred, rows)
    delta_all = Z_all[rows].mean(axis=0) - ref_all
    order = np.argsort(-np.abs(delta_all))
    global_signature = [
        {"feature": names[j], "delta_vs_sameK_correct_z": float(delta_all[j])}
        for j in order[:12]
    ]

    result = {
        "status": "completed",
        "scope": "pre-HPR frames-31-uniform fold-3 Exact-K failures, true K in {2,3,4}",
        "clustering_inputs_exclude_labels": True,
        "label_fields_used_only_for_interpretation": ["true_k", "predicted_k", "error_direction"],
        "rows_total": int(len(true_k)),
        "target_rows_k234": int(target.sum()),
        "correct_rows_k234": int(correct.sum()),
        "failure_rows_k234": int(failures.sum()),
        "features": names,
        "candidate_cluster_counts": candidates,
        "selected_clusters": int(n_clusters),
        "selected_silhouette": float(silhouette_score(Z, labels)),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "global_failure_signature_vs_sameK_correct": global_signature,
        "clusters": clusters,
        "limitations": [
            "One development fold and one seed; this is descriptive, not independent validation.",
            "Clusters use audit-level aggregate/acoustic features, not source-separated note energies.",
            "KMeans imposes approximately spherical clusters after standardization.",
            "Cluster membership is not a correction rule and must not be used to tune the frozen fold.",
        ],
    }

    (args.output / "v273-failure-clustering.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    # Row assignments for reproducibility and case inspection.
    import csv
    with (args.output / "v273-failure-clustering-rows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["global_index", "member", "start_sample", "true_k", "predicted_k",
             "error_delta", "cluster", "pca1", "pca2"]
        )
        for pos, row in enumerate(rows):
            w.writerow(
                [
                    int(global_index[row]),
                    member[row],
                    int(start[row]),
                    int(true_k[row]),
                    int(pred[row]),
                    int(pred[row] - true_k[row]),
                    int(labels[pos]),
                    float(xy[pos, 0]),
                    float(xy[pos, 1]),
                ]
            )

    lines = [
        "# Clustering des défaillances Exact-K K2/K3/K4 — avant HPR",
        "",
        f"Population : **{len(rows)} erreurs** parmi {int(target.sum())} lignes vraies K2/K3/K4.",
        f"KMeans sélectionne **{n_clusters} clusters** (silhouette **{result['selected_silhouette']:.4f}**).",
        "Les labels K et la direction d'erreur n'entrent pas dans le clustering ; ils servent uniquement à l'interprétation.",
        "",
        "## Signature commune de toutes les erreurs vs contrôles corrects du même K",
        "",
        "| Variable | Écart standardisé |",
        "|---|---:|",
    ]
    for rec in global_signature[:8]:
        lines.append(f"| {rec['feature']} | {rec['delta_vs_sameK_correct_z']:+.3f} |")

    lines += ["", "## Clusters", ""]
    for c in clusters:
        lines += [
            f"### Cluster {c['cluster']} — {c['rows']} lignes ({100*c['share_of_failures']:.1f} %)",
            "",
            f"- vrais K : {c['true_k']}",
            f"- direction : {c['error_direction']}",
            "- transitions dominantes : " + ", ".join(
                f"{x['transition']} ({x['rows']})" for x in c["transitions"][:5]
            ),
            "- signatures principales : " + ", ".join(
                f"{x['feature']} {x['delta_vs_sameK_correct_z']:+.2f}σ"
                for x in c["top_signatures_vs_sameK_correct"][:6]
            ),
        ]
        common = c["features_consistent_across_true_k"]
        if common:
            lines.append(
                "- communes à plusieurs vrais K : "
                + ", ".join(
                    f"{x['feature']} ({x['direction']}, {x['mean_delta_z']:+.2f}σ)"
                    for x in common[:5]
                )
            )
        lines.append("")

    lines += [
        "## Portée",
        "",
        "Ce clustering est un diagnostic descriptif sur le fold de développement figé. "
        "Il ne modifie aucune prédiction et ne constitue pas un correcteur.",
    ]
    (args.output / "v273-failure-clustering.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
