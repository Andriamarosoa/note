"""Per-neuron frozen audit of v240_cardinality_hidden2 kernel.

No training and no architecture change.

Starting from the published learned-gate checkpoint, replace ONE output column
(neuron) at a time in v240_cardinality_hidden2.kernel with the corresponding
column from the matched always-on checkpoint. Measure causal decision changes on
the frozen outer fold.

Also evaluate cumulative swaps chosen ONLY by weight-delta norm
(top 1/2/4/8/16/32/64/96 neurons), so cumulative ordering does not use outcome
labels or validation accuracy.

This is an attribution audit, not model selection.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import load_bundle, batch_inputs, require

SEED = 16061 + 1003
FRAMES = 31
H2 = "v240_cardinality_hidden2"

def nested(model):
    import tensorflow as tf
    x = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    require(len(x) == 1, "nested base mismatch")
    return x[0]

def apply_gate(x, g):
    cand = np.asarray(x["candidate_set"], np.float32).copy()
    stats = np.asarray(x["cluster_stats"], np.float32).copy()
    gg = np.asarray(g, np.float32)
    if gg.ndim == 0:
        gg = np.full(len(cand), float(gg), np.float32)
    cand[:, :, V88_FEATURE_DIM:V88_FEATURE_DIM + 5] *= gg[:, None, None]
    stats[:, 4:8] *= gg[:, None]
    return {
        "candidate_set": cand,
        "candidate_mask": np.asarray(x["candidate_mask"], np.float32),
        "cluster_stats": stats,
        "spectral_map": np.asarray(x["spectral_map"], np.float32),
    }

def forward(base, cache, outer, g, batch=256):
    out = []
    g = np.asarray(g)
    for s in range(0, len(outer), batch):
        ids = outer[s:s + batch]
        x = batch_inputs(cache, ids, FRAMES)
        gg = g[s:s + len(ids)] if g.ndim else g
        out.append(np.asarray(base(apply_gate(x, gg), training=False), np.float32))
    return np.concatenate(out)

def load_A(path, idx):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        c = int(r["cluster"]); y = int(r["true_k"]); p = int(r["predicted_k"])
        d = by.setdefault(c, {"n": 0, "u": 0})
        d["n"] += 1; d["u"] += p < y
    cid = max(by, key=lambda c: (by[c]["u"] / by[c]["n"], by[c]["n"]))
    ids = {int(r["global_index"]) for r in rows if int(r["cluster"]) == cid}
    return cid, np.isin(np.asarray(idx, np.int64), list(ids))

def metric(k, p, sel):
    y = k[sel]; q = p[sel]
    return {
        "rows": int(len(y)),
        "correct": int(np.sum(q == y)),
        "exact": float(np.mean(q == y)) if len(y) else None,
        "under": int(np.sum(q < y)),
        "over": int(np.sum(q > y)),
    }

def summarize(k, pred, baseline, reg48, learned_correct293, scopes):
    return {
        "metrics": {name: metric(k, pred, sel) for name, sel in scopes.items()},
        "original_48_recovered": int(np.sum(reg48 & (pred == 3))),
        "learned_correct_293_broken": int(np.sum(learned_correct293 & (pred != 3))),
        "k3_net_correct_vs_learned": int(
            np.sum((k == 3) & (pred == 3)) - np.sum((k == 3) & (baseline == 3))
        ),
        "changed_predictions": int(np.sum(pred != baseline)),
    }

def main():
    ap = argparse.ArgumentParser()
    for n in ("bundle", "config", "always-weights", "learned-weights",
              "saved-predictions", "cluster-rows", "output"):
        ap.add_argument("--" + n, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    cache, parts, _ = load_bundle(a.bundle, a.config)
    outer = np.asarray(parts["outer"], np.int64)
    k = np.minimum(cache["exact"][outer].astype(np.int32), 6)

    with np.load(a.saved_predictions, allow_pickle=False) as z:
        saved = {x: np.asarray(z[x]) for x in z.files}
    np.testing.assert_array_equal(saved["global_index"], outer)
    np.testing.assert_array_equal(saved["k"], k)
    gate = np.asarray(saved["learned_gate_gate"], np.float32)
    always_pred = saved["always_on_predicted"].astype(np.int32)
    learned_pred = saved["learned_gate_predicted"].astype(np.int32)

    mc = build_model("always_on", SEED); mc.load_weights(a.always_weights)
    ml = build_model("learned_gate", SEED); ml.load_weights(a.learned_weights)
    bc = nested(mc); bl = nested(ml)

    wc = [np.asarray(x).copy() for x in bc.get_layer(H2).get_weights()]
    wl = [np.asarray(x).copy() for x in bl.get_layer(H2).get_weights()]
    require(len(wc) == 2 and len(wl) == 2, "expected Dense kernel+bias")
    kc, bc_bias = wc
    kl, bl_bias = wl
    require(kc.shape == kl.shape == (192, 96), f"unexpected h2 kernel {kl.shape}")

    baseline_prob = forward(bl, cache, outer, gate)
    baseline = baseline_prob.argmax(1).astype(np.int32)
    np.testing.assert_array_equal(baseline, learned_pred)

    cid, A = load_A(a.cluster_rows, outer)
    scopes = {
        "global": np.ones(len(k), bool),
        "K2": k == 2,
        "K3": k == 3,
        "K4": k == 4,
        "cluster_A": A,
    }
    reg48 = (k == 3) & (always_pred == 3) & (learned_pred != 3)
    learned_correct293 = (k == 3) & (learned_pred == 3)
    require(int(reg48.sum()) == 48, "K3 regression population drift")
    require(int(learned_correct293.sum()) == 293, "learned correct K3 drift")

    delta = kc - kl
    col_norm = np.linalg.norm(delta, axis=0)
    col_rel = col_norm / (np.linalg.norm(kl, axis=0) + 1e-12)
    order = np.argsort(-col_norm)

    result = {
        "status": "completed",
        "training": False,
        "selection": "none",
        "cluster_A_id": int(cid),
        "layer": H2,
        "kernel_shape": list(kl.shape),
        "baseline": summarize(k, baseline, baseline, reg48, learned_correct293, scopes),
        "single_neuron": [],
        "cumulative_by_weight_delta_norm": {},
        "limitations": [
            "Single-column swaps are frozen hybrid interventions; interactions between neurons can be non-additive.",
            "Ranking by single-neuron outcome is descriptive only and is not used to construct the cumulative interventions.",
            "Cumulative interventions are ordered only by control-vs-learned weight-delta norm, independent of labels/outcomes.",
        ],
    }

    # Single-neuron causal interventions.
    for j in range(kl.shape[1]):
        kk = kl.copy()
        kk[:, j] = kc[:, j]
        bl.get_layer(H2).set_weights([kk, bl_bias])
        pred = forward(bl, cache, outer, gate).argmax(1).astype(np.int32)
        row = {
            "neuron": int(j),
            "weight_delta_norm": float(col_norm[j]),
            "weight_delta_relative": float(col_rel[j]),
            **summarize(k, pred, baseline, reg48, learned_correct293, scopes),
        }
        result["single_neuron"].append(row)

    # Restore.
    bl.get_layer(H2).set_weights([kl, bl_bias])

    # Cumulative swaps ordered only by parameter delta magnitude.
    for n in (1, 2, 4, 8, 16, 32, 64, 96):
        chosen = order[:n]
        kk = kl.copy()
        kk[:, chosen] = kc[:, chosen]
        bl.get_layer(H2).set_weights([kk, bl_bias])
        pred = forward(bl, cache, outer, gate).argmax(1).astype(np.int32)
        result["cumulative_by_weight_delta_norm"][str(n)] = {
            "neurons": [int(x) for x in chosen],
            **summarize(k, pred, baseline, reg48, learned_correct293, scopes),
        }

    bl.get_layer(H2).set_weights([kl, bl_bias])

    # Descriptive rankings only.
    by_recovery = sorted(
        result["single_neuron"],
        key=lambda x: (x["original_48_recovered"] - x["learned_correct_293_broken"],
                       x["original_48_recovered"],
                       -x["learned_correct_293_broken"]),
        reverse=True,
    )
    result["descriptive_top_neurons_by_k3_net"] = [
        {
            "neuron": x["neuron"],
            "recovered48": x["original_48_recovered"],
            "broken293": x["learned_correct_293_broken"],
            "k3_net": x["k3_net_correct_vs_learned"],
            "K2_exact": x["metrics"]["K2"]["exact"],
            "K3_exact": x["metrics"]["K3"]["exact"],
            "K4_exact": x["metrics"]["K4"]["exact"],
            "A_exact": x["metrics"]["cluster_A"]["exact"],
            "weight_delta_norm": x["weight_delta_norm"],
        }
        for x in by_recovery[:20]
    ]

    (a.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Audit par neurone de v240_cardinality_hidden2.kernel",
        "",
        "Aucun entrainement. Une seule colonne du kernel est remplacee a la fois.",
        "",
        f"- Kernel: {kl.shape[0]} x {kl.shape[1]}",
        f"- Population K3 regression: {int(reg48.sum())}",
        f"- K3 correct learned: {int(learned_correct293.sum())}",
        "",
        "## Top 12 interventions unitaires — descriptif",
        "",
        "| Neurone | recup 48 | casse 293 | net K3 | K2 | K3 | K4 | A | ||delta W|| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in by_recovery[:12]:
        m = x["metrics"]
        lines.append(
            f"| {x['neuron']} | {x['original_48_recovered']} | {x['learned_correct_293_broken']} | "
            f"{x['k3_net_correct_vs_learned']:+d} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% | {x['weight_delta_norm']:.4f} |"
        )

    lines += [
        "",
        "## Cumulatif par norme de changement de poids uniquement",
        "",
        "| N neurones | recup 48 | casse 293 | net K3 | K2 | K3 | K4 | A |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (1, 2, 4, 8, 16, 32, 64, 96):
        x = result["cumulative_by_weight_delta_norm"][str(n)]
        m = x["metrics"]
        lines.append(
            f"| {n} | {x['original_48_recovered']} | {x['learned_correct_293_broken']} | "
            f"{x['k3_net_correct_vs_learned']:+d} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% |"
        )

    lines += [
        "",
        "Le classement unitaire par resultat est descriptif uniquement; aucune architecture ni poids n'est promu.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
