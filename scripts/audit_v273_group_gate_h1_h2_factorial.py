"""Frozen 2x2x2 attribution audit for hidden2 activation formation.

No training. No architecture change.

For each hidden2 neuron j, its activation is:
    h2_j = ReLU(h1 @ W2[:, j] + b2[j])

This audit separates three factors:
  H: hidden1 representation source (learned-gate vs always-on)
  W: hidden2 incoming kernel-column source (learned-gate vs always-on)
  B: hidden2 bias source (learned-gate vs always-on)

The downstream final cardinality output layer is held at the learned-gate
checkpoint for all attribution interventions. Thus changes in Exact-K decisions
are caused only by the reconstructed hidden2 activation(s).

Global 2x2x2: all 96 hidden2 units reconstructed from each H/W/B combination.
Per-neuron 2x2x2: only one hidden2 unit is reconstructed/substituted while the
other 95 stay at their learned-gate activations.

Frozen attribution only; no selection or promotion.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import batch_inputs, load_bundle, require

SEED = 16061 + 1003
FRAMES = 31
H1 = "v240_cardinality_hidden1"
H2 = "v240_cardinality_hidden2"
OUT = "cardinality"

def nested(model):
    import tensorflow as tf
    xs = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    require(len(xs) == 1, "nested base mismatch")
    return xs[0]

def apply_gate(x, gate):
    cand = np.asarray(x["candidate_set"], np.float32).copy()
    stats = np.asarray(x["cluster_stats"], np.float32).copy()
    g = np.asarray(gate, np.float32)
    if g.ndim == 0:
        g = np.full(len(cand), float(g), np.float32)
    cand[:, :, V88_FEATURE_DIM:V88_FEATURE_DIM + 5] *= g[:, None, None]
    stats[:, 4:8] *= g[:, None]
    return {
        "candidate_set": cand,
        "candidate_mask": np.asarray(x["candidate_mask"], np.float32),
        "cluster_stats": stats,
        "spectral_map": np.asarray(x["spectral_map"], np.float32),
    }

def collect_h1_h2_probs(base, cache, outer, gate, batch_size=256):
    import tensorflow as tf
    probe = tf.keras.Model(
        base.inputs,
        [base.get_layer(H1).output, base.get_layer(H2).output, base.output],
    )
    h1s, h2s, probs = [], [], []
    gate_arr = np.asarray(gate)
    for start in range(0, len(outer), batch_size):
        ids = outer[start:start + batch_size]
        x = batch_inputs(cache, ids, FRAMES)
        g = gate if gate_arr.ndim == 0 else gate_arr[start:start + len(ids)]
        xx = apply_gate(x, g)
        a, b, p = probe(xx, training=False)
        h1s.append(np.asarray(a, np.float32))
        h2s.append(np.asarray(b, np.float32))
        probs.append(np.asarray(p, np.float32))
    return np.concatenate(h1s), np.concatenate(h2s), np.concatenate(probs)

def relu(x):
    return np.maximum(np.asarray(x, np.float32), 0.0)

def metric(k, pred, sel):
    y = k[sel]
    p = pred[sel]
    return {
        "rows": int(len(y)),
        "correct": int(np.sum(p == y)),
        "exact": float(np.mean(p == y)) if len(y) else None,
        "under": int(np.sum(p < y)),
        "over": int(np.sum(p > y)),
    }

def load_cluster_a(path, global_index):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        c = int(r["cluster"]); y = int(r["true_k"]); p = int(r["predicted_k"])
        d = by.setdefault(c, {"n": 0, "under": 0})
        d["n"] += 1
        d["under"] += p < y
    cid = max(by, key=lambda c: (by[c]["under"] / by[c]["n"], by[c]["n"]))
    ids = {int(r["global_index"]) for r in rows if int(r["cluster"]) == cid}
    return cid, np.isin(np.asarray(global_index, np.int64), list(ids))

def summarize(k, pred, learned_pred, always_pred, scopes):
    k3 = k == 3
    reg48 = k3 & (always_pred == 3) & (learned_pred != 3)
    corr29 = k3 & (always_pred != 3) & (learned_pred == 3)
    learned_correct = k3 & (learned_pred == 3)
    return {
        "metrics": {name: metric(k, pred, sel) for name, sel in scopes.items()},
        "original_48_recovered": int(np.sum(reg48 & (pred == 3))),
        "original_29_corrections_lost": int(np.sum(corr29 & (pred != 3))),
        "learned_correct_293_broken": int(np.sum(learned_correct & (pred != 3))),
        "k3_net_correct_vs_learned": int(
            np.sum(k3 & (pred == 3)) - np.sum(k3 & (learned_pred == 3))
        ),
        "changed_predictions_vs_learned": int(np.sum(pred != learned_pred)),
    }

def logits_to_pred(h2, out_w, out_b):
    return (np.asarray(h2, np.float32) @ out_w + out_b).argmax(1).astype(np.int32)

def source_name(flag):
    return "control" if flag else "learned"

def main():
    ap = argparse.ArgumentParser()
    for name in (
        "bundle", "config", "always-weights", "learned-weights",
        "saved-predictions", "cluster-rows", "output",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    cache, parts, _ = load_bundle(a.bundle, a.config)
    outer = np.asarray(parts["outer"], np.int64)
    k = np.minimum(cache["exact"][outer].astype(np.int32), 6)

    with np.load(a.saved_predictions, allow_pickle=False) as z:
        saved = {key: np.asarray(z[key]) for key in z.files}
    np.testing.assert_array_equal(saved["global_index"], outer)
    np.testing.assert_array_equal(saved["k"], k)
    learned_pred = saved["learned_gate_predicted"].astype(np.int32)
    always_pred = saved["always_on_predicted"].astype(np.int32)
    gate = np.asarray(saved["learned_gate_gate"], np.float32)

    mc = build_model("always_on", SEED); mc.load_weights(a.always_weights)
    ml = build_model("learned_gate", SEED); ml.load_weights(a.learned_weights)
    bc = nested(mc); bl = nested(ml)

    h1_c, h2_c, p_c = collect_h1_h2_probs(bc, cache, outer, 1.0)
    h1_l, h2_l, p_l = collect_h1_h2_probs(bl, cache, outer, gate)
    np.testing.assert_array_equal(p_c.argmax(1), always_pred)
    np.testing.assert_array_equal(p_l.argmax(1), learned_pred)

    w2_c, b2_c = [np.asarray(x, np.float32) for x in bc.get_layer(H2).get_weights()]
    w2_l, b2_l = [np.asarray(x, np.float32) for x in bl.get_layer(H2).get_weights()]
    out_w_l, out_b_l = [np.asarray(x, np.float32) for x in bl.get_layer(OUT).get_weights()]

    require(w2_l.shape == w2_c.shape == (192, 96), f"unexpected W2 {w2_l.shape}")
    require(b2_l.shape == b2_c.shape == (96,), f"unexpected b2 {b2_l.shape}")
    require(h1_l.shape == h1_c.shape == (len(outer), 192), f"unexpected h1 {h1_l.shape}")
    require(h2_l.shape == h2_c.shape == (len(outer), 96), f"unexpected h2 {h2_l.shape}")

    # Exact algebraic replay of hidden2.
    h2_l_replay = relu(h1_l @ w2_l + b2_l)
    h2_c_replay = relu(h1_c @ w2_c + b2_c)
    max_h2_l = float(np.max(np.abs(h2_l_replay - h2_l)))
    max_h2_c = float(np.max(np.abs(h2_c_replay - h2_c)))
    require(max_h2_l < 1e-5, f"learned H2 replay mismatch {max_h2_l}")
    require(max_h2_c < 1e-5, f"control H2 replay mismatch {max_h2_c}")

    pred_l_replay = logits_to_pred(h2_l, out_w_l, out_b_l)
    np.testing.assert_array_equal(pred_l_replay, learned_pred)

    cid, cluster_a = load_cluster_a(a.cluster_rows, outer)
    scopes = {
        "global": np.ones(len(k), bool),
        "K2": k == 2,
        "K3": k == 3,
        "K4": k == 4,
        "cluster_A": cluster_a,
    }
    reg48 = (k == 3) & (always_pred == 3) & (learned_pred != 3)
    corr29 = (k == 3) & (always_pred != 3) & (learned_pred == 3)
    require(int(reg48.sum()) == 48, "48 regression population drift")
    require(int(corr29.sum()) == 29, "29 correction population drift")

    result = {
        "status": "completed",
        "training": False,
        "cluster_A_id": int(cid),
        "population": {"k3_regressions": 48, "k3_corrections": 29},
        "replay": {
            "learned_h2_max_abs": max_h2_l,
            "control_h2_max_abs": max_h2_c,
        },
        "global_factorial_2x2x2": {},
        "per_neuron_factorial_2x2x2": [],
        "limitations": [
            "Frozen attribution only; no retraining or architecture change.",
            "Control hidden1 is the published always-on representation and therefore includes all upstream differences between matched checkpoints.",
            "Per-neuron interventions alter only one reconstructed hidden2 scalar while the other 95 remain learned-gate values.",
            "The final 7-class output layer is held learned for all attribution interventions.",
            "Hybrid H/W/B combinations may be outside the training distribution; they are attribution tests, not deployable models.",
            "No neuron or hybrid is selected or promoted.",
        ],
    }

    # Global all-neuron 2x2x2.
    for h_control in (False, True):
        h1 = h1_c if h_control else h1_l
        for w_control in (False, True):
            w2 = w2_c if w_control else w2_l
            for b_control in (False, True):
                b2 = b2_c if b_control else b2_l
                h2 = relu(h1 @ w2 + b2)
                pred = logits_to_pred(h2, out_w_l, out_b_l)
                label = (
                    f"H_{source_name(h_control)}__"
                    f"W_{source_name(w_control)}__"
                    f"B_{source_name(b_control)}"
                )
                result["global_factorial_2x2x2"][label] = summarize(
                    k, pred, learned_pred, always_pred, scopes
                )

    # Per-neuron factorial: baseline L/L/L omitted because it is identity.
    combos = [
        (True, False, False),   # H only
        (False, True, False),   # W only
        (False, False, True),   # B only
        (True, True, False),    # H+W
        (True, False, True),    # H+B
        (False, True, True),    # W+B
        (True, True, True),     # H+W+B
    ]

    baseline_logits = h2_l @ out_w_l + out_b_l

    for j in range(96):
        row = {
            "neuron": int(j),
            "learned_activation_regressed_mean": float(np.mean(h2_l[reg48, j])),
            "control_activation_regressed_mean": float(np.mean(h2_c[reg48, j])),
            "activation_delta_regressed_mean": float(np.mean(h2_l[reg48, j] - h2_c[reg48, j])),
            "kernel_column_l2_delta": float(np.linalg.norm(w2_l[:, j] - w2_c[:, j])),
            "bias_delta": float(b2_l[j] - b2_c[j]),
            "conditions": {},
        }

        for h_control, w_control, b_control in combos:
            h1 = h1_c if h_control else h1_l
            wcol = w2_c[:, j] if w_control else w2_l[:, j]
            bias = b2_c[j] if b_control else b2_l[j]
            new_act = relu(h1 @ wcol + bias)

            # Efficient exact downstream update for only neuron j.
            delta = new_act - h2_l[:, j]
            logits = baseline_logits + delta[:, None] * out_w_l[j, :][None, :]
            pred = logits.argmax(1).astype(np.int32)
            label = (
                f"H_{source_name(h_control)}__"
                f"W_{source_name(w_control)}__"
                f"B_{source_name(b_control)}"
            )
            row["conditions"][label] = summarize(
                k, pred, learned_pred, always_pred, scopes
            )

        result["per_neuron_factorial_2x2x2"].append(row)

    # Descriptive rankings by net K3 for each causal factor condition.
    result["descriptive_top"] = {}
    for h_control, w_control, b_control in combos:
        label = (
            f"H_{source_name(h_control)}__"
            f"W_{source_name(w_control)}__"
            f"B_{source_name(b_control)}"
        )
        ranked = sorted(
            result["per_neuron_factorial_2x2x2"],
            key=lambda r: (
                r["conditions"][label]["k3_net_correct_vs_learned"],
                r["conditions"][label]["original_48_recovered"],
                -r["conditions"][label]["learned_correct_293_broken"],
            ),
            reverse=True,
        )
        result["descriptive_top"][label] = [
            {
                "neuron": r["neuron"],
                "net_K3": r["conditions"][label]["k3_net_correct_vs_learned"],
                "recovered48": r["conditions"][label]["original_48_recovered"],
                "lost29": r["conditions"][label]["original_29_corrections_lost"],
                "K2": r["conditions"][label]["metrics"]["K2"]["exact"],
                "K3": r["conditions"][label]["metrics"]["K3"]["exact"],
                "K4": r["conditions"][label]["metrics"]["K4"]["exact"],
                "A": r["conditions"][label]["metrics"]["cluster_A"]["exact"],
            }
            for r in ranked[:20]
        ]

    (a.output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Hidden1 x hidden2-kernel x hidden2-bias factorial audit",
        "",
        "Aucun entrainement. Formation de hidden2 decomposee en H/W/B.",
        "",
        "## Factoriel global 2x2x2",
        "",
        "| H1 | W2 | b2 | K2 | K3 | K4 | A | recup48 | perd29 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for h_control in (False, True):
        for w_control in (False, True):
            for b_control in (False, True):
                label = (
                    f"H_{source_name(h_control)}__"
                    f"W_{source_name(w_control)}__"
                    f"B_{source_name(b_control)}"
                )
                x = result["global_factorial_2x2x2"][label]
                m = x["metrics"]
                lines.append(
                    f"| {source_name(h_control)} | {source_name(w_control)} | "
                    f"{source_name(b_control)} | {100*m['K2']['exact']:.2f}% | "
                    f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
                    f"{100*m['cluster_A']['exact']:.2f}% | "
                    f"{x['original_48_recovered']} | {x['original_29_corrections_lost']} |"
                )

    key_labels = [
        ("H_control__W_learned__B_learned", "H1 control uniquement"),
        ("H_learned__W_control__B_learned", "W2 colonne control uniquement"),
        ("H_learned__W_learned__B_control", "bias control uniquement"),
        ("H_control__W_control__B_learned", "H1 + W2 control"),
        ("H_control__W_control__B_control", "H1 + W2 + bias control"),
    ]
    for label, title in key_labels:
        lines += [
            "",
            f"## Top 12 — {title}",
            "",
            "| neurone | net K3 | recup48 | perd29 | K2 | K3 | K4 | A |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in result["descriptive_top"][label][:12]:
            lines.append(
                f"| {r['neuron']} | {r['net_K3']:+d} | {r['recovered48']} | "
                f"{r['lost29']} | {100*r['K2']:.2f}% | {100*r['K3']:.2f}% | "
                f"{100*r['K4']:.2f}% | {100*r['A']:.2f}% |"
            )

    lines += [
        "",
        "Attribution sur checkpoints figes uniquement; aucune promotion automatique.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
