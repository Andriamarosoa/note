"""Factorial hidden2 activation-vs-output audit for V27.3 learned gate.

No training. No architecture change.

This audit separates two learned components at the final Exact-K cardinality path:
  R: hidden2 representation/activation source (learned-gate vs always-on)
  W: final 7-class cardinality Dense source (learned-gate vs always-on)

Global 2x2 conditions:
  R_L x W_L, R_L x W_C, R_C x W_L, R_C x W_C.

Per-neuron factorial interventions start from R_L x W_L and, for each hidden2
unit independently, replace:
  - only that neuron's activation with control activation,
  - only that neuron's output-kernel row with control row,
  - both activation and output-kernel row with control values.

The final bias is held learned for all per-neuron interventions, because the
previous head-component audit measured no K3 recovery from the output bias
alone. A separate global bias control is still reported.

This is a frozen attribution audit; no neuron or hybrid is promoted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import batch_inputs, load_bundle, require

SEED = 16061 + 1003
FRAMES = 31
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


def collect_hidden(base, cache, outer, gate, batch_size=256):
    import tensorflow as tf
    probe = tf.keras.Model(base.inputs, base.get_layer(H2).output)
    acts = []
    probs = []
    gate_arr = np.asarray(gate)
    for start in range(0, len(outer), batch_size):
        ids = outer[start:start + batch_size]
        x = batch_inputs(cache, ids, FRAMES)
        g = gate if gate_arr.ndim == 0 else gate_arr[start:start + len(ids)]
        xx = apply_gate(x, g)
        acts.append(np.asarray(probe(xx, training=False), np.float32))
        probs.append(np.asarray(base(xx, training=False), np.float32))
    return np.concatenate(acts), np.concatenate(probs)


def softmax(logits):
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def logits_from_hidden(hidden, kernel, bias):
    return np.asarray(hidden, np.float32) @ np.asarray(kernel, np.float32) + np.asarray(bias, np.float32)


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


def load_cluster_a(path, global_index):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        c = int(r["cluster"])
        y = int(r["true_k"])
        p = int(r["predicted_k"])
        d = by.setdefault(c, {"n": 0, "under": 0})
        d["n"] += 1
        d["under"] += p < y
    cid = max(by, key=lambda c: (by[c]["under"] / by[c]["n"], by[c]["n"]))
    ids = {int(r["global_index"]) for r in rows if int(r["cluster"]) == cid}
    return cid, np.isin(np.asarray(global_index, np.int64), list(ids))


def main():
    ap = argparse.ArgumentParser()
    for name in (
        "bundle",
        "config",
        "always-weights",
        "learned-weights",
        "saved-predictions",
        "cluster-rows",
        "output",
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
    learned_gate = np.asarray(saved["learned_gate_gate"], np.float32)

    always_wrapper = build_model("always_on", SEED)
    learned_wrapper = build_model("learned_gate", SEED)
    always_wrapper.load_weights(a.always_weights)
    learned_wrapper.load_weights(a.learned_weights)
    always_base = nested(always_wrapper)
    learned_base = nested(learned_wrapper)

    # Published representations.
    h_l, p_l = collect_hidden(learned_base, cache, outer, learned_gate)
    h_c, p_c = collect_hidden(always_base, cache, outer, 1.0)
    np.testing.assert_array_equal(p_l.argmax(1), learned_pred)
    np.testing.assert_array_equal(p_c.argmax(1), always_pred)

    w_l, b_l = [np.asarray(x, np.float32) for x in learned_base.get_layer(OUT).get_weights()]
    w_c, b_c = [np.asarray(x, np.float32) for x in always_base.get_layer(OUT).get_weights()]
    require(w_l.shape == w_c.shape == (96, 7), f"unexpected output kernel shape {w_l.shape}")
    require(h_l.shape == h_c.shape == (len(outer), 96), f"unexpected hidden shape {h_l.shape}")

    # Exact replay from hidden2 proves no omitted layer exists between hidden2 and output.
    replay_ll = softmax(logits_from_hidden(h_l, w_l, b_l))
    replay_cc = softmax(logits_from_hidden(h_c, w_c, b_c))
    require(float(np.max(np.abs(replay_ll - p_l))) < 1e-5, "learned hidden->output replay mismatch")
    require(float(np.max(np.abs(replay_cc - p_c))) < 1e-5, "control hidden->output replay mismatch")

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
        "population": {
            "k3_regressions": int(reg48.sum()),
            "k3_corrections": int(corr29.sum()),
        },
        "replay": {
            "learned_hidden_to_output_max_abs": float(np.max(np.abs(replay_ll - p_l))),
            "control_hidden_to_output_max_abs": float(np.max(np.abs(replay_cc - p_c))),
        },
        "global_factorial": {},
        "global_bias_controls": {},
        "per_neuron_factorial": [],
        "limitations": [
            "Frozen 2x2 attribution audit only; no retraining and no architecture change.",
            "Per-neuron activation substitution uses the corresponding hidden2 scalar from the control checkpoint on the same outer example.",
            "Per-neuron output substitution replaces only one row of the final 7-class kernel; final bias stays learned.",
            "Hybrid conditions can be outside each model's training distribution and are attribution tests, not deployable models.",
            "No neuron or hybrid is selected or promoted.",
        ],
    }

    global_conditions = {
        "R_learned__W_learned": (h_l, w_l, b_l),
        "R_learned__W_control": (h_l, w_c, b_c),
        "R_control__W_learned": (h_c, w_l, b_l),
        "R_control__W_control": (h_c, w_c, b_c),
    }
    global_preds = {}
    for label, (h, w, b) in global_conditions.items():
        pred = logits_from_hidden(h, w, b).argmax(1).astype(np.int32)
        global_preds[label] = pred
        result["global_factorial"][label] = summarize(
            k, pred, learned_pred, always_pred, scopes
        )

    np.testing.assert_array_equal(global_preds["R_learned__W_learned"], learned_pred)
    np.testing.assert_array_equal(global_preds["R_control__W_control"], always_pred)

    # Bias source alone, holding learned representation and kernel fixed.
    for label, bias in {
        "learned_R_learned_W_control_bias": b_c,
        "learned_R_control_W_learned_bias": b_l,
    }.items():
        if "control_W" in label:
            pred = logits_from_hidden(h_l, w_l, bias).argmax(1).astype(np.int32)
        else:
            pred = logits_from_hidden(h_l, w_c, bias).argmax(1).astype(np.int32)
        result["global_bias_controls"][label] = summarize(
            k, pred, learned_pred, always_pred, scopes
        )

    # Per-neuron factorial interventions on learned baseline.
    for j in range(96):
        # Activation-only: replace one hidden activation scalar per example.
        h_act = h_l.copy()
        h_act[:, j] = h_c[:, j]
        p_act = logits_from_hidden(h_act, w_l, b_l).argmax(1).astype(np.int32)

        # Output-row-only: replace one row of final output kernel.
        w_out = w_l.copy()
        w_out[j, :] = w_c[j, :]
        p_out = logits_from_hidden(h_l, w_out, b_l).argmax(1).astype(np.int32)

        # Both neuron activation and output row.
        p_both = logits_from_hidden(h_act, w_out, b_l).argmax(1).astype(np.int32)

        row = {
            "neuron": int(j),
            "activation_only": summarize(k, p_act, learned_pred, always_pred, scopes),
            "output_row_only": summarize(k, p_out, learned_pred, always_pred, scopes),
            "activation_and_output_row": summarize(k, p_both, learned_pred, always_pred, scopes),
            "activation_delta_regressed_mean": float(np.mean(h_l[reg48, j] - h_c[reg48, j])),
            "output_row_l2_delta": float(np.linalg.norm(w_l[j, :] - w_c[j, :])),
            "learned_margin_K3_K2": float(w_l[j, 3] - w_l[j, 2]),
            "learned_margin_K3_K4": float(w_l[j, 3] - w_l[j, 4]),
            "control_margin_K3_K2": float(w_c[j, 3] - w_c[j, 2]),
            "control_margin_K3_K4": float(w_c[j, 3] - w_c[j, 4]),
        }
        result["per_neuron_factorial"].append(row)

    # Descriptive top lists for each intervention family.
    def rank_family(key):
        return sorted(
            result["per_neuron_factorial"],
            key=lambda r: (
                r[key]["k3_net_correct_vs_learned"],
                r[key]["original_48_recovered"],
                -r[key]["learned_correct_293_broken"],
            ),
            reverse=True,
        )[:20]

    result["descriptive_top"] = {
        "activation_only": [
            {
                "neuron": r["neuron"],
                "net_K3": r["activation_only"]["k3_net_correct_vs_learned"],
                "recovered48": r["activation_only"]["original_48_recovered"],
                "lost29": r["activation_only"]["original_29_corrections_lost"],
                "K2": r["activation_only"]["metrics"]["K2"]["exact"],
                "K3": r["activation_only"]["metrics"]["K3"]["exact"],
                "K4": r["activation_only"]["metrics"]["K4"]["exact"],
                "A": r["activation_only"]["metrics"]["cluster_A"]["exact"],
            }
            for r in rank_family("activation_only")
        ],
        "output_row_only": [
            {
                "neuron": r["neuron"],
                "net_K3": r["output_row_only"]["k3_net_correct_vs_learned"],
                "recovered48": r["output_row_only"]["original_48_recovered"],
                "lost29": r["output_row_only"]["original_29_corrections_lost"],
                "K2": r["output_row_only"]["metrics"]["K2"]["exact"],
                "K3": r["output_row_only"]["metrics"]["K3"]["exact"],
                "K4": r["output_row_only"]["metrics"]["K4"]["exact"],
                "A": r["output_row_only"]["metrics"]["cluster_A"]["exact"],
            }
            for r in rank_family("output_row_only")
        ],
        "both": [
            {
                "neuron": r["neuron"],
                "net_K3": r["activation_and_output_row"]["k3_net_correct_vs_learned"],
                "recovered48": r["activation_and_output_row"]["original_48_recovered"],
                "lost29": r["activation_and_output_row"]["original_29_corrections_lost"],
                "K2": r["activation_and_output_row"]["metrics"]["K2"]["exact"],
                "K3": r["activation_and_output_row"]["metrics"]["K3"]["exact"],
                "K4": r["activation_and_output_row"]["metrics"]["K4"]["exact"],
                "A": r["activation_and_output_row"]["metrics"]["cluster_A"]["exact"],
            }
            for r in rank_family("activation_and_output_row")
        ],
    }

    (a.output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Hidden2 activation vs output-weight factorial audit",
        "",
        "Aucun entrainement. Audit 2x2 sur checkpoints publies figes.",
        "",
        "## Factoriel global",
        "",
        "| Representation | Poids sortie | K2 | K3 | K4 | A | recup 48 | perd 29 corrections |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (
        "R_learned__W_learned",
        "R_learned__W_control",
        "R_control__W_learned",
        "R_control__W_control",
    ):
        x = result["global_factorial"][label]
        m = x["metrics"]
        rsrc = "learned" if "R_learned" in label else "control"
        wsrc = "learned" if "W_learned" in label else "control"
        lines.append(
            f"| {rsrc} | {wsrc} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% | {x['original_48_recovered']} | "
            f"{x['original_29_corrections_lost']} |"
        )

    for family, title in (
        ("activation_only", "Activation seule remplacee"),
        ("output_row_only", "Ligne de poids sortie seule remplacee"),
        ("both", "Activation + ligne de poids remplacees"),
    ):
        lines += [
            "",
            f"## Top 12 — {title}",
            "",
            "| neurone | net K3 | recup 48 | perd 29 | K2 | K3 | K4 | A |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in result["descriptive_top"][family][:12]:
            lines.append(
                f"| {r['neuron']} | {r['net_K3']:+d} | {r['recovered48']} | {r['lost29']} | "
                f"{100*r['K2']:.2f}% | {100*r['K3']:.2f}% | "
                f"{100*r['K4']:.2f}% | {100*r['A']:.2f}% |"
            )

    lines += [
        "",
        "Descriptif/causal sur modeles figes uniquement; aucune promotion automatique.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
