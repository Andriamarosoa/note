"""Frozen branch attribution audit for v240_cardinality_context.

No training. No architecture change.

cardinality_context = concat([
    spectral_avg (96),
    spectral_max (96),
    candidate_context (96),
])

This audit separates the three branch sources between the matched learned-gate
and always-on checkpoints while holding hidden1 W/b, hidden2, and the final
7-class output at the learned-gate checkpoint.

It performs:
  1) global 2^3 branch-source combinations;
  2) per-hidden1-unit branch interventions (single branches, pairs, all three),
     changing only one hidden1 scalar at a time while the other 191 remain
     learned;
  3) explicit reporting for hidden1 units previously implicated by the
     context-hidden1 audit.

Frozen attribution only; no model selection or promotion.
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

AVG = "v240_dense_global_average"
MAX = "v240_dense_global_max"
CAND = "candidate_context"
CTX = "v240_cardinality_context"
H1 = "v240_cardinality_hidden1"
H2 = "v240_cardinality_hidden2"
OUT = "cardinality"

TARGET_UNITS = [135, 73, 109, 182, 104, 76, 107, 111, 124, 59, 158, 23, 116]
TARGET_H2 = 37


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


def collect(base, cache, outer, gate, batch_size=256):
    import tensorflow as tf
    probe = tf.keras.Model(
        base.inputs,
        [
            base.get_layer(AVG).output,
            base.get_layer(MAX).output,
            base.get_layer(CAND).output,
            base.get_layer(CTX).output,
            base.get_layer(H1).output,
            base.get_layer(H2).output,
            base.output,
        ],
    )
    out = [[] for _ in range(7)]
    ga = np.asarray(gate)
    for start in range(0, len(outer), batch_size):
        ids = outer[start:start + batch_size]
        x = batch_inputs(cache, ids, FRAMES)
        g = gate if ga.ndim == 0 else ga[start:start + len(ids)]
        vals = probe(apply_gate(x, g), training=False)
        for bucket, val in zip(out, vals):
            bucket.append(np.asarray(val, np.float32))
    return tuple(np.concatenate(x) for x in out)


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
        c = int(r["cluster"])
        y = int(r["true_k"])
        p = int(r["predicted_k"])
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


def branch_label(a_ctl, m_ctl, c_ctl):
    return (
        f"avg_{'control' if a_ctl else 'learned'}__"
        f"max_{'control' if m_ctl else 'learned'}__"
        f"cand_{'control' if c_ctl else 'learned'}"
    )


def mixed_context(avg_l, avg_c, max_l, max_c, cand_l, cand_c, a_ctl, m_ctl, c_ctl):
    return np.concatenate(
        [
            avg_c if a_ctl else avg_l,
            max_c if m_ctl else max_l,
            cand_c if c_ctl else cand_l,
        ],
        axis=1,
    ).astype(np.float32)


def downstream_from_h1(h1, w2, b2, out_w, out_b):
    h2 = relu(h1 @ w2 + b2)
    logits = h2 @ out_w + out_b
    return logits.argmax(1).astype(np.int32), h2


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

    mc = build_model("always_on", SEED)
    ml = build_model("learned_gate", SEED)
    mc.load_weights(a.always_weights)
    ml.load_weights(a.learned_weights)
    bc = nested(mc)
    bl = nested(ml)

    avg_c, max_c, cand_c, ctx_c, h1_c, h2_c, p_c = collect(bc, cache, outer, 1.0)
    avg_l, max_l, cand_l, ctx_l, h1_l, h2_l, p_l = collect(bl, cache, outer, gate)

    np.testing.assert_array_equal(p_c.argmax(1), always_pred)
    np.testing.assert_array_equal(p_l.argmax(1), learned_pred)

    require(avg_l.shape[1] == 96 and max_l.shape[1] == 96 and cand_l.shape[1] == 96,
            f"unexpected branch dims {avg_l.shape} {max_l.shape} {cand_l.shape}")
    require(ctx_l.shape[1] == 288, f"unexpected context dim {ctx_l.shape}")

    replay_ctx_l = np.concatenate([avg_l, max_l, cand_l], axis=1)
    replay_ctx_c = np.concatenate([avg_c, max_c, cand_c], axis=1)
    ctx_err_l = float(np.max(np.abs(replay_ctx_l - ctx_l)))
    ctx_err_c = float(np.max(np.abs(replay_ctx_c - ctx_c)))
    require(ctx_err_l < 1e-6, f"learned context replay mismatch {ctx_err_l}")
    require(ctx_err_c < 1e-6, f"control context replay mismatch {ctx_err_c}")

    w1_l, b1_l = [np.asarray(x, np.float32) for x in bl.get_layer(H1).get_weights()]
    w2_l, b2_l = [np.asarray(x, np.float32) for x in bl.get_layer(H2).get_weights()]
    out_w_l, out_b_l = [np.asarray(x, np.float32) for x in bl.get_layer(OUT).get_weights()]

    require(w1_l.shape == (288, 192), f"unexpected W1 shape {w1_l.shape}")

    h1_replay = relu(ctx_l @ w1_l + b1_l)
    require(float(np.max(np.abs(h1_replay - h1_l))) < 1e-5, "hidden1 replay mismatch")
    pred_replay, h2_replay = downstream_from_h1(h1_l, w2_l, b2_l, out_w_l, out_b_l)
    np.testing.assert_array_equal(pred_replay, learned_pred)
    require(float(np.max(np.abs(h2_replay - h2_l))) < 1e-5, "hidden2 replay mismatch")

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
        "replay": {
            "learned_context_max_abs": ctx_err_l,
            "control_context_max_abs": ctx_err_c,
        },
        "branches": {
            "spectral_avg": [0, 96],
            "spectral_max": [96, 192],
            "candidate_context": [192, 288],
        },
        "global_branch_factorial": {},
        "per_hidden1_unit_branch_factorial": [],
        "target_units": TARGET_UNITS,
        "target_hidden2_neuron": TARGET_H2,
        "limitations": [
            "Frozen attribution only; no retraining or architecture change.",
            "All hidden1/hidden2/output weights and biases are held learned-gate.",
            "Control branch values come from the matched published always-on checkpoint.",
            "Global mixed contexts and per-unit mixed contexts can be outside training distribution.",
            "Per-unit interventions alter only one hidden1 activation; the other 191 remain learned.",
            "No branch, unit, or hybrid is selected or promoted.",
        ],
    }

    # Global 2^3 branch source combinations.
    for a_ctl in (False, True):
        for m_ctl in (False, True):
            for c_ctl in (False, True):
                ctx = mixed_context(
                    avg_l, avg_c, max_l, max_c, cand_l, cand_c,
                    a_ctl, m_ctl, c_ctl,
                )
                h1 = relu(ctx @ w1_l + b1_l)
                pred, h2 = downstream_from_h1(h1, w2_l, b2_l, out_w_l, out_b_l)
                label = branch_label(a_ctl, m_ctl, c_ctl)
                x = summarize(k, pred, learned_pred, always_pred, scopes)
                x["target_h2_37"] = {
                    "regressed_mean": float(np.mean(h2[reg48, TARGET_H2])),
                    "delta_vs_learned_regressed": float(
                        np.mean(h2[reg48, TARGET_H2] - h2_l[reg48, TARGET_H2])
                    ),
                }
                x["target_hidden1_units"] = {
                    str(j): {
                        "regressed_mean": float(np.mean(h1[reg48, j])),
                        "delta_vs_learned_regressed": float(
                            np.mean(h1[reg48, j] - h1_l[reg48, j])
                        ),
                    }
                    for j in TARGET_UNITS
                }
                result["global_branch_factorial"][label] = x

    # Per-hidden1-unit branch interventions.
    branch_combos = [
        (True, False, False),   # avg only
        (False, True, False),   # max only
        (False, False, True),   # candidate only
        (True, True, False),    # avg+max
        (True, False, True),    # avg+cand
        (False, True, True),    # max+cand
        (True, True, True),     # all
    ]

    baseline_h2_pre = h1_l @ w2_l + b2_l
    baseline_h2 = relu(baseline_h2_pre)
    baseline_logits = baseline_h2 @ out_w_l + out_b_l

    mixed_cache = {
        branch_label(a_ctl, m_ctl, c_ctl): mixed_context(
            avg_l, avg_c, max_l, max_c, cand_l, cand_c,
            a_ctl, m_ctl, c_ctl,
        )
        for a_ctl, m_ctl, c_ctl in branch_combos
    }

    for j in range(192):
        row = {
            "unit": int(j),
            "conditions": {},
            "w_to_h2_37": float(w2_l[j, TARGET_H2]),
            "learned_activation_regressed_mean": float(np.mean(h1_l[reg48, j])),
            "control_activation_regressed_mean": float(np.mean(h1_c[reg48, j])),
        }

        for a_ctl, m_ctl, c_ctl in branch_combos:
            label = branch_label(a_ctl, m_ctl, c_ctl)
            ctx = mixed_cache[label]
            new_h1_j = relu(ctx @ w1_l[:, j] + b1_l[j])

            dh1 = new_h1_j - h1_l[:, j]
            new_h2_pre = baseline_h2_pre + dh1[:, None] * w2_l[j, :][None, :]
            new_h2 = relu(new_h2_pre)
            logits = baseline_logits + (new_h2 - baseline_h2) @ out_w_l
            pred = logits.argmax(1).astype(np.int32)

            x = summarize(k, pred, learned_pred, always_pred, scopes)
            x["unit_activation"] = {
                "regressed_mean": float(np.mean(new_h1_j[reg48])),
                "delta_vs_learned_regressed": float(
                    np.mean(new_h1_j[reg48] - h1_l[reg48, j])
                ),
            }
            x["target_h2_37"] = {
                "regressed_mean": float(np.mean(new_h2[reg48, TARGET_H2])),
                "delta_vs_learned_regressed": float(
                    np.mean(new_h2[reg48, TARGET_H2] - h2_l[reg48, TARGET_H2])
                ),
            }
            row["conditions"][label] = x

        result["per_hidden1_unit_branch_factorial"].append(row)

    labels = {
        "avg_only": branch_label(True, False, False),
        "max_only": branch_label(False, True, False),
        "candidate_only": branch_label(False, False, True),
        "avg_max": branch_label(True, True, False),
        "avg_candidate": branch_label(True, False, True),
        "max_candidate": branch_label(False, True, True),
        "all": branch_label(True, True, True),
    }

    result["descriptive_top"] = {}
    for key, label in labels.items():
        ranked = sorted(
            result["per_hidden1_unit_branch_factorial"],
            key=lambda r: (
                r["conditions"][label]["k3_net_correct_vs_learned"],
                r["conditions"][label]["original_48_recovered"],
                -r["conditions"][label]["learned_correct_293_broken"],
            ),
            reverse=True,
        )
        result["descriptive_top"][key] = [
            {
                "unit": r["unit"],
                "net_K3": r["conditions"][label]["k3_net_correct_vs_learned"],
                "recovered48": r["conditions"][label]["original_48_recovered"],
                "lost29": r["conditions"][label]["original_29_corrections_lost"],
                "K2": r["conditions"][label]["metrics"]["K2"]["exact"],
                "K3": r["conditions"][label]["metrics"]["K3"]["exact"],
                "K4": r["conditions"][label]["metrics"]["K4"]["exact"],
                "A": r["conditions"][label]["metrics"]["cluster_A"]["exact"],
                "unit_delta": r["conditions"][label]["unit_activation"]["delta_vs_learned_regressed"],
                "h2_37_delta": r["conditions"][label]["target_h2_37"]["delta_vs_learned_regressed"],
                "w_to_h2_37": r["w_to_h2_37"],
            }
            for r in ranked[:25]
        ]

    (a.output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Cardinality-context branch attribution audit",
        "",
        "Aucun entrainement. Branches: spectral_avg / spectral_max / candidate_context.",
        "",
        "## Factoriel global 2^3",
        "",
        "| avg | max | candidate | K2 | K3 | K4 | A | recup48 | perd29 | delta h2[37] |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a_ctl in (False, True):
        for m_ctl in (False, True):
            for c_ctl in (False, True):
                label = branch_label(a_ctl, m_ctl, c_ctl)
                x = result["global_branch_factorial"][label]
                m = x["metrics"]
                lines.append(
                    f"| {'control' if a_ctl else 'learned'} | "
                    f"{'control' if m_ctl else 'learned'} | "
                    f"{'control' if c_ctl else 'learned'} | "
                    f"{100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | "
                    f"{100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% | "
                    f"{x['original_48_recovered']} | {x['original_29_corrections_lost']} | "
                    f"{x['target_h2_37']['delta_vs_learned_regressed']:+.4f} |"
                )

    for key, title in (
        ("avg_only", "spectral_avg control uniquement"),
        ("max_only", "spectral_max control uniquement"),
        ("candidate_only", "candidate_context control uniquement"),
        ("avg_max", "spectral_avg + spectral_max control"),
        ("avg_candidate", "spectral_avg + candidate_context control"),
        ("max_candidate", "spectral_max + candidate_context control"),
    ):
        lines += [
            "",
            f"## Top 12 hidden1 units — {title}",
            "",
            "| unit | net K3 | recup48 | perd29 | K2 | K3 | K4 | A | delta unit | delta h2[37] |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in result["descriptive_top"][key][:12]:
            lines.append(
                f"| {r['unit']} | {r['net_K3']:+d} | {r['recovered48']} | {r['lost29']} | "
                f"{100*r['K2']:.2f}% | {100*r['K3']:.2f}% | {100*r['K4']:.2f}% | "
                f"{100*r['A']:.2f}% | {r['unit_delta']:+.4f} | {r['h2_37_delta']:+.4f} |"
            )

    lines += [
        "",
        "## Unites deja impliquees — effet de chaque branche seule",
        "",
        "| unit | avg recup | max recup | candidate recup | avg net | max net | candidate net |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_unit = {r["unit"]: r for r in result["per_hidden1_unit_branch_factorial"]}
    for j in TARGET_UNITS:
        row = by_unit[j]
        xa = row["conditions"][labels["avg_only"]]
        xm = row["conditions"][labels["max_only"]]
        xc = row["conditions"][labels["candidate_only"]]
        lines.append(
            f"| {j} | {xa['original_48_recovered']} | {xm['original_48_recovered']} | "
            f"{xc['original_48_recovered']} | {xa['k3_net_correct_vs_learned']:+d} | "
            f"{xm['k3_net_correct_vs_learned']:+d} | {xc['k3_net_correct_vs_learned']:+d} |"
        )

    lines += [
        "",
        "Attribution sur checkpoints figes uniquement; aucune promotion automatique.",
    ]
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
