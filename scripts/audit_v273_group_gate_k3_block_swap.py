"""Frozen block-swap audit for K3 regressions in the learned-gate Exact-K model.

No training. The two matched models start from the published weights of run
37019193261. We replay their nested Exact-K base networks with external gates.

Primary question:
Among the K3 rows that were correct in always_on and wrong in learned_gate,
which learned weight block is responsible for errors that persist even when
router/local-cardinality are restored to gate=1?

Interventions:
- learned base + actual gate (replay)
- learned base + gate=1
- always-on base + gate=1 (replay)
- always-on base + actual learned gate
- learned base + gate=1 with one block transplanted from always-on
- always-on base + gate=1 with the reciprocal learned block transplanted

No result is a deployable correction.
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

BLOCKS = {
    "candidate_set_encoder": [
        "candidate_norm",
        "candidate_hidden1",
        "candidate_hidden2",
        "attention_logits",
        "cluster_norm",
        "cluster_hidden1",
        "cluster_hidden2",
    ],
    "candidate_context": ["candidate_context"],
    "spectral_encoder": [
        "v240_dense_channel_norm",
        "v240_dense_conv1",
        "v240_dense_conv2",
        "v240_dense_conv3",
    ],
    "cardinality_head": [
        "v240_cardinality_hidden1",
        "v240_cardinality_hidden2",
        "cardinality",
    ],
}

def load_cluster_a(path, global_index):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        c = int(r["cluster"]); y = int(r["true_k"]); p = int(r["predicted_k"])
        d = by.setdefault(c, {"n": 0, "under": 0})
        d["n"] += 1; d["under"] += p < y
    cid = max(by, key=lambda c: (by[c]["under"] / by[c]["n"], by[c]["n"]))
    ids = {int(r["global_index"]) for r in rows if int(r["cluster"]) == cid}
    return cid, np.isin(np.asarray(global_index, np.int64), list(ids))

def metric(k, p, sel):
    y = np.asarray(k)[sel]; q = np.asarray(p)[sel]
    return {
        "rows": int(len(y)),
        "exact": float(np.mean(q == y)) if len(y) else None,
        "under": int(np.sum(q < y)),
        "over": int(np.sum(q > y)),
        "pred_hist": np.bincount(q, minlength=7).tolist(),
    }

def gate_inputs(x, gate):
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

def nested_base(wrapper):
    import tensorflow as tf
    nested = [layer for layer in wrapper.layers if isinstance(layer, tf.keras.Model)]
    require(len(nested) == 1, f"expected one nested base, got {[x.name for x in nested]}")
    return nested[0]

def predict_base(base, cache, outer, gate, batch=128):
    gate_arr = np.asarray(gate)
    out = []
    for start in range(0, len(outer), batch):
        ids = outer[start:start + batch]
        x = batch_inputs(cache, ids, FRAMES)
        gg = gate if gate_arr.ndim == 0 else gate_arr[start:start + len(ids)]
        xx = gate_inputs(x, gg)
        out.append(np.asarray(base(xx, training=False), np.float32))
    return np.concatenate(out, axis=0)

def actual_gate(wrapper, cache, outer, batch=128):
    import tensorflow as tf
    gm = tf.keras.Model(wrapper.inputs, wrapper.get_layer("v273_gate_value").output)
    out = []
    for start in range(0, len(outer), batch):
        ids = outer[start:start + batch]
        x = batch_inputs(cache, ids, FRAMES)
        out.append(np.asarray(gm(x, training=False), np.float32).reshape(-1))
    return np.concatenate(out)

def layer_weights(model, names):
    result = {}
    for name in names:
        layer = model.get_layer(name)
        result[name] = [np.array(w, copy=True) for w in layer.get_weights()]
    return result

def set_layer_weights(model, values):
    for name, weights in values.items():
        model.get_layer(name).set_weights(weights)

def block_copy(target, source, names):
    set_layer_weights(target, layer_weights(source, names))

def all_weighted_layer_names(model):
    return [layer.name for layer in model.layers if layer.get_weights()]

def transition_counts(k, ref, test, sel):
    y = np.asarray(k)[sel]; a = np.asarray(ref)[sel]; b = np.asarray(test)[sel]
    return {
        "corrected": int(np.sum((a != y) & (b == y))),
        "regressed": int(np.sum((a == y) & (b != y))),
        "net_correct": int(np.sum(b == y) - np.sum(a == y)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--always-weights", type=Path, required=True)
    ap.add_argument("--learned-weights", type=Path, required=True)
    ap.add_argument("--saved-predictions", type=Path, required=True)
    ap.add_argument("--cluster-rows", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    cache, parts, _ = load_bundle(a.bundle, a.config)
    outer = np.asarray(parts["outer"], np.int64)
    k = np.minimum(cache["exact"][outer].astype(np.int32), 6)

    with np.load(a.saved_predictions, allow_pickle=False) as z:
        saved = {key: np.asarray(z[key]) for key in z.files}
    np.testing.assert_array_equal(saved["global_index"], outer)
    np.testing.assert_array_equal(saved["k"], k)
    always_saved = saved["always_on_predicted"].astype(np.int32)
    learned_saved = saved["learned_gate_predicted"].astype(np.int32)

    always_wrapper = build_model("always_on", SEED)
    learned_wrapper = build_model("learned_gate", SEED)
    always_wrapper.load_weights(a.always_weights)
    learned_wrapper.load_weights(a.learned_weights)
    always_base = nested_base(always_wrapper)
    learned_base = nested_base(learned_wrapper)

    for block, names in BLOCKS.items():
        for name in names:
            require(always_base.get_layer(name).get_weights() is not None, f"missing {block}:{name}")
            learned_base.get_layer(name)

    gate = actual_gate(learned_wrapper, cache, outer)

    learned_actual_prob = predict_base(learned_base, cache, outer, gate)
    learned_actual = learned_actual_prob.argmax(1).astype(np.int32)
    np.testing.assert_array_equal(learned_actual, learned_saved)

    always_one_prob = predict_base(always_base, cache, outer, 1.0)
    always_one = always_one_prob.argmax(1).astype(np.int32)
    np.testing.assert_array_equal(always_one, always_saved)

    learned_one_prob = predict_base(learned_base, cache, outer, 1.0)
    learned_one = learned_one_prob.argmax(1).astype(np.int32)

    always_actual_prob = predict_base(always_base, cache, outer, gate)
    always_actual = always_actual_prob.argmax(1).astype(np.int32)

    cid, A = load_cluster_a(a.cluster_rows, outer)
    K3 = k == 3
    original_reg = K3 & (always_saved == 3) & (learned_saved != 3)
    persistent = original_reg & (learned_one != 3)
    require(int(original_reg.sum()) == 48, "expected 48 original K3 regressions")
    require(int(persistent.sum()) == 34, f"expected 34 persistent regressions, got {persistent.sum()}")

    result = {
        "status": "completed",
        "training": False,
        "cluster_A_id": int(cid),
        "original_k3_regressions": int(original_reg.sum()),
        "persistent_after_gate1": int(persistent.sum()),
        "replay": {
            "learned_actual_matches_saved": True,
            "always_gate1_matches_saved": True,
        },
        "blocks": {},
        "controls": {},
        "limitations": [
            "Block swaps are frozen-model interventions and can create hybrid parameter combinations not seen during training.",
            "A block that rescues a case is evidence that the learned parameters in that block participate in the error, not that the block alone is the sole root cause.",
            "Reciprocal damage tests are included because interactions between blocks can be non-additive.",
        ],
    }

    def summarize(name, pred):
        return {
            "all": metric(k, pred, np.ones(len(k), bool)),
            "K2": metric(k, pred, k == 2),
            "K3": metric(k, pred, K3),
            "K4": metric(k, pred, k == 4),
            "cluster_A": metric(k, pred, A),
            "vs_always_K3": transition_counts(k, always_saved, pred, K3),
            "original_48_recovered": int(np.sum(original_reg & (pred == 3))),
            "persistent_34_recovered": int(np.sum(persistent & (pred == 3))),
        }

    result["controls"]["learned_actual_gate"] = summarize("learned_actual_gate", learned_actual)
    result["controls"]["learned_gate1"] = summarize("learned_gate1", learned_one)
    result["controls"]["always_gate1"] = summarize("always_gate1", always_one)
    result["controls"]["always_actual_gate"] = summarize("always_actual_gate", always_actual)

    learned_original = {name: layer_weights(learned_base, names) for name, names in BLOCKS.items()}
    always_original = {name: layer_weights(always_base, names) for name, names in BLOCKS.items()}

    # Rescue: learned base, gate=1, one control block transplanted.
    for block, names in BLOCKS.items():
        block_copy(learned_base, always_base, names)
        pred = predict_base(learned_base, cache, outer, 1.0).argmax(1).astype(np.int32)
        result["blocks"].setdefault(block, {})["control_into_learned_gate1"] = summarize(block, pred)
        set_layer_weights(learned_base, learned_original[block])

    # Reciprocal damage: always-on base, gate=1, one learned block transplanted.
    for block, names in BLOCKS.items():
        block_copy(always_base, learned_base, names)
        pred = predict_base(always_base, cache, outer, 1.0).argmax(1).astype(np.int32)
        result["blocks"].setdefault(block, {})["learned_into_control_gate1"] = summarize(block, pred)
        set_layer_weights(always_base, always_original[block])

    # Whole-base controls.
    learned_all = layer_weights(learned_base, all_weighted_layer_names(learned_base))
    always_all = layer_weights(always_base, all_weighted_layer_names(always_base))
    set_layer_weights(learned_base, always_all)
    pred = predict_base(learned_base, cache, outer, 1.0).argmax(1).astype(np.int32)
    result["controls"]["all_control_base_into_learned_gate1"] = summarize("all_control", pred)
    np.testing.assert_array_equal(pred, always_saved)
    set_layer_weights(learned_base, learned_all)

    (a.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# K3 block-swap audit",
        "",
        "Aucun entrainement. Deux modeles publies figes.",
        "",
        f"- Regressions K3 originales: **{int(original_reg.sum())}**",
        f"- Persistantes apres gate=1 dans le modele learned: **{int(persistent.sum())}**",
        "",
        "## Controle",
        "",
        f"- Base always-on + gate learned reel: K3 exact {100*result['controls']['always_actual_gate']['K3']['exact']:.2f}%",
        f"- Base learned + gate=1: K3 exact {100*result['controls']['learned_gate1']['K3']['exact']:.2f}%",
        "",
        "## Transplantations de blocs — base learned, gate=1",
        "",
        "| Bloc remis depuis always-on | K3 exact | 48 recuperes | 34 persistants recuperes | K2 exact | K4 exact | A exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for block in BLOCKS:
        x = result["blocks"][block]["control_into_learned_gate1"]
        lines.append(
            f"| {block} | {100*x['K3']['exact']:.2f}% | {x['original_48_recovered']} | "
            f"{x['persistent_34_recovered']} | {100*x['K2']['exact']:.2f}% | "
            f"{100*x['K4']['exact']:.2f}% | {100*x['cluster_A']['exact']:.2f}% |"
        )
    lines += [
        "",
        "## Transplantations reciproques — bloc learned dans base always-on",
        "",
        "| Bloc learned injecte | K3 exact | K3 corrects perdus (net vs always-on) |",
        "|---|---:|---:|",
    ]
    for block in BLOCKS:
        x = result["blocks"][block]["learned_into_control_gate1"]
        lines.append(
            f"| {block} | {100*x['K3']['exact']:.2f}% | {x['vs_always_K3']['net_correct']:+d} |"
        )

    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
