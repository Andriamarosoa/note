"""Internal-only audit of post-train candidate_hidden1 transplantation.

This script never reads outer fold 3 predictions or labels.

Protocol:
1. Train a uniform learned_gate anchor on inner_fit (folds 1/2/4), 8 epochs.
2. Starting from that trained anchor, fine-tune on the same inner_fit with the
   fixed targeted K2/K3-half + K4-quarter weighting while freezing the current
   freeze_local_combo layers: candidate_norm, candidate_hidden2, cardinality.
3. Evaluate both models on inner_val (fold 0).
4. Post-train only, replace candidate_hidden1 in the weighted/frozen model by
   the uniform-anchor candidate_hidden1 weights and re-evaluate fold 0.
5. Cluster disagreement rows using only model-output signatures; labels are
   used only afterwards to characterize clusters.

No outer fold usage. No automatic promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v273_group_gate_ab import build_model, weight_hash, metrics, transitions
from scripts.train_v273_loss_weighting_ab import class_weight_table, batches
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, epoch_order, array_hash, require

SEED = v260.SEED + 1003
EPOCHS = 8
WEIGHT_ARM = "targeted_k23_half_k4_quarter"
FREEZE_LOCAL = ("candidate_norm", "candidate_hidden2", "cardinality")
TRANSPLANT_LAYER = "candidate_hidden1"


def nested_base(model):
    import tensorflow as tf
    xs = [x for x in model.layers if isinstance(x, tf.keras.Model)]
    require(len(xs) == 1, f"expected one nested base, got {[x.name for x in xs]}")
    return xs[0]


def compile_model(model):
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="sparse_categorical_crossentropy",
    )


def entropy(p):
    p = np.asarray(p, np.float64)
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)


def prediction_margin(p):
    p = np.asarray(p, np.float64)
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def layer_delta(anchor_base, tuned_base, name):
    a = anchor_base.get_layer(name).get_weights()
    b = tuned_base.get_layer(name).get_weights()
    require(len(a) == len(b), "layer tensor inventory mismatch")
    num = float(np.sqrt(sum(np.sum((np.asarray(x)-np.asarray(y))**2) for x, y in zip(a,b))))
    den = float(np.sqrt(sum(np.sum(np.asarray(x)**2) for x in a)))
    return {"l2": num, "relative_l2": num / max(den, 1e-12)}


def deterministic_kmeans(x, k, max_iter=100):
    x = np.asarray(x, np.float64)
    n = len(x)
    require(n >= k, "not enough rows for clustering")
    # Deterministic farthest-point initialization.
    centers = [int(np.argmax(np.sum(x*x, axis=1)))]
    while len(centers) < k:
        c = x[centers]
        d2 = np.min(np.sum((x[:,None,:]-c[None,:,:])**2, axis=2), axis=1)
        d2[centers] = -1.0
        centers.append(int(np.argmax(d2)))
    cent = x[centers].copy()
    labels = np.zeros(n, np.int32)
    for _ in range(max_iter):
        d2 = np.sum((x[:,None,:]-cent[None,:,:])**2, axis=2)
        new = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new, labels):
            labels = new
            break
        labels = new
        for j in range(k):
            take = labels == j
            if take.any():
                cent[j] = x[take].mean(axis=0)
    return labels, cent


def silhouette_score(x, labels):
    x = np.asarray(x, np.float64)
    labels = np.asarray(labels, np.int32)
    n = len(x)
    if n < 3 or len(np.unique(labels)) < 2:
        return None
    d = np.sqrt(np.maximum(
        np.sum((x[:,None,:]-x[None,:,:])**2, axis=2), 0.0
    ))
    vals = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = float(d[i, same].mean()) if same.any() else 0.0
        b = float("inf")
        for c in np.unique(labels):
            if c == labels[i]:
                continue
            take = labels == c
            if take.any():
                b = min(b, float(d[i, take].mean()))
        if not np.isfinite(b):
            vals.append(0.0)
        else:
            vals.append((b-a)/max(a,b,1e-12))
    return float(np.mean(vals))


def hist(values):
    return {str(k): int(v) for k, v in sorted(Counter(int(x) for x in values).items())}


def transition_hist(a, b, mask):
    a = np.asarray(a)[mask]
    b = np.asarray(b)[mask]
    c = Counter(f"{int(x)}->{int(y)}" for x, y in zip(a,b))
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def train_one(model, cache, fit, k, table, output, label):
    import tensorflow as tf
    seq = batches(cache, fit, seed=SEED, shuffle=True, k=k, sample_table=table)
    hist = []
    observed = []
    class CB(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            expected = epoch_order(fit, SEED, epoch)
            np.testing.assert_array_equal(seq.order, expected)
            observed.append(array_hash(seq.order))
        def on_epoch_end(self, epoch, logs=None):
            row = {"epoch": int(epoch)+1, **{x: float(y) for x,y in (logs or {}).items()}}
            hist.append(row)
            write_json(output/f"{label}-progress.json", {
                "arm": label, "completed_epoch": int(epoch)+1, "history": hist
            })
    res = model.fit(
        seq, epochs=EPOCHS, shuffle=False, workers=0, max_queue_size=1, verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(), CB()]
    )
    require(len(hist) == EPOCHS and all(np.isfinite(v).all() for v in res.history.values()),
            f"{label} training failure")
    return hist, observed


def main(args):
    import tensorflow as tf
    require(tf.__version__ == "2.15.1", "pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(), "refusing overwrite")
    args.output.mkdir(parents=True)

    cfg = load_config(args.config)
    require(cfg["folds"][3]["epochs"]["v260_uniform"] == EPOCHS, "epoch budget drift")
    cache, parts, _ = load_bundle(args.bundle, args.config)
    k = np.minimum(cache["exact"].astype(np.int32), 6)
    fit = np.asarray(parts["inner_fit"], np.int64)
    val = np.asarray(parts["inner_val"], np.int64)
    outer = np.asarray(parts["outer"], np.int64)
    require(len(fit) and len(val) and not np.intersect1d(fit,val).size, "inner split invalid")
    require(not np.intersect1d(val, outer).size and not np.intersect1d(fit, outer).size,
            "outer leakage into internal audit")

    order = hashlib.sha256()
    for e in range(EPOCHS):
        order.update(epoch_order(fit, SEED, e).tobytes())

    # 1) Internal uniform anchor.
    uniform = build_model("learned_gate", SEED)
    uniform_initial_hash = weight_hash(uniform)
    uniform_table = np.ones(7, np.float32)
    u_hist, u_order = train_one(uniform, cache, fit, k, uniform_table, args.output, "inner_uniform")
    u_base = nested_base(uniform)
    uniform.save_weights(args.output/"inner_uniform.weights.h5")

    u_prob = np.asarray(
        uniform.predict(batches(cache,val), workers=0, max_queue_size=1, verbose=0),
        np.float32,
    )
    require(u_prob.shape == (len(val),7) and np.allclose(u_prob.sum(1),1,atol=1e-5),
            "bad uniform val probabilities")
    U = u_prob.argmax(1).astype(np.int32)

    # Store anchor weights needed for later transplant before clearing anything.
    candidate_hidden1_anchor = [
        np.asarray(w).copy() for w in u_base.get_layer(TRANSPLANT_LAYER).get_weights()
    ]
    anchor_hash = weight_hash(uniform)

    # 2) Internal freeze_local_combo fine-tuning from the trained uniform anchor.
    freeze = build_model("learned_gate", SEED)
    freeze.load_weights(args.output/"inner_uniform.weights.h5")
    require(weight_hash(freeze) == anchor_hash, "internal anchor reload mismatch")
    f_base = nested_base(freeze)
    frozen_before = {}
    for name in FREEZE_LOCAL:
        layer = f_base.get_layer(name)
        frozen_before[name] = [np.asarray(w).copy() for w in layer.get_weights()]
        layer.trainable = False
    for layer in freeze.layers:
        if layer is not f_base:
            layer.trainable = True
    compile_model(freeze)

    weighted_table = class_weight_table(WEIGHT_ARM, k[fit])
    f_hist, f_order = train_one(freeze, cache, fit, k, weighted_table, args.output, "inner_freeze_local_combo")

    for name, before in frozen_before.items():
        after = f_base.get_layer(name).get_weights()
        require(len(before) == len(after) and all(np.array_equal(a,b) for a,b in zip(before,after)),
                "frozen layer changed: "+name)

    freeze.save_weights(args.output/"inner_freeze_local_combo.weights.h5")
    f_prob = np.asarray(
        freeze.predict(batches(cache,val), workers=0, max_queue_size=1, verbose=0),
        np.float32,
    )
    require(f_prob.shape == (len(val),7) and np.allclose(f_prob.sum(1),1,atol=1e-5),
            "bad freeze val probabilities")
    F = f_prob.argmax(1).astype(np.int32)

    drift = layer_delta(u_base, f_base, TRANSPLANT_LAYER)

    # 3) Post-train transplant candidate_hidden1 only.
    f_base.get_layer(TRANSPLANT_LAYER).set_weights(candidate_hidden1_anchor)
    transplanted = f_base.get_layer(TRANSPLANT_LAYER).get_weights()
    require(all(np.array_equal(a,b) for a,b in zip(candidate_hidden1_anchor,transplanted)),
            "candidate_hidden1 transplant failed")
    t_prob = np.asarray(
        freeze.predict(batches(cache,val), workers=0, max_queue_size=1, verbose=0),
        np.float32,
    )
    require(t_prob.shape == (len(val),7) and np.allclose(t_prob.sum(1),1,atol=1e-5),
            "bad transplant val probabilities")
    T = t_prob.argmax(1).astype(np.int32)

    y = k[val]
    um = metrics(y,U); fm = metrics(y,F); tm = metrics(y,T)
    pair_fu = transitions(y,U,F)
    pair_tu = transitions(y,U,T)
    pair_tf = transitions(y,F,T)

    class_net_f = {str(v): int(np.sum((F==y)&(y==v)) - np.sum((U==y)&(y==v))) for v in range(7)}
    class_net_t = {str(v): int(np.sum((T==y)&(y==v)) - np.sum((U==y)&(y==v))) for v in range(7)}

    # Transition-focused diagnostics.
    u_correct = U == y
    f_correct = F == y
    t_correct = T == y
    changed_uf = u_correct != f_correct
    changed_ut = u_correct != t_correct
    reg_uf = u_correct & ~f_correct
    cor_uf = ~u_correct & f_correct
    reg_ut = u_correct & ~t_correct
    cor_ut = ~u_correct & t_correct

    # Cluster disagreement rows using only prediction/probability signatures.
    cluster_mask = changed_uf | changed_ut
    idx = np.where(cluster_mask)[0]
    clustering = {"rows": int(len(idx)), "best_k": None, "best_silhouette": None, "clusters": []}
    if len(idx) >= 8:
        feats = np.concatenate([
            u_prob[idx], f_prob[idx], t_prob[idx],
            (f_prob[idx]-u_prob[idx]), (t_prob[idx]-u_prob[idx]),
            entropy(u_prob[idx])[:,None], entropy(f_prob[idx])[:,None], entropy(t_prob[idx])[:,None],
            prediction_margin(u_prob[idx])[:,None],
            prediction_margin(f_prob[idx])[:,None],
            prediction_margin(t_prob[idx])[:,None],
        ], axis=1)
        mu = feats.mean(axis=0); sd = feats.std(axis=0)
        x = (feats-mu)/np.where(sd>1e-8,sd,1.0)
        best = None
        for kk in range(2, min(5,len(idx)-1)+1):
            labels,_ = deterministic_kmeans(x,kk)
            sil = silhouette_score(x,labels)
            row = (float(sil) if sil is not None else -1.0, kk, labels)
            if best is None or row[0] > best[0]:
                best = row
        if best is not None:
            sil, kk, labels = best
            clustering["best_k"] = int(kk)
            clustering["best_silhouette"] = float(sil)
            for c in range(kk):
                local = labels == c
                rows = idx[local]
                mask = np.zeros(len(y),bool); mask[rows] = True
                clustering["clusters"].append({
                    "cluster": int(c),
                    "rows": int(local.sum()),
                    "true_k_hist": hist(y[mask]),
                    "low_k01_rows": int(np.sum(mask & (y<=1))),
                    "poly_rows": int(np.sum(mask & (y>=2))),
                    "uniform_correct": int(np.sum(mask & u_correct)),
                    "freeze_correct": int(np.sum(mask & f_correct)),
                    "transplant_correct": int(np.sum(mask & t_correct)),
                    "freeze_net_vs_uniform": int(np.sum(mask & f_correct)-np.sum(mask & u_correct)),
                    "transplant_net_vs_uniform": int(np.sum(mask & t_correct)-np.sum(mask & u_correct)),
                    "uniform_to_freeze": transition_hist(U,F,mask),
                    "uniform_to_transplant": transition_hist(U,T,mask),
                })

    confirm = {
        "transplant_beats_freeze_global": bool(tm["exact"] > fm["exact"]),
        "transplant_at_least_uniform_global": bool(tm["exact"] >= um["exact"]),
        "transplant_poly_at_least_freeze_minus_0p25pt": bool(tm["poly_exact"] >= fm["poly_exact"] - 0.0025),
        "transplant_poly_above_uniform": bool(tm["poly_exact"] > um["poly_exact"]),
    }
    confirm["strict_confirmed"] = bool(all(confirm.values()))

    report = {
        "status":"completed",
        "training_scope":"inner_fit only",
        "protocol":{
            "experiment":"v273_inner_candidate_hidden1_posttrain_transplant_audit",
            "fit":"inner_fit folds 1/2/4",
            "validation":"inner_val fold 0",
            "outer_fold_3_loaded":False,
            "uniform_epochs":EPOCHS,
            "weighted_finetune_epochs":EPOCHS,
            "seed":SEED,
            "weighting":WEIGHT_ARM,
            "freeze_local_combo":list(FREEZE_LOCAL),
            "posttrain_transplant":TRANSPLANT_LAYER,
            "epoch_order_sha256":order.hexdigest(),
            "automatic_promotion":False,
        },
        "initial_weight_hash":uniform_initial_hash,
        "uniform_anchor_hash":anchor_hash,
        "weight_table":weighted_table.tolist(),
        "candidate_hidden1_drift_before_transplant":drift,
        "metrics":{
            "uniform":um,
            "freeze_local_combo":fm,
            "posttrain_candidate_hidden1_transplant":tm,
        },
        "paired":{
            "freeze_vs_uniform":pair_fu,
            "transplant_vs_uniform":pair_tu,
            "transplant_vs_freeze":pair_tf,
        },
        "class_net_vs_uniform":{
            "freeze_local_combo":class_net_f,
            "posttrain_transplant":class_net_t,
        },
        "transition_audit":{
            "freeze_vs_uniform":{
                "regressions":int(reg_uf.sum()),"corrections":int(cor_uf.sum()),
                "regression_true_k":hist(y[reg_uf]),"correction_true_k":hist(y[cor_uf]),
                "regression_transitions":transition_hist(U,F,reg_uf),
                "correction_transitions":transition_hist(U,F,cor_uf),
            },
            "transplant_vs_uniform":{
                "regressions":int(reg_ut.sum()),"corrections":int(cor_ut.sum()),
                "regression_true_k":hist(y[reg_ut]),"correction_true_k":hist(y[cor_ut]),
                "regression_transitions":transition_hist(U,T,reg_ut),
                "correction_transitions":transition_hist(U,T,cor_ut),
            },
        },
        "clustering":clustering,
        "confirmation_rule":confirm,
        "history":{"uniform":u_hist,"freeze_local_combo":f_hist},
        "observed_epoch_order_sha256":{"uniform":u_order,"freeze_local_combo":f_order},
    }
    write_json(args.output/"report.json",report)
    np.savez_compressed(
        args.output/"predictions.npz",
        global_index=val,k=y,
        uniform_probability=u_prob,uniform_predicted=U,
        freeze_probability=f_prob,freeze_predicted=F,
        transplant_probability=t_prob,transplant_predicted=T,
    )

    lines=[
        "# Internal audit — post-train candidate_hidden1 transplant","",
        "**Outer fold 3 was not loaded.** Training used inner_fit (folds 1/2/4); all reported metrics below are on inner_val fold 0.","",
        "| Measure | Uniform anchor | freeze_local_combo | + post-train candidate_hidden1 transplant |",
        "|---|---:|---:|---:|",
        f"| exact global | {100*um['exact']:.3f}% | {100*fm['exact']:.3f}% | {100*tm['exact']:.3f}% |",
        f"| exact poly | {100*um['poly_exact']:.3f}% | {100*fm['poly_exact']:.3f}% | {100*tm['poly_exact']:.3f}% |",
    ]
    for v in range(6):
        lines.append(
            f"| K{v} exact | {100*um['by_k'][str(v)]['exact']:.2f}% | "
            f"{100*fm['by_k'][str(v)]['exact']:.2f}% | {100*tm['by_k'][str(v)]['exact']:.2f}% |"
        )
    lines += [
        "",
        f"Freeze vs uniform paired net: **{pair_fu['net_correct']:+d}**.",
        f"Transplant vs uniform paired net: **{pair_tu['net_correct']:+d}**.",
        f"Transplant vs freeze paired net: **{pair_tf['net_correct']:+d}**.",
        f"candidate_hidden1 relative L2 drift during weighted fine-tune: **{drift['relative_l2']:.6f}**.",
        "",
        f"Disagreement clustering: k={clustering['best_k']}, silhouette={clustering['best_silhouette']}.",
        f"Strict internal confirmation: **{confirm['strict_confirmed']}**.",
        "",
        "No outer evaluation, no promotion."
    ]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
