"""Matched A/B: always-on V8.8 count evidence vs learned group gate.

Both arms share the same extra gate network and identical trainable parameter
count. In the control arm, the gate is connected through a zero-effect path so
the historical V8.8 count evidence is preserved exactly. In the treatment arm,
the same gate scales only the V8.8 pseudo-count interface:
  * per-candidate cluster_router + local_cardinality[0:4]
  * derived group stats router_mean/local_count_mean/local_count_weighted/local_nonzero

All remaining candidate, birth, geometry and spectral evidence is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v90_structured_cluster_cardinality import FROZEN_CANDIDATE_DIM
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, batch_inputs, epoch_order, array_hash, require

ARMS = ("always_on", "learned_gate")
FRAMES = 31
BATCH_SIZE = 128


def weight_hash(model):
    h = hashlib.sha256()
    for a in model.get_weights():
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def build_model(arm: str, seed: int, *, spectral_evidence="none"):
    import tensorflow as tf
    if arm not in ARMS:
        raise ValueError(arm)
    base = v260.build_model("uniform", seed, time_frames=FRAMES,
                           spectral_evidence=spectral_evidence)
    if FROZEN_CANDIDATE_DIM != V88_FEATURE_DIM + 8:
        raise RuntimeError("unexpected V8.8 feature layout")

    by_name = {x.name.split(":")[0]: x for x in base.inputs}
    cand = by_name["candidate_set"]
    mask = by_name["candidate_mask"]
    stats = by_name["cluster_stats"]
    spectral = by_name["spectral_map"]

    # Gate uses only non-pseudocount group evidence:
    # candidate_count_log, group_width, fused_mean, fused_max.
    gate_h = tf.keras.layers.Dense(
        12, activation="relu", name="v273_gate_hidden",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 71),
    )(stats[:, :4])
    gate = tf.keras.layers.Dense(
        1, activation="sigmoid", name="v273_gate_value",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 72),
        bias_initializer=tf.keras.initializers.Zeros(),
    )(gate_h)

    # Keep the gate network connected in both arms so parameter counts match.
    if arm == "always_on":
        effective_gate = tf.keras.layers.Lambda(
            lambda g: tf.ones_like(g) + 0.0 * g,
            name="v273_gate_control_passthrough",
        )(gate)
    else:
        effective_gate = gate

    g3 = tf.keras.layers.Lambda(
        lambda g: tf.expand_dims(g, axis=1),
        name="v273_gate_expand",
    )(effective_gate)

    left = cand[:, :, :V88_FEATURE_DIM]
    pseudo = cand[:, :, V88_FEATURE_DIM:V88_FEATURE_DIM + 5]
    right = cand[:, :, V88_FEATURE_DIM + 5:]
    gated_pseudo = tf.keras.layers.Multiply(name="v273_gate_candidate_pseudocount")([pseudo, g3])
    gated_cand = tf.keras.layers.Lambda(
        lambda z: tf.concat(z, axis=-1),
        name="v273_gate_candidate_concat",
    )([left, gated_pseudo, right])

    preserved = stats[:, :4]
    pseudo_stats = stats[:, 4:8]
    gated_stats = tf.keras.layers.Multiply(name="v273_gate_stats_pseudocount")([pseudo_stats, effective_gate])
    merged_stats = tf.keras.layers.Lambda(
        lambda z: tf.concat(z, axis=-1),
        name="v273_gate_stats_concat",
    )([preserved, gated_stats])

    out = base({
        "candidate_set": gated_cand,
        "candidate_mask": mask,
        "cluster_stats": merged_stats,
        "spectral_map": spectral,
    })
    model = tf.keras.Model(base.inputs, out, name="v273_group_gate_" + arm)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="sparse_categorical_crossentropy",
    )
    return model


def batches(cache, indices, *, seed=0, shuffle=False, k=None, batch_size=BATCH_SIZE):
    import tensorflow as tf
    ids0 = np.asarray(indices, np.int64)
    require(len(ids0) > 0 and len(np.unique(ids0)) == len(ids0), "invalid population")

    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch = 0
            self.order = epoch_order(ids0, seed, 0) if shuffle else ids0.copy()
        def __len__(self):
            return (len(ids0) + batch_size - 1) // batch_size
        def __getitem__(self, batch):
            ids = self.order[batch*batch_size:(batch+1)*batch_size]
            x = batch_inputs(cache, ids, FRAMES)
            return x if k is None else (x, k[ids])
        def on_epoch_end(self):
            self.epoch += 1
            if shuffle:
                self.order = epoch_order(ids0, seed, self.epoch)
    return Batches()


def metrics(k, pred):
    k=np.asarray(k); pred=np.asarray(pred)
    poly=k>=2
    out={
        "rows":int(len(k)),
        "exact":float(np.mean(k==pred)),
        "poly_exact":float(np.mean(k[poly]==pred[poly])) if poly.any() else None,
        "under":int(np.sum(pred<k)),
        "over":int(np.sum(pred>k)),
        "by_k":{},
    }
    for v in range(7):
        take=k==v
        out["by_k"][str(v)]={
            "rows":int(take.sum()),
            "exact":float(np.mean(pred[take]==v)) if take.any() else None,
            "under":int(np.sum(pred[take]<v)),
            "over":int(np.sum(pred[take]>v)),
        }
    return out


def transitions(k,a,b):
    k=np.asarray(k);a=np.asarray(a);b=np.asarray(b)
    return {
        "corrected":int(np.sum((a!=k)&(b==k))),
        "regressed":int(np.sum((a==k)&(b!=k))),
        "under_to_correct":int(np.sum((a<k)&(b==k))),
        "over_to_correct":int(np.sum((a>k)&(b==k))),
        "correct_to_under":int(np.sum((a==k)&(b<k))),
        "correct_to_over":int(np.sum((a==k)&(b>k))),
        "net_correct":int(np.sum(b==k)-np.sum(a==k)),
    }


def load_cluster_a(path, global_index):
    rows=list(csv.DictReader(open(path,newline="")))
    by={}
    for r in rows:
        c=int(r["cluster"]); y=int(r["true_k"]); p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0})
        d["n"]+=1; d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    mask=np.isin(np.asarray(global_index,np.int64),list(ids))
    return cid,mask


def train(args):
    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),"refusing to overwrite")

    cfg=load_config(args.config)
    budget=cfg["folds"][3]["epochs"]["v260_uniform"]
    require(budget==8,"budget changed")
    cache,parts,_=load_bundle(args.bundle,args.config)
    k=np.minimum(cache["exact"].astype(np.int32),6)
    fit=parts["final_fit"]; outer=parts["outer"]
    require(len(outer)==15279 and not np.intersect1d(fit,outer).size,"partition drift")
    args.output.mkdir(parents=True)

    seed=v260.SEED+1003
    initial={}
    for arm in ARMS:
        m=build_model(arm,seed)
        initial[arm]={
            "hash":weight_hash(m),
            "params":int(np.sum([np.prod(v.shape) for v in m.trainable_variables]))
        }
        del m; tf.keras.backend.clear_session()
    require(initial["always_on"]["hash"]==initial["learned_gate"]["hash"],"initial weight mismatch")
    require(initial["always_on"]["params"]==initial["learned_gate"]["params"],"parameter mismatch")

    order=hashlib.sha256()
    for e in range(budget):
        order.update(epoch_order(fit,seed,e).tobytes())

    cid,A=load_cluster_a(args.cluster_rows,outer)
    require(int(A.sum())==499,"Cluster A drift")

    report={
        "status":"training",
        "protocol":{
            "experiment":"v273_group_gate_ab",
            "arms":list(ARMS),
            "frames":31,
            "weighting":"uniform",
            "epochs":8,
            "seed":seed,
            "batch_size":128,
            "epoch_order_sha256":order.hexdigest(),
            "only_treatment":"learned multiplicative trust gate on V8.8 router/local-cardinality pseudo-count interface",
            "gate_inputs":"candidate_count_log, group_width, fused_mean, fused_max only",
            "same_initial_weights":True,
            "same_trainable_parameter_count":True,
            "outer_used_for_selection":False,
            "automatic_promotion":False,
        },
        "initial":initial,
        "cluster_A_id":cid,
        "cluster_A_rows":499,
        "arms":{},
    }
    write_json(args.output/"protocol.json",report["protocol"])

    preds={
        "global_index":np.asarray(outer,np.int64),
        "member":np.asarray(cache["members"][outer]),
        "cluster_start_samples":np.asarray(cache["cluster_start_samples"][outer]),
        "k":k[outer],
    }

    for arm in ARMS:
        tf.keras.backend.clear_session()
        model=build_model(arm,seed)
        require(weight_hash(model)==initial[arm]["hash"],"initialization drift")
        seq=batches(cache,fit,seed=seed,shuffle=True,k=k)
        hist_rows=[]; observed=[]
        class CB(tf.keras.callbacks.Callback):
            def on_epoch_begin(self,epoch,logs=None):
                expected=epoch_order(fit,seed,epoch)
                np.testing.assert_array_equal(seq.order,expected)
                observed.append(array_hash(seq.order))
            def on_epoch_end(self,epoch,logs=None):
                row={"epoch":int(epoch)+1,**{x:float(y) for x,y in (logs or {}).items()}}
                hist_rows.append(row)
                write_json(args.output/f"{arm}-progress.json",{
                    "arm":arm,"completed_epoch":int(epoch)+1,"history":hist_rows
                })

        hist=model.fit(
            seq,epochs=budget,shuffle=False,workers=0,max_queue_size=1,verbose=2,
            callbacks=[tf.keras.callbacks.TerminateOnNaN(),CB()]
        )
        require(len(hist_rows)==budget and all(np.isfinite(v).all() for v in hist.history.values()),"training failure")
        prob=np.asarray(model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),np.float32)
        require(prob.shape==(len(outer),7) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5),"bad probabilities")
        pred=prob.argmax(1).astype(np.int32)

        # Inspect learned gate values on held-out outer rows.
        gate_model=tf.keras.Model(model.inputs,model.get_layer("v273_gate_value").output)
        gate=np.asarray(gate_model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0)).reshape(-1)

        model.save_weights(args.output/f"{arm}.weights.h5")
        report["arms"][arm]={
            "metrics":metrics(k[outer],pred),
            "cluster_A_metrics":metrics(k[outer][A],pred[A]),
            "history":hist_rows,
            "gate_summary":{
                "mean":float(gate.mean()),
                "median":float(np.median(gate)),
                "p10":float(np.percentile(gate,10)),
                "p90":float(np.percentile(gate,90)),
                "by_k":{str(v):float(gate[k[outer]==v].mean()) for v in (2,3,4)},
                "cluster_A_mean":float(gate[A].mean()),
            },
            "observed_epoch_order_sha256":observed,
            "weights_sha256":digest(args.output/f"{arm}.weights.h5"),
        }
        preds[f"{arm}_probability"]=prob
        preds[f"{arm}_predicted"]=pred
        preds[f"{arm}_gate"]=gate.astype(np.float32)
        write_json(args.output/"report.json",report)
        np.savez_compressed(args.output/"predictions.npz",**preds)
        del model,gate_model,seq,hist
        tf.keras.backend.clear_session()

    a=preds["always_on_predicted"]; b=preds["learned_gate_predicted"]
    report["status"]="completed"
    report["paired"]={
        "all":transitions(k[outer],a,b),
        "poly":transitions(k[outer][k[outer]>=2],a[k[outer]>=2],b[k[outer]>=2]),
        "cluster_A":transitions(k[outer][A],a[A],b[A]),
    }
    write_json(args.output/"report.json",report)

    bm=report["arms"]["always_on"]["metrics"]
    gm=report["arms"]["learned_gate"]["metrics"]
    ba=report["arms"]["always_on"]["cluster_A_metrics"]
    ga=report["arms"]["learned_gate"]["cluster_A_metrics"]
    lines=[
        "# V27.3 Learned Gate A/B","",
        "| Mesure | Always-on | Learned gate | Delta |",
        "|---|---:|---:|---:|",
        f"| exact global | {100*bm['exact']:.3f}% | {100*gm['exact']:.3f}% | {100*(gm['exact']-bm['exact']):+.3f} pt |",
        f"| exact poly | {100*bm['poly_exact']:.3f}% | {100*gm['poly_exact']:.3f}% | {100*(gm['poly_exact']-bm['poly_exact']):+.3f} pt |",
        f"| under | {bm['under']} | {gm['under']} | {gm['under']-bm['under']:+d} |",
        f"| over | {bm['over']} | {gm['over']} | {gm['over']-bm['over']:+d} |",
    ]
    for v in (2,3,4):
        x=bm["by_k"][str(v)]["exact"]; y=gm["by_k"][str(v)]["exact"]
        lines.append(f"| K{v} exact | {100*x:.3f}% | {100*y:.3f}% | {100*(y-x):+.3f} pt |")
    lines += [
        "",
        f"Cluster A exact: {100*ba['exact']:.2f}% → {100*ga['exact']:.2f}%.",
        f"Cluster A under: {ba['under']} → {ga['under']}; over: {ba['over']} → {ga['over']}.",
        f"Paired global net: {report['paired']['all']['net_correct']:+d}.",
        f"Paired Cluster A net: {report['paired']['cluster_A']['net_correct']:+d}.",
        "",
        f"Gate mean outer: {report['arms']['learned_gate']['gate_summary']['mean']:.3f}; "
        f"K2 {report['arms']['learned_gate']['gate_summary']['by_k']['2']:.3f}, "
        f"K3 {report['arms']['learned_gate']['gate_summary']['by_k']['3']:.3f}, "
        f"K4 {report['arms']['learned_gate']['gate_summary']['by_k']['4']:.3f}, "
        f"A {report['arms']['learned_gate']['gate_summary']['cluster_A_mean']:.3f}.",
        "",
        "Aucune promotion automatique.",
    ]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--cluster-rows",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    train(p.parse_args())
