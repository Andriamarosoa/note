"""Matched loss-weighting A/B for V27.3 learned_gate.

No architecture change. The only intervention is the training loss sample weight:
- uniform: weight 1 for every training row.
- sqrt_balanced: inverse-square-root class-frequency weights derived only from final_fit.

Each arm is trained independently with identical initialization, seed, fold3 split,
epoch order, 31 frames and 8 epochs. Outer fold is evaluation-only.
"""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v273_group_gate_ab import (
    FRAMES, BATCH_SIZE, build_model, weight_hash, metrics, load_cluster_a
)
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import (
    load_bundle, batch_inputs, epoch_order, array_hash, require
)

ARMS=("uniform","sqrt_balanced")


def class_weight_table(arm, fit_k):
    if arm=="uniform":
        return np.ones(7,np.float32)
    if arm=="sqrt_balanced":
        return v260.class_weights(np.asarray(fit_k,np.int32))
    raise ValueError(arm)


def batches(cache, indices, *, seed=0, shuffle=False, k=None, sample_table=None,
            batch_size=BATCH_SIZE):
    import tensorflow as tf
    ids0=np.asarray(indices,np.int64)
    require(len(ids0)>0 and len(np.unique(ids0))==len(ids0),"invalid population")

    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch=0
            self.order=epoch_order(ids0,seed,0) if shuffle else ids0.copy()
        def __len__(self):
            return (len(ids0)+batch_size-1)//batch_size
        def __getitem__(self,batch):
            ids=self.order[batch*batch_size:(batch+1)*batch_size]
            x=batch_inputs(cache,ids,FRAMES)
            if k is None:
                return x
            y=k[ids]
            if sample_table is None:
                return x,y
            return x,y,np.asarray(sample_table[y],np.float32)
        def on_epoch_end(self):
            self.epoch+=1
            if shuffle:
                self.order=epoch_order(ids0,seed,self.epoch)
    return Batches()


def run_arm(args):
    import tensorflow as tf
    require(args.arm in ARMS,"unknown arm")
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),"refusing to overwrite")

    cfg=load_config(args.config)
    budget=cfg["folds"][3]["epochs"]["v260_uniform"]
    require(budget==8,"budget changed")
    cache,parts,_=load_bundle(args.bundle,args.config)
    k=np.minimum(cache["exact"].astype(np.int32),6)
    fit=np.asarray(parts["final_fit"],np.int64)
    outer=np.asarray(parts["outer"],np.int64)
    require(len(outer)==15279 and not np.intersect1d(fit,outer).size,"partition drift")
    args.output.mkdir(parents=True)

    seed=v260.SEED+1003
    model=build_model("learned_gate",seed)
    initial_hash=weight_hash(model)
    params=int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    table=class_weight_table(args.arm,k[fit])

    order=hashlib.sha256()
    for e in range(budget):
        order.update(epoch_order(fit,seed,e).tobytes())

    cid,A=load_cluster_a(args.cluster_rows,outer)
    require(int(A.sum())==499,"Cluster A drift")

    seq=batches(cache,fit,seed=seed,shuffle=True,k=k,sample_table=table)
    hist_rows=[]; observed=[]

    class CB(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            expected=epoch_order(fit,seed,epoch)
            np.testing.assert_array_equal(seq.order,expected)
            observed.append(array_hash(seq.order))
        def on_epoch_end(self,epoch,logs=None):
            row={"epoch":int(epoch)+1,**{x:float(y) for x,y in (logs or {}).items()}}
            hist_rows.append(row)
            write_json(args.output/"progress.json",{
                "arm":args.arm,"completed_epoch":int(epoch)+1,
                "history":hist_rows,
            })

    hist=model.fit(
        seq,epochs=budget,shuffle=False,workers=0,max_queue_size=1,verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),CB()]
    )
    require(len(hist_rows)==budget and all(np.isfinite(v).all() for v in hist.history.values()),
            "training failure")

    prob=np.asarray(
        model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),
        np.float32
    )
    require(prob.shape==(len(outer),7) and np.isfinite(prob).all()
            and np.allclose(prob.sum(1),1,atol=1e-5),"bad probabilities")
    pred=prob.argmax(1).astype(np.int32)

    gate_model=tf.keras.Model(model.inputs,model.get_layer("v273_gate_value").output)
    gate=np.asarray(
        gate_model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0)
    ).reshape(-1)

    model.save_weights(args.output/f"{args.arm}.weights.h5")
    report={
        "status":"completed",
        "protocol":{
            "experiment":"v273_learned_gate_loss_weighting_ab",
            "arm":args.arm,
            "architecture":"learned_gate unchanged",
            "frames":FRAMES,
            "epochs":budget,
            "seed":seed,
            "batch_size":BATCH_SIZE,
            "epoch_order_sha256":order.hexdigest(),
            "only_intervention":"training sample-weight table",
            "weight_table":table.tolist(),
            "weights_derived_from":"final_fit labels only",
            "outer_used_for_selection":False,
            "automatic_promotion":False,
        },
        "initial":{"hash":initial_hash,"params":params},
        "metrics":metrics(k[outer],pred),
        "cluster_A_id":cid,
        "cluster_A_rows":499,
        "cluster_A_metrics":metrics(k[outer][A],pred[A]),
        "history":hist_rows,
        "gate_summary":{
            "mean":float(gate.mean()),
            "by_k":{str(v):float(gate[k[outer]==v].mean()) for v in (2,3,4)},
            "cluster_A_mean":float(gate[A].mean()),
        },
        "observed_epoch_order_sha256":observed,
        "weights_sha256":digest(args.output/f"{args.arm}.weights.h5"),
    }
    write_json(args.output/"report.json",report)
    np.savez_compressed(
        args.output/"predictions.npz",
        global_index=outer,k=k[outer],probability=prob,predicted=pred,gate=gate.astype(np.float32)
    )

    m=report["metrics"]
    lines=[
        f"# V27.3 learned_gate loss weighting — {args.arm}","",
        f"- exact global: {100*m['exact']:.3f}%",
        f"- exact poly: {100*m['poly_exact']:.3f}%",
        f"- K2 exact: {100*m['by_k']['2']['exact']:.3f}%",
        f"- K3 exact: {100*m['by_k']['3']['exact']:.3f}%",
        f"- K4 exact: {100*m['by_k']['4']['exact']:.3f}%",
        f"- under: {m['under']}",
        f"- over: {m['over']}",
        f"- Cluster A exact: {100*report['cluster_A_metrics']['exact']:.3f}%",
        "",
        "Aucune promotion automatique.",
    ]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--cluster-rows",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run_arm(p.parse_args())
