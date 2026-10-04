"""Matched anchored fine-tuning A/B after the transplant audits.

All arms start from the exact same frozen uniform learned_gate checkpoint and
fine-tune for the same 8 epochs with the same targeted K2/K3-half + K4-quarter
sample weights and epoch order.

Arms:
- all_trainable: all layers remain trainable.
- freeze_candidate_hidden2: freeze only candidate_hidden2 at the uniform anchor.
- freeze_local_combo: freeze candidate_norm + candidate_hidden2 + cardinality
  at their uniform-anchor values.

Outer fold 3 is evaluation-only. No automatic promotion.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest,write_json
from scripts.train_v273_group_gate_ab import build_model,weight_hash,metrics,transitions
from scripts.train_v273_loss_weighting_ab import class_weight_table,batches
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,epoch_order,array_hash,require

ARMS=("all_trainable","freeze_candidate_hidden2","freeze_local_combo")
WEIGHT_ARM="targeted_k23_half_k4_quarter"
FREEZE={
    "all_trainable":(),
    "freeze_candidate_hidden2":("candidate_hidden2",),
    "freeze_local_combo":("candidate_norm","candidate_hidden2","cardinality"),
}
SEED=v260.SEED+1003
BATCH=128

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,f"expected one nested base, got {[x.name for x in xs]}")
    return xs[0]

def configure_trainability(model,arm):
    base=nested_base(model)
    frozen=set(FREEZE[arm])
    for layer in base.layers:
        layer.trainable=layer.name not in frozen
    for layer in model.layers:
        if layer is not base:
            layer.trainable=True
    return base

def compile_model(model):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss="sparse_categorical_crossentropy")

def trainable_hash(model):
    h=hashlib.sha256()
    for v in model.trainable_variables:
        h.update(v.name.encode());h.update(str(tuple(v.shape)).encode())
    return h.hexdigest()

def load_cluster_a(path,global_index):
    rows=list(csv.DictReader(open(path,newline=""))); by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0});d["n"]+=1;d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,np.isin(np.asarray(global_index,np.int64),list(ids))

def train(args):
    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),"refusing overwrite")
    args.output.mkdir(parents=True)

    cfg=load_config(args.config)
    budget=cfg["folds"][3]["epochs"]["v260_uniform"]
    require(budget==8,"budget drift")
    cache,parts,_=load_bundle(args.bundle,args.config)
    k=np.minimum(cache["exact"].astype(np.int32),6)
    fit=np.asarray(parts["final_fit"],np.int64)
    outer=np.asarray(parts["outer"],np.int64)
    require(not np.intersect1d(fit,outer).size and len(outer)==15279,"partition drift")
    table=class_weight_table(WEIGHT_ARM,k[fit])

    cid,A=load_cluster_a(args.cluster_rows,outer)
    require(int(A.sum())==499,"Cluster A drift")

    anchor=build_model("learned_gate",SEED)
    anchor.load_weights(args.uniform_anchor)
    anchor_hash=weight_hash(anchor)
    anchor_prob=np.asarray(anchor.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),np.float32)
    anchor_pred=anchor_prob.argmax(1).astype(np.int32)
    anchor_metrics=metrics(k[outer],anchor_pred)

    # The anchor must exactly replay the frozen uniform predictions.
    with np.load(args.uniform_predictions,allow_pickle=False) as z:
        saved={x:np.asarray(z[x]) for x in z.files}
    np.testing.assert_array_equal(saved["global_index"],outer)
    np.testing.assert_array_equal(saved["k"],k[outer])
    np.testing.assert_array_equal(saved["predicted"].astype(np.int32),anchor_pred)
    del anchor; tf.keras.backend.clear_session()

    order=hashlib.sha256()
    for e in range(budget):
        order.update(epoch_order(fit,SEED,e).tobytes())

    report={
        "status":"training",
        "protocol":{
            "experiment":"v273_uniform_anchored_targeted_finetune",
            "anchor":"frozen uniform learned_gate run 37208199201",
            "weighting":WEIGHT_ARM,
            "arms":list(ARMS),
            "freeze":{k:list(v) for k,v in FREEZE.items()},
            "epochs":budget,"seed":SEED,"batch_size":BATCH,
            "epoch_order_sha256":order.hexdigest(),
            "same_anchor_checkpoint":True,
            "same_weighting":True,
            "same_epoch_order":True,
            "outer_used_for_selection":False,
            "automatic_promotion":False,
        },
        "anchor_hash":anchor_hash,
        "anchor_metrics":anchor_metrics,
        "cluster_A_id":int(cid),
        "arms":{},
    }
    write_json(args.output/"protocol.json",report["protocol"])

    pred_out={"global_index":outer,"k":k[outer],"uniform_anchor_predicted":anchor_pred}
    reference=None

    for arm in ARMS:
        tf.keras.backend.clear_session()
        model=build_model("learned_gate",SEED)
        model.load_weights(args.uniform_anchor)
        require(weight_hash(model)==anchor_hash,"anchor mismatch")
        base=configure_trainability(model,arm)
        compile_model(model)

        frozen_before={name:[np.asarray(x).copy() for x in base.get_layer(name).get_weights()]
                       for name in FREEZE[arm]}
        trainable_params=int(sum(np.prod(v.shape) for v in model.trainable_variables))
        trainable_inventory=trainable_hash(model)

        seq=batches(cache,fit,seed=SEED,shuffle=True,k=k,sample_table=table)
        hist=[]; observed=[]
        class CB(tf.keras.callbacks.Callback):
            def on_epoch_begin(self,epoch,logs=None):
                exp=epoch_order(fit,SEED,epoch)
                np.testing.assert_array_equal(seq.order,exp)
                observed.append(array_hash(seq.order))
            def on_epoch_end(self,epoch,logs=None):
                hist.append({"epoch":int(epoch)+1,**{x:float(y) for x,y in (logs or {}).items()}})
                write_json(args.output/f"{arm}-progress.json",{
                    "arm":arm,"completed_epoch":int(epoch)+1,"history":hist
                })

        res=model.fit(seq,epochs=budget,shuffle=False,workers=0,max_queue_size=1,verbose=2,
                      callbacks=[tf.keras.callbacks.TerminateOnNaN(),CB()])
        require(len(hist)==budget and all(np.isfinite(v).all() for v in res.history.values()),"training failure")

        for name,before in frozen_before.items():
            after=base.get_layer(name).get_weights()
            require(len(before)==len(after) and all(np.array_equal(a,b) for a,b in zip(before,after)),
                    "frozen layer changed: "+name)

        prob=np.asarray(model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),np.float32)
        require(prob.shape==(len(outer),7) and np.allclose(prob.sum(1),1,atol=1e-5),"bad probabilities")
        pred=prob.argmax(1).astype(np.int32)
        m=metrics(k[outer],pred)
        ma=metrics(k[outer][A],pred[A])
        model.save_weights(args.output/f"{arm}.weights.h5")

        rec={
            "metrics":m,"cluster_A_metrics":ma,"history":hist,
            "trainable_params":trainable_params,
            "trainable_inventory_sha256":trainable_inventory,
            "observed_epoch_order_sha256":observed,
            "weights_sha256":digest(args.output/f"{arm}.weights.h5"),
            "frozen_layers":list(FREEZE[arm]),
        }
        if reference is None:
            reference=pred
        else:
            rec["paired_vs_all_trainable"]=transitions(k[outer],reference,pred)
        rec["paired_vs_uniform_anchor"]=transitions(k[outer],anchor_pred,pred)
        report["arms"][arm]=rec
        pred_out[f"{arm}_probability"]=prob
        pred_out[f"{arm}_predicted"]=pred
        write_json(args.output/"report.json",report)
        np.savez_compressed(args.output/"predictions.npz",**pred_out)

        del model,base,seq,res
        tf.keras.backend.clear_session()

    report["status"]="completed"
    write_json(args.output/"report.json",report)

    u=anchor_metrics
    lines=[
        "# V27.3 uniform-anchored targeted fine-tuning","",
        "All arms start from the exact same trained uniform learned_gate checkpoint, then receive the same weighted loss for 8 epochs.","",
        "| Arm | Global | Δ vs uniform | Poly | Δ vs uniform poly | K0 | K1 | K2 | K3 | K4 | K5 | Cluster A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m=report["arms"][arm]["metrics"]; a=report["arms"][arm]["cluster_A_metrics"]
        def ex(v):
            x=m["by_k"][str(v)]["exact"]; return 100*x
        lines.append(
            f"| {arm} | {100*m['exact']:.3f}% | {100*(m['exact']-u['exact']):+.3f} pt | "
            f"{100*m['poly_exact']:.3f}% | {100*(m['poly_exact']-u['poly_exact']):+.3f} pt | "
            f"{ex(0):.2f}% | {ex(1):.2f}% | {ex(2):.2f}% | {ex(3):.2f}% | {ex(4):.2f}% | {ex(5):.2f}% | {100*a['exact']:.2f}% |"
        )
    lines += ["","No automatic promotion. Outer fold is evaluation-only."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("bundle","config","uniform-anchor","uniform-predictions","cluster-rows","output"):
        p.add_argument("--"+n,type=Path,required=True)
    train(p.parse_args())
