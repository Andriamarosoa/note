"""Train one audit-justified arm: extend freeze_local_combo with candidate_hidden1.

Starts from the exact frozen uniform learned_gate checkpoint, applies the same
targeted K2/K3-half + K4-quarter weighting for 8 epochs, and freezes:
candidate_norm + candidate_hidden1 + candidate_hidden2 + cardinality.

This is the minimal training intervention justified by residual transplant run
37245028798. Outer fold 3 is evaluation-only. No automatic promotion.
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

ARM="freeze_candidate_stack"
WEIGHT_ARM="targeted_k23_half_k4_quarter"
FROZEN=("candidate_norm","candidate_hidden1","candidate_hidden2","cardinality")
SEED=v260.SEED+1003
EPOCHS=8

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,f"expected one nested base, got {[x.name for x in xs]}")
    return xs[0]

def load_cluster_a(path,global_index):
    rows=list(csv.DictReader(open(path,newline="")));by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0});d["n"]+=1;d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return int(cid),np.isin(np.asarray(global_index,np.int64),list(ids))

def load_reference(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

def main(args):
    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),"refusing overwrite")
    args.output.mkdir(parents=True)

    cfg=load_config(args.config)
    require(cfg["folds"][3]["epochs"]["v260_uniform"]==EPOCHS,"epoch budget drift")
    cache,parts,_=load_bundle(args.bundle,args.config)
    k=np.minimum(cache["exact"].astype(np.int32),6)
    fit=np.asarray(parts["final_fit"],np.int64)
    outer=np.asarray(parts["outer"],np.int64)
    require(not np.intersect1d(fit,outer).size and len(outer)==15279,"partition drift")

    uniform=load_reference(args.uniform_predictions)
    current=load_reference(args.current_predictions)
    np.testing.assert_array_equal(uniform["global_index"],outer)
    np.testing.assert_array_equal(current["global_index"],outer)
    np.testing.assert_array_equal(uniform["k"],k[outer])
    np.testing.assert_array_equal(current["k"],k[outer])
    upred=uniform["predicted"].astype(np.int32)
    cpred=current["freeze_local_combo_predicted"].astype(np.int32)

    cid,A=load_cluster_a(args.cluster_rows,outer)
    require(int(A.sum())==499,"Cluster A drift")

    model=build_model("learned_gate",SEED)
    model.load_weights(args.uniform_anchor)
    anchor_hash=weight_hash(model)
    base=nested_base(model)

    frozen_before={}
    for name in FROZEN:
        layer=base.get_layer(name)
        frozen_before[name]=[np.asarray(w).copy() for w in layer.get_weights()]
        layer.trainable=False
    for layer in model.layers:
        if layer is not base: layer.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss="sparse_categorical_crossentropy")

    table=class_weight_table(WEIGHT_ARM,k[fit])
    seq=batches(cache,fit,seed=SEED,shuffle=True,k=k,sample_table=table)
    order=hashlib.sha256()
    for e in range(EPOCHS):order.update(epoch_order(fit,SEED,e).tobytes())

    hist=[];observed=[]
    class CB(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            exp=epoch_order(fit,SEED,epoch)
            np.testing.assert_array_equal(seq.order,exp)
            observed.append(array_hash(seq.order))
        def on_epoch_end(self,epoch,logs=None):
            hist.append({"epoch":int(epoch)+1,**{x:float(y) for x,y in (logs or {}).items()}})
            write_json(args.output/"progress.json",{"completed_epoch":int(epoch)+1,"history":hist})

    res=model.fit(seq,epochs=EPOCHS,shuffle=False,workers=0,max_queue_size=1,verbose=2,
                  callbacks=[tf.keras.callbacks.TerminateOnNaN(),CB()])
    require(len(hist)==EPOCHS and all(np.isfinite(v).all() for v in res.history.values()),"training failure")

    for name,before in frozen_before.items():
        after=base.get_layer(name).get_weights()
        require(len(before)==len(after) and all(np.array_equal(a,b) for a,b in zip(before,after)),
                "frozen layer changed: "+name)

    prob=np.asarray(model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),np.float32)
    require(prob.shape==(len(outer),7) and np.allclose(prob.sum(1),1,atol=1e-5),"bad probabilities")
    pred=prob.argmax(1).astype(np.int32)
    m=metrics(k[outer],pred); ma=metrics(k[outer][A],pred[A])
    model.save_weights(args.output/f"{ARM}.weights.h5")

    report={
      "status":"completed",
      "protocol":{
        "experiment":"v273_uniform_anchored_candidate_stack_freeze",
        "evidence_run":37245028798,
        "anchor_run":37208199201,
        "current_reference_run":37233793945,
        "weighting":WEIGHT_ARM,
        "frozen_layers":list(FROZEN),
        "seed":SEED,"epochs":EPOCHS,
        "epoch_order_sha256":order.hexdigest(),
        "outer_used_for_selection":False,
        "automatic_promotion":False,
      },
      "anchor_hash":anchor_hash,
      "metrics":m,
      "cluster_A_metrics":ma,
      "paired_vs_uniform":transitions(k[outer],upred,pred),
      "paired_vs_current_freeze_local_combo":transitions(k[outer],cpred,pred),
      "current_reference_metrics":metrics(k[outer],cpred),
      "uniform_reference_metrics":metrics(k[outer],upred),
      "history":hist,
      "observed_epoch_order_sha256":observed,
      "weights_sha256":digest(args.output/f"{ARM}.weights.h5"),
    }
    write_json(args.output/"report.json",report)
    np.savez_compressed(args.output/"predictions.npz",
      global_index=outer,k=k[outer],probability=prob,predicted=pred,
      uniform_predicted=upred,current_freeze_local_combo_predicted=cpred)

    u=report["uniform_reference_metrics"];c=report["current_reference_metrics"]
    lines=[
      "# V27.3 candidate-stack freeze training","",
      "Audit-justified intervention: freeze candidate_norm + candidate_hidden1 + candidate_hidden2 + cardinality.","",
      "| Measure | Uniform | Current freeze_local_combo | New candidate-stack freeze |",
      "|---|---:|---:|---:|",
      f"| exact global | {100*u['exact']:.3f}% | {100*c['exact']:.3f}% | {100*m['exact']:.3f}% |",
      f"| exact poly | {100*u['poly_exact']:.3f}% | {100*c['poly_exact']:.3f}% | {100*m['poly_exact']:.3f}% |",
    ]
    for v in range(6):
        lines.append(f"| K{v} exact | {100*u['by_k'][str(v)]['exact']:.2f}% | {100*c['by_k'][str(v)]['exact']:.2f}% | {100*m['by_k'][str(v)]['exact']:.2f}% |")
    lines += [
      f"| Cluster A exact | — | — | {100*ma['exact']:.2f}% |","",
      f"Paired vs uniform net: {report['paired_vs_uniform']['net_correct']:+d}.",
      f"Paired vs current freeze_local_combo net: {report['paired_vs_current_freeze_local_combo']['net_correct']:+d}.",
      "No automatic promotion."
    ]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("bundle","config","uniform-anchor","uniform-predictions","current-predictions","cluster-rows","output"):
        p.add_argument("--"+n,type=Path,required=True)
    main(p.parse_args())
