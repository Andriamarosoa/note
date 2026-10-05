"""Rotating inner-fold audit of candidate_hidden1 neurons.

Outer fold 3 is excluded from both fit and validation.

For a requested validation fold in {0,1,2,4}:
- fit folds = the other three internal folds;
- train uniform learned_gate for 8 epochs;
- fine-tune from that trained uniform anchor with targeted K2/K3-half + K4-quarter
  weighting for 8 epochs while freezing candidate_norm, candidate_hidden2,
  cardinality;
- evaluate all 96 candidate_hidden1 neuron swaps on the held-out internal fold.

For fold 0, the exact checkpoints from run 37247417133 may be supplied and
reused instead of retraining.

No outer evaluation. No promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest,write_json
from scripts.train_v273_group_gate_ab import build_model,metrics,weight_hash
from scripts.train_v273_loss_weighting_ab import class_weight_table,batches
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,epoch_order,array_hash,require

SEED=v260.SEED+1003
EPOCHS=8
INTERNAL_FOLDS=(0,1,2,4)
FREEZE_LOCAL=("candidate_norm","candidate_hidden2","cardinality")
LAYER="candidate_hidden1"
WEIGHT_ARM="targeted_k23_half_k4_quarter"

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,f"nested base mismatch: {[x.name for x in xs]}")
    return xs[0]

def compile_model(model):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss="sparse_categorical_crossentropy")

def fold_vector(cache,cfg):
    names=np.asarray(cache["members"]).astype(str)
    require(set(names)==set(cfg["member_folds"]),"member inventory drift")
    return np.asarray([cfg["member_folds"][m] for m in names],np.int32)

def train_model(model,cache,fit,k,table,output,label):
    import tensorflow as tf
    seq=batches(cache,fit,seed=SEED,shuffle=True,k=k,sample_table=table)
    hist=[];observed=[]
    class CB(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            exp=epoch_order(fit,SEED,epoch)
            np.testing.assert_array_equal(seq.order,exp)
            observed.append(array_hash(seq.order))
        def on_epoch_end(self,epoch,logs=None):
            hist.append({"epoch":int(epoch)+1,**{x:float(y) for x,y in (logs or {}).items()}})
            write_json(output/f"{label}-progress.json",{"completed_epoch":int(epoch)+1,"history":hist})
    res=model.fit(seq,epochs=EPOCHS,shuffle=False,workers=0,max_queue_size=1,verbose=2,
                  callbacks=[tf.keras.callbacks.TerminateOnNaN(),CB()])
    require(len(hist)==EPOCHS and all(np.isfinite(v).all() for v in res.history.values()),
            label+" training failure")
    return hist,observed

def predict(model,cache,val):
    p=np.asarray(model.predict(batches(cache,val),workers=0,max_queue_size=1,verbose=0),np.float32)
    require(p.shape==(len(val),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5),
            "bad probabilities")
    return p,p.argmax(1).astype(np.int32)

def counts(y,p):
    low=y<=1;poly=y>=2
    return {
      "global":int(np.sum(p==y)),
      "low":int(np.sum((p==y)&low)),
      "poly":int(np.sum((p==y)&poly)),
      "K":{str(v):int(np.sum((p==y)&(y==v))) for v in range(7)}
    }

def delta_counts(base,y,p):
    a=counts(y,base);b=counts(y,p)
    return {
      "global_net":b["global"]-a["global"],
      "low_net":b["low"]-a["low"],
      "poly_net":b["poly"]-a["poly"],
      "K_net":{str(v):b["K"][str(v)]-a["K"][str(v)] for v in range(7)}
    }

def set_swap(layer,freeze_w,uniform_w,neurons):
    kf,bf=[np.asarray(x).copy() for x in freeze_w]
    ku,bu=[np.asarray(x).copy() for x in uniform_w]
    ids=np.asarray(list(neurons),np.int64)
    if len(ids):
        kf[:,ids]=ku[:,ids];bf[ids]=bu[ids]
    layer.set_weights([kf,bf])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--val-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--reuse-uniform-weights",type=Path)
    ap.add_argument("--reuse-freeze-weights",type=Path)
    a=ap.parse_args()
    require(a.val_fold in INTERNAL_FOLDS,"invalid internal validation fold")
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()

    cfg=load_config(a.config)
    cache,parts,_=load_bundle(a.bundle,a.config)
    folds=fold_vector(cache,cfg)
    fit=np.flatnonzero(np.isin(folds,[f for f in INTERNAL_FOLDS if f!=a.val_fold]))
    val=np.flatnonzero(folds==a.val_fold)
    outer=np.flatnonzero(folds==3)
    require(len(fit)>0 and len(val)>0 and len(outer)>0,"empty partition")
    require(not np.intersect1d(fit,val).size and not np.intersect1d(fit,outer).size and
            not np.intersect1d(val,outer).size,"partition leakage")
    require(set(np.unique(folds[fit]).tolist())==set(INTERNAL_FOLDS)-{a.val_fold},"fit fold mismatch")
    k=np.minimum(cache["exact"].astype(np.int32),6)

    reuse=(a.reuse_uniform_weights is not None or a.reuse_freeze_weights is not None)
    require((a.reuse_uniform_weights is None)==(a.reuse_freeze_weights is None),
            "reuse weights must be supplied as a pair")
    if reuse:
        require(a.val_fold==0,"checkpoint reuse allowed only for fold 0")

    # Uniform anchor.
    uniform=build_model("learned_gate",SEED)
    initial_hash=weight_hash(uniform)
    if reuse:
        uniform.load_weights(a.reuse_uniform_weights)
        u_hist=[];u_order=[];uniform_source="reused_from_run_37247417133"
    else:
        u_hist,u_order=train_model(uniform,cache,fit,k,np.ones(7,np.float32),a.output,"uniform")
        uniform_source="trained_in_this_run"
    uniform.save_weights(a.output/"uniform.weights.h5")
    anchor_hash=weight_hash(uniform)
    bu=nested_base(uniform)
    u_prob,U=predict(uniform,cache,val)

    # Freeze-local fine-tune.
    freeze=build_model("learned_gate",SEED)
    if reuse:
        freeze.load_weights(a.reuse_freeze_weights)
        f_hist=[];f_order=[];freeze_source="reused_from_run_37247417133"
    else:
        freeze.load_weights(a.output/"uniform.weights.h5")
        bf=nested_base(freeze)
        frozen_before={}
        for name in FREEZE_LOCAL:
            layer=bf.get_layer(name)
            frozen_before[name]=[np.asarray(w).copy() for w in layer.get_weights()]
            layer.trainable=False
        for layer in freeze.layers:
            if layer is not bf:layer.trainable=True
        compile_model(freeze)
        table=class_weight_table(WEIGHT_ARM,k[fit])
        f_hist,f_order=train_model(freeze,cache,fit,k,table,a.output,"freeze_local_combo")
        for name,before in frozen_before.items():
            after=bf.get_layer(name).get_weights()
            require(all(np.array_equal(x,y) for x,y in zip(before,after)),"frozen layer changed: "+name)
        freeze_source="trained_in_this_run"
    freeze.save_weights(a.output/"freeze_local_combo.weights.h5")
    bf=nested_base(freeze)
    f_prob,F=predict(freeze,cache,val)

    y=k[val]
    um=metrics(y,U);fm=metrics(y,F)
    lu=bu.get_layer(LAYER);lf=bf.get_layer(LAYER)
    wu=[np.asarray(x).copy() for x in lu.get_weights()]
    wf=[np.asarray(x).copy() for x in lf.get_weights()]
    require(len(wu)==2 and len(wf)==2,"candidate_hidden1 must have kernel+bias")
    ku,bu_bias=wu;kf,bf_bias=wf
    require(ku.shape==kf.shape and bu_bias.shape==bf_bias.shape and ku.shape[1]==96,
            f"candidate_hidden1 shape drift: {ku.shape}/{bu_bias.shape}")

    delta=np.sqrt(np.sum((ku-kf)**2,axis=0)+(bu_bias-bf_bias)**2)
    rel=delta/(np.sqrt(np.sum(kf**2,axis=0)+bf_bias**2)+1e-12)

    singles=[]
    for j in range(96):
        set_swap(lf,wf,wu,[j])
        _,P=predict(freeze,cache,val)
        singles.append({
          "neuron":j,
          "weight_delta_norm":float(delta[j]),
          "weight_delta_relative":float(rel[j]),
          **delta_counts(F,y,P),
          "changed_predictions":int(np.sum(P!=F))
        })
    lf.set_weights(wf)

    report={
      "status":"completed","training":not reuse,
      "protocol":{
        "experiment":"v273_candidate_hidden1_rotating_inner_fold",
        "validation_fold":a.val_fold,
        "fit_folds":[f for f in INTERNAL_FOLDS if f!=a.val_fold],
        "outer_fold_3_loaded_for_fit_or_validation":False,
        "epochs_uniform":EPOCHS,"epochs_weighted_finetune":EPOCHS,
        "seed":SEED,"weighting":WEIGHT_ARM,
        "freeze_local_combo":list(FREEZE_LOCAL),
        "neuron_layer":LAYER,
        "automatic_promotion":False
      },
      "sources":{"uniform":uniform_source,"freeze":freeze_source},
      "rows":{"fit":len(fit),"validation":len(val),"outer_excluded":len(outer)},
      "hashes":{"initial":initial_hash,"uniform_anchor":anchor_hash,
                "uniform_weights":digest(a.output/"uniform.weights.h5"),
                "freeze_weights":digest(a.output/"freeze_local_combo.weights.h5")},
      "reference":{"uniform":um,"freeze_local_combo":fm},
      "single_neuron":singles,
      "safe_single_neurons":[x["neuron"] for x in singles
                             if x["global_net"]>0 and x["low_net"]>=0 and x["poly_net"]>=0],
      "history":{"uniform":u_hist,"freeze_local_combo":f_hist},
      "observed_epoch_order":{"uniform":u_order,"freeze_local_combo":f_order}
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    top=sorted(singles,key=lambda x:(x["global_net"],x["poly_net"],x["low_net"]),reverse=True)[:15]
    lines=[
      f"# Rotating inner neuron audit — validation fold {a.val_fold}","",
      f"Fit folds: {[f for f in INTERNAL_FOLDS if f!=a.val_fold]}; outer fold 3 excluded.","",
      f"Uniform: {100*um['exact']:.3f}% global / {100*um['poly_exact']:.3f}% poly.",
      f"freeze_local_combo: {100*fm['exact']:.3f}% global / {100*fm['poly_exact']:.3f}% poly.",
      f"Safe single neurons on this fold: {len(report['safe_single_neurons'])}.","",
      "| neuron | global net | low net | poly net | K0 | K1 | K2 | K3 | K4 |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in top:
        q=x["K_net"]
        lines.append(f"| {x['neuron']} | {x['global_net']:+d} | {x['low_net']:+d} | {x['poly_net']:+d} | "
                     f"{q['0']:+d} | {q['1']:+d} | {q['2']:+d} | {q['3']:+d} | {q['4']:+d} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
