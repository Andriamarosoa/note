"""Train one unique two-fold V27.3 pair for nested reference validation.

The model is trained only on two folds from {0,1,2,4}. Fold 3 and player05 are
forbidden. After training uniform + freeze_local_combo, evaluate:
- fixed hidden1 group [42,52,61,64]
- all 96 single-neuron swaps
on each of the two other allowed folds.

The saved checkpoints can be reused for exact nested group reselection without
retraining this pair.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.train_v273_loss_weighting_ab import class_weight_table
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_candidate_hidden1_multifold_worker import (
    EPOCHS,INTERNAL_FOLDS,FREEZE_LOCAL,LAYER,WEIGHT_ARM,SEED,
    nested_base,compile_model,train_model,predict,delta_counts,set_swap,fold_vector
)

GROUP=(42,52,61,64)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fold-a",type=int,required=True)
    ap.add_argument("--fold-b",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    pair=tuple(sorted((a.fold_a,a.fold_b)))
    require(len(set(pair))==2 and set(pair)<=set(INTERNAL_FOLDS),"invalid pair")
    require(not a.output.exists(),"overwrite")
    a.output.mkdir(parents=True)

    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()

    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    folds=fold_vector(cache,cfg)
    names=np.asarray(cache["members"]).astype(str)
    require("05" not in {x[:2] for x in names},"player05 present")
    fit=np.flatnonzero(np.isin(folds,pair))
    forbidden=np.flatnonzero(np.isin(folds,[3]))
    require(len(fit)>0 and not np.intersect1d(fit,forbidden).size,"fold3 leak")
    require(set(np.unique(folds[fit]).tolist())==set(pair),"pair fit mismatch")
    k=np.minimum(cache["exact"].astype(np.int32),6)

    # Uniform anchor on exactly this pair.
    uniform=build_model("learned_gate",SEED)
    uh,uo=train_model(uniform,cache,fit,k,np.ones(7,np.float32),a.output,"uniform")
    up=a.output/"uniform.weights.h5"
    uniform.save_weights(up)

    # Frozen weighted fine-tune on exactly this pair.
    freeze=build_model("learned_gate",SEED)
    freeze.load_weights(up)
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
    fh,fo=train_model(freeze,cache,fit,k,table,a.output,"freeze_local_combo")
    for name,before in frozen_before.items():
        after=bf.get_layer(name).get_weights()
        require(all(np.array_equal(x,y) for x,y in zip(before,after)),
                "frozen layer changed: "+name)
    fp=a.output/"freeze_local_combo.weights.h5"
    freeze.save_weights(fp)

    bu=nested_base(uniform);bf=nested_base(freeze)
    lu=bu.get_layer(LAYER);lf=bf.get_layer(LAYER)
    wu=[np.asarray(x).copy() for x in lu.get_weights()]
    wf=[np.asarray(x).copy() for x in lf.get_weights()]
    require(wu[0].shape[1]==96 and wf[0].shape==wu[0].shape,"hidden1 shape drift")

    evaluations={}
    for val_fold in [f for f in INTERNAL_FOLDS if f not in pair]:
        val=np.flatnonzero(folds==val_fold)
        require(len(val)>0 and not np.intersect1d(fit,val).size,"pair/val leak")
        y=k[val]
        _,U=predict(uniform,cache,val)
        _,F=predict(freeze,cache,val)

        set_swap(lf,wf,wu,GROUP)
        _,P=predict(freeze,cache,val)
        fixed={**delta_counts(F,y,P),"changed_predictions":int(np.sum(P!=F)),
               "metrics":metrics(y,P)}
        lf.set_weights(wf)

        singles=[]
        for neuron in range(96):
            set_swap(lf,wf,wu,[neuron])
            _,S=predict(freeze,cache,val)
            singles.append({"neuron":int(neuron),
                            **delta_counts(F,y,S),
                            "changed_predictions":int(np.sum(S!=F))})
        lf.set_weights(wf)

        evaluations[str(val_fold)]={
          "validation_fold":int(val_fold),
          "reference":{"uniform":metrics(y,U),"freeze_local_combo":metrics(y,F)},
          "fixed_group_delta":fixed,
          "single_neuron":singles,
        }

    report={
      "status":"completed","training":True,
      "protocol":{
        "experiment":"v273_two_fold_hidden1_nested_checkpoint",
        "fit_folds":list(pair),
        "eval_folds":[f for f in INTERNAL_FOLDS if f not in pair],
        "fold3_used":False,"player05_used":False,
        "epochs_uniform":EPOCHS,"epochs_weighted_finetune":EPOCHS,
        "weighting":WEIGHT_ARM,"freeze_local_combo":list(FREEZE_LOCAL),
        "fixed_reference_group":list(GROUP),
        "single_neurons_saved_for_future_nested_reselection":True,
      },
      "rows":{"fit":int(len(fit))},
      "evaluations":evaluations,
      "history":{"uniform":uh,"freeze_local_combo":fh},
      "epoch_order":{"uniform":uo,"freeze_local_combo":fo},
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=[f"# Two-fold nested checkpoint {pair}","",
           f"Fit folds: **{pair}**. Fold 3/player05 excluded.","",
           "| eval fold | fixed global | low | poly |",
           "|---:|---:|---:|---:|"]
    for f,q in evaluations.items():
        d=q["fixed_group_delta"]
        lines.append(f"| {f} | {d['global_net']:+d} | {d['low_net']:+d} | {d['poly_net']:+d} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__":main()
