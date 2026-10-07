"""Confirmatory nested validation of fixed hidden1 group [42,52,61,64].

For one outer validation fold F in {0,1,2,4}:
- F is never used for model selection.
- on the remaining three folds, train three fresh inner models; each inner model
  trains only on two folds and validates on the third;
- measure the exact effect of the fixed group [42,52,61,64] in each inner-val;
- accept/reject the fixed group using the preregistered 3-fold rule;
- reuse the historical outer-F checkpoint trained on the other three folds;
- evaluate the fixed group once on F. If inner rejected it, nested policy keeps
  the freeze_local_combo baseline on F.

No search over neurons/groups is performed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.train_v273_loss_weighting_ab import class_weight_table
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_candidate_hidden1_multifold_worker import (
    EPOCHS,INTERNAL_FOLDS,FREEZE_LOCAL,LAYER,WEIGHT_ARM,SEED,
    nested_base,compile_model,train_model,predict,counts,delta_counts,set_swap,fold_vector
)

GROUP=(42,52,61,64)

def discover_outer(root,fold):
    hits=[]
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("validation_fold")==fold:
            hits.append((p,r))
    require(len(hits)==1,f"outer fold {fold}: expected one checkpoint report, got {len(hits)}")
    p,r=hits[0]
    uw=p.parent/"uniform.weights.h5"
    fw=p.parent/"freeze_local_combo.weights.h5"
    require(uw.exists() and fw.exists(),f"missing outer weights for fold {fold}")
    return p.parent,r,uw,fw

def train_pair(cache,k,fit,val,outdir,label):
    import tensorflow as tf
    tf.keras.backend.clear_session()

    uniform=build_model("learned_gate",SEED)
    u_hist,u_order=train_model(uniform,cache,fit,k,np.ones(7,np.float32),outdir,label+"-uniform")
    up=outdir/(label+"-uniform.weights.h5")
    uniform.save_weights(up)
    _,U=predict(uniform,cache,val)

    freeze=build_model("learned_gate",SEED)
    freeze.load_weights(up)
    bf=nested_base(freeze)
    frozen_before={}
    for name in FREEZE_LOCAL:
        layer=bf.get_layer(name)
        frozen_before[name]=[np.asarray(w).copy() for w in layer.get_weights()]
        layer.trainable=False
    for layer in freeze.layers:
        if layer is not bf: layer.trainable=True
    compile_model(freeze)
    table=class_weight_table(WEIGHT_ARM,k[fit])
    f_hist,f_order=train_model(freeze,cache,fit,k,table,outdir,label+"-freeze")
    for name,before in frozen_before.items():
        after=bf.get_layer(name).get_weights()
        require(all(np.array_equal(x,y) for x,y in zip(before,after)),
                "frozen layer changed: "+name)
    _,F=predict(freeze,cache,val)

    bu=nested_base(uniform); bf=nested_base(freeze)
    lu=bu.get_layer(LAYER); lf=bf.get_layer(LAYER)
    wu=[np.asarray(x).copy() for x in lu.get_weights()]
    wf=[np.asarray(x).copy() for x in lf.get_weights()]
    set_swap(lf,wf,wu,GROUP)
    _,P=predict(freeze,cache,val)
    lf.set_weights(wf)

    y=k[val]
    return {
      "fit_rows":int(len(fit)),"val_rows":int(len(val)),
      "fit_folds":None,
      "reference":{"uniform":metrics(y,U),"freeze_local_combo":metrics(y,F)},
      "group_delta":{**delta_counts(F,y,P),"changed_predictions":int(np.sum(P!=F)),
                     "metrics":metrics(y,P)},
      "history":{"uniform":u_hist,"freeze":f_hist},
      "epoch_order":{"uniform":u_order,"freeze":f_order},
    }

def strict_inner(items):
    gs=[x["group_delta"]["global_net"] for x in items]
    los=[x["group_delta"]["low_net"] for x in items]
    pos=[x["group_delta"]["poly_net"] for x in items]
    return {
      "accepted":bool(all(x>=0 for x in los) and all(x>=0 for x in pos)
                      and sum(x>0 for x in gs)>=2 and sum(gs)>0),
      "positive_global_folds":int(sum(x>0 for x in gs)),
      "total_global_net":int(sum(gs)),
      "total_low_net":int(sum(los)),
      "total_poly_net":int(sum(pos)),
      "min_global_net":int(min(gs)),
      "min_low_net":int(min(los)),
      "min_poly_net":int(min(pos)),
    }

def outer_eval(cache,k,folds,val_fold,root):
    fold_dir,saved,uw,fw=discover_outer(root,val_fold)
    val=np.flatnonzero(folds==val_fold)
    fit=np.flatnonzero(np.isin(folds,[f for f in INTERNAL_FOLDS if f!=val_fold]))
    require(len(val)>0 and len(fit)>0,"empty outer partition")
    require(not np.intersect1d(val,fit).size,"outer leakage")

    uniform=build_model("learned_gate",SEED); uniform.load_weights(uw)
    freeze=build_model("learned_gate",SEED); freeze.load_weights(fw)
    _,U=predict(uniform,cache,val); _,F=predict(freeze,cache,val)
    y=k[val]
    ref={"uniform":metrics(y,U),"freeze_local_combo":metrics(y,F)}
    historical=saved["reference"]["freeze_local_combo"]
    require(abs(ref["freeze_local_combo"]["exact"]-historical["exact"])<1e-12,
            "outer checkpoint replay drift")

    bu=nested_base(uniform); bf=nested_base(freeze)
    lu=bu.get_layer(LAYER); lf=bf.get_layer(LAYER)
    wu=[np.asarray(x).copy() for x in lu.get_weights()]
    wf=[np.asarray(x).copy() for x in lf.get_weights()]
    set_swap(lf,wf,wu,GROUP)
    _,P=predict(freeze,cache,val)
    lf.set_weights(wf)
    return {
      "fit_folds":[int(x) for x in INTERNAL_FOLDS if x!=val_fold],
      "fit_rows":int(len(fit)),"val_rows":int(len(val)),
      "reference":ref,
      "fixed_group_delta":{**delta_counts(F,y,P),"changed_predictions":int(np.sum(P!=F)),
                           "metrics":metrics(y,P)},
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--outer-fold-root",type=Path,required=True)
    ap.add_argument("--outer-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.outer_fold in INTERNAL_FOLDS,"invalid outer fold")
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()

    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    folds=fold_vector(cache,cfg)
    names=np.asarray(cache["members"]).astype(str)
    require("05" not in {x[:2] for x in names},"player05 present")
    require(set(np.unique(folds).tolist())==set((0,1,2,3,4)),"fold inventory drift")
    k=np.minimum(cache["exact"].astype(np.int32),6)

    remaining=[f for f in INTERNAL_FOLDS if f!=a.outer_fold]
    inner=[]
    for inner_val in remaining:
        inner_fit=[f for f in remaining if f!=inner_val]
        fit=np.flatnonzero(np.isin(folds,inner_fit))
        val=np.flatnonzero(folds==inner_val)
        forbidden=np.flatnonzero(np.isin(folds,[a.outer_fold,3]))
        require(not np.intersect1d(fit,forbidden).size and not np.intersect1d(val,forbidden).size,
                "inner leakage")
        require(set(np.unique(folds[fit]).tolist())==set(inner_fit),"inner fit mismatch")
        q=train_pair(cache,k,fit,val,a.output,f"outer{a.outer_fold}-inner{inner_val}")
        q["inner_val_fold"]=int(inner_val)
        q["fit_folds"]=[int(x) for x in inner_fit]
        inner.append(q)

    acceptance=strict_inner(inner)
    outer=outer_eval(cache,k,folds,a.outer_fold,a.outer_fold_root)
    gd=outer["fixed_group_delta"]
    if acceptance["accepted"]:
        policy_delta={k:gd[k] for k in ("global_net","low_net","poly_net","K_net")}
        policy_delta["changed_predictions"]=gd["changed_predictions"]
        policy_delta["metrics"]=gd["metrics"]
    else:
        policy_delta={
          "global_net":0,"low_net":0,"poly_net":0,
          "K_net":{str(x):0 for x in range(7)},
          "changed_predictions":0,
          "metrics":outer["reference"]["freeze_local_combo"],
        }

    report={
      "status":"completed",
      "training":True,
      "protocol":{
        "experiment":"v273_fixed_hidden1_group_nested_confirmation",
        "fixed_group":list(GROUP),
        "outer_validation_fold":int(a.outer_fold),
        "inner_folds":remaining,
        "fold3_used":False,
        "player05_used":False,
        "inner_training_folds_per_model":2,
        "epochs_uniform":EPOCHS,
        "epochs_weighted_finetune":EPOCHS,
        "weighting":WEIGHT_ARM,
        "freeze_local_combo":list(FREEZE_LOCAL),
        "no_group_search":True,
        "automatic_promotion":False,
      },
      "inner_validation":inner,
      "inner_acceptance":acceptance,
      "outer_evaluation":outer,
      "nested_policy_delta":policy_delta,
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=[
      f"# Fixed hidden1 nested confirmation — outer fold {a.outer_fold}","",
      f"Fixed group: **{list(GROUP)}**. Fold 3 and player 05 excluded.","",
      "## Inner confirmation","","| inner val | fit folds | global | low | poly |",
      "|---:|---|---:|---:|---:|"
    ]
    for q in inner:
        d=q["group_delta"]
        lines.append(f"| {q['inner_val_fold']} | {q['fit_folds']} | {d['global_net']:+d} | {d['low_net']:+d} | {d['poly_net']:+d} |")
    lines += ["",
      f"Inner accepted: **{acceptance['accepted']}**; total global **{acceptance['total_global_net']:+d}**.",
      "",
      "## Held-out outer fold","",
      f"Fixed-group delta: global **{gd['global_net']:+d}**, low **{gd['low_net']:+d}**, poly **{gd['poly_net']:+d}**.",
      f"Nested-policy delta: global **{policy_delta['global_net']:+d}**, low **{policy_delta['low_net']:+d}**, poly **{policy_delta['poly_net']:+d}**."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
