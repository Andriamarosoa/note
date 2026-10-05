"""Inner-val neuron audit for candidate_hidden1 after freeze_local_combo.

No training. Uses only checkpoints produced by the inner-only audit run
37247417133:
- inner_uniform.weights.h5
- inner_freeze_local_combo.weights.h5

Starting from inner freeze_local_combo, restore one candidate_hidden1 output
neuron at a time (kernel column + bias entry) to the corresponding trained
inner-uniform values and evaluate on inner_val fold 0.

Then evaluate cumulative prefixes chosen ONLY from fold-0 single-neuron
validation effects under a strict safety filter:
  low-K (K0/K1) net >= 0 and poly net >= 0 versus freeze_local_combo.
This is inner validation model selection; outer fold 3 is never loaded here.
No automatic promotion.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model, metrics
from scripts.train_v273_loss_weighting_ab import batches
from scripts.v273_window_experiment import load_bundle, require

SEED=v260.SEED+1003
LAYER="candidate_hidden1"
PREFIXES=(1,2,4,8,16,32,64,96)

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,f"nested base mismatch: {[x.name for x in xs]}")
    return xs[0]

def predict(model,cache,val):
    p=np.asarray(model.predict(
        batches(cache,val),workers=0,max_queue_size=1,verbose=0
    ),np.float32)
    require(p.shape==(len(val),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5),
            "bad probabilities")
    return p,p.argmax(1).astype(np.int32)

def counts(y,p):
    low=y<=1; poly=y>=2
    return {
      "global":int(np.sum(p==y)),
      "low":int(np.sum((p==y)&low)),
      "poly":int(np.sum((p==y)&poly)),
      "K":{str(k):int(np.sum((p==y)&(y==k))) for k in range(7)}
    }

def delta_counts(base,y,p):
    a=counts(y,base);b=counts(y,p)
    return {
      "global_net":b["global"]-a["global"],
      "low_net":b["low"]-a["low"],
      "poly_net":b["poly"]-a["poly"],
      "K_net":{str(k):b["K"][str(k)]-a["K"][str(k)] for k in range(7)}
    }

def swap_neurons(base_layer, freeze_w, uniform_w, neurons):
    require(len(freeze_w)==2 and len(uniform_w)==2,"expected Dense kernel+bias")
    kf,bf=[np.asarray(x).copy() for x in freeze_w]
    ku,bu=[np.asarray(x).copy() for x in uniform_w]
    require(kf.shape==ku.shape and bf.shape==bu.shape and kf.ndim==2 and bf.ndim==1,
            f"unexpected weights {kf.shape} {bf.shape}")
    require(kf.shape[1]==bf.shape[0],"output width mismatch")
    ids=np.asarray(list(neurons),np.int64)
    if len(ids):
        kf[:,ids]=ku[:,ids]
        bf[ids]=bu[ids]
    base_layer.set_weights([kf,bf])

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","freeze-weights","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    cache,parts,_=load_bundle(a.bundle,a.config)
    val=np.asarray(parts["inner_val"],np.int64)
    outer=np.asarray(parts["outer"],np.int64)
    require(len(val)>0 and not np.intersect1d(val,outer).size,"inner/outer overlap")
    y=np.minimum(cache["exact"][val].astype(np.int32),6)

    mu=build_model("learned_gate",SEED);mu.load_weights(a.uniform_weights)
    mf=build_model("learned_gate",SEED);mf.load_weights(a.freeze_weights)
    bu=nested_base(mu);bf=nested_base(mf)
    lu=bu.get_layer(LAYER);lf=bf.get_layer(LAYER)
    wu=[np.asarray(x).copy() for x in lu.get_weights()]
    wf=[np.asarray(x).copy() for x in lf.get_weights()]
    require(len(wu)==2 and len(wf)==2,"candidate_hidden1 must be Dense-like")
    kernel_u,bias_u=wu; kernel_f,bias_f=wf
    require(kernel_u.shape==kernel_f.shape and bias_u.shape==bias_f.shape,
            "weight shape mismatch")
    width=int(kernel_f.shape[1])
    require(width==96,f"expected 96 candidate_hidden1 outputs, got {width}")

    u_prob,U=predict(mu,cache,val)
    f_prob,F=predict(mf,cache,val)
    um=metrics(y,U);fm=metrics(y,F)
    baseline_counts=counts(y,F)

    col_delta=np.sqrt(np.sum((kernel_u-kernel_f)**2,axis=0)+(bias_u-bias_f)**2)
    col_rel=col_delta/(np.sqrt(np.sum(kernel_f**2,axis=0)+bias_f**2)+1e-12)

    singles=[]
    for j in range(width):
        swap_neurons(lf,wf,wu,[j])
        prob,p=predict(mf,cache,val)
        d=delta_counts(F,y,p)
        singles.append({
          "neuron":int(j),
          "weight_delta_norm":float(col_delta[j]),
          "weight_delta_relative":float(col_rel[j]),
          **d,
          "metrics":metrics(y,p),
          "changed_predictions":int(np.sum(p!=F))
        })
    lf.set_weights(wf)

    # Validation-selected safe ordering: no single-neuron loss on low-K or poly.
    safe=[x for x in singles if x["low_net"]>=0 and x["poly_net"]>=0 and x["global_net"]>0]
    safe.sort(key=lambda x:(x["global_net"],x["low_net"]+x["poly_net"],x["poly_net"],x["low_net"]),reverse=True)
    safe_order=[x["neuron"] for x in safe]
    remaining=[x["neuron"] for x in sorted(singles,key=lambda x:(x["global_net"],x["poly_net"],x["low_net"]),reverse=True)
               if x["neuron"] not in set(safe_order)]
    validation_order=safe_order+remaining

    cumulative={}
    chosen_best=None
    best_key=None
    for n in PREFIXES:
        chosen=validation_order[:min(n,width)]
        swap_neurons(lf,wf,wu,chosen)
        prob,p=predict(mf,cache,val)
        d=delta_counts(F,y,p); m=metrics(y,p)
        row={
          "neurons":[int(x) for x in chosen],
          **d,"metrics":m,
          "changed_predictions":int(np.sum(p!=F))
        }
        cumulative[str(n)]=row
        # Selection criterion fixed before outer: maximize global correct,
        # subject to no poly degradation and no low-K degradation vs freeze.
        if d["poly_net"]>=0 and d["low_net"]>=0:
            key=(d["global_net"],d["poly_net"],d["low_net"],-n)
            if chosen_best is None or key>best_key:
                chosen_best={"prefix":int(n),**row};best_key=key
    lf.set_weights(wf)

    # Parameter-only cumulative control.
    norm_order=[int(x) for x in np.argsort(-col_delta)]
    cumulative_norm={}
    for n in PREFIXES:
        chosen=norm_order[:min(n,width)]
        swap_neurons(lf,wf,wu,chosen)
        prob,p=predict(mf,cache,val)
        cumulative_norm[str(n)]={
          "neurons":chosen,
          **delta_counts(F,y,p),
          "metrics":metrics(y,p),
          "changed_predictions":int(np.sum(p!=F))
        }
    lf.set_weights(wf)

    full_swap_pred=None
    swap_neurons(lf,wf,wu,range(width))
    _,full_swap_pred=predict(mf,cache,val)
    lf.set_weights(wf)

    result={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_inner_candidate_hidden1_neuron_audit",
        "source_run":37247417133,
        "validation":"inner_val fold 0",
        "outer_fold_3_loaded":False,
        "layer":LAYER,
        "swap_unit":"kernel output column + bias entry",
        "single_neuron_selection_use":"inner validation only",
        "safe_single_filter":"low_net>=0 and poly_net>=0 and global_net>0",
        "prefix_selection":"maximize global net subject to low_net>=0 and poly_net>=0",
        "automatic_promotion":False
      },
      "layer_shape":{"kernel":list(kernel_f.shape),"bias":list(bias_f.shape)},
      "reference":{"uniform":um,"freeze_local_combo":fm},
      "single_neuron":singles,
      "safe_single_neurons":safe_order,
      "safe_single_count":len(safe_order),
      "validation_order":validation_order,
      "cumulative_validation_order":cumulative,
      "cumulative_weight_delta_norm":cumulative_norm,
      "selected_inner_prefix":chosen_best,
      "full_layer_swap":{
        **delta_counts(F,y,full_swap_pred),
        "metrics":metrics(y,full_swap_pred)
      }
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    top=sorted(singles,key=lambda x:(x["global_net"],x["poly_net"],x["low_net"]),reverse=True)[:20]
    lines=[
      "# Inner fold-0 audit — candidate_hidden1 neurons","",
      "No training. Outer fold 3 was not loaded.","",
      f"Reference uniform: {100*um['exact']:.3f}% global / {100*um['poly_exact']:.3f}% poly.",
      f"Reference freeze_local_combo: {100*fm['exact']:.3f}% global / {100*fm['poly_exact']:.3f}% poly.",
      f"Safe single neurons: {len(safe_order)}.","",
      "## Top single-neuron swaps","",
      "| neuron | global net | low K0/1 net | poly net | K0 | K1 | K2 | K3 | K4 | changed |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in top:
        kn=x["K_net"]
        lines.append(
          f"| {x['neuron']} | {x['global_net']:+d} | {x['low_net']:+d} | {x['poly_net']:+d} | "
          f"{kn['0']:+d} | {kn['1']:+d} | {kn['2']:+d} | {kn['3']:+d} | {kn['4']:+d} | {x['changed_predictions']} |"
        )
    lines += ["","## Cumulative validation-selected prefixes","",
              "| N | global net | low net | poly net | global | poly |",
              "|---:|---:|---:|---:|---:|---:|"]
    for n in PREFIXES:
        x=cumulative[str(n)]
        lines.append(f"| {n} | {x['global_net']:+d} | {x['low_net']:+d} | {x['poly_net']:+d} | "
                     f"{100*x['metrics']['exact']:.3f}% | {100*x['metrics']['poly_exact']:.3f}% |")
    if chosen_best:
        lines += ["",f"Selected inner prefix: **{chosen_best['prefix']} neurons**, "
                  f"global net {chosen_best['global_net']:+d}, low net {chosen_best['low_net']:+d}, "
                  f"poly net {chosen_best['poly_net']:+d}."]
    else:
        lines += ["","No cumulative prefix satisfied the safety constraint."]
    lines += ["","No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
