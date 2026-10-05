"""Targeted exact group audit for candidate_hidden1 on rotating inner folds.

Uses only the completed internal single-neuron audits from run 37271262794.
No outer fold data is used for planning, fitting, or evaluation.

A deterministic plan is built from all 4 inner-fold single-neuron effects:
- rank a compact neuron pool by cross-fold stability;
- enumerate groups of size 2, 3, 4 inside that pool;
- rank groups using only the additive proxy from single-neuron effects;
- keep 20 groups per size;
- then evaluate each chosen group exactly by swapping all selected output
  columns + biases together and forwarding the held-out internal fold.

The additive proxy is only a search heuristic. Final group effects are always
measured exactly and may be non-additive.
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.train_v273_loss_weighting_ab import batches
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require

SEED=v260.SEED+1003
FOLDS=(0,1,2,4)
LAYER="candidate_hidden1"
POOL_SIZE=14
PER_SIZE=20
GROUP_SIZES=(2,3,4)

def discover_reports(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        fold=r.get("protocol",{}).get("validation_fold")
        if fold in FOLDS and "single_neuron" in r:
            require(fold not in out,f"duplicate fold report {fold}")
            out[fold]=(p,r)
    require(set(out)==set(FOLDS),f"missing fold reports: {out.keys()}")
    return out

def aggregate_singles(reports):
    rows=[]
    for neuron in range(96):
        per={}
        for f in FOLDS:
            xs=[x for x in reports[f][1]["single_neuron"] if int(x["neuron"])==neuron]
            require(len(xs)==1,f"missing neuron {neuron} fold {f}")
            per[f]=xs[0]
        g=[per[f]["global_net"] for f in FOLDS]
        lo=[per[f]["low_net"] for f in FOLDS]
        po=[per[f]["poly_net"] for f in FOLDS]
        rows.append({
          "neuron":neuron,"per_fold":per,
          "positive_global_folds":sum(x>0 for x in g),
          "nonnegative_low_folds":sum(x>=0 for x in lo),
          "nonnegative_poly_folds":sum(x>=0 for x in po),
          "total_global":sum(g),"total_low":sum(lo),"total_poly":sum(po),
          "min_global":min(g),"min_low":min(lo),"min_poly":min(po)
        })
    rows.sort(key=lambda x:(
        min(x["nonnegative_low_folds"],x["nonnegative_poly_folds"]),
        x["positive_global_folds"],
        x["total_global"],
        x["total_low"]+x["total_poly"],
        x["min_global"]
    ),reverse=True)
    return rows

def group_proxy(group,by_neuron):
    per={}
    penalty=0
    for f in FOLDS:
        g=sum(by_neuron[n]["per_fold"][f]["global_net"] for n in group)
        lo=sum(by_neuron[n]["per_fold"][f]["low_net"] for n in group)
        po=sum(by_neuron[n]["per_fold"][f]["poly_net"] for n in group)
        per[f]={"global_net":g,"low_net":lo,"poly_net":po}
        penalty += max(0,-lo)+max(0,-po)
    gs=[per[f]["global_net"] for f in FOLDS]
    los=[per[f]["low_net"] for f in FOLDS]
    pos=[per[f]["poly_net"] for f in FOLDS]
    return {
      "neurons":list(group),"per_fold":{str(f):per[f] for f in FOLDS},
      "safety_penalty":int(penalty),
      "positive_global_folds":sum(x>0 for x in gs),
      "total_global":sum(gs),"total_low":sum(los),"total_poly":sum(pos),
      "min_global":min(gs),"min_low":min(los),"min_poly":min(pos),
      "proxy_safe":all(x>=0 for x in los) and all(x>=0 for x in pos)
                   and sum(x>0 for x in gs)>=3 and sum(gs)>0
    }

def make_plan(reports):
    agg=aggregate_singles(reports)
    pool=[x["neuron"] for x in agg[:POOL_SIZE]]
    by={x["neuron"]:x for x in agg}
    groups=[]
    selected={}
    for size in GROUP_SIZES:
        cand=[group_proxy(g,by) for g in itertools.combinations(pool,size)]
        cand.sort(key=lambda x:(
            x["proxy_safe"],
            -x["safety_penalty"],
            x["positive_global_folds"],
            x["total_global"],
            x["min_global"],
            x["total_poly"],
            x["total_low"]
        ),reverse=True)
        selected[size]=cand[:PER_SIZE]
        groups.extend(cand[:PER_SIZE])
    return {
      "pool":pool,
      "pool_stats":[x for x in agg if x["neuron"] in pool],
      "groups_by_size":{str(k):v for k,v in selected.items()},
      "groups":groups,
      "selection":"top 20 per size by internal additive proxy; exact effects recomputed"
    }

def find_fold_dir(reports,val_fold):
    p,_=reports[val_fold]
    return p.parent

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,"nested base mismatch")
    return xs[0]

def predict(model,cache,ids):
    p=np.asarray(model.predict(batches(cache,ids),workers=0,max_queue_size=1,verbose=0),np.float32)
    require(p.shape==(len(ids),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5),
            "bad probabilities")
    return p,p.argmax(1).astype(np.int32)

def counts(y,p):
    low=y<=1;poly=y>=2
    return {
      "global":int(np.sum(p==y)),
      "low":int(np.sum((p==y)&low)),
      "poly":int(np.sum((p==y)&poly)),
      "K":{str(k):int(np.sum((p==y)&(y==k))) for k in range(7)}
    }

def delta(base,y,p):
    a=counts(y,base);b=counts(y,p)
    return {
      "global_net":b["global"]-a["global"],
      "low_net":b["low"]-a["low"],
      "poly_net":b["poly"]-a["poly"],
      "K_net":{str(k):b["K"][str(k)]-a["K"][str(k)] for k in range(7)}
    }

def set_group(layer,freeze_w,uniform_w,neurons):
    kf,bf=[np.asarray(x).copy() for x in freeze_w]
    ku,bu=[np.asarray(x).copy() for x in uniform_w]
    ids=np.asarray(neurons,np.int64)
    kf[:,ids]=ku[:,ids];bf[ids]=bu[ids]
    layer.set_weights([kf,bf])

def fold_indices(cache,cfg,val_fold):
    names=np.asarray(cache["members"]).astype(str)
    folds=np.asarray([cfg["member_folds"][m] for m in names],np.int32)
    val=np.flatnonzero(folds==val_fold)
    outer=np.flatnonzero(folds==3)
    require(len(val)>0 and not np.intersect1d(val,outer).size,"partition leak")
    return val

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fold-root",type=Path,required=True)
    ap.add_argument("--val-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.val_fold in FOLDS,"invalid fold")
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    reports=discover_reports(a.fold_root)
    plan=make_plan(reports)
    (a.output/"plan.json").write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")

    fold_dir=find_fold_dir(reports,a.val_fold)
    uw=fold_dir/"uniform.weights.h5"
    fw=fold_dir/"freeze_local_combo.weights.h5"
    require(uw.exists() and fw.exists(),f"missing fold weights in {fold_dir}")

    cfg=load_config(a.config)
    cache,parts,_=load_bundle(a.bundle,a.config)
    val=fold_indices(cache,cfg,a.val_fold)
    y=np.minimum(cache["exact"][val].astype(np.int32),6)

    mu=build_model("learned_gate",SEED);mu.load_weights(uw)
    mf=build_model("learned_gate",SEED);mf.load_weights(fw)
    bu=nested_base(mu);bf=nested_base(mf)
    lu=bu.get_layer(LAYER);lf=bf.get_layer(LAYER)
    wu=[np.asarray(x).copy() for x in lu.get_weights()]
    wf=[np.asarray(x).copy() for x in lf.get_weights()]
    require(len(wu)==2 and len(wf)==2 and wu[0].shape[1]==96,"layer shape drift")

    _,U=predict(mu,cache,val)
    _,F=predict(mf,cache,val)
    reference={"uniform":metrics(y,U),"freeze_local_combo":metrics(y,F)}
    # Ensure exact replay of stored reference metrics within numerical identity.
    saved=reports[a.val_fold][1]["reference"]["freeze_local_combo"]
    require(abs(reference["freeze_local_combo"]["exact"]-saved["exact"])<1e-12,
            "freeze replay metric mismatch")

    results=[]
    for i,g in enumerate(plan["groups"]):
        neurons=[int(x) for x in g["neurons"]]
        set_group(lf,wf,wu,neurons)
        _,P=predict(mf,cache,val)
        results.append({
          "group_id":i,
          "size":len(neurons),
          "neurons":neurons,
          "proxy":g,
          **delta(F,y,P),
          "changed_predictions":int(np.sum(P!=F)),
          "metrics":metrics(y,P)
        })
    lf.set_weights(wf)

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_candidate_hidden1_targeted_group_audit",
        "validation_fold":a.val_fold,
        "outer_fold_3_used":False,
        "pool_size":POOL_SIZE,"groups_per_size":PER_SIZE,
        "group_sizes":list(GROUP_SIZES),
        "planning_source":"single-neuron effects across internal folds 0,1,2,4 only",
        "exact_forward_recomputed":True,
        "automatic_promotion":False
      },
      "plan":plan,
      "reference":reference,
      "groups":results
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    top=sorted(results,key=lambda x:(x["global_net"],x["poly_net"],x["low_net"]),reverse=True)[:20]
    lines=[
      f"# Candidate_hidden1 group audit — fold {a.val_fold}","",
      f"Pool: {plan['pool']}. Exact groups tested: {len(results)}.","",
      "| size | neurons | global net | low net | poly net | changed |",
      "|---:|---|---:|---:|---:|---:|"
    ]
    for x in top:
        lines.append(f"| {x['size']} | {x['neurons']} | {x['global_net']:+d} | "
                     f"{x['low_net']:+d} | {x['poly_net']:+d} | {x['changed_predictions']} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
