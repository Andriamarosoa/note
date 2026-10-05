"""Internal-only guardrail audit above robust candidate_hidden1 group [42,52,61,64].

No training. No outer fold 3 access.

For each rotating internal validation fold, replay:
  freeze_local_combo -> apply selected candidate_hidden1 group -> candidate guardrails.

Guardrails may revert a group prediction to freeze_local_combo only for predefined
fragile transitions:
  k0_up   : freeze predicts 0, group moves to 1/2
  k2_down : freeze predicts 2, group moves to 0/1
  k3_move : freeze predicts 3, group moves to 2/4

A revert is allowed only when the group's swap strength is small:
  P_group[group_pred] - P_group[freeze_pred] <= threshold

The rule family and thresholds are fixed before evaluation.
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
NEURONS=(42,52,61,64)
THRESHOLDS=(0.001,0.0025,0.005,0.01,0.02,0.03,0.05,0.08,0.12,0.20)
ZONES=("k0_up","k2_down","k3_move")
ZONE_SETS=tuple(
    combo for r in range(1,len(ZONES)+1)
    for combo in itertools.combinations(ZONES,r)
)

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

def fold_indices(cache,cfg,val_fold):
    names=np.asarray(cache["members"]).astype(str)
    folds=np.asarray([cfg["member_folds"][m] for m in names],np.int32)
    val=np.flatnonzero(folds==val_fold)
    outer=np.flatnonzero(folds==3)
    require(len(val)>0 and not np.intersect1d(val,outer).size,"partition leak")
    return val

def counts(y,p):
    low=y<=1;poly=y>=2
    return {
      "global":int(np.sum(p==y)),
      "low":int(np.sum((p==y)&low)),
      "poly":int(np.sum((p==y)&poly)),
      "under":int(np.sum(p<y)),
      "over":int(np.sum(p>y)),
    }

def delta(base,y,p):
    a=counts(y,base);b=counts(y,p)
    return {
      "global_net":b["global"]-a["global"],
      "low_net":b["low"]-a["low"],
      "poly_net":b["poly"]-a["poly"],
      "under_delta":b["under"]-a["under"],
      "over_delta":b["over"]-a["over"],
    }

def zone_mask(F,G,zones):
    m=np.zeros(len(F),dtype=bool)
    if "k0_up" in zones:
        m |= (F==0) & np.isin(G,(1,2))
    if "k2_down" in zones:
        m |= (F==2) & np.isin(G,(0,1))
    if "k3_move" in zones:
        m |= (F==3) & np.isin(G,(2,4))
    return m

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
    fold_dir=reports[a.val_fold][0].parent
    uw=fold_dir/"uniform.weights.h5"
    fw=fold_dir/"freeze_local_combo.weights.h5"
    require(uw.exists() and fw.exists(),f"missing weights: {fold_dir}")

    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    val=fold_indices(cache,cfg,a.val_fold)
    y=np.minimum(cache["exact"][val].astype(np.int32),6)

    mu=build_model("learned_gate",SEED);mu.load_weights(uw)
    mf=build_model("learned_gate",SEED);mf.load_weights(fw)
    bu=nested_base(mu);bf=nested_base(mf)
    wu=[np.asarray(x).copy() for x in bu.get_layer(LAYER).get_weights()]
    wf=[np.asarray(x).copy() for x in bf.get_layer(LAYER).get_weights()]
    ku,bu_bias=wu;kf,bf_bias=wf

    Pf,F=predict(mf,cache,val)
    k2=kf.copy();b2=bf_bias.copy()
    ids=np.asarray(NEURONS,np.int64)
    k2[:,ids]=ku[:,ids];b2[ids]=bu_bias[ids]
    bf.get_layer(LAYER).set_weights([k2,b2])
    Pg,G=predict(mf,cache,val)

    ref_group=counts(y,G)
    rules=[]
    idx=np.arange(len(G))
    for zones in ZONE_SETS:
        zmask=zone_mask(F,G,zones)
        swap_strength=Pg[idx,G]-Pg[idx,F]
        for t in THRESHOLDS:
            revert=zmask & (G!=F) & (swap_strength<=t)
            P=G.copy();P[revert]=F[revert]
            rid="+".join(zones)+f"@{t:.4f}"
            rules.append({
              "rule_id":rid,
              "zones":list(zones),
              "threshold":float(t),
              "reverted":int(revert.sum()),
              **delta(G,y,P),
              "metrics":metrics(y,P),
              "counts":counts(y,P)
            })

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_guardrail_robust_group_42_52_61_64",
        "validation_fold":a.val_fold,
        "outer_fold_3_used":False,
        "selected_neurons":list(NEURONS),
        "rule_family":"targeted freeze fallback by group swap strength",
        "zones":list(ZONES),
        "thresholds":list(THRESHOLDS),
        "automatic_promotion":False
      },
      "reference":{
        "freeze_local_combo":counts(y,F),
        "robust_group":ref_group,
        "freeze_metrics":metrics(y,F),
        "group_metrics":metrics(y,G)
      },
      "rules":rules
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    top=sorted(rules,key=lambda x:(x["global_net"],x["poly_net"],x["low_net"]),reverse=True)[:20]
    lines=[
      f"# Internal guardrail audit — fold {a.val_fold}","",
      f"Rules tested: **{len(rules)}**. Outer fold 3 used: **false**.","",
      "| rule | reverted | global net vs group | low net | poly net | under Δ | over Δ |",
      "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for x in top:
        lines.append(f"| {x['rule_id']} | {x['reverted']} | {x['global_net']:+d} | {x['low_net']:+d} | {x['poly_net']:+d} | {x['under_delta']:+d} | {x['over_delta']:+d} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
