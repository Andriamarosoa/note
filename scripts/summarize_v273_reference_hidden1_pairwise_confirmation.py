"""Deduplicated confirmatory nested validation of fixed hidden1 group.

Consumes six unique two-fold checkpoints and the historical three-fold
outer-validation checkpoints. It implements exactly the preregistered fixed-group
accept/reject rule, without searching any alternative group.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_candidate_hidden1_multifold_worker import (
    INTERNAL_FOLDS,LAYER,SEED,nested_base,predict,delta_counts,set_swap,fold_vector
)

GROUP=(42,52,61,64)

def discover_pairs(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_two_fold_hidden1_nested_checkpoint":
            continue
        pair=tuple(sorted(int(x) for x in r["protocol"]["fit_folds"]))
        require(pair not in out,f"duplicate pair {pair}")
        require(not r["protocol"]["fold3_used"] and not r["protocol"]["player05_used"],"scope leak")
        require(r["protocol"]["fixed_reference_group"]==list(GROUP),"group drift")
        out[pair]=(p.parent,r)
    expected={(0,1),(0,2),(0,4),(1,2),(1,4),(2,4)}
    require(set(out)==expected,f"pair inventory {set(out)}")
    return out

def discover_outer(root,fold):
    hits=[]
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("validation_fold")==fold:
            hits.append((p.parent,r))
    require(len(hits)==1,f"outer fold {fold} checkpoint count {len(hits)}")
    d,r=hits[0]
    uw=d/"uniform.weights.h5";fw=d/"freeze_local_combo.weights.h5"
    require(uw.exists() and fw.exists(),f"missing outer weights {fold}")
    return d,r,uw,fw

def acceptance(rows):
    gs=[x["global_net"] for x in rows]
    los=[x["low_net"] for x in rows]
    pos=[x["poly_net"] for x in rows]
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

def eval_outer(cache,k,folds,root,f):
    _,saved,uw,fw=discover_outer(root,f)
    val=np.flatnonzero(folds==f)
    y=k[val]
    uniform=build_model("learned_gate",SEED);uniform.load_weights(uw)
    freeze=build_model("learned_gate",SEED);freeze.load_weights(fw)
    _,U=predict(uniform,cache,val);_,F=predict(freeze,cache,val)
    ref={"uniform":metrics(y,U),"freeze_local_combo":metrics(y,F)}
    require(abs(ref["freeze_local_combo"]["exact"]-saved["reference"]["freeze_local_combo"]["exact"])<1e-12,
            "outer replay drift")
    bu=nested_base(uniform);bf=nested_base(freeze)
    wu=[np.asarray(x).copy() for x in bu.get_layer(LAYER).get_weights()]
    lf=bf.get_layer(LAYER);wf=[np.asarray(x).copy() for x in lf.get_weights()]
    set_swap(lf,wf,wu,GROUP)
    _,P=predict(freeze,cache,val)
    lf.set_weights(wf)
    return {"reference":ref,
            "fixed_group_delta":{**delta_counts(F,y,P),"changed_predictions":int(np.sum(P!=F)),
                                 "metrics":metrics(y,P)}}

def aggregate(rows):
    gs=[x["global_net"] for x in rows];los=[x["low_net"] for x in rows];pos=[x["poly_net"] for x in rows]
    return {
      "total_global_net":int(sum(gs)),"total_low_net":int(sum(los)),"total_poly_net":int(sum(pos)),
      "min_global_net":int(min(gs)),"min_low_net":int(min(los)),"min_poly_net":int(min(pos)),
      "positive_global_folds":int(sum(x>0 for x in gs)),
      "strict_original_rule":bool(all(x>=0 for x in los) and all(x>=0 for x in pos)
                                  and sum(x>0 for x in gs)>=3 and sum(gs)>0)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pair-root",type=Path,required=True)
    ap.add_argument("--outer-root",type=Path,required=True)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    a.output.mkdir(parents=True)

    pairs=discover_pairs(a.pair_root)
    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    folds=fold_vector(cache,cfg)
    names=np.asarray(cache["members"]).astype(str)
    require("05" not in {x[:2] for x in names},"player05 present")
    k=np.minimum(cache["exact"].astype(np.int32),6)

    rotations={}
    fixed_outer=[];policy_outer=[];accepted_all=[]
    for f in INTERNAL_FOLDS:
        remaining=[x for x in INTERNAL_FOLDS if x!=f]
        inner=[]
        for v in remaining:
            pair=tuple(sorted(x for x in remaining if x!=v))
            d,r=pairs[pair]
            q=r["evaluations"][str(v)]["fixed_group_delta"]
            inner.append({"inner_val_fold":int(v),"fit_pair":list(pair),**q})
        acc=acceptance(inner)
        out=eval_outer(cache,k,folds,a.outer_root,f)
        fixed=out["fixed_group_delta"]
        if acc["accepted"]:
            policy=dict(fixed)
        else:
            policy={"global_net":0,"low_net":0,"poly_net":0,
                    "K_net":{str(x):0 for x in range(7)},
                    "changed_predictions":0,
                    "metrics":out["reference"]["freeze_local_combo"]}
        rotations[str(f)]={"inner":inner,"acceptance":acc,"outer":out,"nested_policy_delta":policy}
        accepted_all.append(acc["accepted"]);fixed_outer.append(fixed);policy_outer.append(policy)

    fixed_a=aggregate(fixed_outer);policy_a=aggregate(policy_outer)
    confirmed=bool(all(accepted_all) and fixed_a["strict_original_rule"])
    report={
      "status":"completed",
      "experiment":"v273_reference_hidden1_deduplicated_nested_confirmation",
      "fixed_group":list(GROUP),
      "folds":list(INTERNAL_FOLDS),
      "fold3_used":False,"player05_used":False,
      "no_alternative_group_search":True,
      "rotations":rotations,
      "accepted_all_outer_rotations":bool(all(accepted_all)),
      "fixed_group_outer_summary":fixed_a,
      "nested_policy_outer_summary":policy_a,
      "predeclared_confirmation_passed":confirmed,
      "decision":"reference_confirmed" if confirmed else "reference_not_confirmed_by_nested_validation",
      "computational_note":"six unique two-fold models; semantically equivalent to preregistered 12-model duplicate layout",
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Deduplicated nested confirmation — hidden1 [42,52,61,64]","",
           "Six unique two-fold checkpoints; fold3/player05 excluded; no alternative group search.","",
           "| outer fold | inner accepted | outer global | low | poly | nested policy global |",
           "|---:|---|---:|---:|---:|---:|"]
    for f in INTERNAL_FOLDS:
        r=rotations[str(f)];d=r["outer"]["fixed_group_delta"];p=r["nested_policy_delta"]
        lines.append(f"| {f} | {r['acceptance']['accepted']} | {d['global_net']:+d} | {d['low_net']:+d} | {d['poly_net']:+d} | {p['global_net']:+d} |")
    lines+=["",
      f"All inner rotations accepted: **{all(accepted_all)}**.",
      f"Fixed group totals: global **{fixed_a['total_global_net']:+d}**, low **{fixed_a['total_low_net']:+d}**, poly **{fixed_a['total_poly_net']:+d}**.",
      f"Original strict outer rule: **{fixed_a['strict_original_rule']}**.",
      f"Nested policy total global: **{policy_a['total_global_net']:+d}**.",
      "",
      f"## Predeclared confirmatory verdict: **{'PASS' if confirmed else 'FAIL'}**"
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__":main()
