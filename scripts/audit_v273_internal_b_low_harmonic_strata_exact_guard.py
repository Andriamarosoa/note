"""Exploratory exact-K guard audit using harmonic strata discovered internally.

This is NOT a fresh validation: the gate families come from prior internal
stratified analysis on the same folds. It is an exact-K utility audit to answer:
does restricting 3->2 decisions to the acoustically favorable harmonic regions
turn the weak AUC signal into consistent paired exact-count gains?

Fixed gate families:
  all
  high_register              (FIT upper tertile of median triplet F0)
  moderate_harmonic          (35 <= harmonic-ratio distance < 90 cents)
  high_and_moderate
  high_or_moderate

Classifier:
  logistic regression on the four promising harmonic features only,
  C=1, class_weight=balanced, threshold=0.5.
Target:
  true K2 vs not-K2 among B_low + base K3 rows.
Action:
  3->2 only.

Outer fold 3 is never loaded.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS,NEURONS,SEED,discover_reports,nested_base,predict,fold_ids,
    structural_features,deltas,
)
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    audio_rows,build_blow,BASE_FEATURES,
)

GATES=("all","high_register","moderate_harmonic","high_and_moderate","high_or_moderate")

def model():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=28031))
    ])

def materialize(rows):
    good=np.asarray([x is not None for x in rows],bool)
    rr=[x for x in rows if x is not None]
    keys=list(rr[0].keys()) if rr else []
    A=np.asarray([[r[k] for k in keys] for r in rr],np.float64) if rr else np.zeros((0,0))
    return good,rr,keys,A

def gate_masks(Af,Av,idx,name):
    high_cut=float(np.quantile(Af[:,idx["median_triplet_f0"]],2/3))
    hf=Af[:,idx["median_triplet_f0"]]>=high_cut
    hv=Av[:,idx["median_triplet_f0"]]>=high_cut
    mf=(Af[:,idx["third_harmonic_relation_distance_cents"]]>=35.0)&(Af[:,idx["third_harmonic_relation_distance_cents"]]<90.0)
    mv=(Av[:,idx["third_harmonic_relation_distance_cents"]]>=35.0)&(Av[:,idx["third_harmonic_relation_distance_cents"]]<90.0)
    if name=="all":return np.ones(len(Af),bool),np.ones(len(Av),bool),high_cut
    if name=="high_register":return hf,hv,high_cut
    if name=="moderate_harmonic":return mf,mv,high_cut
    if name=="high_and_moderate":return hf&mf,hv&mv,high_cut
    if name=="high_or_moderate":return hf|mf,hv|mv,high_cut
    raise ValueError(name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",type=Path,required=True)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fold-root",type=Path,required=True)
    ap.add_argument("--val-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.val_fold in FOLDS,"bad fold");require(not a.output.exists(),"overwrite")
    a.output.mkdir(parents=True)

    reports=discover_reports(a.fold_root);fd=reports[a.val_fold][0].parent
    cfg=load_config(a.config);cache,_,_=load_bundle(a.bundle,a.config)
    fit,val=fold_ids(cache,cfg,a.val_fold);y=np.minimum(cache["exact"].astype(np.int32),6)
    yf=y[fit];yv=y[val]

    mu=build_model("learned_gate",SEED);mu.load_weights(fd/"uniform.weights.h5")
    mg=build_model("learned_gate",SEED);mg.load_weights(fd/"freeze_local_combo.weights.h5")
    bu=nested_base(mu);bg=nested_base(mg)
    ku,b1=[np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg,b2=[np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ii=np.asarray(NEURONS,np.int64);kg[:,ii]=ku[:,ii];b2[ii]=b1[ii]
    bg.get_layer("candidate_hidden1").set_weights([kg,b2])

    Pf,Gf=predict(mg,cache,fit);Pv,Gv=predict(mg,cache,val)
    Xcf=structural_features(cache,fit,Pf);Xcv=structural_features(cache,val,Pv)
    fb,vb=build_blow(Xcf,Xcv,yf,Gf,a.val_fold)
    popf=fb&(Gf==3)
    popv=vb&(Gv==3)
    idsf=fit[popf];idsv=val[popv]
    yyf=(yf[popf]==2).astype(np.int32)
    yyv=(yv[popv]==2).astype(np.int32)
    require(len(idsf)>=50 and len(idsv)>=15,"small action population")

    rawf=audio_rows(cache,idsf,a.dataset_dir);rawv=audio_rows(cache,idsv,a.dataset_dir)
    gf,rf,keys,Af=materialize(rawf);gv,rv,keys2,Av=materialize(rawv)
    require(keys==keys2 and len(keys)>0,"feature keys drift")
    yyfg=yyf[gf];yyvg=yyv[gv]
    val_local=np.flatnonzero(popv)[gv]
    idx={k:i for i,k in enumerate(keys)}
    cols=[idx[k] for k in BASE_FEATURES]

    rows=[]
    for gate in GATES:
        mf,mv,cut=gate_masks(Af,Av,idx,gate)
        if int(mf.sum())<30 or len(np.unique(yyfg[mf]))<2:
            continue
        clf=model();clf.fit(Af[mf][:,cols],yyfg[mf])
        p=clf.predict_proba(Av[mv][:,cols])[:,1] if np.any(mv) else np.asarray([])
        choose_local=np.flatnonzero(mv)[p>=0.5] if len(p) else np.asarray([],np.int64)
        chosen_val_positions=val_local[choose_local]
        new=Gv.copy();new[chosen_val_positions]=2
        rec={
          "gate":gate,
          "high_register_fit_cut_hz":cut,
          "fit_gate_rows":int(mf.sum()),
          "val_gate_rows":int(mv.sum()),
          "applied_rows":int(len(chosen_val_positions)),
          **deltas(yv,Gv,new),
          "metrics":metrics(yv,new),
          "by_k_net":{str(k):int(np.sum((new==yv)&(yv==k))-np.sum((Gv==yv)&(yv==k))) for k in range(7)}
        }
        rows.append(rec)

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_harmonic_strata_exact_guard",
        "validation_fold":a.val_fold,"fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "fresh_validation":False,
        "reason_not_fresh":"gate families were discovered on these internal folds in run 37330190680",
        "population":"B_low + base K3",
        "target":"true K2 vs not-K2",
        "action":"3->2 only",
        "features":list(BASE_FEATURES),
        "classifier":"logistic C=1 class_weight=balanced threshold=0.5",
        "automatic_promotion":False
      },
      "rows":rows
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=[f"# Harmonic strata exact guard — fold {a.val_fold}","",
      "| gate | fit rows | val rows | applied | global net | K2 | K3 | corrections | regressions |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in sorted(rows,key=lambda z:(z["global_net"],-z["regressions"]),reverse=True):
        lines.append(f"| {x['gate']} | {x['fit_gate_rows']} | {x['val_gate_rows']} | {x['applied_rows']} | "
                     f"{x['global_net']:+d} | {x['by_k_net']['2']:+d} | {x['by_k_net']['3']:+d} | "
                     f"{x['corrections']} | {x['regressions']} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
