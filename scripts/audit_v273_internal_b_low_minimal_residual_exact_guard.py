"""Minimal register-invariant residual discriminator for B_low/base-K3.

Uses ONLY:
  - best_pair_residual_ratio
  - best_triplet_residual_ratio

Protocol per validation fold 0/1/2/4:
  1) replay robust base and reconstruct B_low from FIT only;
  2) action population = B_low + base prediction K3;
  3) diagnostic training population = action rows with true K in {2,3};
  4) train fixed logistic regression on the two residual features only;
  5) evaluate K2-vs-K3 AUC on held-out validation rows;
  6) apply fixed threshold 0.5 as 3->2 action to ALL valid B_low/base-K3
     validation rows, including K0/K1/K4+, and report paired Exact-K deltas.

Outer fold 3 is never loaded.
No threshold tuning. No register gating. No promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import ALLOWED_PLAYERS,index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS,NEURONS,SEED,discover_reports,nested_base,predict,fold_ids,
    structural_features,deltas,
)
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    transition_spectrum,extract_one,build_blow,
)

FEATURES=("best_pair_residual_ratio","best_triplet_residual_ratio")

def clf():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=28431))
    ])

def auc(y,s):
    y=np.asarray(y,np.int32);s=np.asarray(s,np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def audio_rows(cache,ids,dataset_dir):
    indexed=tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by={t.annotation_member:t for t in indexed};aud={}
    out=[]
    for row in np.asarray(ids,np.int64):
        member=str(cache["members"][row])
        if member not in aud:
            t=by[member]
            wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
            aud[member]=np.asarray(wav.samples,np.float64)/32768.0
        freq,x=transition_spectrum(aud[member],int(cache["cluster_start_samples"][row]))
        out.append(extract_one(freq,x))
    return out

def matrix(rows):
    good=np.asarray([x is not None for x in rows],bool)
    rr=[x for x in rows if x is not None]
    require(len(rr)>0,"no valid harmonic rows")
    keys=list(rr[0].keys())
    A=np.asarray([[r[k] for k in keys] for r in rr],np.float64)
    idx={k:i for i,k in enumerate(keys)}
    return good,A,idx

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",type=Path,required=True)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fold-root",type=Path,required=True)
    ap.add_argument("--val-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.val_fold in FOLDS,"bad fold")
    require(not a.output.exists(),"overwrite")
    a.output.mkdir(parents=True)

    reports=discover_reports(a.fold_root);fd=reports[a.val_fold][0].parent
    cfg=load_config(a.config);cache,_,_=load_bundle(a.bundle,a.config)
    fit,val=fold_ids(cache,cfg,a.val_fold)
    y=np.minimum(cache["exact"].astype(np.int32),6)
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

    actf=fb&(Gf==3); actv=vb&(Gv==3)
    idsf=fit[actf];idsv=val[actv]
    yf_act=yf[actf];yv_act=yv[actv]

    rf=audio_rows(cache,idsf,a.dataset_dir)
    rv=audio_rows(cache,idsv,a.dataset_dir)
    gf,Af,idx=matrix(rf);gv,Av,idx2=matrix(rv)
    require(idx.keys()==idx2.keys(),"feature key drift")

    yf_valid=yf_act[gf];yv_valid=yv_act[gv]
    fit_action_positions=np.flatnonzero(actf)[gf]
    val_action_positions=np.flatnonzero(actv)[gv]

    train=np.isin(yf_valid,(2,3))
    test=np.isin(yv_valid,(2,3))
    ytr=(yf_valid[train]==2).astype(np.int32)
    yte=(yv_valid[test]==2).astype(np.int32)
    require(len(np.unique(ytr))==2 and len(np.unique(yte))==2,"binary collapse")
    cols=[idx[k] for k in FEATURES]

    model=clf();model.fit(Af[train][:,cols],ytr)
    p_k23=model.predict_proba(Av[test][:,cols])[:,1]
    k23_auc=auc(yte,p_k23)

    # Fixed 0.5 action threshold, applied to ALL valid action rows.
    p_all=model.predict_proba(Av[:,cols])[:,1]
    choose=np.flatnonzero(p_all>=0.5)
    chosen_val_positions=val_action_positions[choose]
    new=Gv.copy();new[chosen_val_positions]=2

    by_k={}
    for k in range(7):
        base_k=int(np.sum((Gv==yv)&(yv==k)))
        new_k=int(np.sum((new==yv)&(yv==k)))
        by_k[str(k)]=new_k-base_k

    # K2/K3-only paired effect on the validation action population.
    m23=np.isin(yv,(2,3))
    base23=int(np.sum((Gv==yv)&m23))
    new23=int(np.sum((new==yv)&m23))

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_minimal_pair_triplet_residual_exact_guard",
        "validation_fold":a.val_fold,
        "fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "action_population":"B_low + base K3",
        "training_population":"action rows with true K in {2,3}",
        "features":list(FEATURES),
        "classifier":"logistic C=1 class_weight=balanced",
        "threshold":0.5,
        "action":"3->2 only",
        "register_gating":False,
        "automatic_promotion":False
      },
      "diagnostic":{
        "fit_k23_rows":int(train.sum()),
        "val_k23_rows":int(test.sum()),
        "k2_vs_k3_auc":k23_auc
      },
      "action":{
        "valid_fit_rows":int(len(Af)),
        "valid_val_rows":int(len(Av)),
        "applied_rows":int(len(chosen_val_positions)),
        **deltas(yv,Gv,new),
        "metrics":metrics(yv,new),
        "by_k_net":by_k,
        "k23_exact_net":int(new23-base23)
      }
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    x=report["action"]
    lines=[f"# Minimal pair+triplet residual exact guard — fold {a.val_fold}","",
      f"K2/K3 AUC: **{k23_auc:.3f}**.","",
      f"Applied: **{x['applied_rows']}** rows.",
      f"Global Exact-K net: **{x['global_net']:+d}**.",
      f"K2/K3-only Exact-K net: **{x['k23_exact_net']:+d}**.",
      f"Corrections/regressions: **{x['corrections']}/{x['regressions']}**.","",
      f"By K net: {json.dumps(by_k,sort_keys=True)}",
      "",
      "No outer evaluation and no promotion."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
