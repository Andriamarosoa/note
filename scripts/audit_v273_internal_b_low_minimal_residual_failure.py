"""Audit why the minimal pair+triplet residual discriminator fails on fold 2.

Replays exactly the same B_low + base-K3 population and the same two harmonic
features used in the minimal residual guard. Reports, for each internal fold:

- FIT/VAL class means and medians for K2 vs K3;
- pair-only and triplet-only held-out AUCs;
- joint logistic held-out AUC and standardized coefficients;
- Pearson correlation pair<->triplet overall and within K2/K3;
- covariance condition number on standardized FIT features;
- low/mid/high register held-out AUCs and 3->2 paired K2/K3 net;
- decision-score means for K2/K3.

The audit is descriptive/diagnostic only. Fold 3 is never loaded.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import ALLOWED_PLAYERS,index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS,NEURONS,SEED,discover_reports,nested_base,predict,fold_ids,
    structural_features,
)
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    transition_spectrum,extract_one,build_blow,
)

FEATURES=("best_pair_residual_ratio","best_triplet_residual_ratio")

def auc(y,s):
    y=np.asarray(y,np.int32);s=np.asarray(s,np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def pipe():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=28531))
    ])

def audio_rows(cache,ids,dataset_dir):
    indexed=tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by={t.annotation_member:t for t in indexed};aud={};out=[]
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
    require(len(rr)>0,"no valid rows")
    keys=list(rr[0].keys())
    A=np.asarray([[r[k] for k in keys] for r in rr],np.float64)
    return good,A,{k:i for i,k in enumerate(keys)}

def stats_by_class(y,x):
    out={}
    for label,name in ((1,"K2"),(0,"K3")):
        z=x[y==label]
        out[name]={
          "n":int(len(z)),
          "mean":float(np.mean(z)),
          "median":float(np.median(z)),
          "std":float(np.std(z)),
          "q25":float(np.quantile(z,.25)),
          "q75":float(np.quantile(z,.75)),
        }
    return out

def corr(x,y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])

def oriented_univariate_auc(yfit,xfit,yval,xval):
    pos=xfit[yfit==1];neg=xfit[yfit==0]
    orient=1.0 if float(np.mean(pos))>=float(np.mean(neg)) else -1.0
    return {
      "fit_orientation":int(orient),
      "val_auc":auc(yval,orient*xval),
      "fit_class_mean_delta":float(np.mean(pos)-np.mean(neg)),
      "val_class_mean_delta":float(np.mean(xval[yval==1])-np.mean(xval[yval==0])),
    }

def register_masks(Af,Av,idx):
    med=idx["median_triplet_f0"]
    q1,q2=np.quantile(Af[:,med],[1/3,2/3])
    return (
      {"low":Af[:,med]<q1,"mid":(Af[:,med]>=q1)&(Af[:,med]<q2),"high":Af[:,med]>=q2},
      {"low":Av[:,med]<q1,"mid":(Av[:,med]>=q1)&(Av[:,med]<q2),"high":Av[:,med]>=q2},
      (float(q1),float(q2))
    )

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

    popf=fb&(Gf==3)&np.isin(yf,(2,3))
    popv=vb&(Gv==3)&np.isin(yv,(2,3))
    idsf=fit[popf];idsv=val[popv]
    yyf=(yf[popf]==2).astype(np.int32);yyv=(yv[popv]==2).astype(np.int32)

    rf=audio_rows(cache,idsf,a.dataset_dir);rv=audio_rows(cache,idsv,a.dataset_dir)
    gf,Af,idx=matrix(rf);gv,Av,idx2=matrix(rv)
    require(idx.keys()==idx2.keys(),"key drift")
    yyfg=yyf[gf];yyvg=yyv[gv]
    cols=[idx[k] for k in FEATURES]
    Xf=Af[:,cols];Xv=Av[:,cols]

    uni={}
    for j,k in enumerate(FEATURES):
        uni[k]=oriented_univariate_auc(yyfg,Xf[:,j],yyvg,Xv[:,j])

    m=pipe();m.fit(Xf,yyfg)
    p=m.predict_proba(Xv)[:,1]
    score=m.decision_function(Xv)
    sc=m.named_steps["scale"];lr=m.named_steps["lr"]
    coef=np.asarray(lr.coef_[0],np.float64)

    Zf=sc.transform(Xf)
    cov=np.cov(Zf,rowvar=False)
    eig=np.linalg.eigvalsh(cov)
    cond=float(np.max(eig)/max(float(np.min(eig)),1e-12))

    corr_report={
      "fit_all":corr(Xf[:,0],Xf[:,1]),
      "val_all":corr(Xv[:,0],Xv[:,1]),
      "fit_K2":corr(Xf[yyfg==1,0],Xf[yyfg==1,1]),
      "fit_K3":corr(Xf[yyfg==0,0],Xf[yyfg==0,1]),
      "val_K2":corr(Xv[yyvg==1,0],Xv[yyvg==1,1]),
      "val_K3":corr(Xv[yyvg==0,0],Xv[yyvg==0,1]),
    }

    regf,regv,cuts=register_masks(Af,Av,idx)
    regs={}
    for name in ("low","mid","high"):
        mf=regf[name];mv=regv[name]
        rr={"fit_rows":int(mf.sum()),"val_rows":int(mv.sum())}
        if (mf.sum()>=12 and mv.sum()>=6 and len(np.unique(yyfg[mf]))==2
            and len(np.unique(yyvg[mv]))==2):
            rm=pipe();rm.fit(Xf[mf],yyfg[mf])
            rp=rm.predict_proba(Xv[mv])[:,1]
            pred=(rp>=.5).astype(np.int32)
            correct=int(np.sum((pred==1)&(yyvg[mv]==1)))
            regress=int(np.sum((pred==1)&(yyvg[mv]==0)))
            rr.update({
              "joint_auc":auc(yyvg[mv],rp),
              "applied":int(np.sum(pred==1)),
              "k23_exact_net":int(correct-regress),
              "corrections":correct,
              "regressions":regress,
              "pair_auc":oriented_univariate_auc(yyfg[mf],Xf[mf,0],yyvg[mv],Xv[mv,0])["val_auc"],
              "triplet_auc":oriented_univariate_auc(yyfg[mf],Xf[mf,1],yyvg[mv],Xv[mv,1])["val_auc"],
              "coefficients_standardized":{
                FEATURES[0]:float(rm.named_steps["lr"].coef_[0,0]),
                FEATURES[1]:float(rm.named_steps["lr"].coef_[0,1]),
              }
            })
        regs[name]=rr

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_minimal_residual_failure_audit",
        "validation_fold":a.val_fold,"fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "population":"B_low + base K3 + true K in {2,3}",
        "features":list(FEATURES),
        "classifier":"logistic C=1 class_weight=balanced",
        "automatic_promotion":False
      },
      "rows":{"fit":int(len(Xf)),"val":int(len(Xv)),
              "fit_K2":int(np.sum(yyfg==1)),"fit_K3":int(np.sum(yyfg==0)),
              "val_K2":int(np.sum(yyvg==1)),"val_K3":int(np.sum(yyvg==0))},
      "feature_stats":{
        FEATURES[0]:{"fit":stats_by_class(yyfg,Xf[:,0]),"val":stats_by_class(yyvg,Xv[:,0])},
        FEATURES[1]:{"fit":stats_by_class(yyfg,Xf[:,1]),"val":stats_by_class(yyvg,Xv[:,1])},
      },
      "univariate":uni,
      "joint":{
        "val_auc":auc(yyvg,p),
        "coefficients_standardized":{
          FEATURES[0]:float(coef[0]),FEATURES[1]:float(coef[1])
        },
        "intercept":float(lr.intercept_[0]),
        "score_stats":stats_by_class(yyvg,score),
      },
      "correlation":corr_report,
      "standardized_fit_covariance":cov.tolist(),
      "standardized_fit_condition_number":cond,
      "register_cuts_hz":{"low_upper":cuts[0],"high_lower":cuts[1]},
      "registers":regs,
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=[f"# Minimal residual failure audit — fold {a.val_fold}","",
      f"Joint AUC: **{report['joint']['val_auc']:.3f}**.",
      f"Pair-only AUC: **{uni[FEATURES[0]]['val_auc']:.3f}**.",
      f"Triplet-only AUC: **{uni[FEATURES[1]]['val_auc']:.3f}**.",
      f"FIT pair↔triplet corr: **{corr_report['fit_all']:.3f}**; VAL: **{corr_report['val_all']:.3f}**.",
      f"Condition number: **{cond:.1f}**.",
      f"Standardized LR coefficients: pair **{coef[0]:+.3f}**, triplet **{coef[1]:+.3f}**.","",
      "| register | joint AUC | pair AUC | triplet AUC | K2/K3 net |",
      "|---|---:|---:|---:|---:|"]
    for name in ("low","mid","high"):
        q=regs[name]
        def ff(v):return "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {name} | {ff(q.get('joint_auc'))} | {ff(q.get('pair_auc'))} | {ff(q.get('triplet_auc'))} | {q.get('k23_exact_net','n/a')} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
