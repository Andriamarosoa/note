"""Internal audit of new temporal-onset evidence for B_low/base-K3.

This pilot uses only already-available causal spectral_map data. It does NOT
claim to implement candidate-specific F0 evidence. Instead it asks whether the
previously underused time axis contains extra onset information that separates
true K2 from true K3 inside B_low when the robust base predicts K3.

For each rotating validation fold 0/1/2/4:
  * replay robust hidden1 group [42,52,61,64];
  * reconstruct B_low from FIT only;
  * select B_low + base prediction K3;
  * compare fixed diagnostic logistic models:
      current structural features
      new temporal-onset features only
      current + temporal-onset
  * report K2-vs-K3 AUC and K2-vs-rest AUC;
  * report out-of-fold univariate AUCs for every new feature.

Outer fold 3 is never loaded.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS, NEURONS, SEED, discover_reports, nested_base, predict, fold_ids,
    structural_features,
)
from scripts.train_v273_group_gate_ab import build_model
from scripts.spectral_window import COVERED
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require

DIAG_C=1.0

def diagnostic_model():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(
        C=DIAG_C,max_iter=3000,solver="lbfgs",
        class_weight="balanced",random_state=27731
      ))
    ])

def auc(y,s):
    y=np.asarray(y,dtype=np.int32);s=np.asarray(s,dtype=np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def safe_div(a,b):
    return a/(b+1e-8)

def entropy_rows(x):
    x=np.maximum(np.asarray(x,np.float64),0.0)
    p=x/(x.sum(1,keepdims=True)+1e-12)
    return -np.sum(np.where(p>0,p*np.log(p+1e-12),0.0),axis=1)

def temporal_onset_features(cache,ids):
    """Label-free cluster-onset features from 31x64x3 covered spectral maps."""
    s=np.asarray(cache["spectral"][ids,:COVERED.time_frames],np.float64)
    require(s.ndim==4 and s.shape[1:]==(COVERED.time_frames,64,3),"spectral shape drift")
    centers=np.asarray(COVERED.centers,np.float64)
    pre=centers<0
    early=(centers>=0)&(centers<0.010*44100)
    group=(centers>=0)&(centers<=0.040*44100)
    late=(centers>0.040*44100)
    post=centers>=0
    require(pre.any() and early.any() and group.any() and late.any(),"bad temporal masks")

    lp=s[:,:,:,0]
    pp=s[:,:,:,1]
    fl=s[:,:,:,2]

    # Per-frame total/mean spectral evidence.
    fl_t=fl.mean(2)
    pp_t=pp.mean(2)
    lp_t=lp.mean(2)

    # Band distributions after onset.
    fl_band=fl[:,post,:].sum(1)
    pp_band=pp[:,post,:].sum(1)
    fl_ent=entropy_rows(fl_band)/np.log(fl_band.shape[1])
    pp_ent=entropy_rows(pp_band)/np.log(pp_band.shape[1])

    # Weighted timing of post-onset flux.
    post_cent=centers[post]
    w=fl_t[:,post]
    mass=w.sum(1)+1e-12
    tmean=(w*post_cent[None,:]).sum(1)/mass
    tvar=(w*(post_cent[None,:]-tmean[:,None])**2).sum(1)/mass
    tstd=np.sqrt(np.maximum(tvar,0.0))
    imax=np.argmax(w,axis=1)
    tpeak=post_cent[imax]

    # Number of distinct local temporal flux peaks (strict, no learned threshold).
    peak_count=np.zeros(len(ids),np.float64)
    for i,row in enumerate(w):
        if len(row)>=3:
            peak_count[i]=np.sum((row[1:-1]>row[:-2])&(row[1:-1]>=row[2:]))

    names=[
      "flux_post_pre_delta",
      "flux_early_mean",
      "flux_group_mean",
      "flux_late_mean",
      "flux_late_to_group",
      "flux_peak",
      "flux_peak_delay_samples",
      "flux_time_mean_samples",
      "flux_time_std_samples",
      "flux_temporal_peak_count",
      "flux_band_entropy",
      "positive_pre_post_pre_delta",
      "positive_pre_early_mean",
      "positive_pre_group_mean",
      "positive_pre_late_mean",
      "positive_pre_late_to_group",
      "positive_pre_peak",
      "positive_pre_band_entropy",
      "log_power_post_pre_delta",
      "log_power_group_std",
    ]
    X=np.column_stack([
      fl_t[:,post].mean(1)-fl_t[:,pre].mean(1),
      fl_t[:,early].mean(1),
      fl_t[:,group].mean(1),
      fl_t[:,late].mean(1),
      safe_div(fl_t[:,late].mean(1),fl_t[:,group].mean(1)),
      fl_t[:,post].max(1),
      tpeak,tmean,tstd,peak_count,fl_ent,
      pp_t[:,post].mean(1)-pp_t[:,pre].mean(1),
      pp_t[:,early].mean(1),
      pp_t[:,group].mean(1),
      pp_t[:,late].mean(1),
      safe_div(pp_t[:,late].mean(1),pp_t[:,group].mean(1)),
      pp_t[:,post].max(1),pp_ent,
      lp_t[:,post].mean(1)-lp_t[:,pre].mean(1),
      lp_t[:,group].std(1),
    ]).astype(np.float64)
    require(X.shape==(len(ids),len(names)) and np.isfinite(X).all(),"bad onset features")
    return X,names

def build_blow(Xf,Xv,yf,Gf,val_fold):
    fail=np.isin(yf,(2,3,4))&(Gf!=yf)
    coarse=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370+val_fold))
    ])
    coarse.fit(Xf[fail])
    cf=coarse.predict(Xf);cv=coarse.predict(Xv)
    cs={}
    for c in (0,1):
        m=fail&(cf==c);require(int(m.sum())>20,"small coarse")
        under=int(np.sum(Gf[m]<yf[m]))
        cs[c]={"rows":int(m.sum()),"under":under,
               "under_rate":float(under/max(1,int(m.sum())))}
    a_like=max((0,1),key=lambda c:(cs[c]["under_rate"],cs[c]["rows"]))
    b_like=1-a_like
    b_fail=fail&(cf==b_like)
    sub=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=80,random_state=27420+val_fold))
    ])
    sub.fit(Xf[b_fail])
    sf=sub.predict(Xf);sv=sub.predict(Xv)
    ss={}
    for c in (0,1):
        m=b_fail&(sf==c);require(int(m.sum())>20,"small sub")
        under=int(np.sum(Gf[m]<yf[m]))
        ss[c]={"rows":int(m.sum()),"under":under,
               "under_rate":float(under/max(1,int(m.sum())))}
    low=max((0,1),key=lambda c:(ss[c]["under_rate"],ss[c]["rows"]))
    return (cf==b_like)&(sf==low),(cv==b_like)&(sv==low),cs,ss,b_like,low

def eval_variant(Xfit,yfit,Xval,yval):
    require(len(np.unique(yfit))==2,"fit binary collapse")
    m=diagnostic_model();m.fit(Xfit,yfit)
    p=m.predict_proba(Xval)[:,1]
    return {"auc":auc(yval,p),"rows":int(len(yval)),
            "positive":int(np.sum(yval))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fold-root",type=Path,required=True)
    ap.add_argument("--val-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.val_fold in FOLDS,"bad fold")
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    reports=discover_reports(a.fold_root)
    fold_dir=reports[a.val_fold][0].parent
    uw=fold_dir/"uniform.weights.h5";fw=fold_dir/"freeze_local_combo.weights.h5"
    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    fit,val=fold_ids(cache,cfg,a.val_fold)
    y=np.minimum(cache["exact"].astype(np.int32),6)
    yf=y[fit];yv=y[val]

    mu=build_model("learned_gate",SEED);mu.load_weights(uw)
    mg=build_model("learned_gate",SEED);mg.load_weights(fw)
    bu=nested_base(mu);bg=nested_base(mg)
    ku,bu_bias=[np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg,bg_bias=[np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ii=np.asarray(NEURONS,np.int64)
    kg[:,ii]=ku[:,ii];bg_bias[ii]=bu_bias[ii]
    bg.get_layer("candidate_hidden1").set_weights([kg,bg_bias])

    Pf,Gf=predict(mg,cache,fit);Pv,Gv=predict(mg,cache,val)
    Xcf=structural_features(cache,fit,Pf);Xcv=structural_features(cache,val,Pv)
    Xof,names=temporal_onset_features(cache,fit)
    Xov,names2=temporal_onset_features(cache,val)
    require(names==names2,"feature names drift")
    fit_blow,val_blow,cs,ss,b_like,low=build_blow(Xcf,Xcv,yf,Gf,a.val_fold)

    pop_fit=fit_blow&(Gf==3)
    pop_val=val_blow&(Gv==3)
    # Strict K2-vs-K3 comparison.
    k23_fit=pop_fit&np.isin(yf,(2,3))
    k23_val=pop_val&np.isin(yv,(2,3))
    y23f=(yf[k23_fit]==2).astype(np.int32)
    y23v=(yv[k23_val]==2).astype(np.int32)
    require(len(y23f)>=40 and len(y23v)>=10,"small K23 population")

    # Action-realistic K2-vs-rest comparison.
    yrf=(yf[pop_fit]==2).astype(np.int32)
    yrv=(yv[pop_val]==2).astype(np.int32)

    variants={}
    for label,A,B in (
      ("current",Xcf,Xcv),
      ("onset",Xof,Xov),
      ("current_plus_onset",np.column_stack([Xcf,Xof]),np.column_stack([Xcv,Xov])),
    ):
        variants[label]={
          "k2_vs_k3":eval_variant(A[k23_fit],y23f,B[k23_val],y23v),
          "k2_vs_rest":eval_variant(A[pop_fit],yrf,B[pop_val],yrv)
        }

    # Univariate onset feature AUC; orientation learned from FIT only.
    univ=[]
    for j,name in enumerate(names):
        for task,tf,tv,yyf,yyv in (
          ("k2_vs_k3",k23_fit,k23_val,y23f,y23v),
          ("k2_vs_rest",pop_fit,pop_val,yrf,yrv),
        ):
            pos=Xof[tf,j][yyf==1];neg=Xof[tf,j][yyf==0]
            orient=1.0 if float(np.mean(pos))>=float(np.mean(neg)) else -1.0
            score=orient*Xov[tv,j]
            univ.append({
              "feature":name,"task":task,"orientation":int(orient),
              "auc":auc(yyv,score),
              "fit_mean_positive":float(np.mean(pos)),
              "fit_mean_negative":float(np.mean(neg))
            })

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_temporal_onset_feature_audit",
        "validation_fold":a.val_fold,
        "fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "base":"freeze_local_combo + hidden1 [42,52,61,64]",
        "population":"B_low and base prediction K3",
        "new_feature_family":"cluster-level temporal onset evidence from covered 31x64x3 spectral_map",
        "candidate_specific_F0_evidence":False,
        "diagnostic_classifier":"logistic C=1 class_weight=balanced; same for all variants",
        "automatic_promotion":False
      },
      "feature_names":names,
      "coarse_clusters":{str(k):v for k,v in cs.items()},
      "B_like_cluster":int(b_like),
      "B_subclusters":{str(k):v for k,v in ss.items()},
      "B_low_cluster":int(low),
      "populations":{
        "fit_baseK3":int(pop_fit.sum()),"val_baseK3":int(pop_val.sum()),
        "fit_K2_K3":int(k23_fit.sum()),"val_K2_K3":int(k23_val.sum()),
        "val_true_K2":int(np.sum(y23v==1)),"val_true_K3":int(np.sum(y23v==0))
      },
      "variants":variants,
      "univariate":univ
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=[
      f"# B_low temporal-onset feature audit — fold {a.val_fold}","",
      f"K2-vs-K3 validation rows: **{int(k23_val.sum())}** "
      f"(K2={int(np.sum(y23v))}, K3={int(len(y23v)-np.sum(y23v))}).","",
      "| variant | AUC K2 vs K3 | AUC K2 vs rest |",
      "|---|---:|---:|"
    ]
    for name in ("current","onset","current_plus_onset"):
        x=variants[name]
        lines.append(f"| {name} | {x['k2_vs_k3']['auc']:.3f} | {x['k2_vs_rest']['auc']:.3f} |")
    best=sorted([x for x in univ if x["task"]=="k2_vs_k3"],
                key=lambda x:(x["auc"] if x["auc"] is not None else -1),reverse=True)[:8]
    lines+=["","Best univariate onset features (K2 vs K3):","",
            "| feature | AUC | orientation |","|---|---:|---:|"]
    for x in best:
        lines.append(f"| {x['feature']} | {x['auc']:.3f} | {x['orientation']:+d} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
