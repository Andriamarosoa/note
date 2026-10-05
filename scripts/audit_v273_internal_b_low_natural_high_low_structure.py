"""Internal audit: what structurally distinguishes naturally high vs low
B_low/base-K3 cases, beyond absolute pitch?

Goal
----
The previous experiments showed:
  * natural high-register rows: better K2/K3 separability;
  * low rows shifted upward spectrally or at waveform level: separability worsens.

Therefore absolute frequency location is not sufficient. This audit compares
NATURAL high vs NATURAL low rows using only pitch-normalized / relative
structural descriptors from the same fine-STFT harmonic decomposition.

For each fold 0/1/2/4:
  1. replay robust base and reconstruct B_low using FIT only;
  2. select B_low + base K3 + true K in {2,3};
  3. define natural low/high by FIT tertiles of median_triplet_f0;
  4. derive relative descriptors that should be largely pitch-scale invariant;
  5. test:
       a) how well each descriptor separates natural high from low;
       b) within low and high separately, how well each descriptor separates
          true K2 vs K3;
       c) which descriptors have consistent K2/K3 direction across BOTH
          registers and across folds.

No outer fold 3. No promotion.
"""
from __future__ import annotations
import argparse,json,math
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

RELATIVE_FEATURES=(
  "best_pair_residual_ratio",
  "best_triplet_residual_ratio",
  "residual_drop_ratio",
  "third_coefficient_fraction",
  "third_unique_support_ratio",
  "third_shared_partial_overlap",
  "third_harmonic_relation_distance_cents",
  "third_nearest_spacing_cents",
  "triplet_span_octaves",
  "third_to_median_ratio_log2",
  "third_to_min_ratio_log2",
  "third_to_max_ratio_log2",
)

def auc(y,s):
    y=np.asarray(y,np.int32);s=np.asarray(s,np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def model():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=28331))
    ])

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
        r=extract_one(freq,x)
        if r is not None:
            r=dict(r)
            pair=float(r["best_pair_residual_ratio"])
            trip=float(r["best_triplet_residual_ratio"])
            r["residual_drop_ratio"]=(pair-trip)/(pair+1e-12)
            minf=float(r["min_triplet_f0"]);maxf=float(r["max_triplet_f0"])
            med=float(r["median_triplet_f0"]);third=float(r["third_f0"])
            r["triplet_span_octaves"]=math.log2(maxf/max(minf,1e-12))
            r["third_to_median_ratio_log2"]=math.log2(third/max(med,1e-12))
            r["third_to_min_ratio_log2"]=math.log2(third/max(minf,1e-12))
            r["third_to_max_ratio_log2"]=math.log2(third/max(maxf,1e-12))
        out.append(r)
    return out

def matrix(rows):
    good=np.asarray([x is not None for x in rows],bool)
    rr=[x for x in rows if x is not None]
    require(len(rr)>0,"no harmonic rows")
    keys=list(rr[0].keys())
    A=np.asarray([[r[k] for k in keys] for r in rr],np.float64)
    return good,rr,keys,A,{k:i for i,k in enumerate(keys)}

def fit_oriented_auc(yfit,xfit,yval,xval):
    require(len(xfit)==len(yfit) and len(xval)==len(yval),"shape mismatch")
    if len(np.unique(yfit))<2 or len(np.unique(yval))<2:return None
    pos=xfit[yfit==1];neg=xfit[yfit==0]
    if not len(pos) or not len(neg):return None
    orient=1.0 if np.mean(pos)>=np.mean(neg) else -1.0
    return {
      "auc":auc(yval,orient*xval),
      "orientation":int(orient),
      "fit_pos_mean":float(np.mean(pos)),
      "fit_neg_mean":float(np.mean(neg)),
    }

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
    gf,_,keys,Af,idx=matrix(rf);gv,_,keys2,Av,idx2=matrix(rv)
    require(keys==keys2,"key drift")
    yyfg=yyf[gf];yyvg=yyv[gv]

    med=idx["median_triplet_f0"]
    low_cut=float(np.quantile(Af[:,med],1/3))
    high_cut=float(np.quantile(Af[:,med],2/3))
    lowf=Af[:,med]<low_cut; lowv=Av[:,med]<low_cut
    highf=Af[:,med]>=high_cut; highv=Av[:,med]>=high_cut

    # 1) descriptors that distinguish natural high from low, ignoring labels.
    regf=np.concatenate([np.zeros(int(lowf.sum()),np.int32),np.ones(int(highf.sum()),np.int32)])
    regv=np.concatenate([np.zeros(int(lowv.sum()),np.int32),np.ones(int(highv.sum()),np.int32)])
    Xregf=np.vstack([Af[lowf],Af[highf]])
    Xregv=np.vstack([Av[lowv],Av[highv]])

    register_separation=[]
    k2k3={}
    for feat in RELATIVE_FEATURES:
        j=idx[feat]
        reg=fit_oriented_auc(regf,Xregf[:,j],regv,Xregv[:,j])
        low=fit_oriented_auc(yyfg[lowf],Af[lowf,j],yyvg[lowv],Av[lowv,j]) if lowv.sum()>=8 else None
        high=fit_oriented_auc(yyfg[highf],Af[highf,j],yyvg[highv],Av[highv,j]) if highv.sum()>=8 else None
        register_separation.append({"feature":feat,**(reg or {"auc":None,"orientation":None})})
        k2k3[feat]={"low":low,"high":high}

    # 2) fixed multivariate relative-feature model within each register.
    cols=[idx[k] for k in RELATIVE_FEATURES]
    mult={}
    for name,mf,mv in (("low",lowf,lowv),("high",highf,highv)):
        if mf.sum()>=20 and mv.sum()>=8 and len(np.unique(yyfg[mf]))==2 and len(np.unique(yyvg[mv]))==2:
            clf=model();clf.fit(Af[mf][:,cols],yyfg[mf])
            p=clf.predict_proba(Av[mv][:,cols])[:,1]
            mult[name]={"auc":auc(yyvg[mv],p),"fit_rows":int(mf.sum()),"val_rows":int(mv.sum())}
        else:
            mult[name]={"auc":None,"fit_rows":int(mf.sum()),"val_rows":int(mv.sum())}

    # 3) direction consistency: same K2-vs-K3 orientation in low and high fit sets.
    consistency=[]
    for feat in RELATIVE_FEATURES:
        j=idx[feat]
        def direction(mask):
            y0=yyfg[mask];x=Af[mask,j]
            if len(np.unique(y0))<2:return 0
            return 1 if np.mean(x[y0==1])>=np.mean(x[y0==0]) else -1
        dl=direction(lowf);dh=direction(highf)
        consistency.append({
          "feature":feat,
          "fit_low_direction":dl,
          "fit_high_direction":dh,
          "same_direction":bool(dl!=0 and dl==dh),
          "low_val_auc":None if k2k3[feat]["low"] is None else k2k3[feat]["low"]["auc"],
          "high_val_auc":None if k2k3[feat]["high"] is None else k2k3[feat]["high"]["auc"],
        })

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_natural_high_vs_low_structure",
        "validation_fold":a.val_fold,"fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "population":"B_low + base K3 + true K in {2,3}",
        "register_definition":"FIT tertiles of natural median_triplet_f0",
        "relative_features":list(RELATIVE_FEATURES),
        "automatic_promotion":False
      },
      "cuts_hz":{"low_upper":low_cut,"high_lower":high_cut},
      "rows":{"fit_low":int(lowf.sum()),"fit_high":int(highf.sum()),
              "val_low":int(lowv.sum()),"val_high":int(highv.sum())},
      "register_separation":register_separation,
      "k2k3_by_feature":k2k3,
      "multivariate":mult,
      "direction_consistency":consistency
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=[f"# Natural high vs low structural audit — fold {a.val_fold}","",
      f"Low/high cuts: **{low_cut:.1f} / {high_cut:.1f} Hz**.","",
      f"Relative-feature K2/K3 AUC: low **{mult['low']['auc'] if mult['low']['auc'] is not None else 'n/a'}**, high **{mult['high']['auc'] if mult['high']['auc'] is not None else 'n/a'}**.","",
      "| feature | high-vs-low AUC | K2/K3 low AUC | K2/K3 high AUC | same direction |",
      "|---|---:|---:|---:|---:|"]
    for feat in RELATIVE_FEATURES:
        rr=next(x for x in register_separation if x["feature"]==feat)
        cc=next(x for x in consistency if x["feature"]==feat)
        def ff(v):return "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {feat} | {ff(rr.get('auc'))} | {ff(cc['low_val_auc'])} | {ff(cc['high_val_auc'])} | {cc['same_direction']} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
