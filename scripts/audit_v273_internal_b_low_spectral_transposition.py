"""Internal spectral-transposition audit for low-register B_low/base-K3 cases.

Question:
  If harmonic evidence works better in the naturally high-register stratum,
  does moving the SAME low-register transition spectrum upward recover that
  discriminative signal?

This is a controlled magnitude-spectrum transposition experiment, not yet a
waveform phase-vocoder experiment. For factor r:
    X_shift(f) = X_original(f / r)
which is the magnitude-frequency effect of an ideal upward pitch shift.

Representations compared on exactly the same low-register rows:
  x1 original
  x2 (+12 semitones)
  x4 (+24 semitones)

For each representation, train and evaluate the SAME fixed logistic diagnostic
on the four previously promising harmonic features:
  best_pair_residual_ratio
  best_triplet_residual_ratio
  third_coefficient_fraction
  third_unique_support_ratio

Also test whether a model trained on NATURAL high-register FIT rows transfers
to the shifted low-register validation rows.

Outer fold 3 is never loaded.
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
    structural_features,
)
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    transition_spectrum,extract_one,build_blow,BASE_FEATURES,
)

FACTORS=(1.0,2.0,4.0)

def auc(y,s):
    y=np.asarray(y,np.int32);s=np.asarray(s,np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def model():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=28131))
    ])

def transpose_spectrum(freq,x,factor):
    if factor==1.0:return np.asarray(x,np.float64).copy()
    # Ideal upward magnitude warp: output frequency f receives source f/factor.
    y=np.interp(freq/factor,freq,x,left=0.0,right=0.0)
    # Remove sub-analysis-range energy created by interpolation at the boundary.
    y=np.maximum(y,0.0)
    y/=float(np.sum(y))+1e-12
    return y

def audio_views(cache,ids,dataset_dir):
    indexed=tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by={t.annotation_member:t for t in indexed};aud={}
    views={f:[] for f in FACTORS}
    for row in np.asarray(ids,np.int64):
        member=str(cache["members"][row])
        if member not in aud:
            t=by[member];wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
            aud[member]=np.asarray(wav.samples,np.float64)/32768.0
        freq,x=transition_spectrum(aud[member],int(cache["cluster_start_samples"][row]))
        for factor in FACTORS:
            views[factor].append(extract_one(freq,transpose_spectrum(freq,x,factor)))
    return views

def matrix(rows):
    good=np.asarray([x is not None for x in rows],bool)
    rr=[x for x in rows if x is not None]
    if not rr:return good,[],np.zeros((0,0)),{}
    keys=list(rr[0].keys())
    A=np.asarray([[r[k] for k in keys] for r in rr],np.float64)
    return good,keys,A,{k:i for i,k in enumerate(keys)}

def eval_same_rep(Af,yf,Av,yv,cols):
    require(len(np.unique(yf))==2 and len(np.unique(yv))==2,"binary collapse")
    m=model();m.fit(Af[:,cols],yf)
    p=m.predict_proba(Av[:,cols])[:,1]
    return auc(yv,p)

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
    require(len(idsf)>=40 and len(idsv)>=10,"small K2/K3 population")

    vf=audio_views(cache,idsf,a.dataset_dir)
    vv=audio_views(cache,idsv,a.dataset_dir)

    # Natural-register split is defined ONLY from original x1 FIT representation.
    g1f,k1f,A1f,i1f=matrix(vf[1.0]);g1v,k1v,A1v,i1v=matrix(vv[1.0])
    require(k1f==k1v and len(k1f)>0,"original keys drift")
    med_idx=i1f["median_triplet_f0"]
    low_cut=float(np.quantile(A1f[:,med_idx],1/3))
    high_cut=float(np.quantile(A1f[:,med_idx],2/3))

    # Map original valid rows back to population indices.
    pop_index_f=np.flatnonzero(g1f);pop_index_v=np.flatnonzero(g1v)
    low_orig_f=A1f[:,med_idx]<low_cut
    low_orig_v=A1v[:,med_idx]<low_cut
    high_orig_f=A1f[:,med_idx]>=high_cut
    high_orig_v=A1v[:,med_idx]>=high_cut

    results={}
    # Natural high baseline.
    cols=[i1f[k] for k in BASE_FEATURES]
    high_auc=None
    if high_orig_f.sum()>=20 and high_orig_v.sum()>=8 and len(np.unique(yyf[pop_index_f][high_orig_f]))==2 and len(np.unique(yyv[pop_index_v][high_orig_v]))==2:
        high_auc=eval_same_rep(A1f[high_orig_f],yyf[pop_index_f][high_orig_f],
                               A1v[high_orig_v],yyv[pop_index_v][high_orig_v],cols)

    for factor in FACTORS:
        gf,keysf,Af,idxf=matrix(vf[factor]);gv,keysv,Av,idxv=matrix(vv[factor])
        require(keysf==keysv and keysf==k1f,"transposed keys drift")
        # Intersect representation-valid rows with rows defined low by ORIGINAL x1.
        global_f=np.flatnonzero(gf);global_v=np.flatnonzero(gv)
        low_population_ids_f=set(pop_index_f[low_orig_f].tolist())
        low_population_ids_v=set(pop_index_v[low_orig_v].tolist())
        mf=np.asarray([q in low_population_ids_f for q in global_f],bool)
        mv=np.asarray([q in low_population_ids_v for q in global_v],bool)
        yff=yyf[global_f][mf];yvv=yyv[global_v][mv]
        c=[idxf[k] for k in BASE_FEATURES]
        score=None
        if mf.sum()>=20 and mv.sum()>=8 and len(np.unique(yff))==2 and len(np.unique(yvv))==2:
            score=eval_same_rep(Af[mf],yff,Av[mv],yvv,c)

        # Transfer: train on NATURAL high x1, test shifted low in this representation.
        transfer=None
        if factor>1 and high_orig_f.sum()>=20 and mv.sum()>=8 and len(np.unique(yyf[pop_index_f][high_orig_f]))==2 and len(np.unique(yvv))==2:
            tm=model();tm.fit(A1f[high_orig_f][:,cols],yyf[pop_index_f][high_orig_f])
            p=tm.predict_proba(Av[mv][:,c])[:,1]
            transfer=auc(yvv,p)

        # Track where the low rows land after shift.
        med_shift=Av[mv,idxf["median_triplet_f0"]] if mv.any() else np.asarray([])
        results[f"x{int(factor)}"]={
          "factor":factor,
          "semitones":float(12*np.log2(factor)),
          "low_fit_rows":int(mf.sum()),"low_val_rows":int(mv.sum()),
          "same_representation_auc":score,
          "natural_high_model_transfer_auc":transfer,
          "shifted_low_median_triplet_f0_mean":float(np.mean(med_shift)) if len(med_shift) else None,
          "fraction_shifted_low_above_original_high_cut":float(np.mean(med_shift>=high_cut)) if len(med_shift) else None,
        }

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_low_to_high_spectral_transposition",
        "validation_fold":a.val_fold,"fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "population":"B_low + base K3 + true K in {2,3}; low register defined by original FIT bottom tertile",
        "transposition":"magnitude spectral warp X_shift(f)=X(f/factor)",
        "factors":[1,2,4],
        "features":list(BASE_FEATURES),
        "classifier":"logistic C=1 class_weight=balanced",
        "automatic_promotion":False
      },
      "register_cuts_hz":{"low_upper":low_cut,"high_lower":high_cut},
      "natural_high_auc":high_auc,
      "results":results
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=[f"# Low→high spectral transposition — fold {a.val_fold}","",
      f"Original low/high cuts: **{low_cut:.1f} / {high_cut:.1f} Hz**.",
      f"Natural-high AUC: **{'n/a' if high_auc is None else f'{high_auc:.3f}'}**.","",
      "| representation | same low rows AUC | high-model transfer AUC | shifted low above old high cut |",
      "|---|---:|---:|---:|"]
    for key in ("x1","x2","x4"):
        x=results[key]
        def ff(v):return "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {key} | {ff(x['same_representation_auc'])} | {ff(x['natural_high_model_transfer_auc'])} | "
                     f"{ff(x['fraction_shifted_low_above_original_high_cut'])} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
