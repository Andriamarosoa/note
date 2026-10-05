"""Internal waveform pitch-shift audit for low-register B_low/base-K3 cases.

This is the waveform-level follow-up to the spectral-warp experiment.

For the SAME naturally low-register rows, create three audio views:
  x1  : original waveform
  x2  : +12 semitones, duration-preserving waveform pitch shift
  x4  : +24 semitones, duration-preserving waveform pitch shift

The pitch shift is applied to a local pre+post waveform around the candidate
group start, then the fine-STFT transition and harmonic decomposition are
recomputed from the transformed audio. This tests whether the earlier negative
spectral-warp result was an artifact of moving magnitudes without a waveform
transform.

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

from causal_note.guitarset import ALLOWED_PLAYERS,SAMPLE_RATE,index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS,NEURONS,SEED,discover_reports,nested_base,predict,fold_ids,
    structural_features,
)
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    extract_one,build_blow,BASE_FEATURES,
)

FFT_SIZE=8192
WINDOW=2048
MIN_HZ=65.0
MAX_ANALYSIS_HZ=6000.0
VIEWS=(("x1",0.0),("x2",12.0),("x4",24.0))
PAD=4096

def auc(y,s):
    y=np.asarray(y,np.int32);s=np.asarray(s,np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def model():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=28231))
    ])

def pcm_window(samples,start,length):
    out=np.zeros(length,np.float64)
    l=max(0,int(start));r=min(len(samples),int(start)+length)
    if r>l:
        out[l-int(start):l-int(start)+r-l]=samples[l:r]
    return out

def pitch_shift_local(samples,start,n_steps):
    """Return duration-preserving shifted local signal and onset index."""
    import librosa
    # Extra margin reduces boundary artifacts from the phase vocoder/resampler.
    left=PAD+WINDOW
    right=PAD+WINDOW
    local=pcm_window(samples,int(start)-left,left+right)
    if n_steps:
        shifted=librosa.effects.pitch_shift(
            local.astype(np.float32),
            sr=SAMPLE_RATE,
            n_steps=float(n_steps),
            bins_per_octave=12,
            res_type="soxr_hq",
            scale=False,
        ).astype(np.float64)
    else:
        shifted=local.astype(np.float64)
    if len(shifted)<len(local):
        shifted=np.pad(shifted,(0,len(local)-len(shifted)))
    elif len(shifted)>len(local):
        shifted=shifted[:len(local)]
    return shifted,left

def transition_from_local(local,onset):
    taper=np.hanning(WINDOW)
    pre=pcm_window(local,onset-WINDOW,WINDOW)*taper
    post=pcm_window(local,onset,WINDOW)*taper
    P=np.abs(np.fft.rfft(pre,n=FFT_SIZE))**2
    Q=np.abs(np.fft.rfft(post,n=FFT_SIZE))**2
    freq=np.fft.rfftfreq(FFT_SIZE,1.0/SAMPLE_RATE)
    keep=(freq>=MIN_HZ)&(freq<=MAX_ANALYSIS_HZ)
    freq=freq[keep];P=P[keep];Q=Q[keep]
    X=np.maximum(Q-P,0.0)
    X/=float(np.sum(X))+1e-12
    return freq,X

def audio_views(cache,ids,dataset_dir):
    indexed=tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by={t.annotation_member:t for t in indexed}
    aud={}
    views={name:[] for name,_ in VIEWS}
    for row in np.asarray(ids,np.int64):
        member=str(cache["members"][row])
        if member not in aud:
            t=by[member]
            wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
            aud[member]=np.asarray(wav.samples,np.float64)/32768.0
        start=int(cache["cluster_start_samples"][row])
        for name,steps in VIEWS:
            local,onset=pitch_shift_local(aud[member],start,steps)
            freq,x=transition_from_local(local,onset)
            views[name].append(extract_one(freq,x))
    return views

def matrix(rows):
    good=np.asarray([x is not None for x in rows],bool)
    rr=[x for x in rows if x is not None]
    if not rr:return good,[],np.zeros((0,0)),{}
    keys=list(rr[0].keys())
    A=np.asarray([[r[k] for k in keys] for r in rr],np.float64)
    return good,keys,A,{k:i for i,k in enumerate(keys)}

def eval_same(Af,yf,Av,yv,cols):
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
    require(a.val_fold in FOLDS,"bad fold")
    require(not a.output.exists(),"refusing overwrite")
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

    popf=fb&(Gf==3)&np.isin(yf,(2,3))
    popv=vb&(Gv==3)&np.isin(yv,(2,3))
    idsf=fit[popf];idsv=val[popv]
    yyf=(yf[popf]==2).astype(np.int32);yyv=(yv[popv]==2).astype(np.int32)
    require(len(idsf)>=40 and len(idsv)>=10,"small K2/K3 population")

    vf=audio_views(cache,idsf,a.dataset_dir)
    vv=audio_views(cache,idsv,a.dataset_dir)

    g1f,k1f,A1f,i1f=matrix(vf["x1"])
    g1v,k1v,A1v,i1v=matrix(vv["x1"])
    require(k1f==k1v and len(k1f)>0,"original keys drift")

    med_idx=i1f["median_triplet_f0"]
    low_cut=float(np.quantile(A1f[:,med_idx],1/3))
    high_cut=float(np.quantile(A1f[:,med_idx],2/3))

    pop_index_f=np.flatnonzero(g1f);pop_index_v=np.flatnonzero(g1v)
    low_orig_f=A1f[:,med_idx]<low_cut
    low_orig_v=A1v[:,med_idx]<low_cut
    high_orig_f=A1f[:,med_idx]>=high_cut
    high_orig_v=A1v[:,med_idx]>=high_cut

    cols=[i1f[k] for k in BASE_FEATURES]
    natural_high_auc=None
    if (high_orig_f.sum()>=20 and high_orig_v.sum()>=8
        and len(np.unique(yyf[pop_index_f][high_orig_f]))==2
        and len(np.unique(yyv[pop_index_v][high_orig_v]))==2):
        natural_high_auc=eval_same(
            A1f[high_orig_f],yyf[pop_index_f][high_orig_f],
            A1v[high_orig_v],yyv[pop_index_v][high_orig_v],cols
        )

    results={}
    for name,steps in VIEWS:
        gf,keysf,Af,idxf=matrix(vf[name])
        gv,keysv,Av,idxv=matrix(vv[name])
        require(keysf==keysv and keysf==k1f,"waveform view keys drift")
        global_f=np.flatnonzero(gf);global_v=np.flatnonzero(gv)
        low_ids_f=set(pop_index_f[low_orig_f].tolist())
        low_ids_v=set(pop_index_v[low_orig_v].tolist())
        mf=np.asarray([q in low_ids_f for q in global_f],bool)
        mv=np.asarray([q in low_ids_v for q in global_v],bool)
        yff=yyf[global_f][mf];yvv=yyv[global_v][mv]
        c=[idxf[k] for k in BASE_FEATURES]

        same=None
        if (mf.sum()>=20 and mv.sum()>=8 and
            len(np.unique(yff))==2 and len(np.unique(yvv))==2):
            same=eval_same(Af[mf],yff,Av[mv],yvv,c)

        transfer=None
        if name!="x1" and high_orig_f.sum()>=20 and mv.sum()>=8 and len(np.unique(yvv))==2:
            tm=model()
            tm.fit(A1f[high_orig_f][:,cols],yyf[pop_index_f][high_orig_f])
            p=tm.predict_proba(Av[mv][:,c])[:,1]
            transfer=auc(yvv,p)

        shifted_med=Av[mv,idxf["median_triplet_f0"]] if mv.any() else np.asarray([])
        results[name]={
          "n_steps":steps,
          "factor":float(2**(steps/12.0)),
          "low_fit_rows":int(mf.sum()),
          "low_val_rows":int(mv.sum()),
          "same_representation_auc":same,
          "natural_high_model_transfer_auc":transfer,
          "shifted_low_median_triplet_f0_mean":float(np.mean(shifted_med)) if len(shifted_med) else None,
          "fraction_shifted_low_above_original_high_cut":float(np.mean(shifted_med>=high_cut)) if len(shifted_med) else None,
        }

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_waveform_pitch_shift",
        "validation_fold":a.val_fold,
        "fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "population":"B_low + base K3 + true K in {2,3}; low register defined from original FIT x1 bottom tertile",
        "pitch_shift":"librosa duration-preserving waveform pitch_shift on local pre+post audio",
        "views":{"x1":0,"x2":12,"x4":24},
        "features":list(BASE_FEATURES),
        "classifier":"logistic C=1 class_weight=balanced",
        "automatic_promotion":False
      },
      "register_cuts_hz":{"low_upper":low_cut,"high_lower":high_cut},
      "natural_high_auc":natural_high_auc,
      "results":results
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=[f"# Waveform pitch shift — fold {a.val_fold}","",
      f"Original low/high cuts: **{low_cut:.1f} / {high_cut:.1f} Hz**.",
      f"Natural-high AUC: **{'n/a' if natural_high_auc is None else f'{natural_high_auc:.3f}'}**.","",
      "| view | same low rows AUC | high-model transfer AUC | shifted low above old high cut |",
      "|---|---:|---:|---:|"]
    for name in ("x1","x2","x4"):
        x=results[name]
        def ff(v):return "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {name} | {ff(x['same_representation_auc'])} | "
                     f"{ff(x['natural_high_model_transfer_auc'])} | "
                     f"{ff(x['fraction_shifted_low_above_original_high_cut'])} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
