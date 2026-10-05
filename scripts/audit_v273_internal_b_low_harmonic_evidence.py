"""Internal fine-STFT harmonic evidence audit for B_low/base-K3.

Implements a diagnostic approximation of README features:
  F1 reconstruction_gain_2_to_3
  F3 missing_partial_penalty
plus third-source marginal/unique-support descriptors.

Important:
- exact-K counts assigned onset references, not sustained-note F0 count;
- therefore the spectrum is the positive PRE->POST power transition around the
  candidate-group start, not the raw sustained spectrum;
- F0 hypotheses are estimated from audio only; annotations are never used to
  create or complete the F0 pool;
- outer fold 3 is never loaded.

This is a feature audit, not a production corrector.
"""
from __future__ import annotations
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import ALLOWED_PLAYERS,SAMPLE_RATE,index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS,NEURONS,SEED,discover_reports,nested_base,predict,fold_ids,
    structural_features,
)
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require

FFT_SIZE=8192
WINDOW=2048
MIN_HZ=65.0
MAX_F0_HZ=1800.0
MAX_ANALYSIS_HZ=6000.0
MAX_HARMONICS=10
F0_GRID=np.geomspace(MIN_HZ,MAX_F0_HZ,240)
POOL_SIZE=8
NMS_CENTS=55.0
KERNEL_HZ=18.0

FEATURE_NAMES=(
  "reconstruction_gain_2_to_3",
  "best_pair_residual_ratio",
  "best_triplet_residual_ratio",
  "third_marginal_gain",
  "third_marginal_fraction_of_total_gain",
  "third_missing_partial_penalty",
  "third_observed_harmonic_fraction",
  "third_unique_support_ratio",
  "third_coefficient_fraction",
  "best_triplet_min_pair_cents",
)

def auc(y,s):
    y=np.asarray(y,np.int32);s=np.asarray(s,np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def diagnostic_model():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=27831))
    ])

def pcm_window(samples,start,length):
    out=np.zeros(length,np.float64)
    l=max(0,int(start));r=min(len(samples),int(start)+length)
    if r>l:
        out[l-int(start):l-int(start)+r-l]=samples[l:r]
    return out

def transition_spectrum(samples,start):
    taper=np.hanning(WINDOW)
    pre=pcm_window(samples,start-WINDOW,WINDOW)*taper
    post=pcm_window(samples,start,WINDOW)*taper
    P=np.abs(np.fft.rfft(pre,n=FFT_SIZE))**2
    Q=np.abs(np.fft.rfft(post,n=FFT_SIZE))**2
    norm=max(float(np.sum(taper*taper)),1e-12)
    P/=norm;Q/=norm
    freq=np.fft.rfftfreq(FFT_SIZE,1.0/SAMPLE_RATE)
    keep=(freq>=MIN_HZ)&(freq<=MAX_ANALYSIS_HZ)
    freq=freq[keep];P=P[keep];Q=Q[keep]
    # Positive transition isolates newly arriving acoustic evidence.
    X=np.maximum(Q-P,0.0)
    scale=float(np.sum(X))+1e-12
    return freq,X/scale,P/(float(np.sum(P))+1e-12)

def harmonic_salience(freq,x,f0):
    s=0.0
    for h in range(1,MAX_HARMONICS+1):
        hz=h*f0
        if hz>freq[-1]:break
        j=int(np.argmin(np.abs(freq-hz)))
        lo=max(0,j-2);hi=min(len(freq),j+3)
        s+=float(np.max(x[lo:hi]))/math.sqrt(h)
    return s

def f0_pool(freq,x):
    score=np.asarray([harmonic_salience(freq,x,float(f)) for f in F0_GRID])
    order=np.argsort(-score)
    chosen=[]
    for idx in order:
        f=float(F0_GRID[idx])
        if any(abs(1200.0*math.log2(f/g))<NMS_CENTS for g in chosen):
            continue
        chosen.append(f)
        if len(chosen)>=POOL_SIZE:break
    return chosen

def raw_template(freq,f0):
    d=np.zeros(len(freq),np.float64)
    harmonic_areas=[]
    for h in range(1,MAX_HARMONICS+1):
        hz=h*f0
        if hz>freq[-1]:break
        w=1.0/math.sqrt(h)
        g=np.exp(-0.5*((freq-hz)/KERNEL_HZ)**2)*w
        d+=g
        harmonic_areas.append((hz,w,g))
    norm=np.linalg.norm(d)+1e-12
    return d/norm,harmonic_areas,norm

def nonnegative_cost(G,b,x2,indices):
    """Exact NNLS for <=3 columns by enumerating active subsets."""
    best=(x2,np.zeros(len(indices),np.float64))
    k=len(indices)
    for r in range(1,k+1):
      for active_local in itertools.combinations(range(k),r):
        active=[indices[j] for j in active_local]
        GG=G[np.ix_(active,active)]
        bb=b[active]
        try:a=np.linalg.solve(GG+1e-10*np.eye(len(active)),bb)
        except np.linalg.LinAlgError:continue
        if np.any(a<0):continue
        cost=float(x2-2*np.dot(a,bb)+a@GG@a)
        full=np.zeros(k,np.float64)
        for q,v in zip(active_local,a):full[q]=v
        if cost<best[0]:best=(max(cost,0.0),full)
    return best

def fit_best(D,x,k):
    G=D.T@D;b=D.T@x;x2=float(x@x)
    best=None
    for combo in itertools.combinations(range(D.shape[1]),k):
        cost,a=nonnegative_cost(G,b,x2,list(combo))
        if best is None or cost<best[0]:
            best=(cost,combo,a)
    return best,G,b,x2

def pair_cents(fs):
    vals=[]
    for a,b in itertools.combinations(fs,2):
        vals.append(abs(1200.0*math.log2(a/b)))
    return min(vals) if vals else 0.0

def harmonic_features(freq,x,pre):
    pool=f0_pool(freq,x)
    if len(pool)<3:
        return np.full(len(FEATURE_NAMES),np.nan,np.float64)
    templates=[];parts=[];raw_norm=[]
    for f in pool:
        d,p,n=raw_template(freq,f);templates.append(d);parts.append(p);raw_norm.append(n)
    D=np.column_stack(templates)
    pair,G,b,x2=fit_best(D,x,2)
    trip,_,_,_=fit_best(D,x,3)
    J2,c2,a2=pair;J3,c3,a3=trip
    gain=max(J2-J3,0.0)
    total=max(x2,1e-12)

    # Candidate whose removal loses least reconstruction quality = weakest third source.
    remove=[]
    for local in range(3):
        keep=[c3[j] for j in range(3) if j!=local]
        cost,_=nonnegative_cost(G,b,x2,keep)
        remove.append((max(cost-J3,0.0),local,cost))
    third_gain,weak_local,_=min(remove,key=lambda z:z[0])
    weak_idx=c3[weak_local]
    coeff=float(a3[weak_local])
    coeff_sum=float(np.sum(a3))+1e-12
    fweak=pool[weak_idx]

    # F3 continuous missing-partial penalty using the fitted weak component.
    expected=[];observed=[];unique_obs=[]
    other=[c3[j] for j in range(3) if j!=weak_local]
    for hz,w,graw in parts[weak_idx]:
        band=np.abs(freq-hz)<=max(KERNEL_HZ,0.012*hz)
        if not np.any(band):continue
        # Expected band energy in the normalized template scale.
        g=templates[weak_idx][band]
        exp=coeff*float(np.sum(g))
        obs=float(np.sum(x[band]))
        # Unique support discounts bins strongly covered by the other templates.
        oth=np.zeros(np.sum(band),np.float64)
        for oi in other:oth=np.maximum(oth,templates[oi][band])
        own=templates[weak_idx][band]
        uniqueness=np.clip(1.0-oth/(own+1e-8),0.0,1.0)
        uobs=float(np.sum(x[band]*uniqueness))
        expected.append(exp);observed.append(obs);unique_obs.append(uobs)
    if not expected:
        miss=1.0;obs_frac=0.0;uniq=0.0
    else:
        E=np.asarray(expected);O=np.asarray(observed);U=np.asarray(unique_obs)
        miss=float(np.sum(np.maximum(E-O,0.0))/(np.sum(E)+1e-12))
        obs_frac=float(np.mean(O>=0.25*np.maximum(E,1e-12)))
        uniq=float(np.sum(U)/(np.sum(O)+1e-12))

    fs=[pool[i] for i in c3]
    return np.asarray([
      gain/total,
      J2/total,
      J3/total,
      third_gain/total,
      third_gain/(gain+1e-12),
      miss,
      obs_frac,
      uniq,
      coeff/coeff_sum,
      pair_cents(fs),
    ],np.float64)

def audio_features(cache,ids,dataset_dir):
    indexed=tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by={t.annotation_member:t for t in indexed}
    audio_cache={}
    out=np.zeros((len(ids),len(FEATURE_NAMES)),np.float64)
    for q,row in enumerate(np.asarray(ids,np.int64)):
        member=str(cache["members"][row])
        if member not in audio_cache:
            t=by[member]
            wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
            audio_cache[member]=np.asarray(wav.samples,np.float64)/32768.0
        start=int(cache["cluster_start_samples"][row])
        freq,x,pre=transition_spectrum(audio_cache[member],start)
        out[q]=harmonic_features(freq,x,pre)
    return out

def build_blow(Xf,Xv,yf,Gf,val_fold):
    fail=np.isin(yf,(2,3,4))&(Gf!=yf)
    coarse=Pipeline([("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370+val_fold))])
    coarse.fit(Xf[fail]);cf=coarse.predict(Xf);cv=coarse.predict(Xv)
    stats={}
    for c in (0,1):
        m=fail&(cf==c);under=int(np.sum(Gf[m]<yf[m]))
        stats[c]=(int(m.sum()),float(under/max(1,int(m.sum()))))
    a=max((0,1),key=lambda c:(stats[c][1],stats[c][0]));b=1-a
    bf=fail&(cf==b)
    sub=Pipeline([("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=80,random_state=27420+val_fold))])
    sub.fit(Xf[bf]);sf=sub.predict(Xf);sv=sub.predict(Xv)
    ss={}
    for c in (0,1):
        m=bf&(sf==c);under=int(np.sum(Gf[m]<yf[m]))
        ss[c]=(int(m.sum()),float(under/max(1,int(m.sum()))))
    low=max((0,1),key=lambda c:(ss[c][1],ss[c][0]))
    return (cf==b)&(sf==low),(cv==b)&(sv==low)

def eval_model(Xf,yf,Xv,yv):
    goodf=np.isfinite(Xf).all(1);goodv=np.isfinite(Xv).all(1)
    require(int(goodf.sum())>=30 and int(goodv.sum())>=8,"too few valid harmonic rows")
    m=diagnostic_model();m.fit(Xf[goodf],yf[goodf])
    p=m.predict_proba(Xv[goodv])[:,1]
    return {"auc":auc(yv[goodv],p),"valid_rows":int(goodv.sum()),
            "positive":int(np.sum(yv[goodv]))}

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
    yhf=(yf[popf]==2).astype(np.int32);yhv=(yv[popv]==2).astype(np.int32)
    require(len(idsf)>=40 and len(idsv)>=10,"small population")

    Xhf=audio_features(cache,idsf,a.dataset_dir)
    Xhv=audio_features(cache,idsv,a.dataset_dir)
    current_f=Xcf[popf];current_v=Xcv[popv]

    variants={
      "current":eval_model(current_f,yhf,current_v,yhv),
      "harmonic":eval_model(Xhf,yhf,Xhv,yhv),
      "current_plus_harmonic":eval_model(np.column_stack([current_f,Xhf]),yhf,
                                         np.column_stack([current_v,Xhv]),yhv)
    }
    univ=[]
    for j,name in enumerate(FEATURE_NAMES):
        goodf=np.isfinite(Xhf[:,j]);goodv=np.isfinite(Xhv[:,j])
        pos=Xhf[goodf,j][yhf[goodf]==1];neg=Xhf[goodf,j][yhf[goodf]==0]
        if not len(pos) or not len(neg):continue
        orient=1 if np.mean(pos)>=np.mean(neg) else -1
        univ.append({"feature":name,"auc":auc(yhv[goodv],orient*Xhv[goodv,j]),
                     "orientation":orient,"valid_rows":int(goodv.sum())})

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_fine_STFT_harmonic_evidence",
        "validation_fold":a.val_fold,"fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "population":"B_low + base K3 + true K in {2,3}",
        "audio_feature_source":"positive pre-to-post fine-STFT transition",
        "f0_pool":"audio-only harmonic salience, 65-1800Hz, no annotation completion",
        "features":["F1 reconstruction gain 2->3","F3 missing partial penalty",
                    "third-source marginal/unique support diagnostics"],
        "automatic_promotion":False
      },
      "rows":{"fit":int(len(idsf)),"val":int(len(idsv)),
              "val_K2":int(np.sum(yhv)),"val_K3":int(len(yhv)-np.sum(yhv))},
      "feature_names":list(FEATURE_NAMES),
      "variants":variants,"univariate":univ
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=[f"# Fine-STFT harmonic evidence — fold {a.val_fold}","",
      f"Validation K2/K3 rows: **{len(idsv)}** (K2={int(np.sum(yhv))}, K3={int(len(yhv)-np.sum(yhv))}).","",
      "| variant | AUC | valid rows |","|---|---:|---:|"]
    for n in ("current","harmonic","current_plus_harmonic"):
        x=variants[n];lines.append(f"| {n} | {x['auc']:.3f} | {x['valid_rows']} |")
    lines+=["","Univariate harmonic features:","","| feature | AUC |","|---|---:|"]
    for x in sorted(univ,key=lambda z:z["auc"],reverse=True):
        lines.append(f"| {x['feature']} | {x['auc']:.3f} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
