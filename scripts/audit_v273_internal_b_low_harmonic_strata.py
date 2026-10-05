"""Stratified audit of promising harmonic evidence in B_low/base-K3.

Builds on the fine-STFT harmonic audit. The goal is to explain why the useful
signals (third_coefficient_fraction, pair/triplet residual ratios) vary strongly
across folds.

For each rotating internal fold:
  - replay robust base and reconstruct B_low on FIT only;
  - restrict to B_low + base K3 + true K in {2,3};
  - recompute fine-STFT harmonic decomposition;
  - attach structural strata derived only from audio-side F0 hypotheses:
      register, F0 spacing, harmonic-proximity risk, shared-partial overlap;
  - report per-stratum AUCs for the promising features and a fixed combined
    diagnostic model.

Outer fold 3 is never loaded.
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

BASE_FEATURES=(
  "best_pair_residual_ratio",
  "best_triplet_residual_ratio",
  "third_coefficient_fraction",
  "third_unique_support_ratio",
)

def auc(y,s):
    y=np.asarray(y,np.int32);s=np.asarray(s,np.float64)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def diag_model():
    return Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                               class_weight="balanced",random_state=27931))
    ])

def pcm_window(samples,start,length):
    out=np.zeros(length,np.float64)
    l=max(0,int(start));r=min(len(samples),int(start)+length)
    if r>l:out[l-int(start):l-int(start)+r-l]=samples[l:r]
    return out

def transition_spectrum(samples,start):
    taper=np.hanning(WINDOW)
    pre=pcm_window(samples,start-WINDOW,WINDOW)*taper
    post=pcm_window(samples,start,WINDOW)*taper
    P=np.abs(np.fft.rfft(pre,n=FFT_SIZE))**2
    Q=np.abs(np.fft.rfft(post,n=FFT_SIZE))**2
    freq=np.fft.rfftfreq(FFT_SIZE,1.0/SAMPLE_RATE)
    keep=(freq>=MIN_HZ)&(freq<=MAX_ANALYSIS_HZ)
    freq=freq[keep];P=P[keep];Q=Q[keep]
    X=np.maximum(Q-P,0.0)
    X/=float(np.sum(X))+1e-12
    return freq,X

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
    # Equal salience is common because neighbouring F0s use the same FFT bins.
    # Keep increasing grid order for ties; an unstable sort can change which
    # neighbour survives NMS and therefore the exported pair/triplet residuals.
    order=np.argsort(-score,kind="stable");chosen=[]
    for idx in order:
        f=float(F0_GRID[idx])
        if any(abs(1200.0*math.log2(f/g))<NMS_CENTS for g in chosen):continue
        chosen.append(f)
        if len(chosen)>=POOL_SIZE:break
    return chosen

def template(freq,f0):
    d=np.zeros(len(freq),np.float64)
    for h in range(1,MAX_HARMONICS+1):
        hz=h*f0
        if hz>freq[-1]:break
        d+=np.exp(-0.5*((freq-hz)/KERNEL_HZ)**2)/math.sqrt(h)
    return d/(np.linalg.norm(d)+1e-12)

def nonnegative_cost(G,b,x2,indices):
    best=(x2,np.zeros(len(indices)))
    k=len(indices)
    for r in range(1,k+1):
      for active_local in itertools.combinations(range(k),r):
        active=[indices[j] for j in active_local]
        GG=G[np.ix_(active,active)];bb=b[active]
        try:a=np.linalg.solve(GG+1e-10*np.eye(len(active)),bb)
        except np.linalg.LinAlgError:continue
        if np.any(a<0):continue
        cost=float(x2-2*np.dot(a,bb)+a@GG@a)
        full=np.zeros(k)
        for q,v in zip(active_local,a):full[q]=v
        if cost<best[0]:best=(max(cost,0.0),full)
    return best

def fit_best(D,x,k):
    G=D.T@D;b=D.T@x;x2=float(x@x);best=None
    for combo in itertools.combinations(range(D.shape[1]),k):
        cost,a=nonnegative_cost(G,b,x2,list(combo))
        if best is None or cost<best[0]:best=(cost,combo,a)
    return best,G,b,x2

def cents(a,b):
    return abs(1200.0*math.log2(float(a)/float(b)))

def harmonic_relation_distance(a,b):
    """Distance in cents to low-order harmonic ratios n/m, n,m<=6."""
    r=float(a)/float(b);best=1e9
    for n in range(1,7):
      for m in range(1,7):
        best=min(best,abs(1200.0*math.log2(r/(n/m))))
    return best

def shared_partial_overlap(fa,fb):
    aa=[h*fa for h in range(1,MAX_HARMONICS+1) if h*fa<=MAX_ANALYSIS_HZ]
    bb=[h*fb for h in range(1,MAX_HARMONICS+1) if h*fb<=MAX_ANALYSIS_HZ]
    if not aa or not bb:return 0.0
    shared=0
    for x in aa:
        if any(abs(x-y)<=max(KERNEL_HZ,0.01*x) for y in bb):shared+=1
    return shared/max(1,min(len(aa),len(bb)))

def extract_one(freq,x):
    pool=f0_pool(freq,x)
    if len(pool)<3:return None
    D=np.column_stack([template(freq,f) for f in pool])
    pair,G,b,x2=fit_best(D,x,2);trip,_,_,_=fit_best(D,x,3)
    J2,c2,a2=pair;J3,c3,a3=trip
    # weakest third = smallest reconstruction damage when removed
    rem=[]
    for local in range(3):
        keep=[c3[j] for j in range(3) if j!=local]
        cost,_=nonnegative_cost(G,b,x2,keep)
        rem.append((max(cost-J3,0.0),local))
    _,weak_local=min(rem,key=lambda z:z[0])
    weak_idx=c3[weak_local]
    coeff=float(a3[weak_local])/(float(np.sum(a3))+1e-12)
    fs=[pool[i] for i in c3]
    fweak=pool[weak_idx]
    others=[fs[j] for j in range(3) if j!=weak_local]
    overlap=max(shared_partial_overlap(fweak,o) for o in others)
    hprox=min(harmonic_relation_distance(fweak,o) for o in others)
    spacing=min(cents(fweak,o) for o in others)
    minf=min(fs);maxf=max(fs);medf=float(np.median(fs))
    return {
      "best_pair_residual_ratio":float(J2/(x2+1e-12)),
      "best_triplet_residual_ratio":float(J3/(x2+1e-12)),
      "third_coefficient_fraction":coeff,
      "third_unique_support_ratio":float(max(0.0,1.0-overlap)),
      "third_f0":float(fweak),
      "min_triplet_f0":float(minf),
      "max_triplet_f0":float(maxf),
      "median_triplet_f0":medf,
      "third_nearest_spacing_cents":float(spacing),
      "third_harmonic_relation_distance_cents":float(hprox),
      "third_shared_partial_overlap":float(overlap),
    }

def audio_rows(cache,ids,dataset_dir):
    indexed=tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by={t.annotation_member:t for t in indexed};aud={}
    out=[]
    for row in np.asarray(ids,np.int64):
        member=str(cache["members"][row])
        if member not in aud:
            t=by[member];wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
            aud[member]=np.asarray(wav.samples,np.float64)/32768.0
        freq,x=transition_spectrum(aud[member],int(cache["cluster_start_samples"][row]))
        out.append(extract_one(freq,x))
    return out

def build_blow(Xf,Xv,yf,Gf,val_fold):
    fail=np.isin(yf,(2,3,4))&(Gf!=yf)
    coarse=Pipeline([("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370+val_fold))])
    coarse.fit(Xf[fail]);cf=coarse.predict(Xf);cv=coarse.predict(Xv)
    cs={}
    for c in (0,1):
        m=fail&(cf==c);under=int(np.sum(Gf[m]<yf[m]))
        cs[c]=(int(m.sum()),float(under/max(1,int(m.sum()))))
    a=max((0,1),key=lambda c:(cs[c][1],cs[c][0]));b=1-a
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

def orient_auc(yfit,xfit,yval,xval):
    pos=xfit[yfit==1];neg=xfit[yfit==0]
    if not len(pos) or not len(neg) or len(np.unique(yval))<2:return None
    orient=1.0 if np.mean(pos)>=np.mean(neg) else -1.0
    return auc(yval,orient*xval)

def bins_from_fit(values,kind):
    v=np.asarray(values,np.float64)
    if kind=="register":
        q=np.quantile(v,[1/3,2/3])
        return [(-np.inf,q[0],"low"),(q[0],q[1],"mid"),(q[1],np.inf,"high")]
    if kind=="spacing":
        q=np.quantile(v,[1/3,2/3])
        return [(-np.inf,q[0],"close"),(q[0],q[1],"medium"),(q[1],np.inf,"wide")]
    if kind=="harmonic":
        return [(-np.inf,35.0,"near_harmonic"),(35.0,90.0,"moderate_harmonic"),(90.0,np.inf,"far_harmonic")]
    if kind=="overlap":
        q=np.quantile(v,[1/3,2/3])
        return [(-np.inf,q[0],"low_overlap"),(q[0],q[1],"mid_overlap"),(q[1],np.inf,"high_overlap")]
    raise ValueError(kind)

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
    popf=fb&(Gf==3)&np.isin(yf,(2,3));popv=vb&(Gv==3)&np.isin(yv,(2,3))
    idsf=fit[popf];idsv=val[popv];yyf=(yf[popf]==2).astype(np.int32);yyv=(yv[popv]==2).astype(np.int32)
    rf=audio_rows(cache,idsf,a.dataset_dir);rv=audio_rows(cache,idsv,a.dataset_dir)
    goodf=np.asarray([x is not None for x in rf]);goodv=np.asarray([x is not None for x in rv])
    rf=[x for x in rf if x is not None];rv=[x for x in rv if x is not None]
    yyfg=yyf[goodf];yyvg=yyv[goodv]
    require(len(rf)>=40 and len(rv)>=10,"small valid population")
    keys=list(rf[0].keys())
    Af=np.asarray([[r[k] for k in keys] for r in rf],np.float64)
    Av=np.asarray([[r[k] for k in keys] for r in rv],np.float64)
    idx={k:i for i,k in enumerate(keys)}

    # global combined diagnostic using the four promising features
    cols=[idx[k] for k in BASE_FEATURES]
    model=diag_model();model.fit(Af[:,cols],yyfg)
    pg=model.predict_proba(Av[:,cols])[:,1]
    global_auc=auc(yyvg,pg)

    strata={}
    specs={
      "register":("median_triplet_f0","register"),
      "spacing":("third_nearest_spacing_cents","spacing"),
      "harmonic_proximity":("third_harmonic_relation_distance_cents","harmonic"),
      "partial_overlap":("third_shared_partial_overlap","overlap"),
    }
    for sname,(field,kind) in specs.items():
        fi=idx[field];bins=bins_from_fit(Af[:,fi],kind);rows=[]
        for lo,hi,label in bins:
            mf=(Af[:,fi]>=lo)&(Af[:,fi]<hi)
            mv=(Av[:,fi]>=lo)&(Av[:,fi]<hi)
            if int(mv.sum())<8 or len(np.unique(yyvg[mv]))<2:
                rows.append({"label":label,"fit_rows":int(mf.sum()),"val_rows":int(mv.sum()),"combined_auc":None,"features":{}})
                continue
            m=diag_model();m.fit(Af[mf][:,cols],yyfg[mf]);p=m.predict_proba(Av[mv][:,cols])[:,1]
            feat={}
            for k in BASE_FEATURES:
                j=idx[k];feat[k]=orient_auc(yyfg[mf],Af[mf,j],yyvg[mv],Av[mv,j])
            rows.append({"label":label,"fit_rows":int(mf.sum()),"val_rows":int(mv.sum()),
                         "val_K2":int(np.sum(yyvg[mv])),"val_K3":int(np.sum(~yyvg[mv].astype(bool))),
                         "combined_auc":auc(yyvg[mv],p),"features":feat})
        strata[sname]=rows

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_harmonic_stratified_audit",
        "validation_fold":a.val_fold,"fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "population":"B_low + base K3 + true K in {2,3}",
        "strata":"fit-defined register/spacing/overlap; fixed harmonic-proximity bins",
        "automatic_promotion":False
      },
      "global":{"valid_fit_rows":len(rf),"valid_val_rows":len(rv),"combined_auc":global_auc},
      "feature_keys":keys,"promising_features":list(BASE_FEATURES),"strata":strata
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=[f"# Harmonic stratified audit — fold {a.val_fold}","",f"Global 4-feature AUC: **{global_auc:.3f}**.",""]
    for sname,rows in strata.items():
        lines += [f"## {sname}","", "| stratum | val rows | combined AUC | third coeff | pair residual | triplet residual |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            def ff(v):return "n/a" if v is None else f"{v:.3f}"
            lines.append(f"| {r['label']} | {r['val_rows']} | {ff(r['combined_auc'])} | "
                         f"{ff(r['features'].get('third_coefficient_fraction'))} | "
                         f"{ff(r['features'].get('best_pair_residual_ratio'))} | "
                         f"{ff(r['features'].get('best_triplet_residual_ratio'))} |")
        lines.append("")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
