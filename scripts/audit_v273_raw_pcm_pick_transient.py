"""Raw-PCM pick/transient diagnostic for the frozen residual K2/K3 cohort.

Question: can short broadband pick/strike transients distinguish the 108 K2
cases corrected by the residual guard from the 125 true-K3 cases it regresses?

This is diagnostic only: folds 0/1/2/4, no fold 3, no Exact-K prediction change.
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav

FOLDS=(0,1,2,4)
GROUPS=("K2_corrected","K3_regressed")
PRE_MS=5.0
POST_MS=50.0
FRAME=64
HOP=16
EPS=1e-12

def require(cond,msg):
    if not cond: raise RuntimeError(msg)

def extract_segment(samples, center):
    a=int(round(center-PRE_MS*SAMPLE_RATE/1000))
    b=int(round(center+POST_MS*SAMPLE_RATE/1000))
    left=max(0,-a); right=max(0,b-len(samples))
    seg=np.pad(samples[max(0,a):min(len(samples),b)],(left,right))
    need=int(round((PRE_MS+POST_MS)*SAMPLE_RATE/1000))
    if len(seg)<need: seg=np.pad(seg,(0,need-len(seg)))
    return seg[:need].astype(np.float64,copy=False)

def frame_matrix(x):
    if len(x)<FRAME: x=np.pad(x,(0,FRAME-len(x)))
    n=1+(len(x)-FRAME)//HOP
    return np.stack([x[i*HOP:i*HOP+FRAME] for i in range(n)])

def z(v):
    v=np.asarray(v,np.float64)
    return (v-np.median(v))/(np.std(v)+EPS)

def count_peaks(v, prominence=.75, distance_ms=1.5):
    distance=max(1,int(round(distance_ms*SAMPLE_RATE/(1000*HOP))))
    p,_=find_peaks(z(v),prominence=prominence,distance=distance)
    return p

def safe_quantile(x,q):
    return float(np.quantile(x,q)) if len(x) else 0.0

def transient_features(seg):
    frames=frame_matrix(seg)
    win=np.hanning(FRAME)
    rms=np.sqrt(np.mean(frames**2,axis=1)+EPS)
    diff=np.diff(frames,axis=1,prepend=frames[:,:1])
    diff_rms=np.sqrt(np.mean(diff**2,axis=1)+EPS)

    spec=np.abs(np.fft.rfft(frames*win,n=256))**2
    freqs=np.fft.rfftfreq(256,1/SAMPLE_RATE)
    total=spec.sum(1)+EPS
    hi=spec[:,freqs>=2000].sum(1)/(total)
    centroid=(spec*freqs[None,:]).sum(1)/total
    flat=np.exp(np.mean(np.log(spec+EPS),axis=1))/(np.mean(spec+EPS,axis=1))
    flux=np.r_[0.0,np.sqrt(np.mean(np.maximum(spec[1:]-spec[:-1],0.0),axis=1))]

    times=(np.arange(len(rms))*HOP+FRAME/2)/SAMPLE_RATE*1000-PRE_MS
    post=(times>=0)&(times<=45)
    early=(times>=0)&(times<=20)
    pre=times<0

    pr=count_peaks(rms[post]); pd=count_peaks(diff_rms[post]); ph=count_peaks(hi[post]); pf=count_peaks(flux[post])
    def sep(peaks):
        if len(peaks)<2: return 0.0
        t=times[post][peaks]
        return float(np.median(np.diff(t)))
    x=seg[int(round(PRE_MS*SAMPLE_RATE/1000)):]
    absx=np.abs(x)
    crest=float(np.max(absx)/(np.sqrt(np.mean(x*x))+EPS))
    zcr=float(np.mean(x[1:]*x[:-1]<0)) if len(x)>1 else 0.0
    kurt=float(np.mean(((x-np.mean(x))/(np.std(x)+EPS))**4))
    raw_slope=float(np.max(np.diff(rms[post],prepend=rms[post][0]))/(HOP/SAMPLE_RATE*1000+EPS))

    return {
      "raw_peak_count_45ms":float(len(pr)),
      "diff_peak_count_45ms":float(len(pd)),
      "highband_peak_count_45ms":float(len(ph)),
      "flux_peak_count_45ms":float(len(pf)),
      "raw_peak_spacing_ms":sep(pr),
      "diff_peak_spacing_ms":sep(pd),
      "highband_peak_spacing_ms":sep(ph),
      "flux_peak_spacing_ms":sep(pf),
      "early20_rms_mean":float(rms[early].mean()),
      "early20_diff_rms_mean":float(diff_rms[early].mean()),
      "early20_highband_ratio":float(hi[early].mean()),
      "early20_centroid_hz":float(centroid[early].mean()),
      "early20_flatness":float(flat[early].mean()),
      "early20_flux_mean":float(flux[early].mean()),
      "early20_flux_max":float(flux[early].max()),
      "post45_diff_to_raw":float(diff_rms[post].mean()/(rms[post].mean()+EPS)),
      "pre_to_early_rms_ratio":float(rms[pre].mean()/(rms[early].mean()+EPS)),
      "raw_attack_max_slope":raw_slope,
      "wave_crest_factor":crest,
      "wave_zcr":zcr,
      "wave_kurtosis":kurt,
      "early20_hi_q90":safe_quantile(hi[early],.9),
      "early20_flatness_q90":safe_quantile(flat[early],.9),
    }

def numeric(v):
    a=np.asarray(v,np.float64)
    return {"n":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),
            "q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=Path,required=True)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    rows=[json.loads(x) for x in a.cases.read_text().splitlines()]
    rows=[r for r in rows if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in rows)==108,"K2 cohort drift")
    require(sum(r["group"]=="K3_regressed" for r in rows)==125,"K3 cohort drift")
    require(all(r["fold"] in FOLDS for r in rows),"forbidden fold")

    wanted={r["recording_id"] for r in rows}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage incomplete")
    by=defaultdict(list)
    for r in rows: by[r["recording_id"]].append(r)

    measured=[]
    for member in sorted(by):
        audio=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(audio.samples,np.float64)/32768.0
        for r in by[member]:
            feat=transient_features(extract_segment(samples,int(r["start_sample"])))
            measured.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],
                             "true_K":r["true_K"],"recording_id":member,"features":feat})
        print(json.dumps({"recording":member,"done":len(measured)}),flush=True)
    require(len(measured)==233,"measurement count drift")

    names=list(measured[0]["features"])
    X=np.asarray([[r["features"][n] for n in names] for r in measured],np.float64)
    y=np.asarray([1 if r["group"]=="K3_regressed" else 0 for r in measured],np.int64)
    folds=np.asarray([r["fold"] for r in measured],np.int64)
    require(np.isfinite(X).all(),"non-finite features")

    group_summary={}
    for g in GROUPS:
        idx=np.asarray([r["group"]==g for r in measured])
        group_summary[g]={n:numeric(X[idx,j]) for j,n in enumerate(names)}

    univariate=[]
    for j,n in enumerate(names):
        per={}
        for f in FOLDS:
            fit=folds!=f; val=folds==f
            med3=np.median(X[fit & (y==1),j]); med2=np.median(X[fit & (y==0),j])
            orient=1.0 if med3>=med2 else -1.0
            auc=roc_auc_score(y[val],orient*X[val,j]) if len(np.unique(y[val]))==2 else None
            per[str(f)]={"auc":None if auc is None else float(auc),"orientation":int(orient)}
        vals=[q["auc"] for q in per.values() if q["auc"] is not None]
        univariate.append({"feature":n,"mean_oof_auc":float(np.mean(vals)),
                           "min_oof_auc":float(np.min(vals)),"per_fold":per})
    univariate.sort(key=lambda q:q["mean_oof_auc"],reverse=True)

    multi=[]
    probs=np.zeros(len(y),np.float64)
    for f in FOLDS:
        fit=folds!=f; val=folds==f
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))
        clf.fit(X[fit],y[fit])
        probs[val]=clf.predict_proba(X[val])[:,1]
        multi.append({"fold":f,"auc":float(roc_auc_score(y[val],probs[val])),
                      "n":int(val.sum()),"k2":int(np.sum(val&(y==0))),"k3":int(np.sum(val&(y==1)))})
    multi_auc=float(roc_auc_score(y,probs))

    report={"status":"completed","experiment":"v273_raw_pcm_pick_transient_diagnostic",
            "cases":len(measured),"k2_corrected":int(np.sum(y==0)),"k3_regressed":int(np.sum(y==1)),
            "folds":list(FOLDS),"outer_fold_3_used":False,"prediction_changes":False,
            "window":{"pre_ms":PRE_MS,"post_ms":POST_MS,"frame_samples":FRAME,"hop_samples":HOP},
            "feature_names":names,"group_summary":group_summary,"univariate_oof":univariate,
            "multivariate_oof":{"model":"StandardScaler + LogisticRegression(C=0.1, balanced)",
                                "global_auc":multi_auc,"per_fold":multi}}
    a.output.mkdir(parents=True)
    (a.output/"cases.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in measured))
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=["# Raw-PCM pick/transient diagnostic","",
           f"Cohort: **108 K2 corrected vs 125 K3 regressed**. Fold 3 excluded.","",
           f"Multivariate out-of-fold AUC: **{multi_auc:.3f}**.","",
           "| fold | AUC | K2 | K3 |","|---:|---:|---:|---:|"]
    for q in multi: lines.append(f"| {q['fold']} | {q['auc']:.3f} | {q['k2']} | {q['k3']} |")
    lines += ["","## Best raw-transient features","",
              "| feature | mean OOF AUC | min fold AUC | K2 median | K3 median |",
              "|---|---:|---:|---:|---:|"]
    for q in univariate[:12]:
        n=q["feature"]
        lines.append(f"| {n} | {q['mean_oof_auc']:.3f} | {q['min_oof_auc']:.3f} | "
                     f"{group_summary['K2_corrected'][n]['median']:.6g} | {group_summary['K3_regressed'][n]['median']:.6g} |")
    lines += ["","Diagnostic only; no threshold selection, no Exact-K prediction change, no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__": main()
