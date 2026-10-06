"""Dense raw-PCM attack-event audit for residual K2/K3 cases.

Build a high-resolution novelty curve using the same local derivative-energy ratio
that was strong when centered on annotated onsets, then apply fixed temporal NMS.
This tests whether independent attack events can be recovered without V27.3 candidates.
Diagnostic annotations only measure event coverage and the physical simultaneity ceiling.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_note.guitarset import SAMPLE_RATE,index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav

FOLDS=(0,1,2,4); GROUPS=("K2_corrected","K3_regressed"); EPS=1e-12
PRE=round(.004*SAMPLE_RATE); POST=round(.010*SAMPLE_RATE); FRAME=64; HOP=16
SCAN_PRE=round(.005*SAMPLE_RATE); SCAN_POST=round(.050*SAMPLE_RATE)
STEP=8; SEPS_MS=(3.0,5.0,8.0)

def req(c,m):
    if not c:raise RuntimeError(m)

def local_score(x,t):
    a=max(0,t-PRE);pre=x[a:t]
    if len(pre)<PRE:pre=np.pad(pre,(PRE-len(pre),0))
    post=x[t:min(len(x),t+POST)]
    if len(post)<POST:post=np.pad(post,(0,POST-len(post)))
    dp=np.diff(pre,prepend=pre[:1]); base=np.sqrt(np.mean(dp*dp)+EPS)
    vals=[]
    for i in range(0,max(1,len(post)-FRAME+1),HOP):
        fr=post[i:i+FRAME]
        if len(fr)!=FRAME:continue
        d=np.diff(fr,prepend=fr[:1]);vals.append(np.sqrt(np.mean(d*d)+EPS))
    return float(max(vals)/(base+EPS)) if vals else 0.

def nms(times,scores,sep_ms):
    distance=max(1,round(sep_ms*SAMPLE_RATE/(1000*STEP)))
    peaks,_=find_peaks(scores,distance=distance)
    # include endpoints only if strict local max
    peaks=list(peaks)
    if len(scores)>1 and scores[0]>scores[1]:peaks.append(0)
    if len(scores)>1 and scores[-1]>scores[-2]:peaks.append(len(scores)-1)
    peaks=sorted(set(peaks),key=lambda i:scores[i],reverse=True)
    # scipy distance does not account for manually added endpoints, enforce exact NMS
    chosen=[]
    sep=round(sep_ms*SAMPLE_RATE/1000)
    for i in peaks:
        if all(abs(int(times[i])-int(times[j]))>=sep for j in chosen):chosen.append(i)
    return chosen

def feature_for(times,scores,sep):
    ids=nms(times,scores,sep); vals=sorted([float(scores[i]) for i in ids],reverse=True)
    v=(vals+[0.,0.,0.,0.])[:4]
    return {f"sep{sep:g}_top1":v[0],f"sep{sep:g}_top2":v[1],f"sep{sep:g}_top3":v[2],f"sep{sep:g}_top4":v[3],
            f"sep{sep:g}_top3_over_top2":float(v[2]/(v[1]+EPS)),f"sep{sep:g}_gap23":float(v[1]-v[2]),
            f"sep{sep:g}_count_gt1":float(sum(z>1.0 for z in vals)),f"sep{sep:g}_count_gt1p25":float(sum(z>1.25 for z in vals)),
            f"sep{sep:g}_count_gt1p5":float(sum(z>1.5 for z in vals))},ids

def distinct_true(onsets,sep_ms):
    s=sorted(map(int,onsets)); sep=round(sep_ms*SAMPLE_RATE/1000); groups=[]
    for t in s:
        if not groups or t-groups[-1][-1]>=sep:groups.append([t])
        else:groups[-1].append(t)
    return len(groups)

def main():
    p=argparse.ArgumentParser()
    for n in ("cases","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();req(not a.output.exists(),"refusing overwrite")
    cases=[json.loads(x) for x in a.cases.read_text().splitlines()];cases=[r for r in cases if r["group"] in GROUPS]
    req(sum(r["group"]=="K2_corrected" for r in cases)==108,"K2 drift");req(sum(r["group"]=="K3_regressed" for r in cases)==125,"K3 drift")
    wanted={r["recording_id"] for r in cases};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};req(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for r in cases:by[r["recording_id"]].append(r)
    rows=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);x=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            start=int(r["start_sample"]);times=np.arange(max(PRE,start-SCAN_PRE),min(len(x)-POST,start+SCAN_POST)+1,STEP,dtype=np.int64)
            scores=np.asarray([local_score(x,int(t)) for t in times],float)
            feat={};peaksets={}
            for sep in SEPS_MS:
                f,ids=feature_for(times,scores,sep);feat.update(f);peaksets[str(sep)]=[int(times[i]) for i in ids]
            true=[int(n["onset_sample"]) for n in r["owned_notes"]]
            physical={str(sep):distinct_true(true,sep) for sep in SEPS_MS}
            coverage={}
            for sep in SEPS_MS:
                pk=np.asarray(peaksets[str(sep)],int)
                coverage[str(sep)]=float(np.mean([np.min(np.abs(pk-t))*1000/SAMPLE_RATE<=sep if len(pk) else False for t in true]))
            rows.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],"true_K":r["true_K"],"features":feat,
                         "distinct_true_attacks":physical,"peak_true_coverage":coverage})
        print(json.dumps({"recording":member,"done":len(rows)}),flush=True)
    names=list(rows[0]["features"]);X=np.asarray([[r["features"][n] for n in names] for r in rows],float)
    y=np.asarray([r["group"]=="K3_regressed" for r in rows],int);folds=np.asarray([r["fold"] for r in rows])
    uni=[]
    for j,n in enumerate(names):
        vals=[]
        for f in FOLDS:
            fit=folds!=f;val=folds==f
            orient=1 if np.median(X[fit&(y==1),j])>=np.median(X[fit&(y==0),j]) else -1
            vals.append(float(roc_auc_score(y[val],orient*X[val,j])))
        uni.append({"feature":n,"mean_auc":float(np.mean(vals)),"min_auc":float(np.min(vals)),"per_fold":dict(zip(map(str,FOLDS),vals))})
    uni.sort(key=lambda q:q["mean_auc"],reverse=True)
    probs=np.zeros(len(y));pf=[]
    for f in FOLDS:
        fit=folds!=f;val=folds==f;m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=4000,class_weight="balanced"))
        m.fit(X[fit],y[fit]);probs[val]=m.predict_proba(X[val])[:,1];pf.append({"fold":f,"auc":float(roc_auc_score(y[val],probs[val]))})
    physical={}
    for g in GROUPS:
        rr=[r for r in rows if r["group"]==g];physical[g]={}
        for sep in SEPS_MS:
            vals=[r["distinct_true_attacks"][str(sep)] for r in rr];physical[g][str(sep)]={"median":float(np.median(vals)),"exact_K_distinct":int(sum(v==r["true_K"] for v,r in zip(vals,rr))),
                                                                                       "mean_peak_true_coverage":float(np.mean([r["peak_true_coverage"][str(sep)] for r in rr]))}
    rep={"status":"completed","experiment":"v273_dense_raw_attack_events","outer_fold_3_used":False,"prediction_changes":False,
         "scan_ms":[-5,50],"nms_separations_ms":list(SEPS_MS),"multivariate_oof":{"auc":float(roc_auc_score(y,probs)),"per_fold":pf},
         "univariate_oof":uni,"physical_attack_ceiling":physical}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Dense raw-PCM attack event audit","",f"Multivariate OOF AUC: **{roc_auc_score(y,probs):.3f}**.","",
           "| fold | AUC |","|---:|---:|"]+[f"| {q['fold']} | {q['auc']:.3f} |" for q in pf]+["","## Best fixed raw-event features","",
           "| feature | mean OOF AUC | min fold |","|---|---:|---:|"]+[f"| {q['feature']} | {q['mean_auc']:.3f} | {q['min_auc']:.3f} |" for q in uni[:12]]
    lines+=["","## Physical distinct-attack ceiling",json.dumps(physical,sort_keys=True),"","Raw PCM only for inference features; annotations are diagnostic only."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
