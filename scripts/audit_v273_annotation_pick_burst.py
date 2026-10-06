"""Annotation-conditioned per-note pick-burst diagnostic for residual K2/K3 cases.

Uses annotations only to center acoustic windows. It is diagnostic evidence, not
an inference feature and does not change Exact-K predictions.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_note.guitarset import SAMPLE_RATE,index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav

FOLDS=(0,1,2,4); GROUPS=("K2_corrected","K3_regressed"); EPS=1e-12
PRE=int(round(.004*SAMPLE_RATE)); POST=int(round(.010*SAMPLE_RATE))

def req(c,m):
    if not c: raise RuntimeError(m)

def window(x,s):
    a=s-PRE;b=s+POST
    l=max(0,-a);r=max(0,b-len(x))
    y=np.pad(x[max(0,a):min(len(x),b)],(l,r))
    return y[:PRE+POST]

def spectral(v):
    w=np.hanning(len(v)); p=np.abs(np.fft.rfft(v*w,n=1024))**2
    f=np.fft.rfftfreq(1024,1/SAMPLE_RATE); t=p.sum()+EPS
    return {"hf":float(p[f>=2000].sum()/t),
            "centroid":float((p*f).sum()/t),
            "flatness":float(np.exp(np.mean(np.log(p+EPS)))/(np.mean(p+EPS)))}

def note_burst(seg):
    pre=seg[:PRE]; post=seg[PRE:]
    dpre=np.diff(pre,prepend=pre[:1]); dpost=np.diff(post,prepend=post[:1])
    rp=np.sqrt(np.mean(pre*pre)+EPS); rq=np.sqrt(np.mean(post*post)+EPS)
    dp=np.sqrt(np.mean(dpre*dpre)+EPS); dq=np.sqrt(np.mean(dpost*dpost)+EPS)
    sp=spectral(pre); sq=spectral(post)
    # max short raw/high-pass energy in first 10 ms after onset
    n=64; hop=16
    fr=np.stack([post[i:i+n] for i in range(0,max(1,len(post)-n+1),hop) if len(post[i:i+n])==n])
    if not len(fr): fr=post[None,:]
    er=np.sqrt(np.mean(fr*fr,axis=1)+EPS)
    df=np.diff(fr,axis=1,prepend=fr[:,:1])
    ed=np.sqrt(np.mean(df*df,axis=1)+EPS)
    return {
      "rms_ratio":float(rq/(rp+EPS)),
      "diff_rms_ratio":float(dq/(dp+EPS)),
      "max_rms_ratio":float(er.max()/(rp+EPS)),
      "max_diff_ratio":float(ed.max()/(dp+EPS)),
      "hf_delta":float(sq["hf"]-sp["hf"]),
      "centroid_delta":float(sq["centroid"]-sp["centroid"]),
      "flatness_delta":float(sq["flatness"]-sp["flatness"]),
      "post_crest":float(np.max(np.abs(post))/(rq+EPS)),
      "post_zcr":float(np.mean(post[1:]*post[:-1]<0)),
    }

def aggregate(items):
    out={}
    for n in items[0]:
        v=np.asarray([q[n] for q in items],np.float64)
        out["weakest_"+n]=float(v.min())
        out["median_"+n]=float(np.median(v))
        out["strongest_"+n]=float(v.max())
    return out

def nums(v):
    a=np.asarray(v,np.float64)
    return {"mean":float(a.mean()),"median":float(np.median(a)),
            "q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cases",type=Path,required=True);p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();req(not a.output.exists(),"refusing overwrite")
    rows=[json.loads(x) for x in a.cases.read_text().splitlines()]
    rows=[r for r in rows if r["group"] in GROUPS]
    req(sum(r["group"]=="K2_corrected" for r in rows)==108,"K2 drift")
    req(sum(r["group"]=="K3_regressed" for r in rows)==125,"K3 drift")
    wanted={r["recording_id"] for r in rows}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};req(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in rows: by[r["recording_id"]].append(r)
    result=[]
    for member in sorted(by):
        audio=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        x=np.asarray(audio.samples,np.float64)/32768.
        for r in by[member]:
            bursts=[note_burst(window(x,int(n["onset_sample"]))) for n in r["owned_notes"]]
            req(len(bursts)==r["true_K"],"owned-note drift")
            result.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],"true_K":r["true_K"],
                           "recording_id":member,"bursts":bursts,"features":aggregate(bursts)})
        print(json.dumps({"recording":member,"done":len(result)}),flush=True)
    names=list(result[0]["features"])
    X=np.asarray([[r["features"][n] for n in names] for r in result]); y=np.asarray([r["group"]=="K3_regressed" for r in result],int); folds=np.asarray([r["fold"] for r in result])
    summary={}
    for g in GROUPS:
        idx=np.asarray([r["group"]==g for r in result])
        summary[g]={n:nums(X[idx,j]) for j,n in enumerate(names)}
    uni=[]
    for j,n in enumerate(names):
        per=[]
        for f in FOLDS:
            fit=folds!=f;val=folds==f
            orient=1 if np.median(X[fit&(y==1),j])>=np.median(X[fit&(y==0),j]) else -1
            per.append(float(roc_auc_score(y[val],orient*X[val,j])))
        uni.append({"feature":n,"mean_auc":float(np.mean(per)),"min_auc":float(np.min(per)),"per_fold":dict(zip(map(str,FOLDS),per))})
    uni.sort(key=lambda q:q["mean_auc"],reverse=True)
    probs=np.zeros(len(y))
    pf=[]
    for f in FOLDS:
        fit=folds!=f;val=folds==f
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))
        m.fit(X[fit],y[fit]);probs[val]=m.predict_proba(X[val])[:,1]
        pf.append({"fold":f,"auc":float(roc_auc_score(y[val],probs[val]))})
    auc=float(roc_auc_score(y,probs))
    rep={"status":"completed","experiment":"v273_annotation_centered_pick_burst","annotation_conditioned":True,"inference_feature":False,
         "outer_fold_3_used":False,"cases":len(result),"k2":108,"k3":125,"window_ms":{"pre":4,"post":10},
         "summary":summary,"univariate_oof":uni,"multivariate_oof":{"auc":auc,"per_fold":pf}}
    a.output.mkdir(parents=True)
    (a.output/"cases.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in result))
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Annotation-centered pick-burst diagnostic","",f"Multivariate OOF AUC: **{auc:.3f}**.","",
           "| fold | AUC |","|---:|---:|"]+[f"| {q['fold']} | {q['auc']:.3f} |" for q in pf]+["","| feature | mean OOF AUC | min fold | K2 median | K3 median |","|---|---:|---:|---:|---:|"]
    for q in uni[:12]:
        n=q["feature"];lines.append(f"| {n} | {q['mean_auc']:.3f} | {q['min_auc']:.3f} | {summary['K2_corrected'][n]['median']:.5g} | {summary['K3_regressed'][n]['median']:.5g} |")
    lines+=["","Annotations only center the acoustic windows; no Exact-K decision is changed."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
