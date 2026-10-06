"""Inference-safe candidate-aligned pick-burst diagnostic.

Uses only retained candidate timestamps already present in the frozen V27.3
bundle plus raw audio. Ground-truth labels are used only for evaluation.
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
from scripts.v273_window_experiment import load_bundle
from scripts.audit_v273_annotation_pick_burst import window,note_burst

FOLDS=(0,1,2,4); GROUPS=("K2_corrected","K3_regressed")
CLUSTER_WINDOW_SAMPLES=round(.040*SAMPLE_RATE)
MERGE_SAMPLES=round(.003*SAMPLE_RATE)
EPS=1e-12

def req(c,m):
    if not c: raise RuntimeError(m)

def clusters(samples,bursts):
    order=np.argsort(samples); samples=np.asarray(samples)[order]; bursts=[bursts[i] for i in order]
    groups=[]; cur=[]
    last=None
    for s,b in zip(samples,bursts):
        if last is None or int(s)-int(last)<=MERGE_SAMPLES:
            cur.append((int(s),b))
        else:
            groups.append(cur);cur=[(int(s),b)]
        last=int(s)
    if cur:groups.append(cur)
    out=[]
    for g in groups:
        # strongest local transient represents this temporal proposal group
        best=max(g,key=lambda q:q[1]["max_diff_ratio"])
        out.append({"sample":best[0],**best[1],"members":len(g)})
    return out

def pad_top(v,n=5,default=0.0):
    a=sorted([float(x) for x in v],reverse=True)
    return (a+[default]*n)[:n]

def row_features(groups):
    d=pad_top([g["max_diff_ratio"] for g in groups])
    r=pad_top([g["max_rms_ratio"] for g in groups])
    dr=pad_top([g["diff_rms_ratio"] for g in groups])
    fl=pad_top([g["flatness_delta"] for g in groups])
    hf=pad_top([g["hf_delta"] for g in groups])
    return {
      "pick_diff_top1":d[0],"pick_diff_top2":d[1],"pick_diff_top3":d[2],"pick_diff_top4":d[3],
      "pick_diff_top3_over_top2":float(d[2]/(d[1]+EPS)),
      "pick_diff_top3_over_top1":float(d[2]/(d[0]+EPS)),
      "pick_diff_gap_2_3":float(d[1]-d[2]),
      "pick_rms_top3":r[2],"pick_rms_top3_over_top2":float(r[2]/(r[1]+EPS)),
      "diff_rms_top3":dr[2],"flatness_top3":fl[2],"hf_delta_top3":hf[2],
      "temporal_cluster_count":float(len(groups)),
      "strong_pick_count_gt1":float(sum(g["max_diff_ratio"]>1.0 for g in groups)),
      "strong_pick_count_gt1p25":float(sum(g["max_diff_ratio"]>1.25 for g in groups)),
    }

def stats(v):
    a=np.asarray(v,float);return {"median":float(np.median(a)),"mean":float(a.mean()),"q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}

def main():
    p=argparse.ArgumentParser()
    for n in ("cases","dataset","bundle","config","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();req(not a.output.exists(),"refusing overwrite")
    rows=[json.loads(x) for x in a.cases.read_text().splitlines()]
    rows=[r for r in rows if r["group"] in GROUPS]
    req(sum(r["group"]=="K2_corrected" for r in rows)==108,"K2 drift");req(sum(r["group"]=="K3_regressed" for r in rows)==125,"K3 drift")
    cache,_,_=load_bundle(a.bundle,a.config)
    wanted={r["recording_id"] for r in rows};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};req(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in rows:by[r["recording_id"]].append(r)
    measured=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);x=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            rid=int(r["row_id"]); req(str(np.asarray(cache["members"])[rid])==member,"row/member mismatch")
            start=int(np.asarray(cache["cluster_start_samples"])[rid]); req(abs(start-int(r["start_sample"]))<=1,"start mismatch")
            mask=np.asarray(cache["mask"])[rid]>0
            rel=np.asarray(cache["sequence"])[rid,mask,-2].astype(float)
            samp=start+np.rint(rel*CLUSTER_WINDOW_SAMPLES).astype(np.int64)
            bursts=[note_burst(window(x,int(s))) for s in samp]
            cg=clusters(samp,bursts);feat=row_features(cg)
            measured.append({"row_id":rid,"fold":r["fold"],"group":r["group"],"true_K":r["true_K"],"recording_id":member,
                             "candidate_samples":samp.tolist(),"candidate_clusters":cg,"features":feat})
        print(json.dumps({"recording":member,"done":len(measured)}),flush=True)
    names=list(measured[0]["features"]);X=np.asarray([[r["features"][n] for n in names] for r in measured],float)
    y=np.asarray([r["group"]=="K3_regressed" for r in measured],int);folds=np.asarray([r["fold"] for r in measured])
    summary={}
    for g in GROUPS:
        idx=np.asarray([r["group"]==g for r in measured]); summary[g]={n:stats(X[idx,j]) for j,n in enumerate(names)}
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
        fit=folds!=f;val=folds==f
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))
        m.fit(X[fit],y[fit]);probs[val]=m.predict_proba(X[val])[:,1]
        pf.append({"fold":f,"auc":float(roc_auc_score(y[val],probs[val]))})
    auc=float(roc_auc_score(y,probs))
    rep={"status":"completed","experiment":"v273_candidate_aligned_pick_burst","inference_safe_inputs":True,"annotation_centering":False,
         "outer_fold_3_used":False,"cases":len(measured),"merge_ms":3.0,"multivariate_oof":{"auc":auc,"per_fold":pf},
         "univariate_oof":uni,"summary":summary}
    a.output.mkdir(parents=True);(a.output/"cases.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in measured));(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Candidate-aligned pick-burst diagnostic","",f"Inference-safe candidate/audio OOF AUC: **{auc:.3f}**.","","| fold | AUC |","|---:|---:|"]+[f"| {q['fold']} | {q['auc']:.3f} |" for q in pf]+["","| feature | mean OOF AUC | min fold | K2 median | K3 median |","|---|---:|---:|---:|---:|"]
    for q in uni[:12]:
        n=q["feature"];lines.append(f"| {n} | {q['mean_auc']:.3f} | {q['min_auc']:.3f} | {summary['K2_corrected'][n]['median']:.5g} | {summary['K3_regressed'][n]['median']:.5g} |")
    lines+=["","Only frozen candidate timestamps + raw PCM are used as inputs. Labels are evaluation-only. No Exact-K output changed."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
