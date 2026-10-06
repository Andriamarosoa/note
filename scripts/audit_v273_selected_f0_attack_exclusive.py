"""Inference-safe selected-F0 attack + exclusivity diagnostic.

For each frozen residual-decomposition triplet F0, compute harmonic attack novelty
and exclusive-vs-shared harmonic attack support from raw PCM. No annotation
frequency enters feature computation. Labels are only K2/K3 evaluation targets.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_attack_novelty import raw_powers,component_metrics
from scripts.audit_v273_exclusive_harmonic_support import attack_spectrum,support_against_set
from scripts.v273_residual_audit import FOLDS,require

GROUPS=("K2_corrected","K3_regressed")
NOV_METRICS=("pre_norm","post1_norm","onset_contrast_raw","onset_contrast_norm","log_ratio_raw","late_retention_raw","positive_retained_fraction")
EX_METRICS=("unique_attack_fraction","shared_attack_fraction","attack_energy_total")
EPS=1e-12

def summarize_components(vals,prefix):
    out={}
    for k in vals[0]:
        x=np.asarray([v[k] for v in vals],float)
        if k=="attack_energy_total":x=np.log1p(np.maximum(x,0))
        out[f"{prefix}_{k}_min"]=float(x.min())
        out[f"{prefix}_{k}_median"]=float(np.median(x))
        out[f"{prefix}_{k}_max"]=float(x.max())
        out[f"{prefix}_{k}_range"]=float(x.max()-x.min())
    return out

def row_features(row,freq,powers):
    f0=np.asarray(row["decomposition"]["triplet_f0"],float);require(len(f0)==3,"triplet drift")
    nov=[];exc=[];attack=attack_spectrum(powers)
    for i,q in enumerate(f0):
        n=component_metrics(freq,powers,q);nov.append({k:float(n[k]) for k in NOV_METRICS})
        peers=[float(f0[j]) for j in range(3) if j!=i]
        e=support_against_set(freq,attack,q,peers);exc.append({k:float(e[k]) for k in EX_METRICS})
    feat={}
    feat.update(summarize_components(nov,"nov"))
    feat.update(summarize_components(exc,"exc"))
    # Joint permutation-invariant evidence: a component should both attack and own some spectrum.
    unique=np.asarray([v["unique_attack_fraction"] for v in exc],float)
    post=np.asarray([v["post1_norm"] for v in nov],float)
    contrast=np.asarray([v["onset_contrast_raw"] for v in nov],float)
    retained=np.asarray([v["positive_retained_fraction"] for v in nov],float)
    for name,x in {
        "joint_unique_x_post":unique*post,
        "joint_unique_x_contrast":unique*np.maximum(contrast,0),
        "joint_unique_x_retained":unique*retained,
    }.items():
        feat[name+"_min"]=float(x.min());feat[name+"_median"]=float(np.median(x));feat[name+"_max"]=float(x.max());feat[name+"_range"]=float(x.max()-x.min())
    # Explicit weakest proposed source descriptors.
    iu=int(np.argmin(unique)); ip=int(np.argmin(post))
    feat["weak_unique_post1_norm"]=float(post[iu]);feat["weak_unique_contrast"]=float(contrast[iu]);feat["weak_unique_fraction"]=float(unique[iu])
    feat["weak_post_unique_fraction"]=float(unique[ip]);feat["weak_post_post1_norm"]=float(post[ip]);feat["weak_post_contrast"]=float(contrast[ip])
    return feat

def eval_variant(rows,names):
    X=np.asarray([[r["features"][n] for n in names] for r in rows],float); y=np.asarray([r["group"]=="K3_regressed" for r in rows],int); f=np.asarray([r["fold"] for r in rows])
    p=np.zeros(len(y));per=[]
    for fold in FOLDS:
        fit=f!=fold;val=f==fold
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=4000,class_weight="balanced"))
        m.fit(X[fit],y[fit]);p[val]=m.predict_proba(X[val])[:,1];per.append({"fold":fold,"auc":float(roc_auc_score(y[val],p[val]))})
    return {"auc":float(roc_auc_score(y,p)),"per_fold":per}

def main():
    ap=argparse.ArgumentParser()
    for n in ("cases","dataset","output"):ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"refusing overwrite")
    cases=[json.loads(x) for x in a.cases.read_text().splitlines()];cases=[r for r in cases if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in cases)==108,"K2 drift");require(sum(r["group"]=="K3_regressed" for r in cases)==125,"K3 drift")
    require(all(r["fold"] in FOLDS and r["fold"]!=3 for r in cases),"fold leak")
    wanted={r["recording_id"] for r in cases};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in cases:by[r["recording_id"]].append(r)
    rows=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);samples=np.asarray(au.samples,float)/32768.
        for r in sorted(by[member],key=lambda z:z["row_id"]):
            freq,powers,_=raw_powers(samples,int(r["start_sample"]))
            rows.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],"features":row_features(r,freq,powers)})
        print(json.dumps({"recording":member,"done":len(rows)}),flush=True)
    all_names=list(rows[0]["features"])
    nov=[n for n in all_names if n.startswith("nov_")]
    exc=[n for n in all_names if n.startswith("exc_")]
    joint=[n for n in all_names if n.startswith("joint_") or n.startswith("weak_")]
    variants={"novelty_only":nov,"exclusive_only":exc,"joint_only":joint,"combined":all_names}
    results={k:eval_variant(rows,v) for k,v in variants.items()}
    # Univariate fold-oriented diagnostic for interpretation.
    X=np.asarray([[r["features"][n] for n in all_names] for r in rows]);y=np.asarray([r["group"]=="K3_regressed" for r in rows],int);fold=np.asarray([r["fold"] for r in rows])
    uni=[]
    for j,n in enumerate(all_names):
        vals=[]
        for f in FOLDS:
            fit=fold!=f;val=fold==f
            orient=1 if np.median(X[fit&(y==1),j])>=np.median(X[fit&(y==0),j]) else -1
            vals.append(float(roc_auc_score(y[val],orient*X[val,j])))
        uni.append({"feature":n,"mean_auc":float(np.mean(vals)),"min_auc":float(np.min(vals)),"per_fold":dict(zip(map(str,FOLDS),vals))})
    uni.sort(key=lambda q:q["mean_auc"],reverse=True)
    rep={"status":"completed","experiment":"v273_selected_f0_attack_exclusive","inference_features_annotation_free":True,"outer_fold_3_used":False,"prediction_changes":False,
         "cases":len(rows),"variants":results,"univariate_oof":uni}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Selected-F0 attack + exclusivity diagnostic","",
           "| variant | OOF AUC | fold0 | fold1 | fold2 | fold4 |","|---|---:|---:|---:|---:|---:|"]
    for n,q in results.items():
        d={x["fold"]:x["auc"] for x in q["per_fold"]};lines.append(f"| {n} | {q['auc']:.3f} | {d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} | {d[4]:.3f} |")
    lines+=["","## Best single features","","| feature | mean OOF AUC | min fold |","|---|---:|---:|"]+[f"| {q['feature']} | {q['mean_auc']:.3f} | {q['min_auc']:.3f} |" for q in uni[:12]]
    lines+=["","All inference features use only selected residual F0s + raw PCM. No annotation F0 is used."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
