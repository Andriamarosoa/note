"""Candidate-level true-attack verifier diagnostic on the frozen residual cohort.

Labels are constructed one-to-one from annotations only for training/evaluation on
internal folds 0/1/2/4. The diagnostic asks whether raw pick-burst evidence adds
candidate identity information beyond the frozen V27.3 candidate representation.

No outer fold 3. No Exact-K prediction changes or promotion.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_note.guitarset import SAMPLE_RATE,index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.v273_window_experiment import load_bundle
from scripts.audit_v273_annotation_pick_burst import window,note_burst

FOLDS=(0,1,2,4); GROUPS=("K2_corrected","K3_regressed")
CLUSTER_WINDOW_SAMPLES=round(.040*SAMPLE_RATE); MATCH_MS=3.0; MATCH=round(MATCH_MS*SAMPLE_RATE/1000)
PICK_NAMES=("rms_ratio","diff_rms_ratio","max_rms_ratio","max_diff_ratio","hf_delta","centroid_delta","flatness_delta","post_crest","post_zcr")

def req(c,m):
    if not c: raise RuntimeError(m)

def labels_for(cand,true):
    c=np.asarray(cand,int);t=np.asarray(true,int); y=np.zeros(len(c),int)
    cost=np.abs(t[:,None]-c[None,:]);ri,ci=linear_sum_assignment(cost)
    matched=[]
    for a,b in zip(ri,ci):
        ok=cost[a,b]<=MATCH
        if ok:y[b]=1
        matched.append({"true_index":int(a),"candidate_index":int(b),"distance_ms":float(cost[a,b]*1000/SAMPLE_RATE),"within_3ms":bool(ok)})
    return y,matched

def third(v):
    a=np.sort(np.asarray(v,float))[::-1]
    return float(a[2]) if len(a)>=3 else 0.0

def fourth(v):
    a=np.sort(np.asarray(v,float))[::-1]
    return float(a[3]) if len(a)>=4 else 0.0

def model():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=4000,class_weight="balanced"))

def main():
    p=argparse.ArgumentParser()
    for n in ("cases","dataset","bundle","config","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();req(not a.output.exists(),"refusing overwrite")
    cases=[json.loads(x) for x in a.cases.read_text().splitlines()];cases=[r for r in cases if r["group"] in GROUPS]
    req(sum(r["group"]=="K2_corrected" for r in cases)==108,"K2 drift");req(sum(r["group"]=="K3_regressed" for r in cases)==125,"K3 drift")
    cache,_,_=load_bundle(a.bundle,a.config)
    wanted={r["recording_id"] for r in cases};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};req(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for r in cases:by[r["recording_id"]].append(r)
    rows=[];cand=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);x=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            rid=int(r["row_id"]);mask=np.asarray(cache["mask"])[rid]>0;seq=np.asarray(cache["sequence"])[rid,mask].astype(float)
            start=int(np.asarray(cache["cluster_start_samples"])[rid]);samples=start+np.rint(seq[:,-2]*CLUSTER_WINDOW_SAMPLES).astype(np.int64)
            true=[int(n["onset_sample"]) for n in r["owned_notes"]]; lab,matched=labels_for(samples,true)
            picks=[note_burst(window(x,int(s))) for s in samples]
            idx=[]
            for j,(s,q,y,pb) in enumerate(zip(samples,seq,lab,picks)):
                rec={"row_index":len(rows),"slot":j,"fold":r["fold"],"group":r["group"],"label":int(y),
                     "sample":int(s),"frozen":q.tolist(),"pick":[float(pb[n]) for n in PICK_NAMES]}
                idx.append(len(cand));cand.append(rec)
            rows.append({"row_id":rid,"fold":r["fold"],"group":r["group"],"true_K":r["true_K"],"candidate_indices":idx,"matching":matched})
        print(json.dumps({"recording":member,"rows":len(rows),"candidates":len(cand)}),flush=True)

    cf=np.asarray([q["fold"] for q in cand]); cy=np.asarray([q["label"] for q in cand])
    frozen=np.asarray([q["frozen"] for q in cand],float);pick=np.asarray([q["pick"] for q in cand],float)
    modes={"frozen":frozen,"pick":pick,"combined":np.concatenate([frozen,pick],axis=1)}
    results={}
    for name,X in modes.items():
        p_oof=np.zeros(len(cand),float);fold_auc=[]
        for f in FOLDS:
            fit=cf!=f;val=cf==f;m=model();m.fit(X[fit],cy[fit]);p_oof[val]=m.predict_proba(X[val])[:,1]
            fold_auc.append({"fold":f,"candidate_auc":float(roc_auc_score(cy[val],p_oof[val])),
                             "positive":int(cy[val].sum()),"candidates":int(val.sum())})
        ry=[];thirds=[];sums=[];margins=[];folds=[]
        for r in rows:
            pr=p_oof[np.asarray(r["candidate_indices"],int)];ry.append(1 if r["group"]=="K3_regressed" else 0)
            thirds.append(third(pr));sums.append(float(pr.sum()));margins.append(third(pr)-fourth(pr));folds.append(r["fold"])
        ry=np.asarray(ry);folds=np.asarray(folds);thirds=np.asarray(thirds);sums=np.asarray(sums);margins=np.asarray(margins)
        group_pf=[]
        for f in FOLDS:
            v=folds==f; group_pf.append({"fold":f,"third_auc":float(roc_auc_score(ry[v],thirds[v])),
                                         "sum_auc":float(roc_auc_score(ry[v],sums[v])),
                                         "third_minus_fourth_auc":float(roc_auc_score(ry[v],margins[v]))})
        results[name]={"candidate_auc_global":float(roc_auc_score(cy,p_oof)),"candidate_per_fold":fold_auc,
                       "group_third_auc_global":float(roc_auc_score(ry,thirds)),"group_sum_auc_global":float(roc_auc_score(ry,sums)),
                       "group_third_minus_fourth_auc_global":float(roc_auc_score(ry,margins)),"group_per_fold":group_pf}
    # Oracle coverage under same one-to-one 3 ms candidate label definition.
    oracle={}
    for g in GROUPS:
        rr=[r for r in rows if r["group"]==g];counts=[]
        for r in rr:
            counts.append(sum(cand[i]["label"] for i in r["candidate_indices"]))
        oracle[g]={"rows":len(rr),"exact_candidate_coverage":int(sum(c==r["true_K"] for c,r in zip(counts,rr))),
                   "mean_matched":float(np.mean(counts)),"median_matched":float(np.median(counts))}
    rep={"status":"completed","experiment":"v273_candidate_true_attack_verifier","match_ms":MATCH_MS,"outer_fold_3_used":False,
         "prediction_changes":False,"candidates":len(cand),"positive_candidates":int(cy.sum()),"oracle_candidate_coverage":oracle,"results":results}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Candidate true-attack verifier diagnostic","",f"Candidate labels: one-to-one annotation match within {MATCH_MS:.1f} ms. Fold 3 excluded.","",
           "| features | candidate AUC | group AUC (3rd p) | group AUC (sum p) | group AUC (p3-p4) |",
           "|---|---:|---:|---:|---:|"]
    for n,q in results.items():lines.append(f"| {n} | {q['candidate_auc_global']:.3f} | {q['group_third_auc_global']:.3f} | {q['group_sum_auc_global']:.3f} | {q['group_third_minus_fourth_auc_global']:.3f} |")
    lines+=["","## Combined per-fold group AUC (3rd candidate probability)","",
            "| fold | AUC |","|---:|---:|"]+[f"| {q['fold']} | {q['third_auc']:.3f} |" for q in results["combined"]["group_per_fold"]]
    lines+=["","## 3 ms candidate coverage ceiling","",json.dumps(oracle,sort_keys=True),
            "","Diagnostic only. No Exact-K output is modified."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
