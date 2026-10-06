"""Audit candidate-to-true-attack alignment and inference-safe fused-score pick routing.

Diagnostic annotations quantify where candidate timing/ranking fails. Inference-safe
variants use only frozen V27.3 candidate timestamps, fused_birth score, and raw PCM.
No Exact-K prediction is changed.
"""
from __future__ import annotations
import argparse,json,math
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
CLUSTER_WINDOW_SAMPLES=round(.040*SAMPLE_RATE); MERGE=round(.003*SAMPLE_RATE); EPS=1e-12

def req(c,m):
    if not c: raise RuntimeError(m)

def auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def group_candidates(samples,scores,bursts):
    order=np.argsort(samples); s=np.asarray(samples)[order]; q=np.asarray(scores)[order]; b=[bursts[i] for i in order]
    groups=[]; cur=[]; last=None
    for si,qi,bi in zip(s,q,b):
        if last is None or int(si)-int(last)<=MERGE: cur.append((int(si),float(qi),bi))
        else: groups.append(cur);cur=[(int(si),float(qi),bi)]
        last=int(si)
    if cur:groups.append(cur)
    out=[]
    for g in groups:
        # represent temporal group by candidate with strongest frozen fused score
        rep=max(g,key=lambda z:z[1])
        out.append({"sample":rep[0],"fused":rep[1],"burst":rep[2],"members":len(g)})
    return out

def selection_features(groups, mode):
    if mode=="fused":
        sel=sorted(groups,key=lambda g:g["fused"],reverse=True)
    elif mode=="pick":
        sel=sorted(groups,key=lambda g:g["burst"]["max_diff_ratio"],reverse=True)
    elif mode=="fused_pick":
        fs=np.asarray([g["fused"] for g in groups],float); ps=np.asarray([g["burst"]["max_diff_ratio"] for g in groups],float)
        fz=(fs-np.median(fs))/(np.std(fs)+EPS); pz=(ps-np.median(ps))/(np.std(ps)+EPS)
        sel=[groups[i] for i in np.argsort(-(fz+pz))]
    else: raise ValueError(mode)
    def val(i,k,default=0.):
        return float(sel[i]["burst"][k]) if i<len(sel) else default
    def fused(i):
        return float(sel[i]["fused"]) if i<len(sel) else 0.
    d=[val(i,"max_diff_ratio") for i in range(4)]
    r=[val(i,"max_rms_ratio") for i in range(4)]
    return {
      "sel1_diff":d[0],"sel2_diff":d[1],"sel3_diff":d[2],"sel4_diff":d[3],
      "sel3_over_sel2":float(d[2]/(d[1]+EPS)),"sel2_minus_sel3":float(d[1]-d[2]),
      "sel1_rms":r[0],"sel2_rms":r[1],"sel3_rms":r[2],
      "sel1_fused":fused(0),"sel2_fused":fused(1),"sel3_fused":fused(2),
      "group_count":float(len(groups)),
    }

def nearest_diag(cand_samples,cand_scores,true_onsets,true_strengths):
    cs=np.asarray(cand_samples,int); cq=np.asarray(cand_scores,float); ts=np.asarray(true_onsets,int)
    cost=np.abs(ts[:,None]-cs[None,:])
    ri,ci=linear_sum_assignment(cost)
    dt=np.full(len(ts),np.inf); sc=np.zeros(len(ts)); rank=np.full(len(ts),999,int)
    order=np.argsort(-cq); ranks=np.empty(len(cq),int); ranks[order]=np.arange(1,len(cq)+1)
    for a,b in zip(ri,ci):
        dt[a]=cost[a,b]*1000/SAMPLE_RATE; sc[a]=cq[b]; rank[a]=ranks[b]
    nearest=np.min(cost,axis=1)*1000/SAMPLE_RATE
    weak=int(np.argmin(np.asarray(true_strengths,float)))
    return {
      "nearest_ms":nearest.tolist(),"assigned_ms":dt.tolist(),"assigned_fused":sc.tolist(),"assigned_rank":rank.tolist(),
      "max_nearest_ms":float(np.max(nearest)),"median_nearest_ms":float(np.median(nearest)),
      "coverage_1ms":float(np.mean(nearest<=1)),"coverage_3ms":float(np.mean(nearest<=3)),"coverage_5ms":float(np.mean(nearest<=5)),
      "weakest_nearest_ms":float(nearest[weak]),"weakest_assigned_ms":float(dt[weak]),
      "weakest_fused":float(sc[weak]),"weakest_rank":int(rank[weak]),
    }

def summarize(vals):
    a=np.asarray(vals,float);return {"n":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),"q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}

def eval_mode(rows,mode):
    names=list(rows[0]["features"][mode]); X=np.asarray([[r["features"][mode][n] for n in names] for r in rows],float)
    y=np.asarray([r["group"]=="K3_regressed" for r in rows],int); f=np.asarray([r["fold"] for r in rows])
    probs=np.zeros(len(y)); per=[]
    for fold in FOLDS:
        fit=f!=fold;val=f==fold
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))
        m.fit(X[fit],y[fit]);probs[val]=m.predict_proba(X[val])[:,1]
        per.append({"fold":fold,"auc":float(roc_auc_score(y[val],probs[val]))})
    return {"mode":mode,"global_auc":float(roc_auc_score(y,probs)),"per_fold":per}

def main():
    p=argparse.ArgumentParser()
    for n in ("cases","dataset","bundle","config","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args(); req(not a.output.exists(),"refusing overwrite")
    cases=[json.loads(x) for x in a.cases.read_text().splitlines()]; cases=[r for r in cases if r["group"] in GROUPS]
    req(sum(r["group"]=="K2_corrected" for r in cases)==108,"K2 drift"); req(sum(r["group"]=="K3_regressed" for r in cases)==125,"K3 drift")
    cache,_,_=load_bundle(a.bundle,a.config)
    wanted={r["recording_id"] for r in cases}; tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}; req(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in cases:by[r["recording_id"]].append(r)
    out=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member); x=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            rid=int(r["row_id"]); req(str(np.asarray(cache["members"])[rid])==member,"member mismatch")
            mask=np.asarray(cache["mask"])[rid]>0; seq=np.asarray(cache["sequence"])[rid,mask]
            start=int(np.asarray(cache["cluster_start_samples"])[rid]); req(abs(start-int(r["start_sample"]))<=1,"start mismatch")
            samples=start+np.rint(seq[:,-2].astype(float)*CLUSTER_WINDOW_SAMPLES).astype(np.int64)
            fused=seq[:,-3].astype(float)
            bursts=[note_burst(window(x,int(s))) for s in samples]
            groups=group_candidates(samples,fused,bursts)
            true_onsets=[int(n["onset_sample"]) for n in r["owned_notes"]]
            true_b=[note_burst(window(x,s)) for s in true_onsets]
            diag=nearest_diag(samples,fused,true_onsets,[q["max_diff_ratio"] for q in true_b])
            out.append({"row_id":rid,"fold":r["fold"],"group":r["group"],"true_K":r["true_K"],"recording_id":member,
                        "alignment":diag,
                        "features":{m:selection_features(groups,m) for m in ("fused","pick","fused_pick")}})
        print(json.dumps({"recording":member,"done":len(out)}),flush=True)

    alignment={}
    for g in GROUPS:
        rr=[r for r in out if r["group"]==g]; alignment[g]={}
        for k in ("max_nearest_ms","median_nearest_ms","coverage_1ms","coverage_3ms","coverage_5ms","weakest_nearest_ms","weakest_assigned_ms","weakest_fused","weakest_rank"):
            alignment[g][k]=summarize([r["alignment"][k] for r in rr])
    # Explicit K3 weakest-note coverage counts
    k3=[r for r in out if r["group"]=="K3_regressed"]
    weak_cov={str(t):int(sum(r["alignment"]["weakest_nearest_ms"]<=t for r in k3)) for t in (1,3,5,10,20)}
    modes=[eval_mode(out,m) for m in ("fused","pick","fused_pick")]
    rep={"status":"completed","experiment":"v273_candidate_attack_alignment","outer_fold_3_used":False,"prediction_changes":False,
         "cases":len(out),"alignment":alignment,"k3_weakest_note_coverage_counts":weak_cov,"inference_safe_variants":modes}
    a.output.mkdir(parents=True);(a.output/"cases.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in out));(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Candidate ↔ true attack alignment audit","",
           "## Annotation-conditioned alignment diagnostic","",
           "| group | median max-nearest ms | median weakest-nearest ms | median weakest fused | median weakest rank |",
           "|---|---:|---:|---:|---:|"]
    for g in GROUPS:
        z=alignment[g]; lines.append(f"| {g} | {z['max_nearest_ms']['median']:.3f} | {z['weakest_nearest_ms']['median']:.3f} | {z['weakest_fused']['median']:.4f} | {z['weakest_rank']['median']:.1f} |")
    lines += ["",f"K3 weakest-note candidate coverage: {weak_cov}","",
              "## Inference-safe candidate/audio variants","",
              "| selection | OOF AUC | fold0 | fold1 | fold2 | fold4 |","|---|---:|---:|---:|---:|---:|"]
    for q in modes:
        d={x["fold"]:x["auc"] for x in q["per_fold"]};lines.append(f"| {q['mode']} | {q['global_auc']:.3f} | {d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} | {d[4]:.3f} |")
    lines+=["","Diagnostic only. Annotation is used only to explain alignment; inference-safe variants use frozen candidates + fused score + PCM."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))
if __name__=="__main__":main()
