"""Nested fit-only calibration for the stage-2 exc+onset logistic veto.

Outer fold:
  - Stage1 thresholds fitted on the other 3 internal folds.
  - Final stage2 logistic fitted on stage1 survivors in those 3 folds.

Stage2 decision threshold:
  - Selected only from INNER out-of-fold predictions inside the outer-fit folds.
  - Candidate thresholds are fixed before evaluation: 0.50..0.90 by 0.05.
  - Feasible threshold must have non-negative veto gain on every inner validation fold.
  - Among feasible thresholds choose max total inner gain, then higher precision,
    then the higher threshold (more conservative).
  - If none is feasible, disable stage2 for that outer fold (threshold > 1).

Outer held fold is never used to choose threshold. Fold 3 excluded.
"""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_oof_source_onset_contrast_guard import build_oof
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_stage2_exclusivity_failure import features
from scripts.v273_residual_audit import FOLDS,require

NAMES=("exc_shared_range","onset_contrast")
THRESHOLDS=tuple(round(x,2) for x in np.arange(.50,.901,.05))

def style(member):
    m=re.match(r"^\d{2}_([A-Za-z]+)\d+-",Path(member).name)
    require(m is not None,"bad style")
    return m.group(1)

def med(vals):
    require(len(vals)>=10,f"small median {len(vals)}")
    return float(np.median(np.asarray(vals,float)))

def s1_thresholds(rows):
    return {"pre":med([r["features"]["pre"] for r in rows]),
            "amp":med([r["features"]["amp_range"] for r in rows])}

def s1_hit(r,t):
    return r["features"]["pre"]>=t["pre"] and r["features"]["amp_range"]<=t["amp"]

def mdl():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))

def train(rows):
    X=np.asarray([[r["features"][n] for n in NAMES] for r in rows],float)
    y=np.asarray([r["y"]==3 for r in rows],int)
    require(len(rows)>=20 and len(np.unique(y))==2,"bad train")
    m=mdl();m.fit(X,y);return m

def prob(m,r):
    return float(m.predict_proba(np.asarray([[r["features"][n] for n in NAMES]],float))[0,1])

def counts(rows,mask):
    y=np.asarray([r["y"] for r in rows],int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));g=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"applied":int(m.sum()),"corrections":c,"regressions":g,"other_k_actions":o,"net":c-g}

def inner_oof(outer_fit):
    folds=sorted({r["fold"] for r in outer_fit})
    preds=[]
    for inner_val in folds:
        tr=[r for r in outer_fit if r["fold"]!=inner_val and r["y"] in (2,3)]
        va=[r for r in outer_fit if r["fold"]==inner_val and r["y"] in (2,3)]
        t=s1_thresholds(tr)
        tr_surv=[r for r in tr if not s1_hit(r,t)]
        va_surv=[r for r in va if not s1_hit(r,t)]
        m=train(tr_surv)
        for r in va_surv:preds.append({"fold":inner_val,"y":r["y"],"p":prob(m,r)})
    return preds

def choose_threshold(preds):
    fold_ids=sorted({q["fold"] for q in preds})
    candidates=[]
    for th in THRESHOLDS:
        per=[];reg=corr=0
        for f in fold_ids:
            qq=[q for q in preds if q["fold"]==f and q["p"]>=th]
            r=sum(q["y"]==3 for q in qq);c=sum(q["y"]==2 for q in qq)
            per.append({"fold":int(f),"reg":r,"corr":c,"gain":r-c})
            reg+=r;corr+=c
        feasible=all(q["gain"]>=0 for q in per)
        precision=None if reg+c==0 else reg/(reg+c)
        candidates.append({"threshold":th,"reg":reg,"corr":corr,"gain":reg-c,
                           "precision":precision,"per_fold":per,"feasible":feasible})
    feasible=[q for q in candidates if q["feasible"] and q["reg"]+q["corr"]>0]
    if not feasible:return 1.01,candidates,None
    best=max(feasible,key=lambda q:(q["gain"],-1 if q["precision"] is None else q["precision"],q["threshold"]))
    return float(best["threshold"]),candidates,best

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"overwrite")
    action_inputs(a.exports);rows_map=build_oof(a.exports)
    wanted={r["member"] for r in rows_map.values()}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for rid,r in rows_map.items():
        r["style"]=style(r["member"])
        if r["source_apply"]:by[r["member"]].append((rid,r["start"]))
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rows_map[rid]["features"]=features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rows_map.values())}),flush=True)
    rows=[r for r in rows_map.values() if r["source_apply"]]
    require(len(rows)==427 and all(r.get("features") is not None for r in rows),"row drift")
    require(sum(r["y"]==2 for r in rows)==108 and sum(r["y"]==3 for r in rows)==125,"K drift")

    s1=np.zeros(len(rows),bool);s2=np.zeros(len(rows),bool);outer=[]
    for f in FOLDS:
        outer_fit=[r for r in rows if r["fold"]!=f]
        fit_k23=[r for r in outer_fit if r["y"] in (2,3)]
        t=s1_thresholds(fit_k23)
        final_train=[r for r in fit_k23 if not s1_hit(r,t)]
        final_model=train(final_train)
        preds=inner_oof(outer_fit)
        th,cands,best=choose_threshold(preds)
        val_idx=[i for i,r in enumerate(rows) if r["fold"]==f]
        probs=[]
        for i in val_idx:
            r=rows[i];s1[i]=s1_hit(r,t)
            if s1[i]:continue
            pval=prob(final_model,r);probs.append(pval);s2[i]=pval>=th
        outer.append({"fold":int(f),"selected_threshold":th,"inner_best":best,
                      "inner_candidates":cands,"final_train_survivors":len(final_train),
                      "outer_survivor_prob_median":None if not probs else float(np.median(probs))})

    b1=counts(rows,s1);b2=counts(rows,s2);final=counts(rows,~(s1|s2));source=counts(rows,np.ones(len(rows),bool)))
    require((b1["corrections"],b1["regressions"])==(24,48),"stage1 drift")
    per=[]
    for f in FOLDS:
        m=np.asarray([r["fold"]==f for r in rows],bool)
        q2=counts(rows,s2&m);fin=counts(rows,(~(s1|s2))&m)
        per.append({"fold":int(f),"stage2_block":q2,"stage2_gain":q2["regressions"]-q2["corrections"],"final":fin})

    rep={"status":"completed","experiment":"v273_stage2_nested_calibration",
         "features":list(NAMES),"threshold_grid":list(THRESHOLDS),"source":source,
         "stage1_block":b1,"stage2_block":b2,"final":final,
         "incremental_gain":b2["regressions"]-b2["corrections"],
         "per_fold":per,"outer_calibration":outer,
         "strict_pass":{"all_folds_stage2_nonnegative":all(q["stage2_gain"]>=0 for q in per),
                        "incremental_gain_positive":b2["regressions"]>b2["corrections"]},
         "outer_fold_3_used":False,"prediction_changes":False,
         "held_fold_used_for_threshold":False,"posthoc_internal_reuse":True,"automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Nested-calibrated stage-2 veto","",
           "Threshold chosen from inner-OOF fit folds only; held outer fold never used.","",
           f"Stage1 leaves net **{source['net'] + (b1['regressions']-b1['corrections']):+d}**.",
           f"Stage2 blocks **{b2['regressions']} K3 / {b2['corrections']} K2**, gain **{rep['incremental_gain']:+d}**.",
           f"Final net: **{final['net']:+d}**.","",
           "| outer fold | selected p threshold | stage2 K3 | stage2 K2 | gain | final net |",
           "|---:|---:|---:|---:|---:|---:|"]
    om={q["fold"]:q for q in outer}
    for q in per:
        b=q["stage2_block"];lines.append(f"| {q['fold']} | {om[q['fold']]['selected_threshold']:.2f} | {b['regressions']} | {b['corrections']} | {q['stage2_gain']:+d} | {q['final']['net']:+d} |")
    lines+=["",f"Strict pass: **{rep['strict_pass']}**.","","Internal/post-hoc only; no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
