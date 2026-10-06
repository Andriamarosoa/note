"""LOFO logistic stage-2 veto on survivors of the robust pre+range guard.

Stage 1: fixed pre_high AND amp_range_low, thresholds = medians from fit folds.

Stage 2: balanced logistic regression trained only on stage-1 surviving K2/K3
actions in fit folds. Predeclared feature sets:
  - exc + onset
  - exc + onset + post
No probability-threshold tuning: veto when p(K3 regression) >= 0.5.
No fold 3. Post-hoc internal audit only.
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

SETS={
 "exc_onset":("exc_shared_range","onset_contrast"),
 "exc_onset_post":("exc_shared_range","onset_contrast","post1_norm"),
}

def style(member):
    m=re.match(r"^\d{2}_([A-Za-z]+)\d+-",Path(member).name)
    require(m is not None,f"bad style {member}")
    return m.group(1)

def med(vals):
    require(len(vals)>=20,f"small median {len(vals)}")
    return float(np.median(np.asarray(vals,float)))

def stage1_thresholds(rows):
    return {"pre":med([r["features"]["pre"] for r in rows]),
            "amp":med([r["features"]["amp_range"] for r in rows])}

def stage1_hit(r,t):
    return r["features"]["pre"]>=t["pre"] and r["features"]["amp_range"]<=t["amp"]

def model():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))

def summarize(rows,mask):
    y=np.asarray([r["y"] for r in rows],int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));g=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"applied":int(m.sum()),"corrections":c,"regressions":g,"other_k_actions":o,"net":c-g}

def train_stage2(fit_surv,names):
    X=np.asarray([[r["features"][n] for n in names] for r in fit_surv],float)
    y=np.asarray([r["y"]==3 for r in fit_surv],int)
    require(len(np.unique(y))==2 and len(y)>=30,"bad stage2 train")
    m=model();m.fit(X,y)
    return m

def eval_oof(rows,names):
    s1=np.zeros(len(rows),bool);s2=np.zeros(len(rows),bool)
    diagnostics={}
    for f in FOLDS:
        fit=[r for r in rows if r["fold"]!=f and r["y"] in (2,3)]
        t=stage1_thresholds(fit)
        fit_surv=[r for r in fit if not stage1_hit(r,t)]
        m=train_stage2(fit_surv,names)
        val_idx=[i for i,r in enumerate(rows) if r["fold"]==f]
        probs=[]
        for i in val_idx:
            r=rows[i];s1[i]=stage1_hit(r,t)
            if s1[i]:continue
            p=float(m.predict_proba(np.asarray([[r["features"][n] for n in names]],float))[0,1])
            probs.append(p);s2[i]=p>=.5
        diagnostics[str(f)]={"fit_survivors":len(fit_surv),"val_survivor_probs_n":len(probs),
                             "prob_median":None if not probs else float(np.median(probs))}
    return s1,s2,diagnostics

def style_transfer(rows,names,st):
    held=[r for r in rows if r["style"]==st]
    s1=[];s2=[]
    cache={}
    for r in held:
        f=r["fold"];key=str(f)
        if key not in cache:
            fit=[x for x in rows if x["style"]!=st and x["fold"]!=f and x["y"] in (2,3)]
            t=stage1_thresholds(fit)
            surv=[x for x in fit if not stage1_hit(x,t)]
            cache[key]=(t,train_stage2(surv,names))
        t,m=cache[key]
        h1=stage1_hit(r,t);s1.append(h1)
        if h1:s2.append(False)
        else:
            p=float(m.predict_proba(np.asarray([[r["features"][n] for n in names]],float))[0,1])
            s2.append(p>=.5)
    return summarize(held,s1),summarize(held,s2),summarize(held,~(np.asarray(s1)|np.asarray(s2)))

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

    source=summarize(rows,np.ones(len(rows),bool)); results={}
    for name,names in SETS.items():
        s1,s2,diag=eval_oof(rows,names)
        b1=summarize(rows,s1);b2=summarize(rows,s2);final=summarize(rows,~(s1|s2))
        require((b1["corrections"],b1["regressions"])==(24,48),"stage1 drift")
        per=[]
        for f in FOLDS:
            mask=np.asarray([r["fold"]==f for r in rows],bool)
            q2=summarize(rows,s2&mask);fin=summarize(rows,(~(s1|s2))&mask)
            per.append({"fold":int(f),"stage2_block":q2,"stage2_gain":q2["regressions"]-q2["corrections"],"final":fin})
        sty=[]
        for st in sorted({r["style"] for r in rows}):
            _,q2,fin=style_transfer(rows,names,st)
            sty.append({"style":st,"stage2_block":q2,"stage2_gain":q2["regressions"]-q2["corrections"],"final":fin})
        results[name]={"features":list(names),"stage1_block":b1,"stage2_block":b2,"final":final,
                       "incremental_gain":b2["regressions"]-b2["corrections"],
                       "per_fold":per,"leave_one_style_out":sty,"diagnostics":diag,
                       "all_folds_nonnegative":all(q["stage2_gain"]>=0 for q in per),
                       "all_styles_nonnegative":all(q["stage2_gain"]>=0 for q in sty)}

    rep={"status":"completed","experiment":"v273_stage2_lofo_logistic",
         "source":source,"results":results,"probability_threshold":0.5,
         "outer_fold_3_used":False,"prediction_changes":False,"threshold_search":False,
         "posthoc_internal_reuse":True,"automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Stage-2 LOFO logistic audit","",
           "Balanced logistic, C=0.1, p>=0.5 fixed. Trained only on stage-1 survivors in fit folds.","",
           "| feature set | K3 blocked | K2 blocked | stage2 gain | final net | folds >=0 | styles >=0 |",
           "|---|---:|---:|---:|---:|---|---|"]
    for name,q in results.items():
        b=q["stage2_block"];lines.append(f"| {name} | {b['regressions']} | {b['corrections']} | {q['incremental_gain']:+d} | {q['final']['net']:+d} | {q['all_folds_nonnegative']} | {q['all_styles_nonnegative']} |")
    lines+=["","## Per-fold incremental gains","","| set | f0 | f1 | f2 | f4 |","|---|---:|---:|---:|---:|"]
    for name,q in results.items():
        d={x["fold"]:x["stage2_gain"] for x in q["per_fold"]}
        lines.append(f"| {name} | {d[0]:+d} | {d[1]:+d} | {d[2]:+d} | {d[4]:+d} |")
    lines+=["","Internal/post-hoc only; no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
