"""Targeted second-stage audit after the robust pre+range veto.

Goal: explain why exclusivity-range helps fold 2 but hurts fold 1/BN, and test
only fixed combinations already justified by prior residual-signature audits.

Stage 1 (fixed): pre_high AND amp_range_low.
Stage-2 signals: exclusivity range HIGH, onset contrast LOW, post1 norm LOW.
All thresholds are fit-fold medians. Fold 3 excluded.
"""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_oof_source_onset_contrast_guard import build_oof
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_internal_b_low_harmonic_strata import transition_spectrum,f0_pool,template,fit_best
from scripts.audit_v273_attack_novelty import raw_powers,component_metrics
from scripts.audit_v273_exclusive_harmonic_support import attack_spectrum,support_against_set
from scripts.v273_residual_audit import FOLDS,require

PATTERNS={
 "exc_only":("exc_high",),
 "onset_only":("onset_low",),
 "post_only":("post_low",),
 "exc_onset":("exc_high","onset_low"),
 "exc_post":("exc_high","post_low"),
 "exc_onset_post":("exc_high","onset_low","post_low"),
}

def style(member):
    m=re.match(r"^\d{2}_([A-Za-z]+)\d+-",Path(member).name)
    require(m is not None,f"bad style {member}")
    return m.group(1)

def features(samples,start):
    freq,x=transition_spectrum(samples,start)
    pool=f0_pool(freq,x)
    if len(pool)<3:return None
    D=np.column_stack([template(freq,f) for f in pool])
    trip,_,_,_=fit_best(D,x,3)
    _,combo,amps=trip
    f0=np.asarray([pool[i] for i in combo],float)
    amps=np.asarray(amps,float)
    if len(f0)!=3 or len(amps)!=3:return None
    rf,powers,_=raw_powers(samples,start)
    nov=[component_metrics(rf,powers,float(q)) for q in f0]
    attack=attack_spectrum(powers)
    shared=[]
    for i,q in enumerate(f0):
        peers=[float(f0[j]) for j in range(3) if j!=i]
        shared.append(float(support_against_set(rf,attack,float(q),peers)["shared_attack_fraction"]))
    return {
      "pre":float(np.median([q["pre_norm"] for q in nov])),
      "amp_range":float(np.ptp(amps)),
      "exc_shared_range":float(np.ptp(np.asarray(shared,float))),
      "onset_contrast":float(np.median([q["onset_contrast_norm"] for q in nov])),
      "post1_norm":float(np.median([q["post1_norm"] for q in nov])),
    }

def med(vals):
    require(len(vals)>=20,f"small median fit: {len(vals)}")
    return float(np.median(np.asarray(vals,float)))

def fit_thresholds(rows):
    return {
      "pre_high":med([r["features"]["pre"] for r in rows]),
      "amp_range_low":med([r["features"]["amp_range"] for r in rows]),
      "exc_high":med([r["features"]["exc_shared_range"] for r in rows]),
      "onset_low":med([r["features"]["onset_contrast"] for r in rows]),
      "post_low":med([r["features"]["post1_norm"] for r in rows]),
    }

def stage1_hit(r,t):
    q=r["features"]
    return q["pre"]>=t["pre_high"] and q["amp_range"]<=t["amp_range_low"]

def cond(r,t,name):
    q=r["features"]
    if name=="exc_high":return q["exc_shared_range"]>=t["exc_high"]
    if name=="onset_low":return q["onset_contrast"]<=t["onset_low"]
    if name=="post_low":return q["post1_norm"]<=t["post_low"]
    raise KeyError(name)

def stage2_hit(r,t,pattern):
    return all(cond(r,t,n) for n in pattern)

def summarize(rows,mask):
    y=np.asarray([r["y"] for r in rows],int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));g=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"applied":int(m.sum()),"corrections":c,"regressions":g,"other_k_actions":o,"net":c-g}

def oof_eval(rows,pattern):
    s1=np.zeros(len(rows),bool);s2=np.zeros(len(rows),bool);thr={}
    for f in FOLDS:
        fit=[r for r in rows if r["fold"]!=f and r["y"] in (2,3)]
        t1=fit_thresholds(fit)
        fit_surv=[r for r in fit if not stage1_hit(r,t1)]
        t2=fit_thresholds(fit_surv)
        thr[str(f)]={"stage1":{"pre_high":t1["pre_high"],"amp_range_low":t1["amp_range_low"]},
                     "stage2":{"exc_high":t2["exc_high"],"onset_low":t2["onset_low"],"post_low":t2["post_low"]},
                     "fit_survivors":len(fit_surv)}
        for i,r in enumerate(rows):
            if r["fold"]!=f:continue
            s1[i]=stage1_hit(r,t1)
            if not s1[i]:s2[i]=stage2_hit(r,t2,pattern)
    return s1,s2,thr

def style_transfer(rows,pattern,style_name):
    held=[r for r in rows if r["style"]==style_name]
    v1=[];v2=[]
    for r in held:
        fit=[x for x in rows if x["style"]!=style_name and x["fold"]!=r["fold"] and x["y"] in (2,3)]
        t1=fit_thresholds(fit)
        h1=stage1_hit(r,t1);v1.append(h1)
        if h1:
            v2.append(False);continue
        fit_surv=[x for x in fit if not stage1_hit(x,t1)]
        t2=fit_thresholds(fit_surv)
        v2.append(stage2_hit(r,t2,pattern))
    return summarize(held,v1),summarize(held,v2),summarize(held,np.logical_not(np.logical_or(v1,v2)))

def medians_by_class(rows,fold=None,style_name=None):
    rr=[r for r in rows if r["y"] in (2,3)]
    if fold is not None:rr=[r for r in rr if r["fold"]==fold]
    if style_name is not None:rr=[r for r in rr if r["style"]==style_name]
    out={}
    for key in ("exc_shared_range","onset_contrast","post1_norm"):
        out[key]={}
        for y in (2,3):
            vals=[r["features"][key] for r in rr if r["y"]==y]
            out[key][str(y)]={"n":len(vals),"median":None if not vals else float(np.median(vals))}
    return out

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
    require(len(rows)==427,"action drift")
    require(all(r.get("features") is not None for r in rows),"feature missing")
    require(sum(r["y"]==2 for r in rows)==108 and sum(r["y"]==3 for r in rows)==125,"K2/K3 drift")

    results={}
    for name,pat in PATTERNS.items():
        s1,s2,thr=oof_eval(rows,pat)
        source=summarize(rows,np.ones(len(rows),bool))
        b1=summarize(rows,s1);b2=summarize(rows,s2)
        final=summarize(rows,~(s1|s2))
        per=[]
        for f in FOLDS:
            m=np.asarray([r["fold"]==f for r in rows],bool)
            q2=summarize(rows,s2&m);fin=summarize(rows,(~(s1|s2))&m)
            per.append({"fold":int(f),"stage2_block":q2,"stage2_gain":q2["regressions"]-q2["corrections"],"final":fin})
        sty=[]
        for st in sorted({r["style"] for r in rows}):
            _,q2,fin=style_transfer(rows,pat,st)
            sty.append({"style":st,"stage2_block":q2,"stage2_gain":q2["regressions"]-q2["corrections"],"final":fin})
        results[name]={"pattern":list(pat),"source":source,"stage1_block":b1,"stage2_block":b2,
                       "final":final,"incremental_gain":b2["regressions"]-b2["corrections"],
                       "per_fold":per,"leave_one_style_out":sty,"thresholds":thr,
                       "all_folds_nonnegative":all(q["stage2_gain"]>=0 for q in per),
                       "all_styles_nonnegative":all(q["stage2_gain"]>=0 for q in sty)}

    diag={"fold1":medians_by_class(rows,fold=1),"fold2":medians_by_class(rows,fold=2),"style_BN":medians_by_class(rows,style_name="BN")}
    report={"status":"completed","experiment":"v273_stage2_exclusivity_failure_audit",
            "patterns":results,"diagnostic_medians":diag,"outer_fold_3_used":False,
            "prediction_changes":False,"threshold_search":False,"posthoc_internal_reuse":True,"automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Stage-2 exclusivity failure audit","",
           "All thresholds are fit-fold medians; no numeric threshold search.","",
           "| pattern | stage2 K3 blocked | stage2 K2 blocked | gain | final net | all folds >=0 | all styles >=0 |",
           "|---|---:|---:|---:|---:|---|---|"]
    for name,q in results.items():
        b=q["stage2_block"]
        lines.append(f"| {name} | {b['regressions']} | {b['corrections']} | {q['incremental_gain']:+d} | {q['final']['net']:+d} | {q['all_folds_nonnegative']} | {q['all_styles_nonnegative']} |")
    lines+=["","## Per-fold gains for combined candidates","","| pattern | f0 | f1 | f2 | f4 |","|---|---:|---:|---:|---:|"]
    for name in ("exc_only","exc_onset","exc_post","exc_onset_post"):
        d={q["fold"]:q["stage2_gain"] for q in results[name]["per_fold"]}
        lines.append(f"| {name} | {d[0]:+d} | {d[1]:+d} | {d[2]:+d} | {d[4]:+d} |")
    lines+=["","Diagnostic medians: "+json.dumps(diag,sort_keys=True),"","Diagnostic/post-hoc internal audit only."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
