"""Post-hoc confidence sweep for stage2 exc+onset logistic.

Stage1 fixed: fit-fold medians for pre_high AND amp_range_low.
Stage2 model fixed: balanced logistic C=.1 on exc_shared_range + onset_contrast.
Only decision confidence is swept: p >= .55, .60, .65.

Reports full OOF, per-fold, per-player, per-style gains. This sweep is explicitly
post-hoc after fold1 probability diagnostics; no automatic promotion.
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

PTH=(.55,.60,.65);NAMES=("exc_shared_range","onset_contrast")

def meta(member):
    stem=Path(member).stem
    m=re.match(r"^(\d{2})_([A-Za-z]+)(\d+)-(.+?)_(comp|solo)$",stem);require(m is not None,"bad member")
    p,s,v,piece,part=m.groups();return {"player":p,"style":s,"part":part}

def med(v):
    require(len(v)>=20,"small med");return float(np.median(np.asarray(v,float)))

def s1thr(rows):return {"pre":med([r["features"]["pre"] for r in rows]),"amp":med([r["features"]["amp_range"] for r in rows])}
def s1(r,t):return r["features"]["pre"]>=t["pre"] and r["features"]["amp_range"]<=t["amp"]

def train(rows):
    X=np.asarray([[r["features"][n] for n in NAMES] for r in rows],float);y=np.asarray([r["y"]==3 for r in rows],int)
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"));m.fit(X,y);return m
def prob(m,r):return float(m.predict_proba(np.asarray([[r["features"][n] for n in NAMES]],float))[0,1])

def stat(rows,mask):
    y=np.asarray([r["y"] for r in rows],int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));g=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"applied":int(m.sum()),"corr":c,"reg":g,"other":o,"gain":g-c}

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"overwrite")
    action_inputs(a.exports);rm=build_oof(a.exports)
    wanted={r["member"] for r in rm.values()};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for rid,r in rm.items():
        r.update(meta(r["member"]))
        if r["source_apply"]:by[r["member"]].append((rid,r["start"]))
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);samples=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rm[rid]["features"]=features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rm.values())}),flush=True)
    rows=[r for r in rm.values() if r["source_apply"]];require(len(rows)==427 and all(r.get("features") is not None for r in rows),"drift")

    base_s1=np.zeros(len(rows),bool);pk=np.full(len(rows),np.nan)
    for f in FOLDS:
        fit=[r for r in rows if r["fold"]!=f and r["y"] in (2,3)]
        t=s1thr(fit);surv=[r for r in fit if not s1(r,t)];m=train(surv)
        for i,r in enumerate(rows):
            if r["fold"]!=f:continue
            base_s1[i]=s1(r,t)
            if not base_s1[i]:pk[i]=prob(m,r)
    b1=stat(rows,base_s1);require((b1["corr"],b1["reg"])==(24,48),"stage1 drift")

    results={}
    for th in PTH:
        s2=(~base_s1)&(pk>=th)
        final=~(base_s1|s2);b2=stat(rows,s2)
        per_fold=[];per_player=[];per_style=[]
        for f in FOLDS:
            m=np.asarray([r["fold"]==f for r in rows],bool);q=stat(rows,s2&m);per_fold.append({"fold":int(f),**q})
        for pl in sorted({r["player"] for r in rows}):
            m=np.asarray([r["player"]==pl for r in rows],bool);q=stat(rows,s2&m);per_player.append({"player":pl,**q})
        for st in sorted({r["style"] for r in rows}):
            m=np.asarray([r["style"]==st for r in rows],bool);q=stat(rows,s2&m);per_style.append({"style":st,**q})
        fin=stat(rows,final)
        results[str(th)]={"stage2":b2,"final":fin,"incremental_gain":b2["gain"],
                          "per_fold":per_fold,"per_player":per_player,"per_style":per_style,
                          "all_folds_nonnegative":all(q["gain"]>=0 for q in per_fold),
                          "all_players_nonnegative":all(q["gain"]>=0 for q in per_player),
                          "all_styles_nonnegative":all(q["gain"]>=0 for q in per_style)}

    rep={"status":"completed","experiment":"v273_stage2_confidence_sweep","thresholds":list(PTH),
         "stage1_block":b1,"results":results,"posthoc_after_fold1_diagnostic":True,
         "outer_fold_3_used":False,"prediction_changes":False,"automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Stage2 high-confidence sweep","",
           "Post-hoc sweep after fold1 diagnosis; no automatic promotion.","",
           "| p threshold | K3 blocked | K2 blocked | gain | final net | folds >=0 | players >=0 | styles >=0 |",
           "|---:|---:|---:|---:|---:|---|---|---|"]
    for th in PTH:
        q=results[str(th)];b=q["stage2"];lines.append(f"| {th:.2f} | {b['reg']} | {b['corr']} | {q['incremental_gain']:+d} | {q['final']['gain']:+d} | {q['all_folds_nonnegative']} | {q['all_players_nonnegative']} | {q['all_styles_nonnegative']} |")
    lines+=["","## Per-fold gains","","| p | f0 | f1 | f2 | f4 |","|---:|---:|---:|---:|---:|"]
    for th in PTH:
        d={q["fold"]:q["gain"] for q in results[str(th)]["per_fold"]};lines.append(f"| {th:.2f} | {d[0]:+d} | {d[1]:+d} | {d[2]:+d} | {d[4]:+d} |")
    lines+=["","Post-hoc internal audit only."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
