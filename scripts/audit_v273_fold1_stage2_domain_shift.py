"""Fold-1 stage2 domain-shift audit.

Replays the robust stage1 pre+range veto and the exc+onset logistic stage2 trained
on folds 0/2/4 with fixed p>=0.5. Then diagnoses fold-1 stage2 true vetoes (K3)
vs false vetoes (K2) by player, recording, style and comp/solo.

No prediction changes. Fold 3 excluded.
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

FEATURES=("exc_shared_range","onset_contrast")

def meta(member):
    stem=Path(member).stem
    m=re.match(r"^(\d{2})_([A-Za-z]+)(\d+)-(.+?)_(comp|solo)$",stem)
    require(m is not None,f"bad member {member}")
    player,style,variant,piece,part=m.groups()
    return {"player":player,"style":style,"variant":style+variant,"part":part,"piece":piece}

def med(vals):
    require(len(vals)>=20,"small median")
    return float(np.median(np.asarray(vals,float)))

def s1_thresholds(rows):
    return {"pre":med([r["features"]["pre"] for r in rows]),
            "amp":med([r["features"]["amp_range"] for r in rows])}

def s1_hit(r,t):
    return r["features"]["pre"]>=t["pre"] and r["features"]["amp_range"]<=t["amp"]

def train(rows):
    X=np.asarray([[r["features"][n] for n in FEATURES] for r in rows],float)
    y=np.asarray([r["y"]==3 for r in rows],int)
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))
    m.fit(X,y);return m

def p3(m,r):
    return float(m.predict_proba(np.asarray([[r["features"][n] for n in FEATURES]],float))[0,1])

def stats(vals):
    a=np.asarray(vals,float)
    return {"n":len(a),"mean":None if not len(a) else float(a.mean()),
            "median":None if not len(a) else float(np.median(a)),
            "q25":None if not len(a) else float(np.quantile(a,.25)),
            "q75":None if not len(a) else float(np.quantile(a,.75))}

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
        r.update(meta(r["member"]))
        if r["source_apply"]:by[r["member"]].append((rid,r["start"]))
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rows_map[rid]["features"]=features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rows_map.values())}),flush=True)

    rows=[r for r in rows_map.values() if r["source_apply"]]
    require(len(rows)==427 and all(r.get("features") is not None for r in rows),"row drift")
    fit=[r for r in rows if r["fold"]!=1 and r["y"] in (2,3)]
    t=s1_thresholds(fit)
    fit_surv=[r for r in fit if not s1_hit(r,t)]
    model=train(fit_surv)
    fold1=[r for r in rows if r["fold"]==1]
    for r in fold1:
        r["stage1_veto"]=s1_hit(r,t)
        r["pK3"]=None if r["stage1_veto"] else p3(model,r)
        r["stage2_veto"]=bool(not r["stage1_veto"] and r["pK3"]>=.5)

    blocked=[r for r in fold1 if r["stage2_veto"]]
    require(sum(r["y"]==3 for r in blocked)==4 and sum(r["y"]==2 for r in blocked)==7,
            f"fold1 replay drift K3={sum(r['y']==3 for r in blocked)} K2={sum(r['y']==2 for r in blocked)}")

    breakdown={}
    for key in ("player","style","part","variant","member"):
        vals=[]
        for g in sorted({r[key] for r in blocked}):
            rr=[r for r in blocked if r[key]==g]
            k3=sum(r["y"]==3 for r in rr);k2=sum(r["y"]==2 for r in rr)
            vals.append({key:g,"blocked_K3":k3,"blocked_K2":k2,"gain":k3-k2,"n":len(rr)})
        breakdown[key]=vals

    groups={}
    for label,rr in (
        ("false_veto_K2",[r for r in blocked if r["y"]==2]),
        ("true_veto_K3",[r for r in blocked if r["y"]==3]),
        ("surviving_K2",[r for r in fold1 if not r["stage1_veto"] and not r["stage2_veto"] and r["y"]==2]),
        ("surviving_K3",[r for r in fold1 if not r["stage1_veto"] and not r["stage2_veto"] and r["y"]==3]),
    ):
        groups[label]={}
        for key in ("pre","amp_range","exc_shared_range","onset_contrast","post1_norm"):
            groups[label][key]=stats([r["features"][key] for r in rr])
        groups[label]["pK3"]=stats([r["pK3"] for r in rr if r["pK3"] is not None])

    # Recording-level normalization diagnostic: feature minus median among fold1 stage1 survivors in same recording.
    survivors=[r for r in fold1 if not r["stage1_veto"]]
    rec_medians={}
    for member in {r["member"] for r in survivors}:
        rr=[r for r in survivors if r["member"]==member]
        rec_medians[member]={k:float(np.median([r["features"][k] for r in rr]))
                             for k in ("exc_shared_range","onset_contrast","post1_norm","pre","amp_range")}
    normalized={}
    for label,rr in (("false_veto_K2",[r for r in blocked if r["y"]==2]),
                     ("true_veto_K3",[r for r in blocked if r["y"]==3])):
        normalized[label]={}
        for k in ("exc_shared_range","onset_contrast","post1_norm","pre","amp_range"):
            normalized[label][k]=stats([r["features"][k]-rec_medians[r["member"]][k] for r in rr])

    report={"status":"completed","experiment":"v273_fold1_stage2_domain_shift",
            "fold1_stage2":{"blocked_K3":4,"blocked_K2":7,"gain":-3},
            "breakdown":breakdown,"feature_groups":groups,
            "recording_centered_diagnostic":normalized,
            "outer_fold_3_used":False,"prediction_changes":False,"posthoc_internal_reuse":True}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Fold-1 stage2 domain-shift audit","",
           "Replayed stage2: **4 K3 blocked / 7 K2 blocked = -3**.","",
           "## By player","","| player | K3 | K2 | gain |","|---|---:|---:|---:|"]
    for q in breakdown["player"]:lines.append(f"| {q['player']} | {q['blocked_K3']} | {q['blocked_K2']} | {q['gain']:+d} |")
    lines+=["","## By recording","","| recording | K3 | K2 | gain |","|---|---:|---:|---:|"]
    for q in breakdown["member"]:lines.append(f"| {q['member']} | {q['blocked_K3']} | {q['blocked_K2']} | {q['gain']:+d} |")
    lines+=["","Feature-group summary: "+json.dumps(groups,sort_keys=True),
            "","Recording-centered diagnostic: "+json.dumps(normalized,sort_keys=True),
            "","Diagnostic only; no new guard selected."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
