"""FIT-only single-feature guard for residual 3->2 actions.

Feature: median selected-F0 normalized onset contrast, chosen before this run from
the preceding annotation-free selected-F0 diagnostic. Threshold and orientation
are selected on FIT folds only; held-out fold is untouched. K2_corrected means
apply is correct; K3_regressed means apply is a regression.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_attack_novelty import raw_powers,component_metrics
from scripts.v273_residual_audit import FOLDS,require

FEATURE="nov_onset_contrast_norm_median"
GROUPS=("K2_corrected","K3_regressed")

def feat(row,freq,powers):
    f0=np.asarray(row["decomposition"]["triplet_f0"],float);require(len(f0)==3,"triplet drift")
    x=[component_metrics(freq,powers,q)["onset_contrast_norm"] for q in f0]
    return float(np.median(x))

def candidates(x):
    u=np.unique(np.asarray(x,float))
    if len(u)==1:return [float(u[0])]
    mids=(u[:-1]+u[1:])/2
    return [float(u[0]-1e-12),*map(float,mids),float(u[-1]+1e-12)]

def score(groups,values,orientation,thr):
    # apply correction when oriented feature <= threshold (K2-like)
    apply=orientation*np.asarray(values)<=thr
    g=np.asarray(groups)
    corr=int(np.sum(apply&(g=="K2_corrected")));reg=int(np.sum(apply&(g=="K3_regressed")))
    return {"actions":int(apply.sum()),"corrections":corr,"regressions":reg,"net":corr-reg}

def select(groups,values):
    v=np.asarray(values,float);g=np.asarray(groups)
    best=None
    for orient in (1,-1):
        z=orient*v
        for t in candidates(z):
            q=score(g,v,orient,t)
            key=(q["net"],-q["regressions"],q["corrections"],-q["actions"])
            if best is None or key>best[0]:best=(key,orient,float(t),q)
    if best[3]["net"]<=0:return None
    return {"orientation":best[1],"threshold":best[2],"fit":best[3]}

def main():
    p=argparse.ArgumentParser()
    for n in ("cases","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"refusing overwrite")
    all_rows=[json.loads(x) for x in a.cases.read_text().splitlines()]
    rows=[r for r in all_rows if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in rows)==108,"K2 drift");require(sum(r["group"]=="K3_regressed" for r in rows)==125,"K3 drift")
    wanted={r["recording_id"] for r in rows};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};require(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for r in rows:by[r["recording_id"]].append(r)
    measured=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);s=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            freq,powers,_=raw_powers(s,int(r["start_sample"]))
            measured.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],"value":feat(r,freq,powers)})
        print(json.dumps({"recording":member,"done":len(measured)}),flush=True)
    fold=np.asarray([r["fold"] for r in measured]);groups=np.asarray([r["group"] for r in measured]);values=np.asarray([r["value"] for r in measured])
    reports=[]
    for f in FOLDS:
        fit=fold!=f;val=fold==f;sel=select(groups[fit],values[fit])
        base=score(groups[val],values[val],1,float("inf"))
        guarded={"actions":0,"corrections":0,"regressions":0,"net":0} if sel is None else score(groups[val],values[val],sel["orientation"],sel["threshold"])
        reports.append({"fold":f,"selected":sel,"val_base_all_actions":base,"val_guarded":guarded,
                        "val_rows":int(val.sum()),"val_k2":int(np.sum(groups[val]=="K2_corrected")),"val_k3":int(np.sum(groups[val]=="K3_regressed"))})
    total={k:sum(r["val_guarded"][k] for r in reports) for k in ("actions","corrections","regressions","net")}
    base={k:sum(r["val_base_all_actions"][k] for r in reports) for k in ("actions","corrections","regressions","net")}
    rep={"status":"completed","experiment":"v273_selected_f0_onset_contrast_exact_guard","feature":FEATURE,"outer_fold_3_used":False,
         "threshold_selection":"FIT-only exhaustive midpoint; maximize net, then fewer regressions, more corrections, fewer actions; abstain if FIT net <=0",
         "folds":reports,"total_guarded":total,"total_base_action_cohort":base,"prediction_changes":False,"automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Selected-F0 onset-contrast Exact-K guard","",f"Base action cohort: **{base['corrections']}/{base['regressions']} net {base['net']:+d}**.",f"Guarded OOF: **{total['corrections']}/{total['regressions']} net {total['net']:+d}**.","",
           "| fold | apply | corrections | regressions | net |","|---:|---:|---:|---:|---:|"]
    for r in reports:
        q=r["val_guarded"];lines.append(f"| {r['fold']} | {q['actions']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["","Fold 3 excluded. This evaluates only the K2-corrected/K3-regressed residual action cohort; no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
