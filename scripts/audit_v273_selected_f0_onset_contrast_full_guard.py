"""Full-population Exact-K audit of the selected-F0 onset-contrast guard.

Uses frozen residual-failure exports from source run 37356100423. The feature is
computed without annotation F0 from the residual triplet hypotheses (min/median/max
F0) and raw PCM. Threshold/orientation are chosen on FIT K2/K3 action rows only,
then applied to ALL held-out valid B_low + base-K3 action rows, including other K.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_internal_b_low_harmonic_strata import transition_spectrum,extract_one
from scripts.audit_v273_attack_novelty import raw_powers,component_metrics
from scripts.v273_residual_audit import FOLDS,require,write_json

FEATURE="nov_onset_contrast_norm_median"
EPS=1e-12

def compute_feature(samples,start):
    freq_t,x=transition_spectrum(samples,start)
    r=extract_one(freq_t,x)
    if r is None:return None
    f0=[float(r["min_triplet_f0"]),float(r["median_triplet_f0"]),float(r["max_triplet_f0"])]
    freq,powers,_=raw_powers(samples,start)
    v=[float(component_metrics(freq,powers,q)["onset_contrast_norm"]) for q in f0]
    return float(np.median(v))

def thresholds(z):
    u=np.unique(np.asarray(z,float))
    if len(u)==1:return [float(u[0])]
    mids=(u[:-1]+u[1:])/2
    return [float(u[0]-1e-12),*map(float,mids),float(u[-1]+1e-12)]

def account(y,apply):
    y=np.asarray(y,int);apply=np.asarray(apply,bool)
    corr=int(np.sum(apply&(y==2)));reg=int(np.sum(apply&(y==3)));other=int(np.sum(apply&~np.isin(y,(2,3))))
    return {"rows":int(len(y)),"applied":int(apply.sum()),"corrections":corr,"regressions":reg,
            "other_k_actions":other,"global_net":corr-reg}

def select(y,v):
    y=np.asarray(y,int);v=np.asarray(v,float);m=np.isin(y,(2,3));yy=y[m];vv=v[m]
    require(np.any(yy==2) and np.any(yy==3),"fit binary collapse")
    best=None
    for orient in (1,-1):
        z=orient*vv
        for t in thresholds(z):
            ap=z<=t;q=account(yy,ap)
            key=(q["global_net"],-q["regressions"],q["corrections"],-q["applied"])
            if best is None or key>best[0]:best=(key,orient,float(t),q)
    if best[3]["global_net"]<=0:return None
    return {"orientation":best[1],"threshold":best[2],"fit_k23":best[3]}

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"refusing overwrite")
    inputs,records=action_inputs(a.exports)
    wanted={m for m,_ in records.values()};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};require(set(tracks)==wanted,"missing audio")
    by={}
    for row,(member,start) in records.items():by.setdefault(member,[]).append((int(row),int(start)))
    computed={};started=time.monotonic()
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);s=np.asarray(au.samples,float)/32768.
        for row,start in by[member]:computed[row]=compute_feature(s,start)
        print(json.dumps({"recording":member,"rows":len(computed),"seconds":time.monotonic()-started}),flush=True)
    require(len(computed)==len(records),"feature coverage drift")

    reports=[]
    for f in FOLDS:
        arr=inputs[f];data={}
        for split in ("fit","val"):
            action=arr[split+"_b_low"]&(arr[split+"_base_k"]==3)
            valid=arr[split+"_valid"]
            ids=arr[split+"_ids"][action][valid]
            y=arr[split+"_true_k"][action][valid].astype(int)
            vals=np.asarray([np.nan if computed[int(i)] is None else computed[int(i)] for i in ids],float)
            good=np.isfinite(vals)
            data[split]={"ids":ids[good],"y":y[good],"v":vals[good],"excluded":int((~good).sum())}
        fit,val=data["fit"],data["val"]
        sel=select(fit["y"],fit["v"])
        base_apply=np.ones(len(val["y"]),bool)
        if sel is None:guard_apply=np.zeros(len(val["y"]),bool)
        else:guard_apply=sel["orientation"]*val["v"]<=sel["threshold"]
        reports.append({"fold":f,"selected":sel,"fit_rows":int(len(fit["y"])),"val_rows":int(len(val["y"])),
                        "excluded_fit":fit["excluded"],"excluded_val":val["excluded"],
                        "val_base_all_actions":account(val["y"],base_apply),
                        "val_guarded":account(val["y"],guard_apply),
                        "guarded_by_k":{str(k):int(np.sum(guard_apply&(val["y"]==k))) for k in range(7)}})
    keys=("rows","applied","corrections","regressions","other_k_actions","global_net")
    total_guard={k:sum(r["val_guarded"][k] for r in reports) for k in keys}
    total_base={k:sum(r["val_base_all_actions"][k] for r in reports) for k in keys}
    require(total_base["corrections"]==108 and total_base["regressions"]==125 and total_base["global_net"]==-17,
            "source action accounting changed")
    report={"status":"completed","experiment":"v273_selected_f0_onset_contrast_full_population_guard",
            "feature":FEATURE,"source_run":37356100423,"outer_fold_3_used":False,
            "threshold_selection":"FIT K2/K3 only; exhaustive midpoint; maximize net then fewer regressions/more corrections/fewer actions; abstain if FIT net<=0",
            "action_population":"all valid B_low + base K3, including other true K","folds":reports,
            "total_base_all_actions":total_base,"total_guarded":total_guard,"automatic_promotion":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# Selected-F0 onset-contrast full-population Exact-K guard","",
           f"Frozen source all-actions control: **{total_base['corrections']} corrections / {total_base['regressions']} regressions / {total_base['other_k_actions']} other-K actions, net {total_base['global_net']:+d}**.",
           f"Guarded OOF: **{total_guard['corrections']} corrections / {total_guard['regressions']} regressions / {total_guard['other_k_actions']} other-K actions, net {total_guard['global_net']:+d}**.","",
           "| fold | actions | corr | reg | other K | net |","|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        q=r["val_guarded"];lines.append(f"| {r['fold']} | {q['applied']} | {q['corrections']} | {q['regressions']} | {q['other_k_actions']} | {q['global_net']:+d} |")
    lines+=["","Fold 3 excluded. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
