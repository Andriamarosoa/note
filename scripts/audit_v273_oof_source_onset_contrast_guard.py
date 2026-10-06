"""OOF-source-action full-population guard for selected-F0 onset contrast.

The original residual action status is reconstructed ONLY from each fold's held-out
VAL predictions. For a held-out fold f, threshold/orientation are learned from
original OOF actions belonging to the other internal folds. This matches the
protocol used by the earlier +6 K2/K3 audit while retaining all other-K actions.
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
from scripts.v273_residual_audit import FOLDS,require,verify_export,write_json

FEATURE="nov_onset_contrast_norm_median"

def feature(samples,start):
    ft,x=transition_spectrum(samples,start); r=extract_one(ft,x)
    if r is None:return None
    f0=(r["min_triplet_f0"],r["median_triplet_f0"],r["max_triplet_f0"])
    freq,powers,_=raw_powers(samples,start)
    return float(np.median([component_metrics(freq,powers,float(q))["onset_contrast_norm"] for q in f0]))

def account(y,apply):
    y=np.asarray(y,int);a=np.asarray(apply,bool)
    corr=int(np.sum(a&(y==2)));reg=int(np.sum(a&(y==3)));other=int(np.sum(a&~np.isin(y,(2,3))))
    return {"rows":int(len(y)),"applied":int(a.sum()),"corrections":corr,"regressions":reg,
            "other_k_actions":other,"global_net":corr-reg}

def thresholds(z):
    u=np.unique(np.asarray(z,float))
    if len(u)==1:return [float(u[0])]
    return [float(u[0]-1e-12),*map(float,(u[:-1]+u[1:])/2),float(u[-1]+1e-12)]

def select(y,v):
    y=np.asarray(y,int);v=np.asarray(v,float)
    m=np.isin(y,(2,3))&np.isfinite(v); y=y[m];v=v[m]
    require(np.any(y==2) and np.any(y==3),"binary collapse")
    best=None
    for orient in (1,-1):
        z=orient*v
        for t in thresholds(z):
            q=account(y,z<=t); key=(q["global_net"],-q["regressions"],q["corrections"],-q["applied"])
            if best is None or key>best[0]:best=(key,orient,float(t),q)
    return None if best[3]["global_net"]<=0 else {"orientation":best[1],"threshold":best[2],"fit_k23":best[3]}

def build_oof(exports):
    rows={}
    for f in FOLDS:
        folder=exports/f"fold-{f}";verify_export(folder)
        with np.load(folder/"replay.npz",allow_pickle=False) as z: arr=dict(z)
        action=arr["val_b_low"]&(arr["val_base_k"]==3); valid=arr["val_valid"]
        ids=arr["val_ids"][action][valid]; y=arr["val_true_k"][action][valid].astype(int)
        members=arr["val_recording"][action][valid].astype(str); starts=arr["val_start_sample"][action][valid].astype(int)
        probs=np.asarray(arr["val_probability"],float)
        require(len(ids)==len(probs),"val probability alignment drift")
        ff=arr["val_fold"][action][valid].astype(int);require(np.all(ff==f),"OOF fold drift")
        for rid,yy,m,s,p in zip(ids,y,members,starts,probs):
            rid=int(rid); item={"row_id":rid,"fold":f,"y":int(yy),"member":str(m),"start":int(s),"source_probability":float(p),"source_apply":bool(p>=.5)}
            require(rid not in rows,"OOF row duplicated");rows[rid]=item
    return rows

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"refusing overwrite")
    # also validates all source exports/identities
    action_inputs(a.exports)
    rows=build_oof(a.exports)
    wanted={r["member"] for r in rows.values()};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};require(set(tracks)==wanted,"audio coverage")
    by={}
    for rid,r in rows.items():by.setdefault(r["member"],[]).append((rid,r["start"]))
    started=time.monotonic()
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);s=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rows[rid]["value"]=feature(s,start)
        print(json.dumps({"recording":member,"done":sum("value" in r for r in rows.values()),"seconds":time.monotonic()-started}),flush=True)

    ordered=list(rows.values())
    source_y=np.asarray([r["y"] for r in ordered]); source_a=np.asarray([r["source_apply"] for r in ordered])
    control=account(source_y,source_a)
    require(control["corrections"]==108 and control["regressions"]==125 and control["other_k_actions"]==194 and control["global_net"]==-17,
            "OOF source accounting drift")

    reports=[]
    for f in FOLDS:
        fit=[r for r in ordered if r["fold"]!=f and r["source_apply"] and r.get("value") is not None]
        val=[r for r in ordered if r["fold"]==f and r["source_apply"]]
        sel=select([r["y"] for r in fit],[r["value"] for r in fit])
        yy=np.asarray([r["y"] for r in val],int); good=np.asarray([r.get("value") is not None for r in val],bool)
        apply=np.zeros(len(val),bool)
        if sel is not None:
            vv=np.asarray([np.nan if r.get("value") is None else r["value"] for r in val],float)
            apply[good]=sel["orientation"]*vv[good]<=sel["threshold"]
        q=account(yy,apply)
        reports.append({"fold":f,"selected":sel,"fit_source_actions":len(fit),"val_source_actions":len(val),
                        "feature_missing_val":int((~good).sum()),"val_guarded":q,
                        "by_k_actions":{str(k):int(np.sum(apply&(yy==k))) for k in range(7)}})
    keys=("rows","applied","corrections","regressions","other_k_actions","global_net")
    total={k:sum(r["val_guarded"][k] for r in reports) for k in keys}
    report={"status":"completed","experiment":"v273_selected_f0_onset_contrast_oof_source_guard","feature":FEATURE,
            "source_run":37356100423,"outer_fold_3_used":False,
            "source_action_definition":"each row's own held-out VAL residual probability >= 0.5",
            "fit_policy":"for heldout fold f, train threshold only on OOF source actions from folds != f with true K2/K3",
            "source_control":control,"folds":reports,"total_guarded":total,"automatic_promotion":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# OOF-source selected-F0 onset-contrast full-population guard","",
           f"Original OOF residual actions: **{control['corrections']}/{control['regressions']}**, other-K **{control['other_k_actions']}**, net **{control['global_net']:+d}**.",
           f"Filtered OOF actions: **{total['corrections']}/{total['regressions']}**, other-K **{total['other_k_actions']}**, net **{total['global_net']:+d}**.","",
           "| fold | source actions | kept | corr | reg | other K | net |","|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        q=r["val_guarded"];lines.append(f"| {r['fold']} | {r['val_source_actions']} | {q['applied']} | {q['corrections']} | {q['regressions']} | {q['other_k_actions']} | {q['global_net']:+d} |")
    lines+=["","Fold 3 excluded. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
