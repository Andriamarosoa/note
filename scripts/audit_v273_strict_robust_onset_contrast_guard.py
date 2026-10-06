"""Strict-fold-robust OOF guard for selected-F0 onset contrast.

For each held-out fold, candidate thresholds are evaluated on the OTHER three OOF
folds separately. A threshold is admissible only if K2/K3 Exact-K net is >=0 on
EVERY FIT fold and total FIT net >0. Selection then maximizes minimum fold net,
total net, fewer regressions, more corrections, fewer actions.
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

def account(y,a):
    y=np.asarray(y,int);a=np.asarray(a,bool)
    c=int(np.sum(a&(y==2)));r=int(np.sum(a&(y==3)));o=int(np.sum(a&~np.isin(y,(2,3))))
    return {"rows":int(len(y)),"applied":int(a.sum()),"corrections":c,"regressions":r,"other_k_actions":o,"global_net":c-r}

def thresholds(z):
    u=np.unique(np.asarray(z,float))
    if len(u)==1:return [float(u[0])]
    return [float(u[0]-1e-12),*map(float,(u[:-1]+u[1:])/2),float(u[-1]+1e-12)]

def select(rows,heldout):
    rr=[r for r in rows if r["fold"]!=heldout and r["source_apply"] and r.get("value") is not None and r["y"] in (2,3)]
    y=np.asarray([r["y"] for r in rr],int);v=np.asarray([r["value"] for r in rr],float);f=np.asarray([r["fold"] for r in rr],int)
    require(set(np.unique(f))==set(x for x in FOLDS if x!=heldout),"fit fold coverage drift")
    best=None
    for orient in (1,-1):
        z=orient*v
        for t in thresholds(z):
            a=z<=t
            per={}
            for ff in FOLDS:
                if ff==heldout:continue
                m=f==ff;per[str(ff)]=account(y[m],a[m])
            nets=[q["global_net"] for q in per.values()]
            total=account(y,a)
            if min(nets)<0 or total["global_net"]<=0:continue
            key=(min(nets),total["global_net"],-total["regressions"],total["corrections"],-total["applied"])
            if best is None or key>best[0]:best=(key,orient,float(t),total,per)
    if best is None:return None
    return {"orientation":best[1],"threshold":best[2],"fit_total":best[3],"fit_by_fold":best[4],"min_fit_fold_net":best[0][0]}

def build_oof(exports):
    rows={}
    for f in FOLDS:
        folder=exports/f"fold-{f}";verify_export(folder)
        with np.load(folder/"replay.npz",allow_pickle=False) as z:arr=dict(z)
        action=arr["val_b_low"]&(arr["val_base_k"]==3);valid=arr["val_valid"]
        ids=arr["val_ids"][action][valid];y=arr["val_true_k"][action][valid].astype(int)
        members=arr["val_recording"][action][valid].astype(str);starts=arr["val_start_sample"][action][valid].astype(int)
        probs=np.asarray(arr["val_probability"],float);ff=arr["val_fold"][action][valid].astype(int)
        require(len(ids)==len(probs) and np.all(ff==f),"OOF alignment drift")
        for rid,yy,m,s,p in zip(ids,y,members,starts,probs):
            rid=int(rid);require(rid not in rows,"duplicate OOF row")
            rows[rid]={"row_id":rid,"fold":f,"y":int(yy),"member":str(m),"start":int(s),"source_apply":bool(p>=.5)}
    return rows

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"refusing overwrite");action_inputs(a.exports)
    rows=build_oof(a.exports);wanted={r["member"] for r in rows.values()}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};require(set(tracks)==wanted,"audio")
    by={}
    for rid,r in rows.items():by.setdefault(r["member"],[]).append((rid,r["start"]))
    started=time.monotonic()
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);s=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rows[rid]["value"]=feature(s,start)
        print(json.dumps({"recording":member,"done":sum("value" in r for r in rows.values()),"seconds":time.monotonic()-started}),flush=True)
    rr=list(rows.values());y=np.asarray([r["y"] for r in rr]);src=np.asarray([r["source_apply"] for r in rr])
    control=account(y,src);require(control["corrections"]==108 and control["regressions"]==125 and control["other_k_actions"]==194,"source accounting drift")
    reports=[]
    for f in FOLDS:
        sel=select(rr,f)
        val=[r for r in rr if r["fold"]==f and r["source_apply"]]; yy=np.asarray([r["y"] for r in val],int)
        vv=np.asarray([np.nan if r.get("value") is None else r["value"] for r in val],float);good=np.isfinite(vv);ap=np.zeros(len(val),bool)
        if sel is not None:ap[good]=sel["orientation"]*vv[good]<=sel["threshold"]
        reports.append({"fold":f,"selected":sel,"source_actions":len(val),"feature_missing":int((~good).sum()),"val_guarded":account(yy,ap),
                        "by_k_actions":{str(k):int(np.sum(ap&(yy==k))) for k in range(7)}})
    keys=("rows","applied","corrections","regressions","other_k_actions","global_net")
    total={k:sum(r["val_guarded"][k] for r in reports) for k in keys}
    strict=all(r["val_guarded"]["global_net"]>=0 for r in reports) and total["global_net"]>0
    report={"status":"completed","experiment":"v273_onset_contrast_strict_fold_robust_guard","feature":FEATURE,"source_run":37356100423,
            "outer_fold_3_used":False,"selection":"threshold must be >=0 net on every FIT fold; total FIT net >0; maximize min fold net then total net",
            "source_control":control,"folds":reports,"total_guarded":total,"strict_internal_pass":strict,"automatic_promotion":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# Strict-fold-robust onset-contrast guard","",
           f"Source: **{control['corrections']}/{control['regressions']}**, other-K {control['other_k_actions']}, net {control['global_net']:+d}.",
           f"Guarded: **{total['corrections']}/{total['regressions']}**, other-K {total['other_k_actions']}, net {total['global_net']:+d}.",
           f"Strict internal pass: **{strict}**.","","| fold | source | kept | corr | reg | other K | net | min FIT-fold net |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        q=r["val_guarded"];mn="abstain" if r["selected"] is None else str(r["selected"]["min_fit_fold_net"])
        lines.append(f"| {r['fold']} | {r['source_actions']} | {q['applied']} | {q['corrections']} | {q['regressions']} | {q['other_k_actions']} | {q['global_net']:+d} | {mn} |")
    lines+=["","Fold 3 excluded. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
