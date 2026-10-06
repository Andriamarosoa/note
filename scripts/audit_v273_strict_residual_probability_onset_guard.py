"""Strict OOF conjunctive guard: residual K2 probability AND selected-F0 onset contrast.

Only original held-out OOF residual actions (p>=0.5) are eligible. For each held-out
fold, (p0, onset threshold, orientation) is selected on the other three OOF folds.
A candidate policy is admissible only if its K2/K3 Exact-K net is >=0 on EVERY
FIT fold and total FIT net >0. Held-out promotion criterion is all fold nets >=0
and positive total net.
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

def feature(samples,start):
    ft,x=transition_spectrum(samples,start);r=extract_one(ft,x)
    if r is None:return None
    f0=(r["min_triplet_f0"],r["median_triplet_f0"],r["max_triplet_f0"])
    freq,powers,_=raw_powers(samples,start)
    return float(np.median([component_metrics(freq,powers,float(q))["onset_contrast_norm"] for q in f0]))

def account(y,a):
    y=np.asarray(y,int);a=np.asarray(a,bool)
    c=int(np.sum(a&(y==2)));r=int(np.sum(a&(y==3)));o=int(np.sum(a&~np.isin(y,(2,3))))
    return {"rows":int(len(y)),"applied":int(a.sum()),"corrections":c,"regressions":r,"other_k_actions":o,"global_net":c-r}

def mids(x,lower=None):
    u=np.unique(np.asarray(x,float))
    if lower is not None:u=u[u>=lower]
    require(len(u)>0,"no thresholds")
    if len(u)==1:return [float(u[0])]
    return [float(u[0]),*map(float,(u[:-1]+u[1:])/2),float(u[-1]+1e-12)]

def build_oof(exports):
    rows={}
    for f in FOLDS:
        folder=exports/f"fold-{f}";verify_export(folder)
        with np.load(folder/"replay.npz",allow_pickle=False) as z:arr=dict(z)
        action=arr["val_b_low"]&(arr["val_base_k"]==3);valid=arr["val_valid"]
        ids=arr["val_ids"][action][valid];y=arr["val_true_k"][action][valid].astype(int)
        mem=arr["val_recording"][action][valid].astype(str);start=arr["val_start_sample"][action][valid].astype(int)
        p=np.asarray(arr["val_probability"],float);ff=arr["val_fold"][action][valid].astype(int)
        require(len(ids)==len(p) and np.all(ff==f),"OOF alignment drift")
        for rid,yy,m,s,pp in zip(ids,y,mem,start,p):
            rid=int(rid);require(rid not in rows,"duplicate OOF row")
            rows[rid]={"row_id":rid,"fold":f,"y":int(yy),"member":str(m),"start":int(s),
                       "source_probability":float(pp),"source_apply":bool(pp>=.5)}
    return rows

def select(rows,heldout):
    rr=[r for r in rows if r["fold"]!=heldout and r["source_apply"] and r["y"] in (2,3) and r.get("value") is not None]
    y=np.asarray([r["y"] for r in rr],int);p=np.asarray([r["source_probability"] for r in rr],float)
    v=np.asarray([r["value"] for r in rr],float);f=np.asarray([r["fold"] for r in rr],int)
    fitfolds=[x for x in FOLDS if x!=heldout];require(set(np.unique(f))==set(fitfolds),"fit fold drift")
    pth=mids(p,0.5)
    best=None
    for orient in (1,-1):
        z=orient*v
        for zt in mids(z):
            zm=z<=zt
            if not np.any(zm):continue
            for pt in pth:
                a=zm&(p>=pt)
                per={};nets=[]
                for ff in fitfolds:
                    m=f==ff;q=account(y[m],a[m]);per[str(ff)]=q;nets.append(q["global_net"])
                total=account(y,a)
                if min(nets)<0 or total["global_net"]<=0:continue
                key=(min(nets),total["global_net"],-total["regressions"],total["corrections"],-total["applied"],pt)
                if best is None or key>best[0]:
                    best=(key,orient,float(zt),float(pt),total,per)
    if best is None:return None
    return {"orientation":best[1],"onset_threshold":best[2],"probability_threshold":best[3],
            "fit_total":best[4],"fit_by_fold":best[5],"min_fit_fold_net":best[0][0]}

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
        val=[r for r in rr if r["fold"]==f and r["source_apply"]];yy=np.asarray([r["y"] for r in val],int)
        vv=np.asarray([np.nan if r.get("value") is None else r["value"] for r in val],float)
        pp=np.asarray([r["source_probability"] for r in val],float);good=np.isfinite(vv);ap=np.zeros(len(val),bool)
        if sel is not None:
            ap[good]=(pp[good]>=sel["probability_threshold"])&(sel["orientation"]*vv[good]<=sel["onset_threshold"])
        reports.append({"fold":f,"selected":sel,"source_actions":len(val),"feature_missing":int((~good).sum()),
                        "val_guarded":account(yy,ap),"by_k_actions":{str(k):int(np.sum(ap&(yy==k))) for k in range(7)}})
    keys=("rows","applied","corrections","regressions","other_k_actions","global_net")
    total={k:sum(r["val_guarded"][k] for r in reports) for k in keys}
    passed=all(r["val_guarded"]["global_net"]>=0 for r in reports) and total["global_net"]>0
    report={"status":"completed","experiment":"v273_strict_conjunctive_residual_probability_onset_guard",
            "inputs":["original OOF residual probability","selected-F0 median normalized onset contrast"],
            "outer_fold_3_used":False,"source_control":control,"folds":reports,"total_guarded":total,
            "promotion_criterion":"all held-out fold nets >=0 and total net >0","strict_internal_pass":passed,
            "automatic_promotion":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# Strict conjunctive residual-probability + onset guard","",
           f"Source: **{control['corrections']}/{control['regressions']}**, other-K {control['other_k_actions']}, net {control['global_net']:+d}.",
           f"Guarded: **{total['corrections']}/{total['regressions']}**, other-K {total['other_k_actions']}, net {total['global_net']:+d}.",
           f"Strict internal pass: **{passed}**.","",
           "| fold | source | kept | corr | reg | other K | net | p threshold | onset threshold |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        q=r["val_guarded"];s=r["selected"];pt="abstain" if s is None else f"{s['probability_threshold']:.6f}";ot="abstain" if s is None else f"{s['onset_threshold']:.6f}"
        lines.append(f"| {r['fold']} | {r['source_actions']} | {q['applied']} | {q['corrections']} | {q['regressions']} | {q['other_k_actions']} | {q['global_net']:+d} | {pt} | {ot} |")
    lines+=["","Fold 3 excluded. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
