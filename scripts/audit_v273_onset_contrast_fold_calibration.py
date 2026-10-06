"""Audit fold calibration of the selected-F0 onset-contrast feature on original OOF residual actions.

No new Exact-K policy is evaluated. This describes per-fold K2/K3 distributions,
AUCs, within-fold oracle split, and the absolute-threshold transfer selected from
the other folds. Purpose: determine whether fold 2 failure is ranking failure or
cross-fold calibration shift.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
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

def build_oof(exports):
    rows={}
    for f in FOLDS:
        folder=exports/f"fold-{f}";verify_export(folder)
        with np.load(folder/"replay.npz",allow_pickle=False) as z:arr=dict(z)
        action=arr["val_b_low"]&(arr["val_base_k"]==3);valid=arr["val_valid"]
        ids=arr["val_ids"][action][valid];y=arr["val_true_k"][action][valid].astype(int)
        m=arr["val_recording"][action][valid].astype(str);s=arr["val_start_sample"][action][valid].astype(int)
        p=np.asarray(arr["val_probability"],float);ff=arr["val_fold"][action][valid].astype(int)
        require(len(ids)==len(p) and np.all(ff==f),"alignment drift")
        for rid,yy,mm,ss,pp in zip(ids,y,m,s,p):
            rid=int(rid);require(rid not in rows,"duplicate")
            rows[rid]={"row_id":rid,"fold":f,"y":int(yy),"member":str(mm),"start":int(ss),"source_apply":bool(pp>=.5)}
    return rows

def summ(x):
    x=np.asarray(x,float)
    return {"n":int(len(x)),"mean":float(x.mean()),"median":float(np.median(x)),
            "q10":float(np.quantile(x,.1)),"q25":float(np.quantile(x,.25)),
            "q75":float(np.quantile(x,.75)),"q90":float(np.quantile(x,.9))}

def account(y,a):
    y=np.asarray(y,int);a=np.asarray(a,bool)
    c=int(np.sum(a&(y==2)));r=int(np.sum(a&(y==3)))
    return {"applied":int(a.sum()),"corrections":c,"regressions":r,"net":c-r}

def candidate_thresholds(z):
    u=np.unique(np.asarray(z,float))
    if len(u)==1:return [float(u[0])]
    return [float(u[0]-1e-12),*map(float,(u[:-1]+u[1:])/2),float(u[-1]+1e-12)]

def pooled_threshold(rows,heldout):
    rr=[r for r in rows if r["fold"]!=heldout and r["source_apply"] and r["y"] in (2,3) and r.get("value") is not None]
    y=np.asarray([r["y"] for r in rr],int);v=np.asarray([r["value"] for r in rr],float)
    best=None
    for orient in (1,-1):
        z=orient*v
        for t in candidate_thresholds(z):
            q=account(y,z<=t);key=(q["net"],-q["regressions"],q["corrections"],-q["applied"])
            if best is None or key>best[0]:best=(key,orient,float(t),q)
    return {"orientation":best[1],"threshold":best[2],"fit":best[3]}

def oracle_threshold(y,v):
    best=None
    for orient in (1,-1):
        z=orient*v
        for t in candidate_thresholds(z):
            q=account(y,z<=t);key=(q["net"],-q["regressions"],q["corrections"],-q["applied"])
            if best is None or key>best[0]:best=(key,orient,float(t),q)
    return {"orientation":best[1],"threshold":best[2],"result":best[3]}

def percentile(sorted_fit,x):
    return float(np.searchsorted(sorted_fit,x,side="right")/len(sorted_fit))

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
    rr=list(rows.values());folds={}
    for f in FOLDS:
        val=[r for r in rr if r["fold"]==f and r["source_apply"] and r["y"] in (2,3) and r.get("value") is not None]
        y=np.asarray([r["y"] for r in val],int);v=np.asarray([r["value"] for r in val],float)
        # orient AUC to K2 using fit direction only
        fit=[r for r in rr if r["fold"]!=f and r["source_apply"] and r["y"] in (2,3) and r.get("value") is not None]
        yf=np.asarray([r["y"] for r in fit],int);vf=np.asarray([r["value"] for r in fit],float)
        orient=1 if np.median(vf[yf==2])<=np.median(vf[yf==3]) else -1
        auc=float(roc_auc_score((y==2).astype(int),-orient*v))
        sel=pooled_threshold(rr,f);zfit=np.sort(sel["orientation"]*vf);thr_pct=percentile(zfit,sel["threshold"])
        apply=sel["orientation"]*v<=sel["threshold"]
        oracle=oracle_threshold(y,v)
        folds[str(f)]={"K2":summ(v[y==2]),"K3":summ(v[y==3]),"oriented_auc":auc,
                       "fit_selected":sel,"fit_threshold_percentile":thr_pct,"val_at_fit_threshold":account(y,apply),
                       "val_oracle":oracle}
    report={"status":"completed","experiment":"v273_onset_contrast_fold_calibration","outer_fold_3_used":False,
            "source_actions_only":True,"folds":folds,"prediction_changes":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# Onset-contrast fold calibration audit","",
           "| fold | K2 med | K3 med | oriented AUC | FIT threshold | threshold percentile | transferred net | oracle net |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in FOLDS:
        q=folds[str(f)];lines.append(f"| {f} | {q['K2']['median']:.6f} | {q['K3']['median']:.6f} | {q['oriented_auc']:.3f} | {q['fit_selected']['threshold']:.6f} | {q['fit_threshold_percentile']:.3f} | {q['val_at_fit_threshold']['net']:+d} | {q['val_oracle']['result']['net']:+d} |")
    lines+=["","Diagnostic only; oracle values describe fold separability and are not a deployable policy."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
