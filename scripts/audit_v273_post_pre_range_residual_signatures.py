"""Describe residual K2 corrections vs K3 regressions after the strict pre+range veto.

Replays the fixed fold-wise median veto (pre_high + amp_range_low), verifies the
84 remaining corrections / 77 remaining regressions, then recomputes the broad
signature audit without changing any prediction.

Special attention is given to fold 2, whose kept net remains negative after the
veto. Fold-2 feature orientation is learned from the other internal folds only.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_correction_regression_signatures import build_features,feature_report
from scripts.audit_v273_full_oof_stable_signature_veto import extract_features
from scripts.v273_residual_audit import FOLDS,require

GROUPS=("K2_corrected","K3_regressed")
GUARD=(("pre_high","nov_pre_norm_median","high"),("amp_range_low","amp_range","low"))

def guard_thresholds(rows,fold):
    fit=[r for r in rows if r["fold"]!=fold]
    out={}
    for alias,key,direction in GUARD:
        vals=[r["guard_features"][key] for r in fit]
        out[alias]={"feature":key,"direction":direction,"threshold":float(np.median(vals))}
    return out

def veto(r,thr):
    return r["guard_features"]["nov_pre_norm_median"]>=thr["pre_high"]["threshold"] and r["guard_features"]["amp_range"]<=thr["amp_range_low"]["threshold"]

def fold2_report(rows,names):
    y=np.asarray([r["group"]=="K3_regressed" for r in rows],int)
    folds=np.asarray([r["fold"] for r in rows],int)
    out=[]
    for n in names:
        x=np.asarray([r["features"].get(n,np.nan) for r in rows],float)
        fit=(folds!=2)&np.isfinite(x); val=(folds==2)&np.isfinite(x)
        if fit.sum()<20 or val.sum()<4 or len(np.unique(y[fit]))<2 or len(np.unique(y[val]))<2:continue
        orient=1 if np.median(x[fit&(y==1)])>=np.median(x[fit&(y==0)]) else -1
        a=float(roc_auc_score(y[val],orient*x[val]))
        fit_auc=float(roc_auc_score(y[fit],orient*x[fit]))
        out.append({"feature":n,"fold2_auc":a,"other_folds_auc":fit_auc,
                    "direction":"higher_in_regressions" if orient==1 else "lower_in_regressions",
                    "fold2_K2_median":float(np.median(x[val&(y==0)])),
                    "fold2_K3_median":float(np.median(x[val&(y==1)]))})
    out.sort(key=lambda q:q["fold2_auc"],reverse=True)
    return out

def main():
    p=argparse.ArgumentParser()
    for n in ("cases","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"overwrite")
    cases=[json.loads(s) for s in a.cases.read_text().splitlines()]
    cases=[r for r in cases if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in cases)==108,"K2 drift")
    require(sum(r["group"]=="K3_regressed" for r in cases)==125,"K3 drift")
    require(all(r["fold"] in FOLDS and r["fold"]!=3 for r in cases),"fold leak")
    wanted={r["recording_id"] for r in cases}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for r in cases:by[r["recording_id"]].append(r)
    rows=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for r in sorted(by[member],key=lambda z:z["row_id"]):
            rows.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],"recording_id":member,
                         "guard_features":extract_features(samples,int(r["start_sample"])),
                         "features":build_features(r,samples)})
        print(json.dumps({"recording":member,"done":len(rows)}),flush=True)
    require(all(r["guard_features"] is not None for r in rows),"guard feature missing")

    thresholds={str(f):guard_thresholds(rows,f) for f in FOLDS}
    for r in rows:r["veto"]=veto(r,thresholds[str(r["fold"])])
    blocked_k2=sum(r["veto"] and r["group"]=="K2_corrected" for r in rows)
    blocked_k3=sum(r["veto"] and r["group"]=="K3_regressed" for r in rows)
    require((blocked_k2,blocked_k3)==(24,48),f"guard replay drift {(blocked_k2,blocked_k3)}")
    kept=[r for r in rows if not r["veto"]]
    require(sum(r["group"]=="K2_corrected" for r in kept)==84,"kept K2 drift")
    require(sum(r["group"]=="K3_regressed" for r in kept)==77,"kept K3 drift")

    common=sorted(set.intersection(*(set(r["features"]) for r in kept)))
    reports=[q for n in common if (q:=feature_report(kept,n)) is not None]
    reports.sort(key=lambda q:(q["stable_4_of_4"],q["min_oriented_auc"],q["mean_oriented_auc"]),reverse=True)
    stable=[q for q in reports if q["stable_4_of_4"] and q["min_oriented_auc"]>=.55]
    f2=fold2_report(kept,common)

    byfold={}
    for f in FOLDS:
        rr=[r for r in kept if r["fold"]==f]
        byfold[str(f)]={"K2_corrected":sum(r["group"]=="K2_corrected" for r in rr),
                        "K3_regressed":sum(r["group"]=="K3_regressed" for r in rr),
                        "net":sum(r["group"]=="K2_corrected" for r in rr)-sum(r["group"]=="K3_regressed" for r in rr)}

    rep={"status":"completed","experiment":"v273_post_pre_range_residual_signatures",
         "guard_replay":{"blocked_K2":blocked_k2,"blocked_K3":blocked_k3,"gain":blocked_k3-blocked_k2},
         "kept":{"K2_corrected":84,"K3_regressed":77},"kept_by_fold":byfold,
         "features_evaluated":len(reports),"stable_residual_signatures":stable,
         "fold2_oriented_from_other_folds":f2,
         "outer_fold_3_used":False,"prediction_changes":False,"threshold_search":False}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Residual signatures after pre+range veto","",
           "Guard replay: blocked **48 K3 regressions / 24 K2 corrections**. Remaining: **77 regressions / 84 corrections**.","",
           "## Remaining Exact-K balance by fold","",
           "| fold | K2 corrections | K3 regressions | net |","|---:|---:|---:|---:|"]
    for f in FOLDS:
        q=byfold[str(f)];lines.append(f"| {f} | {q['K2_corrected']} | {q['K3_regressed']} | {q['net']:+d} |")
    lines+=["","## Stable residual signatures across all four folds","",
            "| feature | direction in regressions | K2 median | K3 median | mean AUC | min fold AUC |",
            "|---|---|---:|---:|---:|---:|"]
    for q in stable[:20]:
        lines.append(f"| {q['feature']} | {q['direction']} | {q['K2_corrected']['median']:.6g} | {q['K3_regressed']['median']:.6g} | {q['mean_oriented_auc']:.3f} | {q['min_oriented_auc']:.3f} |")
    lines+=["","## Fold 2 — orientation learned from folds 0/1/4","",
            "| feature | direction | fold2 AUC | other-fold AUC | fold2 K2 median | fold2 K3 median |",
            "|---|---|---:|---:|---:|---:|"]
    for q in f2[:15]:
        lines.append(f"| {q['feature']} | {q['direction']} | {q['fold2_auc']:.3f} | {q['other_folds_auc']:.3f} | {q['fold2_K2_median']:.6g} | {q['fold2_K3_median']:.6g} |")
    lines+=["","Diagnostic only; no second guard selected in this run."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
