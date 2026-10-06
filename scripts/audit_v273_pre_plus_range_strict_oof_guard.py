"""Strict internal audit for the pre_high + amp_range_low veto.

For each held-out internal fold, thresholds are medians from source-action K2/K3
rows in the other internal folds only. The fixed signal directions and pair were
chosen by prior completed audits. No numeric threshold search.

Reports full 427-action Exact-K accounting, per-fold robustness, and style transfer.
Fold 3 excluded. No automatic promotion.
"""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_oof_source_onset_contrast_guard import build_oof,account
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_full_oof_stable_signature_veto import extract_features,hit
from scripts.v273_residual_audit import FOLDS,require

PATTERN=("pre_high","amp_range_low")
SIGNALS={
 "pre_high":("nov_pre_norm_median","high"),
 "amp_range_low":("amp_range","low"),
}

def style(member):
    m=re.match(r"^\d{2}_([A-Za-z]+)\d+-",Path(member).name)
    require(m is not None,f"bad style: {member}")
    return m.group(1)

def fit_thresholds(rows):
    out={}
    for alias,(feature,direction) in SIGNALS.items():
        vals=[r["features"][feature] for r in rows if r.get("features") is not None and r["y"] in (2,3)]
        require(len(vals)>=20,f"small fit {alias}")
        out[alias]={"feature":feature,"direction":direction,"threshold":float(np.median(vals))}
    return out

def blocked(rows,veto):
    k3=k2=other=0
    for r,v in zip(rows,veto):
        if not v:continue
        if r["y"]==3:k3+=1
        elif r["y"]==2:k2+=1
        else:other+=1
    return {"blocked_reg":k3,"blocked_corr":k2,"blocked_other":other,"gain":k3-k2,
            "precision":None if k3+k2==0 else k3/(k3+k2)}

def kept_account(rows,veto):
    y=np.asarray([r["y"] for r in rows],int)
    keep=np.asarray([not v for v in veto],bool)
    return account(y,keep)

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"overwrite")
    action_inputs(a.exports); rows_map=build_oof(a.exports)
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
        for rid,start in by[member]:rows_map[rid]["features"]=extract_features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rows_map.values())}),flush=True)

    rows=[r for r in rows_map.values() if r["source_apply"]]
    require(len(rows)==427,"source action count drift")
    require(sum(r["y"]==2 for r in rows)==108 and sum(r["y"]==3 for r in rows)==125,"k23 drift")

    fold_thresholds={}
    veto=[]
    for r in rows:
        f=r["fold"]
        if str(f) not in fold_thresholds:
            fit=[x for x in rows if x["fold"]!=f and x.get("features") is not None and x["y"] in (2,3)]
            fold_thresholds[str(f)]=fit_thresholds(fit)
        veto.append(r.get("features") is not None and hit(r["features"],fold_thresholds[str(f)],PATTERN))

    source=account(np.asarray([r["y"] for r in rows],int),np.ones(len(rows),bool))
    b=blocked(rows,veto);kept=kept_account(rows,veto)
    require(source["corrections"]==108 and source["regressions"]==125 and source["other_k_actions"]==194,"source drift")

    per_fold=[]
    for f in FOLDS:
        rr=[r for r in rows if r["fold"]==f]
        vv=[v for r,v in zip(rows,veto) if r["fold"]==f]
        bb=blocked(rr,vv);kk=kept_account(rr,vv)
        per_fold.append({"fold":int(f),"blocked":bb,"kept":kk})

    # Leave-one-style-out + leave-fold-out transfer with same fixed pair.
    style_reports=[]
    for st in sorted({r["style"] for r in rows}):
        held=[r for r in rows if r["style"]==st]
        byfold={}
        vv=[]
        for r in held:
            f=r["fold"]
            if str(f) not in byfold:
                fit=[x for x in rows if x["style"]!=st and x["fold"]!=f and x.get("features") is not None and x["y"] in (2,3)]
                byfold[str(f)]=fit_thresholds(fit)
            vv.append(r.get("features") is not None and hit(r["features"],byfold[str(f)],PATTERN))
        q=blocked(held,vv);q["style"]=st;style_reports.append(q)

    report={"status":"completed","experiment":"v273_pre_plus_range_strict_oof_guard",
            "pattern":list(PATTERN),"threshold_rule":"median on other internal folds only",
            "source_control":source,"blocked_total":b,"kept_total":kept,
            "net_improvement_vs_source":kept["global_net"]-source["global_net"],
            "per_fold":per_fold,"leave_one_style_out":style_reports,
            "strict_internal_pass":{
              "all_folds_nonnegative_gain":all(q["blocked"]["gain"]>=0 for q in per_fold),
              "all_styles_nonnegative_gain":all(q["gain"]>=0 for q in style_reports),
              "global_positive_gain":b["gain"]>0,
            },
            "outer_fold_3_used":False,"prediction_changes":False,"posthoc_internal_reuse":True,"automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Pre-high + amp-range strict OOF veto","",
           f"Source: **108 corrections / 125 regressions / 194 other-K**, net **{source['global_net']:+d}**.",
           f"Blocked by veto: **{b['blocked_reg']} regressions / {b['blocked_corr']} corrections / {b['blocked_other']} other-K**, gain **{b['gain']:+d}**.",
           f"Kept actions: **{kept['corrections']} corrections / {kept['regressions']} regressions / {kept['other_k_actions']} other-K**, net **{kept['global_net']:+d}**.","",
           "## Per-fold blocked gain","",
           "| fold | blocked K3 | blocked K2 | gain | kept net |","|---:|---:|---:|---:|---:|"]
    for q in per_fold:
        lines.append(f"| {q['fold']} | {q['blocked']['blocked_reg']} | {q['blocked']['blocked_corr']} | {q['blocked']['gain']:+d} | {q['kept']['global_net']:+d} |")
    lines+=["","## Leave-one-style-out transfer","","| style | blocked K3 | blocked K2 | gain | precision |","|---|---:|---:|---:|---:|"]
    for q in style_reports:
        pr="n/a" if q["precision"] is None else f"{q['precision']:.3f}"
        lines.append(f"| {q['style']} | {q['blocked_reg']} | {q['blocked_corr']} | {q['gain']:+d} | {pr} |")
    lines+=["",f"Strict internal pass: **{report['strict_internal_pass']}**.","","Internal/post-hoc only; fold 3 excluded; no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
