"""Generalization audit for the pre_high + amp_range_low + amp_cv_low veto.

Checks whether the +26 blocked-regression advantage found internally is concentrated
in particular GuitarSet players/styles/tracks, and whether the same fixed directions
transfer when thresholds are recalibrated without the held-out player/style AND
without the held-out fold.

Diagnostic/post-hoc on internal folds only. No fold 3. No model promotion.
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

PATTERN=("pre_high","amp_range_low","amp_cv_low")
SIGNALS={
 "pre_high":("nov_pre_norm_median","high"),
 "amp_range_low":("amp_range","low"),
 "amp_cv_low":("amp_cv","low"),
}

def meta(member):
    stem=Path(member).stem
    m=re.match(r"^(\d{2})_([A-Za-z]+)(\d+)-(.+?)_(comp|solo)$",stem)
    require(m is not None,f"unparsed GuitarSet name: {member}")
    player,style,variant,rest,part=m.groups()
    return {
      "player":player,
      "style":style,
      "variant":style+variant,
      "part":part,
      "piece":f"{style}{variant}-{rest}",
      "track_family":f"{style}{variant}-{rest}_{part}",
    }

def thresholds_from(rows):
    out={}
    for alias,(feature,direction) in SIGNALS.items():
        vals=[r["features"][feature] for r in rows if r.get("features") is not None and r["y"] in (2,3)]
        require(len(vals)>=8,f"small threshold fit for {alias}")
        out[alias]={"feature":feature,"direction":direction,"threshold":float(np.median(vals))}
    return out

def group_block(rows,veto):
    src=np.asarray([r["source_apply"] for r in rows],bool)
    y=np.asarray([r["y"] for r in rows],int)
    v=np.asarray(veto,bool)&src
    blocked_corr=int(np.sum(v&(y==2)));blocked_reg=int(np.sum(v&(y==3)))
    blocked_other=int(np.sum(v&~np.isin(y,(2,3))))
    return {
      "source_actions":int(src.sum()),"blocked":int(v.sum()),
      "blocked_corr":blocked_corr,"blocked_reg":blocked_reg,"blocked_other":blocked_other,
      "veto_gain":blocked_reg-blocked_corr,
      "blocked_k23_precision":None if blocked_corr+blocked_reg==0 else blocked_reg/(blocked_corr+blocked_reg),
    }

def fixed_group_breakdown(rows,group_key,pattern_thresholds):
    vals=[]
    for g in sorted({r[group_key] for r in rows}):
        rr=[r for r in rows if r[group_key]==g]
        veto=[]
        for r in rr:
            if not r["source_apply"] or r.get("features") is None:
                veto.append(False);continue
            veto.append(hit(r["features"],pattern_thresholds[str(r["fold"])],PATTERN))
        q=group_block(rr,veto);q[group_key]=g;vals.append(q)
    return vals

def transfer_holdout(rows,group_key):
    reports=[]
    groups=sorted({r[group_key] for r in rows})
    for g in groups:
        held=[r for r in rows if r[group_key]==g and r["source_apply"]]
        require(held,f"empty holdout {group_key}={g}")
        veto=[];thr_by_fold={}
        for r in held:
            f=r["fold"]
            key=str(f)
            if key not in thr_by_fold:
                fit=[x for x in rows if x["source_apply"] and x[group_key]!=g and x["fold"]!=f and x.get("features") is not None and x["y"] in (2,3)]
                thr_by_fold[key]=thresholds_from(fit)
            veto.append(r.get("features") is not None and hit(r["features"],thr_by_fold[key],PATTERN))
        q=group_block(held,veto)
        # fold-wise sign transfer inside holdout group
        pf=[]
        for f in FOLDS:
            rr=[r for r in held if r["fold"]==f]
            if not rr: continue
            vv=[]
            for r in rr:
                vv.append(r.get("features") is not None and hit(r["features"],thr_by_fold[str(f)],PATTERN))
            z=group_block(rr,vv);z["fold"]=int(f);pf.append(z)
        q.update({group_key:g,"thresholds_by_fold":thr_by_fold,"per_fold":pf})
        reports.append(q)
    return reports

def concentration(groups):
    gains=np.asarray([q["veto_gain"] for q in groups],int)
    positives=[int(x) for x in gains if x>0]
    total=int(gains.sum())
    abs_total=int(np.abs(gains).sum())
    return {
      "groups":len(groups),"positive_groups":int(np.sum(gains>0)),"zero_groups":int(np.sum(gains==0)),
      "negative_groups":int(np.sum(gains<0)),"total_gain":total,
      "max_group_gain":int(gains.max()) if len(gains) else 0,
      "min_group_gain":int(gains.min()) if len(gains) else 0,
      "largest_positive_share_of_net":None if total<=0 else float(max(positives,default=0)/total),
      "absolute_contribution_concentration":None if abs_total==0 else float(np.max(np.abs(gains))/abs_total),
    }

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","pattern-report","output"):p.add_argument("--"+n.replace("_","-"),dest=n.replace("-","_"),type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"refusing overwrite")
    action_inputs(a.exports)
    rows_map=build_oof(a.exports)
    pattern=json.loads(a.pattern_report.read_text())
    require(pattern["experiment"]=="v273_stable_signature_intersections","pattern drift")
    pattern_thresholds=pattern["thresholds"]

    wanted={r["member"] for r in rows_map.values()}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for rid,r in rows_map.items():
        r.update(meta(r["member"]))
        if r["source_apply"]:by[r["member"]].append((rid,r["start"]))
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rows_map[rid]["features"]=extract_features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rows_map.values())}),flush=True)

    rows=list(rows_map.values())
    y=np.asarray([r["y"] for r in rows],int);src=np.asarray([r["source_apply"] for r in rows],bool)
    control=account(y,src)
    require(control["corrections"]==108 and control["regressions"]==125 and control["other_k_actions"]==194 and control["global_net"]==-17,"source drift")

    fixed_veto=[]
    for r in rows:
        fixed_veto.append(bool(r["source_apply"] and r.get("features") is not None and hit(r["features"],pattern_thresholds[str(r["fold"])],PATTERN)))
    fixed=group_block(rows,fixed_veto)
    require(fixed["blocked_corr"]==19 and fixed["blocked_reg"]==45 and fixed["veto_gain"]==26,"fixed veto drift")

    fixed_breakdowns={}
    for key in ("player","style","part","variant","piece","track_family"):
        gg=fixed_group_breakdown(rows,key,pattern_thresholds)
        fixed_breakdowns[key]={"groups":gg,"concentration":concentration(gg)}

    transfer={}
    for key in ("player","style","part"):
        gg=transfer_holdout(rows,key)
        transfer[key]={"groups":gg,"concentration":concentration(gg),
                       "all_groups_nonnegative":all(q["veto_gain"]>=0 for q in gg),
                       "positive_groups":sum(q["veto_gain"]>0 for q in gg)}

    # player x style matrix under fixed discovery thresholds
    matrix=[]
    for player in sorted({r["player"] for r in rows}):
        for style in sorted({r["style"] for r in rows}):
            rr=[r for r in rows if r["player"]==player and r["style"]==style]
            if not rr:continue
            vv=[bool(r["source_apply"] and r.get("features") is not None and hit(r["features"],pattern_thresholds[str(r["fold"])],PATTERN)) for r in rr]
            q=group_block(rr,vv);q.update(player=player,style=style);matrix.append(q)

    report={"status":"completed","experiment":"v273_stable_signature_generalization",
            "pattern":list(PATTERN),"source_control":control,"fixed_discovery_thresholds":fixed,
            "outer_fold_3_used":False,"prediction_changes":False,"posthoc_internal_reuse":True,
            "fixed_breakdowns":fixed_breakdowns,"leave_one_group_out_transfer":transfer,
            "player_style_matrix":matrix,
            "interpretation_guardrails":[
              "A positive veto_gain means more K3 regressions than K2 corrections were blocked.",
              "Other-K blocked actions are reported but do not enter Exact-K correction-regression net.",
              "Leave-one-group-out recalibrates only medians from other groups and other folds; directions are frozen.",
              "These internal folds have already been inspected and are not untouched final validation."
            ]}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    lines=["# Stable-signature veto generalization audit","",
           f"Fixed veto: blocked **{fixed['blocked_reg']} regressions / {fixed['blocked_corr']} corrections**, gain **{fixed['veto_gain']:+d}**.","",
           "## Fixed-threshold breakdown by player","",
           "| player | blocked K3 | blocked K2 | other-K | gain | K3 precision |",
           "|---|---:|---:|---:|---:|---:|"]
    for q in fixed_breakdowns["player"]["groups"]:
        pr="n/a" if q["blocked_k23_precision"] is None else f"{q['blocked_k23_precision']:.3f}"
        lines.append(f"| {q['player']} | {q['blocked_reg']} | {q['blocked_corr']} | {q['blocked_other']} | {q['veto_gain']:+d} | {pr} |")
    lines+=["","## Fixed-threshold breakdown by style","",
            "| style | blocked K3 | blocked K2 | gain | K3 precision |","|---|---:|---:|---:|---:|"]
    for q in fixed_breakdowns["style"]["groups"]:
        pr="n/a" if q["blocked_k23_precision"] is None else f"{q['blocked_k23_precision']:.3f}"
        lines.append(f"| {q['style']} | {q['blocked_reg']} | {q['blocked_corr']} | {q['veto_gain']:+d} | {pr} |")
    lines+=["","## Leave-one-player-out + leave-fold-out median transfer","",
            "| player held out | blocked K3 | blocked K2 | gain | precision |","|---|---:|---:|---:|---:|"]
    for q in transfer["player"]["groups"]:
        pr="n/a" if q["blocked_k23_precision"] is None else f"{q['blocked_k23_precision']:.3f}"
        lines.append(f"| {q['player']} | {q['blocked_reg']} | {q['blocked_corr']} | {q['veto_gain']:+d} | {pr} |")
    lines+=["","## Leave-one-style-out + leave-fold-out median transfer","",
            "| style held out | blocked K3 | blocked K2 | gain | precision |","|---|---:|---:|---:|---:|"]
    for q in transfer["style"]["groups"]:
        pr="n/a" if q["blocked_k23_precision"] is None else f"{q['blocked_k23_precision']:.3f}"
        lines.append(f"| {q['style']} | {q['blocked_reg']} | {q['blocked_corr']} | {q['veto_gain']:+d} | {pr} |")
    lines+=["",
            f"Player transfer all non-negative: **{transfer['player']['all_groups_nonnegative']}**.",
            f"Style transfer all non-negative: **{transfer['style']['all_groups_nonnegative']}**.",
            "",
            "Internal/post-hoc audit only; no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
