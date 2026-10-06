"""Full OOF accounting for three predeclared five-signal veto patterns.

Patterns and fold-specific median thresholds come unchanged from completed run
37530685909. Features are reconstructed from raw PCM + the residual harmonic
decomposition for every original OOF source action (including other-K rows).

This is a post-hoc accounting/transport check on already inspected internal folds,
not fresh validation. No fold 3 and no model promotion.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_oof_source_onset_contrast_guard import build_oof,account
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_internal_b_low_harmonic_strata import transition_spectrum,f0_pool,template,fit_best
from scripts.audit_v273_attack_novelty import raw_powers,component_metrics
from scripts.v273_residual_audit import FOLDS,require

EPS=1e-12
PATTERNS={
 "trio_best_net":("pre_high","amp_range_low","amp_cv_low"),
 "four_best_precision":("onset_low","pre_high","amp_range_low","amp_cv_low"),
 "all_five":("onset_low","pre_high","amp_range_low","amp3_ratio_high","amp_cv_low"),
}
ALIAS_TO_FEATURE={
 "onset_low":"nov_onset_contrast_norm_median",
 "pre_high":"nov_pre_norm_median",
 "amp_range_low":"amp_range",
 "amp3_ratio_high":"amp3_over_amp1",
 "amp_cv_low":"amp_cv",
}

def extract_features(samples,start):
    freq,x=transition_spectrum(samples,start)
    pool=f0_pool(freq,x)
    if len(pool)<3:return None
    D=np.column_stack([template(freq,f) for f in pool])
    trip,_,_,_=fit_best(D,x,3)
    _,combo,amps=trip
    f0=np.asarray([pool[i] for i in combo],float);amps=np.asarray(amps,float)
    if len(f0)!=3 or len(amps)!=3:return None
    sf=np.sort(amps)[::-1]
    rf,powers,_=raw_powers(samples,start)
    nov=[component_metrics(rf,powers,float(q)) for q in f0]
    return {
      "nov_onset_contrast_norm_median":float(np.median([q["onset_contrast_norm"] for q in nov])),
      "nov_pre_norm_median":float(np.median([q["pre_norm"] for q in nov])),
      "amp_range":float(np.ptp(amps)),
      "amp3_over_amp1":float(sf[2]/(sf[0]+EPS)),
      "amp_cv":float(amps.std()/(abs(amps.mean())+EPS)),
    }

def hit(features,thresholds,pattern):
    for alias in pattern:
        spec=thresholds[alias];v=features[ALIAS_TO_FEATURE[alias]]
        if spec["direction"]=="low":
            if not v<=float(spec["threshold"]):return False
        else:
            if not v>=float(spec["threshold"]):return False
    return True

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","pattern-report","pattern-rows","output"):p.add_argument("--"+n.replace("_","-"),dest=n.replace("-","_"),type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"refusing overwrite")
    action_inputs(a.exports)
    rows=build_oof(a.exports)
    pattern=json.loads(a.pattern_report.read_text())
    require(pattern["experiment"]=="v273_stable_signature_intersections","pattern report drift")
    old_rows={int(json.loads(s)["row_id"]):json.loads(s) for s in a.pattern_rows.read_text().splitlines()}
    require(len(old_rows)==233,"pattern row count drift")

    wanted={r["member"] for r in rows.values()}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for rid,r in rows.items():
        if r["source_apply"]:by[r["member"]].append((rid,r["start"]))
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:
            rows[rid]["features"]=extract_features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rows.values())}),flush=True)

    ordered=[r for r in rows.values()]
    y=np.asarray([r["y"] for r in ordered],int);src=np.asarray([r["source_apply"] for r in ordered],bool)
    control=account(y,src)
    require(control["corrections"]==108 and control["regressions"]==125 and control["other_k_actions"]==194 and control["global_net"]==-17,"source accounting drift")

    # Transport agreement on the same 233 K2/K3 action rows used to discover the patterns.
    agreement={}
    for alias in ALIAS_TO_FEATURE:
        agree=0;n=0
        for r in ordered:
            if not r["source_apply"] or r["y"] not in (2,3) or r["row_id"] not in old_rows or r.get("features") is None:continue
            thr=pattern["thresholds"][str(r["fold"])][alias]
            now=hit(r["features"],{alias:thr},(alias,))
            old=bool(old_rows[r["row_id"]]["signature_bits"][alias])
            agree+=int(now==old);n+=1
        agreement[alias]={"n":n,"agree":agree,"rate":None if n==0 else agree/n}

    reports={}
    for name,pat in PATTERNS.items():
        veto=np.zeros(len(ordered),bool);missing=0
        per=[]
        for i,r in enumerate(ordered):
            if not r["source_apply"]:continue
            if r.get("features") is None:
                missing+=1;continue
            veto[i]=hit(r["features"],pattern["thresholds"][str(r["fold"])],pat)
        kept=src&~veto
        total=account(y,kept)
        blocked={"rows":int(veto.sum()),"corrections":int(np.sum(veto&(y==2))),"regressions":int(np.sum(veto&(y==3))),
                 "other_k_actions":int(np.sum(veto&~np.isin(y,(2,3))))}
        for f in FOLDS:
            m=np.asarray([r["fold"]==f for r in ordered],bool)
            q=account(y[m],kept[m])
            b={"blocked":int(veto[m].sum()),"blocked_corr":int(np.sum(veto[m]&(y[m]==2))),
               "blocked_reg":int(np.sum(veto[m]&(y[m]==3))),"blocked_other":int(np.sum(veto[m]&~np.isin(y[m],(2,3))))}
            per.append({"fold":int(f),"kept":q,**b})
        reports[name]={"pattern":list(pat),"feature_missing_source_actions":missing,
                       "blocked":blocked,"kept_action_account":total,
                       "net_improvement_vs_source":total["global_net"]-control["global_net"],"per_fold":per}

    rep={"status":"completed","experiment":"v273_full_oof_stable_signature_veto",
         "source_control":control,"outer_fold_3_used":False,"prediction_changes":False,
         "posthoc_internal_reuse":True,"automatic_promotion":False,
         "reconstruction_agreement_with_discovery_run":agreement,"patterns":reports}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Full OOF stable-signature veto accounting","",
           f"Source: **{control['corrections']} corrections / {control['regressions']} regressions / {control['other_k_actions']} other-K**, net **{control['global_net']:+d}**.","",
           "This is post-hoc internal accounting, not fresh validation.","",
           "| pattern | blocked corr | blocked reg | blocked other | kept corr | kept reg | kept other | new net | delta |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,q in reports.items():
        b=q["blocked"];k=q["kept_action_account"]
        lines.append(f"| {name} | {b['corrections']} | {b['regressions']} | {b['other_k_actions']} | {k['corrections']} | {k['regressions']} | {k['other_k_actions']} | {k['global_net']:+d} | {q['net_improvement_vs_source']:+d} |")
    lines+=["","## Reconstruction agreement with discovery features","","| signal | agreement |","|---|---:|"]
    for alias,q in agreement.items():lines.append(f"| {alias} | {q['agree']}/{q['n']} = {q['rate']:.3f} |")
    lines+=["","Fold 3 excluded. No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
