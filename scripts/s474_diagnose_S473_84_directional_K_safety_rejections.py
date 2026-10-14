"""S474 audit 84 original GuitarSet S473 inner-validation thresholds.

S473 tried 4 folds x 3 source predicted K (2,3,4) x 7 fixed confidence
cuts and safely abstained. This reads archived inner-partition evidence
ONLY, quantifies correction/regression reasons by source K and true K,
without selecting any threshold or changing original S80.
"""
from __future__ import annotations
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np
CUTS=(.30,.40,.50,.60,.70,.80,.90)
def need(x,s):
    if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def failures(row):
    v=[]
    if row["fixes"]<5:v.append("few_fixes")
    if row["net"]<3:v.append("net_below_three")
    if row["fixes"]<2*row["regressions"]:v.append("fixes_less_than_twice_regressions")
    for k in (1,2,3,4):
        if row["each_trueK"][str(k)]["net"]<0:
            v.append("true_K"+str(k)+"_regression")
    return v
def run(s473,out):
    need(not out.exists(),"S474 immutable S473 diagnosis already exists")
    source=json.loads(s473.read_text())
    need(source["status"]=="S473_ACTUAL_V88_SONGHELD_DIRECTIONAL_K2_K3_K4_SPECIALISTS"
         and source["original_source_events"]==59309
         and source["count_K_changes"]==0
         and source["nonofficial_GuitarSet_pair_poly"]["net"]==0
         and source["no_promotion"],
         "S474 original S473 source report needs complete zero-intervention proof")
    reasons=collections.Counter()
    byK={}
    checked=0
    totalfix=0
    for a in source["per_source_predicted_K"]:
        k=int(a["original_predicted_K"])
        rows=[]
        need(len(a["fold_specs"])==4,"S474 four native real source fold experts required")
        for fold in a["fold_specs"]:
            need(fold["selected_cut"] is None
                 and len(fold["all_threshold_evidence"])==len(CUTS),
                 "S474 S473 fold chosen a K specialist; cannot claim conservative total abstention")
            options=fold["all_threshold_evidence"]
            for c,rec in zip(CUTS,options):
                need(round(rec["cut"],2)==c,"S474 immutable threshold grid changed")
                why=failures(rec)
                need(bool(why) and not rec["safe"],
                     "S474 found a safe candidate that was incorrectly rejected")
                reasons.update(why);checked+=1
                rows.append(dict(source_K=k,
                    fold=int(fold["outer_fold"]),cut=c,
                    corrections=int(rec["fixes"]),
                    regressions=int(rec["regressions"]),
                    net=int(rec["net"]),
                    failure_conditions=why,
                    by_actual_K={str(j):dict(rec["each_trueK"][str(j)])
                                 for j in range(7)}))
        summary=[]
        for c in CUTS:
            rr=[x for x in rows if x["cut"]==c]
            cor=sum(x["corrections"] for x in rr)
            reg=sum(x["regressions"] for x in rr)
            totalfix+=cor
            summary.append(dict(threshold=c,corrections=cor,regressions=reg,
                 net=cor-reg,
                 failed_folds=sum(bool(x["failure_conditions"]) for x in rr),
                 regression_true_K={
                    str(j):sum(x["by_actual_K"][str(j)]["reg"] for x in rr)
                    for j in (1,2,3,4)},
                 corrections_true_K={
                    str(j):sum(x["by_actual_K"][str(j)]["fix"] for x in rr)
                    for j in (1,2,3,4)}))
        byK[str(k)]=dict(source_count=a["all_source_events"],
            aggregate_by_threshold=summary,
            most_positive_unpromoted_threshold=max(summary,key=lambda q:q["net"]),
            full_original_train_partitions=rows)
    need(checked==4*3*len(CUTS),"S474 expected exactly 84 separately calibrated K source decisions")
    out.mkdir(parents=True)
    result=dict(status="S474_READONLY_S473_84_K2K3K4_INNER_THRESHOLD_REJECTION_AUDIT",
       source_V88_guitarset_events=59309,
       independent_fourfold_songheld_inner_calibration_per_source_K=True,
       thresholds_tested=checked,
       all_gate_options_unsafe=True,
       failure_categories=dict(reasons),
       all_trial_corrections_before_gate_aggregated=totalfix,
       by_original_predicted_K=byK,
       no_retraining_no_external_IDMT_or_MAESTRO_label_access=True,
       no_official_source_K_or_live_promotion=True,
       not_independent_external_score=True,
       no_promotion=True,
       S473_report_SHA256=sha(s473))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S474 DIAGNOSIS",json.dumps(dict(
       total_thresholds=checked,
       failure_reasons=dict(reasons),
       best_candidate_per_predicted_K={
            k:v["most_positive_unpromoted_threshold"] for k,v in byK.items()},
       no_promotion=True)),flush=True)
def selftest():
    sample=dict(fixes=8,regressions=1,net=7,
                each_trueK={str(j):dict(net=0) for j in range(7)})
    need(failures(sample)==[],"S474 safety gate incorrectly blocks successful no-regression trial")
    sample["each_trueK"]["3"]["net"]=-1
    need(failures(sample)==["true_K3_regression"],
         "S474 classify regression by actual musical poly K")
    print("S474 SELFTEST PASS risky K-specific regression causal labels")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--s473",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.s473 and a.out,"S474 missing S473 and out");run(a.s473,a.out)
