"""S457 — inspect why all four S454 acoustic K gates abstained.

Strictly read-only report analysis: original GuitarSet S454 nested validation
results + external S453 source root-cause. NEVER use already-scored IDMT
labels to choose thresholds. Diagnostic counts are training/development only.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path

GATES=(0,1,2,3,4)
CUTS=(.45,.55,.65,.75,.85,.92)
def need(x,msg):
    if not x:raise ValueError(msg)
def blockers(row):
    v=row["overall"]
    reason=[]
    if v["fixes"]<5:reason.append("less_than_five_corrections")
    if v["net"]<5:reason.append("net_gain_below_five")
    if v["fixes"]<2*v["regressions"]:reason.append("fixes_less_than_twice_regressions")
    for kk in (2,3,4):
        if row["by_true_K"][str(kk)]["net"]<0:
            reason.append("true_K"+str(kk)+"_regressions")
    return reason

def run(s454,s453,out):
    need(not out.exists(),"S457 immutable experiment diagnosis already exists")
    source=json.loads(s454.read_text())
    ext=json.loads(s453.read_text())
    need(source["status"]=="S454_REAL_ACOUSTIC_1098D_COMP_DISJOINT_K0_K6_CANDIDATE"
         and source["real_source_N"]==59309
         and source["original_K_reference_guitarset_global"]==53529
         and source["original_K_reference_guitarset_poly"]==6054
         and source["no_promotion"] and source["no_external_IDMT_labels_or_weights"]
         and ext["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and ext["matched_true_poly_events"]==21
         and ext["original_stage_ledger"]["S80"]["true_poly"]["exact"]==7,
         "S457 expects exactly S454 trained source and verified S453 provenance")
    failures=collections.Counter()
    perK={}
    fold_reasons=[]
    gate_count=0
    for fold in source["all_model_folds"]:
        gate_count+=len(fold["gates"])
        for k in GATES:
            rec=fold["inner_by_predicted_K"][str(k)]
            options=rec["candidate_thresholds"]
            need(len(options)==len(CUTS)
                 and [round(c["threshold"],2) for c in options]==list(CUTS),
                 "S457 source K thresholds or score grid changed")
            for v in options:
                why=blockers(v)
                if not why:
                    need(rec["safe_gate_selected"] is not None,
                         "S457 found a safe gate never selected by S454")
                else:
                    for reason in why:failures[reason]+=1
                fold_reasons.append(dict(fold=int(fold["fold"]),
                     source_predicted_K=int(k),threshold=v["threshold"],
                     validation_examples=rec["n"],corrections=v["overall"]["fixes"],
                     regressions=v["overall"]["regressions"],
                     net=v["overall"]["net"],failed_criteria=why,
                     trueK2to4={str(t):v["by_true_K"][str(t)] for t in (2,3,4)}))
    for k in GATES:
        subset=[r for r in fold_reasons if r["source_predicted_K"]==k]
        summed=[]
        for cut in CUTS:
            vals=[r for r in subset if r["threshold"]==cut]
            summed.append(dict(threshold=cut,composition_fold_validations=len(vals),
                corrections=sum(v["corrections"] for v in vals),
                regressions=sum(v["regressions"] for v in vals),
                net=sum(v["net"] for v in vals),
                unsafe_fold_thresholds=sum(bool(v["failed_criteria"]) for v in vals)))
        perK[str(k)]=dict(candidate_policy_tried=len(subset),
            best_unsafe_counterfactual=max(summed,key=lambda s:s["net"]),
            by_fixed_threshold=summed)
    need(len(fold_reasons)==4*len(GATES)*len(CUTS)
         and gate_count==0
         and source["paired_validation"]["all"]["net"]==0
         and source["paired_validation"]["true_poly"]["net"]==0,
         "S457 S454 was not complete conservative abstention; cannot assert failure")
    external=dict(
         K0_toK6_true_poly_original_correct=ext["original_stage_ledger"]["S80"]["true_poly"]["exact"],
         real_true_poly=21,
         original_wrong_K0_on_true_poly=ext["original_stage_ledger"]["S80"]["wrong_K0_true_poly"],
         original_K_under=ext["original_stage_ledger"]["S80"]["true_poly"]["K_under"],
         original_K_over=ext["original_stage_ledger"]["S80"]["true_poly"]["K_over"])
    out.mkdir(parents=True)
    result=dict(status="S457_S454_INNER_K_GATE_ABSTENTION_TRAINING_ONLY_DIAGNOSTIC",
       S454_trained_fold_count=4,all_120_threshold_checks_rejected=True,
       original_reference_unchanged=True,
       no_model_retraining=True,no_threshold_retuning=True,
       source_GuitarSet_historical_training_overlap=True,
       external_IDMT_XML_was_previously_opened=True,
       external_info_used_for_prediction_model_or_thresholds=False,
       not_independent_validation=True,
       no_promotion=True,
       failure_causes_counts=dict(failures),
       per_original_predicted_K=perK,
       external_K_error_context=external)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"failed_gates.jsonl").write_text(
         "".join(json.dumps(r)+"\n" for r in fold_reasons))
    print("S457 GATE ROOT-CAUSE",json.dumps(dict(
        total_threshold_checks=len(fold_reasons),
        all_rejected=gate_count==0,
        failures=dict(failures),
        by_predicted_K_best={k:v["best_unsafe_counterfactual"]
            for k,v in perK.items()},
        external=external)),flush=True)

def selftest():
    good=dict(overall=dict(fixes=6,regressions=2,net=4),
              by_true_K={str(k):dict(net=0) for k in range(7)})
    assert blockers(good)==["net_gain_below_five"]
    good["overall"]["net"]=5
    assert blockers(good)==[]
    good["by_true_K"]["3"]["net"]=-1
    assert blockers(good)==["true_K3_regressions"]
    print("S457 SELFTEST PASS explicit gate regression reasons")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for field in ("s454","s453","out"):p.add_argument("--"+field,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for field in ("s454","s453","out"):
            need(getattr(a,field)is not None,"S457 missing --"+field)
        run(a.s454,a.s453,a.out)
