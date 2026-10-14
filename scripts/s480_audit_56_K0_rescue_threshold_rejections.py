"""S480 forensic 56 candidate K0 rescue gates from S478 frozen GuitarSet.
Four outer fold nested-validation ×14 thresholds. Why do they fail?
Quantify fixes / regression true K0 and missing true poly K0 recovery.
NO refit, no alternative thresholds and no external labels.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
def need(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(source,out):
    need(not out.exists(),"S480 immutable audit exists")
    a=json.loads(source.read_text())
    need(a["status"]=="S478_GUITARSET_ONLY_V88_K0_SOURCE_RESIDUAL_4FOLD"
         and len(a["four_GuitarSet_train_only_outerfolds"])==4
         and a["total_directional_K0_proposals"]==0 and a["no_promotion"],
         "S480 must diagnose actual completed no-deployment S478 from GuitarSet")
    why=collections.Counter();per=[]
    for f in a["four_GuitarSet_train_only_outerfolds"]:
        need(f["selected_cut"] is None
             and len(f["inner_all_thresholds"])==14,
             "S480 source K0 gates all evaluated and all abstained prerequisite")
        for trial in f["inner_all_thresholds"]:
            reasons=[]
            if trial["fixes"]<5:reasons.append("fewer_than_5_true_fixes")
            if trial["net"]<5:reasons.append("net_gain_less_than_5")
            if trial["fixes"]<3*max(1,trial["regressions"]):
                reasons.append("not_3fixes_per_regression")
            if trial["regressions"]>2:reasons.append("more_than_2_regressions")
            if trial["true_K0_lost"]>2:reasons.append("more_than_2_real_K0_discarded")
            need(reasons and not trial["valid"],"S480 falsely rejected a safe proposal")
            why.update(reasons)
            per.append(dict(fold=int(f["fold"]),cut=float(trial["cut"]),
              fixes=int(trial["fixes"]),regressions=int(trial["regressions"]),
              net=int(trial["net"]),trueK0_lost=int(trial["true_K0_lost"]),
              K1_rescued=int(trial["source_K0_wrong_rescued_K1"]),
              K2_rescued=int(trial["source_K0_wrong_rescued_K2"]),
              rejection_reasons=reasons))
    bycut=[]
    for c in sorted({x["cut"] for x in per}):
        groups=[x for x in per if x["cut"]==c]
        cor=sum(x["fixes"] for x in groups)
        reg=sum(x["regressions"] for x in groups)
        bycut.append(dict(threshold=c,corrections=cor,regressions=reg,
               net=cor-reg,corrected_trueK1=sum(x["K1_rescued"] for x in groups),
               corrected_trueK2=sum(x["K2_rescued"] for x in groups)))
    best=max(bycut,key=lambda q:(q["net"],-q["regressions"]))
    out.mkdir(parents=True)
    result=dict(status="S480_S478_ORIGINAL_V88_K0_RESCUE_INNER_THRESHOLD_FAILURE_DIAGNOSIS",
        examined_fold_thresholds=len(per),
        rejection_reasons_overlapping=dict(why),
        all_thresholds_rejected=True,
        full_guitarset_nested_validation_only=True,
        per_fold_threshold=per,
        aggregate_by_fixed_threshold=bycut,
        best_but_NOT_selected=best,
        original_GuitarSet_K0_true_poly_missing=931,
        source_S478_SHA256=sha(source),
        no_models_trained_or_threshold_optimized=True,
        no_MAESTRO_or_IDMT_labels_consulted=True,
        no_promotion=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S480 WHY K0 FAILS",json.dumps(dict(
        all=56,reasons=dict(why),best_unselected=best,
        by_threshold=bycut)),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s478",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    z=p.parse_args();run(z.s478,z.out)
