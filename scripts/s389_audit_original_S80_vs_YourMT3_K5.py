"""S389. Strictly descriptive paired S80-vs-YourMT3 K5 proposal coverage audit.

NOTE: S272 4-fold fold-neutral GuitarSet scores three training-overlap models
per example. This is NOT held-piece OOF, an independent model comparison, a
trained K5 rescue, or permission to promote anything.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def ensure(test, msg):
    if not test:
        raise ValueError(msg)

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def histogram(x, mask):
    return {f"K{j}": int(np.count_nonzero(mask & (x == j))) for j in range(7)}

def run(s80file, yourfile, oldfile, out):
    ensure(not out.exists(), "S389 evidence must be immutable")
    a = load(s80file)
    b = load(yourfile)
    c = load(oldfile)
    ia = np.argsort(a["row_id"].astype(np.int64))
    ib = np.argsort(b["global_index"])
    ic = np.argsort(c["global_index"])
    ids = a["row_id"][ia].astype(np.int64)
    ensure(len(ids) == 59309 and np.array_equal(ids,b["global_index"][ib])
           and np.array_equal(ids,c["global_index"][ic])
           and len(set(ids)) == 59309, "source IDs not identical")
    y = b["k"][ib]
    ensure(np.array_equal(y,c["true_K"][ic]), "labels not identical")
    base = b["baseline"][ib]
    ensure(np.array_equal(base,a["incumbent_freeze_K"][ia]),
           "frozen baseline differs")
    fold = b["fold"][ib]
    ensure(set(map(int,np.unique(fold))) == {0,1,2,4}, "heldfold inventory differs")
    s80 = a["s80_K"][ia]
    mt = b["predicted"][ib]
    mask = y == 5
    ensure(int(mask.sum()) == 355 and int((y>=2).sum()) == 7385
           and int((s80==y).sum()) == 53327
           and int((mt==y).sum()) == 51328
           and int((mask & (s80==5)).sum()) == 99
           and int((mask & (mt==5)).sum()) == 189,
           "frozen K5 or global score drift")
    stages = {
        "historical_freeze_audit_only": base,
        "old_S80_OOF_audit_only": c["s80_K"][ic],
        "S72": a["s72_K"][ia],
        "S79": a["s79_K"][ia],
        "PR16_policy22": a["original_PR16_policy22_K"][ia],
        "S80_reconstructed": s80,
    }
    stage_stats = {
        key: {"correct_true_K5":int((mask & (p==5)).sum()),
              "predicted_K5_total":int((p==5).sum()),
              "on_true_K5_predicted_distribution":histogram(p,mask)}
        for key,p in [*stages.items(),("YourMT3+_full_decoder",mt)]
    }
    oracle = np.logical_or.reduce([v==5 for v in stages.values()])
    deploy_only = np.logical_or.reduce([v==5 for k,v in stages.items()
                                        if k!="old_S80_OOF_audit_only"])
    matched = dict(
        both_correct=int((mask & (s80==5) & (mt==5)).sum()),
        S80_only=int((mask & (s80==5) & (mt!=5)).sum()),
        YourMT3_only=int((mask & (s80!=5) & (mt==5)).sum()),
        both_wrong=int((mask & (s80!=5) & (mt!=5)).sum()),
        S80_K5_pred_K4=int((mask & (s80==4)).sum()),
        YourMT3_only_but_S80_pred_K4=int((mask&(mt==5)&(s80==4)).sum()))
    ensure(sum(matched[k] for k in ("both_correct","S80_only","YourMT3_only","both_wrong"))==355,
           "pairwise ledger drift")
    report={
        "status":"S389_K5_SOURCE_COVERAGE_READ_ONLY_NO_PROMOTION",
        "original_wav_events":59309,"true_K5_events":355,
        "source_runs":{"S80":38314291222,"YourMT3+":37605163312,
                       "old_S80":38187186049},
        "paired_true_K5":matched,
        "source_stage_inventory":stage_stats,
        "oracle_K5_candidate_coverage_with_historical_OOF":int((mask&oracle).sum()),
        "oracle_K5_candidate_coverage_without_historical_OOF":int((mask&deploy_only).sum()),
        "K5_correct_of_YourMT3_uncovered_by_any_stage":int((mask&(mt==5)&~oracle).sum()),
        "original_K5_s80_vs_s72_changes":int((mask&(s80!=stages["S72"])).sum()),
        "original_K5_final_tail_actions":{
            k:int((mask&a[k][ia]).sum())
            for k in ("action_K0","action_K1","action_K2")},
        "by_fold_true_K5":{
            str(f):{"n":int((mask&(fold==f)).sum()),
                    "S80_correct":int((mask&(fold==f)&(s80==5)).sum()),
                    "YourMT3_correct":int((mask&(fold==f)&(mt==5)).sum())}
            for f in (0,1,2,4)},
        "training_overlap_S80":True,
        "YourMT3_GuitarSet_training_overlap_unexcluded":True,
        "independent_or_equal_latency_test":False,
        "trained_new_model":False,"promotion":False,
        "next_step":"Build a new upstream K5 proposal producer, trained on disjoint training material only, then score against frozen S80 on a composition-disjoint K5-rich holdout."
    }
    ensure(report["oracle_K5_candidate_coverage_with_historical_OOF"]==112
           and report["oracle_K5_candidate_coverage_without_historical_OOF"]==109
           and matched["S80_K5_pred_K4"]==224
           and matched["YourMT3_only_but_S80_pred_K4"]==124
           and report["original_K5_s80_vs_s72_changes"]==0,
           "source candidate K5 accounting changed")
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print("S389 PASS READ-ONLY: NOT INDEPENDENT, NO REFERENCE CHANGE")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("s80","your","old","out"):
        parser.add_argument("--"+flag,required=True,type=Path)
    args=parser.parse_args()
    run(args.s80,args.your,args.old,args.out)
