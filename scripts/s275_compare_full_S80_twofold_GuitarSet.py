"""S275 paired DEV diagnostic: original S272 full S80 vs S274-gated full S80.

The exact same 59,309 GuitarSet native events are scored only after the new
counterfactual source-only predictions were frozen in an earlier CI job.
These predictions are NOT out-of-fold (3/4 source folds saw training rows).
Do not use either score to replace official freeze_local_combo.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
PRED={"row_id","incumbent_freeze_K","s72_K","s79_K","s80_K",
      "original_PR16_policy22_K","K1_by_fold_threshold_action",
      "K1_by_fold_threshold_scores","action_K1","action_K0","action_K2"}
TRUTH={"row_id","true_K","original_fold","original_freeze_K"}
def need(v,msg):
    if not v: raise ValueError(msg)
def load(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==keys,"source schema mismatch "+str(p))
        return {k:z[k].copy() for k in z.files}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ledger(y,original,new,mask):
    n=int(mask.sum())
    need(n>0,"empty class")
    y=y[mask];b=original[mask];c=new[mask]
    baseok=b==y;candok=c==y
    fix=int(np.count_nonzero(~baseok&candok))
    reg=int(np.count_nonzero(baseok&~candok))
    return dict(n=n,original_S272_correct=int(baseok.sum()),
        S275_twofold_correct=int(candok.sum()),
        original_S272_ExactK=float(baseok.mean()),
        S275_twofold_ExactK=float(candok.mean()),
        corrections=fix,regressions=reg,net_change=fix-reg,
        changes=int(np.count_nonzero(b!=c)),
        changed_both_wrong=int(np.count_nonzero((b!=c)&~baseok&~candok)),
        metric_type="training-overlap DEV diagnostic, NOT independent/OoF")
def run(original,candidate,truth,out):
    need(not out.exists(),"immutable postfrozen audit")
    b=load(original,PRED);c=load(candidate,PRED);t=load(truth,TRUTH)
    ids=b['row_id'];n=len(ids);y=t['true_K']
    need(n==59309 and np.array_equal(ids,c['row_id'])
         and np.array_equal(ids,t['row_id'])
         and np.array_equal(b['incumbent_freeze_K'],c['incumbent_freeze_K'])
         and np.array_equal(b['incumbent_freeze_K'],t['original_freeze_K'])
         and int((y>=2).sum())==7385,
         "immutable GuitarSet paired native events, truth and baseline")
    old=b['s80_K'];new=c['s80_K']
    need(int(np.count_nonzero((old==y)&(y>=2)))==5852,
         "original S273 S80 full reconstructed measured baseline drift")
    report=dict(
        status="S275_FULL_S80_FROZEN_COUNTERFACTUAL_GUITARSET_DEV_ONLY",
        source_original_S272_sha256=sha(original),
        source_S275_twofold_sha256=sha(candidate),
        labels_opened_only_in_scoring_job=True,
        full_S69_S70_S71_S72_S79_S80_all_replayed=True,
        global_K0to6=ledger(y,old,new,np.ones(n,bool)),
        true_poly_K2to6=ledger(y,old,new,y>=2),
        per_true_K={f"K{k}":ledger(y,old,new,y==k) for k in range(7)
                    if np.any(y==k)},
        original_training_overlap_three_nonheldfold_models=True,
        song_disjoint_independent_validation=False,
        no_IDMT_XML_used=True,
        official_S80_promoted=False,
        official_reference="freeze_local_combo")
    out.mkdir(parents=True)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','global_K0to6',
        'true_poly_K2to6','original_training_overlap_three_nonheldfold_models',
        'official_S80_promoted')},indent=2),flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser()
    for key in ('original','candidate','truth','out'):
        a.add_argument('--'+key,type=Path,required=True)
    z=a.parse_args();run(z.original,z.candidate,z.truth,z.out)
