"""S273: measure GuitarSet new S80 SAME events against old 43.8727% S80.
Warning: new fourfold deploy puts three fold-specific trained models on
rows that may have been used for fitting. Score is in-sample descriptive,
NEVER held-piece OOF or independent promotion evidence.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def require(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def measure(true,old,new,mask):
    n=int(np.count_nonzero(mask))
    require(n>0,'empty scoring cohort')
    o=old[mask]==true[mask];v=new[mask]==true[mask]
    corrections=int(np.count_nonzero(~o&v))
    regressions=int(np.count_nonzero(o&~v))
    oldcorrect=int(np.count_nonzero(o));newcorrect=int(np.count_nonzero(v))
    require(newcorrect-oldcorrect==corrections-regressions,'paired score change ledger')
    return dict(n=n,old_S80_historical_correct=oldcorrect,
        new_S80_foldneutral_correct=newcorrect,
        old_S80_historical_exactK=float(oldcorrect/n),
        new_S80_foldneutral_exactK=float(newcorrect/n),
        difference_pct_points=100*(newcorrect-oldcorrect)/n,
        corrections=corrections,regressions=regressions,
        score_type='historical training-overlap diagnostic, NOT independent and NOT 4fold OOF')
def run(source,candidate,old,out):
    require(not out.exists(),'do not overwrite finalized scoring report')
    with np.load(source,allow_pickle=False) as z:
        require(set(z.files)=={'row_id','true_K','original_fold','original_freeze_K'},
            'reference true K only in final scoring boundary')
        src={k:z[k].copy() for k in z.files}
    with np.load(candidate,allow_pickle=False) as z:
        expected={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
                  'original_PR16_policy22_K','K1_by_fold_threshold_action',
                  'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
        require(set(z.files)==expected,'S272 score input is original frozen exact full new S80 output only')
        new={k:z[k].copy() for k in z.files}
    with np.load(old,allow_pickle=False) as z:
        require({'global_index','true_K','s80_K','baseline_K'}.issubset(z.files),
              'original S80 43.87 comparator must be historical unretrained fixture')
        ids=z['global_index'].copy();t=z['true_K'].copy();h=z['s80_K'].copy();b=z['baseline_K'].copy()
    order=np.argsort(ids)
    ids,t,h,b=ids[order],t[order],h[order],b[order]
    n=len(t)
    true=src['true_K'].astype(np.int16)
    cand=new['s80_K'].astype(np.int16)
    parent=src['original_freeze_K'].astype(np.int16)
    require(n==59309 and int(np.count_nonzero(true>=2))==7385,
            'immutable GuitarSet original 59309 and 7385 true poly counts')
    require(np.array_equal(src['row_id'].astype(str),ids.astype(str)) and
            np.array_equal(new['row_id'].astype(str),ids.astype(str)) and
            np.array_equal(t,true) and np.array_equal(parent,b) and
            np.array_equal(parent,new['incumbent_freeze_K']),
            'old S80 vs new full S80 source misaligned event IDs, reference, original held-out labels')
    require(np.isin(cand,range(7)).all() and
            int(np.count_nonzero((h==true)&(true>=2)))==3240,
            'original historical S80 poly 3240/7385 unrevised')
    global_stats=measure(true,h,cand,np.ones(n,bool))
    poly=measure(true,h,cand,true>=2)
    byclass={f'K{k}':measure(true,h,cand,true==k) for k in range(7)
              if np.any(true==k)}
    oldref=int(np.count_nonzero(parent==true))
    new_vs_official=int(np.count_nonzero(cand==true))
    changed=parent!=cand
    modifications=dict(changes_from_official=int(changed.sum()),
        fixes_from_official=int(np.count_nonzero(changed&(parent!=true)&(cand==true))),
        regressions_from_official=int(np.count_nonzero(changed&(parent==true)&(cand!=true))),
        changed_both_wrong=int(np.count_nonzero(changed&(parent!=true)&(cand!=true))),
        freeze_local_combo_correct=oldref,new_candidate_correct=new_vs_official)
    report=dict(
        status='S273_MEASURED_NEW_FOLDNEUTRAL_S80_ON_ORIGINAL_GUITARSET_WITH_TRAIN_CONTAMINATION',
        native_events=n,true_polyphonic_events=int((true>=2).sum()),
        historical_original_S80_source_run=38187186049,
        candidate_new_S80_GuitarSet_run=None,
        source_true_K_SHA256=sha(source),candidate_new_S80_SHA256=sha(candidate),
        historical_original_S80_SHA256=sha(old),
        exactK_global_K0to6=global_stats,
        exactK_polyphonic_trueK2to6=poly,
        exactK_by_true_K=byclass,
        candidate_vs_official_freeze=modifications,
        new_rule_is_real_fourfold_original_weight_counterfactual=True,
        original_training_data_seen_by_three_nonheldfold_models=True,
        new_candidate_score_not_pure_outof_fold=True,
        original_historical_S80_OOF_comparator_same_members=True,
        direct_generalization_equivalence_to_IDMT_not_claimed=True,
        original_official_freeze_local_combo_preserved=True,
        independent_promotion_gate_satisfied=False,S80_promoted=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('source','candidate','old','out'):a.add_argument('--'+k,required=True,type=Path)
    q=a.parse_args();run(q.source,q.candidate,q.old,q.out)
