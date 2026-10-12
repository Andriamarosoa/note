"""S228 independent final S80 deployment correctness and promotion-gate audit.

Independently reproduce source-only S227 final K0 and K2 formula with raw
12 original PR16 neural distributions, 36 original PR16 K decisions and
193 original S46 K decisions, without importing the S147 tail source.
Verify all 57 predicted K, three action masks, original K1 fold quorum,
baseline differences and no true labels / promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
BAD={'true_K','truth','label','labels','target','midi','correct','regression','y'}
def must(v,msg):
    if not v:raise ValueError(msg)
def load(path,fields):
    with np.load(path,allow_pickle=False) as z:
        must(set(z.files)==set(fields) and not BAD.intersection(z.files),
             'strict independent source only '+str(path))
        return {k:z[k].copy() for k in z.files}
def run(s227,s219,s85,out):
    must(not out.exists(),'immutable S228 audit output')
    final=load(s227,{'row_id','incumbent_freeze_K','s72_K','s79_K',
         's80_K','original_PR16_policy22_K','K1_by_fold_threshold_action',
         'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'})
    pr=load(s85,{'member','start_samples','end_samples',
         'incumbent_freeze_K','pr16_policy_K','pr16_vote_count',
         'pr16_policy22_K','pr16_network_probabilities'})
    s46=load(s219,{'row_id','variant_ids','s18_consensus_K',
         's46_policy_K','per_fold_proposal_K','per_fold_proposal_strength',
         'proposal_rejected_due_different_S18_parent',
         'conflicting_active_fold_proposals','active_fold_count'})
    ids=list(map(str,final['row_id']));n=len(ids)
    must(n==57 and np.array_equal(final['row_id'],s46['row_id']),
         'independent S227 and S219 original event order')
    match={f'{v}:{int(t)}':i for i,(v,t) in enumerate(zip(pr['member'],pr['start_samples']))}
    must(all(i in match for i in ids),'no original PR16 event matching')
    ix=[match[i] for i in ids]
    policy=pr['pr16_policy_K'][ix]
    prob=pr['pr16_network_probabilities'][ix]
    s46_votes=s46['s46_policy_K']
    must(policy.shape==(57,36) and prob.shape==(57,12,7) and
         s46_votes.shape==(57,193),'original raw source vote count')
    base=final['s72_K'];k1=final['s79_K']
    pp22=pr['pr16_policy22_K'][ix]
    must(np.array_equal(pp22,final['original_PR16_policy22_K']),
         'historic PR16 K1 candidate mismatch')
    orig_k1=(base==1)&(pp22>base)
    quorum=(final['K1_by_fold_threshold_action'].sum(axis=0)>=3)&orig_k1
    must(np.array_equal(quorum,final['action_K1']) and
         np.array_equal(np.where(quorum,pp22,base),k1),
         'original four fold K1 source model quorum mismatch')
    meanp=prob.mean(axis=1)
    counts_pr=np.zeros((n,7),np.float64)
    counts_s46=np.zeros((n,7),np.float64)
    for k in range(7):
        counts_pr[:,k]=np.count_nonzero(policy==k,axis=1)/36.
        counts_s46[:,k]=np.count_nonzero(s46_votes==k,axis=1)/193.
    fused=.7*meanp+.2*counts_pr+.1*counts_s46
    winner0=meanp.argmax(axis=1)
    winner2=counts_s46.argmax(axis=1)
    zero=(base==0)&(winner0>=2)
    two=(base==2)&(winner2!=base)&(
        fused[np.arange(n),winner2]>=fused[np.arange(n),base]+.4)
    must(not np.any(quorum&(zero|two)),'K1 conflict with K0/K2')
    recomputed=k1.copy()
    recomputed[zero]=winner0[zero]
    recomputed[two]=winner2[two]
    checks=dict(action_K0_exact=np.array_equal(zero,final['action_K0']),
       action_K2_exact=np.array_equal(two,final['action_K2']),
       action_K1_quorum_exact=np.array_equal(quorum,final['action_K1']),
       final_S80_exact=np.array_equal(recomputed,final['s80_K']),
       original_freeze_K_alignment=np.array_equal(final['incumbent_freeze_K'],
                                     pr['incumbent_freeze_K'][ix]),
       K0_to_K6_only=np.isin(recomputed,range(7)).all())
    must(all(checks.values()),'independent S80 source replay disagrees '+str(checks))
    altered=np.flatnonzero(recomputed!=final['incumbent_freeze_K'])
    report=dict(status='S228_INDEPENDENT_COMPLETE_ORIGINAL_WEIGHT_S80_NEW_WAV_REPLAY',
       full_model_source_run=38297654063,
       event_count=n,PR16_votes=36,PR16_original_neural_heads=12,
       S46_original_policies=193,S79_trained_fold_models=4,
       independent_formula_checks=checks,
       source_original_model_K1_actions=int(quorum.sum()),
       original_final_K0_actions=int(zero.sum()),
       original_final_K2_actions=int(two.sum()),
       event_K_histogram=np.bincount(recomputed,minlength=7).tolist(),
       final_prediction_changed_vs_freeze_count=len(altered),
       changed_unlabeled_event_ids=[ids[i] for i in altered],
       independent_ExactK_global=None,independent_ExactK_poly=None,
       note='All events from GOAT were previously exposed; no MIDI or truth available in runtime',
       models_untouched=True,truth_or_MIDI_read=False,
       new_songs_evaluated=False,promotion_eligibility_verified=False,
       official_S80_promoted=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('s227','s219','s85','out'):a.add_argument('--'+k,required=True,type=Path)
    p=a.parse_args()
    run(p.s227,p.s219,p.s85,p.out)
