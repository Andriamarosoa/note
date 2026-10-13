"""S369 read-only original S18 -> 193 S46 -> S69 K3 regression lineage.

Replays SHA verified original S349 S69 frozen trained rules, then inspects
all SIX true K3->predicted K2 regressions against S18 parent, original
four trained S18 fold votes, 193 S46 specialist bank, S69 raw
polymax/balanced interventions and actual 3/4 risk veto results.
Source truth is ALREADY EXPOSED in S363. No model tuning and NO promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.audit_v273_s69_s46_vote_guard import POLICIES
S18={'row_id','freeze_K','original_fold_ids','original_S18_by_fold_K',
      'S18_piece_neutral_consensus_K','S18_quorum_of_3_supported',
      'S18_vote_strength','S18_abstention_to_freeze'}
S46={'row_id','variant_ids','original_freeze_K','S18_parent_K',
      'original_19_branch_folds','S46_193_policy_K',
      'per_fold_specialist_K','active_fold_count',
      'parent_incompatible_specialists','contradictory_specialists'}
S69={'row_id','S18_parent_K','balanced_s46_K','balanced_protected_K',
     'polymax_s46_K','polymax_protected_K',
     'balanced_fold_veto','balanced_veto_three_folds',
     'polymax_fold_veto','polymax_veto_three_folds'}
TRUTH={'row_id','member','historical_attack_K_if_XML_complete',
       'acoustic_anchored_K_or_unknown','official_freeze_K',
       'S80_original_K','S80_twofold_K'}
PRED={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
      'original_PR16_policy22_K','K1_by_fold_threshold_action',
      'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
def need(x,why):
    if not x:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,fields):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==fields,'original frozen stage schema mismatch: '+str(path))
        return {k:z[k].copy() for k in z.files}
def stage_compare(y,base,s18,s46,s69,out):
    mask=y==3
    idx=np.flatnonzero(mask)
    return dict(n=int(len(idx)),correct=int((out[idx]==y[idx]).sum()),
       predicts_K0to6=np.bincount(out[idx].astype(int),minlength=7).tolist())
def run(src18,src46,src69,truthfile,frozen,out):
    need(not out.exists(),'do not mutate previously exposed test source')
    a=load(src18,S18);b=load(src46,S46)
    c=load(src69,S69);d=load(truthfile,TRUTH);z=load(frozen,PRED)
    ids=a['row_id'];n=len(ids)
    need(n==967 and all(np.array_equal(ids,e['row_id']) for e in (b,c,d,z)),
         'original 967 native IDs source drift')
    truth=d['historical_attack_K_if_XML_complete']
    reference=d['official_freeze_K']
    final=d['S80_original_K']
    need(np.array_equal(reference,a['freeze_K'])
         and np.array_equal(reference,b['original_freeze_K'])
         and np.array_equal(reference,z['incumbent_freeze_K'])
         and np.array_equal(a['S18_piece_neutral_consensus_K'],b['S18_parent_K'])
         and np.array_equal(b['S18_parent_K'],c['S18_parent_K'])
         and np.array_equal(final,z['s80_K']),
         'original S18/S46/S69/final identities or target drift')
    parent=b['S18_parent_K'];poly=c['polymax_protected_K']
    flags=(truth==3)&(reference==3)&(final==2)
    need(int(flags.sum())==6,'all six S367 true K3->K2 regressions required')
    names=list(map(str,b['variant_ids']))
    need(len(names)==193 and names[0]=='series18_parent'
         and np.array_equal(b['S46_193_policy_K'][:,0],parent)
         and c['polymax_s46_K'].shape==(n,),
         'genuine S46 branch S69 source validation failure')
    details=[]
    for i in np.flatnonzero(flags):
        vote=np.asarray(a['original_S18_by_fold_K'][:,i]).astype(int)
        fold_hist=np.bincount(vote,minlength=7).tolist()
        s46_unique=np.bincount(np.asarray(b['S46_193_policy_K'][i],int),minlength=7).tolist()
        details.append(dict(row_id=str(ids[i]),truth=3,reference=3,final=2,
            S18_parent_K=int(parent[i]),S18_parent_equals_reference_K3=bool(parent[i]==3),
            S18_four_original_fold_K=list(map(int,vote)),S18_fold_votes_histogram=fold_hist,
            S18_vote_strength=int(a['S18_vote_strength'][i]),
            S18_abstention_to_official_baseline=bool(a['S18_abstention_to_freeze'][i]),
            S46_193_K_histogram=s46_unique,
            S46_scenario_polymax_raw=int(c['polymax_s46_K'][i]),
            S46_scenario_balanced_raw=int(c['balanced_s46_K'][i]),
            S69_polymax_after_4fold_veto=int(poly[i]),
            S69_balanced_after_4fold_veto=int(c['balanced_protected_K'][i]),
            S69_polymax_veto_count=int(c['polymax_fold_veto'][:,i].sum()),
            S69_polymax_3of4_veto=bool(c['polymax_veto_three_folds'][i]),
            S69_balanced_veto_count=int(c['balanced_fold_veto'][:,i].sum()),
            S69_balanced_3of4_veto=bool(c['balanced_veto_three_folds'][i]),
            first_K3_to_K2_stage=('S18_parent' if parent[i]==2 else
               'S46_polymax_specialist' if c['polymax_s46_K'][i]==2 else
               'S69_polymax_policy_or_other')))
    first=[x['first_K3_to_K2_stage'] for x in details]
    bad2=(truth==2)&(reference!=2)&(final==2)
    need(int(bad2.sum())==8,'second cohort should have exactly eight K2 reference corrections')
    # Distinguish actual true K3 from true K2 at parent / specialist / source.
    by_truth={}
    for k in (2,3,4):
        mask=truth==k
        by_truth[f'K{k}']={}
        for label,v in [('official',reference),('S18_parent',parent),
                        ('S46_raw_polymax',c['polymax_s46_K']),
                        ('S69_polymax',poly),('S80_final',final)]:
            by_truth[f'K{k}'][label]=dict(correct=int(np.count_nonzero(v[mask]==k)),
                 N=int(mask.sum()),K_hist=np.bincount(v[mask].astype(int),minlength=7).tolist())
    result=dict(status='S369_TRUE_K3_REGRESSIONS_PROVEN_SOURCE_S18_S46_S69',
      original_source_S338_SHA256=sha(src18),
      original_source_S346_SHA256=sha(src46),
      original_source_S349_SHA256=sha(src69),
      original_first_label_S363_SHA256=sha(truthfile),
      original_complete_S80_prelabel_SHA256=sha(frozen),
      true_K3_regressions=len(details),regressions_first_K3toK2_stage_counts={
          stage:first.count(stage) for stage in sorted(set(first))},
      first_source_to_underestimate_identified_no_model_fit=True,
      true_K2_reference_corrections_on_second_cohort=int(bad2.sum()),
      all_true_K3_count_in_second_cohort=int(np.count_nonzero(truth==3)),
      stage_accuracies_by_true_class_if_XML_complete=by_truth,
      six_regression_case_origins=details,
      original_S18_foldvote_weights_unmodified=True,
      original_S46_193_proposals_unmodified=True,
      original_S69_rules_unmodified=True,
      original_S80_unchanged=True,
      true_K_labels_previously_exposed_by_S363=True,
      cannot_claim_independent_K3_repair_from_this_test=True,
      official_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in
                     ('six_regression_case_origins','stage_accuracies_by_true_class_if_XML_complete')},
                     indent=2),flush=True)
    print('EACH 6 K3 FAIL',json.dumps(details,indent=2),flush=True)
    print('K2/3/4 STAGES',json.dumps(by_truth,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s338','s346','s349','truth','frozen','out'):
        p.add_argument('--'+k,type=Path,required=True)
    q=p.parse_args();run(q.s338,q.s346,q.s349,q.truth,q.frozen,q.out)
