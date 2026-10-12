"""S220: independent Counter vote audit of S219 conditional sparse gates.

No import of S217 or S219 inference. Recompute all 57x193 original-branch
vote outcomes with explicit lists and Counter; verify parent abstention,
active fold consensus, 2 surviving corrections, and blocked mismatch.
No labels or accuracy.
"""
from __future__ import annotations
import argparse,re,json
from pathlib import Path
from collections import Counter
import numpy as np
FOLDS=(0,1,2,4)
BAD={'label','labels','truth','true_K','midi','target','correct','accuracy','y','regression'}
def need(v,msg):
    if not v:raise ValueError(msg)
def load(p,required):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(required) and not BAD.intersection(z.files),'label-free exact keys '+str(p))
        return {k:z[k].copy() for k in z.files}
def run(src,old,sparse,out):
    need(not out.exists(),'immutable independent S220 audit')
    before=load(old,{'row_id','variant_ids','original_freeze_K',
      's18_consensus_K','policy_consensus_K','policy_has_three_fold_votes',
      'policy_fold_highest_vote_count','parent_abstained_to_freeze',
      'original_fold_s18_K','fold_s46_consensus_K','fold_s46_has_strict_majority'})
    after=load(sparse,{'row_id','variant_ids','s18_consensus_K','s46_policy_K',
      'per_fold_proposal_K','per_fold_proposal_strength',
      'proposal_rejected_due_different_S18_parent',
      'conflicting_active_fold_proposals','active_fold_count'})
    n=len(before['row_id']);P=193
    need(n==57 and np.array_equal(before['row_id'],after['row_id']) and
         np.array_equal(before['variant_ids'],after['variant_ids']) and
         np.array_equal(before['s18_consensus_K'],after['s18_consensus_K']),
         'same event, policy and label-free parent as S217')
    bank={f:[] for f in FOLDS}
    for file in sorted(src.glob('branch*_fold*/s46/predictions.npz')):
        m=re.fullmatch(r'branch(\d+)_fold(\d+)',file.parent.parent.name)
        need(m is not None and int(m.group(2)) in FOLDS,'unknown branch')
        z=load(file,{'row_id','variant_ids','predictions'})
        need(np.array_equal(z['row_id'],before['row_id']) and
             np.array_equal(z['variant_ids'],before['variant_ids']) and
             z['predictions'].shape==(n,P),'invalid source S46 shape')
        bank[int(m.group(2))].append(z['predictions'])
    need([len(bank[f]) for f in FOLDS]==[5,4,5,5],'all historical branches required')
    expected=np.repeat(before['s18_consensus_K'][:,None],P,axis=1).astype(np.int8)
    fproposal=np.full((4,n,P),-1,np.int8)
    fstrength=np.zeros((4,n,P),np.int8)
    ignored=np.zeros((4,n,P),bool)
    conflicts=np.zeros((n,P),bool)
    active=np.zeros((n,P),np.int8)
    for i in range(n):
        for j in range(P):
            signals=[]
            for fi,fold in enumerate(FOLDS):
                parent=int(before['original_fold_s18_K'][fi,i])
                choices=[int(z[i,j]) for z in bank[fold] if int(z[i,j])!=parent]
                counts=Counter(choices)
                strongest=max(counts.values(),default=0)
                fstrength[fi,i,j]=strongest
                if strongest < len(bank[fold])//2+1:continue
                winners=[k for k,v in counts.items() if v==strongest]
                need(len(winners)==1,'strict majority of branches cannot tie')
                candidate=winners[0]
                fproposal[fi,i,j]=candidate
                if parent!=int(before['s18_consensus_K'][i]):
                    ignored[fi,i,j]=True
                else:
                    signals.append(candidate)
            active[i,j]=len(signals)
            if j==0:continue
            if len(set(signals))>1:
                conflicts[i,j]=True
            elif signals:
                expected[i,j]=signals[0]
    checks={
      'all_S46_output_cells':np.array_equal(expected,after['s46_policy_K']),
      'S18_parent_preserved':np.array_equal(expected[:,0],before['s18_consensus_K']),
      'all_4_fold_candidate_values':np.array_equal(fproposal,after['per_fold_proposal_K']),
      'all_candidate_vote_counts':np.array_equal(fstrength,after['per_fold_proposal_strength']),
      'all_parent_mismatch_rejection_masks':np.array_equal(ignored,after['proposal_rejected_due_different_S18_parent']),
      'all_actual_contradiction_masks':np.array_equal(conflicts,after['conflicting_active_fold_proposals']),
      'all_active_fold_counts':np.array_equal(active,after['active_fold_count'])}
    need(all(checks.values()),'S219 independent Counter replay diverged '+str(checks))
    differences=expected[:,1:]!=before['s18_consensus_K'][:,None]
    events=np.flatnonzero(differences.any(axis=1))
    evidence=[]
    for i in events:
        variants=np.flatnonzero(differences[i])+1
        evidence.append(dict(row_id=str(before['row_id'][i]),
            parent_K=int(before['s18_consensus_K'][i]),
            corrected_variant_ids=list(map(str,before['variant_ids'][variants])),
            corrected_predicted_K=list(map(int,expected[i,variants])),
            original_fold_parent_K=list(map(int,before['original_fold_s18_K'][:,i])),
            active_fold_counts=list(map(int,active[i,variants]))))
    report=dict(status='S220_INDEPENDENT_SPARSE_19BRANCH_193POLICY_REPLAY',
       source_run=38242983478,source_consensus_run=38294699823,
       branch_count=19,events=n,policies=P,all_checks_passed=checks,
       independent_algorithm='collections.Counter direct per-event branch votes; no S219 imports',
       original_sparse_source_interventions=int(np.count_nonzero(
          np.stack([v for f in FOLDS for v in bank[f]])[:,:,1:] !=
          np.stack([v for f in FOLDS for v in bank[f]])[:,:,0,None])),
       accepted_non_parent_policy_cells=int(differences.sum()),
       unique_events_with_S46_interventions=int(len(events)),
       rejected_mismatched_fold_s18=int(ignored.sum()),
       conflicting_active_fold_policy_events=int(conflicts.sum()),
       kept_S46_evidence=evidence,
       no_labels_read=True,independent_ExactK_scored=False,
       original_S80_completely_deployable=False,S80_promoted=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('s215','s217','s219','out'):
        p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args()
    run(a.s215,a.s217,a.s219,a.out)
