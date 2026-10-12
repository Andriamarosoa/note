"""S219: piece-neutral CONDITIONAL SPECIALIST aggregation for sparse S46 gates.

S217 3/4 unanimous vote over final K suppresses ALL 602 sparse original S46
branch interventions. That's a structural error: a fold that outputs its S18
parent has no *proposal*, and must be distinguished from a fold actively
opposing another fold's correction. Pre-label rules:
  (a) S18 remains S217 original 4-fold majority with fallback freeze.
  (b) Per S46 policy, >=strict majority of 4/5 same-fold branch
      checkpoints must actively change their own S18 parent to the SAME K.
  (c) Only proposals whose original fold S18 == aggregated S18 may act.
  (d) If at least one eligible fold proposes and ALL active folds agree,
      accept K. Contradictory active folds -> fallback aggregated S18.
  (e) Unchanged folds abstain, not veto. No identity-based piece choice,
      no label-based weights, and all dissent/abstention flags retained.

This is a new candidate deployment policy, NOT old historical S80.
Use alone ONLY for label-free output plumbing and independent future tests.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np
FOLDS=(0,1,2,4)
BAD={'true_K','truth','label','labels','target','midi','correct','y','accuracy','regression'}
def need(v,why):
    if not v:raise ValueError(why)
def read(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(keys) and not BAD.intersection(z.files),
             'S219 accepts no labels or extra data '+str(p))
        return {k:z[k].copy() for k in z.files}
def solve(x,folds,parent):
    """All 19 original fold branches [19,N,193], original aggregated S18 [N]."""
    x=np.asarray(x,dtype=np.int8);folds=np.asarray(folds,dtype=np.int8)
    parent=np.asarray(parent,dtype=np.int8)
    need(x.ndim==3 and x.shape[0]==19 and x.shape[2]==193,
         '19 source S46 193 policy matrix required')
    n=x.shape[1]
    need(parent.shape==(n,) and [int((folds==f).sum()) for f in FOLDS]==[5,4,5,5],
         'S18 one parent and exactly four historical fold groups required')
    need(np.isin(x,np.arange(7)).all() and np.isin(parent,np.arange(7)).all(),
         'exact K classes 0 through 6 only')
    parent_by_fold=np.stack([x[folds==f, :, 0][0] for f in FOLDS])
    for fi,fold in enumerate(FOLDS):
        need(np.array_equal(x[folds==fold,:,0],
            np.broadcast_to(parent_by_fold[fi],x[folds==fold,:,0].shape)),
            'S18 fold parents must not vary inside held-piece fold')
    per_fold_k=np.full((4,n,193),-1,np.int8)
    per_fold_strength=np.zeros((4,n,193),np.int8)
    per_fold_dissent=np.zeros((4,n,193),bool)
    unsupported=np.zeros((4,n,193),bool)
    for fi,fold in enumerate(FOLDS):
        branch=x[folds==fold]
        quorum=len(branch)//2+1
        active=branch!=parent_by_fold[fi][None,:,None]
        quorum_counts=np.stack([np.sum((branch==k)&active,axis=0)
                   for k in range(7)],axis=-1)
        top=quorum_counts.argmax(-1).astype(np.int8)
        votes=quorum_counts.max(-1).astype(np.int8)
        accepted=votes>=quorum
        per_fold_k[fi]=np.where(accepted,top,-1)
        per_fold_strength[fi]=votes
        per_fold_dissent[fi]=(active.sum(0)>votes) & accepted
        unsupported[fi]=(active.sum(0)>0) & (~accepted)
    # Distinguish specialist proposals from uninformed folds (no change).
    matching_parent=parent_by_fold==parent[None,:]
    eligible=(per_fold_k>=0)&matching_parent[:,:,None]
    ignored_wrong_parent=(per_fold_k>=0)&(~matching_parent[:,:,None])
    accepted=np.broadcast_to(parent[:,None],(n,193)).copy()
    agreement=np.ones((n,193),bool)
    fold_support=np.zeros((n,193),np.int8)
    conflicting=np.zeros((n,193),bool)
    for k in range(7):
        supporters=np.sum(eligible & (per_fold_k==k),axis=0)
        fold_support=np.maximum(fold_support,supporters.astype(np.int8))
    actives=eligible.sum(axis=0)
    # No folds proposing -> abstain; all active folds propose same K ->
    # accept. No selected held-piece checkpoint, no fold chosen by ID.
    for k in range(7):
        votes=np.sum(eligible&(per_fold_k==k),axis=0)
        only_this=(votes>0)&(votes==actives)
        accepted[only_this]=k
    conflicting=actives>fold_support
    accepted[conflicting]=np.broadcast_to(parent[:,None],(n,193))[conflicting]
    agreement=~conflicting
    accepted[:,0]=parent
    need(np.isin(accepted,range(7)).all() and
         np.array_equal(accepted[:,0],parent),'parent identity and output class range')
    return dict(predictions=accepted,fold_proposals=per_fold_k,
       fold_proposal_strength=per_fold_strength,
       fold_proposal_dissent=per_fold_dissent,
       weak_active_proposals=unsupported,
       fold_parent_K=parent_by_fold,
       eligible_fold_proposals=eligible,
       wrong_parent_proposals_ignored=ignored_wrong_parent,
       contradictory_fold_proposal=conflicting,
       active_fold_support=actives,
       no_active_fold_proposals=(actives==0),
       all_active_fold_proposals_agree=agreement)

def selftest():
    f=np.array([0]*5+[1]*4+[2]*5+[4]*5,np.int8)
    base=np.array([2,3,4],np.int8)
    x=np.broadcast_to(base[None,:,None],(19,3,193)).copy()
    # 5 unanimous fold4 experts proposing 1; other folds unchanged
    x[f==4,0,1:]=1
    r=solve(x,f,base)
    need(np.all(r['predictions'][0,1:]==1) and
         np.all(r['predictions'][0,0]==2),
         'silent folds must abstain, not veto one unanimously active fold')
    # less than majority in a 4-member fold must be ignored
    x=np.broadcast_to(base[None,:,None],(19,3,193)).copy()
    ix=np.flatnonzero(f==1)[:2]
    x[ix,0,1:]=1
    r=solve(x,f,base)
    need(np.all(r['predictions'][0,1:]==2),
         'two of four proposing is not a credible source majority')
    # conflict between two qualified active folds -> base fallback
    x[f==0,0,1:]=0
    x[f==4,0,1:]=1
    r=solve(x,f,base)
    need(np.all(r['predictions'][0,1:]==2) and
         np.all(r['contradictory_fold_proposal'][0,1:]),
         'conflicting real specialists must abstain to aggregated S18')
    # Unsupported S18 fold cannot override a dominant fold parent.
    x=np.broadcast_to(base[None,:,None],(19,3,193)).copy()
    x[f==4,0,0]=3
    x[f==4,0,1:]=1
    r=solve(x,f,base)
    need(np.all(r['predictions'][0,1:]==2) and
         np.all(r['wrong_parent_proposals_ignored'][3,0,1:]),
         'correction from mismatched S18 parent must not be used')
    y=np.roll(x,8,axis=0);q=np.roll(f,8)
    need(np.array_equal(solve(x,f,base)['predictions'],
                        solve(y,q,base)['predictions']),
         'branch order cannot determine winner')
    print('S219 PASS: silent folds abstain; weak branch votes rejected; conflicting proposals rejected; mismatched parent rejected; identity invariance')

def run(root,s217,out):
    need(not out.exists(),'immutable S219 candidate')
    base=read(s217,{'row_id','variant_ids','original_freeze_K','s18_consensus_K',
       'policy_consensus_K','policy_has_three_fold_votes',
       'policy_fold_highest_vote_count','parent_abstained_to_freeze',
       'original_fold_s18_K','fold_s46_consensus_K',
       'fold_s46_has_strict_majority'})
    ids=base['row_id'];n=len(ids)
    source=[];folds=[];seen=[]
    for p in sorted(root.glob('branch*_fold*/s46/predictions.npz')):
        m=re.fullmatch(r'branch(\d+)_fold(\d+)',p.parent.parent.name)
        need(m is not None,'S46 file must have original branch name')
        branch=int(m.group(1));fold=int(m.group(2))
        z=read(p,{'row_id','variant_ids','predictions'})
        need(np.array_equal(z['row_id'],ids) and
             np.array_equal(z['variant_ids'],base['variant_ids']) and
             z['predictions'].shape==(n,193),'original S46 event identity')
        source.append(z['predictions']);folds.append(fold);seen.append(branch)
    need(sorted(seen)==list(range(19)),'19 exact historical branches')
    fold_ids=np.asarray(folds,np.int8)
    inp=np.stack(source)
    result=solve(inp,fold_ids,base['s18_consensus_K'])
    for perm in (np.arange(18,-1,-1),np.roll(np.arange(19),3)):
        q=solve(inp[perm],fold_ids[perm],base['s18_consensus_K'])
        for k in result:
            need(np.array_equal(q[k],result[k]),'order-dependent sparse proposal: '+k)
    need(np.array_equal(result['fold_parent_K'],base['original_fold_s18_K']),
         'S219 cannot silently change S217 S18 parent rule')
    nochange=base['policy_consensus_K'][:,1:]==base['s18_consensus_K'][:,None]
    need(nochange.all(),'S217 S46 source proof unexpectedly preserved votes')
    original_changes=np.count_nonzero(inp[:,:,1:]!=inp[:,:,0,None])
    res=result['predictions'];parent=base['s18_consensus_K']
    active_changes=int(np.count_nonzero(res[:,1:]!=parent[:,None]))
    changed_vs_s217=int(np.count_nonzero(res!=base['policy_consensus_K']))
    report=dict(status='S219_SPARSE_CONDITIONAL_S46_SOURCE_BRANCH_AGGREGATION',
       based_on_parent_run=38294189867,
       sample_count=n,branches=19,folds=list(FOLDS),policy_variants=193,
       original_S18_parent_policy='S217 majority3of4 else freeze unchanged',
       fold_within_quorum='strict_majority_of_4_or_5_original_piece_checkpoints',
       across_fold='among active folds agree; inactive folds abstain',
       reject_mismatched_original_S18_parent=True,
       conflicting_specialists_abstain_to_S18=True,
       previous_S217_corrections_survived=0,
       original_S46_interventions_across_19_branches=int(original_changes),
       new_S46_interventions_after_sparse_fusion=active_changes,
       new_S46_cells_changed_vs_S217=changed_vs_s217,
       fold_proposals_K_counts=[int((result['fold_proposals'][i]>=0).sum()) for i in range(4)],
       proposals_ignored_wrong_S18_parent=int(result['wrong_parent_proposals_ignored'].sum()),
       fold_proposals_conflicting_cells=int(result['contradictory_fold_proposal'].sum()),
       branch_identity_permutation_invariant=True,
       no_original_S18_fold_identity_selected=True,
       labels_read=False,score_measured=False,model_trained=False,
       new_deployment_accuracy_unverified=True,
       original_historical_S80_equivalence=False,S80_promoted=False)
    out.mkdir(parents=True)
    np.savez_compressed(out/'conditional_sparse_193_policy_prediction.npz',
       row_id=ids,variant_ids=base['variant_ids'],
       s18_consensus_K=parent,
       s46_policy_K=res,
       per_fold_proposal_K=result['fold_proposals'],
       per_fold_proposal_strength=result['fold_proposal_strength'],
       proposal_rejected_due_different_S18_parent=result['wrong_parent_proposals_ignored'],
       conflicting_active_fold_proposals=result['contradictory_fold_proposal'],
       active_fold_count=result['active_fold_support'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--s215',type=Path)
    p.add_argument('--s217',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,k) is not None for k in ('s215','s217','out')),
             'original source and S217 S18 consensus required')
        selftest()
        run(a.s215,a.s217,a.out)
