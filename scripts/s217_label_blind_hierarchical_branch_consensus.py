"""S217: label-blind, piece-neutral hierarchical deployment consensus for S18/S46.

Research policy fixed before seeing any new song labels:
* Each historic held-piece network carries ONE branch vote in its own fold.
* Within every fold (4/5 branches) a STRICT majority wins, otherwise its
  original fold S18 prediction wins. Ties never use branch filename order.
* The four folds have equal weight, irrespective of 4 vs 5 branches.
  Require >=3/4 matching fold predictions; otherwise abstain to the
  original freeze_local_combo K for S18, or the already aggregated S18 K
  for other S46 policies.
* S46 column 0 is explicitly the aggregated S18 parent. Any disagreement
  is exposed as a confidence/abstention flag, not hidden or scored.

This is a NEW fully deterministic deploy rule, not an original S80
historical checkpoint, and not evidence of improved Exact-K.
No truth/MIDI, fitting, score selection, or piece-index oracle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
NAMES_PER_FOLD=(5,4,5,5)
BAD={'true_K','truth','label','labels','target','midi','correct','y','accuracy','regression'}
CANDIDATES=np.arange(7,dtype=np.int8)

def require(c,m):
    if not c:raise ValueError(m)

def strict_npz(p,keys):
    with np.load(p,allow_pickle=False) as z:
        require(set(z.files)==set(keys) and not BAD.intersection(z.files),
                'strict label-free input fields for '+str(p))
        return {k:z[k].copy() for k in z.files}

def count_and_vote(stack,base,quorum):
    """Votes: [number_of_voters,N,P]; baseline [N,P], never voter-order tie."""
    arr=np.asarray(stack)
    base=np.asarray(base)
    require(arr.ndim==3 and base.shape==arr.shape[1:],
            'voting geometry')
    require(np.isin(arr,CANDIDATES).all() and np.isin(base,CANDIDATES).all(),
            'all K0 through K6 in expected support')
    require(1<=quorum<=len(arr),'invalid consensus quorum')
    counts=np.stack([np.sum(arr==k,axis=0) for k in CANDIDATES],axis=-1)
    top=counts.argmax(axis=-1).astype(np.int8)
    strength=counts.max(axis=-1).astype(np.int8)
    supported=strength>=quorum
    winner=np.where(supported,top,base).astype(np.int8)
    return winner,supported,strength

def aggregate(branches,folds,freeze):
    """19 x N x 193 -> N x 193 with named 4-fold equal-weight consensus."""
    x=np.asarray(branches,dtype=np.int8)
    f=np.asarray(folds,dtype=np.int8)
    b=np.asarray(freeze,dtype=np.int8)
    require(x.ndim==3 and x.shape[0]==19 and x.shape[2]==193,
            'nineteen saved original 193-policy branches required')
    n=x.shape[1]
    require(n>0 and f.shape==(19,) and b.shape==(n,),
            'branch fold IDs and baseline align')
    require(np.isin(x,CANDIDATES).all() and np.isin(b,CANDIDATES).all(),
            'no missing K output')
    require([int((f==fold).sum()) for fold in FOLDS]==list(NAMES_PER_FOLD),
            'original 5/4/5/5 fold weights required')
    parents=np.stack([x[f==fold, :, 0][0] for fold in FOLDS])
    for fold,parent in zip(FOLDS,parents):
        require(np.array_equal(x[f==fold,:,0],np.broadcast_to(parent,
                         x[f==fold,:,0].shape)),
                'S18 parent differs between branches of the same fold')
    parent,supported_parent,parent_strength=count_and_vote(
        parents[:,:,None],b[:,None],3)
    parent=parent[:,0]
    per_fold=[];within_supported=[];within_strength=[]
    for fold,ref in zip(FOLDS,parents):
        members=x[f==fold]
        q=members.shape[0]//2+1
        predicted,consensus,confidence=count_and_vote(
            members,np.broadcast_to(ref[:,None],(n,193)),q)
        require(np.array_equal(predicted[:,0],ref),'within fold parent changed')
        per_fold.append(predicted)
        within_supported.append(consensus)
        within_strength.append(confidence)
    fold_pred=np.stack(per_fold)
    support=np.stack(within_supported)
    fold_strength=np.stack(within_strength)
    # A single original S18 parent is used if the fold policy votes lack 3/4.
    aggregated,across_supported,across_strength=count_and_vote(
         fold_pred,np.broadcast_to(parent[:,None],(n,193)),3)
    aggregated[:,0]=parent
    across_supported[:,0]=supported_parent[:,0]
    across_strength[:,0]=parent_strength[:,0]
    require(np.array_equal(aggregated[:,0],parent),'aggregated S18 column invariant')
    require(np.isin(aggregated,CANDIDATES).all(),'invalid combined K')
    # On 57/19 GOAT, S46 can still be != frozen parent, but those moves
    # require independent held-piece agreement and >=3 real fold votes.
    return {'predictions':aggregated,'s18_K':parent,
            'within_fold_K':fold_pred,'within_fold_agreement':support,
            'within_fold_top_vote_count':fold_strength,
            'across_fold_agreement':across_supported,
            'across_fold_top_vote_count':across_strength,
            'original_S18_folds':parents,
            'parent_fallback_to_freeze':~supported_parent[:,0],
            'fallback_to_aggregated_S18':~across_supported[:,1:],
            'original_19_branch_K':x}

def selftest():
    b=np.array([2,3,0,1],np.int8)
    x=np.zeros((19,4,193),np.int8)
    f=np.array([0]*5+[1]*4+[2]*5+[4]*5,dtype=np.int8)
    x[:]=b[None,:,None]
    # Two folds favor K4 but fold #2 K3 and fold #4 K2:
    # no 3/4 quorum -> fallback to frozen b on event 0.
    x[f==0,0,:]=4
    x[f==1,0,:]=4
    x[f==2,0,:]=3
    x[f==4,0,:]=2
    # In fold 0, one minority BFIRST heldpiece cannot overturn
    # strict majority, regardless of order.
    x[0,1,1:]=5
    r=aggregate(x,f,b)
    require(r['s18_K'].tolist()==[2,3,0,1],'tie must abstain to frozen K')
    require(r['parent_fallback_to_freeze'][0] and not r['parent_fallback_to_freeze'][1],
            'four fold confidence')
    require(np.array_equal(r['predictions'],aggregate(x[::-1],f[::-1],b)['predictions']),
            'heldpiece input ordering cannot change any prediction')
    require(np.array_equal(r['predictions'],aggregate(np.roll(x,7,axis=0),
                    np.roll(f,7),b)['predictions']),
            'shifted input ordering cannot change predictions')
    # Fold with 4 piece checkpoints is NOT downweighted vs fold with 5.
    x[:]=b[None,:,None]
    x[f==0,1,1:]=4
    x[f==1,1,1:]=4
    x[f==2,1,1:]=4
    r=aggregate(x,f,b)
    require(np.all(r['predictions'][1,1:]==4) and r['s18_K'][1]==3,
            '3/4 fold change must be carried through consensus')
    # An internally tied 4-piece branch must never get a false majority.
    x[:]=b[None,:,None]
    one=np.flatnonzero(f==1)[:2]
    x[one,2,1:]=5
    r=aggregate(x,f,b)
    require(np.all(r['within_fold_K'][1,2,1:]==b[2]),
            '2-vs-2 within four-piece fold must abstain to fold parent')
    print('S217 TEST PASS: fold weight balance, 3/4 vote, branch permutation, ties, no K invention')

def run(root,s18file,out):
    require(not out.exists(),'immutable S217 aggregation evidence')
    source=strict_npz(s18file,{'row_id','freeze_K','fold_ids','global_K',
                     'poly_K','S8_K','S9_K','S13_K','S14_K',
                     'S15_K','S16_K','S17_K','S18_K'})
    ids=source['row_id'];freeze=source['freeze_K'].astype(np.int8)
    require(len(ids)>0 and len(np.unique(ids))==len(ids),
            'stable unique original audio event IDs')
    paths=sorted(root.glob('branch*_fold*/s46/predictions.npz'))
    require(len(paths)==19,'S215 original branch census missing')
    branch_outputs=[];branch_names=[];folds=[]
    names=None
    for file in paths:
        m=re.fullmatch(r'branch(\d+)_fold(\d+)',file.parent.parent.name)
        require(m is not None,'non-historic fold branch path '+str(file))
        branch=int(m.group(1));fold=int(m.group(2))
        require(fold in FOLDS and branch not in branch_names,
                'invalid fold or duplicate branch identity')
        current=strict_npz(file,{'row_id','variant_ids','predictions'})
        require(np.array_equal(current['row_id'],ids),
                'S215 branch source mismatch')
        p=current['predictions']
        require(p.shape==(len(ids),193),'S46 policy number')
        if names is None:names=current['variant_ids']
        require(np.array_equal(current['variant_ids'],names),
                'S46 policies drift across branches')
        contract=strict_npz(file.parent.parent/'proposals.npz',
            {'row_id','parent_K','family_ids','proposal_K'})
        require(np.array_equal(contract['row_id'],ids) and
                np.array_equal(contract['parent_K'],p[:,0]),
                'S18 parent not aligned with S45 branch')
        branch_outputs.append(p.astype(np.int8))
        branch_names.append(branch)
        folds.append(fold)
    require(sorted(branch_names)==list(range(19)) and
            list(map(int,source['fold_ids']))==list(FOLDS),
            'complete stable historical branch and fold identities')
    inp=np.stack(branch_outputs);folds=np.asarray(folds,np.int8)
    result=aggregate(inp,folds,freeze)
    orig=source['S18_K'].astype(np.int8)
    require(orig.shape==(4,len(ids)),'S18 authentic 4 parent source geometry')
    for fi,fold in enumerate(FOLDS):
        require(np.array_equal(orig[fi],result['original_S18_folds'][fi]),
                'S215 S18 parent differs from frozen upstream S18 chain fold '+str(fold))
    # Order and family identity invariance tested over all 19 real branches;
    # no historical piece label chosen, no truth labels read.
    for perm in (np.arange(18,-1,-1),np.roll(np.arange(19),7)):
        second=aggregate(inp[perm],folds[perm],freeze)
        for key in ('predictions','s18_K','within_fold_K',
                    'across_fold_agreement','within_fold_agreement'):
            require(np.array_equal(result[key],second[key]),
                    'aggregation depends on held-piece branch enumeration: '+key)
    n=len(ids)
    unequal=np.any(orig!=orig[0:1],axis=0)
    diff=(inp!=result['predictions'][None])
    # Reasons for abstention are tracked independently of predictions.
    report=dict(status='S217_ORIGINAL_S18_AND_S46_19_BRANCH_HIERARCHICAL_CONSENSUS',
       source_run_id=38242983478,source_audit_run_id=38243390558,
       sample_count=n,source_branch_count=19,source_folds=list(FOLDS),
       S46_policy_count=193,prespecified_rule='strict_majority_of_4or5_per_fold_then_3of4_fold_vote_else_parent',
       within_fold_strict_majority=True,fold_equal_weight=True,
       fold_quorum=3,parent_fallback='freeze_local_combo',
       S46_fallback='aggregated_S18',
       originally_disagreeing_S18_events=int(unequal.sum()),
       remaining_multi_valued_S18_decisions=0,
       unresolved_S18_parent_events_flagged=int(result['parent_fallback_to_freeze'].sum()),
       S18_aggregated_changed_vs_freeze=int(np.count_nonzero(result['s18_K']!=freeze)),
       S46_policy_event_differences_vs_aggregated_S18=int(np.count_nonzero(result['predictions'][:,1:]!=result['s18_K'][:,None])),
       within_fold_ambiguous_policy_event_votes=[
           int(np.count_nonzero(~result['within_fold_agreement'][i])) for i in range(4)],
       across_fold_no_three_vote_policy_event_cells=int(np.count_nonzero(~result['across_fold_agreement'])),
       original_S18_fold_disagreement_event_ids=list(map(str,ids[unequal])),
       original_19_branch_policy_events_disagreeing_with_aggregated_result=int(diff.sum()),
       branch_permutation_invariant=True,decision_has_no_fold_or_piece_oracle=True,
       all_19_original_branch_outputs_preserved=True,
       source_truth_or_MIDI_read=False,model_weights_changed=False,
       historical_original_S80_rule=False,
       new_song_independent_ExactK_measured=False,
       S69_S80_downstream_replayed=False,S80_promoted=False)
    out.mkdir(parents=True)
    np.savez_compressed(out/'one_piece_neutral_193_policy_prediction.npz',
       row_id=ids,variant_ids=names,original_freeze_K=freeze,
       s18_consensus_K=result['s18_K'],policy_consensus_K=result['predictions'],
       policy_has_three_fold_votes=result['across_fold_agreement'],
       policy_fold_highest_vote_count=result['across_fold_top_vote_count'],
       parent_abstained_to_freeze=result['parent_fallback_to_freeze'],
       original_fold_s18_K=result['original_S18_folds'],
       fold_s46_consensus_K=result['within_fold_K'],
       fold_s46_has_strict_majority=result['within_fold_agreement'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--s215',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(all(getattr(a,x) is not None for x in ('s215','s18','out')),
                'required 19-branch S215 and S18 label-free sources')
        selftest()
        run(a.s215,a.s18,a.out)
