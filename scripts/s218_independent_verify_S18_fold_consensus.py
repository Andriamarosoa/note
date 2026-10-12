"""S218 INDEPENDENT implementation of S217 voting, using only Counter loops.

Deliberately DOES NOT import any S217 source code/function. Recomputes each
of the 57 x 193 policy choices from 19 archived original frozen model
branches, verifies branch and fold voting, abstention and full output byte
arrays. Records the critical fact when 3/4 voting rejects all S46
corrections; no label, MIDI, accuracy, or optimization.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from collections import Counter
from pathlib import Path
import numpy as np
FOLDS=(0,1,2,4)
FOLD_SIZES=(5,4,5,5)
BAD={'true_K','truth','label','labels','target','midi','correct','y','accuracy','regression'}
def need(x,why):
    if not x:raise ValueError(why)
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(fields) and not BAD.intersection(z.files),
             'disallow extraneous or label fields in '+str(p))
        return {k:z[k].copy() for k in z.files}
def decide(votes,needed,base):
    counts=Counter(map(int,votes))
    top=max(counts.values())
    winners=[k for k,q in counts.items() if q==top]
    if top>=needed:
        need(len(winners)==1,'quorum selected multiple incompatible K values')
        return int(winners[0]),True,top
    return int(base),False,top
def run(source_path,consensus_path,s18_path,out):
    need(not out.exists(),'immutable independent audit required')
    reference=load(consensus_path,{'row_id','variant_ids','original_freeze_K',
      's18_consensus_K','policy_consensus_K','policy_has_three_fold_votes',
      'policy_fold_highest_vote_count','parent_abstained_to_freeze',
      'original_fold_s18_K','fold_s46_consensus_K',
      'fold_s46_has_strict_majority'})
    acoustic=load(s18_path,{'row_id','freeze_K','fold_ids','global_K','poly_K',
      'S8_K','S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'})
    ids=reference['row_id'];n=len(ids);policies=193
    need(np.array_equal(ids,acoustic['row_id']) and n==57 and
         np.array_equal(reference['original_freeze_K'],acoustic['freeze_K']),
         'independent original freeze from recovered S18 source')
    paths=sorted(source_path.glob('branch*_fold*/s46/predictions.npz'))
    need(len(paths)==19,'all S215 original 19 branches required')
    bank={fold:[] for fold in FOLDS}
    files=[]
    for f in paths:
        m=re.fullmatch(r'branch(\d+)_fold(\d+)',f.parent.parent.name)
        need(m is not None,'nonoriginal branch '+str(f))
        k=int(m.group(2));need(k in FOLDS,'illegal fold')
        p=load(f,{'row_id','variant_ids','predictions'})
        need(np.array_equal(ids,p['row_id']) and
             np.array_equal(reference['variant_ids'],p['variant_ids']) and
             p['predictions'].shape==(n,policies),'branch identifiers policy census')
        bank[k].append(p['predictions'].astype(np.int8))
        files.append(f)
    need(tuple(len(bank[k]) for k in FOLDS)==FOLD_SIZES,'branch counts')
    base=acoustic['freeze_K']
    parents=acoustic['S18_K']
    need(parents.shape==(4,n),'original fold S18 outputs')
    pred=np.empty((n,policies),np.int8)
    strict=np.empty((4,n,policies),bool)
    foldK=np.empty((4,n,policies),np.int8)
    foldCount=np.empty((n,policies),np.int8)
    acrossOK=np.empty((n,policies),bool)
    abstain=np.empty(n,bool)
    parentK=np.empty(n,np.int8)
    corrections_by_fold=np.zeros(4,int)
    original_s46_dissent_by_fold=np.zeros(4,int)
    for fi,fold in enumerate(FOLDS):
        for piece in bank[fold]:
            need(np.array_equal(piece[:,0],parents[fi]),
                 'independent S18 fold parent inconsistent')
            original_s46_dissent_by_fold[fi]+=int(np.count_nonzero(
                 piece[:,1:]!=piece[:,0,None]))
    for i in range(n):
        p,ok,q=decide(parents[:,i],3,base[i])
        parentK[i]=p
        abstain[i]=not ok
        pred[i,0]=p
        acrossOK[i,0]=ok
        foldCount[i,0]=q
        for j in range(policies):
            onefold=[]
            for fi,fold in enumerate(FOLDS):
                votes=[int(z[i,j]) for z in bank[fold]]
                candidate,yes,count=decide(votes,len(votes)//2+1,parents[fi,i])
                foldK[fi,i,j]=candidate
                strict[fi,i,j]=yes
                onefold.append(candidate)
            if j==0:continue
            candidate,yes,count=decide(onefold,3,p)
            pred[i,j]=candidate
            acrossOK[i,j]=yes
            foldCount[i,j]=count
    checks={'full_193_policy_prediction':np.array_equal(pred,reference['policy_consensus_K']),
        'S18_parent_predictions':np.array_equal(parentK,reference['s18_consensus_K']),
        'all_fold_majority_masks':np.array_equal(strict,reference['fold_s46_has_strict_majority']),
        'all_fold_majority_values':np.array_equal(foldK,reference['fold_s46_consensus_K']),
        'all_193_policy_three_fold_masks':np.array_equal(acrossOK,reference['policy_has_three_fold_votes']),
        'all_193_policy_vote_strengths':np.array_equal(foldCount,reference['policy_fold_highest_vote_count']),
        'parent_abstention_masks':np.array_equal(abstain,reference['parent_abstained_to_freeze']),
        'unaltered_4fold_original_S18':np.array_equal(parents,reference['original_fold_s18_K'])}
    need(all(checks.values()),'S217 independent implementation disagreement '+str(checks))
    disagreement=np.any(parents!=parents[:1],axis=0)
    need(disagreement.sum()==6,'unexpected original six parent disagreement events')
    parents_for_changed=parents[:,disagreement]
    detailed=[]
    for pos in np.flatnonzero(disagreement):
        detailed.append({'event_id':str(ids[pos]),
            'four_original_S18_votes':list(map(int,parents[:,pos])),
            'original_freeze_K':int(base[pos]),
            'S18_consensus_K':int(parentK[pos]),
            'S18_vote_count':int(foldCount[pos,0]),
            'no_3_of_4_abstain':bool(abstain[pos])})
    source_s46_nonparents=int(sum(original_s46_dissent_by_fold))
    after_nonparent=int(np.count_nonzero(pred[:,1:]!=parentK[:,None]))
    report=dict(status='S218_INDEPENDENT_AGGREGATOR_EXACT_REPLAY',
        all_checks_passed=checks,independent_python_algorithm='pure collections.Counter vote loops; zero imports from S217',
        source_run=38242983478,consensus_run=38294189867,
        rows=n,variants=193,branches=19,folds=list(FOLDS),
        prior_s18_fold_disagreements=int(disagreement.sum()),
        final_number_of_s18_values_per_event=1,
        final_S18_abstentions=int(abstain.sum()),
        original_S46_branch_corrections_before_vote=source_s46_nonparents,
        original_S46_dissent_by_fold=list(map(int,original_s46_dissent_by_fold)),
        S46_policy_event_corrections_surviving_3_of_4=int(after_nonparent),
        S46_active_source_signal_survived=bool(after_nonparent>0),
        six_disagreement_diagnostics=detailed,
        no_true_K_or_midi_read=True,no_model_retraining=True,
        independent_ExactK_accuracy_measured=False,S80_promoted=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('s215','s217','s18','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    run(a.s215,a.s217,a.s18,a.out)
