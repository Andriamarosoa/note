"""S294: deploy frozen S69 fold-specific historical veto rules WITHOUT fold oracle.

Inputs are S247 original-source conditional specialist S46 predictions and
S248 original 36 PR16 policy predictions from the same native audio events.
Execute each frozen held-fold S69 veto as a separate counterfactual view,
then require at least 3 of 4 fold-rule votes to reject an intervention.
No new fitting; no chosen held-piece/fold identity; label-free. This is
a newly versioned deployment adaptation, not original S69 historical OOF.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.audit_v273_s69_s46_vote_guard import POLICIES,signals
FOLDS=(0,1,2,4)
BAD={'true_K','truth','label','labels','target','midi','correct','accuracy','y','regression'}
def need(v,msg):
    if not v:raise ValueError(msg)
def load(path,expected):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(expected) and not BAD.intersection(z.files),
             'unexpected fields or labels '+str(path))
        return {k:z[k].copy() for k in z.files}
def veto_four_folds(base,proposed,votes,rules):
    n=len(base)
    bank=signals(np.column_stack([(votes==k).sum(1) for k in range(7)]),
                 base,proposed)
    across=np.zeros((4,n),bool)
    for fi,fold in enumerate(FOLDS):
        for rule in rules:
            if int(rule['held_fold'])!=fold:continue
            src=int(rule['source']);dst=int(rule['target'])
            score=rule['score'];threshold=float(rule['threshold'])
            need(score in bank and src in range(7) and dst in range(7),
                 'frozen S69 rule unsupported or illegal')
            decision=(base==src)&(proposed==dst)&(bank[score]>=threshold)
            need(not np.any(across[fi]&decision),'original S69 overlapping fold vetoes')
            across[fi]|=decision
    stable_veto=(across.sum(axis=0)>=3)&(base!=proposed)
    adjusted=proposed.copy();adjusted[stable_veto]=base[stable_veto]
    need(np.all((adjusted==proposed)|(adjusted==base)),
         'veto illegally invented a K')
    return adjusted,across,stable_veto
def selftest():
    x=np.array([2,3,1,0],np.int8)
    p=np.array([3,2,0,1],np.int8)
    votes=np.tile(x[:,None],(1,36))
    rules=[dict(held_fold=f,source=2,target=3,
                score='base_votes',threshold=0.) for f in FOLDS[:3]]
    new,m,v=veto_four_folds(x,p,votes,rules)
    need(v.tolist()==[True,False,False,False] and new.tolist()==[2,2,0,1],
         '3-of-4 label-free folded veto')
    rules=rules[:2]
    new,_,v=veto_four_folds(x,p,votes,rules)
    need(not v.any() and np.array_equal(new,p),'2 folds must not veto')
    print('S294 PASS: 3/4 folded S69 veto, 2/4 abstention, no labels')

def run(s247file,pr16file,rulesfile,out):
    need(not out.exists(),'immutable original S69 new-piece research')
    s=load(s247file,{'row_id','variant_ids','original_freeze_K','S18_parent_K',
        'original_19_branch_folds','S46_193_policy_K','per_fold_specialist_K',
        'active_fold_count','parent_incompatible_specialists',
        'contradictory_specialists'})
    v=load(pr16file,{'member','start_samples','end_samples',
        'incumbent_freeze_K','pr16_policy_K','pr16_vote_count',
        'pr16_policy22_K','pr16_network_probabilities'})
    ids=list(map(str,s['row_id']))
    n=len(ids);need(n==506 and len(set(ids))==n,'506 original official native IDMT events')
    orig=list(map(str,v['member']))
    keys=[f'{x}:{int(y)}' for x,y in zip(orig,v['start_samples'])]
    need(len(keys)==len(set(keys)),'PR16 duplicate acoustic events')
    idx={k:i for i,k in enumerate(keys)}
    need(all(k in idx for k in ids),'PR16 cannot join 506 S247 native events')
    positions=np.array([idx[k] for k in ids],int)
    vv=v['pr16_policy_K'][positions].astype(np.int8)
    need(vv.shape==(n,36) and
         np.isin(vv,range(7)).all(),'real 36 PR16 votes on original audio')
    need(np.array_equal(v['pr16_vote_count'][positions],
         np.column_stack([(vv==k).sum(1) for k in range(7)])),
         'PR16 original 36 vote census')
    policy=s['S46_193_policy_K'];parent=s['S18_parent_K']
    names=list(map(str,s['variant_ids']))
    need(policy.shape==(n,193) and len(names)==len(set(names))==193 and
         names[0]=='series18_parent' and np.array_equal(policy[:,0],parent),
         'complete aligned S46 after source branch aggregation')
    historical=json.loads(rulesfile.read_text())
    need(set(historical['policies'])==set(POLICIES),
         'immutable historical S69 balanced and polymax rule inventory')
    result={'row_id':np.array(ids),'S18_parent_K':parent}
    metrics={}
    for label,(key,cost) in POLICIES.items():
        need(key in names,'selected original S69 source S46 policy missing '+key)
        raw=policy[:,names.index(key)]
        applied,fold_veto,veto=veto_four_folds(parent,raw,vv,
                                      historical['policies'][label]['rules'])
        result[label+'_s46_K']=raw
        result[label+'_protected_K']=applied
        result[label+'_fold_veto']=fold_veto
        result[label+'_veto_three_folds']=veto
        metrics[label]=dict(historical_rules=len(historical['policies'][label]['rules']),
          unprotected_changes=int((raw!=parent).sum()),
          vetoed_changes=int(veto.sum()),
          final_changes_vs_parent=int((applied!=parent).sum()),
          each_historical_rule_fold_veto_actions=list(map(int,fold_veto.sum(axis=1))))
    out.mkdir(parents=True)
    np.savez_compressed(out/'S294_original_S69_fourfold_rule_consensus_IDMT.npz',**result)
    report=dict(status='S294_S69_GENUINE_ORIGINAL_VETO_FOURFOLD_ON_506_NATIVE_IDMT',
        count=n,pr16_original_network_heads=12,pr16_real_policies=36,
        original_S46_policy_bank=193,
        S69_frozen_original_policy_names=list(POLICIES),
        newly_fixed_unseen_fold_policy='4 counterfactual original fold rules; 3 of 4 veto quorum; no selected held-fold',
        original_training_thresholds_unchanged=True,
        source_original_pr16_sha256=hashlib.sha256(pr16file.read_bytes()).hexdigest(),
        source_historic_S69_rules_sha256=hashlib.sha256(rulesfile.read_bytes()).hexdigest(),
        policies=metrics,
        truth_or_midi_used=False,models_retrained=False,
        independent_accuracy_measured=False,historical_S69_new_piece_identity_proven=False,
        S70_S80_replayed=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--self-test',action='store_true')
    for k in ('s247','pr16','rules','out'):
        a.add_argument('--'+k,type=Path)
    q=a.parse_args()
    if q.self_test:selftest()
    else:
        need(all(getattr(q,k) is not None for k in ('s247','pr16','rules','out')),
             'original evidence required')
        selftest()
        run(q.s247,q.pr16,q.rules,q.out)
