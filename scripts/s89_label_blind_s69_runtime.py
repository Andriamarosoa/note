"""S89: label-free S69 fixed veto rules applied to original S46 + PR16 votes.

Rules learned historically and pinned in S69 report; this module never
reads labels or refits thresholds. Fold keys 0,1,2,4 are ONLY valid for
historical replay, NOT for an unseen audio recording (no fold assignment).
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.audit_v273_s69_s46_vote_guard import POLICIES,signals
FOLDS={0,1,2,4};FORBIDDEN={'true_K','truth','label','labels','y','target'}
def check(c,m):
    if not c:raise ValueError(m)
def strict(path,keys):
    with np.load(path,allow_pickle=False) as z:
        check(set(z.files)==set(keys) and not FORBIDDEN.intersection(z.files),
              'source must be strictly label-free '+str(path))
        return {k:z[k].copy() for k in z.files}
def run(s46file,pr16file,rulesfile,out):
    check(not out.exists(),'refuse overwrite')
    s=strict(s46file,('row_id','fold','variant_ids','predictions'))
    v=strict(pr16file,('row_id','predictions'))
    ids=s['row_id'];f=s['fold'];n=len(ids)
    check(n>0 and len(np.unique(ids))==n and set(f)==FOLDS,
          'only historic 0/1/2/4 fold identities can replay existing rules')
    check(np.array_equal(ids,v['row_id']),'PR16 vote alignment')
    p=s['predictions'];votes=v['predictions']
    check(p.shape==(n,193) and votes.shape==(n,36),'policy census')
    names=list(map(str,s['variant_ids']))
    check(names[0]=='series18_parent' and len(set(names))==193,'S46 names')
    report=json.loads(Path(rulesfile).read_text())
    check(set(report['policies'])==set(POLICIES),'frozen S69 rule inventory')
    base=p[:,0]
    count=np.stack([(votes==k).sum(1) for k in range(7)],1)
    outputs={'row_id':ids,'fold':f,'s18_K':base}
    detail={}
    for kind,(policy,cost) in POLICIES.items():
        check(policy in names,'S46 source policy absent: '+policy)
        original=p[:,names.index(policy)]
        bank=signals(count,base,original)
        applied=original.copy()
        rules=report['policies'][kind]['rules']
        changed=np.zeros(n,bool)
        for rule in rules:
            held=int(rule['held_fold'])
            source=int(rule['source']);target=int(rule['target'])
            val=rule['score'];threshold=float(rule['threshold'])
            check(held in FOLDS and val in bank,'unexpected S69 frozen rule')
            decision=(f==held)&(base==source)&(original==target)&(bank[val]>=threshold)
            check(not np.any(changed&decision),'double veto on same event')
            changed |= decision
            applied[decision]=base[decision]
        outputs[kind+'_s46_K']=original
        outputs[kind+'_protected_K']=applied
        detail[kind]={'rules':len(rules),'veto_actions':int(changed.sum())}
    out.mkdir(parents=True)
    np.savez_compressed(out/'predictions.npz',**outputs)
    summary={'status':'S69_FROZEN_RULES_OOF_EXACT_INFERENCE',
             'no_truth_labels_read':True,'row_count':n,'policies':detail,
             'independent_validation':False,'new_audio_supported':False,
             'reason':'Frozen S69 rule sets indexed by historical held folds; new music needs a distinct deployment policy'}
    (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s46','pr16','rules','output'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();run(a.s46,a.pr16,a.rules,a.output)
