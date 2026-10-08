"""S22 — consensus of recalculated A states and out-of-piece B trust gates.

No new model fit and no label-informed decoding; evaluates a fixed policy inventory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require
def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

PASS_ID=dict(A1=0,ABA2=1,ABA4=3,A4_noB=4)
FAMILIES={
    'A1_ABA2':('A1','ABA2'),
    'ABA2_ABA4':('ABA2','ABA4'),
    'ABA4_noB4':('ABA4','A4_noB'),
    'A1_ABA2_ABA4':('A1','ABA2','ABA4'),
    'ABA2_ABA4_Bsupports':('ABA2','ABA4'),
    'all4':('A1','ABA2','ABA4','A4_noB'),
}
RISK_MODES=('logistic_AND_hgb','logistic_AND_noB_hgb','average_models')
CUTS=(.30,.40,.50,.60,.70,.80)

def decoded(parent,A,B,risk):
    require(A.ndim==3 and A.shape[1:]==(5,7),'A shape')
    require(B.ndim==3 and B.shape[1:]==(4,7),'B shape')
    n=len(parent)
    require(len(A)==n==len(B),'A/B identity')
    candidates={name:A[:,idx].argmax(1).astype(np.int8)
                for name,idx in PASS_ID.items()}
    out={'series18_parent':parent.copy()}
    metadata={}
    ii=np.arange(n)
    for fam,members in FAMILIES.items():
        chosen=candidates[members[-1]].copy()
        agrees=(chosen!=parent)
        for name in members:
            agrees &= (candidates[name]==chosen)
        if fam=='ABA2_ABA4_Bsupports':
            agrees&=(B[ii,3,chosen]>B[ii,3,parent])
        for mode in RISK_MODES:
            scores=[]
            for name in members:
                if mode=='logistic_AND_hgb':
                    keys=(f'with_B_logistic_{name}',f'with_B_hgb_{name}')
                elif mode=='logistic_AND_noB_hgb':
                    keys=(f'with_B_logistic_{name}',
                          f'without_B_features_hgb_{name}')
                else:
                    keys=(f'with_B_logistic_{name}',f'with_B_hgb_{name}')
                for key in keys:
                    require(key in risk,'missing prelearned trust '+key)
                    scores.append(np.nan_to_num(
                        risk[key].astype(np.float32),nan=-1.0))
            array=np.stack(scores)
            trust=np.mean(array,axis=0) if mode=='average_models' else np.min(array,axis=0)
            for c in CUTS:
                key=f'series22__{fam}__{mode}__p>{c:g}'
                change=agrees&(trust>c)
                out[key]=np.where(change,chosen,parent).astype(np.int8)
                metadata[key]=dict(changed=int(change.sum()),agreement=int(agrees.sum()))
    require(len(out)==109,'108 consensus policies plus parent')
    return out,metadata


def selftest():
    n=8;parent=np.arange(n,dtype=np.int8)%7
    A=np.zeros((n,5,7),np.float32)
    for t in range(5):
        A[np.arange(n),t,(parent+1)%7]=1.
    B=np.ones((n,4,7),np.float32)*.3
    B[np.arange(n),3,(parent+1)%7]=.9
    risk={f'{view}_{algo}_{name}':np.ones(n,np.float32)*.95
          for view in ('with_B','without_B_features')
          for algo in ('logistic','hgb') for name in PASS_ID}
    out,meta=decoded(parent,A,B,risk)
    assert len(out)==109 and len(meta)==108
    for key,vec in out.items():
        if key=='series18_parent':continue
        assert np.array_equal(vec,(parent+1)%7)
    B[np.arange(n),3,(parent+1)%7]=.1
    out,_=decoded(parent,A,B,risk)
    assert np.array_equal(out[
        'series22__ABA2_ABA4_Bsupports__average_models__p>0.8'],parent)
    print('PASS: 108 A↔B consensus gates, no true-K access, B support reversible')


def audit(a):
    require(not a.output.exists(),'cannot overwrite prior consensus audit')
    d=read(a.s20)
    ids=d['global_index'];y=d['true_K'];fold=d['fold']
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'cohort drift')
    variants=list(map(str,d['variant_ids']))
    require('series18_parent' in variants,'series18 absent in S20')
    parent=d['predictions'][:,variants.index('series18_parent')]
    require(metrics(y,parent)['correct']==49178 and
            metrics(y,parent)['poly']['correct']==2998,
            'parent preservation drift')
    freeze=d['predictions'][:,variants.index('freeze_parent')]
    Adata=read(a.s20_prob)
    A=np.asarray(Adata['A_probs'],np.float32)
    B=np.asarray(Adata['B_compatibility'],np.float32)
    require(np.array_equal(Adata['global_index'],ids),'S20 source alignment drift')
    gate=read(a.s21)
    require(np.array_equal(gate['global_index'],ids),'crosspiece trust ID drift')
    scores={k:gate['probabilities'][:,j] for j,k
            in enumerate(map(str,gate['variant_ids']))}
    out,decisions=decoded(parent,A,B,scores)
    out['freeze_reference']=freeze.copy()
    require(len(out)==110,'110 policies with freeze')
    reports={}
    for key,pred in out.items():
        pair=paired(y,parent,pred)
        neu=int(np.sum((parent!=pred)&(parent!=y)&(pred!=y)))
        reports[key]=dict(metrics=metrics(y,pred),versus_parent=pair,
            versus_freeze=paired(y,freeze,pred),
            changed=int((parent!=pred).sum()),neutral=neu,
            decision=decisions.get(key),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                vs_parent=paired(y[fold==f],parent[fold==f],pred[fold==f]))
                for f in FOLDS})
    ranked=sorted(reports,key=lambda k:(reports[k]['metrics']['correct'],
        reports[k]['metrics']['poly']['correct']),reverse=True)
    candidates=[name for name in out if name.startswith('series22__') and
        reports[name]['versus_parent']['global']['corrections']>0 and
        reports[name]['versus_parent']['global']['regressions']==0 and
        reports[name]['metrics']['poly']['correct']>=2998]
    result=dict(status='completed',independent_validation=False,
        posthoc_exposed_development_data=True,promotion=False,
        source_training_run_S20=37858383973,
        source_training_run_S21=37859454327,
        all_decisions_label_free=True,
        no_new_labels_in_fit=True,
        policy_count=len(out),policies=reports,
        strict_no_loss_candidates=candidates,
        best_global=ranked[0])
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,
        true_K=y,fold=fold,variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    lines=['# S22 — consensus A ↔ B ↔ A and learned trust','',
        'All 108 fixed policies are exploratory, not an independently unseen evaluation.',
        '', '| Candidate | Exact global | Exact poly | Fixed vs S18 | Broken vs S18 | Neutral | Remaining regressions |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ranked:
        o=reports[name]
        m=o['metrics'];x=o['versus_parent']['global'];f=o['versus_freeze']['global']
        lines.append(f"| {name} | {m['exact']*100:.4f}% | {m['poly']['exact']*100:.4f}% | "
            f"{x['corrections']} | {x['regressions']} | {o['neutral']} | "
            f"{f['regressions']} |")
    lines+=['',f"Best global: {ranked[0]}",
        f"Strict zero-loss candidates: {len(candidates)}",
        'No training or evaluation label affects any of the 108 decisions.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s20',type=Path)
    p.add_argument('--s20-prob',type=Path)
    p.add_argument('--s21',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.s20 and a.s20_prob and a.s21 and a.output,'source arguments')
        audit(a)
