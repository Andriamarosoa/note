"""S36a: predeclared K-source/destination guards applied to S35 decisions.

No change to S35 routing or experts. No true K ever used to choose a guard.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

SOURCES=('series35__lambda1__threshold0.05',
         'series35__lambda4__threshold0')
RULES=('all_changed','source_le1','source_le2','source_le3',
       'only_K_increase','only_K_decrease','guard_poly_to_low',
       'guard_234','guard_2plus','both_le1')

def load(path):
    with np.load(path,allow_pickle=False) as q:
        return {k:q[k] for k in q.files}

def source_pred(s,name):
    options=list(map(str,s['variant_ids']))
    if name not in options:raise ValueError('missing S35 source '+name)
    return s['predictions'][:,options.index(name)]

def keep(old,new,name):
    changed=old!=new
    if name=='all_changed':return changed
    if name=='source_le1':return changed&(old<=1)
    if name=='source_le2':return changed&(old<=2)
    if name=='source_le3':return changed&(old<=3)
    if name=='only_K_increase':return changed&(new>old)
    if name=='only_K_decrease':return changed&(new<old)
    if name=='guard_poly_to_low':return changed&~((old>=2)&(new<=1))
    if name=='guard_234':return changed&~np.isin(old,[2,3,4])
    if name=='guard_2plus':return changed&(old<=1)
    if name=='both_le1':return changed&(old<=1)&(new<=1)
    raise ValueError('unknown mask')

def selftest():
    p=np.array([0,1,2,3,4,5],np.int8)
    q=np.array([1,2,1,4,3,2],np.int8)
    assert keep(p,q,'source_le1').tolist()==[True,True,False,False,False,False]
    assert keep(p,q,'guard_234').tolist()==[True,True,False,False,False,True]
    assert keep(p,q,'guard_poly_to_low').tolist()==[True,True,False,True,True,True]
    assert len(SOURCES)==2 and len(RULES)==10
    print('PASS: 20 fixed label-blind K guards; no H9 or test truth at inference')

def run(s,out):
    if out.exists():raise ValueError('no overwrite')
    y=s['true_K'];fold=s['fold'];ids=s['global_index']
    if len(y)!=59309 or set(fold.tolist())!={0,1,2,4}:
        raise ValueError('source native cohort changed')
    original=source_pred(s,'series18_parent')
    if (original==y).sum()!=49178:raise ValueError('S18 reference altered')
    chosen={'series18_parent':original.copy()}
    for name in SOURCES:
        base=source_pred(s,name)
        chosen[name]=base.copy()
        for rule in RULES:
            permitted=keep(original,base,rule)
            chosen[f'series36a__{name}__{rule}']=np.where(
                permitted,base,original).astype(np.int8)
    measures={}
    poly=(y>=2)
    for name,pred in chosen.items():
        c=(original!=y)&(pred==y)
        r=(original==y)&(pred!=y)
        n=(original!=y)&(pred!=y)&(original!=pred)
        measures[name]=dict(
            correct=int((pred==y).sum()),
            exact_global=float((pred==y).mean()),
            exact_poly=float((pred[poly]==y[poly]).mean()),
            corrections=int(c.sum()),regressions=int(r.sum()),
            neutral=int(n.sum()),net=int(c.sum()-r.sum()),
            by_true_K={str(k):dict(correct=int(((y==k)&(pred==y)).sum()),
                fixed=int(((y==k)&c).sum()),
                lost=int(((y==k)&r).sum())) for k in range(7)},
            by_original_K={str(k):dict(corrections=int(((original==k)&c).sum()),
                regressions=int(((original==k)&r).sum())) for k in range(7)},
            by_fold={str(k):dict(fixed=int(((fold==k)&c).sum()),
                lost=int(((fold==k)&r).sum())) for k in (0,1,2,4)})
    winners=[k for k in chosen if k.startswith('series36a__')
        and measures[k]['corrections']>0 and
        measures[k]['regressions']==0 and
        measures[k]['exact_poly']>=2998/7385]
    ranked=sorted(measures,key=lambda k:(measures[k]['correct'],
        measures[k]['exact_poly']),reverse=True)
    out.mkdir(parents=True)
    report=dict(status='completed',predeclared_label_blind_masks=True,
        H9_excluded=True,no_models_refitted=True,
        historical_S18_unchanged=True,unseen_music_validation=False,
        no_automatic_promotion=True,strict_zero_loss_candidates=winners,
        best_global=ranked[0],all_policies=measures)
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(out/'predictions.npz',global_index=ids,
        true_K=y,fold=fold,variant_ids=np.asarray(list(chosen)),
        predictions=np.column_stack(list(chosen.values())))
    lines=['# S36a: polyphonic K-source guards on S35, H9 excluded','',
        '22 policies tested; original S35 outputs reused unchanged.',
        'Labels are for paired corrections/regressions only, never masks.',
        '| Policy | Global | Poly | Fix | Break | Net |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ranked:
        z=measures[name]
        lines.append(f"| {name} | {z['exact_global']*100:.4f}% | "
             f"{z['exact_poly']*100:.4f}% | {z['corrections']} | "
             f"{z['regressions']} | {z['net']} |")
    lines.extend(['',f'Strict zero-loss candidates: {len(winners)}',
        'Cohort already used for many experiments; this is exploratory.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s35',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        if not a.s35 or not a.output:raise ValueError('source and output mandatory')
        run(load(a.s35),a.output)
