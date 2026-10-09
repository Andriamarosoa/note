"""Reconcile all regression fixes: auditable OR/AND series13/14 combinations."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

S13_SAFE='series13__flow_logistic__p0_gt0.975'
S14_SAFE='series14__flow_mean__p0_gt0.9'
S13_WIDE='series13__flow_logistic__p0_gt0.9'
S14_WIDE='series14__flow_logistic__p0_gt0.9'

def get(z,name):
    variants=list(map(str,z['variant_ids']))
    require(name in variants,'missing policy '+name)
    return np.asarray(z['predictions'][:,variants.index(name)],dtype=np.int8)

def read(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def audit(a):
    require(not a.output.exists(),'refuse to overwrite previous experiment')
    u=read(a.series13);v=read(a.series14)
    d=read(a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=(d[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold'))
    for name,z in [('series13',u),('series14',v)]:
        for key,expect in [('global_index',ids),('true_K',y),('fold',fold)]:
            require(np.array_equal(z[key],expect),name+' cohort alignment '+key)
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'native size/fold drift')
    s9=get(u,'series9_parent')
    require(np.array_equal(s9,get(v,'series9_parent')),'series9 parent mismatch')
    require(metrics(y,s9)['correct']==49162,'reference scoring invariant')
    safe13=get(u,S13_SAFE);safe14=get(v,S14_SAFE)
    wide13=get(u,S13_WIDE);wide14=get(v,S14_WIDE)
    require(paired(y,s9,safe13)['global']['corrections']==3 and
        paired(y,s9,safe13)['global']['regressions']==0,'source 13 safety drift')
    require(paired(y,s9,safe14)['global']['corrections']==4 and
        paired(y,s9,safe14)['global']['regressions']==0,'source 14 safety drift')
    # Both source gates are 'rollback-only': their actions can only select the
    # original parent decision or freeze. No label or K truth enters decoding.
    policies={'series9_parent':s9.copy(),
              'series13_safe':safe13,'series14_safe':safe14,
              'series13_wide':wide13,'series14_wide':wide14}
    for suffix,x,z in [('safe',safe13,safe14),('wide',wide13,wide14)]:
        xx=x!=s9;zz=z!=s9
        require(np.array_equal(x,np.where(xx,b,s9)),'series13 rollback lineage')
        require(np.array_equal(z,np.where(zz,b,s9)),'series14 rollback lineage')
        policies[f'series15__{suffix}_OR']=np.where(xx|zz,b,s9).astype(np.int8)
        policies[f'series15__{suffix}_AND']=np.where(xx&zz,b,s9).astype(np.int8)
    require(len(policies)==9,'expected nine frozen policy vectors')
    reports={}
    for name,pred in policies.items():
        reports[name]=dict(metrics=metrics(y,pred),versus_series9=paired(y,s9,pred),
            versus_freeze=paired(y,b,pred),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                paired=paired(y[fold==f],s9[fold==f],pred[fold==f]))
                for f in FOLDS})
    union=policies['series15__safe_OR']
    changed13=set(np.flatnonzero(safe13!=s9).tolist())
    changed14=set(np.flatnonzero(safe14!=s9).tolist())
    common=sorted(changed13&changed14)
    exclusive13=sorted(changed13-changed14)
    exclusive14=sorted(changed14-changed13)
    explain=dict(source13_count=len(changed13),source14_count=len(changed14),
        intersection_count=len(common),
        only_series13=len(exclusive13),only_series14=len(exclusive14),
        union_count=len(changed13|changed14))
    if all(paired(y,s9,s)['global']['regressions']==0
        for s in (safe13,safe14)):
        require(paired(y,s9,union)['global']['regressions']==0,
                'union must not lose any correct decision')
    rows=[]
    for i in np.flatnonzero(union!=s9):
        outcome=('correction' if (union[i]==y[i] and s9[i]!=y[i]) else
                 'regression' if (s9[i]==y[i] and union[i]!=y[i]) else
                 'neutral')
        rows.append(dict(global_index=int(ids[i]),fold=int(fold[i]),
            true_K=int(y[i]),freeze_K=int(b[i]),
            parent_K=int(s9[i]),joint_K=int(union[i]),
            source13=bool(i in changed13),source14=bool(i in changed14),
            result=outcome))
    a.output.mkdir(parents=True)
    with (a.output/'safe-union-changed-cases.csv').open('w',newline='') as f:
        fields=['global_index','fold','true_K','freeze_K','parent_K',
                'joint_K','source13','source14','result']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,
        fold=fold,variant_ids=np.asarray(list(policies)),
        predictions=np.column_stack(list(policies.values())))
    summary=dict(status='completed',cohort_previously_exposed=True,
        candidate_selection_posthoc=True,independent_validation=False,
        automatic_promotion=False,source_runs=[37854384791,37855070118],
        overlap=explain,policies=reports,
        safe_combined_correct_event_ids=[r['global_index'] for r in rows
            if r['result']=='correction'],
        no_correction_lost=reports['series15__safe_OR']['versus_series9']['global']['regressions']==0)
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    lines=['# Series15 — conservation OR/AND, full native replay','',
        'Two independently trained heads; union policy selected after exposed-data audits.',
        '**Not an independent validation or production policy.**','',
        '| Variant | Global exact | Poly exact | Regressions corrected | Corrections lost | Regressions remaining vs freeze |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in sorted(reports.items(),key=lambda q:-q[1]['metrics']['correct']):
        m=r['metrics'];p=r['versus_series9']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
                     f"{100*m['poly']['exact']:.4f}% | "
                     f"{p['corrections']} | {p['regressions']} | "
                     f"{r['versus_freeze']['global']['regressions']} |")
    lines+=['',f'Safe overlap: {explain}', '',
        '| Event ID | Fold | True K | Before | After | Source13 | Source14 |',
        '|---:|---:|---:|---:|---:|---|---|']
    for c in rows:
        lines.append(f"| {c['global_index']} | {c['fold']} | {c['true_K']} | "
            f"{c['parent_K']} | {c['joint_K']} | "
            f"{c['source13']} | {c['source14']} |")
    lines+=['','All variants and all row-level joint changes archived, never overwrite.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--series13',type=Path,required=True)
    p.add_argument('--series14',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())
