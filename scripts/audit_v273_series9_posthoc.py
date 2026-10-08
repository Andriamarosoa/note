"""Reproduce series9 posthoc K5 guard audit on archived series8 artifact.

The candidate was discovered after observing K5 failures on exposed data.
This is an independent computation replay, NOT independent validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require
from scripts.loop_v273_policy_gate import GLOBAL, POLY

BEST='series8__poly_parent__votes__C0.3__cost0.5'
WANTED='series9__veto_P_ge5_and_G_le3'

def evaluate(a):
    require(not a.output.exists(),'refuse overwrite')
    with np.load(a.input,allow_pickle=False) as z:
        ids=z['global_index']; y=z['true_K'];fold=z['fold']
        names=list(map(str,z['variant_ids']));matrix=z['predictions']
    require(len(ids)==len(y)==len(fold)==59309,'native cohort scope')
    require(matrix.shape[0]==len(ids) and len(set(ids.tolist()))==len(ids),
            'predictions/identity schema')
    require(set(fold.tolist())==set(FOLDS),'fold set changed')
    with np.load(a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz',
                 allow_pickle=False) as native:
        require(np.array_equal(native['native_global_index'], ids), 'baseline IDs drift')
        require(np.array_equal(native['native_truth'], y), 'baseline truth drift')
        require(np.array_equal(native['native_fold'], fold), 'baseline fold drift')
        freeze = native['native_baseline']
    require(int(np.sum(freeze==y))==48454, 'freeze target drift')
    for name in (GLOBAL,POLY,BEST):
        require(name in names,'missing parent '+name)
    get=lambda name:matrix[:,names.index(name)].astype(np.int8)
    g,p,s=get(GLOBAL),get(POLY),get(BEST)
    masks={
        'veto_P_ge5':p>=5,
        'veto_P_ge4':p>=4,
        'veto_P_ge3':p>=3,
        'veto_P_eq5':p==5,
        'veto_P_eq6':p==6,
        'veto_P_ge5_and_G_ltP':(p>=5)&(g<p),
        'veto_P_ge4_and_G_ltP':(p>=4)&(g<p),
        'veto_P_ge3_and_G_ltP':(p>=3)&(g<p),
        'veto_P_ge5_and_G_le3':(p>=5)&(g<=3),
        'veto_P_ge4_and_G_le3':(p>=4)&(g<=3),
        'veto_P_gtG':p>g,
        'veto_P_ltG':p<g,
    }
    policies={POLY:p,BEST:s}
    for name,mask in masks.items():
        policies['series9__'+name]=np.where(mask,p,s).astype(np.int8)
    require(len(policies)==14,'candidate inventory')
    reports={}
    for name,pred in policies.items():
        reports[name]=dict(metrics=metrics(y,pred),vs_freeze=paired(y,freeze,pred),vs_series8=paired(y,s,pred),
            vs_poly_parent=paired(y,p,pred),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                vs_series8=paired(y[fold==f],s[fold==f],pred[fold==f])) for f in FOLDS})
    wanted=reports[WANTED]
    require(wanted['metrics']['correct']==49162 and wanted['metrics']['poly']['correct']==2989,
            'local exploratory counts not reproduced')
    change=wanted['vs_series8']['global']
    require(change['corrections']==2 and change['regressions']==1 and change['net']==1,
            'paired audit not reproduced')
    rank=sorted(reports,key=lambda name:(
        reports[name]['metrics']['poly']['correct'],reports[name]['metrics']['correct']),reverse=True)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,fold=fold,
        variant_ids=np.asarray(list(policies)),predictions=np.column_stack(list(policies.values())))
    changes=np.flatnonzero(s!=policies[WANTED])
    row_detail=[dict(global_index=int(ids[i]),fold=int(fold[i]),true_K=int(y[i]),
        global_parent_K=int(g[i]),poly_parent_K=int(p[i]),
        series8_K=int(s[i]),series9_K=int(policies[WANTED][i]),
        outcome=('correction' if policies[WANTED][i]==y[i]
            else 'regression' if s[i]==y[i] else 'neutral')) for i in changes]
    report=dict(status='reproduced',validation_independent=False,
        posthoc_rule=True,development_exposed=True,
        policy_count=len(policies),best_exploratory=WANTED,policies=reports,
        winner_changed_cases=row_detail)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    lines=['# Series9 K5 veto — independent arithmetic replay of posthoc screen','',
        'NOT independent validation; all results derive from exposed development data.','',
        '| Policy | Correct global | Correct poly | Regressions vs freeze | Poly regressions vs freeze | Corrections vs S8 | Regressions vs S8 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in rank:
        m=reports[name]['metrics'];q=reports[name]['vs_series8']['global']
        w=reports[name]['vs_freeze']
        lines.append(f"| {name} | {m['correct']} | {m['poly']['correct']} | "
                     f"{w['global']['regressions']} | {w['poly']['regressions']} | "
                     f"{q['corrections']} | {q['regressions']} |")
    lines+=['',f'Exploratory improvement: {WANTED}',
      f"Total regressions versus freeze: {wanted['vs_freeze']['global']['regressions']} global, {wanted['vs_freeze']['poly']['regressions']} poly.",
      f"Total corrections versus freeze: {wanted['vs_freeze']['global']['corrections']} global, {wanted['vs_freeze']['poly']['corrections']} poly.",
      f"Total wrong predictions: {len(y)-wanted['metrics']['correct']} global, {int((y>=2).sum())-wanted['metrics']['poly']['correct']} poly.",
      f'Changes relative to S8: {len(changes)}; corrections 2; regressions 1; net +1.',
      'All 14 candidates preserved; none promoted.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    evaluate(p.parse_args())
