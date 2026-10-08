"""Reproducible global K0–K6 audit of every series9 correction and regression."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

WANTED='series9__veto_P_ge5_and_G_le3'

def analyze(a):
    require(not a.output.exists(),'never overwrite audit')
    with np.load(a.input,allow_pickle=False) as z:
        ids=z['global_index']; y=z['true_K'];fold=z['fold']
        variants=list(map(str,z['variant_ids']))
        require(WANTED in variants,'missing series9')
        p=z['predictions'][:,variants.index(WANTED)]
    with np.load(a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz',
                 allow_pickle=False) as z:
        for key,target in [('native_global_index',ids),('native_truth',y),('native_fold',fold)]:
            require(np.array_equal(z[key],target),'native alignment: '+key)
        b=z['native_baseline']; eligible=z['native_position']
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'native cohort changed')
    q=paired(y,b,p)
    require(q['global']['corrections']==2934 and q['global']['regressions']==2226,'pairwise total changed')
    require(q['poly']['corrections']==1336 and q['poly']['regressions']==877,'poly total changed')
    changed=b!=p; fix=(b!=y)&(p==y);reg=(b==y)&(p!=y)
    def counter(mask):
        d=Counter()
        for i in np.flatnonzero(mask):
            d[(int(b[i]),int(p[i]))]+=1
        return [{'from_K':int(k[0]),'to_K':int(k[1]),'count':int(v)}
                for k,v in sorted(d.items(),key=lambda z:(-z[1],z[0]))]
    masks={}
    masks['by_true_k']={str(k):{'rows':int((y==k).sum()),
        'corrections':int((fix&(y==k)).sum()),'regressions':int((reg&(y==k)).sum()),
        'net':int((fix&(y==k)).sum()-(reg&(y==k)).sum()),
        'other_changed':int((changed&~fix&~reg&(y==k)).sum())}
        for k in range(7)}
    masks['by_fold']={str(k):{'rows':int((fold==k).sum()),
        'corrections':int((fix&(fold==k)).sum()),'regressions':int((reg&(fold==k)).sum()),
        'poly_corrections':int((fix&(fold==k)&(y>=2)).sum()),
        'poly_regressions':int((reg&(fold==k)&(y>=2)).sum())}
        for k in FOLDS}
    masks['by_eligible']={}
    eligible_mask=np.zeros(len(y),bool);eligible_mask[eligible]=True
    for name,m in [('baseline_K234_eligible',eligible_mask),('outside_K234',~eligible_mask)]:
        masks['by_eligible'][name]={
            'rows':int(m.sum()),'corrections':int((fix&m).sum()),
            'regressions':int((reg&m).sum()),
            'net':int((fix&m).sum()-(reg&m).sum())}
    masks['regression_transitions']=counter(reg)
    masks['correction_transitions']=counter(fix)
    reports=dict(status='completed',independent_validation=False,
        target=WANTED,records=59309,metrics=metrics(y,p),paired=q,
        signatures=masks)
    a.output.mkdir(parents=True)
    with (a.output/'changed-cases.csv').open('w',newline='') as fd:
        w=csv.writer(fd)
        w.writerow(['global_id','fold','true_K','freeze_K','series9_K','outcome'])
        for i in np.flatnonzero(changed):
            w.writerow([int(ids[i]),int(fold[i]),int(y[i]),int(b[i]),int(p[i]),
                'correction' if fix[i] else 'regression' if reg[i] else 'neither'])
    (a.output/'report.json').write_text(json.dumps(reports,indent=2,sort_keys=True)+'\n')
    lines=['# Native series9 regression taxonomy','',
        'Research cohort already exposed; no new model and no independent validation.',
        '', '| True K | Corrections | Regressions | Net |',
        '|---:|---:|---:|---:|']
    for k,r in masks['by_true_k'].items():
        lines.append(f"| {k} | {r['corrections']} | {r['regressions']} | {r['net']:+d} |")
    lines+=['', '| Fold | Corrections | Regressions | Poly regressions |',
        '|---:|---:|---:|---:|']
    for k,r in masks['by_fold'].items():
        lines.append(f"| {k} | {r['corrections']} | {r['regressions']} | {r['poly_regressions']} |")
    lines+=['','## Eligible vs unaddressed',
        json.dumps(masks['by_eligible'],sort_keys=True),
        '','## Main regression directions','',
        '| Freeze K | Proposed K | Regressions |','|---:|---:|---:|']
    for x in masks['regression_transitions'][:20]:
        lines.append(f"| {x['from_K']} | {x['to_K']} | {x['count']} |")
    lines+=['','Global corrections: 2934; global regressions: 2226.',
        'Poly corrections: 1336; poly regressions: 877.',
        'All changed cases persisted in changed-cases.csv.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args())
