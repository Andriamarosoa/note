"""Independent decoding replay of all 24 fixed K0 birth vetoes; no model refit."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

CUTOFFS=(.85,.90,.95,.975)
HEADS=('summary_logistic','summary_hgb','flow_logistic','flow_hgb','flow_both','flow_mean')
SAFE='series13__flow_logistic__p0_gt0.975'
SECOND='series13__flow_logistic__p0_gt0.9'

def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def replay(a):
    require(not a.output.exists(),'never overwrite verified native audit')
    old=read(a.input/'predictions.npz')
    scores=read(a.input/'probabilities.npz')
    report=json.loads((a.input/'report.json').read_text())
    ids,y,fold=(old[k] for k in ('global_index','true_K','fold'))
    names=list(map(str,old['variant_ids']))
    matrix=old['predictions']
    prob_ids=list(map(str,scores['heads']))
    require(len(ids)==59309 and matrix.shape==(len(ids),26),'prediction shape')
    require(np.array_equal(scores['global_index'],ids),'score identity drift')
    require(set(fold.tolist())==set(FOLDS),'held fold drift')
    require(set(prob_ids)==set(HEADS),'head inventory changed')
    require(report['status']=='completed' and not report['validation_independent'],
            'missing development safety statement')
    from pathlib import Path
    with np.load(a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz',
                 allow_pickle=False) as f:
        require(np.array_equal(f['native_global_index'],ids),'reference identity drift')
        require(np.array_equal(f['native_truth'],y),'reference truth drift')
        require(np.array_equal(f['native_fold'],fold),'reference fold drift')
        b=f['native_baseline']
    s9=matrix[:,names.index('series9_parent')]
    relevant=(b==0)&(s9!=0)
    require(np.array_equal(scores['relevant'],relevant),'native eligibility mismatch')
    score=np.asarray(scores['probabilities'],dtype=np.float64)
    require(score.shape==(len(ids),6),'probability dimension drift')
    replayed={}
    for head in HEADS:
        h=score[:,prob_ids.index(head)]
        require(np.isfinite(h[relevant]).all(),'nonfinite score '+head)
        for c in CUTOFFS:
            name=f'series13__{head}__p0_gt{c:g}'
            pred=np.where(relevant&(h>c),b,s9).astype(np.int8)
            require(name in names,'missing candidate '+name)
            require(np.array_equal(pred,matrix[:,names.index(name)]),
                    'native decision discrepancy: '+name)
            replayed[name]=pred
    require(len(replayed)==24 and len(names)==26,'variant count')
    for x in report['records']:
        if 'model_file' in x:
            require(x['piece'] not in x['train_pieces'],'fit uses evaluated piece')
            path=a.input/'models'/x['model_file']
            require(path.is_file(),'missing model '+str(path))
    files={}
    for p in (a.input/'models').glob('*.joblib'):
        hasher=hashlib.sha256()
        with p.open('rb') as fd:
            for chunk in iter(lambda:fd.read(1024*1024),b''):
                hasher.update(chunk)
        files[p.name]=hasher.hexdigest()
    require(len(files)==76,'expected model set not archived')
    safe=replayed[SAFE]
    m=metrics(y,safe)
    c=paired(y,s9,safe)
    f=paired(y,b,safe)
    require(m['correct']==49165 and m['poly']['correct']==2989,
            'safe candidate accuracy mismatches CI')
    require(c['global']['corrections']==3 and c['global']['regressions']==0,
            'zero correction loss assertion failed')
    require(f['global']['corrections']==2934 and f['global']['regressions']==2223,
            'freeze paired regression invariant')
    details={}
    for tag in (SAFE,SECOND):
        pred=replayed[tag]
        fixed=np.flatnonzero((s9!=y)&(pred==y))
        broken=np.flatnonzero((s9==y)&(pred!=y))
        cases=[]
        for i in np.flatnonzero(s9!=pred):
            cases.append(dict(global_index=int(ids[i]),fold=int(fold[i]),
                true_K=int(y[i]),freeze_K=int(b[i]),series9_K=int(s9[i]),
                new_K=int(pred[i]),p0=float(score[i,prob_ids.index('flow_logistic')]),
                effect='correction' if i in set(fixed.tolist()) else
                       'regression' if i in set(broken.tolist()) else 'neutral'))
        details[tag]=dict(metrics=metrics(y,pred),vs_s9=paired(y,s9,pred),
            vs_freeze=paired(y,b,pred),changed_rows=cases,
            folds={str(k):paired(y[fold==k],s9[fold==k],pred[fold==k])
                   for k in FOLDS})
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(dict(status='replayed',
        not_independent_validation=True,not_a_new_model=True,
        all_24_probabilistic_gates_bitwise_reproduced=True,
        full_models_sha256=files,safe_variant=SAFE,proof=details),
        indent=2,sort_keys=True)+'\n')
    for tag in (SAFE,SECOND):
        with (a.output/('cases-'+tag.replace('.','_')+'.csv')).open('w',newline='') as fd:
            records=details[tag]['changed_rows']
            if not records:continue
            writer=csv.DictWriter(fd,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    lines=['# Native series13 regression proof replay','',
        'All 24 policies regenerated from stored probabilities with EXACT equality.',
        f'Fitted models checked by SHA256: {len(files)}.',
        'No independent validation; all event labels were previously exposed.',
        '',
        '| Policy | New corrections | New regressions | Remaining regressions vs freeze | Corrections preserved vs freeze |',
        '|---|---:|---:|---:|---:|']
    for name,d in details.items():
        c=d['vs_s9']['global'];f=d['vs_freeze']['global']
        lines.append(f"| {name} | {c['corrections']} | {c['regressions']} | {f['regressions']} | {f['corrections']} |")
    lines+=['','## Exact recovered native events for zero-loss candidate','',
        '| ID | Fold | True K | Freeze | Series9 | Series13 | p(K0) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for z in details[SAFE]['changed_rows']:
        lines.append(f"| {z['global_index']} | {z['fold']} | {z['true_K']} | {z['freeze_K']} | {z['series9_K']} | {z['new_K']} | {z['p0']:.6f} |")
    lines+=['','No training/labels used to make the corrections on the held pieces.',
        'These three descriptive cases do not establish generalization.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    replay(p.parse_args())
