"""S56 frozen-vote consensus for K0/K1/K5 with optional S54 K2 rescue.

Reuses 36 pre-trained PR16 policies; they are correlated policies, NOT 36
independent heads. Research/development folds; no independent promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.yourmt3_exactk_common import metrics, paired, require

FOLDS=(0,1,2,4)
SOURCES=(0,1,5)
THRESHOLDS=(24,28,30,32,34,36)


def joint_consensus(proposals, baseline, s54, threshold, rescue_k2):
    """Inference-only: never reads ground-truth labels or recording IDs."""
    proposals=np.asarray(proposals)
    baseline=np.asarray(baseline)
    s54=np.asarray(s54)
    require(proposals.shape==(len(baseline),36) and s54.shape==baseline.shape,
            "36-policy consensus requires complete native rows")
    require(24<=threshold<=36 and np.isin(baseline,np.arange(7)).all(),
            "invalid consensus threshold or frozen class")
    agreement=np.stack([(proposals==k).sum(1) for k in range(7)],axis=1)
    agreement[np.arange(len(baseline)),baseline]=0
    offer=np.argmax(agreement,axis=1)
    support=agreement[np.arange(len(baseline)),offer]
    eligible=np.isin(baseline,SOURCES)
    accept=eligible&(offer!=baseline)&(support>=threshold)
    result=baseline.copy()
    result[accept]=offer[accept]
    if rescue_k2:
        # Restore S54's K2 proposal only where consensus otherwise keeps.
        supplement=eligible&(result==baseline)&(s54==2)&(s54!=baseline)
        result[supplement]=2
    require(np.array_equal(result[~eligible],baseline[~eligible]),
            "baseline outside K0/K1/K5 modified")
    return result


def load(source, critic):
    with np.load(source,allow_pickle=False) as a:
        ids,y,b,fold,proposals,labels=(a[k] for k in
            ('global_index','true_K','baseline_K','fold','predictions','variant_ids'))
        archive={k:np.array(x) for k,x in zip(
            ('ids','y','b','fold','proposals','labels'),
            (ids,y,b,fold,proposals,labels))}
    with np.load(critic,allow_pickle=False) as z:
        cid,cy,cb,cf,cp=(z[k] for k in
            ('global_index','true_K','frozen_baseline_K','fold','predicted_K'))
    require(len(archive['ids'])==59309 and len(archive['labels'])==36,
            "native size or policy count mismatch")
    order=np.argsort(cid)
    ix=np.searchsorted(cid[order],archive['ids'])
    require((ix<len(order)).all() and
            np.array_equal(cid[order[ix]],archive['ids']) and
            np.array_equal(cy[order[ix]],archive['y']) and
            np.array_equal(cb[order[ix]],archive['b']) and
            np.array_equal(cf[order[ix]],archive['fold']),
            "S54 versus PR16 native alignment failed")
    archive['s54']=cp[order[ix]]
    require(np.isin(archive['fold'],FOLDS).all() and
            np.isin(archive['b'],np.arange(7)).all(),
            "forbidden fold or K")
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pr16',type=Path,required=True)
    parser.add_argument('--s54',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    require(not a.output.exists(),"cannot overwrite research evidence")
    d=load(a.pr16,a.s54)
    ids,y,b,fold=(d[k] for k in ('ids','y','b','fold'))
    baseline=metrics(y,b)
    outcomes={}
    produced={'global_index':ids,'true_K':y,'frozen_baseline_K':b,'fold':fold,
              's54_K':d['s54']}
    for threshold in THRESHOLDS:
        for rescue in (False,True):
            name=f'consensus{threshold}'+('_plus_K2' if rescue else '')
            p=joint_consensus(d['proposals'],b,d['s54'],threshold,rescue)
            paired_result=paired(y,b,p)
            outcomes[name]=dict(metrics=metrics(y,p),paired=paired_result,
                folds={str(f):paired(y[fold==f],b[fold==f],p[fold==f])['global']
                       for f in FOLDS},
                source={str(k):paired(y[b==k],b[b==k],p[b==k])['global']
                        for k in SOURCES})
            produced[name+'_K']=p.astype(np.int8)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'predictions.npz',**produced)
    report=dict(status='completed',independent_validation=False,
        automatic_promotion=False,thresholds=list(THRESHOLDS),
        source_baselines=list(SOURCES),baseline=baseline,
        archived_pr16_sha256=hashlib.sha256(a.pr16.read_bytes()).hexdigest(),
        archived_s54_sha256=hashlib.sha256(a.s54.read_bytes()).hexdigest(),
        policies=outcomes,
        limitations=['36 policies are correlated and do not represent 36 independent heads',
            'Thresholds explored on historically exposed folds: no unbiased threshold selection',
            'PR16 producer lineage is not nested for a newly trained routing meta-model',
            'S54 was cross-fitted on the same exposed folds',
            'Never apply any candidate automatically'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    lines=['# S56 consensus (development only)','',
        '36 frozen correlated policies, candidate baselines K0/K1/K5 only.',
        'All 59,309 native events scored; no independent validation or promotion.',
        '',
        '| Policy | Fixes | Regressions | Net | K2+ correct |',
        '|---|---:|---:|---:|---:|']
    for name,v in outcomes.items():
        p=v['paired']['global']
        lines.append(f"| {name} | {p['corrections']} | {p['regressions']} | "
                     f"{p['net']:+d} | {v['metrics']['poly']['correct']} |")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
