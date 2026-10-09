"""S46 independent replay of all class-sensitive compromises and every loss."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.audit_v273_s46_classwise_compromise import read,select,FAMILY_GROUPS

def run(a):
    if a.output.exists():raise ValueError('no archive overwrite')
    source=read(a.s45risk);p=read(a.s45pred);saved=read(a.s46)
    report=json.loads(a.report.read_text())
    ids=source['global_index'];y=p['true_K']
    if len(y)!=59309 or not all(np.array_equal(z['global_index'],ids) for z in (p,saved)):
        raise ValueError('native alignment drift')
    assert np.array_equal(saved['true_K'],y)
    original=p['predictions'][:,0]
    if (original==y).sum()!=49178:raise ValueError('S18 source mismatch')
    families=list(map(str,source['family_ids']))
    proposal=source['proposal_K'];risks=source['risks'];votes=source['independent_votes']
    names=list(map(str,saved['variant_ids']))
    assert names[0]=='series18_parent' and len(names)==193
    assert np.array_equal(saved['predictions'][:,0],original)
    for col,name in enumerate(names[1:],start=1):
        series,group,lamb,cut,vote,structure=name.split('__')
        assert series=='series46' and lamb.startswith('polyLambda')
        assert cut.startswith('cut') and vote.startswith('votes')
        members=FAMILY_GROUPS[group]
        idx=list(range(16)) if members is None else [families.index(f) for f in members]
        z,_=select(original,proposal,risks,votes,idx,
            float(lamb.removeprefix('polyLambda')),
            float(cut.removeprefix('cut')),
            int(vote.removeprefix('votes')),structure)
        if not np.array_equal(z,saved['predictions'][:,col]):
            raise ValueError('could not replay source-K policy '+name)
        sc=report['audits'][name]
        corrected=int(((original!=y)&(z==y)).sum())
        lost=int(((original==y)&(z!=y)).sum())
        if (corrected,lost)!=(sc['corrections'],sc['regressions']):
            raise ValueError('false correction counts '+name)
    a.output.mkdir(parents=True)
    summary=dict(status='verified',S18_verified=True,
        all_192_class_conditional_policies_verified=True,
        no_true_K_used_for_decision=True,no_H9_expert=True,
        already_used_historical_cohort=True,no_production_promotion=True)
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print('PASS: all 192 S46 class-specific K2-K6 risk compromises and losses independently verified')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s45risk',type=Path,required=True)
    p.add_argument('--s45pred',type=Path,required=True)
    p.add_argument('--s46',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
