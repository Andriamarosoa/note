"""S39 diagnostic only: S38 true-novel correction complementarity versus S18.

Oracle upper bounds are NEVER policies nor allowed as model scores.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

KEYS=('raw335__hgb','raw335__extra_trees',
      'morph825__hgb','morph825__extra_trees')

def read(p):
    with np.load(p,allow_pickle=False) as q:return {k:q[k] for k in q.files}

def run(a):
    if a.output.exists():raise ValueError('existing S39 report')
    p=read(a.pred)
    q=read(a.probs)
    truth=p['true_K'];n=len(truth)
    ids=p['global_index']
    if n!=59309 or not np.array_equal(ids,q['global_index']):
        raise ValueError('wrong native cohort')
    variants=list(map(str,p['variant_ids']))
    if variants[0]!='series18_parent':
        raise ValueError('S18 benchmark moved')
    parent=p['predictions'][:,0]
    proba=q['poly_probs']
    if proba.shape!=(n,4,7) or list(map(str,q['variant_ids']))!=list(KEYS):
        raise ValueError('not four S38 standalone experts')
    correct_parent=(parent==truth)
    poly=truth>=2
    assert correct_parent.sum()==49178 and (correct_parent&poly).sum()==2998
    report={'status':'completed','diagnostic_only_no_runtime_oracle':True,
        'all_models_exclude_their_test_fold':True,
        'H9_not_a_model_input':True,
        'already_used_development_compositions':True,
        'standalone':{},'oracle_upper_bound':{},
        'no_promotion':True}
    successes=np.zeros((len(truth),len(KEYS)),bool)
    for j,key in enumerate(KEYS):
        candidate=proba[:,j].argmax(axis=1)
        correct=candidate==truth
        successes[:,j]=correct
        groups={}
        for k,mask in [('all',np.ones(n,bool)),
                       ('poly',poly),
                       *[(f'K{i}',truth==i) for i in range(7)],
                       *[(f'fold{f}',p['fold']==f) for f in (0,1,2,4)]]:
            alone_base=correct_parent & ~correct & mask
            alone_new=correct & ~correct_parent & mask
            both=correct_parent & correct & mask
            neither=~correct_parent & ~correct & mask
            groups[k]=dict(total=int(mask.sum()),
                only_S18_correct=int(alone_base.sum()),
                only_expert_correct=int(alone_new.sum()),
                both_correct=int(both.sum()),
                neither_correct=int(neither.sum()))
        report['standalone'][key]=groups
    # Labels are intentionally consulted only *here*, so the union below
    # is a not-deployable theoretical ceiling, not an architecture.
    any_new=successes.any(axis=1)
    oracle=correct_parent | any_new
    report['oracle_upper_bound']=dict(
        global_total_correct=int(oracle.sum()),
        global_exact=float(oracle.mean()),
        poly_total_correct=int((oracle&poly).sum()),
        poly_exact=float((oracle&poly).sum()/poly.sum()),
        extra_poly_beyond_S18=int((any_new&~correct_parent&poly).sum()),
        complementary_cases=int((any_new&~correct_parent).sum()),
        reachable_only_with_true_K_oracle=True)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    lines=['# S39 independently tabulated S38 complementarity','',
        '**Oracle is a non-deployable diagnostic using truth after predictions.**',
        '| Expert | New correct events vs S18 | New poly correct events vs S18 | S18 correct/expert wrong poly | Both wrong poly |',
        '|---|---:|---:|---:|---:|']
    for key in KEYS:
        g=report['standalone'][key]
        lines.append(f"| {key} | {g['all']['only_expert_correct']} | "
             f"{g['poly']['only_expert_correct']} | "
             f"{g['poly']['only_S18_correct']} | "
             f"{g['poly']['neither_correct']} |")
    bound=report['oracle_upper_bound']
    lines+=['',f"ORACLE NON-DEPLOYABLE max poly union: {bound['poly_exact']*100:.4f}%",
          f"ORACLE NON-DEPLOYABLE extra true-poly cases: {bound['extra_poly_beyond_S18']}",
          'No label-based chooser was trained or run.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pred',type=Path,required=True)
    p.add_argument('--probs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
