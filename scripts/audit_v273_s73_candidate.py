"""S73: audit 7 Exact-K only if complete original case-level predictions exist."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def require(ok,msg):
    if not ok: raise ValueError(msg)

def load(file):
    with np.load(file,allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}

def pair(y,old,new,mask):
    o=old==y;n=new==y;change=old!=new
    return dict(n=int(mask.sum()),old_correct=int((o&mask).sum()),
      new_correct=int((n&mask).sum()),
      corrections=int(((~o)&n&mask).sum()),
      regressions=int((o&(~n)&mask).sum()),
      changed=int((change&mask).sum()),
      wrong_to_wrong=int((change&(~o)&(~n)&mask).sum()))

def audit(s72_path,manifest_path,dest,s73_path=None,variant=None):
    require(not dest.exists(),'never overwrite prior audit')
    manifest=json.loads(manifest_path.read_text())
    a=load(s72_path)
    ids=a['global_index'];y=a['true_K'];fold=a['fold'];b=a['s72_poly_poly_guard_K']
    require(len(ids)==len(set(map(int,ids)))==manifest['total']==59309,'identity drift')
    require(set(map(int,fold))==set(manifest['folds']),'fold drift')
    require(int((y>=2).sum())==manifest['poly_n']==7385,'poly drift')
    require(int((b==y).sum())==manifest['s72']['global_correct']==49238,'S72 global drift')
    require(int(((b==y)&(y>=2)).sum())==manifest['s72']['poly_correct']==3202,'S72 poly drift')
    report=dict(status='BLOCKED_MISSING_S73_CASE_PREDICTIONS',promotion='NOT_APPROVED',
                incumbent='S72_poly_poly_guard',candidate_variant=variant,
                saved_aggregate_variants=manifest['variants'],actual_case_metrics=None)
    if s73_path is not None:
        require(variant in manifest['variants'],'unknown declared variant')
        v=load(s73_path)
        for key in ('global_index','true_K','fold'):
            require(key in v and np.array_equal(v[key],a[key]),'S73 identity mismatch: '+key)
        require('s73_K' in v and v['s73_K'].shape==b.shape,'missing complete case predictions')
        p=v['s73_K'];require(np.isin(p,range(7)).all(),'invalid predicted K')
        global_metrics=pair(y,b,p,np.ones(len(y),bool))
        poly_metrics=pair(y,b,p,y>=2)
        target=manifest['variants'][variant]
        require(global_metrics['new_correct']==target['global_correct'] and
                poly_metrics['new_correct']==target['poly_correct'],
                'candidate does not reproduce S73 archived aggregate')
        report['actual_case_metrics']={
          'global':global_metrics,'poly':poly_metrics,
          'by_true_K':{str(k):pair(y,b,p,y==k) for k in range(7)},
          'by_fold':{str(f):pair(y,b,p,fold==f) for f in manifest['folds']},
          'by_transition':{str(k)+'->'+str(j):pair(y,b,p,(b==k)&(p==j))
                           for k in range(7) for j in range(7)
                           if k!=j and np.any((b==k)&(p==j))}}
        report['status']='HISTORICAL_CASE_AUDIT_ONLY'
    dest.mkdir(parents=True)
    (dest/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('s72','manifest','out','s73'):
        p.add_argument('--'+n,type=Path,required=n not in ('s73',))
    p.add_argument('--variant')
    args=p.parse_args()
    r=audit(args.s72,args.manifest,args.out,args.s73,args.variant)
    print(r['status'],'promotion:',r['promotion'])
