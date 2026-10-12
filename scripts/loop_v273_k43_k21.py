"""S18: two held-piece binary vetoes for K4→K3 and K2→K1."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require
from scripts.loop_v273_policy_gate import GLOBAL, POLY
from scripts.loop_v273_native_risk import read, full_members_and_geometry, designs, get_model
from scripts.loop_v273_birth_flow import collect

PARENT='series17__flow_logistic__k32__pbase_gt0.99'
DIRECTIONS=((4,3),(2,1))
SCOPES=('k43','k21','both')
ARMS=('summary_logistic','summary_hgb','flow_logistic','flow_hgb','flow_both','flow_mean')
THRESHOLDS=(.75,.85,.90,.95,.975,.99)

def extract_policy(z, name):
    keys=list(map(str,z['variant_ids']))
    require(name in keys,'missing policy '+name)
    return z['predictions'][:,keys.index(name)].astype(np.int8)

def make_policies(b,parent,heads):
    out={'series17_parent':parent.copy(),'freeze_parent':b.copy()}
    scopes={}
    for (source,dest),prefix in zip(DIRECTIONS,('k43','k21')):
        mask=(b==source)&(parent==dest)
        scopes[prefix]=mask
        for arm in ARMS:
            require(f'{prefix}__{arm}' in heads,'missing probability '+prefix+' '+arm)
            require(np.isfinite(heads[f'{prefix}__{arm}'][mask]).all(),
                    'missing evaluated rows for '+prefix+' '+arm)
    for arm in ARMS:
        for cost in THRESHOLDS:
            for scope in SCOPES:
                veto=np.zeros(len(b),bool)
                for key in ('k43','k21'):
                    if scope in (key,'both'):
                        veto|=scopes[key]&(heads[f'{key}__{arm}']>cost)
                name=f'series18__{arm}__{scope}__pbase_gt{cost:g}'
                out[name]=np.where(veto,b,parent).astype(np.int8)
    require(len(out)==110,'expected 108+2 variants')
    return out,scopes

def selftest():
    b=np.array([4,4,2,2,0,4,2],np.int8)
    p=np.array([3,4,1,1,1,3,1],np.int8)
    scores={f'{key}__{arm}':np.ones(len(b),np.float32)*.995
        for key in ('k43','k21') for arm in ARMS}
    out,m=make_policies(b,p,scores)
    assert len(out)==110 and int(m['k43'].sum())==2 and int(m['k21'].sum())==3
    assert np.array_equal(out['series18__flow_mean__both__pbase_gt0.99'],
                          np.array([4,4,2,2,1,4,2],np.int8))
    assert out['series18__flow_mean__k43__pbase_gt0.99'][2]==1
    assert out['series18__flow_mean__k21__pbase_gt0.99'][0]==3
    print('PASS: K4→3 and K2→1 strictly-scoped 108 head decisions')


def train(a):
    require(not a.output.exists(),'cannot overwrite research evidence')
    t0=time.monotonic()
    d=read(a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=(d[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold'))
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'native cohort mismatch')
    old=read(a.series17)
    require(np.array_equal(old['global_index'],ids),'series15 ids mismatch')
    parent=extract_policy(old,PARENT)
    require(metrics(y,parent)['correct']==49174,'series15 total mismatch')
    require(paired(y,b,parent)['global']['corrections']==2934 and
        paired(y,b,parent)['global']['regressions']==2214,'parent pairwise mismatch')
    five=read(a.root/'analysis/evidence/v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(five['global_index'],ids),'series5 ids mismatch')
    g=extract_policy(five,GLOBAL);p=extract_policy(five,POLY)
    # source series15 differs from S9 only in 7 K0→K1 cases.
    require(np.all((parent==g)|(parent==p)|(parent==b)),
            'parent contains unsupported model decisions')
    competitor=read(a.root/'analysis/evidence/v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(competitor['global_index'],ids),'benchmark identity changed')
    yourmt3=competitor['yourmt3_K']
    require(int((yourmt3==y).sum())==51328,'benchmark changed')
    pieces,geometry,provenance=full_members_and_geometry(a.root,ids,y,b,fold)
    audio,flow,feature_sources=collect(a.features,ids,y,b,fold,d)
    vote=designs(d,ids,b,g,p,geometry)['votes_time']
    features={'summary':np.column_stack([vote,audio]).astype(np.float32),
              'flow':np.column_stack([vote,audio,flow]).astype(np.float32)}
    require(features['summary'].shape==(len(y),90) and
        features['flow'].shape==(len(y),335),'feature schemas changed')
    probs={f'{prefix}__{rep}_{alg}':np.full(len(y),np.nan,np.float32)
           for prefix in ('k43','k21') for rep in ('summary','flow')
           for alg in ('logistic','hgb')}
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    records=[]
    for (source,dest),prefix in zip(DIRECTIONS,('k43','k21')):
        relevant=(b==source)&(parent==dest)
        base_correct=(y==b).astype(np.int8)
        for piece in sorted(set(pieces.tolist())):
            held=(pieces==piece)&relevant
            if not held.any():
                continue
            f=int(np.unique(fold[held])[0])
            fit=(fold==f)&(pieces!=piece)&relevant
            fi=np.flatnonzero(fit);ti=np.flatnonzero(held)
            require(np.all(fold[fit]==f) and not np.any(fit&(pieces==piece)),
                    'piece leakage')
            if len(fi)<30 or len(set(base_correct[fi].tolist()))<2:
                for rep in ('summary','flow'):
                    for alg in ('logistic','hgb'):
                        probs[f'{prefix}__{rep}_{alg}'][ti]=0.
                records.append(dict(direction=prefix,piece=piece,fold=f,
                    status='insufficient',fit_rows=len(fi),held_rows=len(ti)))
                continue
            for rep,features_raw in features.items():
                scaler=StandardScaler().fit(features_raw[fit])
                xx=np.clip(scaler.transform(features_raw[fit]),-6,6)
                zz=np.clip(scaler.transform(features_raw[held]),-6,6)
                for alg in ('logistic','hgb'):
                    model=get_model(alg)
                    with threadpool_limits(limits=1):
                        model.fit(xx,base_correct[fit])
                        prob=model.predict_proba(zz)[:,list(model.classes_).index(1)]
                    key=f'{prefix}__{rep}_{alg}'
                    probs[key][ti]=prob.astype(np.float32)
                    name=f'{prefix}__{piece}__{rep}_{alg}.joblib'
                    joblib.dump(dict(model=model,scaler=scaler,fold=f,piece=piece,
                        direction=prefix,fit_ids=ids[fi],held_ids=ids[ti],
                        labels_bin_count=np.bincount(base_correct[fi],minlength=2).tolist()),
                        a.output/'models'/name,compress=3)
                    records.append(dict(direction=prefix,piece=piece,fold=f,
                        key=key,status='trained',fit_rows=len(fi),held_rows=len(ti),
                        fit_pieces=sorted(set(pieces[fit].tolist())),
                        fit_hash=hashlib.sha256(ids[fi].astype('<i8').tobytes()).hexdigest(),
                        held_hash=hashlib.sha256(ids[ti].astype('<i8').tobytes()).hexdigest(),
                        file=name))
            print(json.dumps(dict(direction=prefix,piece=piece,fold=f,
                fit=int(fit.sum()),held=int(held.sum()),
                elapsed_seconds=round(time.monotonic()-t0,1))),flush=True)
        for rep in ('summary','flow'):
            h=probs[f'{prefix}__{rep}_logistic'];q=probs[f'{prefix}__{rep}_hgb']
            require(np.isfinite(h[relevant]).all() and np.isfinite(q[relevant]).all(),
                    'unpredicted held rows '+prefix)
        probs[f'{prefix}__flow_both']=np.minimum(
            probs[f'{prefix}__flow_logistic'],probs[f'{prefix}__flow_hgb'])
        probs[f'{prefix}__flow_mean']=(probs[f'{prefix}__flow_logistic']+
            probs[f'{prefix}__flow_hgb'])/2
    policies,eligible=make_policies(b,parent,probs)
    results={}
    for name,v in policies.items():
        results[name]=dict(metrics=metrics(y,v),
            vs_parent=paired(y,parent,v),vs_freeze=paired(y,b,v),
            vs_yourmt3=paired(y,yourmt3,v),
            by_fold={str(f):dict(metrics=metrics(y[fold==f],v[fold==f]),
                vs_parent=paired(y[fold==f],parent[fold==f],v[fold==f]),
                vs_freeze=paired(y[fold==f],b[fold==f],v[fold==f]))
                for f in FOLDS})
    qualified=[name for name in policies if name.startswith('series18__') and
        results[name]['vs_parent']['global']['corrections']>0 and
        results[name]['vs_parent']['global']['regressions']==0 and
        results[name]['vs_parent']['poly']['regressions']==0]
    ranked=sorted(results,key=lambda k:(
        results[k]['metrics']['correct'],results[k]['metrics']['poly']['correct']),reverse=True)
    report=dict(status='completed',independent_validation=False,
        already_exposed_data=True,posthoc_parent=True,automatic_promotion=False,
        source=provenance,feature_sources=feature_sources,
        direction_candidate_rows={key:int(value.sum()) for key,value in eligible.items()},
        policy_count=len(policies),fit_records=records,policies=results,
        zero_loss_candidates=qualified,best_global=ranked[0],
        elapsed_seconds=round(time.monotonic()-t0,2))
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,fold=fold,
        variant_ids=np.asarray(list(policies)),predictions=np.column_stack(list(policies.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=ids,
        variant_ids=np.asarray(list(probs)),probabilities=np.column_stack(list(probs.values())))
    lines=['# S16 — K1→K2 and K2→K3 held-piece conditional risk','',
        '**Research only. Exposed development cohort, not independent validation.**',
        f"Direction populations: {report['direction_candidate_rows']}",'',
        '| Candidate | Exact global | Exact poly | Regressions fixed vs S15 | Corrections lost vs S15 | Remaining regressions |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ranked:
        r=results[key];m=r['metrics'];comp=r['vs_parent']['global']
        lines.append(f"| {key} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {comp['corrections']} | "
            f"{comp['regressions']} | {r['vs_freeze']['global']['regressions']} |")
    lines+=['',f"Best global in S16: {ranked[0]}",
        f"Zero-loss candidates: {len(qualified)}",
        'All 108 policy vectors, fit exclusions and weights retained. No automatic promotion.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--series17',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features is not None and a.series17 is not None and
                a.output is not None,'missing inputs')
        train(a)
