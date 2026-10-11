"""Series13: held-piece K0 birth detection using 42-frame harmonic flow."""
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
from scripts.yourmt3_exactk_common import FOLDS,digest,metrics,paired,require
from scripts.audit_v273_yourmt3_target import align
from scripts.loop_v273_native_risk import (read,full_members_and_geometry,designs,get_model)
from scripts.loop_v273_policy_gate import GLOBAL,POLY

PARENT='series9__veto_P_ge5_and_G_le3'
CUTOFFS=(.85,.90,.95,.975)
TYPES=('summary_logistic','summary_hgb','flow_logistic','flow_hgb',
       'flow_both','flow_mean')

def flow_profile(logstates):
    z=np.asarray(logstates,np.float32)
    require(z.ndim==3 and z.shape[1:]==(42,49),'harmonic time profile shape')
    times=np.arange(42,dtype=np.float64)*256*1000/44100.-80.
    pre=(times>=-65)&(times<-18)
    onset=(times>=-12)&(times<=42)
    late=(times>70)&(times<140)
    delta_times=.5*(times[:-1]+times[1:])
    flux_on=(delta_times>=-12)&(delta_times<=120)
    require(pre.sum()>=5 and onset.sum()>=5 and late.sum()>=5,'phase masks invalid')
    A=np.expm1(z)
    require(np.isfinite(A).all() and (A>=0).all(),'invalid original activation')
    pre_m=A[:,pre].mean(axis=1);on_m=A[:,onset].mean(axis=1)
    late_m=A[:,late].mean(axis=1)
    birth=np.maximum(on_m-pre_m,0)
    positive=np.maximum(np.diff(A,axis=1)[:,flux_on],0).sum(axis=1)
    out=np.column_stack([np.log1p(m) for m in (pre_m,on_m,late_m,birth,positive)])
    require(out.shape==(len(z),245) and np.isfinite(out).all(),'flow feature contract')
    return out.astype(np.float32)

def collect(a,ids,y,b,fold,prepared):
    feats=[]
    reports=[]
    for f in FOLDS:
        ps=list(a.rglob(f'features-fold-{f}.npz'))
        require(len(ps)==1,'full archive inventory '+str(f))
        path=ps[0]
        rep=json.loads(path.with_name('report.json').read_text())
        require(rep['status']=='completed' and rep['fold']==f,'feature report integrity')
        require(digest(path)==rep['feature_sha256'],'audio archive SHA changed')
        with np.load(path,allow_pickle=False) as z:
            index=z['global_index'].copy();labels=z['k'].copy()
            bases=z['baseline'].copy();folds=z['fold'].copy()
            summary=z['summary'].astype(np.float32)
            flow=flow_profile(z['sequence'])
        require(summary.shape==(len(index),58),'summary shape drift')
        feats.append(dict(id=index,truth=labels,baseline=bases,fold=folds,
                          summary=summary,flow=flow))
        reports.append(dict(fold=f,sha256=rep['feature_sha256'],rows=len(index)))
    merged={k:np.concatenate([d[k] for d in feats]) for k in feats[0]}
    order=align(merged['id'],ids)
    for k,expected in [('truth',y),('baseline',b),('fold',fold)]:
        require(np.array_equal(merged[k][order],expected),'aligned acoustic cohort '+k)
    summary=merged['summary'][order]
    flow=merged['flow'][order]
    pos=prepared['native_position']
    require(np.allclose(summary[pos],prepared['features'],rtol=3e-5,atol=3e-5),
            'old 58 acoustic feature drift')
    return summary,flow,reports

def decode(y,b,parent,prob):
    require(len(prob)==6,'all probability heads missing')
    eligible=(b==0)&(parent!=b)
    names={'series9_parent':parent.copy(),'freeze_parent':b.copy()}
    decisions={}
    for kind in TYPES:
        for c in CUTOFFS:
            name=f'series13__{kind}__p0_gt{c:g}'
            x=prob[kind]
            require(np.isfinite(x[eligible]).all(),'nonfinite head '+kind)
            mask=eligible & (x>c)
            out=parent.copy()
            out[mask]=b[mask]
            names[name]=out.astype(np.int8)
            decisions[name]=int(mask.sum())
    require(len(names)==26,'policy count must be 24+2')
    return names,decisions

def selftest():
    z=np.zeros((7,42,49),dtype=np.float32)
    z[0,10:20]=1
    x=flow_profile(z)
    assert x.shape==(7,245)
    b=np.array([0,0,1],np.int8);p=np.array([1,2,2],np.int8)
    prob={k:np.array([.99,.96,np.nan]) for k in TYPES}
    out,dec=decode(None,b,p,prob)
    assert len(out)==26 and out['series13__flow_both__p0_gt0.975'][0]==0
    assert out['series13__flow_both__p0_gt0.975'][1]==p[1]
    print('PASS: all 42 time windows, 245 flow features and 24 gate policies')

def train(a):
    require(not a.output.exists(),'archive overwrite denied')
    root=a.root/'analysis/evidence'
    d=read(root/'v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=(d[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold'))
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'cohort drift')
    parent_data=read(a.series9)
    require(np.array_equal(parent_data['global_index'],ids),'series9 identity drift')
    names=list(map(str,parent_data['variant_ids']))
    require(PARENT in names,'parent absent')
    s9=parent_data['predictions'][:,names.index(PARENT)]
    require(metrics(y,s9)['correct']==49162,'parent accuracy drift')
    source=read(root/'v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(source['global_index'],ids),'source alignment drift')
    src=list(map(str,source['variant_ids']))
    g=source['predictions'][:,src.index(GLOBAL)]
    p=source['predictions'][:,src.index(POLY)]
    require(np.all((s9==g)|(s9==p)),'lineage drift')
    comp=read(root/'v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(comp['global_index'],ids),'competitor ID drift')
    bench=comp['yourmt3_K']
    pieces,geometry,provenance=full_members_and_geometry(a.root,ids,y,b,fold)
    summary,trajectory,files=collect(a.features,ids,y,b,fold,d)
    votes=designs(d,ids,b,g,p,geometry)['votes_time']
    representations={
        'summary':np.column_stack([votes,summary]).astype(np.float32),
        'flow':np.column_stack([votes,summary,trajectory]).astype(np.float32)}
    require(representations['summary'].shape==(len(y),90),'summary design drift')
    require(representations['flow'].shape==(len(y),335),'flow design drift')
    relevant=(b==0)&(s9!=0)
    target=(y==0).astype(np.int8)
    probs={f'{rep}_{alg}':np.full(len(y),np.nan,dtype=np.float32)
         for rep in ('summary','flow') for alg in ('logistic','hgb')}
    records=[]
    a.output.mkdir(parents=True);(a.output/'models').mkdir()
    t0=time.monotonic()
    for piece in sorted(set(pieces.tolist())):
        held=(pieces==piece)&relevant
        if not held.any():
            records.append(dict(piece=piece,status='no_eligible_changes'));continue
        f=int(np.unique(fold[held])[0])
        fit=(fold==f)&(pieces!=piece)&(b==0)
        train_pos=np.flatnonzero(fit);test_pos=np.flatnonzero(held)
        require(len(train_pos)>=100 and len(np.unique(target[fit]))==2,
                'insufficient K0/new-onset examples')
        require(not np.any((pieces==piece)&fit),'held piece in training')
        for rep,raw in representations.items():
            scaler=StandardScaler().fit(raw[fit])
            xx=np.clip(scaler.transform(raw[fit]),-6,6)
            xt=np.clip(scaler.transform(raw[held]),-6,6)
            for alg in ('logistic','hgb'):
                mdl=get_model(alg)
                with threadpool_limits(limits=1):
                    mdl.fit(xx,target[fit])
                    result=mdl.predict_proba(xt)[:,list(mdl.classes_).index(1)]
                key=f'{rep}_{alg}'
                probs[key][held]=result.astype(np.float32)
                filename=f'{piece}__{key}.joblib'
                joblib.dump(dict(model=mdl,scaler=scaler,piece=piece,fold=f,
                    train_ids=ids[train_pos],held_ids=ids[test_pos],key=key),
                    a.output/'models'/filename,compress=3)
                records.append(dict(piece=piece,fold=f,key=key,
                    train_rows=len(train_pos),held_rows=len(test_pos),
                    train_true_k0=int(target[fit].sum()),
                    train_true_nonzero=int((1-target[fit]).sum()),
                    train_pieces=sorted(set(pieces[fit].tolist())),
                    train_hash=hashlib.sha256(ids[train_pos].astype('<i8').tobytes()).hexdigest(),
                    held_hash=hashlib.sha256(ids[test_pos].astype('<i8').tobytes()).hexdigest(),
                    model_file=filename))
        print(json.dumps(dict(piece=piece,fold=f,held=int(held.sum()),
            elapsed_seconds=round(time.monotonic()-t0,2))),flush=True)
    for v in probs.values():require(np.isfinite(v[relevant]).all(),'missing K0 risk scores')
    probs['flow_both']=np.minimum(probs['flow_logistic'],probs['flow_hgb'])
    probs['flow_mean']=(probs['flow_logistic']+probs['flow_hgb'])/2.
    candidates,decisions=decode(y,b,s9,probs)
    report={}
    for name,pred in candidates.items():
        report[name]=dict(metrics=metrics(y,pred),vs_series9=paired(y,s9,pred),
            vs_freeze=paired(y,b,pred),vs_yourmt3=paired(y,bench,pred),
            veto_count=decisions.get(name,0),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                vs_series9=paired(y[fold==f],s9[fold==f],pred[fold==f])) for f in FOLDS})
    ranks=sorted(report,key=lambda n: (report[n]['metrics']['correct'],
        report[n]['metrics']['poly']['correct']),reverse=True)
    good=[n for n in report if n.startswith('series13__') and
          report[n]['vs_series9']['global']['net']>0 and
          report[n]['vs_series9']['poly']['net']>=0]
    result=dict(status='completed',validation_independent=False,
        promotion=False,posthoc_parent=True,grouping='other pieces within held fold',
        harmonic_components_are_proxies_not_pitch_labels=True,
        record_count=len(records),model_sources=files,provenance=provenance,
        records=records,policies=report,both_improved_over_s9=good,
        best_global=ranks[0],elapsed_seconds=time.monotonic()-t0)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,fold=fold,
        variant_ids=np.asarray(list(candidates)),
        predictions=np.column_stack(list(candidates.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=ids,
        relevant=relevant,heads=np.asarray(list(probs)),
        probabilities=np.column_stack(list(probs.values())))
    lines=['# Series 13 — Harmonic birth trajectory K0 risk','',
        'Development cohort, NO independent validation.','',
        '| Policy | Global exact | Poly exact | Regressions avoided | Corrections lost | Total regressions vs freeze |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ranks:
        m=report[name]['metrics'];p=report[name]['vs_series9']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | {100*m['poly']['exact']:.4f}% | "
            f"{p['corrections']} | {p['regressions']} | "
            f"{report[name]['vs_freeze']['global']['regressions']} |")
    lines+=['',f'Both global and poly improved vs series9: {len(good)}',
        f'Best in series: {ranks[0]}',
        'Keep all variants and model weights, no promotion.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--root',type=Path,default=Path('.'))
    args.add_argument('--features',type=Path)
    args.add_argument('--series9',type=Path)
    args.add_argument('--output',type=Path)
    args.add_argument('--self-test',action='store_true')
    a=args.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.series9 and a.output,'missing input arguments')
        train(a)
