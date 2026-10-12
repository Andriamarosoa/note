"""Series12: full acoustic summaries for K0–K6 plus two-model veto consensus."""
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
from scripts.loop_v273_policy_gate import GLOBAL,POLY
from scripts.loop_v273_native_risk import (
    read,full_members_and_geometry,designs,get_model)

SOURCE='series9__veto_P_ge5_and_G_le3'
THRESHOLDS=(0.,.10,.25)
ARMS=('logistic','hgb','both','mean')
SCOPES=('k0','all')

def full_acoustics(features_dir,ids,fold,prepared):
    all_rows=[]
    sha_records=[]
    for f in FOLDS:
        paths=list(features_dir.rglob(f'features-fold-{f}.npz'))
        require(len(paths)==1,'missing or duplicate acoustic archive for fold '+str(f))
        p=paths[0]
        meta=json.loads(p.with_name('report.json').read_text())
        require(meta['status']=='completed' and meta['fold']==f,'feature metadata drift')
        require(digest(p)==meta['feature_sha256'],'feature SHA256 mismatch')
        with np.load(p,allow_pickle=False) as z:
            rows={k:z[k] for k in ('global_index','summary','k','fold','baseline')}
        require(rows['summary'].shape[1]==58,'audio dimension changed')
        require(np.all(rows['fold']==f),'feature fold drift')
        require(np.isfinite(rows['summary']).all(),'nonfinite audio input')
        all_rows.append(rows)
        sha_records.append(dict(fold=f,sha256=meta['feature_sha256'],
                                rows=len(rows['global_index'])))
    merged={k:np.concatenate([x[k] for x in all_rows]) for k in all_rows[0]}
    order=align(merged['global_index'],ids)
    require(np.array_equal(merged['fold'][order],fold),'source/test fold drift')
    require(np.array_equal(merged['k'][order],prepared['native_truth']),'truth drift')
    require(np.array_equal(merged['baseline'][order],prepared['native_baseline']),
            'native baseline drift')
    sound=np.asarray(merged['summary'][order],dtype=np.float32)
    eligible=prepared['native_position']
    require(np.allclose(sound[eligible],prepared['features'],atol=3e-5,rtol=3e-5),
            'historical acoustic values drift')
    return sound,sha_records

def emit_decisions(series9,b,changed,probs):
    n=len(b)
    require(series9.shape==b.shape and changed.shape==b.shape,'decoder shape')
    ii=np.flatnonzero(changed)
    predictions={'series9_parent':series9.copy(),'freeze_parent':b.copy()}
    records={}
    required={'logistic','hgb'}
    require(required==set(probs),'probability keys')
    margins={}
    for name,prob in probs.items():
        require(np.isfinite(prob[ii]).all() and np.allclose(prob[ii].sum(1),1,atol=1e-4),
                'probability normalization '+name)
        margins[name]=prob[ii,b[ii]]-prob[ii,series9[ii]]
    # Predeclared multi-model agreement; NO labels or folds in decoding.
    margins['mean']=(margins['logistic']+margins['hgb'])/2.
    for arm in ARMS:
        for scope in SCOPES:
            scope_mask=(b[ii]==0) if scope=='k0' else np.ones(len(ii),bool)
            for threshold in THRESHOLDS:
                name=f'series12__{arm}__{scope}__cost{threshold:g}'
                if arm=='both':
                    veto=scope_mask & (margins['logistic']>threshold) & (margins['hgb']>threshold)
                else:
                    veto=scope_mask & (margins[arm]>threshold)
                pred=series9.copy()
                pred[ii[veto]]=b[ii[veto]]
                predictions[name]=pred.astype(np.int8)
                records[name]={'veto_count':int(veto.sum()),
                               'eligible_changed_rows':int(scope_mask.sum())}
    require(len(predictions)==26,'predeclared inventory is 24+2')
    return predictions,records

def selftest():
    pred=np.array([1,1,2,2,3],np.int8)
    b=np.array([0,0,2,3,3],np.int8)
    changed=b!=pred
    pl=np.full((5,7),np.nan,dtype=np.float32)
    ph=np.full((5,7),np.nan,dtype=np.float32)
    for i in np.flatnonzero(changed):
        pl[i,:]=0
        ph[i,:]=0
        pl[i,b[i]]=0.8;pl[i,pred[i]]=0.2
        ph[i,b[i]]=0.6;ph[i,pred[i]]=0.4
    r,m=emit_decisions(pred,b,changed,{'logistic':pl,'hgb':ph})
    assert len(r)==26
    assert r['series12__both__k0__cost0.25'][0]==pred[0]
    assert r['series12__both__k0__cost0.1'][0]==b[0]
    assert r['series12__both__k0__cost0.1'][3]==pred[3]
    print('PASS: thresholds, consensus, scope and 26 variant identities')

def run(a):
    require(not a.output.exists(),'refuse overwrite of models or decisions')
    root=a.root/'analysis/evidence'
    d=read(root/'v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=(d[k] for k in ('native_global_index','native_truth',
                                   'native_baseline','native_fold'))
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'cohort invariant')
    incoming=read(a.series9)
    require(np.array_equal(ids,incoming['global_index']),'series9 alignment')
    n=list(map(str,incoming['variant_ids']))
    require(SOURCE in n,'missing series9 source')
    s9=incoming['predictions'][:,n.index(SOURCE)]
    require(metrics(y,s9)['correct']==49162,'series9 baseline drift')
    other=read(root/'v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(ids,other['global_index']),'upstream policy alignment')
    nm=list(map(str,other['variant_ids']))
    g=other['predictions'][:,nm.index(GLOBAL)]
    p=other['predictions'][:,nm.index(POLY)]
    require(np.all((s9==g)|(s9==p)),'source candidate lineage invalid')
    competitor=read(root/'v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(competitor['global_index'],ids),'yourmt identity drift')
    target=competitor['yourmt3_K']
    require(int((target==y).sum())==51328,'benchmark target changed')
    pieces,geometry,member_source=full_members_and_geometry(a.root,ids,y,b,fold)
    sound,feature_sources=full_acoustics(a.features,ids,fold,d)
    old=designs(d,ids,b,g,p,geometry)['votes_time']
    x=np.column_stack([old,sound]).astype(np.float32)
    require(x.shape==(len(ids),90) and np.isfinite(x).all(),'full-audio gate schema')
    changed=s9!=b
    candidate=(g!=b)|(p!=b)
    require(np.all(changed<=candidate),'unsupported series9 decision')
    prob={arm:np.full((len(ids),7),np.nan,dtype=np.float32)
          for arm in ('logistic','hgb')}
    t0=time.monotonic()
    records=[]
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    for piece in sorted(set(pieces.tolist())):
        held=(pieces==piece)&changed
        if not held.any():
            records.append(dict(piece=piece,status='no_changes'));continue
        ff=int(np.unique(fold[held])[0])
        fit=(fold==ff)&(pieces!=piece)&candidate
        fit_pos=np.flatnonzero(fit); held_pos=np.flatnonzero(held)
        require(len(fit_pos)>=100 and len(set(y[fit].tolist()))>=2,'few training examples')
        require(np.all(fold[fit]==ff) and not np.any((pieces==piece)&fit),
                'cross-piece training overlap')
        scaler=StandardScaler().fit(x[fit])
        train=np.clip(scaler.transform(x[fit]),-6,6)
        test=np.clip(scaler.transform(x[held]),-6,6)
        for model_name in ('logistic','hgb'):
            model=get_model(model_name)
            with threadpool_limits(limits=1):
                model.fit(train,y[fit])
                local=model.predict_proba(test)
            whole=np.zeros((len(held_pos),7),np.float32)
            whole[:,model.classes_.astype(int)]=local
            prob[model_name][held]=whole
            name=f'{piece}__{model_name}.joblib'
            joblib.dump(dict(model=model,scaler=scaler,piece=piece,
                fold=ff,fit_ids=ids[fit_pos],test_ids=ids[held_pos],
                feature_sha256=feature_sources),a.output/'models'/name,compress=3)
            records.append(dict(piece=piece,fold=ff,model=model_name,
                fit_rows=len(fit_pos),held_rows=len(held_pos),
                fit_pieces=sorted(set(pieces[fit].tolist())),
                fit_sha256=hashlib.sha256(ids[fit_pos].astype('<i8').tobytes()).hexdigest(),
                held_sha256=hashlib.sha256(ids[held_pos].astype('<i8').tobytes()).hexdigest(),
                model_file=name))
        print(json.dumps(dict(piece=piece,fold=ff,held=int(held.sum()),
            elapsed_seconds=round(time.monotonic()-t0,1))),flush=True)
    candidates,decision=emit_decisions(s9,b,changed,prob)
    reports={}
    for name,pred in candidates.items():
        reports[name]=dict(metrics=metrics(y,pred),
            versus_series9=paired(y,s9,pred),versus_freeze=paired(y,b,pred),
            versus_yourmt3=paired(y,target,pred),
            decisions=decision.get(name),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                versus_series9=paired(y[fold==f],s9[fold==f],pred[fold==f]),
                versus_freeze=paired(y[fold==f],b[fold==f],pred[fold==f]))
                for f in FOLDS})
    ordered=sorted(reports,key=lambda n:(reports[n]['metrics']['correct'],
        reports[n]['metrics']['poly']['correct']),reverse=True)
    useful=[name for name in candidates if name.startswith('series12__')
            and reports[name]['versus_series9']['global']['net']>0
            and reports[name]['versus_series9']['poly']['net']>=0]
    result=dict(status='completed',validation_independent=False,
        parent_posthoc_selected=True,promotion=False,models=len(records),
        policy_count=len(candidates),feature_sources=feature_sources,
        member_source=member_source,records=records,policies=reports,
        candidates_improve_both_vs_s9=useful,
        best_global_in_series=ordered[0],elapsed_seconds=time.monotonic()-t0)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,fold=fold,
        variant_ids=np.asarray(list(candidates)),
        predictions=np.column_stack(list(candidates.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=ids,changed=changed,
        algorithms=np.asarray(list(prob)),
        probabilities=np.stack(list(prob.values()),axis=1))
    lines=['# Series12 — Full-acoustic K0–K6 veto and agreement','',
        'Developed on exposed dataset; not a new independent test.','',
        '| Variant | Global exact | Poly exact | Regressions avoided | Corrections lost | Remaining regressions |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ordered:
        m=reports[name]['metrics'];s=reports[name]['versus_series9']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | {100*m['poly']['exact']:.4f}% | "
                     f"{s['corrections']} | {s['regressions']} | "
                     f"{reports[name]['versus_freeze']['global']['regressions']} |")
    lines+=['',f"Best global: {ordered[0]}",
        f"Global and poly improved vs series9: {len(useful)}",
        "All models, probabilistic margins and decisions retained. No promotion."]
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--features',type=Path)
    parser.add_argument('--series9',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:selftest()
    else:
        require(args.features and args.series9 and args.output,'source arguments required')
        run(args)
