"""Series14: targeted, held-piece K0->K1 regression risk, never widen actions."""
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
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require
from scripts.loop_v273_policy_gate import GLOBAL,POLY
from scripts.loop_v273_native_risk import read,full_members_and_geometry,designs,get_model
from scripts.loop_v273_birth_flow import collect

PARENT='series9__veto_P_ge5_and_G_le3'
CUTS=(.75,.85,.90,.95,.975)
HEADS=('summary_logistic','summary_hgb','flow_logistic','flow_hgb','flow_both','flow_mean')

def decode(b,s9,risk):
    mask=(b==0)&(s9==1)
    policies={'series9_parent':s9.copy(),'freeze_parent':b.copy()}
    changed={}
    for name in HEADS:
        require(name in risk,'missing head '+name)
        require(np.isfinite(risk[name][mask]).all(),'missing held piece score '+name)
        for cutoff in CUTS:
            key=f'series14__{name}__p0_gt{cutoff:g}'
            select=mask&(risk[name]>cutoff)
            pred=s9.copy()
            pred[select]=0
            policies[key]=pred
            changed[key]=int(select.sum())
    require(len(policies)==32,'candidate count drift')
    return policies,changed

def selftest():
    b=np.array([0,0,2,0,1,0],np.int8)
    p=np.array([1,2,3,1,0,1],np.int8)
    prob={n:np.array([.99,np.nan,np.nan,.87,np.nan,.65]) for n in HEADS}
    a,counts=decode(b,p,prob)
    assert len(a)==32 and a['series14__flow_both__p0_gt0.975'][0]==0
    assert a['series14__flow_both__p0_gt0.975'][3]==1
    assert a['series14__flow_both__p0_gt0.75'][1]==2
    assert a['series14__flow_both__p0_gt0.75'][4]==0
    print('PASS: 30 static policies, K0->K1-only, thresholds and no invented action')

def run(a):
    require(not a.output.exists(),'refuse to overwrite audit')
    root=a.root/'analysis/evidence'
    d=read(root/'v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=(d[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold'))
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'native alignment')
    prev=read(a.series9)
    require(np.array_equal(prev['global_index'],ids),'parent global indexes')
    keys=list(map(str,prev['variant_ids']))
    require(PARENT in keys,'missing fixed source parent')
    s9=prev['predictions'][:,keys.index(PARENT)]
    require(metrics(y,s9)['correct']==49162,'parent record mismatch')
    producers=read(root/'v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(producers['global_index'],ids),'producer ids')
    names=list(map(str,producers['variant_ids']))
    g=producers['predictions'][:,names.index(GLOBAL)]
    p=producers['predictions'][:,names.index(POLY)]
    require(np.all((s9==g)|(s9==p)),'invalid source class lineage')
    comp=read(root/'v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(comp['global_index'],ids),'YourMT3 row mismatch')
    yourmt3=comp['yourmt3_K']
    require(int((yourmt3==y).sum())==51328,'reference metric drift')
    pieces,geometry,provenance=full_members_and_geometry(a.root,ids,y,b,fold)
    audio,flow,feature_sources=collect(a.features,ids,y,b,fold,d)
    votes=designs(d,ids,b,g,p,geometry)['votes_time']
    features={'summary':np.column_stack([votes,audio]).astype(np.float32),
              'flow':np.column_stack([votes,audio,flow]).astype(np.float32)}
    require(features['summary'].shape==(len(y),90),'summary shape')
    require(features['flow'].shape==(len(y),335),'flow shape')
    relevant=(b==0)&(s9==1)
    t0=time.monotonic()
    require(int((relevant & (y==0)).sum())==691,'known 0->1 regression mismatch')
    target=(y==0).astype(np.int8)
    output={n:np.zeros(len(y),np.float32) for n in HEADS[:4]}
    records=[]
    a.output.mkdir(parents=True);(a.output/'models').mkdir()
    for piece in sorted(set(pieces.tolist())):
        held=(pieces==piece)&relevant
        if not held.any():continue
        ff=int(np.unique(fold[held])[0])
        train=(fold==ff)&(pieces!=piece)&relevant
        fitid=np.flatnonzero(train);testid=np.flatnonzero(held)
        require(np.all(fold[train]==ff) and not (train&(pieces==piece)).any(),
                'piece/fold exclusion')
        if len(fitid)<30 or len(np.unique(target[train]))<2:
            records.append(dict(piece=piece,fold=ff,status='insufficient',
                                fit_rows=len(fitid),test_rows=len(testid)))
            continue
        for key,values in features.items():
            scaler=StandardScaler().fit(values[train])
            xfit=np.clip(scaler.transform(values[train]),-6,6)
            xtest=np.clip(scaler.transform(values[held]),-6,6)
            for kind in ('logistic','hgb'):
                model=get_model(kind)
                with threadpool_limits(limits=1):
                    model.fit(xfit,target[train])
                    result=model.predict_proba(xtest)[:,list(model.classes_).index(1)]
                output[f'{key}_{kind}'][held]=result.astype(np.float32)
                filename=f'{piece}__{key}_{kind}.joblib'
                joblib.dump(dict(model=model,scaler=scaler,fold=ff,piece=piece,
                    fit_global_ids=ids[fitid],held_global_ids=ids[testid],
                    class_0_count=int((target[train]==0).sum()),
                    class_1_count=int((target[train]==1).sum())),
                    a.output/'models'/filename,compress=3)
                records.append(dict(piece=piece,fold=ff,representation=key,
                    algorithm=kind,model_file=filename,
                    fit_rows=len(fitid),held_rows=len(testid),
                    train_pieces=sorted(set(pieces[train].tolist())),
                    fit_sha256=hashlib.sha256(ids[fitid].astype('<i8').tobytes()).hexdigest(),
                    test_sha256=hashlib.sha256(ids[testid].astype('<i8').tobytes()).hexdigest()))
        print(json.dumps(dict(stage='trained_piece',piece=piece,
             fit_rows=int(train.sum()),test_rows=int(held.sum()),
             seconds=round(time.monotonic()-t0,1))),flush=True)
    for name in HEADS[:4]:
        require(np.isfinite(output[name][relevant]).all(),'bad head value')
    output['flow_both']=np.minimum(output['flow_logistic'],output['flow_hgb'])
    output['flow_mean']=(output['flow_logistic']+output['flow_hgb'])/2.
    candidates,decisions=decode(b,s9,output)
    reports={}
    for name,pred in candidates.items():
        reports[name]=dict(metrics=metrics(y,pred),versus_series9=paired(y,s9,pred),
            versus_freeze=paired(y,b,pred),versus_yourmt3=paired(y,yourmt3,pred),
            vetoed=decisions.get(name,None),
            by_fold={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                paired=paired(y[fold==f],s9[fold==f],pred[fold==f]))
                for f in FOLDS})
    ranked=sorted(reports,key=lambda key:(reports[key]['metrics']['correct'],
        reports[key]['metrics']['poly']['correct']),reverse=True)
    qualifies=[key for key in candidates if key.startswith('series14__') and
         reports[key]['versus_series9']['global']['net']>0 and
         reports[key]['versus_series9']['poly']['net']>=0]
    result=dict(status='completed',cohort_exposed=True,independent_validation=False,
        promotion=False,conditional_training=True,parent_posthoc=True,
        count_policies=len(candidates),audit=reports,records=records,
        feature_sha256=feature_sources,source=provenance,
        candidates_improve_both=qualifies,best_global=ranked[0])
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,fold=fold,
        variant_ids=np.asarray(list(candidates)),
        predictions=np.column_stack(list(candidates.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=ids,
        relevant=relevant,variant_ids=np.asarray(list(output)),
        probabilities=np.column_stack(list(output.values())))
    lines=['# Series14 — conditional K0→K1 discrimination','',
        'Held-piece crossfit, model performance from exposed cohort only. No promotion.',
        '','| Policy | Global | Poly | Regressions avoided | Corrections lost | Remaining regressions |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ranked:
        v=reports[name];m=v['metrics'];s=v['versus_series9']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
            f"{100*m['poly']['exact']:.4f}% | {s['corrections']} | {s['regressions']} | "
            f"{v['versus_freeze']['global']['regressions']} |")
    lines+=['',f'Best global: {ranked[0]}',
        f'Policies improving global without poly loss: {len(qualifies)}',
        'All 30 policies and training records preserved; no independent validation.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--features',type=Path)
    a.add_argument('--series9',type=Path)
    a.add_argument('--output',type=Path)
    a.add_argument('--self-test',action='store_true')
    args=a.parse_args()
    if args.self_test:selftest()
    else:
        require(args.features is not None and args.series9 is not None and args.output is not None,
                'missing required inputs')
        run(args)
