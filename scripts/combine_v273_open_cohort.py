"""Joint seven-class combination, excluding each receiver piece from every fit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.yourmt3_exactk_common import FOLDS,digest,metrics,paired,require
from scripts.audit_v273_yourmt3_target import align
from scripts.train_v273_open_cohort import piece_names,id_hash

FAMILIES=('probabilities__logistic','probabilities__trees','context__logistic','context__trees')


def prepare(a):
    with np.load(a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz',allow_pickle=False) as z:
        ids,y,base,fold=(z[k] for k in ['native_global_index','native_truth','native_baseline','native_fold'])
    with np.load(a.root/'analysis/evidence/v273-yourmt3-target/comparison/row-evidence.npz',allow_pickle=False) as z:
        require(np.array_equal(z['global_index'],ids),'reference alignment')
        previous=z['current_best_K'];competitor=z['yourmt3_K']
    with np.load(a.series2/'probabilities.npz',allow_pickle=False) as z:
        order=align(z['global_index'],ids);probability=z['probabilities'][order]
    blocks=[];feature_sources=[]
    for f in FOLDS:
        paths=list(a.features.rglob(f'features-fold-{f}.npz'));require(len(paths)==1,'feature inventory')
        path=paths[0];r=json.loads(path.with_name('report.json').read_text())
        require(digest(path)==r['feature_sha256'],'feature checksum')
        with np.load(path,allow_pickle=False) as z:
            blocks.append({k:z[k] for k in ['global_index','k','baseline','fold','member','summary','geometry']})
        feature_sources.append(dict(fold=f,sha256=r['feature_sha256']))
    all_features={k:np.concatenate([d[k] for d in blocks]) for k in blocks[0]}
    order=align(all_features['global_index'],ids)
    for key,value in [('k',y),('baseline',base),('fold',fold)]:
        require(np.array_equal(all_features[key][order],value),'feature metadata '+key)
    pieces=piece_names(all_features['member'][order])
    require(len(set(pieces))==19,'piece inventory')
    for piece in np.unique(pieces):require(len(set(fold[pieces==piece]))==1,'piece crosses folds')
    minimal=np.column_stack([np.log(np.maximum(probability,1e-7)).reshape(len(ids),-1),np.eye(7)[base]])
    context=np.column_stack([minimal,all_features['summary'][order],all_features['geometry'][order]])
    require(minimal.shape==(59309,91) and context.shape==(59309,195),'combination input shape')
    require(np.isfinite(context).all(),'nonfinite context')
    source=json.loads((a.series2/'report.json').read_text())
    for r in source['model_sources']:
        for m in r['records']:
            require(r['held_fold'] not in m['train_folds'],'upstream fold leak')
            require(set(m['train_folds'])==set(FOLDS)-{r['held_fold']},'upstream training scope')
    return dict(ids=ids,y=y,base=base,fold=fold,piece=pieces,previous=previous,competitor=competitor,
                raw_probabilities=probability,designs={'probabilities':minimal,'context':context},
                sources=dict(series2_report_sha256=digest(a.series2/'report.json'),
                             series2_probability_sha256=digest(a.series2/'probabilities.npz'),
                             features=feature_sources))


def infer(state,raw):
    x=np.clip((raw-state['mean'])/state['scale'],-6,6)
    local=state['model'].predict_proba(x)
    p=np.zeros((len(raw),7),np.float64);p[:,state['model'].classes_.astype(int)]=local
    require(np.isfinite(p).all() and np.allclose(p.sum(1),1),'invalid combined probabilities')
    return p


def evaluate(d,probability):
    ids,y,base,fold=d['ids'],d['y'],d['base'],d['fold'];new={};reports={}
    all_wrong=np.all(d['raw_probabilities'].argmax(2)!=y[:,None],axis=1)
    for family,q in probability.items():
        for tilt in (1.,1.5):
            adjusted=q.copy();adjusted[:,2:]*=tilt;adjusted/=adjusted.sum(1,keepdims=True)
            for parent_name,parent in [('freeze',base),('previous',d['previous'])]:
                for alpha in (.6,.8,1.):
                    p=alpha*adjusted+(1-alpha)*np.eye(7)[parent]
                    winner=p.argmax(1)
                    pred=np.where(p[np.arange(len(y)),winner]>p[np.arange(len(y)),parent],winner,parent)
                    name=f'open_k0k6_series3__{family}__tilt{tilt:g}__{parent_name}__mix{alpha:g}'
                    new[name]=pred.astype(np.int8)
                    expected=float(np.sum(q[np.arange(len(y)),pred]-q[np.arange(len(y)),parent]))
                    actual=paired(y,parent,pred)['global']['net']
                    reports[name]=dict(metrics=metrics(y,pred),versus_freeze=paired(y,base,pred),
                        versus_previous_best=paired(y,d['previous'],pred),versus_yourmt3=paired(y,d['competitor'],pred),
                        expected_net_vs_parent=expected,actual_net_vs_parent=actual,
                        optimism_vs_parent=expected-actual,
                        correct_despite_all_12_producer_argmax_wrong=int(np.sum((pred==y)&all_wrong)),
                        folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                            versus_freeze=paired(y[fold==f],base[fold==f],pred[fold==f])) for f in FOLDS})
    require(len(new)==48,'announced policy inventory')
    return new,reports


def main(a):
    require(not a.output.exists() or a.resume,'refusing output overwrite')
    require(not (a.output/'report.json').exists(),'completed run cannot be resumed')
    d=prepare(a);t0=time.monotonic()
    a.output.mkdir(parents=True,exist_ok=a.resume);(a.output/'models').mkdir(exist_ok=a.resume)
    probabilities={f:np.full((len(d['y']),7),np.nan) for f in FAMILIES}
    records=[];splits={};resumed=0
    for piece in sorted(set(d['piece'])):
        test=np.flatnonzero(d['piece']==piece);fold=int(d['fold'][test[0]])
        fit=np.flatnonzero((d['fold']==fold)&(d['piece']!=piece))
        require(len(fit)>0 and not set(d['ids'][fit])&set(d['ids'][test]),'split overlap')
        require(not set(d['piece'][fit])&set(d['piece'][test]),'piece leakage')
        require(np.all(d['fold'][fit]==fold),'calibrator outside its producer-excluded fold')
        splits[piece+'__fit']=d['ids'][fit];splits[piece+'__test']=d['ids'][test]
        for family in FAMILIES:
            representation,kind=family.split('__');raw=d['designs'][representation]
            path=Path('models')/(piece+'__'+family+'.joblib')
            restore=a.replay or (a.output if a.resume and (a.output/path).exists() else None)
            if restore:
                state=joblib.load(restore/path)
                require(np.array_equal(state['fit_ids'],d['ids'][fit]) and
                        np.array_equal(state['test_ids'],d['ids'][test]),'saved model split drift')
                require(state['family']==family and state['held_piece']==piece and state['fold']==fold,
                        'saved model identity drift')
                scaler=StandardScaler().fit(raw[fit])
                require(np.array_equal(state['mean'],scaler.mean_) and
                        np.array_equal(state['scale'],scaler.scale_),'saved normalization drift')
                if not a.replay:resumed+=1
            else:
                scaler=StandardScaler().fit(raw[fit]);x=np.clip(scaler.transform(raw[fit]),-6,6)
                if kind=='logistic':
                    model=LogisticRegression(C=.1,solver='lbfgs',max_iter=500,tol=1e-7)
                else:
                    model=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.05,max_iter=100,
                        max_leaf_nodes=15,max_depth=4,min_samples_leaf=40,l2_regularization=10,
                        early_stopping=False,random_state=27406)
                with threadpool_limits(limits=1):model.fit(x,d['y'][fit])
                state=dict(model=model,mean=scaler.mean_,scale=scaler.scale_,fit_ids=d['ids'][fit],
                           test_ids=d['ids'][test],held_piece=piece,fold=fold,family=family)
                joblib.dump(state,a.output/path,compress=('xz',3))
            with threadpool_limits(limits=1):probabilities[family][test]=infer(state,raw[test])
            model_path=(a.replay if a.replay else a.output)/path
            records.append(dict(held_piece=piece,fold=fold,family=family,fit_rows=len(fit),test_rows=len(test),
                fit_id_sha256=id_hash(d['ids'][fit]),test_id_sha256=id_hash(d['ids'][test]),
                fit_pieces=sorted(set(d['piece'][fit])),producer_excluded_fold=fold,
                class_counts=np.bincount(d['y'][fit],minlength=7).tolist(),
                classes=state['model'].classes_.astype(int).tolist(),
                model=str(path),model_sha256=digest(model_path),
                solver_iterations=np.asarray(getattr(state['model'],'n_iter_',0)).reshape(-1).astype(int).tolist()))
            print(json.dumps(dict(stage='replay' if a.replay else 'fit',piece=piece,family=family,
                records=len(records),elapsed_seconds=round(time.monotonic()-t0,2))),flush=True)
    require(all(np.isfinite(p).all() for p in probabilities.values()),'incomplete crossfit predictions')
    new,reports=evaluate(d,probabilities)
    np.savez_compressed(a.output/'predictions.npz',global_index=d['ids'],true_K=d['y'],baseline_K=d['base'],
        fold=d['fold'],piece=d['piece'],variant_ids=np.asarray(list(new)),predictions=np.column_stack(list(new.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=d['ids'],family_ids=np.asarray(FAMILIES),
                        probabilities=np.stack([probabilities[f] for f in FAMILIES],axis=1))
    np.savez_compressed(a.output/'splits.npz',**splits)
    if a.replay:
        with np.load(a.replay/'probabilities.npz',allow_pickle=False) as z:
            actual=np.stack([probabilities[f] for f in FAMILIES],axis=1)
            require(np.allclose(actual,z['probabilities'],atol=1e-12,rtol=0),'probability replay drift')
            maxdiff=float(np.max(np.abs(actual-z['probabilities'])))
        with np.load(a.replay/'predictions.npz',allow_pickle=False) as z:
            require(np.array_equal(np.column_stack(list(new.values())),z['predictions']),'decision replay drift')
    else:maxdiff=None
    report=dict(status='completed',protocol_commit='668d98e025f34f04c7a46232b51cbb391c84268b',
        independent_validation=False,automatic_promotion=False,fold_label_free_final_system=False,
        held_piece_excluded_from_all_fits=True,producer_excludes_entire_receiver_fold=True,
        models=len(records),resumed_models=resumed,policies=reports,records=records,sources=d['sources'],
        replay_max_abs=maxdiff,elapsed_seconds=time.monotonic()-t0,
        files_sha256={p.name:digest(p) for p in a.output.glob('*.npz')})
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    top=sorted(reports,key=lambda n:reports[n]['metrics']['correct'],reverse=True)
    for name in top[:10]:
        r=reports[name];print(json.dumps(dict(policy=name,global_exact=r['metrics']['exact'],
            poly_exact=r['metrics']['poly']['exact'],paired=r['versus_freeze']['global'],
            versus_yourmt3=r['versus_yourmt3']['global'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--series2',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',type=Path)
    p.add_argument('--resume',action='store_true')
    main(p.parse_args())
