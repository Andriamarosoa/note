"""S21: cross-piece learned reliability of B and A's revised K proposals.

Train only on OTHER pieces within held fold. At inference choose S20 raw A proposal
or keep S18 based on its independently learned probability of correctness.
"""
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
from scripts.loop_v273_native_risk import read,get_model
from scripts.train_v273_aba_recurrent import prepare

SEED=27321
SOURCES=('A1','ABA2','ABA4','A4_noB')
STEPS=(0,1,3,4)
VIEWS=('with_B','without_B_features')
ALGOS=('logistic','hgb')
CUTS=(.20,.30,.40,.50,.60,.70)

def get(z,policy):
    arr=list(map(str,z['variant_ids']))
    require(policy in arr,'missing '+policy)
    return z['predictions'][:,arr.index(policy)].astype(np.int8)

def source_data(z,ids,parent):
    require(np.array_equal(z['global_index'],ids),'recurrent source identity')
    A=np.asarray(z['A_probs'],np.float32)
    B=np.asarray(z['B_compatibility'],np.float32)
    require(A.shape==(len(ids),5,7) and B.shape==(len(ids),4,7),
            'source messaging matrix wrong shape')
    require(np.isfinite(A).all() and np.isfinite(B).all(),
            'unfinished message producer')
    require(np.allclose(A.sum(axis=2),1,atol=1e-4),'A distributions not normalized')
    return A,B


def make_features(idx,s,view,base,A,B,parent):
    """No truth, fold, or piece identifier; candidate K comes from A only."""
    idx=np.asarray(idx,dtype=np.int64)
    cand=A[idx,STEPS[s]].argmax(1).astype(np.int64)
    old=parent[idx].astype(np.int64)
    one=np.eye(7,dtype=np.float32)
    count=len(idx)
    p=A[idx,STEPS[s]]
    oldprob=p[np.arange(count),old]
    candprob=p[np.arange(count),cand]
    b=B[idx] if view=='with_B' else np.zeros((count,4,7),np.float32)
    bp=b[np.arange(count),0,cand]
    bo=b[np.arange(count),0,old]
    b_last_prop=b[np.arange(count),3,cand]
    b_last_old=b[np.arange(count),3,old]
    b_average=b.mean(axis=1)
    stat=np.column_stack([
        oldprob,candprob,candprob-oldprob,
        -np.sum(p*np.log(p+1e-9),axis=1),
        bp,bo,bp-bo,b_last_prop,b_last_old,
        b_last_prop-b_last_old,
        b_average[np.arange(count),cand],
        b_average[np.arange(count),old],
        np.abs(cand-old),(cand>=2).astype(np.float32),
        (old>=2).astype(np.float32),
        np.max(p,axis=1),
        (p[:,4]-A[idx,0,4]),
        (p[:,3]-A[idx,0,3]),
        (p[:,2]-A[idx,0,2])
    ]).astype(np.float32)
    source_one=np.zeros((count,4),np.float32)
    source_one[:,s]=1
    features=np.column_stack([
        base[idx],A[idx].reshape(count,35),
        b.reshape(count,28),one[old],one[cand],source_one,stat
    ]).astype(np.float32)
    require(features.shape==(count,190),'meta-gate feature dimension drift')
    require(np.isfinite(features).all(),'bad risk gate features')
    return features,cand

def sample_training(held,fold,pieces,parent,base,A,B,y,view):
    fit=(fold==np.unique(fold[held])[0])&~held
    xchunks=[];labels=[];events=[]
    for s in range(4):
        indices=np.flatnonzero(fit)
        proposal=A[indices,STEPS[s]].argmax(1)
        selected=indices[proposal!=parent[indices]]
        if not len(selected):continue
        x,cand=make_features(selected,s,view,base,A,B,parent)
        xchunks.append(x)
        labels.append((cand==y[selected]).astype(np.int32))
        events.append(selected)
    require(bool(xchunks),'missing training alternatives')
    xx=np.concatenate(xchunks,axis=0)
    yy=np.concatenate(labels,axis=0)
    require(len(yy)>100 and len(np.unique(yy))==2,
            'too few changed proposals to fit trust model')
    return xx,yy,np.concatenate(events),fit

def decode(parent,A,scores):
    proposals={s:A[:,STEPS[i]].argmax(1).astype(np.int8)
               for i,s in enumerate(SOURCES)}
    out={'series18_parent':parent.copy()}
    events={}
    for view in VIEWS:
        for algo in ALGOS:
            for source in SOURCES:
                cand=proposals[source]
                for cutoff in CUTS:
                    key=f'series21__{view}__{algo}__{source}__p>{cutoff:g}'
                    eligible=(cand!=parent)&(scores[f'{view}_{algo}_{source}']>cutoff)
                    pred=np.where(eligible,cand,parent).astype(np.int8)
                    out[key]=pred
                    events[key]=int(eligible.sum())
    require(len(out)==97,'96 gate policies plus one parent')
    return out,events


def selftest():
    n=7
    old=np.array([0,1,2,3,4,5,6],np.int8)
    A=np.zeros((n,5,7),np.float32)
    for i in range(5):
        A[np.arange(n),i,(old+i+1)%7]=1
    B=np.ones((n,4,7),np.float32)*.5
    base=np.ones((n,90),np.float32)
    for view in VIEWS:
        xx,cand=make_features(np.arange(n),0,view,base,A,B,old)
        assert xx.shape==(n,190) and np.array_equal(cand,(old+1)%7)
        if view=='without_B_features':
            assert np.all(xx[:,125:153]==0)
    scores={f'{v}_{algo}_{s}':np.ones(n,np.float32)*.99
            for v in VIEWS for algo in ALGOS for s in SOURCES}
    out,_=decode(old,A,scores)
    assert len(out)==97
    assert np.array_equal(
        out['series21__with_B__logistic__A1__p>0.7'],(old+1)%7)
    print('PASS: 190 source-only features, B/no-B ablation and 96 fixed gates')


def train(args):
    require(not args.output.exists(),'refuse to overwrite research artifacts')
    d=prepare(args)
    s20=read(args.series20)
    A,B=source_data(s20,d['ids'],d['parent'])
    decisions=read(args.s20_predictions)
    require(np.array_equal(decisions['global_index'],d['ids']),'S20 decision IDs')
    require(np.array_equal(get(decisions,'series18_parent'),d['parent']),'S18 source drift')
    for j,stage in enumerate(SOURCES):
        key=dict(A1='series20__pass1__raw',ABA2='series20__pass2__raw',
                 ABA4='series20__pass4__raw',A4_noB='series20__no_B_pass4__raw')[stage]
        require(np.array_equal(A[:,STEPS[j]].argmax(1),get(decisions,key)),
                'S20 raw proposal drift '+stage)
    X=np.asarray(d['X'][:,:90],np.float32)
    require(X.shape==(59309,90),'sound/votes summary drift')
    n=len(d['ids'])
    scores={f'{v}_{algo}_{s}':np.full(n,np.nan,np.float32)
            for v in VIEWS for algo in ALGOS for s in SOURCES}
    records=[];start=time.monotonic()
    args.output.mkdir(parents=True)
    (args.output/'models').mkdir()
    for piece in sorted(set(d['pieces'].tolist())):
        held=d['pieces']==piece
        f=int(np.unique(d['fold'][held])[0])
        record=dict(piece=piece,fold=f,models={})
        for view in VIEWS:
            xtrain,ytrain,train_idx,fit=sample_training(
                held,d['fold'],d['pieces'],d['parent'],X,A,B,d['y'],view)
            scaler=StandardScaler().fit(xtrain)
            standardized=np.clip(scaler.transform(xtrain),-6,6)
            for algo in ALGOS:
                model=get_model(algo)
                with threadpool_limits(limits=1):
                    model.fit(standardized,ytrain)
                require(np.array_equal(model.classes_,np.array([0,1])),
                        'model no longer binary')
                for si,source in enumerate(SOURCES):
                    test_idx=np.flatnonzero(held)
                    eligible=A[test_idx,STEPS[si]].argmax(1)!=d['parent'][test_idx]
                    test_idx=test_idx[eligible]
                    if not len(test_idx):continue
                    xx,cand=make_features(test_idx,si,view,X,A,B,d['parent'])
                    with threadpool_limits(limits=1):
                        prob=model.predict_proba(
                            np.clip(scaler.transform(xx),-6,6))[:,1]
                    scores[f'{view}_{algo}_{source}'][test_idx]=prob.astype(np.float32)
                filename=f'{piece}__{view}__{algo}.joblib'
                joblib.dump(dict(model=model,scaler=scaler,held_ids=d['ids'][held],
                    fit_ids=d['ids'][fit],fit_event_ids=d['ids'][train_idx],
                    piece=piece,fold=f,view=view,algorithm=algo),
                    args.output/'models'/filename,compress=3)
                record['models'][f'{view}_{algo}']=dict(
                    file=filename,train_rows=len(ytrain),
                    train_positives=int(ytrain.sum()),
                    train_pieces=sorted(set(d['pieces'][fit].tolist())),
                    fit_hash=hashlib.sha256(
                        d['ids'][fit].astype('<i8').tobytes()).hexdigest())
        records.append(record)
        print(json.dumps(dict(piece=piece,fold=f,fitted=len(records),
            elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
    for si,source in enumerate(SOURCES):
        proposed=A[:,STEPS[si]].argmax(1)
        eligible=proposed!=d['parent']
        for view in VIEWS:
            for algo in ALGOS:
                z=scores[f'{view}_{algo}_{source}']
                require(np.isfinite(z[eligible]).all(),'nonfinite held-piece gate')
                require(np.all((z[eligible]>=0)&(z[eligible]<=1)),
                        'invalid trust probability')
    out,decisions=decode(d['parent'],A,scores)
    out['freeze_reference']=d['freeze'].copy()
    require(len(out)==98,'source and parent count changed')
    audits={}
    for name,pred in out.items():
        audits[name]=dict(metrics=metrics(d['y'],pred),
            vs_parent=paired(d['y'],d['parent'],pred),
            vs_freeze=paired(d['y'],d['freeze'],pred),
            vs_yourmt3=paired(d['y'],d['yourmt3'],pred),
            gate_count=decisions.get(name),
            folds={str(f):dict(metrics=metrics(d['y'][d['fold']==f],pred[d['fold']==f]),
                vs_parent=paired(d['y'][d['fold']==f],
                    d['parent'][d['fold']==f],pred[d['fold']==f]))
                for f in FOLDS})
    ranking=sorted(audits,key=lambda k:(
        audits[k]['metrics']['correct'],audits[k]['metrics']['poly']['correct']),reverse=True)
    zero_loss=[name for name in out if name.startswith('series21__') and
        audits[name]['vs_parent']['global']['regressions']==0 and
        audits[name]['vs_parent']['global']['corrections']>0 and
        audits[name]['metrics']['poly']['correct']>=2998]
    report=dict(status='completed',independent_validation=False,
        previously_exposed_cohort=True,not_promoted=True,
        parent_source='series18 verified conservative policy',
        held_piece_excluded_in_all_meta_fits=True,
        source_S20_also_crosspiece=True,
        views=list(VIEWS),algorithms=list(ALGOS),
        source_policies=list(SOURCES),thresholds=list(CUTS),
        feature_count=190,models=len(records)*4,
        score_count=len(scores),policy_count=len(out),
        ablation_B_messages_are_removed_from_meta_features_only=True,
        records=records,audits=audits,
        no_loss_research_candidates=zero_loss,
        best_global=ranking[0],
        elapsed_seconds=round(time.monotonic()-start,2))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(args.output/'predictions.npz',global_index=d['ids'],
        true_K=d['y'],fold=d['fold'],variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    np.savez_compressed(args.output/'scores.npz',global_index=d['ids'],
        variant_ids=np.asarray(list(scores)),
        probabilities=np.column_stack(list(scores.values())))
    lines=['# Série21 — arbitration apprise de la fiabilité des sélections A/B','',
        'Previously exposed development cohort, no new external validation or promotion.',
        '', '| Policy | Global | Poly | Corrections vs S18 | Regressions vs S18 | Regressions vs freeze |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ranking:
        m=audits[name]['metrics']
        p=audits[name]['vs_parent']['global']
        q=audits[name]['vs_freeze']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
            f"{100*m['poly']['exact']:.4f}% | {p['corrections']} | "
            f"{p['regressions']} | {q['regressions']} |")
    lines+=['',f'Best global: {ranking[0]}',
        f'Strict no-lost-correction candidates: {len(zero_loss)}',
        'All 96 variants, every trained gate and every rejected alternative archived.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--series20',type=Path)
    p.add_argument('--s20-predictions',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.series20 and
                a.s20_predictions and a.output,'all four data sources required')
        train(a)
