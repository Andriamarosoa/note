"""Nested, piece-held-out learned risk gate for V27.3 series-5 producers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require
from scripts.loop_v273_policy_gate import GLOBAL, POLY

ARMS=('votes','votes_acoustic')
CS=(.03,.3)
COSTS=(0.,.10,.20)
REVERSE_COSTS=(0.,.05,.10,.20,.30,.50)

def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def design(d,base,g,p):
    n=len(g)
    one=np.eye(7,dtype=np.float32)
    vote=np.column_stack([one[base],one[g],one[p],
        (p-g).astype(np.float32),(p-base).astype(np.float32),
        (g-base).astype(np.float32),
        (p>=2).astype(np.float32),(g>=2).astype(np.float32),
        (p==base).astype(np.float32),(g==base).astype(np.float32)])
    require(vote.shape==(n,28),'vote schema changed')
    acoustic=np.asarray(d['features'],dtype=np.float32)
    require(acoustic.shape==(n,58),'acoustic feature schema changed')
    return {'votes':vote,'votes_acoustic':np.column_stack([vote,acoustic])}

def decision_advantage(model,scaler,raw):
    q=model.predict_proba(np.clip(scaler.transform(raw),-6,6))
    prob=np.zeros((len(raw),3),np.float64)
    prob[:,model.classes_.astype(int)]=q
    return prob[:,1]-prob[:,0]

def main(a):
    root=a.root/'analysis/evidence'
    require(not a.output.exists(),'refusing overwrite')
    d=read(root/'v273-regression-loops/prepared/inputs.npz')
    ids,y,base,fold,pos=[d[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold','native_position')]
    require(len(y)==59309 and len(pos)==7493,'cohort scope')
    require(np.array_equal(ids[pos],d['global_index']),'eligible ID alignment')
    require(np.array_equal(y[pos],d['truth']) and np.array_equal(base[pos],d['baseline']),'eligible label alignment')
    require(np.array_equal(fold[pos],d['fold']),'eligible fold alignment')
    archive=read(root/'v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(ids,archive['global_index']),'producer ID alignment')
    names=list(map(str,archive['variant_ids']))
    require(GLOBAL in names and POLY in names,'missing source policies')
    g=np.asarray(archive['predictions'][:,names.index(GLOBAL)],dtype=np.int8)
    p=np.asarray(archive['predictions'][:,names.index(POLY)],dtype=np.int8)
    src=read(root/'v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(src['global_index'],ids),'YourMT3 alignment')
    competitor=src['yourmt3_K']
    require(int(np.sum(competitor==y))==51328 and int(np.sum((competitor==y)&(y>=2)))==4028,
            'YourMT3 target drift')
    x=design(d,base[pos],g[pos],p[pos])
    pieces=np.asarray(d['piece']).astype(str)
    efold=d['fold']; ey=d['truth']; ge=g[pos]; pe=p[pos]
    require(len(np.unique(pieces))==19,'piece inventory')
    disagree=ge!=pe
    # Three outcomes: only G correct, only P correct, or neither correct.
    target=np.where(ge==ey,0,np.where(pe==ey,1,2)).astype(np.int8)
    require(not np.any(disagree & (ge==ey) & (pe==ey)),'two correct distinct classes')
    adv={f'{arm}__C{c:g}':np.zeros(len(pos),np.float32) for arm in ARMS for c in CS}
    records=[]
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    for piece in sorted(set(pieces)):
        piece_rows=pieces==piece
        held=piece_rows & disagree
        if not np.any(held):
            records.append(dict(piece=piece,status='no_disagreements',test_rows=0))
            continue
        f=int(np.unique(efold[held])[0])
        require(np.all(efold[piece_rows]==f),'piece spans folds')
        train=(efold==f)&(~piece_rows)&disagree
        require(not np.any(train & piece_rows) and np.all(efold[train]==f),
                'gate scope violation')
        at=np.flatnonzero(held);learn=np.flatnonzero(train)
        if len(learn)<30 or len(set(target[train].tolist()))<2:
            records.append(dict(piece=piece,fold=f,status='not_enough_training',
                fit_rows=len(learn),test_rows=len(at)))
            continue
        for arm in ARMS:
            scaler=StandardScaler().fit(x[arm][train])
            xx=np.clip(scaler.transform(x[arm][train]),-6,6)
            for C in CS:
                model=LogisticRegression(C=C,solver='lbfgs',
                    max_iter=500,tol=1e-6)
                model.fit(xx,target[train])
                name=f'{arm}__C{C:g}'
                adv[name][held]=decision_advantage(model,scaler,x[arm][held]).astype(np.float32)
                filename=f'{piece}__{name}.joblib'
                joblib.dump(dict(model=model,scaler=scaler,
                    fit_ids=ids[pos[learn]],test_ids=ids[pos[at]],
                    fold=f,arm=arm,C=C,parent_global=GLOBAL,parent_poly=POLY),
                    a.output/'models'/filename,compress=3)
                records.append(dict(piece=piece,fold=f,arm=arm,C=C,status='trained',
                    fit_rows=len(learn),test_rows=len(at),
                    outcome_counts=np.bincount(target[train],minlength=3).tolist(),
                    train_piece_count=len(set(pieces[train])),
                    fit_sha256=hashlib.sha256(ids[pos[learn]].astype('<i8').tobytes()).hexdigest(),
                    test_sha256=hashlib.sha256(ids[pos[at]].astype('<i8').tobytes()).hexdigest(),
                    model=filename))
    outputs={GLOBAL:g.copy(),POLY:p.copy()}
    switches={}
    for name,diff in adv.items():
        for cost in COSTS:
            key=f'series7__{name}__cost{cost:g}'
            pred=g.copy()
            choose=disagree & (diff>cost)
            pred[pos[choose]]=p[pos[choose]]
            outputs[key]=pred
            switches[key]=int(choose.sum())
    # Series 8: preserve the stronger polyphonic parent and veto only
    # changes with a sufficiently strong learned advantage for G.
    for name,diff in adv.items():
        for cost in REVERSE_COSTS:
            key=f'series8__poly_parent__{name}__cost{cost:g}'
            pred=p.copy()
            choose=disagree & (diff < -cost)
            pred[pos[choose]]=g[pos[choose]]
            outputs[key]=pred
            switches[key]=int(choose.sum())
    require(len(outputs)==38,'policy count')
    results={}
    for name,pred in outputs.items():
        results[name]=dict(metrics=metrics(y,pred),
            switches=switches.get(name,None),
            vs_global=paired(y,g,pred),
            vs_poly=paired(y,p,pred),
            vs_freeze=paired(y,base,pred),
            vs_yourmt3=paired(y,competitor,pred),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                vs_global=paired(y[fold==f],g[fold==f],pred[fold==f]))
                for f in FOLDS})
    ranks=sorted(results,key=lambda name:(
        results[name]['metrics']['poly']['correct'],
        results[name]['metrics']['correct']),reverse=True)
    report=dict(status='completed',population='exposed development GuitarSet cohort',
        training='same-fold other-piece fit; outer piece excluded',
        upstream_producers_exclude_target_fold=True,validation_independent=False,
        promotion=False,policies_count=len(outputs),records=records,
        policies=results,best_poly=ranks[0])
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,
        true_K=y,fold=fold,variant_ids=np.asarray(list(outputs)),
        predictions=np.column_stack(list(outputs.values())))
    np.savez_compressed(a.output/'gate-advantages.npz',
        eligible_global_index=ids[pos],fold=efold,disagree=disagree,
        variants=np.asarray(list(adv)),advantage=np.column_stack(list(adv.values())))
    lines=['# Series 7 and 8: nested learned risk gates with poly parent preserved','',
      'Each evaluated piece was excluded from the gate fit; no independent unseen validation.','',
      '| Policy | Global | Poly | Corrections vs G | Regressions vs G | K5 | K6 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name in ranks:
        m=results[name]['metrics'];b=results[name]['vs_global']['global']
        lines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {b['corrections']} | {b['regressions']} | "
            f"{m['by_k']['5']['exact']*100:.2f}% | {m['by_k']['6']['exact']*100:.2f}% |")
    beat=sum(v['metrics']['correct']>51328 and v['metrics']['poly']['correct']>4028
             for v in results.values())
    lines+=['',f'Best poly candidate: {ranks[0]}',
        f'Policies beating YourMT3+ on both metrics: {beat}',
        'No automatic promotion; development validation only.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

def selftest():
    d={'features':np.zeros((6,58))}
    b=np.array([0,1,2,3,4,5],dtype=np.int8)
    g=np.array([0,1,2,3,4,5],dtype=np.int8)
    p=np.array([1,1,3,2,5,5],dtype=np.int8)
    x=design(d,b,g,p)
    assert x['votes'].shape==(6,28)
    assert x['votes_acoustic'].shape==(6,86)
    sc=StandardScaler().fit(x['votes'])
    model=LogisticRegression(C=.1,max_iter=100).fit(
        np.clip(sc.transform(x['votes']),-6,6),np.array([0,1,2,0,1,2]))
    assert np.isfinite(decision_advantage(model,sc,x['votes'])).all()
    print('PASS: learned risk gate input and output schema')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,default=Path('model/v273-learned-loop'))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:selftest()
    else:main(args)
