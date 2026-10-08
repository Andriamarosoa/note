"""Series11: label-independent inputs, cross-piece learning, K0/all vetoes."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.yourmt3_exactk_common import FOLDS, digest, metrics, paired, require
from scripts.audit_v273_yourmt3_target import align, UPSTREAM_SHA256
from scripts.loop_v273_policy_gate import GLOBAL, POLY
from scripts.train_v273_open_cohort import piece_names

SOURCE='series9__veto_P_ge5_and_G_le3'
ARMS=('votes_time','votes_time_audio')
ALGORITHMS=('logistic','hgb')
COSTS=(0.,.10,.25)
SCOPES=('k0','all')

def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def full_members_and_geometry(root,ids,truth,baseline,fold):
    p=root/'analysis/evidence/v273-yourmt3-target/upstream/yourmt3-exactk-summary.zip'
    require(digest(p)==UPSTREAM_SHA256,'native member source SHA changed')
    with zipfile.ZipFile(p) as z:
        with np.load(io.BytesIO(z.read('predictions.npz')),allow_pickle=False) as m:
            order=align(m['global_index'],ids)
            for key,expected in [('k',truth),('baseline',baseline),('fold',fold)]:
                require(np.array_equal(m[key][order],expected),'source '+key+' drift')
            # Explicitly DO NOT load m['predicted']: this is YourMT3+, only for evaluation.
            members=m['member'][order]
            starts=m['starts'][order].astype(np.int64)
            ends=m['ends'][order].astype(np.int64)
    pieces=piece_names(members)
    require(len(set(pieces.tolist()))==19,'native piece inventory')
    require(np.all((ends>=starts)&(ends-starts<=1764)),'timing invariant')
    for piece in set(pieces):
        require(len(set(fold[pieces==piece].tolist()))==1,'piece crosses folds')
    last=np.zeros(len(ids),dtype=np.float32)
    for member in np.unique(members):
        at=np.flatnonzero(members==member)
        order=at[np.argsort(starts[at])]
        if len(order)>1:
            last[order[1:]]=np.clip(
                (starts[order[1:]]-starts[order[:-1]])/44100.,0,2)
        last[order[0]]=2.
    geom=np.column_stack([(ends-starts)/1764.,
        np.log1p(last), np.minimum(last,.2)/.2]).astype(np.float32)
    return pieces,geom,dict(archive_sha256=digest(p),member_count=len(set(members.tolist())),
        pieces=19,geometry_shape=list(geom.shape))

def designs(prepared,ids,b,g,p,geom):
    n=len(ids)
    one=np.eye(7,dtype=np.float32)
    votes=np.column_stack([one[b],one[g],one[p],
        (g-b).astype(np.float32),(p-b).astype(np.float32),
        (g-p).astype(np.float32),
        (g==p).astype(np.float32), (g==b).astype(np.float32),
        (p==b).astype(np.float32),
        (g>=2).astype(np.float32),(p>=2).astype(np.float32),
        geom])
    require(votes.shape==(n,32),'votes design shape')
    acoustic=np.zeros((n,58),dtype=np.float32)
    present=np.zeros(n,dtype=np.float32)
    positions=prepared['native_position']
    require(np.array_equal(ids[positions],prepared['global_index']),
        'old acoustic ID alignment')
    acoustic[positions]=prepared['features'].astype(np.float32)
    present[positions]=1.
    require(np.isfinite(acoustic).all(),'nonfinite acoustic source')
    expanded=np.column_stack([votes,acoustic,present]).astype(np.float32)
    require(expanded.shape==(n,91),'expanded design shape')
    return {'votes_time':votes,'votes_time_audio':expanded}

def get_model(algorithm):
    if algorithm=='logistic':
        return LogisticRegression(C=.1,solver='lbfgs',max_iter=500,tol=1e-6)
    if algorithm=='hgb':
        return HistGradientBoostingClassifier(loss='log_loss',learning_rate=.05,
            max_iter=100,max_leaf_nodes=15,max_depth=4,min_samples_leaf=40,
            l2_regularization=10,early_stopping=False,random_state=27411)
    raise ValueError(algorithm)

def self_test():
    b=np.array([0,1,2,3],np.int8)
    g=np.array([1,1,3,3],np.int8)
    p=np.array([0,2,2,4],np.int8)
    prep={'native_position':np.array([2]),'global_index':np.array([12]),
          'features':np.ones((1,58),np.float32)}
    ids=np.array([10,11,12,13])
    geom=np.ones((4,3),np.float32)
    x=designs(prep,ids,b,g,p,geom)
    assert x['votes_time'].shape==(4,32)
    assert x['votes_time_audio'].shape==(4,91)
    assert x['votes_time_audio'][2,-1]==1.
    assert x['votes_time_audio'][0,-1]==0.
    print('PASS: masks, votes, acoustic eligibility and full decision design')

def evaluate(a):
    require(not a.output.exists(),'never overwrite native gate run')
    root=a.root/'analysis/evidence'
    prep=read(root/'v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=(prep[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold'))
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'cohort drift')
    series9=read(a.series9)
    require(np.array_equal(series9['global_index'],ids),'series9 identity drift')
    names=list(map(str,series9['variant_ids']))
    require(SOURCE in names,'missing series9 parent')
    s9=series9['predictions'][:,names.index(SOURCE)]
    require(metrics(y,s9)['correct']==49162,'series9 scoring drift')
    series5=read(root/'v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(series5['global_index'],ids),'series5 identity drift')
    n=list(map(str,series5['variant_ids']))
    require(GLOBAL in n and POLY in n,'source producer missing')
    g=series5['predictions'][:,n.index(GLOBAL)]
    p=series5['predictions'][:,n.index(POLY)]
    require(np.all((s9==g)|(s9==p)),'parent lineage no longer G/P')
    competitor=read(root/'v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(competitor['global_index'],ids),'competitor ID drift')
    target=competitor['yourmt3_K']
    require(int((target==y).sum())==51328,'target scoring drift')
    pieces,geom,origin=full_members_and_geometry(a.root,ids,y,b,fold)
    # Earlier acoustic features are independent of audio ground truth.
    require(np.array_equal(pieces[prep['native_position']].astype(str),
        prep['piece'].astype(str)),'old piece provenance drift')
    x=designs(prep,ids,b,g,p,geom)
    changed=(s9!=b)
    candidate=(g!=b)|(p!=b)
    require(np.all(changed<=candidate),'no unsupported series9 action')
    probs={f'{arm}__{kind}':np.full((len(y),7),np.nan,dtype=np.float32)
        for arm in ARMS for kind in ALGORITHMS}
    record=[]
    t0=time.monotonic()
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    for piece in sorted(set(pieces.tolist())):
        test=(pieces==piece)&changed
        if not test.any():
            record.append(dict(piece=piece,status='no_changes'))
            continue
        f=int(np.unique(fold[test])[0])
        train=(fold==f)&(pieces!=piece)&candidate
        ti=np.flatnonzero(test);fi=np.flatnonzero(train)
        require(not np.any(train&test) and not np.any((pieces==piece)&train),
            'piece leakage')
        require(len(fi)>=100 and len(np.unique(y[train]))>=2,
            'insufficient fit sample classes')
        for arm in ARMS:
            scaler=StandardScaler().fit(x[arm][train])
            xx=np.clip(scaler.transform(x[arm][train]),-6,6)
            zz=np.clip(scaler.transform(x[arm][test]),-6,6)
            for kind in ALGORITHMS:
                model=get_model(kind)
                with threadpool_limits(limits=1):
                    model.fit(xx,y[train])
                    pred=model.predict_proba(zz)
                full=np.zeros((len(ti),7),dtype=np.float32)
                full[:,model.classes_.astype(int)]=pred
                key=f'{arm}__{kind}'
                probs[key][test]=full
                name=f'{piece}__{key}.joblib'
                joblib.dump(dict(model=model,scaler=scaler,fit_ids=ids[fi],
                    test_ids=ids[ti],fold=f,piece=piece,arm=arm,kind=kind,
                    classes=model.classes_.astype(int).tolist()),a.output/'models'/name,
                    compress=3)
                record.append(dict(piece=piece,fold=f,arm=arm,kind=kind,
                    fit_rows=len(fi),test_changed_rows=len(ti),model=name,
                    fit_pieces=sorted(set(pieces[train].tolist())),
                    fit_hash=hashlib.sha256(ids[fi].astype('<i8').tobytes()).hexdigest(),
                    test_hash=hashlib.sha256(ids[ti].astype('<i8').tobytes()).hexdigest()))
        print(json.dumps(dict(piece=piece,fold=f,test_changed=int(test.sum()),
            elapsed_seconds=round(time.monotonic()-t0,1))),flush=True)
    # Unchanged cases can have NaNs by design; no model decision needed.
    for key,prob in probs.items():
        require(np.isfinite(prob[changed]).all(),key+' missing held prediction')
        require(np.allclose(prob[changed].sum(1),1,atol=1e-4),key+' norm')
    candidates={'series9_parent':s9.copy(),'freeze_parent':b.copy()}
    decisions={}
    ii=np.flatnonzero(changed)
    for key,prob in probs.items():
        advantage=prob[ii,b[ii]]-prob[ii,s9[ii]]
        for scope in SCOPES:
            mask=(b[ii]==0) if scope=='k0' else np.ones(len(ii),bool)
            for cost in COSTS:
                name=f'series11__{key}__{scope}__cost{cost:g}'
                veto=mask & (advantage>cost)
                pred=s9.copy();pred[ii[veto]]=b[ii[veto]]
                candidates[name]=pred.astype(np.int8)
                decisions[name]=dict(veto_count=int(veto.sum()),
                    candidate_rows=int(mask.sum()))
    require(len(candidates)==26,'policy inventory announced')
    report={}
    for name,pred in candidates.items():
        report[name]=dict(metrics=metrics(y,pred),
            versus_series9=paired(y,s9,pred),versus_freeze=paired(y,b,pred),
            versus_yourmt3=paired(y,target,pred),
            decisions=decisions.get(name),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                versus_series9=paired(y[fold==f],s9[fold==f],pred[fold==f]),
                versus_freeze=paired(y[fold==f],b[fold==f],pred[fold==f]))
                for f in FOLDS})
    order=sorted(report,key=lambda k:(report[k]['metrics']['correct'],
        report[k]['metrics']['poly']['correct']),reverse=True)
    useful=[name for name in candidates if name.startswith('series11__')
            and report[name]['versus_series9']['global']['net']>0
            and report[name]['versus_series9']['poly']['net']>=0]
    summary=dict(status='completed',independent_validation=False,
        parent_selected_on_development_data=True,promotion=False,
        features_from_original_audio_only=True,
        target_seen_only_during_other_piece_fit=True,
        models=len(record),policy_count=len(candidates),
        origin=origin,records=record,policies=report,
        improvement_candidates=useful,best_global_in_series=order[0],
        elapsed_seconds=time.monotonic()-t0)
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,fold=fold,
        variant_ids=np.asarray(list(candidates)),
        predictions=np.column_stack(list(candidates.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=ids,
        changed=changed,variant_ids=np.asarray(list(probs)),
        probabilities=np.stack(list(probs.values()),axis=1))
    lines=['# Series11 — Native cross-piece regression guard','',
        'Predictions are development results, NOT new independent validation.','',
        '| Policy | Global | Poly | Corrections avoided regression vs S9 | Corrections lost vs S9 | Regressions vs freeze |',
        '|---|---:|---:|---:|---:|---:|']
    for name in order:
        entry=report[name];m=entry['metrics'];pair=entry['versus_series9']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | {100*m['poly']['exact']:.4f}% | "
            f"{pair['corrections']} | {pair['regressions']} | "
            f"{entry['versus_freeze']['global']['regressions']} |")
    lines+=['',f"Global best: {order[0]}.",
        f"Strict improvements global with no poly regression: {len(useful)}.",
        'All 26 decision vectors and fitted models are retained, no promotion.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--series9',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test()
    else:
        require(a.series9 and a.output,'need series9 path and output')
        evaluate(a)
