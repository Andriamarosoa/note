"""S28: independent numerical replay of 152 piece-held learned head schedulers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_adaptive_scheduler import ACTIONS,decode,get_pred,read,digest
from scripts.yourmt3_exactk_common import metrics,paired,require


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for data in iter(lambda:file.read(2<<20),b''):
            h.update(data)
    return h.hexdigest()


def verify(args):
    require(not args.output.exists(),'refusing to overwrite replay evidence')
    d=prepare(args)
    source27=read(args.s27)
    source25=read(args.s25)
    p=read(args.input/'predictions.npz')
    g=read(args.input/'scores.npz')
    routes=read(args.input/'routes.npz')
    report=json.loads((args.input/'report.json').read_text())
    require(report['status']=='completed' and
            report['held_piece_excluded_from_scheduler_fit'] and
            not report['independent_validation'],'training split declaration missing')
    for z in (p,g,routes,source27,source25):
        require(np.array_equal(z['global_index'],d['ids']),'source index mismatch')
    require(np.array_equal(p['true_K'],d['y']) and
            np.array_equal(p['fold'],d['fold']),'true K/fold misalignment')
    require(list(map(str,g['action_names']))==list(ACTIONS),'actions changed')
    scores=np.asarray(g['estimated_fix_and_regress'],np.float32)
    require(scores.shape==(59309,8,2),'missing learned routing scores')
    actions=np.column_stack([get_pred(
        source27 if name.startswith('series27__') else source25,name)
        for name in ACTIONS]).astype(np.int8)
    require(np.array_equal(actions,g['action_predictions']),
            'candidate predictions sourced from wrong runs')
    out,chosen=decode(d['parent'],actions,scores)
    out['freeze_reference']=d['freeze']
    keys=list(map(str,p['variant_ids']))
    require(len(keys)==14 and set(keys)==set(out),
            'missing training policies in replay')
    for name,vec in out.items():
        require(np.array_equal(vec,p['predictions'][:,keys.index(name)]),
            'predictions differ from deterministic decoder '+name)
    route_keys=list(map(str,routes['variant_ids']))
    require(set(route_keys)==set(chosen),'routing choices inventory drift')
    for name,arr in chosen.items():
        require(np.array_equal(arr,routes['chosen_action'][:,route_keys.index(name)]),
                'route assignment changed '+name)
    X=np.column_stack([
        d['X'][:,:90],
        np.eye(7,dtype=np.float32)[d['parent']]
    ]).astype(np.float32)
    require(X.shape==(59309,97),'raw routing features changed')
    require(len(report['trained_splits'])==19 and
            report['models_count']==152,'lost scheduler checkpoints')
    maxerr=0.
    sha={}
    for record in report['trained_splits']:
        piece=record['piece'];f=int(record['fold'])
        held=d['pieces']==piece
        fit=(d['fold']==f)&~held
        require(piece not in set(d['pieces'][fit].tolist()),
                'incorrect held-piece exclusion')
        require(record['held_count']==int(held.sum()) and
                record['train_count']==int(fit.sum()),'training cohort drift')
        require(record['fit_SHA256']==digest(d['ids'][fit]) and
                record['held_SHA256']==digest(d['ids'][held]),
                'input training sample hashing drift')
        require(len(record['actions'])==8,'missing one route model')
        for i,spec in enumerate(record['actions']):
            require(spec['action']==ACTIONS[i],'changed model target')
            file=args.input/'models'/spec['file']
            require(file.exists(),'learned route model missing')
            state=joblib.load(file)
            require(state['action']==ACTIONS[i] and
                state['held_piece']==piece and
                state['fold']==f and
                np.array_equal(state['train_ids'],d['ids'][fit]) and
                np.array_equal(state['test_ids'],d['ids'][held]),
                'wrong fit/held provenance of learned route')
            with threadpool_limits(limits=2):
                z=state['model'].predict(np.clip(
                    state['scaler'].transform(X[held]),-6,6))
            arr=np.clip(z,0,1)
            maxerr=max(maxerr,float(np.abs(arr-scores[held,i]).max()))
            sha[file.name]=digest_file(file)
    require(len(sha)==152,'not all gate weights verified')
    require(maxerr<2e-6,'cannot reproduce routing scores numerically')
    checks={}
    for name,vec in out.items():
        m=metrics(d['y'],vec)
        parent=paired(d['y'],d['parent'],vec)
        require(m['correct']==report['audits'][name]['metrics']['correct'] and
                m['poly']['correct']==report['audits'][name]['metrics']['poly']['correct'],
                'stored per-K score mismatch '+name)
        checks[name]=dict(metrics=m,vs_parent=parent)
    args.output.mkdir(parents=True)
    result=dict(status='verified',all_152_route_models_reloaded=True,
        all_14_prediction_vectors_identical=True,
        all_12_routes_identical=True,source_previously_exposed=True,
        independent_unseen_validation=False,no_promotion=True,
        max_route_score_absolute_difference=maxerr,
        model_SHA256=sha,score_audit=checks)
    (args.output/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    lines=['# S28 verified learned neural-head scheduler','',
        f'152 trained route estimators loaded; max probability-score error {maxerr:.9g}.',
        '14 complete policies and 12 route vectors exactly regenerated.',
        'The underlying development cohort is not new music. No promotion.',
        '', '| Policy | Global | Poly | Added corrections | New regressions |',
        '|---|---:|---:|---:|---:|']
    for name,q in checks.items():
        m=q['metrics'];pa=q['vs_parent']['global']
        lines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{pa['corrections']} | {pa['regressions']} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--features',type=Path,required=True)
    a.add_argument('--s18',type=Path,required=True)
    a.add_argument('--s27',type=Path,required=True)
    a.add_argument('--s25',type=Path,required=True)
    a.add_argument('--input',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    verify(a.parse_args())
