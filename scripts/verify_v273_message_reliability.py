"""S21: independent replay of 96 learned K proposals and held-piece gate weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require
from scripts.loop_v273_native_risk import read
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_message_reliability import (
    SOURCES,STEPS,VIEWS,ALGOS,CUTS,get,source_data,make_features,decode)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()

def replay(args):
    require(not args.output.exists(),'do not overwrite independent audit')
    d=prepare(args)
    v=read(args.input/'predictions.npz')
    s=read(args.input/'scores.npz')
    report=json.loads((args.input/'report.json').read_text())
    require(report['status']=='completed' and
            not report['independent_validation'] and
            report['held_piece_excluded_in_all_meta_fits'],
            'model source or separation report is unexpected')
    for z in (v,s):
        require(np.array_equal(z['global_index'],d['ids']),'global index mismatch')
    require(np.array_equal(v['true_K'],d['y']) and
            np.array_equal(v['fold'],d['fold']),
            'truth/fold identity changed')
    source=read(args.series20)
    A,B=source_data(source,d['ids'],d['parent'])
    base=d['X'][:,:90].astype(np.float32)
    score_names=list(map(str,s['variant_ids']))
    stored={key:s['probabilities'][:,i] for i,key in enumerate(score_names)}
    policies,_=decode(d['parent'],A,stored)
    policies['freeze_reference']=d['freeze'].copy()
    variant_names=list(map(str,v['variant_ids']))
    require(set(variant_names)==set(policies),'full policy inventory changed')
    for name,pred in policies.items():
        require(np.array_equal(pred,v['predictions'][:,variant_names.index(name)]),
                'decision replay mismatch '+name)
    model_hashes={}
    max_probability_diff=0.0
    require(len(report['records'])==19,'missing held piece records')
    for rec in report['records']:
        piece=rec['piece']
        held=d['pieces']==piece
        fold=int(np.unique(d['fold'][held])[0])
        fit=(d['fold']==fold)&~held
        require(fold==rec['fold'] and piece not in set(
            d['pieces'][fit].tolist()),'piece or fold leakage')
        i=np.flatnonzero(held)
        for view in VIEWS:
            for algo in ALGOS:
                filename=rec['models'][f'{view}_{algo}']['file']
                f=args.input/'models'/filename
                require(f.is_file(),'missing weight '+filename)
                loaded=joblib.load(f)
                require(loaded['piece']==piece and loaded['fold']==fold and
                        loaded['view']==view and loaded['algorithm']==algo,
                        'incorrect fitted estimator provenance')
                require(np.array_equal(loaded['held_ids'],d['ids'][held]) and
                        np.array_equal(loaded['fit_ids'],d['ids'][fit]),
                        'held piece was used by fit')
                require(rec['models'][f'{view}_{algo}']['fit_hash']==
                    hashlib.sha256(d['ids'][fit].astype('<i8').tobytes()).hexdigest(),
                    'fit sample hash changed')
                model_hashes[filename]=sha256(f)
                model=loaded['model']
                scaler=loaded['scaler']
                for si,source_name in enumerate(SOURCES):
                    cand=A[i,STEPS[si]].argmax(1)
                    eligible=i[cand!=d['parent'][i]]
                    if not len(eligible):continue
                    xx,_=make_features(eligible,si,view,base,A,B,d['parent'])
                    with threadpool_limits(limits=1):
                        replayed=model.predict_proba(
                            np.clip(scaler.transform(xx),-6,6))[:,1]
                    key=f'{view}_{algo}_{source_name}'
                    previous=stored[key][eligible]
                    max_probability_diff=max(max_probability_diff,
                        float(np.max(np.abs(replayed-previous))))
    require(len(model_hashes)==76,'expected 76 fitted models')
    require(max_probability_diff<2e-5,
            'held-piece probabilities not reproduced from fitted weights')
    calculated={}
    for name,pred in policies.items():
        m=metrics(d['y'],pred)
        original=report['audits'][name]['metrics']
        require(m['correct']==original['correct'] and
                m['poly']['correct']==original['poly']['correct'],
                'scoring mismatch '+name)
        calculated[name]=dict(metrics=m,
            vs_parent=paired(d['y'],d['parent'],pred),
            vs_freeze=paired(d['y'],d['freeze'],pred))
    result=dict(status='verified',all_96_policies_exactly_replayed=True,
        all_76_crosspiece_models_reloaded=True,independent_new_data_validation=False,
        original_cohort_exposed=True,source_meta_train_labels_excluded_test_piece=True,
        model_sha256=model_hashes,max_probability_replay_difference=max_probability_diff,
        policies=calculated)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    ranking=sorted(calculated,key=lambda n:calculated[n]['metrics']['correct'],reverse=True)
    lines=['# S21 independent numerical audit of learned feedback reliability','',
        'All 96 decision policies exactly regenerated and all 76 fitted gate models reloaded.',
        f'Max deviation from stored held-piece trust probability: {max_probability_diff:.9g}',
        'This validates numerical reproduction, NOT unseen music performance.',
        '', '| Policy | Exact global | Exact poly | Corrected vs S18 | Regessed vs S18 |',
        '|---|---:|---:|---:|---:|']
    for name in ranking:
        m=calculated[name]['metrics'];p=calculated[name]['vs_parent']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
            f"{100*m['poly']['exact']:.4f}% | {p['corrections']} | {p['regressions']} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--s18',type=Path,required=True)
    p.add_argument('--series20',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    replay(p.parse_args())
