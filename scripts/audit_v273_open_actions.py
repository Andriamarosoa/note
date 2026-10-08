"""Predeclared K0/K1 action ablation; no model fitting and no tuned thresholds."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS, digest, metrics, paired, require


def keep_ties(p, baseline):
    winner = p.argmax(1)
    return np.where(p[np.arange(len(p)), winner] > p[np.arange(len(p)), baseline],
                    winner, baseline)


def main(a):
    require(not a.output.exists(), 'refusing overwrite')
    root = a.root
    memory = root/'analysis/evidence/v273-regression-loops'
    with np.load(memory/'prepared/inputs.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    ids, y, base, fold, pos = (d[k] for k in
        ['native_global_index','native_truth','native_baseline','native_fold','native_position'])
    with np.load(memory/'posthoc-audit/appended-candidates.npz', allow_pickle=False) as z:
        key = 'regression_loops_posthoc__group_audit__cost_1.15__no_favorable_group'
        require(np.array_equal(z['global_index'], ids), 'parent alignment')
        current = z['predictions'][:, z['variant_ids'].tolist().index(key)]
    require(digest(a.coherent) == '707fc1681e2b0b1ef01805710186c4119ad697a0b52ed3330698e48f34c2f77b',
            'coherent source drift')
    with np.load(a.coherent, allow_pickle=False) as z:
        for key, expected in [('global_index',ids),('true_K',y),('fold',fold),
                              ('frozen_baseline_K',base),('eligible_global_index',ids[pos])]:
            require(np.array_equal(z[key],expected), 'coherent alignment '+key)
        coherent = z['class_probability']
        archived_coherent = z['predicted_K']
    archive = root/'analysis/evidence/v273-audit-memory/source-archives/upstream-producers.zip'
    with zipfile.ZipFile(archive) as z:
        raw = z.read('learned_corrector/correctors.npz')
        import hashlib
        require(hashlib.sha256(raw).hexdigest() ==
                'c08344dea641f4e6b30530a845113298369895d3c08256a8f1622a0863294ed5',
                'corrector cache drift')
        manifest = json.loads(z.read('learned_corrector/manifest.json'))
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        for key, expected in [('eligible_global_index',ids[pos]),('truth',y[pos]),
                              ('baseline',base[pos]),('fold',fold[pos])]:
            require(np.array_equal(z[key],expected), 'corrector alignment '+key)
        q = np.empty((len(pos),7),np.float32)
        provenance = []
        for held in FOLDS:
            train = tuple(f for f in FOLDS if f != held)
            record = next(r for r in manifest['producers'] if tuple(r['train_folds']) == train)
            fit = np.isin(fold[pos],train); test = fold[pos] == held
            require(np.array_equal(record['train_global_ids'],ids[pos][fit]), 'producer fit IDs')
            values = z['fit_'+'_'.join(map(str,train))]
            require(np.isnan(values[fit]).all() and np.isfinite(values[test]).all(), 'producer exclusions')
            q[test] = values[test]
            provenance.append(dict(held_fold=held, producer_train_folds=list(train),
                                   producer_train_id_sha256=record['train_id_sha256']))
    with zipfile.ZipFile(root/'analysis/evidence/v273-yourmt3-target/upstream/yourmt3-exactk-summary.zip') as z:
        with np.load(io.BytesIO(z.read('predictions.npz')), allow_pickle=False) as m:
            order = np.argsort(m['global_index']); order = order[np.searchsorted(m['global_index'][order],ids)]
            require(np.array_equal(m['global_index'][order],ids) and np.array_equal(m['k'][order],y), 'benchmark alignment')
            competitor = m['predicted'][order]
    variants = {}; sources = {'S8':q,'coherent':coherent}
    for source, p in sources.items():
        require(p.shape == (7493,7) and np.isfinite(p).all() and (p>=0).all()
                and np.allclose(p.sum(1),1,atol=1e-6), 'probability schema')
        open_action = keep_ties(p,base[pos])
        pred = base.copy(); pred[pos] = open_action
        variants[source+'__open_argmax'] = pred
        for threshold in (.50,.65,.80,.90,.95):
            apply = (open_action < 2) & (p[np.arange(len(p)),open_action] >= threshold)
            pred = current.copy(); pred[pos[apply]] = open_action[apply]
            variants[f'{source}__low_gate_{threshold:.2f}'] = pred
        if source == 'S8':
            projected = p[:,2:].copy()
            projected[np.arange(len(p)),base[pos]-2] += p[:,:2].sum(1)
            restricted = keep_ties(projected,base[pos]-2)+2
        else:
            winner = p[:,2:].argmax(1)+2
            restricted = np.where(p[np.arange(len(p)),winner] > p[np.arange(len(p)),base[pos]],winner,base[pos])
        pred = base.copy();pred[pos] = restricted
        variants[source+'__restricted_control'] = pred
    require(np.array_equal(variants['coherent__restricted_control'],archived_coherent),'coherent replay drift')
    with np.load(memory/'all-candidate-decisions.npz',allow_pickle=False) as z:
        k=z['variant_ids'].tolist().index('catalogue8_local__new_singleton_alone_K')
        require(np.array_equal(variants['S8__restricted_control'],z['predictions'][:,k]), 'S8 control drift')
    report = dict(status='completed',training_performed=False,automatic_promotion=False,
        independent_validation=False,protocol_commit='22b55fb96e3a27348fa8b51c8b2bbfe99b28e2ea',
        policies={name:dict(metrics=metrics(y,pred),versus_freeze=paired(y,base,pred),
                            versus_current=paired(y,current,pred),versus_yourmt3=paired(y,competitor,pred),
                            folds={str(f):paired(y[fold==f],base[fold==f],pred[fold==f]) for f in FOLDS})
                  for name,pred in variants.items()},producer_exclusions=provenance,
        source_coherent_sha256=digest(a.coherent),source_producers_sha256=digest(archive))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,baseline_K=base,fold=fold,
                        variant_ids=np.asarray(list(variants)),predictions=np.column_stack(list(variants.values())))
    np.savez_compressed(a.output/'source-probabilities.npz',eligible_global_index=ids[pos],
                        S8=q,coherent=coherent,baseline_K=base[pos],fold=fold[pos])
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    for name,r in report['policies'].items():
        print(json.dumps(dict(policy=name,global_exact=r['metrics']['exact'],poly_exact=r['metrics']['poly']['exact'],
                              global_paired=r['versus_freeze']['global'],poly_paired=r['versus_freeze']['poly'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--coherent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
