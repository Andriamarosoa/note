"""V26 paired categorical count weighting experiment.

Fresh shared count encoder in both arms; no V24 latent/checkpoint reuse during
inner selection. Saved outer V24 ranking is used only for fixed realization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave
import zipfile

import numpy as np

from scripts import train_v240_categorical_k_candidate_subset as v240
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v177_candidate_centric as v177
from scripts.audit_v240_selection import ranked_ids, score, aggregate
from scripts.evaluate_v8_boundaries import _reference_positions

ARMS = ('weighted', 'uniform')
SEED = 16061
MAX_EPOCHS = 20
SOURCE_RUN = 34168832668
SOURCE_SHA = '980a052e6cd9eee7896962ceb3a458e0ae258334'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def class_weights(fit_k):
    counts = np.bincount(np.asarray(fit_k, dtype=np.int32), minlength=7)
    table = np.ones(7, dtype=np.float64)
    nz = counts > 0
    table[nz] = np.sqrt(len(fit_k) / (7 * counts[nz]))
    table = np.clip(table, .35, 4.)
    table /= np.mean(table[np.asarray(fit_k, dtype=np.int32)])
    return table.astype(np.float32)


def arm_weights(arm, fit_k):
    if arm == 'weighted':
        return class_weights(fit_k)
    if arm == 'uniform':
        return np.ones(7, dtype=np.float32)
    raise ValueError(arm)


def build_model(arm, seed):
    from scripts import train_v250_count_only as v250
    if arm not in ARMS:
        raise ValueError(arm)
    return v250.build_model('categorical', seed)


def validate_partitions(ctx):
    outer, fit, val, final = [np.asarray(ctx[k], dtype=np.int64) for k in
                             ('outer_idx','meta_fit_idx','meta_val_idx','final_fit_idx')]
    if any(len(x) == 0 or len(np.unique(x)) != len(x) for x in (outer,fit,val,final)):
        raise RuntimeError('empty or duplicate partition')
    if np.intersect1d(outer, final).size or np.intersect1d(fit, val).size:
        raise RuntimeError('partition leakage')
    if not np.array_equal(np.sort(np.concatenate([fit,val])), np.sort(final)):
        raise RuntimeError('inner partitions do not cover final fit')
    if sorted(np.concatenate([outer,final]).tolist()) != list(range(len(ctx['k']))):
        raise RuntimeError('outer/final coverage mismatch')
    return outer,fit,val,final


def cardinality(k, pred):
    k = np.asarray(k); pred = np.asarray(pred)
    poly = k >= 2
    return {'exact': float(np.mean(k == pred)),
            'poly_rows': int(poly.sum()), 'poly_correct': int(np.sum(k[poly] == pred[poly])),
            'poly_exact': float(np.mean(k[poly] == pred[poly])) if poly.any() else None,
            'under': int(np.sum(pred < k)), 'over': int(np.sum(pred > k)),
            'k0_rows': int(np.sum(k == 0)), 'k0_false_birth_rows': int(np.sum((k == 0) & (pred > 0))),
            'confusion_true_by_predicted': np.bincount(k*7+pred, minlength=49).reshape(7,7).tolist()}


def verify_source(fold_dir, fold):
    expected = json.loads((Path(__file__).resolve().parents[1]/'analysis/v240-final-audit.json').read_text())
    paths = []
    for suffix in ('json','npz'):
        name = f'report-fold-{fold}.json' if suffix == 'json' else f'predictions-fold-{fold}.npz'
        hits = list(fold_dir.glob('**/'+name))
        hashes = [x['sha256'] for x in expected['source_files'] if Path(x['path']).name == name]
        if len(hits) != 1 or len(hashes) != 1 or digest(hits[0]) != hashes[0]:
            raise RuntimeError('noncanonical source artifact: '+name)
        paths.append(hits[0])
    return paths


def train(args):
    import tensorflow as tf
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.outer_fold not in range(5):
        raise ValueError('invalid fold')
    ctx = v171._fold_context(args)
    outer, fit, val, final = validate_partitions(ctx)
    cache, k = ctx['cache'], ctx['k'].astype(np.int32)
    allowed = {t.annotation_member for t in ctx['train_split']}
    if set(cache['members']) != allowed or allowed & {t.annotation_member for t in ctx['validation']}:
        raise RuntimeError('dataset/cache mismatch')
    rp, pp = verify_source(args.fold_dir, args.outer_fold)
    source = json.loads(rp.read_text())
    with np.load(pp, allow_pickle=False) as z:
        idx, truth = np.asarray(z['global_index']), np.asarray(z['k'])
        rank = np.asarray(z['candidate_rank_distribution'])
        saved_ids = np.asarray(z['candidate_selected_ids'])
        saved_k = np.asarray(z['categorical_k'])
        members = np.asarray(z['member']).astype(str)
        if not np.all(np.asarray(z['outer_fold']) == args.outer_fold):
            raise RuntimeError('source fold mismatch')
    if not np.array_equal(idx, outer) or not np.array_equal(truth, k[outer]) or not np.array_equal(members, cache['members'][outer].astype(str)):
        raise RuntimeError('source outer alignment mismatch')
    mask = cache['mask'][outer]
    if not np.isfinite(rank).all():
        raise RuntimeError('invalid frozen ranking')
    replay_ids = ranked_ids(rank, mask, saved_k)
    if any(set(a[a>=0]) != set(b[b>=0]) for a,b in zip(replay_ids,saved_ids)):
        raise RuntimeError('frozen ranking does not replay V24 IDs')
    samples, reconstruction = v102._reconstruct_candidates(cache)
    tracks = tuple(t for t in ctx['train_split'] if t.annotation_member in set(members))
    refs = {}
    for t in tracks:
        with zipfile.ZipFile(t.audio_zip) as z, z.open(t.audio_member) as stream, wave.open(stream) as wav:
            n = wav.getnframes()
        refs[t.annotation_member] = _reference_positions(t,n)[0]
    base_map = v177._direct_prediction_map(cache,samples,outer,saved_ids,np.ones(len(outer),bool))
    baseline = score(refs,base_map,2205)
    source_metric = source['strata']['aggregate'][v240.MODEL_KEY]['metrics']['global']
    if any(baseline[x] != source_metric[x] for x in ('true_positive','false_positive','false_negative')):
        raise RuntimeError('baseline metric replay mismatch')
    args.output_dir.mkdir(parents=True)
    weights = {arm: {'fit':arm_weights(arm,k[fit]), 'final':arm_weights(arm,k[final])} for arm in ARMS}
    provenance = {'source_run':SOURCE_RUN,'source_sha':SOURCE_SHA,'source_prediction_sha256':digest(pp),
                  'source_report_sha256':digest(rp),'frozen_rank_sha256':array_digest(rank),
                  'frozen_mask_sha256':array_digest(mask),
                  'frozen_outer_timestamps_sha256':array_digest(np.concatenate([samples[i] for i in outer])),
                  'partition_sha256':{key:array_digest(ctx[key]) for key in ('outer_idx','meta_fit_idx','meta_val_idx','final_fit_idx')}}
    protocol = {'experiment':'v260_count_weighting_ab','only_arm_treatment':'inverse-square-root class weights versus uniform weights for training and inner NLL',
                'arms':list(ARMS),'fresh_count_encoder':True,'pretrained_v24_features_used_for_fit':False,
                'frozen_saved_v24_ranking':True,'count_decoder':'argmax P(K=0..6)',
                'historical_validation_or_locked12_evaluated':False,'outer_used_for_epoch_selection':False,
                'class_weights_fit_partition_only':True,'threshold_tuning':False,'seed':SEED,
                'max_epochs':MAX_EPOCHS,'epoch_selection':'inner arm-specific NLL, patience 4; refit from scratch',
                'class_weights_by_arm':{a:{p:w.tolist() for p,w in tables.items()} for a,tables in weights.items()}}
    report = {'protocol':protocol,'provenance':provenance,'fold':args.outer_fold,'rows':len(outer),
              'baseline_v240':baseline,'baseline_v104':source['strata']['aggregate']['v104']['metrics']['global'],
              'baseline_v240_cardinality':cardinality(k[outer],saved_k),'reconstruction':reconstruction,'arms':{}}
    prediction = {'global_index':outer,'member':members,'k':k[outer],'v240_k':saved_k}
    (args.output_dir/'protocol.json').write_text(json.dumps(report,indent=2)+'\n')
    for arm in ARMS:
        fit_table, final_table = weights[arm]['fit'], weights[arm]['final']
        probe = build_model(arm,SEED+100+args.outer_fold)
        hist = probe.fit(v102._inputs(cache,fit),k[fit],sample_weight=fit_table[k[fit]],
                         validation_data=(v102._inputs(cache,val),k[val],fit_table[k[val]]),
                         epochs=MAX_EPOCHS,batch_size=128,shuffle=True,verbose=2,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,min_delta=0,restore_best_weights=True),
                                    tf.keras.callbacks.TerminateOnNaN()])
        if not all(np.isfinite(values).all() for key, values in hist.history.items() if 'loss' in key):
            raise RuntimeError('nonfinite inner loss')
        epochs = int(np.argmin(hist.history['val_loss']))+1
        del probe
        model = build_model(arm,SEED+1000+args.outer_fold)
        final_hist = model.fit(v102._inputs(cache,final),k[final],sample_weight=final_table[k[final]],
                              epochs=epochs,batch_size=128,shuffle=True,verbose=2,
                              callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        if len(final_hist.history['loss']) != epochs or not np.isfinite(final_hist.history['loss']).all():
            raise RuntimeError('invalid final training')
        probability = np.asarray(model.predict(v102._inputs(cache,outer),batch_size=128,verbose=0))
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(1),1,atol=1e-5):
            raise RuntimeError('invalid count distribution')
        pred = np.argmax(probability,axis=1).astype(np.int32)
        ids = ranked_ids(rank,mask,pred)
        pmap = v177._direct_prediction_map(cache,samples,outer,ids,np.ones(len(outer),bool))
        metric = {str(ms):score(refs,pmap,round(ms*v102.SAMPLE_RATE/1000)) for ms in (5,10,20,50)}
        realized = np.sum(ids>=0,axis=1)
        report['arms'][arm] = {'epochs':epochs,'metrics_by_tolerance_ms':metric,
                               'categorical_k':cardinality(k[outer],pred),'realized_k':cardinality(k[outer],realized),
                               'rows_candidate_shortage':int(np.sum(realized<pred)),
                               'inner_history':hist.history,'final_history':final_hist.history}
        model.save_weights(args.output_dir/f'{arm}.weights.h5')
        prediction[arm+'_probability']=probability; prediction[arm+'_selected_ids']=ids
        (args.output_dir/f'report-fold-{args.outer_fold}.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        np.savez_compressed(args.output_dir/f'predictions-fold-{args.outer_fold}.npz',**prediction)
        print(json.dumps({'fold':args.outer_fold,'arm':arm,'epochs':epochs,'global':metric['50'],
                          'poly_exact':report['arms'][arm]['categorical_k']['poly_exact']}),flush=True)
        del model
    return report


def summarize(args):
    reports=[]; indices=[]
    for fold in range(5):
        paths=list(args.input_dir.glob(f'**/report-fold-{fold}.json'))
        if len(paths)!=1: raise RuntimeError('missing/duplicate fold')
        r=json.loads(paths[0].read_text())
        if r['protocol']['experiment'] != 'v260_count_weighting_ab' or set(r['arms']) != set(ARMS) or r['fold']!=fold: raise RuntimeError('incomplete fold')
        with np.load(paths[0].with_name(f'predictions-fold-{fold}.npz'),allow_pickle=False) as z:
            indices.extend(z['global_index'].tolist())
        reports.append(r)
    if sorted(indices)!=list(range(76768)): raise RuntimeError('outer coverage mismatch')
    result={'source_run':SOURCE_RUN,'outer_rows':len(indices),'folds':reports,
            'baseline_v240':aggregate([r['baseline_v240'] for r in reports]),
            'baseline_v104':aggregate([r['baseline_v104'] for r in reports]),'arms':{}}
    for arm in ARMS:
        rows=[r['arms'][arm] for r in reports]
        result['arms'][arm]={'global':aggregate([r['metrics_by_tolerance_ms']['50'] for r in rows]),
                             'poly_exact':sum(r['categorical_k']['poly_correct'] for r in rows)/sum(r['categorical_k']['poly_rows'] for r in rows),
                             'k_confusion':np.sum([r['categorical_k']['confusion_true_by_predicted'] for r in rows],axis=0).tolist()}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for x in ('dataset-dir','cache-dir','fold-dir','output-dir'): t.add_argument('--'+x,type=Path,required=True)
    t.add_argument('--outer-fold',type=int,required=True)
    s=sub.add_parser('summarize'); s.add_argument('--input-dir',type=Path,required=True); s.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); train(args) if args.command=='train' else summarize(args)
