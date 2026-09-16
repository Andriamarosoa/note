"""Train the V27.3 count networks with paired native decay inputs.

The candidate sources were rebuilt. This is a new paired development
experiment, never a claim that the archived 42.6019% model was restored.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import v273_decay_native_inputs as native
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_class_conditional_fusion as v104
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v260_count_weighting as v260
from scripts import audit_v270_class_conditional_fusion as v270
from scripts import audit_v271_poly_rescue as v271
from scripts import train_v272_poly_conditional_count as v272
from scripts import train_v273_selective_transition_corrector as v273
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_native_protocol import load_config, verify_native_cache


def evidence_table(spectral, batch_size=512):
    """Bound feature-extraction memory; do not convert the whole cache to f64."""
    if len(spectral) == 0 or batch_size <= 0:
        raise ValueError('nonempty spectra and positive batch size required')
    values = np.empty((len(spectral), native.FEATURE_DIM), dtype=np.float32)
    reliable_bands = reliable_rows = nonzero_rows = 0
    for start in range(0, len(spectral), batch_size):
        end = min(start + batch_size, len(spectral))
        feature, audit = native.native_decay_features(spectral[start:end])
        values[start:end] = feature
        reliable_bands += int(audit['reliable'].sum())
        reliable_rows += int(audit['reliable'].any(axis=1).sum())
        nonzero_rows += int(np.any(feature > 0, axis=1).sum())
    return values, {'rows': len(values), 'reliable_rows': reliable_rows,
                    'reliable_bands': reliable_bands, 'nonzero_rows': nonzero_rows,
                    'feature_sha256': v260.array_digest(values)}


def epoch_order(indices, seed, epoch):
    indices = np.asarray(indices, dtype=np.int64)
    return indices[np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(indices))]


def make_sequence(cache, indices, features, arm, *, k=None, table=None,
                  poly=False, seed=0, shuffle=False, batch_size=128):
    """Read only a batch of the large spectral cache and explicitly pair order."""
    import tensorflow as tf
    if arm not in native.ARMS:
        raise ValueError(arm)
    base = np.asarray(indices, dtype=np.int64)
    if len(base) == 0 or len(np.unique(base)) != len(base):
        raise ValueError('empty or duplicate sequence indices')
    if k is not None and (table is None or (poly and np.any(np.asarray(k)[base] < 2))):
        raise ValueError('invalid training targets or weights')

    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch = 0
            self.order = epoch_order(base, seed, 0) if shuffle else base.copy()

        def __len__(self):
            return (len(base) + batch_size - 1) // batch_size

        def __getitem__(self, batch):
            idx = self.order[batch * batch_size:(batch + 1) * batch_size]
            inputs = v102._inputs(cache, idx)
            inputs['decay_evidence'] = (features[idx] if arm == 'decay' else
                                        np.zeros((len(idx), native.FEATURE_DIM), dtype=np.float32))
            if k is None:
                return inputs
            target = np.asarray(k)[idx] - (2 if poly else 0)
            return inputs, target, table[target]

        def on_epoch_end(self):
            self.epoch += 1
            if shuffle:
                self.order = epoch_order(base, seed, self.epoch)

    return Batches()


def weights_digest(model):
    h = hashlib.sha256()
    for weight in model.get_weights():
        h.update(str((weight.shape, str(weight.dtype))).encode())
        h.update(np.ascontiguousarray(weight).tobytes())
    return h.hexdigest()


def train_component(ctx, features, arm, component, phase, epochs, fold, output):
    import tensorflow as tf
    poly = component == 'v272_uniform'
    k = ctx['k']
    fit = ctx['meta_fit_idx'] if phase == 'inner' else ctx['final_fit_idx']
    predict = ctx['meta_val_idx'] if phase == 'inner' else ctx['outer_idx']
    if poly:
        fit = fit[k[fit] >= 2]
        table = v272.poly_class_weights(k[fit], 'uniform')
        seed = v272.SEED + (100 + 10 * fold if phase == 'inner' else 1000 + fold)
    else:
        table = v260.arm_weights(component.removeprefix('v260_'), k[fit])
        seed = v260.SEED + (100 if phase == 'inner' else 1000) + fold
    maximum = v272.MAX_EPOCHS if poly else v260.MAX_EPOCHS
    if not 1 <= epochs <= maximum:
        raise ValueError('invalid frozen epoch budget')
    model, original = native.build_count_model(component, seed)
    initial = weights_digest(model)
    del original
    train = make_sequence(ctx['cache'], fit, features, arm, k=k, table=table,
                          poly=poly, seed=seed, shuffle=True)
    kwargs = {}
    if phase == 'inner':
        val = predict[k[predict] >= 2] if poly else predict
        kwargs['validation_data'] = make_sequence(
            ctx['cache'], val, features, arm, k=k, table=table, poly=poly)
    print(json.dumps({'arm': arm, 'component': component, 'phase': phase,
                      'epochs': epochs, 'fit_rows': len(fit), 'seed': seed}), flush=True)
    history = model.fit(train, epochs=epochs, shuffle=False, workers=0,
                        max_queue_size=1, verbose=2,
                        callbacks=[tf.keras.callbacks.TerminateOnNaN()], **kwargs)
    if len(history.history.get('loss', [])) != epochs or any(
            not np.isfinite(values).all() for values in history.history.values()):
        raise RuntimeError('incomplete or nonfinite count training')
    probability = np.asarray(model.predict(
        make_sequence(ctx['cache'], predict, features, arm),
        workers=0, max_queue_size=1, verbose=0), dtype=np.float32)
    validator = v272._validate_poly_probability if poly else v271._validate_probability
    validator(probability, f'{arm}/{component}/{phase}')
    if len(probability) != len(predict):
        raise RuntimeError('count prediction coverage mismatch')
    root = output/arm/phase/component
    root.mkdir(parents=True)
    weight_file = root/'model.weights.h5'
    model.save_weights(weight_file)
    np.savez_compressed(root/'probability.npz', global_index=predict, probability=probability)
    order_hash = hashlib.sha256()
    for epoch in range(epochs):
        order_hash.update(epoch_order(fit, seed, epoch).tobytes())
    projection = np.asarray(model.get_layer('native_decay_projection').get_weights()[0])
    if arm == 'control' and np.any(projection != 0):
        raise RuntimeError('zero-evidence control projection changed')
    report = {'seed': seed, 'epochs': epochs, 'fit_rows': len(fit),
              'initial_weights_sha256': initial, 'weights_sha256': digest(weight_file),
              'epoch_row_order_sha256': order_hash.hexdigest(),
              'class_weights': table.tolist(), 'history': history.history,
              'projection_l2': float(np.linalg.norm(projection))}
    write_json(root/'training.json', report)
    del model, train, kwargs
    tf.keras.backend.clear_session()
    gc.collect()
    return probability, report


def calibrate(k_inner, anchor, probabilities):
    """Use only inner labels for the historical V27.1 and V27.3 rules."""
    low = v270.fuse_counts(anchor, np.argmax(probabilities['v260_uniform'], axis=1), 'low_k_fusion')
    rescue = v271.select_threshold(k_inner, anchor, low, probabilities['v260_weighted'])
    base, _, _, _ = v271.apply_poly_rescue(anchor, low, probabilities['v260_weighted'], rescue['threshold'])
    transitions = v273.select_transition_thresholds(k_inner, base, probabilities['v272_uniform'])
    return {'v271': rescue, 'v273': transitions}


def decode(anchor, probabilities, calibration):
    """Labels are deliberately absent from the runtime decode interface."""
    low = v270.fuse_counts(anchor, np.argmax(probabilities['v260_uniform'], axis=1), 'low_k_fusion')
    base, _, _, _ = v271.apply_poly_rescue(anchor, low, probabilities['v260_weighted'], calibration['v271']['threshold'])
    return v273.apply_transition_corrector(base, probabilities['v272_uniform'], calibration['v273']['thresholds'])[0]


def run(args):
    import tensorflow as tf
    if tf.__version__ != '2.15.1':
        raise RuntimeError('the historical TensorFlow 2.15.1 runtime is required')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cfg = load_config(args.fold_manifest)
    verified = verify_native_cache(args.cache_dir, args.fold_manifest)
    ctx = v171._fold_context(args)
    outer, fit, val, final = v260.validate_partitions(ctx)
    k, members = ctx['k'], ctx['members']
    cfg_fold = cfg['folds'][args.outer_fold]
    args.output_dir.mkdir(parents=True)
    features, evidence = evidence_table(ctx['cache']['spectral'])
    write_json(args.output_dir/'protocol.json', {
        'experiment': 'v273_native_decay_paired_rebuilt_sources',
        'config_sha256': digest(args.fold_manifest), 'fold': args.outer_fold,
        'inputs': verified, 'evidence': evidence, 'epochs': cfg_fold['epochs'],
        'pairing': 'identical compositions, examples, seeds, epoch order, architecture and epoch budgets',
        'control': 'zero acoustic decay evidence', 'decay': 'native acoustic decay evidence',
        'additional_output_corrector': False, 'historical_v273_reproduced': False,
        'partition_sha256': {name: v260.array_digest(ctx[name]) for name in
                            ('outer_idx', 'meta_fit_idx', 'meta_val_idx', 'final_fit_idx')},
    })
    inner, inner_paths, meta = v271._load_inner_v104(args.expert_dir, args.outer_fold, final, val, k, members)
    shared_probability, anchor_epochs = v271._inner_v104_probability(inner, meta)
    inner_anchor = np.argmax(shared_probability, axis=1).astype(np.int32)
    np.savez_compressed(args.output_dir/'shared-inner-anchor.npz', global_index=val, probability=shared_probability)
    calibration, training = {}, {arm: {} for arm in native.ARMS}
    for arm in native.ARMS:
        probabilities = {}
        for component in native.COMPONENTS:
            probabilities[component], record = train_component(
                ctx, features, arm, component, 'inner', cfg_fold['epochs'][component], args.outer_fold, args.output_dir)
            training[arm]['inner/'+component] = record
        calibration[arm] = calibrate(k[val], inner_anchor, probabilities)
    # Freeze both sets of thresholds before any outer count prediction.
    write_json(args.output_dir/'frozen-calibration.json', calibration)
    frozen_hash = digest(args.output_dir/'frozen-calibration.json')
    paths = list(args.anchor_dir.rglob(f'v104-nested-eval-{args.outer_fold}.npz'))
    if len(paths) != 1:
        raise RuntimeError('expected exactly one shared deployment anchor')
    with np.load(paths[0], allow_pickle=False) as z:
        if not np.array_equal(z['global_index'], outer) or not np.array_equal(z['k'], k[outer]) or not np.array_equal(z['member'].astype(str), members[outer]):
            raise RuntimeError('shared outer anchor alignment mismatch')
        outer_anchor = np.asarray(z['pred104_deploy'], dtype=np.int32)
    predictions, results = {}, {}
    for arm in native.ARMS:
        probabilities = {}
        for component in native.COMPONENTS:
            probabilities[component], record = train_component(
                ctx, features, arm, component, 'final', cfg_fold['epochs'][component], args.outer_fold, args.output_dir)
            training[arm]['final/'+component] = record
        predictions[arm] = decode(outer_anchor, probabilities, calibration[arm])
        results[arm] = {'cardinality': v270.cardinality_report(k[outer], predictions[arm]),
                        'event_50ms': v104._metrics_for_indices(ctx['cache'], ctx['train_split'], outer, predictions[arm])['global']}
    for key, a in training['control'].items():
        b = training['decay'][key]
        for paired in ('seed', 'epochs', 'fit_rows', 'initial_weights_sha256', 'epoch_row_order_sha256', 'class_weights'):
            if a[paired] != b[paired]:
                raise RuntimeError(f'paired training mismatch: {key}/{paired}')
    if digest(args.output_dir/'frozen-calibration.json') != frozen_hash:
        raise RuntimeError('calibration changed during outer evaluation')
    report = {'experiment': 'v273_native_decay_paired_rebuilt_sources', 'fold': args.outer_fold,
              'config_sha256': digest(args.fold_manifest), 'calibration_sha256': frozen_hash,
              'historical_v273_reproduced': False, 'additional_output_corrector': False,
              'outer_labels_used_for_selection': False, 'identical_pairing_verified': True,
              'shared_inner_anchor_epochs': anchor_epochs, 'evidence': evidence,
              'shared_outer_anchor_sha256': digest(paths[0]),
              'inner_expert_sha256': {p.name: digest(p) for p in inner_paths},
              'calibration': calibration, 'training': training, 'arms': results,
              'historical_reference_different_cluster_population': cfg_fold['archived_cardinality']}
    np.savez_compressed(args.output_dir/f'predictions-fold-{args.outer_fold}.npz',
                        global_index=outer, outer_fold=np.full(len(outer), args.outer_fold, dtype=np.int16),
                        member=members[outer], k=k[outer], shared_anchor=outer_anchor,
                        control=predictions['control'], decay=predictions['decay'])
    write_json(args.output_dir/f'report-fold-{args.outer_fold}.json', report)
    print(json.dumps({'fold': args.outer_fold, 'arms': results}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-dir', 'cache-dir', 'expert-dir', 'anchor-dir', 'fold-manifest', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--outer-fold', type=int, choices=range(5), required=True)
    run(p.parse_args())
