"""Repair ln adaptation by freezing the trained control and its calibration.

Only outer fold 3 is evaluated. Hyperparameters are selected using its inner
validation partition before final fitting/evaluation. The outer fold was
previously audited, so this remains development evidence, not a fresh test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
from pathlib import Path
import shutil

import numpy as np

from scripts.audit_v273_native_paired_results import load_npz, read_json, require, verify_inventory
from scripts.audit_v273_fold3_ablation import metrics
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v260_count_weighting as v260
from scripts import train_v272_poly_conditional_count as v272
from scripts import train_v273_native_paired as paired
from scripts import v273_decay_native_inputs as native
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_native_protocol import load_config, verify_native_cache


FOLD = 3
TEACHER_WEIGHTS = (0.0, 1.0)
STRENGTHS = (0.25, 0.5, 1.0)


def candidate_name(teacher_weight, strength):
    return f'teacher{teacher_weight:g}_strength{strength:g}'


def select_inner(records):
    """Labels/metrics from outer evaluation cannot enter this interface."""
    base = records['control']['metrics']
    references = [base, records['control'].get('archived_metrics', base)]
    eligible = []
    for name, record in records.items():
        if name == 'control':
            continue
        m = record['metrics']
        require(m['rows'] == base['rows'] and m['poly_rows'] == base['poly_rows'], 'selection populations differ')
        if all(m['poly_correct'] > ref['poly_correct'] and m['correct'] >= ref['correct']
               and m['poly_over'] <= ref['poly_over'] and m['over'] <= ref['over'] for ref in references):
            eligible.append(name)
    if not eligible:
        return 'control'
    return max(eligible, key=lambda name: (
        records[name]['metrics']['poly_correct'], records[name]['metrics']['correct'],
        -records[name]['metrics']['poly_over'], -records[name]['metrics']['over'],
        -records[name]['strength'], records[name]['teacher_weight']))


def freeze_and_build_head(extended, original, component):
    """Share the existing bias-free projection; freeze every original weight."""
    import tensorflow as tf
    for layer in extended.layers:
        layer.trainable = layer.name == 'native_decay_projection'
    projection = extended.get_layer('native_decay_projection')
    context_dim = int(original.get_layer('v240_cardinality_context').output.shape[-1])
    context = tf.keras.Input((context_dim,), name='frozen_context')
    evidence = tf.keras.Input((native.FEATURE_DIM,), name='decay_evidence')
    hidden = tf.keras.layers.Add()([context, projection(evidence)])
    head_name = 'v272_poly_cardinality' if component == 'v272_uniform' else 'cardinality'
    for name in ('v240_cardinality_hidden1', 'v240_cardinality_dropout',
                 'v240_cardinality_hidden2', head_name):
        # Frozen dropout is explicitly disabled, including during adaptation.
        hidden = original.get_layer(name)(hidden, training=False)
    head = tf.keras.Model([context, evidence], hidden)
    require(len(head.trainable_weights) == 1 and head.trainable_weights[0] is projection.kernel,
            'a legacy weight would be trained')
    require(not original.trainable_weights, 'legacy model is not frozen')
    return head


def cached_batches(context, features, targets, weights, seed, batch_size=128):
    import tensorflow as tf
    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch = 0
            self.order = paired.epoch_order(np.arange(len(context)), seed, 0)

        def __len__(self):
            return (len(context)+batch_size-1)//batch_size

        def __getitem__(self, batch):
            idx = self.order[batch*batch_size:(batch+1)*batch_size]
            return {'frozen_context': context[idx], 'decay_evidence': features[idx]}, targets[idx], weights[idx]

        def on_epoch_end(self):
            self.epoch += 1
            self.order = paired.epoch_order(np.arange(len(context)), seed, self.epoch)
    return Batches()


def adapt_component(ctx, features, component, phase, teacher_weights, strengths, root, output):
    import tensorflow as tf
    directory = root/'control'/phase/component
    record = read_json(directory/'training.json')
    require(digest(directory/'model.weights.h5') == record['weights_sha256'], 'source weights changed')
    fit_all = ctx['fit'] if phase == 'inner' else ctx['final_fit']
    evaluate = ctx['val'] if phase == 'inner' else ctx['outer']
    poly = component == 'v272_uniform'
    if poly:
        fit_all = fit_all[ctx['k'][fit_all] >= 2]
        table = v272.poly_class_weights(ctx['k'][fit_all], 'uniform')
    else:
        table = v260.arm_weights(component.removeprefix('v260_'), ctx['k'][fit_all])
    require(len(fit_all) == record['fit_rows'], 'fit population differs from frozen control')
    require(np.array_equal(table, record['class_weights']), 'class weighting differs')
    # All omitted rows have exactly zero gradient for the bias-free projection.
    # Filtering is predeclared; this changes the Adam schedule from full training.
    fit = fit_all[np.any(features[fit_all] != 0, axis=1)]
    require(len(fit) > 0 and not np.intersect1d(fit, evaluate).size, 'invalid adapter split')
    model, original = native.build_count_model(component, record['seed'])
    require(paired.weights_digest(model) == record['initial_weights_sha256'], 'source architecture differs')
    model.load_weights(directory/'model.weights.h5')
    projection = model.get_layer('native_decay_projection')
    require(not np.any(projection.get_weights()[0]), 'control ln projection must be zero')
    original_hash = paired.weights_digest(original)
    probe = tf.keras.Model(model.inputs, [original.get_layer('v240_cardinality_context').output, model.output])
    train_context, train_base = probe.predict(paired.make_sequence(ctx['cache'], fit, features, 'control'),
                                               workers=0, max_queue_size=1, verbose=0)
    val_context, val_base = probe.predict(paired.make_sequence(ctx['cache'], evaluate, features, 'control'),
                                           workers=0, max_queue_size=1, verbose=0)
    saved = load_npz(directory/'probability.npz')
    require(np.array_equal(saved['global_index'], evaluate), 'saved evaluation ordering differs')
    require(np.allclose(val_base, saved['probability'], atol=2e-5, rtol=1e-5), 'control inference does not replay')
    require(np.array_equal(val_base.argmax(1), saved['probability'].argmax(1)), 'control argmax changed')
    head = freeze_and_build_head(model, original, component)
    # Compare zero-input interventions through exactly the same cached graph.
    # Tiny CPU differences at archived calibration boundaries are recorded;
    # selection must improve both the live and archived inner control.
    cached_base = np.asarray(head.predict({'frozen_context': val_context,
        'decay_evidence': np.zeros_like(features[evaluate])}, batch_size=128, verbose=0))
    require(np.allclose(cached_base, val_base, atol=2e-5, rtol=1e-5), 'cached control differs')
    out = output/phase/component
    out.mkdir(parents=True)
    result, reports = {}, {}
    classes = 5 if poly else 7
    labels = ctx['k'][fit]-(2 if poly else 0)
    val_inputs = {'frozen_context': val_context, 'decay_evidence': features[evaluate]}
    for teacher_weight in teacher_weights:
        projection.set_weights([np.zeros_like(projection.get_weights()[0])])
        head.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss='categorical_crossentropy')
        target = (np.eye(classes, dtype=np.float32)[labels] + teacher_weight*train_base)/(1+teacher_weight)
        require(np.isfinite(target).all() and np.allclose(target.sum(1), 1, atol=1e-5), 'invalid soft labels')
        batches = cached_batches(train_context, features[fit], target, table[labels], record['seed'])
        print({'phase': phase, 'component': component, 'teacher_weight': teacher_weight,
               'trainable_tensors': len(head.trainable_weights), 'active_fit_rows': len(fit),
               'epochs': record['epochs']}, flush=True)
        history = head.fit(batches, epochs=record['epochs'], shuffle=False, workers=0,
                           max_queue_size=1, verbose=2, callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        require(len(history.history['loss']) == record['epochs'] and
                np.isfinite(history.history['loss']).all(), 'incomplete adaptation')
        require(paired.weights_digest(original) == original_hash, 'adaptation changed a legacy weight')
        kernel = projection.get_weights()[0].copy()
        np.save(out/f'projection-teacher{teacher_weight:g}.npy', kernel, allow_pickle=False)
        for strength in strengths:
            name = candidate_name(teacher_weight, strength)
            projection.set_weights([kernel*strength])
            probability = np.asarray(head.predict(val_inputs, batch_size=128, verbose=0), dtype=np.float32)
            require(np.isfinite(probability).all() and np.allclose(probability.sum(1), 1, atol=1e-5), 'invalid probabilities')
            zero = ~np.any(features[evaluate] != 0, axis=1)
            require(np.array_equal(probability[zero], cached_base[zero]), 'zero-input parity broken')
            result[name] = probability
            np.savez_compressed(out/f'{name}-probability.npz', global_index=evaluate, probability=probability)
        reports[str(teacher_weight)] = {'source_weight_sha256': record['weights_sha256'],
            'legacy_weight_tensor_sha256_before': original_hash,
            'legacy_weight_tensor_sha256_after': paired.weights_digest(original),
            'source_fit_rows': len(fit_all), 'active_fit_rows': len(fit),
            'fit_global_index_sha256': v260.array_digest(fit), 'seed': record['seed'],
            'epochs': record['epochs'], 'class_weights': table.tolist(),
            'projection_l2': float(np.linalg.norm(kernel)), 'loss': history.history['loss'],
            'baseline_replay_max_error': float(np.max(np.abs(val_base-saved['probability'])))}
        if phase == 'final':
            require(len(teacher_weights) == len(strengths) == 1, 'final model was not selected internally')
            # Save a full native model compatible with the existing builder.
            full_probability = np.asarray(model.predict(
                paired.make_sequence(ctx['cache'], evaluate, features, 'decay'),
                workers=0, max_queue_size=1, verbose=0), dtype=np.float32)
            require(np.allclose(full_probability, probability, atol=2e-5, rtol=1e-5), 'cached/full model mismatch')
            require(np.array_equal(full_probability.argmax(1), probability.argmax(1)), 'full model argmax differs')
            result[name] = full_probability
            np.savez_compressed(out/f'{name}-probability.npz', global_index=evaluate, probability=full_probability)
            model.save_weights(out/'model.weights.h5')
            reports[str(teacher_weight)]['exported_model_sha256'] = digest(out/'model.weights.h5')
    require(paired.weights_digest(original) == original_hash, 'legacy weights changed')
    write_json(out/'training.json', reports)
    del model, original, probe, head, batches
    tf.keras.backend.clear_session()
    gc.collect()
    return result, (cached_base if phase == 'inner' else val_base)


def run(args):
    import tensorflow as tf
    require(tf.__version__ == '2.15.1', 'TensorFlow 2.15.1 required')
    require(args.fold == FOLD and not args.output_dir.exists(), 'wrong fold or existing output')
    verify_inventory(args.fold_dir)
    root = args.fold_dir/'paired'
    protocol, source_report = read_json(root/'protocol.json'), read_json(root/'report-fold-3.json')
    require(protocol['fold'] == source_report['fold'] == FOLD, 'wrong source fold')
    require(protocol['config_sha256'] == digest(args.config), 'source config differs')
    verified = verify_native_cache(args.cache_dir, args.config)
    require(verified == protocol['inputs'], 'source cache differs')
    cfg = load_config(args.config)
    cache = v100._load_spectral_caches(args.cache_dir)
    row_fold = np.array([cfg['member_folds'][str(m)] for m in cache['members']], dtype=np.int16)
    meta = cfg['folds'][FOLD]['meta_fold']
    ctx = {'cache': cache, 'k': np.minimum(cache['exact'], 6).astype(np.int32),
           'fit': np.flatnonzero((row_fold != FOLD) & (row_fold != meta)),
           'val': np.flatnonzero(row_fold == meta), 'final_fit': np.flatnonzero(row_fold != FOLD),
           'outer': np.flatnonzero(row_fold == FOLD)}
    for name, old in (('fit', 'meta_fit_idx'), ('val', 'meta_val_idx'), ('final_fit', 'final_fit_idx'), ('outer', 'outer_idx')):
        require(v260.array_digest(ctx[name]) == protocol['partition_sha256'][old], 'partition differs: '+name)
    features, evidence = paired.evidence_table(cache['spectral'])
    require(evidence == protocol['evidence'], 'ln input implementation differs')
    calibration_file = root/'frozen-calibration.json'
    calibration_hash = digest(calibration_file)
    require(calibration_hash == source_report['calibration_sha256'], 'source calibration differs')
    calibration = read_json(calibration_file)['control']
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'protocol.json', {
        'outer_fold': FOLD, 'inner_validation_fold': meta, 'config_sha256': digest(args.config),
        'legacy_weights_frozen': True, 'control_thresholds_frozen': True,
        'additional_output_corrector': False, 'teacher_weights': list(TEACHER_WEIGHTS),
        'strengths': list(STRENGTHS), 'epochs': cfg['folds'][FOLD]['epochs'],
        'learning_rate': 2e-4, 'batch_size': 128, 'dropout_disabled': True,
        'fit_only_nonzero_evidence': True, 'automatic_promotion': False,
        'outer_fold_previously_observed': True, 'evidence': evidence})
    inner_anchor_pack = load_npz(root/'shared-inner-anchor.npz')
    require(np.array_equal(inner_anchor_pack['global_index'], ctx['val']), 'inner anchor ordering differs')
    inner_anchor = inner_anchor_pack['probability'].argmax(1)
    inner_probabilities = {candidate_name(t, s): {} for t, s in itertools.product(TEACHER_WEIGHTS, STRENGTHS)}
    baseline = {}
    for component in native.COMPONENTS:
        probabilities, baseline[component] = adapt_component(
            ctx, features, component, 'inner', TEACHER_WEIGHTS, STRENGTHS, root, args.output_dir)
        for name, p in probabilities.items():
            inner_probabilities[name][component] = p
    inner_control = paired.decode(inner_anchor, baseline, calibration)
    k_val = ctx['k'][ctx['val']]
    base_score = metrics(k_val, inner_control)
    expected = source_report['calibration']['control']['v273']['selected']
    archived_probabilities = {c: load_npz(root/'control/inner'/c/'probability.npz')['probability'] for c in native.COMPONENTS}
    archived_control = paired.decode(inner_anchor, archived_probabilities, calibration)
    archived_score = metrics(k_val, archived_control)
    require(all(archived_score[key] == expected[key] for key in ('rows', 'correct', 'poly_rows', 'poly_correct', 'over', 'under')),
            'archived inner control decisions differ')
    records = {'control': {'teacher_weight': None, 'strength': 0.0, 'metrics': base_score,
                          'archived_metrics': archived_score}}
    for teacher, strength in itertools.product(TEACHER_WEIGHTS, STRENGTHS):
        name = candidate_name(teacher, strength)
        pred = paired.decode(inner_anchor, inner_probabilities[name], calibration)
        require(np.array_equal(pred[~np.any(features[ctx['val']] != 0, axis=1)],
                               inner_control[~np.any(features[ctx['val']] != 0, axis=1)]), 'zero-evidence decisions changed internally')
        records[name] = {'teacher_weight': teacher, 'strength': strength, 'metrics': metrics(k_val, pred)}
    selected = select_inner(records)
    selection = {'selected': selected, 'candidates': records,
                 'rule': 'strictly improve inner poly_correct over both live and archived control; do not reduce global correct or increase global/poly overcounts; tie by smaller strength',
                 'inner_control_numerical_decision_changes': ctx['val'][inner_control != archived_control].tolist(),
                 'outer_labels_used': False, 'calibration_sha256': calibration_hash}
    write_json(args.output_dir/'selection.json', selection)
    selection_hash = digest(args.output_dir/'selection.json')
    print({'inner_selection': selection}, flush=True)
    final_probabilities, final_base = {}, {}
    if selected == 'control':
        for component in native.COMPONENTS:
            p = load_npz(root/'control/final'/component/'probability.npz')
            require(np.array_equal(p['global_index'], ctx['outer']), 'control outer ordering differs')
            final_probabilities[component] = final_base[component] = p['probability']
            out = args.output_dir/'final'/component
            out.mkdir(parents=True)
            shutil.copyfile(root/'control/final'/component/'model.weights.h5', out/'model.weights.h5')
    else:
        chosen = records[selected]
        for component in native.COMPONENTS:
            p, final_base[component] = adapt_component(
                ctx, features, component, 'final', [chosen['teacher_weight']], [chosen['strength']], root, args.output_dir)
            final_probabilities[component] = p[selected]
    require(digest(args.output_dir/'selection.json') == selection_hash, 'selection changed after final fitting')
    saved = load_npz(root/'predictions-fold-3.npz')
    idx, k = ctx['outer'], ctx['k'][ctx['outer']]
    require(np.array_equal(saved['global_index'], idx) and np.array_equal(saved['k'], k), 'outer ground truth differs')
    require(np.array_equal(saved['member'], cache['members'][idx].astype(str)), 'outer member ordering differs')
    control = paired.decode(saved['shared_anchor'], final_base, calibration)
    prediction = paired.decode(saved['shared_anchor'], final_probabilities, calibration)
    require(np.array_equal(control, saved['control']), 'outer control decisions differ')
    zero = ~np.any(features[idx] != 0, axis=1)
    require(np.array_equal(prediction[zero], control[zero]), 'zero-evidence safety property failed')
    require(digest(calibration_file) == calibration_hash, 'calibration was altered')
    new_old_over = (k >= 2) & (saved['control'] == k) & (saved['decay'] > k)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'outer_folds_evaluated': [FOLD],
              'selected_before_outer_evaluation': selected, 'selection_sha256': selection_hash,
              'automatic_promotion': False, 'zero_evidence_rows': int(zero.sum()),
              'zero_evidence_decision_changes': int(np.sum(prediction[zero] != control[zero])),
              'metrics': {'control': metrics(k, control), 'previous_full_retraining_ln': metrics(k, saved['decay']),
                          'selected_frozen_model': metrics(k, prediction)},
              'vs_control_poly_corrected': int(np.sum((k >= 2) & (control != k) & (prediction == k))),
              'vs_control_poly_regressed': int(np.sum((k >= 2) & (control == k) & (prediction != k))),
              'previous_35_new_overcounts_corrected': int(np.sum(new_old_over & (prediction == k))),
              'previous_35_new_overcounts_still_over': int(np.sum(new_old_over & (prediction > k))),
              'new_poly_overcounts_vs_control': int(np.sum((k >= 2) & (control == k) & (prediction > k))),
              'focus': [{'global_index': i, 'truth': int(k[j]), 'control': int(control[j]),
                         'previous_ln': int(saved['decay'][j]), 'selected': int(prediction[j])}
                        for i in (50105, 42907) for j in np.flatnonzero(idx == i)],
              'limits': ['Single previously audited outer fold; development result.',
                         'Shared upstream sources retain the limitations of the rebuilt paired experiment.',
                         'Frozen adaptation repairs zero-input drift; it does not guarantee improvement on nonzero inputs.']}
    write_json(args.output_dir/'report.json', report)
    write_json(args.output_dir/'calibration.json', calibration)
    np.savez_compressed(args.output_dir/'predictions.npz', global_index=idx, k=k, member=saved['member'],
                        shared_anchor=saved['shared_anchor'], control=control, previous_ln=saved['decay'],
                        selected=prediction, ln_nonzero=~zero,
                        **{component: p for component, p in final_probabilities.items()})
    lines = ['# Correction de la dérive ln — fold 3', '',
             f"Choix effectué sur la validation interne : `{selected}`. Poids du contrôle et seuils figés.", '',
             '| Variante | Exact K polyphonique | Surcomptages polyphoniques |', '|---|---:|---:|']
    for name, m in report['metrics'].items():
        lines.append(f"| {name} | {m['poly_correct']}/{m['poly_rows']} ({100*m['poly_correct']/m['poly_rows']:.4f} %) | {m['poly_over']} |")
    lines += ['', f"Parité lorsque ln=0 : {report['zero_evidence_decision_changes']} changements sur {report['zero_evidence_rows']} exemples.",
              f"Parmi les 35 nouveaux surcomptages précédents : {report['previous_35_new_overcounts_corrected']} corrigés.", '',
              'Les résultats de ce fold déjà examiné constituent une vérification de développement. Aucune promotion automatique.']
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print({'final_report': report}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', type=int, choices=[FOLD], default=FOLD)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--fold-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    p.add_argument('--output-dir', type=Path, required=True)
    run(p.parse_args())
