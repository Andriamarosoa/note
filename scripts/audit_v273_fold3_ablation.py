"""Single outer-fold diagnostic: remove decay evidence at fixed saved weights.

No fit, threshold selection, other-fold evaluation, or model promotion occurs.
Zeroing a learned input is a post-hoc intervention, not a retrained control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import itertools
from pathlib import Path

import numpy as np

from scripts.audit_v273_native_paired_results import load_npz, read_json, require, verify_inventory
from scripts import audit_v270_class_conditional_fusion as v270
from scripts import audit_v271_poly_rescue as v271
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v260_count_weighting as v260
from scripts import train_v273_native_paired as paired
from scripts import train_v273_selective_transition_corrector as v273
from scripts import v273_decay_native_inputs as native
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_native_protocol import load_config, verify_native_cache


FOLD = 3
FOCUS = (50105, 42907)
COMPONENTS = native.COMPONENTS


def load_fold_cache(directory, member_folds):
    """Preserve original shard/row order, retaining inputs only for outer 3."""
    keys = ('spectral', 'sequence', 'mask', 'stats', 'exact', 'members')
    parts = {key: [] for key in keys}
    indices, offset = [], 0
    for path in sorted(Path(directory).rglob('v100-spectral-shard-*.npz')):
        with np.load(path, allow_pickle=False) as z:
            require(int(z['schema_version'][0]) == 1, 'cache schema changed')
            members = z['members'].astype(str)
            selected = np.flatnonzero([member_folds[m] == FOLD for m in members])
            indices.append(offset + selected)
            offset += len(members)
            if len(selected):
                for key in keys:
                    value = z[key]
                    require(len(value) == len(members), 'cache row mismatch: '+key)
                    parts[key].append(value[selected])
    require(offset > 0 and all(parts.values()), 'empty fold cache')
    return ({key: np.concatenate(values) for key, values in parts.items()},
            np.concatenate(indices).astype(np.int64), offset)


def metrics(k, prediction):
    poly = k >= 2
    return {'rows': len(k), 'correct': int(np.sum(prediction == k)),
            'over': int(np.sum(prediction > k)), 'under': int(np.sum(prediction < k)),
            'poly_rows': int(poly.sum()), 'poly_correct': int(np.sum(poly & (prediction == k))),
            'poly_over': int(np.sum(poly & (prediction > k))),
            'poly_under': int(np.sum(poly & (prediction < k)))}


def effect(k, before, after, mask):
    return {'rows': int(mask.sum()),
            'changed': int(np.sum(mask & (before != after))),
            'corrected': int(np.sum(mask & (before != k) & (after == k))),
            'regressed': int(np.sum(mask & (before == k) & (after != k))),
            'still_over': int(np.sum(mask & (after > k))),
            'still_under': int(np.sum(mask & (after < k)))}


def stages(anchor, probability, calibration):
    low = v270.fuse_counts(anchor, np.argmax(probability['v260_uniform'], axis=1), 'low_k_fusion')
    base = v271.apply_poly_rescue(anchor, low, probability['v260_weighted'], calibration['v271']['threshold'])[0]
    final, active, proposal, margin, direction = v273.apply_transition_corrector(
        base, probability['v272_uniform'], calibration['v273']['thresholds'])
    return {'low': low, 'base': base, 'final': final, 'active': active,
            'proposal': proposal, 'margin': margin, 'direction': direction}


def feature_table(spectral):
    values, evidence = paired.evidence_table(spectral)
    reliable = np.empty(len(spectral), dtype=np.int16)
    for start in range(0, len(spectral), 512):
        end = min(start+512, len(spectral))
        _, audit = native.native_decay_features(spectral[start:end])
        reliable[start:end] = audit['reliable'].sum(axis=1)
    return values, reliable, evidence


def run(args):
    import tensorflow as tf
    require(tf.__version__ == '2.15.1', 'TensorFlow 2.15.1 required')
    require(args.fold == FOLD, 'only outer fold 3 is authorized for this audit')
    require(not args.output_dir.exists(), 'output directory already exists')
    files_verified = verify_inventory(args.fold_dir)
    root = args.fold_dir/'paired'
    report = read_json(root/'report-fold-3.json')
    protocol = read_json(root/'protocol.json')
    config = load_config(args.config)
    require(report['fold'] == protocol['fold'] == FOLD, 'wrong saved fold')
    require(report['config_sha256'] == protocol['config_sha256'] == digest(args.config), 'config changed')
    verified = verify_native_cache(args.cache_dir, args.config)
    require(verified == protocol['inputs'], 'inputs differ from training inputs')
    saved = load_npz(root/'predictions-fold-3.npz')
    cache, global_index, total_source_rows = load_fold_cache(args.cache_dir, config['member_folds'])
    require(total_source_rows == config['expected_rows'], 'source row order incomplete')
    require(np.array_equal(global_index, saved['global_index']), 'outer row ordering differs')
    require(v260.array_digest(global_index) == protocol['partition_sha256']['outer_idx'], 'fold coverage mismatch')
    require(np.all(saved['outer_fold'] == FOLD), 'another outer fold present')
    k = np.minimum(cache['exact'], 6).astype(np.int32)
    require(np.array_equal(k, saved['k']), 'saved labels differ from native input labels')
    require(np.array_equal(cache['members'].astype(str), saved['member']), 'member alignment mismatch')
    anchor_file = args.fold_dir/'anchor/v104-nested-eval-3.npz'
    require(digest(anchor_file) == report['shared_outer_anchor_sha256'], 'shared anchor changed')
    anchor = load_npz(anchor_file)
    require(np.array_equal(anchor['global_index'], global_index), 'anchor index mismatch')
    require(np.array_equal(anchor['pred104_deploy'], saved['shared_anchor']), 'anchor prediction mismatch')
    anchor = saved['shared_anchor']
    calibration_path = root/'frozen-calibration.json'
    calibration_hash = digest(calibration_path)
    calibration = read_json(calibration_path)
    require(calibration_hash == report['calibration_sha256'] and calibration == report['calibration'], 'calibration changed')
    args.output_dir.mkdir(parents=True)
    features, reliable, evidence = feature_table(cache['spectral'])
    nonzero = np.any(features != 0, axis=1)
    local_index = np.arange(len(k), dtype=np.int64)
    probabilities = {'control': {}, 'decay': {}, 'decay_zero': {}}
    replays = {}
    for arm in native.ARMS:
        for component in COMPONENTS:
            directory = root/arm/'final'/component
            record = read_json(directory/'training.json')
            require(record == report['training'][arm]['final/'+component], 'training record changed')
            require(digest(directory/'model.weights.h5') == record['weights_sha256'], 'saved weights changed')
            expected = load_npz(directory/'probability.npz')
            require(np.array_equal(expected['global_index'], global_index), 'saved probability ordering differs')
            model, original = native.build_count_model(component, record['seed'])
            require(paired.weights_digest(model) == record['initial_weights_sha256'], 'model initialization differs')
            del original
            model.load_weights(directory/'model.weights.h5')
            before = paired.weights_digest(model)
            projection = model.get_layer('native_decay_projection').get_weights()[0]
            require(np.isclose(np.linalg.norm(projection), record['projection_l2'], rtol=1e-6), 'projection tensor mismatch')
            if arm == 'control':
                require(not np.any(projection), 'control has an active ln projection')
            p = np.asarray(model.predict(
                paired.make_sequence(cache, local_index, features, arm),
                workers=0, max_queue_size=1, verbose=0), dtype=np.float32)
            require(np.allclose(p, expected['probability'], atol=2e-5, rtol=1e-5), 'saved inference does not replay')
            require(np.array_equal(p.argmax(axis=1), expected['probability'].argmax(axis=1)), 'replay argmax differs')
            probabilities[arm][component] = p
            replay = {'max_abs_probability_error': float(np.max(np.abs(p-expected['probability']))),
                      'weight_tensor_sha256': before, 'weight_file_sha256': record['weights_sha256'],
                      'projection_l2': float(np.linalg.norm(projection))}
            if arm == 'decay':
                zero = np.asarray(model.predict(
                    paired.make_sequence(cache, local_index, features, 'control'),
                    workers=0, max_queue_size=1, verbose=0), dtype=np.float32)
                require(np.isfinite(zero).all() and np.allclose(zero.sum(axis=1), 1, atol=1e-5), 'invalid ablated probabilities')
                require(np.allclose(p[~nonzero], zero[~nonzero], atol=1e-7, rtol=1e-7), 'zero-feature rows changed under ablation')
                probabilities['decay_zero'][component] = zero
                replay['ablation_max_abs_probability_change'] = float(np.max(np.abs(p-zero)))
            require(paired.weights_digest(model) == before, 'inference mutated model weights')
            replays[arm+'/'+component] = replay
            write_json(args.output_dir/'inference-progress.json', replays)
            print({'completed': arm+'/'+component, **replay}, flush=True)
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    outputs = {arm: stages(anchor, probabilities[arm], calibration[arm]) for arm in native.ARMS}
    for arm in native.ARMS:
        require(np.array_equal(outputs[arm]['final'], saved[arm]), 'replayed final decisions differ from saved decisions')
    outputs['decay_zero'] = stages(anchor, probabilities['decay_zero'], calibration['decay'])
    masks, intervention_scores = {}, {}
    for switches in itertools.product((False, True), repeat=len(COMPONENTS)):
        key = ''.join('1' if x else '0' for x in switches)
        mixed = {component: probabilities['decay' if on else 'decay_zero'][component]
                 for component, on in zip(COMPONENTS, switches)}
        masks[key] = paired.decode(anchor, mixed, calibration['decay'])
        intervention_scores[key] = metrics(k, masks[key])
    require(np.array_equal(masks['111'], saved['decay']), 'all-on intervention differs')
    require(np.array_equal(masks['000'], outputs['decay_zero']['final']), 'all-off intervention differs')
    poly = k >= 2
    c, d, zero = (outputs[a]['final'] for a in ('control', 'decay', 'decay_zero'))
    newly_up = (poly & (outputs['control']['base'] == k) & (outputs['decay']['base'] == k)
                & (c == k) & (d > k))
    cohorts = {'all_poly': poly, 'decay_poly_overcounts': poly & (d > k),
               'new_poly_regressions': poly & (c == k) & (d != k),
               'new_poly_overcounts': poly & (c == k) & (d > k),
               'new_upward_transitions_from_shared_correct_base': newly_up}
    cohort_results = {}
    for name, selected in cohorts.items():
        cohort_results[name] = {'rows': int(selected.sum()),
                               'with_nonzero_ln': int(np.sum(selected & nonzero)),
                               'zero_ln_global_indices': global_index[selected & ~nonzero].tolist(),
                               'remove_all_ln': effect(k, d, zero, selected),
                               'remove_v272_ln_only': effect(k, d, masks['110'], selected)}
    # Separate the V272 runtime input from already changed upstream decisions.
    fixed_base = outputs['decay']['base']
    v272_fixed_base = {}
    for arm, threshold_arm in itertools.product(('decay', 'decay_zero'), native.ARMS):
        pred = v273.apply_transition_corrector(fixed_base, probabilities[arm]['v272_uniform'],
                                               calibration[threshold_arm]['v273']['thresholds'])[0]
        v272_fixed_base[arm+'/'+threshold_arm] = metrics(k, pred)
    cases = []
    for index in global_index[poly & (c == k) & (d != k)]:
        row = int(np.flatnonzero(global_index == index)[0])
        cases.append({'global_index': int(index), 'member': str(saved['member'][row]), 'true_k': int(k[row]),
                      'ln_nonzero': bool(nonzero[row]), 'ln_max': float(features[row].max()),
                      'reliable_bands': int(reliable[row]),
                      'control': int(c[row]), 'decay': int(d[row]), 'decay_zero': int(zero[row]),
                      'v272_zero_only': int(masks['110'][row]),
                      'control_base': int(outputs['control']['base'][row]),
                      'decay_base': int(outputs['decay']['base'][row])})
    focus = []
    for index in FOCUS:
        row = int(np.flatnonzero(global_index == index)[0])
        _, acoustic = native.native_decay_features(cache['spectral'][row:row+1])
        bands = np.flatnonzero(features[row, 64:] > 0)
        focus.append({'global_index': index, 'member': str(saved['member'][row]), 'true_k': int(k[row]),
                      'ln_nonzero': bool(nonzero[row]), 'ln_max': float(features[row].max()),
                      'reliable_bands': int(reliable[row]),
                      'predictions_by_active_ln_heads': {name: int(pred[row]) for name, pred in masks.items()},
                      'networks': {arm: {'v272_probability': probabilities[arm]['v272_uniform'][row].tolist(),
                                         **{name: (float(value[row]) if name == 'margin' else int(value[row]))
                                            for name, value in outputs[arm].items()}}
                                   for arm in ('control', 'decay', 'decay_zero')},
                      'active_bands': [{'band': int(b), 'mean_feature': float(features[row, b]),
                                        'max_feature': float(features[row, 64+b]),
                                        'power_slope_per_second': float(acoustic['power_slope_per_second'][0, b]),
                                        'log_power_rmse': float(acoustic['log_power_rmse'][0, b]),
                                        'observed_log_power': cache['spectral'][row, :, b, 0].astype(float).tolist()}
                                       for b in bands]})
    require(digest(calibration_path) == calibration_hash, 'audit mutated calibration')
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'audited_outer_folds': [FOLD],
              'training_performed': False, 'thresholds_reselected': False, 'automatic_promotion': False,
              'post_hoc_intervention': True, 'runtime': {'tensorflow': tf.__version__, 'numpy': np.__version__},
              'source_run_id': 35099819151, 'source_commit': 'a192bb3137304b5206fdd45a837995c95f80fcbb',
              'config_sha256': digest(args.config), 'calibration_sha256': calibration_hash,
              'files_verified': files_verified, 'native_input_provenance': verified,
              'evaluated_rows': len(k), 'features': evidence, 'inference_replay': replays,
              'metrics': {arm: metrics(k, value['final']) for arm, value in outputs.items()},
              'stage_poly_correct': {arm: {name: int(np.sum(poly & (value[name] == k))) for name in ('low', 'base', 'final')}
                                     for arm, value in outputs.items()},
              'switch_order': list(COMPONENTS), 'one_means_ln_active_at_saved_decay_weights': True,
              'interventions_fixed_decay_calibration': intervention_scores,
              'cohorts': cohort_results, 'v272_probabilities_x_thresholds_fixed_decay_base': v272_fixed_base,
              'frozen_transition_thresholds': {arm: calibration[arm]['v273']['thresholds'] for arm in native.ARMS},
              'poly_regression_cases': cases, 'focus_cases': focus,
              'limitations': ['Outer fold 3 only; selected after observing regressions.',
                              'Zeroing a feature at learned weights is not retraining without that feature.',
                              'A change under this intervention establishes model sensitivity, not the physical acoustic cause.',
                              'No independent annotation or waveform audit; beating, resonance and onset identity remain unproven.',
                              'The common candidate-source leakage and rebuilt-data limitations of the original paired protocol remain.']}
    write_json(args.output_dir/'audit.json', result)
    arrays = {'global_index': global_index, 'k': k, 'member': saved['member'], 'features': features,
              'reliable_bands': reliable, 'shared_anchor': anchor}
    arrays.update({arm+'__'+name: values for arm, ps in probabilities.items() for name, values in ps.items()})
    arrays.update({'prediction__'+key: value for key, value in masks.items()})
    arrays['prediction__control'] = c
    np.savez_compressed(args.output_dir/'audit-arrays.npz', **arrays)
    write_report(args.output_dir/'audit.md', result)
    print({'fold': FOLD, 'metrics': result['metrics'], 'cohorts': {
        name: {key: value for key, value in record.items() if key != 'zero_ln_global_indices'}
        for name, record in cohort_results.items()}}, flush=True)


def write_report(path, result):
    lines = ['# Audit causal limité au fold externe 3', '',
             'Inférence des six réseaux sauvegardés, puis retrait de l’indice ln dans les trois réseaux du bras ln, à poids et seuils constants. Aucun entraînement ni réétalonnage.', '',
             '| Variante | Exact K polyphonique | Surcomptages polyphoniques | Sous-comptages polyphoniques |',
             '|---|---:|---:|---:|']
    for name, label in [('control', 'Contrôle entraîné sans indice ln'), ('decay', 'Modèle entraîné avec ln'),
                        ('decay_zero', 'Même modèle ln, indice coupé à l’inférence')]:
        m = result['metrics'][name]
        lines.append(f"| {label} | {m['poly_correct']}/{m['poly_rows']} ({100*m['poly_correct']/m['poly_rows']:.4f} %) | {m['poly_over']} | {m['poly_under']} |")
    lines += ['', '## Régressions et action directe de l’indice', '']
    for name in ('new_poly_regressions', 'new_poly_overcounts', 'new_upward_transitions_from_shared_correct_base'):
        r = result['cohorts'][name]
        lines.append(f"- `{name}` : {r['rows']} cas, dont {r['with_nonzero_ln']} avec un indice ln non nul. En coupant tous les indices : {r['remove_all_ln']['corrected']} corrigés ; en coupant seulement l’entrée ln de V272 : {r['remove_v272_ln_only']['corrected']} corrigés.")
    lines += ['', '## Deux cas suivis', '', '| Indice global | Vrai K | Contrôle | Avec ln | Ln coupé | Maximum de l’indice ln |', '|---|---:|---:|---:|---:|---:|']
    for case in result['focus_cases']:
        n = case['networks']
        lines.append(f"| {case['global_index']} | {case['true_k']} | {n['control']['final']} | {n['decay']['final']} | {n['decay_zero']['final']} | {case['ln_max']:.6g} |")
    lines += ['', '## Portée', '',
              'Les probabilités sauvegardées et toutes les décisions finales des deux bras doivent être reproduites avant de conclure. Les poids et seuils sont vérifiés inchangés après l’intervention.', '',
              'Cet audit mesure l’effet direct de l’entrée ln dans les réseaux déjà appris. Les erreurs qui persistent peuvent refléter les autres poids appris et les seuils ; leur persistance ne démontre pas une cause acoustique particulière. Le retrait à l’inférence peut aussi supprimer des informations utiles.', '',
              'Le fold 3 a été choisi après constat des régressions. Aucun résultat des autres folds n’est évalué ici. Ce diagnostic ne constitue ni un nouveau modèle validé ni une justification de promotion.']
    path.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=[FOLD], default=FOLD)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--fold-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
