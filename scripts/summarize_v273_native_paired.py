"""Aggregate the paired native experiment without promoting a new reference."""
import argparse
from pathlib import Path
import json
import numpy as np

from scripts import audit_v270_class_conditional_fusion as metrics
from scripts import train_v273_selective_transition_corrector as v273
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_boundaries import group_stem
from scripts.v273_native_protocol import load_config


def summarize(input_dir, config, output_dir):
    cfg = load_config(config)
    reports, parts = [], []
    for fold in range(5):
        rp = list(input_dir.rglob(f'report-fold-{fold}.json'))
        pp = list(input_dir.rglob(f'predictions-fold-{fold}.npz'))
        if len(rp) != 1 or len(pp) != 1:
            raise RuntimeError('expected one paired result for every fold')
        r = json.loads(rp[0].read_text())
        if r['experiment'] != 'v273_native_decay_paired_rebuilt_sources' or r['fold'] != fold:
            raise RuntimeError('paired report identity mismatch')
        if r['config_sha256'] != digest(config) or r['identical_pairing_verified'] is not True or r['outer_labels_used_for_selection'] is not False:
            raise RuntimeError('paired protocol mismatch')
        with np.load(pp[0], allow_pickle=False) as z:
            a = {key: z[key] for key in z.files}
        if not np.all(a['outer_fold'] == fold):
            raise RuntimeError('outer prediction fold mismatch')
        if any(cfg['member_folds'].get(str(m)) != fold for m in a['member']):
            raise RuntimeError('historical composition assignment changed')
        for arm in ('control', 'decay'):
            if metrics.cardinality_report(a['k'], a[arm]) != r['arms'][arm]['cardinality']:
                raise RuntimeError('reported count scores differ from saved predictions')
        reports.append(r)
        parts.append(a)
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(merged['global_index'])
    merged = {key: value[order] for key, value in merged.items()}
    if not np.array_equal(merged['global_index'], np.arange(cfg['expected_rows'])):
        raise RuntimeError('outer predictions must cover every rebuilt row exactly once')
    k = merged['k']
    if int(np.sum(k >= 2)) != cfg['expected_poly_rows']:
        raise RuntimeError('polyphonic outer coverage mismatch')
    arms = {arm: {'cardinality': metrics.cardinality_report(k, merged[arm]),
                  'event_50ms': v273._aggregate_event([r['arms'][arm]['event_50ms'] for r in reports])}
            for arm in ('control', 'decay')}
    poly = k >= 2
    improved = (merged['decay'] == k) & (merged['control'] != k)
    regressed = (merged['control'] == k) & (merged['decay'] != k)
    groups = np.array([group_stem(str(m)) for m in merged['member']])
    per_group = []
    for group in sorted(set(groups)):
        mask = (groups == group) & poly
        per_group.append({'composition': group, 'poly_rows': int(mask.sum()),
                          'net_poly_correct': int(improved[mask].sum() - regressed[mask].sum())})
    rows = np.array([g['poly_rows'] for g in per_group])
    net = np.array([g['net_poly_correct'] for g in per_group])
    choices = np.random.default_rng(27361).integers(0, len(rows), size=(10000, len(rows)))
    denominators = rows[choices].sum(axis=1)
    bootstrap = 100 * net[choices].sum(axis=1)[denominators > 0] / denominators[denominators > 0]
    delta = 100 * (arms['decay']['cardinality']['poly_exact'] - arms['control']['cardinality']['poly_exact'])
    result = {
        'experiment': 'v273_native_decay_paired_rebuilt_sources', 'completed_folds': 5,
        'historical_v273_reproduced': False, 'automatic_promotion': False,
        'config_sha256': digest(config), 'arms': arms,
        'paired_delta_poly_percentage_points': delta,
        'paired_poly_corrected_rows': int(improved[poly].sum()),
        'paired_poly_regressed_rows': int(regressed[poly].sum()),
        'paired_delta_global_percentage_points': 100 * (arms['decay']['cardinality']['exact'] - arms['control']['cardinality']['exact']),
        'composition_bootstrap_95_percentile_pp': np.quantile(bootstrap, [.025, .975]).tolist(),
        'uncertainty_scope': 'one training seed; resample compositions, not independent clusters',
        'archived_reference_different_cluster_population': cfg['reference'],
        'archived_reference_comparison_is_paired': False,
        'per_composition': per_group,
        'per_fold': {str(r['fold']): r['arms'] for r in reports},
        'verdict': 'positive paired development gain' if delta > 0 else 'no paired polyphonic Exact-K gain',
    }
    if output_dir.exists():
        raise FileExistsError(output_dir)
    write_json(output_dir/'summary.json', result)
    np.savez_compressed(output_dir/'paired-predictions.npz', **merged)
    ci = result['composition_bootstrap_95_percentile_pp']
    text = ('# V27.3 : comparaison native avec/sans décroissance\n\n'
            f"Cinq folds, {len(k)} clusters, {int(poly.sum())} polyphoniques.\n\n"
            '| Bras | Exact-K polyphonique | Exact-K global | F1 événements 50 ms |\n'
            '|---|---:|---:|---:|\n')
    for arm, label in (('control', 'Témoin sans indice'), ('decay', 'Indice de décroissance ln')):
        card = arms[arm]['cardinality']
        text += f"| {label} | {100*card['poly_exact']:.4f} % ({card['poly_correct']}/{card['poly_rows']}) | {100*card['exact']:.4f} % | {100*arms[arm]['event_50ms']['f1']:.4f} % |\n"
    text += (f'\nÉcart apparié polyphonique : **{delta:+.4f} points**. '
             f'Intervalle bootstrap par composition : [{ci[0]:+.4f} ; {ci[1]:+.4f}] points.\n\n'
             'Les candidats ont été reconstruits avec de nouveaux poids. La population de '
             'clusters diffère de l’archive V27.3 : son score de 42,6019 % reste une référence '
             'historique distincte. Aucun remplacement automatique de V27.3.\n')
    (output_dir/'summary.md').write_text(text)
    print(text)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input-dir', 'config', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    summarize(args.input_dir, args.config, args.output_dir)
