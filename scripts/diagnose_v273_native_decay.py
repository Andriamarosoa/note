"""Locate count regressions by replaying already frozen probabilities and rules.

Crossed arms are post-hoc diagnostics, never new validated candidate models.
No threshold is optimized and no model is trained by this program.
"""
import argparse
from datetime import datetime, timezone
import itertools
from pathlib import Path

import numpy as np

from scripts.audit_v273_native_paired_results import load_npz, read_json, require, verify_inventory
from scripts.train_v273_native_paired import decode
from scripts.audit_v270_class_conditional_fusion import fuse_counts
from scripts.audit_v271_poly_rescue import apply_poly_rescue
from scripts.train_v273_selective_transition_corrector import apply_transition_corrector, TRANSITIONS, transition_name
from scripts.rebuild_v273_sources import digest, write_json


ARMS = ('control', 'decay')
COMPONENTS = ('v260_uniform', 'v260_weighted', 'v272_uniform')


def diagnose(root, folds, output):
    details = []
    aggregate = {'rows': 0, 'poly_rows': 0, 'probabilities_x_calibration': {},
                 'stages': {}, 'transition_effects': {}, 'raw_v272_poly_correct': {a: 0 for a in ARMS}}
    for fold in folds:
        stage = root/f'fold-{fold}'
        verify_inventory(stage)
        base = stage/'paired'
        saved = load_npz(base/f'predictions-fold-{fold}.npz')
        k, anchor = saved['k'], saved['shared_anchor']
        poly = k >= 2
        correct = lambda pred: int(np.sum(poly & (pred == k)))
        calibration = read_json(base/'frozen-calibration.json')
        report = read_json(base/f'report-fold-{fold}.json')
        require(digest(base/'frozen-calibration.json') == report['calibration_sha256'], 'calibration changed')
        probabilities = {}
        for arm in ARMS:
            probabilities[arm] = {}
            for component in COMPONENTS:
                p = load_npz(base/arm/'final'/component/'probability.npz')
                require(np.array_equal(p['global_index'], saved['global_index']), 'probability row mismatch')
                probabilities[arm][component] = p['probability']
            require(np.array_equal(decode(anchor, probabilities[arm], calibration[arm]), saved[arm]), 'official decision replay mismatch')
        crossed = {f'{a}/{b}': correct(decode(anchor, probabilities[a], calibration[b]))
                   for a, b in itertools.product(ARMS, repeat=2)}
        stages, effects, raw = {}, {}, {}
        for arm in ARMS:
            p = probabilities[arm]
            low = fuse_counts(anchor, np.argmax(p['v260_uniform'], axis=1), 'low_k_fusion')
            mid = apply_poly_rescue(anchor, low, p['v260_weighted'], calibration[arm]['v271']['threshold'])[0]
            final, _, _, _, direction = apply_transition_corrector(mid, p['v272_uniform'], calibration[arm]['v273']['thresholds'])
            stages[arm] = {'low_k_fusion': correct(low), 'poly_rescue': correct(mid), 'final_v273': correct(final)}
            raw[arm] = correct(np.argmax(p['v272_uniform'], axis=1)+2)
            effects[arm] = {}
            for i, pair in enumerate(TRANSITIONS):
                name = transition_name(*pair)
                active = direction == i
                good = int(np.sum(active & poly & (final == k) & (mid != k)))
                bad = int(np.sum(active & poly & (final != k) & (mid == k)))
                effects[arm][name] = {'activated_all_rows': int(active.sum()), 'corrected_poly': good,
                                      'regressed_poly': bad, 'net_poly': good-bad}
        inner = {a: {'before_transitions': calibration[a]['v273']['baseline']['poly_correct'],
                     'after_transitions': calibration[a]['v273']['selected']['poly_correct'],
                     'poly_rows': calibration[a]['v273']['selected']['poly_rows']} for a in ARMS}
        details.append({'fold': fold, 'rows': len(k), 'poly_rows': int(poly.sum()),
                        'archive_sha256': digest(root/f'fold-{fold}.zip'),
                        'predictions_sha256': digest(base/f'predictions-fold-{fold}.npz'),
                        'probabilities_x_calibration': crossed, 'stages': stages,
                        'transition_effects': effects, 'raw_v272_poly_correct': raw,
                        'inner_calibration_counts': inner})
        aggregate['rows'] += len(k)
        aggregate['poly_rows'] += int(poly.sum())
        for key, value in crossed.items():
            aggregate['probabilities_x_calibration'][key] = aggregate['probabilities_x_calibration'].get(key, 0)+value
        for arm in ARMS:
            aggregate['raw_v272_poly_correct'][arm] += raw[arm]
            for name, value in stages[arm].items():
                key = arm+'/'+name
                aggregate['stages'][key] = aggregate['stages'].get(key, 0)+value
            for name, values in effects[arm].items():
                for metric, value in values.items():
                    key = arm+'/'+name+'/'+metric
                    aggregate['transition_effects'][key] = aggregate['transition_effects'].get(key, 0)+value
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'run_id': 35099819151, 'producer_commit': 'a192bb3137304b5206fdd45a837995c95f80fcbb',
              'folds': folds, 'post_hoc_diagnostic_only': True, 'automatic_promotion': False,
              'crossed_key_order': 'network probabilities / frozen calibration',
              'counts_are_polyphonic_correct_rows': True, 'per_fold': details, 'aggregate': aggregate}
    write_json(output, result)
    print(aggregate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(len(args.folds) == len(set(args.folds)), 'duplicate folds')
    diagnose(args.input_dir, args.folds, args.output)
