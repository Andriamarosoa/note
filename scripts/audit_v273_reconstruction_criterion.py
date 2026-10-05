"""Diagnose why global NNLS chooses non-owned F0s despite full pool coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path

import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_decay_guard import extraction_tables, best_batch
from scripts.audit_v273_internal_residual_acoustics import GROUPS, MATCH_CENTS, match_frequencies
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

ARMS = ((.5, 'pool64_t0.5'), (2., 'pool64_t2'))
NEAR_CENTS = 150.0
RELATION_ORDERS = tuple(range(2, 11))


def distance_to_integer_relation(value, reference, direction):
    targets = np.array([reference * n if direction == 'harmonic' else reference / n
                        for n in RELATION_ORDERS])
    cents = np.abs(1200 * np.log2(value / targets))
    i = int(np.argmin(cents))
    return float(cents[i]), int(RELATION_ORDERS[i])


def active_foreign_frequencies(row):
    end = row['start_sample'] + h.WINDOW
    return np.array([440 * 2 ** ((n['midi'] - 69) / 12) for n in row['nearby_notes']
                     if not n['own'] and n['onset_sample'] < end and n['offset_sample'] > row['start_sample']], float)


def valid_owned_combinations(expected, pool):
    matches = [np.flatnonzero(np.abs(1200 * np.log2(pool / hz)) <= MATCH_CENTS).tolist()
               for hz in expected]
    combinations = set()
    for choice in itertools.product(*matches):
        if len(set(choice)) != len(choice):
            continue
        combo = tuple(sorted(choice))
        if match_frequencies(expected, pool[list(combo)])[0] == len(expected):
            combinations.add(combo)
    return sorted(combinations)


def best_restricted(D, x, combinations):
    G, b, x2 = D.T @ D, D.T @ x, float(x @ x)
    best = None
    for combo in combinations:
        cost, coefficient = h.nonnegative_cost(G, b, x2, list(combo))
        if best is None or cost < best[0]:
            best = (cost, combo, coefficient)
    return best, G, x2


def relation_trace(value, expected, foreign, selected, D, local):
    fundamental = np.abs(1200 * np.log2(value / expected))
    out = {
        'frequency_hz': float(value),
        'nearest_owned_cents': float(fundamental.min()),
        'owned_neighbor_55_to_150': bool(MATCH_CENTS < fundamental.min() <= NEAR_CENTS),
        'owned_harmonic': False, 'owned_subharmonic': False,
        'foreign_fundamental': False, 'foreign_harmonic': False, 'foreign_subharmonic': False,
    }
    for reference in expected:
        for direction in ('harmonic', 'subharmonic'):
            distance, order = distance_to_integer_relation(value, reference, direction)
            key = 'owned_' + direction
            if distance <= MATCH_CENTS and (not out[key] or distance < out[key + '_cents']):
                out[key] = True; out[key + '_cents'] = distance; out[key + '_order'] = order
    if len(foreign):
        distance = np.abs(1200 * np.log2(value / foreign))
        out['foreign_fundamental'] = bool(distance.min() <= MATCH_CENTS)
        out['foreign_fundamental_cents'] = float(distance.min())
        for reference in foreign:
            for direction in ('harmonic', 'subharmonic'):
                distance, order = distance_to_integer_relation(value, reference, direction)
                key = 'foreign_' + direction
                if distance <= MATCH_CENTS and (not out[key] or distance < out[key + '_cents']):
                    out[key] = True; out[key + '_cents'] = distance; out[key + '_order'] = order
    other = np.delete(selected, local)
    out['nearest_selected_hz'] = float(np.min(np.abs(value - other)))
    out['nearest_selected_cents'] = float(np.min(np.abs(1200 * np.log2(value / other))))
    cosine = np.delete(D[:, local].T @ D, local)
    out['max_template_cosine_to_selected'] = float(cosine.max())
    out['unclassified'] = not any(out[k] for k in ('owned_neighbor_55_to_150', 'owned_harmonic',
                                                    'owned_subharmonic', 'foreign_fundamental',
                                                    'foreign_harmonic', 'foreign_subharmonic'))
    return out


def synthetic_control(D, combo, expected, pool):
    coefficient = np.linspace(.5, 1.5, len(combo))
    x = D[:, combo] @ coefficient
    winner, _ = best_batch(D, x, len(combo))
    return match_frequencies(expected, pool[np.asarray(winner[1])])[0] == len(expected), float(winner[0])


def aggregate(rows, arm):
    out = {'rows': len(rows), 'pool_complete': sum(r['arms'][arm]['pool_complete'] for r in rows)}
    covered = [r['arms'][arm] for r in rows if r['arms'][arm]['pool_complete']]
    out['covered_rows'] = len(covered)
    out['selected_owned_matches'] = dict(Counter(str(r['selected_owned_matches']) for r in covered))
    gaps = np.array([r['oracle_minus_selected_ratio'] for r in covered])
    out['oracle_minus_selected_ratio'] = ({'mean': float(gaps.mean()), 'median': float(np.median(gaps)),
                                            'q25': float(np.quantile(gaps, .25)), 'q75': float(np.quantile(gaps, .75)),
                                            'maximum': float(gaps.max())} if len(gaps) else None)
    traces = [t for r in covered for t in r['unmatched_components']]
    keys = ('owned_neighbor_55_to_150', 'owned_harmonic', 'owned_subharmonic',
            'foreign_fundamental', 'foreign_harmonic', 'foreign_subharmonic', 'unclassified')
    out['unmatched_components'] = len(traces)
    out['relation_component_counts_nonexclusive'] = {k: sum(t[k] for t in traces) for k in keys}
    out['relation_case_counts_nonexclusive'] = {k: sum(any(t[k] for t in r['unmatched_components']) for r in covered) for k in keys}
    out['close_selected_pair_cases'] = {str(hz): sum(r['minimum_selected_spacing_hz'] < hz for r in covered)
                                        for hz in (h.KERNEL_HZ, 2 * h.KERNEL_HZ)}
    out['high_selected_template_cosine_cases'] = {str(v): sum(r['maximum_selected_template_cosine'] >= v for r in covered)
                                                   for v in (.8, .9, .95)}
    out['synthetic_complete'] = sum(r['synthetic_complete'] for r in covered)
    out['synthetic_max_residual'] = max((r['synthetic_residual'] for r in covered), default=None)
    return out


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    ranking_report = json.loads((a.ranking / 'report.json').read_text())
    capacity_report = json.loads((a.capacity / 'report.json').read_text())
    require(ranking_report['status'] == capacity_report['status'] == 'completed', 'incomplete source')
    require(not ranking_report['outer_fold_3_used'] and not capacity_report['outer_fold_3_used'], 'fold 3 source')
    require(sha256_file(a.cases) == ranking_report['source_cases_sha256'], 'cases changed')
    annotations = {r['row_id']: r for r in map(json.loads, a.cases.read_text().splitlines())}
    ranking = {r['row_id']: r for r in map(json.loads, (a.ranking / 'cases.jsonl').read_text().splitlines())}
    require(set(annotations) == set(ranking) and len(annotations) == 488, 'population changed')
    with np.load(a.ranking / 'spectra.npz', allow_pickle=False) as z:
        freq, spectra = z['freq'], dict(zip(z['row_id'], z['x']))
    with np.load(a.capacity / 'features.npz', allow_pickle=False) as z:
        require(tuple(z['variants']) == ('pool8_t0.5', 'pool8_t2', 'pool64_t0.5', 'pool64_t2'), 'variants changed')
        archived_features = dict(zip(z['row_id'], z['features']))
        archived_triplets = dict(zip(z['row_id'], z['triplet_f0']))
        archived_pairs = dict(zip(z['row_id'], z['pair_f0']))
    _, _, dictionaries = extraction_tables(freq)
    results, max_residual_error, max_frequency_error = [], 0., 0.
    for row_id, row in annotations.items():
        require(row['fold'] in FOLDS, 'fold 3')
        expected = np.array([n['frequency_hz'] for n in row['owned_notes']])
        foreign = active_foreign_frequencies(row)
        pool_ids = np.array(ranking[row_id]['arms']['s0.5']['selected_grid_ids'][:64])
        pool = h.F0_GRID[pool_ids]
        x = spectra[row_id]
        result = {k: row[k] for k in ('row_id', 'fold', 'group', 'true_K')}
        result['arms'] = {}
        combinations = valid_owned_combinations(expected, pool)
        for exponent, name in ARMS:
            D = dictionaries[exponent][:, pool_ids]
            global_fit, x2 = best_batch(D, x, len(expected))
            chosen_local = np.array(global_fit[1]); chosen = pool[chosen_local]
            archive_index = 2 if exponent == .5 else 3
            saved = archived_triplets[row_id][archive_index] if len(expected) == 3 else archived_pairs[row_id][archive_index]
            max_frequency_error = max(max_frequency_error, float(np.max(np.abs(chosen - saved))))
            saved_cost = archived_features[row_id][archive_index, 1 if len(expected) == 3 else 0]
            max_residual_error = max(max_residual_error, abs(global_fit[0] / (x2 + 1e-12) - saved_cost))
            count, matches = match_frequencies(expected, chosen)
            matched_components = {m['component'] for m in matches}
            restricted = best_restricted(D, x, combinations)[0] if combinations else None
            pair_cosine = D[:, chosen_local].T @ D[:, chosen_local]
            upper = pair_cosine[np.triu_indices(len(expected), 1)]
            arm = {'pool_complete': len(combinations) > 0, 'selected_f0': chosen.tolist(),
                   'selected_owned_matches': count, 'selected_matches': matches,
                   'selected_residual_ratio': global_fit[0] / (x2 + 1e-12),
                   'minimum_selected_spacing_hz': float(np.diff(np.sort(chosen)).min()),
                   'minimum_selected_spacing_cents': min(h.cents(u, v) for u, v in itertools.combinations(chosen, 2)),
                   'maximum_selected_template_cosine': float(upper.max()),
                   'unmatched_components': [relation_trace(chosen[i], expected, foreign, chosen, D[:, chosen_local], i)
                                            for i in range(len(chosen)) if i not in matched_components]}
            if restricted:
                oracle_hz = pool[list(restricted[1])]
                arm.update(oracle_f0=oracle_hz.tolist(), oracle_residual_ratio=restricted[0] / (x2 + 1e-12),
                           oracle_minus_selected_ratio=(restricted[0] - global_fit[0]) / (x2 + 1e-12))
                complete, residual = synthetic_control(D, restricted[1], expected, pool)
                arm.update(synthetic_complete=complete, synthetic_residual=residual)
            result['arms'][name] = arm
        results.append(result)
    require(max_frequency_error == 0 and max_residual_error < 1e-10, 'archived extraction replay changed')
    summary = {}
    for group in (*GROUPS, 'K3', 'K2', 'all'):
        subset = [r for r in results if (True if group == 'all' else
                  r['true_K'] == int(group[1]) if len(group) == 2 else r['group'] == group)]
        summary[group] = {'rows': len(subset), 'arms': {name: aggregate(subset, name) for _, name in ARMS}}
    report = {'status': 'completed', 'folds': list(FOLDS), 'outer_fold_3_used': False,
              'annotation_use': 'diagnostic only; never inference input', 'neural_training': False,
              'cases': len(results), 'arms': [name for _, name in ARMS], 'summary': summary,
              'archive_replay_max_abs_residual_error': max_residual_error,
              'archive_replay_max_abs_frequency_error': max_frequency_error,
              'source_sha256': {'cases': sha256_file(a.cases), 'ranking_report': sha256_file(a.ranking / 'report.json'),
                                'ranking_cases': sha256_file(a.ranking / 'cases.jsonl'),
                                'spectra': sha256_file(a.ranking / 'spectra.npz'),
                                'capacity_features': sha256_file(a.capacity / 'features.npz'),
                                'script': sha256_file(__file__)},
              'limitations': ['Frequency relations are non-exclusive and do not identify a physical source.',
                              'Oracle combinations use annotations only to diagnose the frozen criterion.',
                              'Internal folds were previously inspected; no untouched test claim.',
                              'Normal residual path only; no compressed-path inference.']}
    a.output.mkdir(parents=True)
    (a.output / 'cases.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in results))
    report['cases_sha256'] = sha256_file(a.output / 'cases.jsonl')
    write_json(a.output / 'report.json', report)
    print(json.dumps(summary['K3'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ranking', 'capacity', 'cases', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
