"""Annotation-only frequency diagnosis for two frozen reconstruction arms."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_reconstruction_criterion import (
    active_foreign_frequencies, distance_to_integer_relation,
)
from scripts.audit_v273_internal_residual_acoustics import GROUPS, MATCH_CENTS, match_frequencies
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json


def component_relations(value, expected, foreign):
    distance = np.abs(1200 * np.log2(value / expected))
    out = {'owned_neighbor_55_to_150': bool(MATCH_CENTS < distance.min() <= 150),
           'owned_harmonic': False, 'owned_subharmonic': False,
           'foreign_fundamental': False, 'foreign_harmonic': False, 'foreign_subharmonic': False}
    for reference in expected:
        for direction in ('harmonic', 'subharmonic'):
            d, _ = distance_to_integer_relation(value, reference, direction)
            out['owned_' + direction] |= d <= MATCH_CENTS
    if len(foreign):
        out['foreign_fundamental'] = bool(np.min(np.abs(1200 * np.log2(value / foreign))) <= MATCH_CENTS)
        for reference in foreign:
            for direction in ('harmonic', 'subharmonic'):
                d, _ = distance_to_integer_relation(value, reference, direction)
                out['foreign_' + direction] |= d <= MATCH_CENTS
    out['unclassified'] = not any(out.values())
    return out


def aggregate(rows, name):
    arms = [r['arms'][name] for r in rows]
    relations = [t for a in arms for t in a['unmatched_components']]
    keys = ('owned_neighbor_55_to_150', 'owned_harmonic', 'owned_subharmonic',
            'foreign_fundamental', 'foreign_harmonic', 'foreign_subharmonic', 'unclassified')
    return {'selected_owned_matches': dict(Counter(str(a['selected_owned_matches']) for a in arms)),
            'selected_all_owned': sum(a['selected_all_owned'] for a in arms),
            'unmatched_components': len(relations),
            'relation_component_counts_nonexclusive': {k: sum(t[k] for t in relations) for k in keys},
            'relation_case_counts_nonexclusive': {k: sum(any(t[k] for t in a['unmatched_components']) for a in arms) for k in keys}}


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    annotations = [json.loads(s) for s in a.cases.read_text().splitlines()]
    require(len(annotations) == 488 and all(r['fold'] in FOLDS for r in annotations), 'cohort changed')
    ranking_report = json.loads((a.ranking / 'report.json').read_text())
    require(sha256_file(a.cases) == ranking_report['source_cases_sha256'], 'cases source changed')
    ranking = {r['row_id']: r for r in map(json.loads, (a.ranking / 'cases.jsonl').read_text().splitlines())}
    with np.load(a.guard / 'features.npz', allow_pickle=False) as z:
        names = tuple(z['variants'])
        require(len(names) == 2 and len(set(names)) == 2, 'expected two named variants')
        pairs = dict(zip(z['row_id'], z['pair_f0']))
        triplets = dict(zip(z['row_id'], z['triplet_f0']))
    results = []
    for row in annotations:
        row_id = row['row_id']; expected = np.array([n['frequency_hz'] for n in row['owned_notes']])
        foreign = active_foreign_frequencies(row)
        pool_ids = ranking[row_id]['arms']['s0.5']['selected_grid_ids'][:64]
        pool_matches = match_frequencies(expected, h.F0_GRID[pool_ids])[0]
        out = {k: row[k] for k in ('row_id', 'fold', 'group', 'true_K')}
        out['pool_all_owned'] = pool_matches == len(expected); out['arms'] = {}
        selected_matches = []
        for i, name in enumerate(names):
            chosen = triplets[row_id][i] if row['true_K'] == 3 else pairs[row_id][i]
            count, matches = match_frequencies(expected, chosen)
            matched = {m['component'] for m in matches}
            trace = [component_relations(chosen[j], expected, foreign) for j in range(len(chosen)) if j not in matched]
            out['arms'][name] = {'selected_f0': chosen.tolist(), 'selected_owned_matches': count,
                                 'selected_all_owned': count == len(expected), 'unmatched_components': trace}
            selected_matches.append(count)
        out['second_minus_first_matches'] = selected_matches[1] - selected_matches[0]
        results.append(out)
    summary = {}
    for group in (*GROUPS, 'K3', 'K2', 'all'):
        subset = [r for r in results if (True if group == 'all' else
                  r['true_K'] == int(group[1]) if len(group) == 2 else r['group'] == group)]
        summary[group] = {'rows': len(subset), 'pool_all_owned': sum(r['pool_all_owned'] for r in subset),
                          'second_improved_matches': sum(r['second_minus_first_matches'] > 0 for r in subset),
                          'second_equal_matches': sum(r['second_minus_first_matches'] == 0 for r in subset),
                          'second_worsened_matches': sum(r['second_minus_first_matches'] < 0 for r in subset),
                          'arms': {name: aggregate(subset, name) for name in names}}
    report = {'status': 'completed', 'folds': list(FOLDS), 'outer_fold_3_used': False,
              'annotation_use': 'diagnostic only; never inference input', 'cases': len(results),
              'variants': list(names),
              'summary': summary, 'source_sha256': {'cases': sha256_file(a.cases),
                  'ranking_report': sha256_file(a.ranking / 'report.json'),
                  'ranking_cases': sha256_file(a.ranking / 'cases.jsonl'),
                  'guard_features': sha256_file(a.guard / 'features.npz'), 'script': sha256_file(__file__)},
              'limitations': ['Frequency relations are non-exclusive and do not identify a physical source.',
                              'Internal folds were previously inspected.',
                              'Normal residual path only; no compressed-path inference.']}
    a.output.mkdir(parents=True)
    (a.output / 'cases.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in results))
    report['cases_sha256'] = sha256_file(a.output / 'cases.jsonl')
    write_json(a.output / 'report.json', report)
    print(json.dumps({'K3': summary['K3'], 'K2': summary['K2']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ranking', 'guard', 'cases', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
