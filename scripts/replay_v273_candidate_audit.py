"""Replay candidate traces, frozen model decisions and annotated F0 coverage."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_candidate_capacity_guard import NAMES, VARIANTS
from scripts.audit_v273_candidate_ranking import CAPACITIES, trace_nms, trace_notes
from scripts.audit_v273_harmonic_decay_guard import EXPONENTS, extraction_tables
from scripts.audit_v273_internal_residual_acoustics import GROUPS, match_frequencies
from scripts.replay_v273_harmonic_decay_guard import verify as verify_models
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json


def verify(ranking, capacity, cases):
    report = json.loads((ranking / 'report.json').read_text())
    require(report['status'] == 'completed' and report['folds'] == list(FOLDS) and not report['outer_fold_3_used'], 'invalid ranking report')
    require(sha256_file(cases) == report['source_cases_sha256'], 'annotation case hash changed')
    require(sha256_file(ranking / 'cases.jsonl') == report['cases_sha256'], 'trace hash changed')
    annotations = {r['row_id']: r for r in map(json.loads, cases.read_text().splitlines())}
    rows = [json.loads(s) for s in (ranking / 'cases.jsonl').read_text().splitlines()]
    require({r['row_id'] for r in rows} == set(annotations), 'case population changed')
    with np.load(ranking / 'scores.npz', allow_pickle=False) as z:
        require(np.array_equal(z['grid'], h.F0_GRID) and np.array_equal(z['exponents'], EXPONENTS), 'grid changed')
        scores = dict(zip(z['row_id'], z['score']))
    require(sha256_file(ranking / 'spectra.npz') == report['source_spectra_sha256'], 'spectrum checkpoint changed')
    with np.load(ranking / 'spectra.npz', allow_pickle=False) as z:
        freq, spectra = z['freq'], dict(zip(z['row_id'], z['x']))
    require(set(spectra) == set(annotations), 'spectrum population changed')
    neighbors, active, _ = extraction_tables(freq)
    with np.load(capacity / 'features.npz', allow_pickle=False) as z:
        require(tuple(z['variants']) == NAMES, 'capacity variants changed')
        features = dict(zip(z['row_id'], z['features']))
        pairs = dict(zip(z['row_id'], z['pair_f0']))
        triplets = dict(zip(z['row_id'], z['triplet_f0']))
    coverage = []
    for row in rows:
        ref = annotations[row['row_id']]
        require(row['fold'] == ref['fold'] and row['fold'] in FOLDS, 'fold changed')
        require(row['group'] == ref['group'] and row['true_K'] == ref['true_K'], 'labels changed')
        expected = np.array([n['frequency_hz'] for n in ref['owned_notes']])
        sampled = spectra[row['row_id']][neighbors]
        peaks = sampled.max(axis=2) * active
        dominant_bins = np.take_along_axis(neighbors, sampled.argmax(axis=2)[..., None], axis=2)[..., 0]
        pools = {}
        for i, exponent in enumerate(EXPONENTS):
            score = scores[row['row_id']][i]
            recomputed = np.zeros(len(h.F0_GRID))
            for j in range(h.MAX_HARMONICS):
                recomputed += peaks[:, j] / (math.sqrt(j + 1) if exponent == .5 else (j + 1) ** exponent)
            require(np.array_equal(recomputed, score), 'salience scores changed')
            order, selected, blocked = trace_nms(score)
            arm = row['arms'][f's{exponent:g}']
            require(selected.tolist() == arm['selected_grid_ids'], 'NMS changed')
            require(trace_notes(expected, score, order, selected, blocked) == arm['notes'], 'note trace changed')
            for cap in CAPACITIES:
                require(match_frequencies(expected, h.F0_GRID[order[:cap]])[0] == arm['raw_matches'][str(cap)], 'raw coverage changed')
                require(match_frequencies(expected, h.F0_GRID[selected[:cap]])[0] == arm['nms_matches'][str(cap)], 'NMS coverage changed')
            count, match = match_frequencies(expected, h.F0_GRID[selected[:8]])
            require(match == arm['pool_8_pairs'], 'matching changed')
            require(sum(n['status'] == 'represented_in_8' for n in arm['notes']) - count == arm['unmatched_despite_individual_presence'], 'individual-count mismatch')
            require(len(set(arm['pool_8_fundamental_peak_bins'])) == arm['pool_8_fundamental_unique_peak_bins'], 'peak-bin count mismatch')
            require(dominant_bins[selected[:8], 0].tolist() == arm['pool_8_fundamental_peak_bins'], 'peak bins changed')
            pools[exponent] = selected
        values = features[row['row_id']]
        require(np.all(values[2:, :2] <= values[:2, :2] + 1e-10), 'enlarged pool residual increased')
        require(np.all(values[:, 1] <= values[:, 0] + 1e-10), 'triplet residual above pair')
        out = {k: row[k] for k in ('row_id', 'fold', 'true_K', 'group')}
        out['arms'] = {}
        for i, (name, (size, _)) in enumerate(zip(NAMES, VARIANTS)):
            pool_hz = h.F0_GRID[pools[.5][:size]]
            pair_hz, trip_hz = pairs[row['row_id']][i], triplets[row['row_id']][i]
            require(np.isin(pair_hz, pool_hz).all() and np.isin(trip_hz, pool_hz).all(), 'chosen frequency outside pool')
            out['arms'][name] = {f'{part}_all_owned': match_frequencies(expected, freq)[0] == len(expected)
                                for part, freq in (('pool', pool_hz), ('pair', pair_hz), ('triplet', trip_hz))}
        coverage.append(out)
    # Independently reaggregate report counts from the verified traces.
    for group, summary in report['groups'].items():
        subset = [r for r in rows if (r['true_K'] == int(group[1]) if len(group) == 2 else r['group'] == group)]
        require(len(subset) == summary['rows'], 'group size changed')
        for name, saved in summary['arms'].items():
            aa = [r['arms'][name] for r in subset]
            notes = [n for a in aa for n in a['notes']]
            require(dict(Counter(n['status'] for n in notes)) == saved['note_status'], 'note totals changed')
            for stage in ('raw', 'nms'):
                require({str(cap): sum(r['arms'][name][stage + '_matches'][str(cap)] == r['true_K'] for r in subset) for cap in CAPACITIES} == saved[stage + '_complete'], 'coverage totals changed')
            ranks = [n['first_nms_rank'] for n in notes if n['first_nms_rank'] is not None]
            require(np.percentile(ranks, [0, 25, 50, 75, 100]).tolist() == saved['finite_nms_rank_percentiles'], 'rank quantiles changed')
            require(sum(a['unmatched_despite_individual_presence'] for a in aa) == saved['unmatched_despite_individual_presence'], 'ambiguous-match total changed')
            require(float(np.mean([a['pool_8_fundamental_unique_peak_bins'] for a in aa])) == saved['pool_fundamental_unique_peak_bins_mean'], 'bin summary changed')
    summary = {}
    for group in (*GROUPS, 'K3', 'K2'):
        subset = [r for r in coverage if (r['true_K'] == int(group[1]) if len(group) == 2 else r['group'] == group)]
        summary[group] = {'rows': len(subset), 'arms': {name: {k: sum(r['arms'][name][k] for r in subset)
                                    for k in ('pool_all_owned', 'pair_all_owned', 'triplet_all_owned')} for name in NAMES}}
    return {'status': 'verified', 'ranking_cases': len(rows), 'outer_fold_3_used': False,
            'models': verify_models(capacity, NAMES), 'frequency_coverage': summary,
            'script_sha256': sha256_file(__file__), 'source_cases_sha256': sha256_file(cases),
            'limitations': ['Salience is recomputed from archived spectra; waveform preprocessing and residuals are not recomputed here.',
                            'Annotations evaluate frozen frequency outputs only; never used in inference.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ranking', 'capacity', 'cases', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    result = verify(a.ranking, a.capacity, a.cases)
    write_json(a.output, result)
    print(json.dumps(result, indent=2))
