"""Trace candidate losses before/after NMS on the frozen acoustic cohort."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_decay_guard import EXPONENTS, extraction_tables
from scripts.audit_v273_internal_residual_acoustics import GROUPS, MATCH_CENTS, match_frequencies
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

CAPACITIES = (8, 16, 32, 64, 240)


def trace_nms(scores, grid=h.F0_GRID, threshold=h.NMS_CENTS):
    order = np.argsort(-scores, kind='stable')
    selected, blocked = [], {}
    for i in order:
        blockers = [j for j in selected if h.cents(grid[i], grid[j]) < threshold]
        if blockers:
            blocked[int(i)] = int(blockers[0])
        else:
            selected.append(int(i))
    return order, np.array(selected), blocked


def trace_notes(expected, score, order, selected, blocked, grid=h.F0_GRID):
    raw_rank = np.empty(len(order), int)
    raw_rank[order] = np.arange(1, len(order) + 1)
    selected_rank = {int(i): rank + 1 for rank, i in enumerate(selected)}
    detail = []
    for hz in expected:
        matching = np.flatnonzero(np.abs(1200 * np.log2(grid / hz)) <= MATCH_CENTS)
        surviving = [i for i in matching if i in selected_rank]
        rank = min((selected_rank[i] for i in surviving), default=None)
        reason = ('outside_grid' if not len(matching) else
                  'suppressed_by_nms' if rank is None else
                  'after_capacity_8' if rank > 8 else 'represented_in_8')
        first = min(matching, key=lambda i: raw_rank[i]) if len(matching) else None
        detail.append({'frequency_hz': float(hz), 'status': reason,
                       'first_raw_rank': None if first is None else int(raw_rank[first]),
                       'first_nms_rank': rank,
                       'first_raw_tie_rank_min': None if first is None else int(np.sum(score > score[first]) + 1),
                       'first_raw_tie_rank_max': None if first is None else int(np.sum(score >= score[first])),
                       'matching_grid_ids': matching.tolist(),
                       'blockers': {str(int(i)): blocked[int(i)] for i in matching if i in blocked}})
    return detail


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    source = json.loads((a.audit / 'report.json').read_text())
    case_file = a.audit / 'cases.jsonl'
    require(sha256_file(case_file) == source['cases_sha256'], 'cases changed')
    config = json.loads(a.config.read_text())
    require(sha256_file(a.config) == source['config_sha256'], 'config changed')
    rows = [json.loads(s) for s in case_file.read_text().splitlines()]
    require(len(rows) == 488 and all(r['fold'] in FOLDS and config['member_folds'][r['recording_id']] == r['fold'] for r in rows), 'cohort changed')
    with np.load(a.spectra, allow_pickle=False) as z:
        freq = z['freq']; spectra = dict(zip(z['row_id'], z['x']))
    require(set(spectra) == {r['row_id'] for r in rows}, 'spectrum rows mismatch')
    reference = json.loads(a.coverage.read_text())
    require(reference['source_cases_sha256'] == source['cases_sha256'], 'coverage source changed')
    previous = {r['row_id']: r for r in reference['cases']}
    neighbors, active, _ = extraction_tables(freq)
    results, all_scores = [], []
    for row in rows:
        x = spectra[row['row_id']]
        expected = np.array([n['frequency_hz'] for n in row['owned_notes']])
        sampled = x[neighbors]
        peaks = sampled.max(axis=2) * active
        dominant_bins = np.take_along_axis(neighbors, sampled.argmax(axis=2)[..., None], axis=2)[..., 0]
        out = {k: row[k] for k in ('row_id', 'fold', 'group', 'true_K')}
        out['arms'] = {}
        score_row = []
        for exponent in EXPONENTS:
            score = np.zeros(len(h.F0_GRID))
            for j in range(h.MAX_HARMONICS):
                score += peaks[:, j] / (math.sqrt(j + 1) if exponent == .5 else (j + 1) ** exponent)
            order, selected, blocked = trace_nms(score)
            notes = trace_notes(expected, score, order, selected, blocked)
            match_count, pairs = match_frequencies(expected, h.F0_GRID[selected[:8]])
            require(match_count == previous[row['row_id']]['arms'][f's{exponent:g}_t0.5']['pool_matches'], 'coverage replay changed')
            out['arms'][f's{exponent:g}'] = {
                'notes': notes, 'selected_grid_ids': selected.tolist(),
                'raw_matches': {str(cap): match_frequencies(expected, h.F0_GRID[order[:cap]])[0] for cap in CAPACITIES},
                'nms_matches': {str(cap): match_frequencies(expected, h.F0_GRID[selected[:cap]])[0] for cap in CAPACITIES},
                'pool_8_pairs': pairs,
                'pool_8_fundamental_peak_bins': dominant_bins[selected[:8], 0].tolist(),
                'pool_8_fundamental_unique_peak_bins': len(set(dominant_bins[selected[:8], 0])),
                'unmatched_despite_individual_presence': sum(n['status'] == 'represented_in_8' for n in notes) - match_count,
            }
            score_row.append(score)
        all_scores.append(score_row)
        results.append(out)
    report = {'status': 'completed', 'cases': len(rows), 'folds': list(FOLDS), 'outer_fold_3_used': False,
              'annotation_use': 'diagnostic only', 'source_cases_sha256': source['cases_sha256'],
              'source_spectra_sha256': sha256_file(a.spectra), 'source_coverage_sha256': sha256_file(a.coverage),
              'script_sha256': sha256_file(__file__), 'groups': {}}
    for group in (*GROUPS, 'K3', 'K2'):
        subset = [r for r in results if (r['true_K'] == int(group[1]) if len(group) == 2 else r['group'] == group)]
        report['groups'][group] = {'rows': len(subset), 'arms': {}}
        for name in results[0]['arms']:
            arms = [r['arms'][name] for r in subset]
            notes = [n for arm in arms for n in arm['notes']]
            ranks = [n['first_nms_rank'] for n in notes if n['first_nms_rank'] is not None]
            report['groups'][group]['arms'][name] = {
                'note_status': dict(Counter(n['status'] for n in notes)),
                'raw_complete': {str(cap): sum(r['arms'][name]['raw_matches'][str(cap)] == r['true_K'] for r in subset) for cap in CAPACITIES},
                'nms_complete': {str(cap): sum(r['arms'][name]['nms_matches'][str(cap)] == r['true_K'] for r in subset) for cap in CAPACITIES},
                'finite_nms_rank_percentiles': np.percentile(ranks, [0, 25, 50, 75, 100]).tolist(),
                'pool_fundamental_unique_peak_bins_mean': float(np.mean([arm['pool_8_fundamental_unique_peak_bins'] for arm in arms])),
                'unmatched_despite_individual_presence': sum(arm['unmatched_despite_individual_presence'] for arm in arms),
            }
    a.output.mkdir(parents=True)
    (a.output / 'cases.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in results))
    np.savez_compressed(a.output / 'scores.npz', row_id=np.array([r['row_id'] for r in rows]),
                        score=np.array(all_scores), grid=h.F0_GRID, exponents=EXPONENTS)
    report['cases_sha256'] = sha256_file(a.output / 'cases.jsonl')
    write_json(a.output / 'report.json', report)
    print(json.dumps(report['groups']['K3'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'spectra', 'coverage', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
