"""Annotation-only evaluation of F0s from the nine audio-only decay variants."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_decay_guard import (
    EXPONENTS, VARIANTS, NAMES, best_batch, extraction_tables,
)
from scripts.audit_v273_internal_residual_acoustics import match_frequencies, GROUPS
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json


def run(a):
    rows = [json.loads(line) for line in a.cases.read_text().splitlines()]
    require(all(r['fold'] in FOLDS for r in rows), 'outer fold')
    with np.load(a.spectra, allow_pickle=False) as z:
        freq = z['freq']; spectra = dict(zip(z['row_id'], z['x']))
    with np.load(a.features, allow_pickle=False) as z:
        features = dict(zip(z['row_id'], z['features']))
    neighbors, active, dictionaries = extraction_tables(freq)
    results = []
    max_error = 0.0
    for row in rows:
        x = spectra[row['row_id']]
        expected = [n['frequency_hz'] for n in row['owned_notes']]
        peaks = x[neighbors].max(axis=2) * active
        pools = {}
        for exponent in EXPONENTS:
            score = np.zeros(len(h.F0_GRID))
            for j in range(h.MAX_HARMONICS):
                denominator = math.sqrt(j + 1) if exponent == .5 else (j + 1) ** exponent
                score += peaks[:, j] / denominator
            chosen = []
            for i in np.argsort(-score, kind='stable'):
                if any(h.cents(h.F0_GRID[i], h.F0_GRID[k]) < h.NMS_CENTS for k in chosen):
                    continue
                chosen.append(i)
                if len(chosen) == h.POOL_SIZE:
                    break
            pools[exponent] = np.asarray(chosen)
        out = {'row_id': row['row_id'], 'fold': row['fold'], 'group': row['group'], 'true_K': row['true_K'], 'arms': {}}
        for i, (score_exp, template_exp) in enumerate(VARIANTS):
            pool = pools[score_exp]
            D = dictionaries[template_exp][:, pool]
            pair, x2 = best_batch(D, x, 2); trip, _ = best_batch(D, x, 3)
            error = float(np.max(np.abs(np.array([pair[0], trip[0]]) / (x2 + 1e-12) - features[row['row_id']][i, :2])))
            require(error < 1e-10, 'deployment feature replay mismatch')
            max_error = max(max_error, error)
            match_pool = match_frequencies(expected, h.F0_GRID[pool])[0]
            match_trip = match_frequencies(expected, h.F0_GRID[pool[list(trip[1])]])[0]
            out['arms'][NAMES[i]] = {'pool_matches': match_pool, 'triplet_matches': match_trip,
                                      'pool_all_owned': match_pool == len(expected),
                                      'triplet_all_owned': match_trip == len(expected)}
        results.append(out)
    summary = {g: {name: {k: sum(r['arms'][name][k] for r in results if r['group'] == g)
                         for k in ('pool_all_owned', 'triplet_all_owned')}
                   for name in NAMES} for g in GROUPS}
    write_json(a.output, {'status': 'completed', 'diagnostic_only': True, 'outer_fold_3_used': False,
                          'feature_replay_max_abs_error': max_error, 'summary': summary, 'cases': results,
                          'source_cases_sha256': sha256_file(a.cases), 'script_sha256': sha256_file(__file__)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'spectra', 'features', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
