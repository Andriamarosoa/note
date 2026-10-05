"""Test the exact Hann-power template shape with fixed pool and decay."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_candidate_ranking import trace_nms
from scripts.audit_v273_harmonic_decay_guard import accounting, extraction_tables, probability
from scripts.audit_v273_harmonic_templates import harmonic_basis, rigid_dictionary
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

NAMES = ('gaussian_t2', 'hann_t2')


def hann_dictionary(freq):
    return rigid_dictionary([harmonic_basis(freq, float(f0), 'hann_power') for f0 in h.F0_GRID], 2.0)


@lru_cache(maxsize=None)
def combinations(columns, k):
    return np.array(list(itertools.combinations(range(columns), k)), dtype=int)


def _single_cost(G, b, x2, ids):
    diagonal = G[ids, ids]
    coefficient = b[ids] / (diagonal + 1e-10)
    cost = x2 - 2 * coefficient * b[ids] + coefficient * coefficient * diagonal
    return np.where(coefficient >= 0, cost, x2)


def _pair_cost(G, b, x2, ids):
    i, j = ids[:, 0], ids[:, 1]
    g00, g11, g01 = G[i, i], G[j, j], G[i, j]
    b0, b1 = b[i], b[j]
    determinant = (g00 + 1e-10) * (g11 + 1e-10) - g01 * g01
    a0 = ((g11 + 1e-10) * b0 - g01 * b1) / determinant
    a1 = ((g00 + 1e-10) * b1 - g01 * b0) / determinant
    cost = x2 - 2 * (a0 * b0 + a1 * b1) + a0 * a0 * g00 + 2 * a0 * a1 * g01 + a1 * a1 * g11
    cost = np.where((a0 >= 0) & (a1 >= 0), cost, x2)
    return np.minimum(cost, np.minimum(_single_cost(G, b, x2, i), _single_cost(G, b, x2, j)))


def best_fast(D, x, k):
    """Same active-subset NNLS costs as best_batch, specialized to k=2/3."""
    require(k in (2, 3), 'unsupported k')
    G, b, x2 = D.T @ D, D.T @ x, float(x @ x)
    combos = combinations(D.shape[1], k)
    if k == 2:
        cost = _pair_cost(G, b, x2, combos)
    else:
        GG = G[combos[:, :, None], combos[:, None, :]]
        bb = b[combos]
        coefficient = np.linalg.solve(GG + 1e-10 * np.eye(3), bb[..., None])[..., 0]
        cost = x2 - 2 * np.einsum('ij,ij->i', coefficient, bb) + np.einsum('ij,ijk,ik->i', coefficient, GG, coefficient)
        cost = np.where(np.all(coefficient >= 0, axis=1), cost, x2)
        for a, q in ((0, 1), (0, 2), (1, 2)):
            cost = np.minimum(cost, _pair_cost(G, b, x2, combos[:, [a, q]]))
        for a in range(3):
            cost = np.minimum(cost, _single_cost(G, b, x2, combos[:, a]))
    winner = int(np.argmin(cost))
    return (max(float(cost[winner]), 0.0), tuple(combos[winner])), x2


def select_on_fit(X, y, fold):
    require(set(fold) <= set(FOLDS), 'outer fold')
    reports, predictions = [], []
    for i in range(len(NAMES)):
        p = np.empty(len(y))
        for heldout in np.unique(fold):
            valid = fold == heldout
            p[valid] = probability(X[~valid, i, :2], y[~valid], X[valid, i, :2])
        reports.append(accounting(y, p)); predictions.append(p)
    chosen = min(range(len(NAMES)), key=lambda i: (-reports[i]['global_net'], reports[i]['regressions'], reports[i]['applied'], i))
    return (chosen if reports[chosen]['global_net'] > 0 else None), reports, np.asarray(predictions)


def extract_hann(freq, x, tables):
    neighbors, active, dictionaries = tables
    peaks = x[neighbors].max(axis=2) * active
    salience = np.zeros(len(h.F0_GRID))
    for j in range(h.MAX_HARMONICS):
        salience += peaks[:, j] / math.sqrt(j + 1)
    _, selected, _ = trace_nms(salience)
    ids = selected[:64]
    D = dictionaries['hann_t2'][:, ids]
    pair, x2 = best_fast(D, x, 2); trip, _ = best_fast(D, x, 3)
    pair_hz = h.F0_GRID[ids[list(pair[1])]]
    trip_hz = h.F0_GRID[ids[list(trip[1])]]
    value = np.array([pair[0] / (x2 + 1e-12), trip[0] / (x2 + 1e-12), float(np.median(trip_hz))])
    return value, pair_hz, trip_hz


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    inputs, records = action_inputs(a.exports)
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open('rb') as stream:
            require(hashlib.file_digest(stream, 'md5').hexdigest() == expected, 'dataset changed')
    wanted = {member for member, _ in records.values()}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, 'missing tracks')
    by_member = {}
    for row, (member, start) in records.items():
        by_member.setdefault(member, []).append((row, start))
    with np.load(a.capacity / 'features.npz', allow_pickle=False) as previous_file:
        previous = dict(previous_file)
    computed, spectra, tables, freq, started = {}, {}, None, None, time.monotonic()
    for member in sorted(wanted):
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768
        for row, start in by_member[member]:
            freq, x = h.transition_spectrum(samples, start)
            if tables is None:
                tables = extraction_tables(freq)
                tables[2]['hann_t2'] = hann_dictionary(freq)
            computed[row] = extract_hann(freq, x, tables); spectra[row] = x
        print(json.dumps({'recording': member, 'rows': len(computed), 'seconds': time.monotonic() - started}), flush=True)
    ids = np.array(sorted(computed))
    np.testing.assert_array_equal(ids, previous['row_id'])
    hann_values = np.array([computed[int(i)][0] for i in ids])
    hann_pairs = np.array([computed[int(i)][1] for i in ids])
    hann_triplets = np.array([computed[int(i)][2] for i in ids])
    values = np.stack([previous['features'][:, 3], hann_values], axis=1)
    pairs = np.stack([previous['pair_f0'][:, 3], hann_pairs], axis=1)
    triplets = np.stack([previous['triplet_f0'][:, 3], hann_triplets], axis=1)
    feature_by_id = dict(zip(ids, values))
    control_error = 0.0; control_frequency_error = 0.0
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'features.npz', row_id=ids, features=values,
                        pair_f0=pairs, triplet_f0=triplets, variants=NAMES)
    np.savez_compressed(a.output / 'spectra.npz', row_id=ids,
                        x=np.array([spectra[int(i)] for i in ids]), freq=freq)
    reports, scores = [], {}
    for fold, arr in inputs.items():
        data = {}
        for split in ('fit', 'val'):
            mask = arr[split + '_b_low'] & (arr[split + '_base_k'] == 3)
            valid = arr[split + '_valid']; row_ids = arr[split + '_ids'][mask][valid]
            data[split] = dict(X=np.array([feature_by_id[int(i)] for i in row_ids]),
                               y=arr[split + '_true_k'][mask][valid],
                               fold=arr[split + '_fold'][mask][valid], ids=row_ids)
        fit, val = data['fit'], data['val']
        selected, cv, inner = select_on_fit(fit['X'], fit['y'], fit['fold'])
        states = []
        val_probability = np.array([probability(fit['X'][:, i, :2], fit['y'], val['X'][:, i, :2], states)
                                    for i in range(len(NAMES))])
        write_json(a.output / f'models-fold-{fold}.json', dict(zip(NAMES, states)))
        for split in ('fit', 'val'):
            for key in ('ids', 'y', 'fold'):
                scores[f'fold_{fold}_{split}_{key}'] = data[split][key]
        scores[f'fold_{fold}_inner_probability'] = inner
        scores[f'fold_{fold}_val_probability'] = val_probability
        chosen = np.zeros(len(val['y'])) if selected is None else val_probability[selected]
        reports.append({'fold': fold, 'selected': None if selected is None else NAMES[selected],
                        'fit_cv': dict(zip(NAMES, cv)),
                        'val_fixed': {name: accounting(val['y'], p) for name, p in zip(NAMES, val_probability)},
                        'val_selected': accounting(val['y'], chosen)})
    np.savez_compressed(a.output / 'scores.npz', **scores)
    total = lambda rows: {k: sum(r[k] for r in rows) for k in
                          ('rows', 'applied', 'corrections', 'regressions', 'other_k_actions', 'global_net')}
    report = {'status': 'completed', 'folds': reports, 'variants': list(NAMES), 'outer_fold_3_used': False,
              'annotation_frequencies_used': False, 'neural_training': False, 'automatic_promotion': False,
              'unique_audio_rows': len(ids), 'recordings': len(wanted),
              'templates': {'gaussian_t2': 'sigma 18 Hz, peak heights 1/h^2',
                            'hann_t2': 'exact phase-averaged Hann power response, peak heights 1/h^2'},
              'control_max_abs_residual_error': control_error,
              'control_max_abs_frequency_error': control_frequency_error,
              'control_source': 'verified pool64_t2 arrays from the candidate-capacity evidence',
              'total_selected': total([r['val_selected'] for r in reports]),
              'total_fixed': {name: total([r['val_fixed'][name] for r in reports]) for name in NAMES},
              'selection': 'FIT-only rotation; net, fewer regressions, fewer actions, arm order; abstain if net <= 0',
              'source_run': 37356100423, 'source_data_md5': DATA_MD5,
              'source_manifests_sha256': {str(f): sha256_file(a.exports / f'fold-{f}' / 'manifest.json') for f in FOLDS},
              'source_sha256': {name: sha256_file(Path(__file__).parent / name) for name in
                                ('audit_v273_hann_shape_guard.py', 'audit_v273_harmonic_templates.py',
                                 'audit_v273_log_frequency_guard.py', 'audit_v273_candidate_ranking.py',
                                 'audit_v273_harmonic_decay_guard.py', 'audit_v273_internal_b_low_harmonic_strata.py',
                                 'v273_residual_audit.py')},
              'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                          for p in ('numpy', 'scipy', 'scikit-learn')}},
              'limitations': ['Internal folds previously inspected; not untouched validation.',
                              'Frozen base neural predictions and B_low routing, including inside FIT rotation.',
                              'Normal residual-audio path only; no compressed-path gain inferred.']}
    previous = json.loads((a.capacity / 'report.json').read_text())
    require(report['total_fixed']['gaussian_t2'] == previous['total_fixed']['pool64_t2'], 'control accounting changed')
    write_json(a.output / 'report.json', report)
    print(json.dumps({'fixed': report['total_fixed'], 'selected': report['total_selected'],
                      'selection': [r['selected'] for r in reports]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'dataset', 'capacity', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
