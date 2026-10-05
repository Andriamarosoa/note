"""Two post windows on frozen normal-audio action rows, without pitch labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_candidate_ranking import trace_nms
from scripts.audit_v273_hann_shape_guard import combinations, _single_cost, _pair_cost
from scripts.audit_v273_harmonic_decay_guard import accounting, extraction_tables, probability
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5, match_frequencies
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

NAMES = ('first_window', 'joint_windows')
PROTOCOL = Path('analysis/v273-temporal-stability-protocol.md')


def post_spectra(samples, start):
    """Both positive residuals share the original pre window, not successive deltas."""
    taper = np.hanning(h.WINDOW)
    freq = np.fft.rfftfreq(h.FFT_SIZE, 1 / SAMPLE_RATE)
    keep = (freq >= h.MIN_HZ) & (freq <= h.MAX_ANALYSIS_HZ)
    power = []
    for offset in (-h.WINDOW, 0, h.WINDOW):
        wave = h.pcm_window(samples, start + offset, h.WINDOW) * taper
        power.append((np.abs(np.fft.rfft(wave, n=h.FFT_SIZE)) ** 2)[keep])
    residual = np.maximum(np.array(power[1:]) - power[0], 0)
    mass = residual.sum(axis=1)
    x = residual / (mass[:, None] + 1e-12)
    padding = [max(0, -start - offset) + max(0, start + offset + h.WINDOW - len(samples))
               for offset in (-h.WINDOW, 0, h.WINDOW)]
    return freq[keep], x, mass, np.array([p.sum() for p in power]), np.minimum(padding, h.WINDOW)


def costs_for_combinations(D, x, k):
    """Exhaustive NNLS active subsets, normalized separately for each window."""
    require(k in (2, 3), 'unsupported k')
    G, b, x2 = D.T @ D, D.T @ x, float(x @ x)
    combo = combinations(D.shape[1], k)
    if k == 2:
        cost = _pair_cost(G, b, x2, combo)
    else:
        GG = G[combo[:, :, None], combo[:, None, :]]
        bb = b[combo]
        coefficient = np.linalg.solve(GG + 1e-10 * np.eye(3), bb[..., None])[..., 0]
        cost = x2 - 2 * np.einsum('ij,ij->i', coefficient, bb) + np.einsum('ij,ijk,ik->i', coefficient, GG, coefficient)
        cost = np.where(np.all(coefficient >= 0, axis=1), cost, x2)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            cost = np.minimum(cost, _pair_cost(G, b, x2, combo[:, [i, j]]))
        for i in range(3):
            cost = np.minimum(cost, _single_cost(G, b, x2, combo[:, i]))
    return np.maximum(cost, 0) / (x2 + 1e-12), combo


def pool_from_first(x, tables):
    neighbors, active, _ = tables
    peaks = x[neighbors].max(axis=2) * active
    salience = np.zeros(len(h.F0_GRID))
    for j in range(h.MAX_HARMONICS):
        salience += peaks[:, j] / math.sqrt(j + 1)
    return trace_nms(salience)[1][:64]


def extract_windows(xs, tables):
    pool = pool_from_first(xs[0], tables)
    D = tables[2][2.0][:, pool]
    residuals = np.zeros((3, 2))  # first, second, common-frequency joint
    selected = {}
    for k in (2, 3):
        per_window = np.array([costs_for_combinations(D, x, k)[0] for x in xs])
        all_costs = np.vstack([per_window, per_window.mean(axis=0)])
        winner = all_costs.argmin(axis=1)
        residuals[:, k - 2] = all_costs[np.arange(3), winner]
        selected[k] = h.F0_GRID[pool[combinations(len(pool), k)[winner]]]
    return residuals, selected[2], selected[3], pool


def extract(a):
    require(not a.output.exists(), 'refusing overwrite')
    inputs, records = action_inputs(a.exports)
    config = json.loads(a.config.read_text())
    for f in FOLDS:
        manifest = json.loads((a.exports / f'fold-{f}/manifest.json').read_text())
        require(sha256_file(a.config) == manifest['config_sha256'], 'config changed')
        for split in ('fit', 'val'):
            arr = inputs[f]
            for member, fold in zip(arr[split + '_recording'], arr[split + '_fold']):
                require(int(fold) in FOLDS and config['member_folds'][str(member)] == int(fold), 'member fold mismatch')
    require(len(records) == 1666, 'action population changed')
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open('rb') as stream:
            require(hashlib.file_digest(stream, 'md5').hexdigest() == expected, 'dataset checksum changed')
    with np.load(a.reference / 'spectra.npz', allow_pickle=False) as z:
        reference_freq = z['freq']; reference_spectra = dict(zip(z['row_id'], z['x']))
    with np.load(a.reference / 'features.npz', allow_pickle=False) as z:
        reference_features = dict(zip(z['row_id'], z['features'][:, 0]))
        reference_pairs = dict(zip(z['row_id'], z['pair_f0'][:, 0]))
        reference_triplets = dict(zip(z['row_id'], z['triplet_f0'][:, 0]))
    require(set(records) == set(reference_spectra), 'reference population changed')
    by_member = {}
    for row, (member, start) in records.items():
        require(config['member_folds'][member] in FOLDS, 'excluded recording')
        by_member.setdefault(member, []).append((row, start))
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in by_member}
    require(len(tracks) == len(by_member) == 122, 'recording population changed')
    tables = extraction_tables(reference_freq)
    results, max_spectrum_error, max_cost_error, max_frequency_error = {}, 0., 0., 0.
    started = time.monotonic()
    for member in sorted(tracks):
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768
        for row, start in by_member[member]:
            freq, xs, mass, power, padding = post_spectra(samples, start)
            np.testing.assert_array_equal(freq, reference_freq)
            max_spectrum_error = max(max_spectrum_error, float(np.max(np.abs(xs[0] - reference_spectra[row]))))
            residuals, pairs, triplets, pool = extract_windows(xs, tables)
            max_cost_error = max(max_cost_error, float(np.max(np.abs(residuals[0] - reference_features[row][:2]))))
            max_frequency_error = max(max_frequency_error, float(np.max(np.abs(pairs[0] - reference_pairs[row]))),
                                      float(np.max(np.abs(triplets[0] - reference_triplets[row]))))
            # Preserve the verified control inputs bit for bit after recomputation.
            value = np.column_stack([residuals[[0, 2]], np.median(triplets[[0, 2]], axis=1)])
            value[0] = reference_features[row]
            results[row] = (value, pairs, triplets, pool, xs[1], mass, power, padding, residuals[1])
        print(json.dumps({'recording': member, 'rows': len(results), 'seconds': time.monotonic() - started}), flush=True)
    require(max_spectrum_error == 0 and max_cost_error < 1e-10 and max_frequency_error == 0, 'first-window control drift')
    ids = np.array(sorted(results))
    columns = [np.array([results[int(row)][j] for row in ids]) for j in range(9)]
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'features.npz', row_id=ids, features=columns[0],
                        pair_f0=columns[1][:, [0, 2]], triplet_f0=columns[2][:, [0, 2]], variants=NAMES)
    np.savez_compressed(a.output / 'temporal.npz', row_id=ids, pair_f0=columns[1], triplet_f0=columns[2],
                        pool_ids=columns[3], positive_mass=columns[5], power=columns[6],
                        padding=columns[7], second_residuals=columns[8])
    np.savez_compressed(a.output / 'spectra-second.npz', row_id=ids, freq=reference_freq, x=columns[4])
    pair_stability = [match_frequencies(p[0], p[1])[0] for p in columns[1]]
    trip_stability = [match_frequencies(p[0], p[1])[0] for p in columns[2]]
    report = {'status': 'completed', 'folds': list(FOLDS), 'outer_fold_3_used': False,
              'annotation_frequencies_used': False, 'neural_training': False,
              'unique_audio_rows': len(ids), 'recordings': len(tracks),
              'first_window_max_spectrum_error': max_spectrum_error,
              'first_window_max_residual_error': max_cost_error,
              'first_window_max_frequency_error': max_frequency_error,
              'pair_persistence_counts': dict(Counter(str(v) for v in pair_stability)),
              'triplet_persistence_counts': dict(Counter(str(v) for v in trip_stability)),
              'zero_residual_windows': np.sum(columns[5] == 0, axis=0).tolist(),
              'padded_windows': np.sum(columns[7] > 0, axis=0).tolist(),
              'added_lookahead_ms': h.WINDOW * 1000 / SAMPLE_RATE,
              'total_lookahead_ms': 2 * h.WINDOW * 1000 / SAMPLE_RATE,
              'source_run': 37356100423, 'source_data_md5': DATA_MD5,
              'source_manifests_sha256': {str(f): sha256_file(a.exports / f'fold-{f}/manifest.json') for f in FOLDS},
              'source_sha256': {'reference_spectra': sha256_file(a.reference / 'spectra.npz'),
                                'reference_features': sha256_file(a.reference / 'features.npz'),
                                'config': sha256_file(a.config), 'protocol': sha256_file(PROTOCOL),
                                'script': sha256_file(__file__)},
              'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                          for p in ('numpy', 'scipy', 'scikit-learn')}}}
    write_json(a.output / 'extraction-report.json', report)
    print(json.dumps(report, sort_keys=True), flush=True)


def select_on_fit(X, y, folds, saved_models=None):
    require(set(folds) <= set(FOLDS), 'excluded FIT fold')
    reports, predictions = [], []
    for i, name in enumerate(NAMES):
        p = np.empty(len(y))
        for heldout in np.unique(folds):
            valid = folds == heldout
            state = []
            p[valid] = probability(X[~valid, i, :2], y[~valid], X[valid, i, :2], state)
            if saved_models is not None:
                saved_models.setdefault(name, {})[str(int(heldout))] = state[0]
        reports.append(accounting(y, p)); predictions.append(p)
    chosen = min(range(len(NAMES)), key=lambda i: (-reports[i]['global_net'], reports[i]['regressions'], reports[i]['applied'], i))
    return chosen if reports[chosen]['global_net'] > 0 else None, reports, np.asarray(predictions)


def evaluate(a):
    require(not (a.output / 'report.json').exists(), 'refusing overwrite')
    extraction = json.loads((a.output / 'extraction-report.json').read_text())
    require(extraction['triplet_persistence_counts'].get('3', 0) < 1666, 'no temporal change; Phase B not authorized')
    require(extraction['source_sha256']['protocol'] == sha256_file(PROTOCOL), 'protocol changed')
    inputs, _ = action_inputs(a.exports)
    with np.load(a.output / 'features.npz', allow_pickle=False) as z:
        feature_by_id = dict(zip(z['row_id'], z['features']))
    with np.load(a.reference / 'scores.npz', allow_pickle=False) as z:
        reference_scores = dict(z)
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
        internal_models = {}
        selected, cv, inner = select_on_fit(fit['X'], fit['y'], fit['fold'], internal_models)
        states = []
        val_probability = np.array([probability(fit['X'][:, i, :2], fit['y'], val['X'][:, i, :2], states)
                                    for i in range(len(NAMES))])
        for kind, actual in (('inner_probability', inner[0]), ('val_probability', val_probability[0])):
            np.testing.assert_array_equal(actual, reference_scores[f'fold_{fold}_{kind}'][0])
        write_json(a.output / f'models-fold-{fold}.json', dict(zip(NAMES, states)))
        write_json(a.output / f'inner-models-fold-{fold}.json', internal_models)
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
              'unique_audio_rows': 1666, 'recordings': 122,
              'total_selected': total([r['val_selected'] for r in reports]),
              'total_fixed': {name: total([r['val_fixed'][name] for r in reports]) for name in NAMES},
              'control_max_probability_error': 0.,
              'selection': 'FIT-only rotation; net, fewer regressions, fewer actions, arm order; abstain if net <= 0',
              'source_sha256': {'extraction': sha256_file(a.output / 'extraction-report.json'),
                                'features': sha256_file(a.output / 'features.npz'),
                                'protocol': sha256_file(PROTOCOL), 'script': sha256_file(__file__)},
              'limitations': ['Previously inspected internal folds, not untouched final validation.',
                              'Normal audio only; no compressed-path inference.',
                              '46.44 ms extra lookahead; foreign later onsets can enter.',
                              'Shared frequency set with independent amplitudes is not strict persistence.']}
    previous = json.loads((a.reference / 'report.json').read_text())
    require(report['total_fixed']['first_window'] == previous['total_fixed']['gaussian_t2'], 'control accounting drift')
    write_json(a.output / 'report.json', report)
    print(json.dumps({'fixed': report['total_fixed'], 'selected': report['total_selected'],
                      'selections': [r['selected'] for r in reports]}, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('extract', 'evaluate'))
    for name in ('exports', 'reference', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    a = p.parse_args()
    (extract if a.phase == 'extract' else evaluate)(a)
