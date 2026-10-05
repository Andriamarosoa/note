"""Test one fixed log-frequency reconstruction measure on frozen action rows."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from scripts.audit_v273_harmonic_decay_guard import accounting, best_batch, extraction_tables, probability
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5
from scripts.v273_residual_audit import FOLDS, require, sha256_file, verify_export, write_json

NAMES = ('linear_hz', 'log_hz')


def measure_roots(freq):
    return (np.ones_like(freq), np.sqrt(h.MIN_HZ / freq))


def select_on_fit(X, y, fold):
    require(set(fold) <= set(FOLDS), 'outer fold')
    scores, predictions = [], []
    for i in range(len(NAMES)):
        p = np.empty(len(y))
        for heldout in np.unique(fold):
            valid = fold == heldout
            p[valid] = probability(X[~valid, i, :2], y[~valid], X[valid, i, :2])
        scores.append(accounting(y, p)); predictions.append(p)
    chosen = min(range(len(NAMES)), key=lambda i: (-scores[i]['global_net'], scores[i]['regressions'], scores[i]['applied'], i))
    return (chosen if scores[chosen]['global_net'] > 0 else None), scores, np.asarray(predictions)


def extract(freq, x, tables):
    neighbors, active, dictionaries = tables
    peaks = x[neighbors].max(axis=2) * active
    salience = np.zeros(len(h.F0_GRID))
    for j in range(h.MAX_HARMONICS):
        salience += peaks[:, j] / math.sqrt(j + 1)
    _, selected, _ = trace_nms(salience)
    ids = selected[:64]
    D = dictionaries[2.0][:, ids]
    values, pairs, triplets = [], [], []
    for root in measure_roots(freq):
        weighted_D, weighted_x = D * root[:, None], x * root
        pair, x2 = best_batch(weighted_D, weighted_x, 2)
        trip, _ = best_batch(weighted_D, weighted_x, 3)
        pair_hz = h.F0_GRID[ids[list(pair[1])]]
        trip_hz = h.F0_GRID[ids[list(trip[1])]]
        values.append([pair[0] / (x2 + 1e-12), trip[0] / (x2 + 1e-12), float(np.median(trip_hz))])
        pairs.append(pair_hz); triplets.append(trip_hz)
    return np.array(values), np.array(pairs), np.array(triplets)


def action_inputs(exports):
    inputs, records = {}, {}
    for fold in FOLDS:
        folder = exports / f'fold-{fold}'
        verify_export(folder)
        with np.load(folder / 'replay.npz', allow_pickle=False) as z:
            arr = dict(z)
        inputs[fold] = arr
        for split in ('fit', 'val'):
            mask = arr[split + '_b_low'] & (arr[split + '_base_k'] == 3)
            valid = arr[split + '_valid']
            for row, member, start, row_fold in zip(*(arr[split + '_' + key][mask][valid]
                                                       for key in ('ids', 'recording', 'start_sample', 'fold'))):
                require(int(row_fold) in FOLDS, 'outer fold')
                value = str(member), int(start)
                require(int(row) not in records or records[int(row)] == value, 'mixed row identity')
                records[int(row)] = value
    return inputs, records


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    inputs, records = action_inputs(a.exports)
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open('rb') as stream:
            require(hashlib.file_digest(stream, 'md5').hexdigest() == expected, 'dataset changed')
    wanted = {member for member, _ in records.values()}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, 'missing tracks')
    computed, tables, started = {}, None, time.monotonic()
    by_member = {}
    for row, (member, start) in records.items():
        by_member.setdefault(member, []).append((row, start))
    for member in sorted(wanted):
        track = tracks[member]
        audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768
        for row, start in by_member[member]:
            freq, x = h.transition_spectrum(samples, start)
            if tables is None:
                tables = extraction_tables(freq)
            computed[row] = extract(freq, x, tables)
        print(json.dumps({'recording': member, 'rows': len(computed), 'seconds': time.monotonic() - started}), flush=True)
    ids = np.array(sorted(computed))
    values = np.array([computed[int(i)][0] for i in ids])
    pairs = np.array([computed[int(i)][1] for i in ids])
    triplets = np.array([computed[int(i)][2] for i in ids])
    with np.load(a.capacity / 'features.npz', allow_pickle=False) as previous:
        np.testing.assert_array_equal(ids, previous['row_id'])
        control_error = float(np.max(np.abs(values[:, 0, :2] - previous['features'][:, 3, :2])))
        control_frequency_error = float(max(np.max(np.abs(pairs[:, 0] - previous['pair_f0'][:, 3])),
                                            np.max(np.abs(triplets[:, 0] - previous['triplet_f0'][:, 3]))))
        require(control_error < 1e-10 and control_frequency_error == 0, 'linear-Hz control changed')
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'features.npz', row_id=ids, features=values,
                        pair_f0=pairs, triplet_f0=triplets, variants=NAMES)
    reports, scores = [], {}
    for fold, arr in inputs.items():
        data = {}
        for split in ('fit', 'val'):
            mask = arr[split + '_b_low'] & (arr[split + '_base_k'] == 3)
            valid = arr[split + '_valid']
            row_ids = arr[split + '_ids'][mask][valid]
            data[split] = dict(X=np.array([computed[int(i)][0] for i in row_ids]),
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
              'weighting': {'linear_hz': 'w(f)=1', 'log_hz': f'w(f)={h.MIN_HZ}/f; fixed before evaluation'},
              'control_max_abs_residual_error': control_error,
              'control_max_abs_frequency_error': control_frequency_error,
              'total_selected': total([r['val_selected'] for r in reports]),
              'total_fixed': {name: total([r['val_fixed'][name] for r in reports]) for name in NAMES},
              'selection': 'FIT-only rotation; net, fewer regressions, fewer actions, arm order; abstain if net <= 0',
              'source_run': 37356100423, 'source_data_md5': DATA_MD5,
              'source_manifests_sha256': {str(f): sha256_file(a.exports / f'fold-{f}' / 'manifest.json') for f in FOLDS},
              'source_sha256': {name: sha256_file(Path(__file__).parent / name) for name in
                                ('audit_v273_log_frequency_guard.py', 'audit_v273_candidate_ranking.py',
                                 'audit_v273_harmonic_decay_guard.py', 'audit_v273_internal_b_low_harmonic_strata.py',
                                 'v273_residual_audit.py')},
              'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                          for p in ('numpy', 'scipy', 'scikit-learn')}},
              'limitations': ['Internal folds previously inspected; not untouched validation.',
                              'Frozen base neural predictions and B_low routing, including inside FIT rotation.',
                              'Normal residual-audio path only; no compressed-path gain inferred.']}
    previous = json.loads((a.capacity / 'report.json').read_text())
    require(report['total_fixed']['linear_hz'] == previous['total_fixed']['pool64_t2'], 'control accounting changed')
    write_json(a.output / 'report.json', report)
    print(json.dumps({'fixed': report['total_fixed'], 'selected': report['total_selected'],
                      'selection': [r['selected'] for r in reports]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'dataset', 'capacity', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
