"""Isolate candidate capacity and template decay on frozen internal rows."""
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
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

VARIANTS = ((8, .5), (8, 2.), (64, .5), (64, 2.))
NAMES = tuple(f'pool{n}_t{t:g}' for n, t in VARIANTS)


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
    scores = np.zeros(len(h.F0_GRID))
    for j in range(h.MAX_HARMONICS):
        scores += peaks[:, j] / math.sqrt(j + 1)
    _, selected, _ = trace_nms(scores)
    values, triplets, pairs = [], [], []
    for capacity, exponent in VARIANTS:
        ids = selected[:capacity]
        D = dictionaries[exponent][:, ids]
        pair, x2 = best_batch(D, x, 2); trip, _ = best_batch(D, x, 3)
        pair_hz = h.F0_GRID[ids[list(pair[1])]]
        trip_hz = h.F0_GRID[ids[list(trip[1])]]
        values.append([pair[0] / (x2 + 1e-12), trip[0] / (x2 + 1e-12), float(np.median(trip_hz))])
        pairs.append(pair_hz); triplets.append(trip_hz)
    return np.array(values), np.array(pairs), np.array(triplets)


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    inputs, records = {}, {}
    for fold in FOLDS:
        from scripts.v273_residual_audit import verify_export
        folder = a.exports / f'fold-{fold}'
        verify_export(folder)
        with np.load(folder / 'replay.npz', allow_pickle=False) as z:
            arr = dict(z)
        inputs[fold] = arr
        for split in ('fit', 'val'):
            mask = arr[split + '_b_low'] & (arr[split + '_base_k'] == 3)
            valid = arr[split + '_valid']
            for row, member, start, f in zip(*(arr[split + '_' + key][mask][valid] for key in ('ids', 'recording', 'start_sample', 'fold'))):
                require(int(f) in FOLDS, 'outer fold')
                value = str(member), int(start)
                require(int(row) not in records or records[int(row)] == value, 'mixed row identity')
                records[int(row)] = value
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open('rb') as stream:
            require(hashlib.file_digest(stream, 'md5').hexdigest() == expected, 'dataset changed')
    wanted = {m for m, _ in records.values()}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, 'missing tracks')
    computed, tables, started = {}, None, time.monotonic()
    for member in sorted(wanted):
        t = tracks[member]
        audio = decode_pcm16_mono_wav(t.audio_zip, t.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768
        for row, (m, start) in records.items():
            if m == member:
                freq, x = h.transition_spectrum(samples, start)
                if tables is None:
                    tables = extraction_tables(freq)
                computed[row] = extract(freq, x, tables)
        print(json.dumps({'recording': member, 'rows': len(computed), 'seconds': time.monotonic() - started}), flush=True)
    ids = np.array(sorted(computed))
    values = np.array([computed[int(i)][0] for i in ids])
    pairs = np.array([computed[int(i)][1] for i in ids])
    triplets = np.array([computed[int(i)][2] for i in ids])
    with np.load(a.decay / 'features.npz', allow_pickle=False) as previous:
        np.testing.assert_array_equal(ids, previous['row_id'])
        control_error = float(np.max(np.abs(values[:, :2, :2] - previous['features'][:, [0, 2], :2])))
        require(control_error < 1e-10, 'pool-8 controls changed')
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
        vp = np.array([probability(fit['X'][:, i, :2], fit['y'], val['X'][:, i, :2], states) for i in range(len(NAMES))])
        write_json(a.output / f'models-fold-{fold}.json', dict(zip(NAMES, states)))
        for split in ('fit', 'val'):
            for key in ('ids', 'y', 'fold'):
                scores[f'fold_{fold}_{split}_{key}'] = data[split][key]
        scores[f'fold_{fold}_inner_probability'] = inner
        scores[f'fold_{fold}_val_probability'] = vp
        chosen_p = np.zeros(len(val['y'])) if selected is None else vp[selected]
        reports.append({'fold': fold, 'selected': None if selected is None else NAMES[selected],
                        'fit_cv': dict(zip(NAMES, cv)),
                        'val_fixed': {name: accounting(val['y'], p) for name, p in zip(NAMES, vp)},
                        'val_selected': accounting(val['y'], chosen_p)})
    np.savez_compressed(a.output / 'scores.npz', **scores)
    totals = lambda rr: {k: sum(r[k] for r in rr) for k in ('rows', 'applied', 'corrections', 'regressions', 'other_k_actions', 'global_net')}
    report = {'status': 'completed', 'folds': reports, 'variants': list(NAMES), 'outer_fold_3_used': False,
              'annotation_frequencies_used': False, 'neural_training': False, 'automatic_promotion': False,
              'unique_audio_rows': len(ids), 'recordings': len(wanted), 'control_max_abs_residual_error': control_error,
              'total_selected': totals([r['val_selected'] for r in reports]),
              'total_fixed': {name: totals([r['val_fixed'][name] for r in reports]) for name in NAMES},
              'selection': 'FIT-only rotation; net, fewer regressions, fewer actions, arm order; abstain if net <= 0',
              'source_manifests_sha256': {str(f): sha256_file(a.exports / f'fold-{f}' / 'manifest.json') for f in FOLDS},
              'source_data_md5': DATA_MD5,
              'source_sha256': {name: sha256_file(Path(__file__).parent / name) for name in
                                ('audit_v273_candidate_capacity_guard.py', 'audit_v273_candidate_ranking.py',
                                 'audit_v273_harmonic_decay_guard.py', 'audit_v273_internal_b_low_harmonic_strata.py', 'v273_residual_audit.py')},
              'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')}},
              'limitations': ['Internal folds previously inspected; not untouched validation.',
                              'Frozen base neural predictions and B_low routing, including inside FIT rotation.',
                              'Normal residual-audio path only; no compressed-path gain inferred.']}
    previous = json.loads((a.decay / 'report.json').read_text())
    for name, old in zip(NAMES[:2], ('s0.5_t0.5', 's0.5_t2')):
        require(report['total_fixed'][name] == previous['total_fixed'][old], 'control counts changed')
    write_json(a.output / 'report.json', report)
    print(json.dumps({'fixed': report['total_fixed'], 'selected': report['total_selected'], 'selection': [r['selected'] for r in reports]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'dataset', 'decay', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
