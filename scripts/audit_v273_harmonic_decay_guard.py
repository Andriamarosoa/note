"""FIT-only selection of harmonic decay exponents on frozen action populations.

No annotated frequencies, new neural training, or fold 3. Both score decay and
template decay use the fixed grid (0.5, 1, 2); all other acoustic knobs are fixed.
"""
from __future__ import annotations

import argparse
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
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5
from scripts.v273_residual_audit import FOLDS, fit_classifier, model_state, require, sha256_file, verify_export, write_json

EXPONENTS = (0.5, 1.0, 2.0)
VARIANTS = tuple((s, t) for s in EXPONENTS for t in EXPONENTS)
NAMES = tuple(f's{s:g}_t{t:g}' for s, t in VARIANTS)


def best_batch(D, x, k):
    """The existing exhaustive NNLS calculation, batched across combinations."""
    G, b, x2 = D.T @ D, D.T @ x, float(x @ x)
    combos = np.array(list(itertools.combinations(range(D.shape[1]), k)), dtype=int)
    GG = G[combos[:, :, None], combos[:, None, :]]
    bb = b[combos]
    costs = np.full(len(combos), x2)
    coefficients = np.zeros((len(combos), k))
    for n in range(1, k + 1):
        for active in itertools.combinations(range(k), n):
            ids = list(active)
            ga = GG[:, ids][:, :, ids]
            ba = bb[:, ids]
            aa = np.linalg.solve(ga + 1e-10 * np.eye(n), ba[..., None])[..., 0]
            value = x2 - 2 * np.einsum('ij,ij->i', aa, ba) + np.einsum('ij,ijk,ik->i', aa, ga, aa)
            better = np.all(aa >= 0, axis=1) & (value < costs)
            costs[better] = np.maximum(value[better], 0)
            coefficients[better] = 0
            for j, column in enumerate(ids):
                coefficients[better, column] = aa[better, j]
    winner = int(costs.argmin())
    return (float(costs[winner]), tuple(combos[winner]), coefficients[winner]), x2


def extraction_tables(freq):
    centers = h.F0_GRID[:, None] * np.arange(1, h.MAX_HARMONICS + 1)
    nearest = np.abs(centers[:, :, None] - freq).argmin(axis=2)
    neighbors = np.clip(nearest[:, :, None] + np.arange(-2, 3), 0, len(freq) - 1)
    active = centers <= freq[-1]
    dictionaries = {}
    for exponent in EXPONENTS:
        D = np.zeros((len(freq), len(h.F0_GRID)))
        for harmonic in range(1, h.MAX_HARMONICS + 1):
            B = np.exp(-0.5 * ((freq[:, None] - harmonic * h.F0_GRID) / h.KERNEL_HZ) ** 2)
            B[:, ~active[:, harmonic - 1]] = 0
            denominator = math.sqrt(harmonic) if exponent == .5 else harmonic ** exponent
            D += B / denominator
        dictionaries[exponent] = D / (np.linalg.norm(D, axis=0) + 1e-12)
    return neighbors, active, dictionaries


def extract_variants(freq, x, tables):
    neighbors, active, dictionaries = tables
    peaks = x[neighbors].max(axis=2) * active
    pools = {}
    for exponent in EXPONENTS:
        scores = np.zeros(len(h.F0_GRID))
        for j in range(h.MAX_HARMONICS):
            denominator = math.sqrt(j + 1) if exponent == .5 else (j + 1) ** exponent
            scores += peaks[:, j] / denominator
        selected = []
        for i in np.argsort(-scores, kind='stable'):
            if any(h.cents(h.F0_GRID[i], h.F0_GRID[j]) < h.NMS_CENTS for j in selected):
                continue
            selected.append(int(i))
            if len(selected) == h.POOL_SIZE:
                break
        pools[exponent] = selected
    result = []
    for score_exponent, template_exponent in VARIANTS:
        pool = pools[score_exponent]
        D = dictionaries[template_exponent][:, pool]
        pair, x2 = best_batch(D, x, 2)
        trip, _ = best_batch(D, x, 3)
        result.append([pair[0] / (x2 + 1e-12), trip[0] / (x2 + 1e-12),
                       float(np.median(h.F0_GRID[np.asarray(pool)[list(trip[1])]]))])
    return np.asarray(result)


def accounting(y, probability):
    action = probability >= .5
    correct = int(np.sum(action & (y == 2)))
    regress = int(np.sum(action & (y == 3)))
    return {'rows': len(y), 'applied': int(action.sum()), 'corrections': correct,
            'regressions': regress, 'other_k_actions': int(np.sum(action & ~np.isin(y, (2, 3)))),
            'global_net': correct - regress}


def probability(Xfit, yfit, Xval, states=None):
    train = np.isin(yfit, (2, 3))
    require(len(np.unique(yfit[train])) == 2, 'one-class FIT')
    model = fit_classifier(Xfit[train], (yfit[train] == 2).astype(int))
    if states is not None:
        states.append(model_state(model))
    return model.predict_proba(Xval)[:, 1]


def select_on_fit(X, y, fold):
    """Selection consumes FIT only, with recording-disjoint folds inherited."""
    scores = []
    predictions = []
    for variant in range(len(NAMES)):
        p = np.empty(len(y))
        for heldout in np.unique(fold):
            valid = fold == heldout
            p[valid] = probability(X[~valid, variant, :2], y[~valid], X[valid, variant, :2])
        scores.append(accounting(y, p))
        predictions.append(p)
    chosen = min(range(len(NAMES)), key=lambda i: (-scores[i]['global_net'], scores[i]['regressions'], scores[i]['applied'], i))
    if scores[chosen]['global_net'] <= 0:
        chosen = None
    return chosen, scores, np.asarray(predictions)


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    inputs, records = {}, {}
    for fold in FOLDS:
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
            if m != member:
                continue
            freq, x = h.transition_spectrum(samples, start)
            if tables is None:
                tables = extraction_tables(freq)
            computed[row] = extract_variants(freq, x, tables)
        print(json.dumps({'recording': member, 'rows': len(computed), 'seconds': time.monotonic() - started}), flush=True)
    ids = np.array(sorted(computed))
    values = np.array([computed[int(i)] for i in ids])
    with np.load(a.stable / 'stable-features.npz', allow_pickle=False) as previous:
        np.testing.assert_array_equal(ids, previous['row_id'])
        control_error = float(np.max(np.abs(values[:, 0, :2] - previous['features'][:, :2])))
        require(control_error < 1e-10, 'stable control extraction changed')
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'features.npz', row_id=ids, features=values, variants=NAMES)
    scores, reports = {}, []
    for fold, arr in inputs.items():
        data = {}
        for split in ('fit', 'val'):
            mask = arr[split + '_b_low'] & (arr[split + '_base_k'] == 3)
            valid = arr[split + '_valid']
            row_ids = arr[split + '_ids'][mask][valid]
            data[split] = dict(X=np.array([computed[int(i)] for i in row_ids]),
                               y=arr[split + '_true_k'][mask][valid],
                               fold=arr[split + '_fold'][mask][valid], ids=row_ids)
        fit, val = data['fit'], data['val']
        selected, cv, inner_p = select_on_fit(fit['X'], fit['y'], fit['fold'])
        states = []
        vp = np.array([probability(fit['X'][:, i, :2], fit['y'], val['X'][:, i, :2], states) for i in range(len(NAMES))])
        write_json(a.output / f'models-fold-{fold}.json', dict(zip(NAMES, states)))
        chosen_p = np.zeros(len(val['y'])) if selected is None else vp[selected]
        scores[f'fold_{fold}_inner_probability'] = inner_p
        scores[f'fold_{fold}_val_probability'] = vp
        scores[f'fold_{fold}_val_ids'] = val['ids']
        scores[f'fold_{fold}_fit_ids'] = fit['ids']
        for split in ('fit', 'val'):
            for key in ('y', 'fold'):
                scores[f'fold_{fold}_{split}_{key}'] = data[split][key]
        reports.append({'fold': fold, 'selected': None if selected is None else NAMES[selected],
                        'fit_cv': dict(zip(NAMES, cv)),
                        'val_fixed': {name: accounting(val['y'], p) for name, p in zip(NAMES, vp)},
                        'val_selected': accounting(val['y'], chosen_p)})
    np.savez_compressed(a.output / 'scores.npz', **scores)
    totals = lambda rr: {k: sum(r[k] for r in rr) for k in ('rows', 'applied', 'corrections', 'regressions', 'other_k_actions', 'global_net')}
    report = {'status': 'completed', 'folds': reports, 'outer_fold_3_used': False,
              'annotation_frequencies_used': False, 'neural_training': False,
              'automatic_promotion': False, 'unique_audio_rows': len(ids), 'recordings': len(wanted),
              'control_max_abs_residual_error': control_error, 'source_run': 37356100423,
              'selection': 'FIT-only inner fold rotation; maximize net, fewer regressions, fewer actions, grid order; abstain if net <= 0',
              'total_selected': totals([r['val_selected'] for r in reports]),
              'total_fixed': {name: totals([r['val_fixed'][name] for r in reports]) for name in NAMES},
              'source_sha256': {name: sha256_file(Path(__file__).parent / name) for name in
                                ('audit_v273_harmonic_decay_guard.py', 'audit_v273_internal_b_low_harmonic_strata.py', 'v273_residual_audit.py')},
              'source_data_md5': DATA_MD5,
              'source_manifests_sha256': {str(f): sha256_file(a.exports / f'fold-{f}' / 'manifest.json') for f in FOLDS},
              'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')}},
              'limitations': ['Internal folds previously inspected; exploratory evaluation, not an untouched test set.',
                              'Base neural predictions and B_low routing remain frozen inside FIT selection.',
                              'VAL outcomes of individual variants are descriptive and never select the variant.']}
    previous = json.loads((a.stable / 'report.json').read_text())
    for key, value in previous['totals']['stable'].items():
        require(report['total_fixed'][NAMES[0]][key] == value, 'stable control accounting changed')
    write_json(a.output / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'dataset', 'stable', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
