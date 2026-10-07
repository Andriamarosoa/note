"""FIT-only selection of existing feature families on frozen internal exports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import index_guitarset
from scripts.audit_v273_internal_residual_acoustics import DATA_MD5
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_low_band_fold2_vs_fold4 import inference_features
from scripts.audit_v273_residual_feature_family_lofo import (
    ATTACK, GEOM, VARIANTS, classifier, counts, meta, robust_threshold,
)
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.v273_residual_audit import FOLDS, model_state, require, sha256_file, write_json

PROTOCOL = Path('analysis/v273-feature-family-nested-protocol.md')
NAMES = tuple(VARIANTS)
ALL_FEATURES = VARIANTS['base_geom_attack']
POLICIES = ('fixed050', 'robust_base', 'robust_family')
DEPENDENCIES = (
    'scripts/audit_v273_feature_family_nested.py',
    'scripts/audit_v273_residual_feature_family_lofo.py',
    'scripts/audit_v273_low_band_fold2_vs_fold4.py',
    'scripts/audit_v273_internal_b_low_harmonic_strata.py',
    'scripts/audit_v273_selected_f0_attack_exclusive.py',
    'scripts/audit_v273_attack_novelty.py',
    'scripts/audit_v273_exclusive_harmonic_support.py',
    'scripts/v273_residual_audit.py',
)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def columns(name):
    return [ALL_FEATURES.index(n) for n in VARIANTS[name]]


def fit_model(X, y, name):
    use = np.isin(y, (2, 3))
    require(len(set(y[use])) == 2, 'training class collapse')
    m = classifier().fit(np.array(X[use], dtype=float, order='C'), (y[use] == 2).astype(int))
    state = model_state(m)
    state['features'] = list(VARIANTS[name])
    return state


def score(X, state, name):
    require(state['features'] == list(VARIANTS[name]) and state['classes'] == [0, 1], 'model schema changed')
    X = np.asarray(X, float)
    return expit(((X - state['scale_mean']) / state['scale_scale']) @ np.asarray(state['coef']) + state['intercept'])


def split(data, fold, part):
    prefix = f'fold_{fold}_{part}_'
    return {k[len(prefix):]: v for k, v in data.items() if k.startswith(prefix)}


def validate_split(fit, val, heldout):
    require(heldout in FOLDS, 'excluded heldout fold')
    require(set(fit['fold']) == set(FOLDS) - {heldout}, 'excluded or missing FIT fold')
    require(set(val['fold']) == {heldout}, 'wrong VAL fold')
    require(not set(fit['row_id']) & set(val['row_id']), 'FIT/VAL row overlap')
    require(not set(fit['recording']) & set(val['recording']), 'FIT/VAL recording overlap')
    for d in (fit, val):
        require(d['X'].shape == (len(d['y']), len(ALL_FEATURES)), 'feature shape changed')
        require(np.isfinite(d['X']).all() and np.isin(d['y'], range(7)).all(), 'invalid inputs')


def rows(d):
    return [dict(true_k=int(y), fold=int(f), recording_id=str(r), **meta(str(r)))
            for y, f, r in zip(d['y'], d['fold'], d['recording'])]


def inner_models(fit):
    require(set(fit['fold']) <= set(FOLDS) and len(set(fit['fold'])) == 3, 'excluded or incomplete inner folds')
    probabilities = np.empty((len(NAMES), len(fit['y'])))
    states = {}
    for i, name in enumerate(NAMES):
        states[name] = {}
        X = fit['X'][:, columns(name)]
        for heldout in sorted(set(fit['fold'])):
            use = fit['fold'] != heldout
            require(not set(fit['recording'][use]) & set(fit['recording'][~use]), 'inner recording overlap')
            state = fit_model(X[use], fit['y'][use], name)
            probabilities[i, ~use] = score(X[~use], state, name)
            states[name][str(int(heldout))] = state
    return probabilities, states


def choose_on_fit(fit, probabilities):
    """No VAL argument or global label access; both policies may abstain."""
    r = rows(fit)
    require(probabilities.shape == (len(NAMES), len(r)), 'FIT probability shape')
    fixed = {name: counts(r, probabilities[i], .5) for i, name in enumerate(NAMES)}
    winner = min(NAMES, key=lambda n: (-fixed[n]['net'], fixed[n]['regressions'], fixed[n]['actions'], NAMES.index(n)))
    fixed_choice = {'family': winner, 'threshold': .5} if fixed[winner]['net'] > 0 else None
    best = {name: robust_threshold(r, probabilities[i])[0] for i, name in enumerate(NAMES)}
    use = np.isin(fit['y'], (2, 3))
    aucs = {name: float(roc_auc_score(fit['y'][use] == 2, probabilities[i, use])) for i, name in enumerate(NAMES)}
    baseline = best['base']
    baseline_choice = None if baseline is None else {'family': 'base', 'threshold': baseline['threshold']}
    base_net = 0 if baseline is None else baseline['total']['net']
    candidates = [n for n in NAMES[1:] if best[n] is not None]
    chosen = baseline_choice
    if candidates:
        n = max(candidates, key=lambda n: (best[n]['total']['net'], aucs[n], -len(VARIANTS[n]), -NAMES.index(n)))
        if best[n]['total']['net'] > base_net:
            chosen = {'family': n, 'threshold': best[n]['threshold']}
    return {'fixed050': fixed_choice, 'robust_base': baseline_choice, 'robust_family': chosen}, {
        'fixed050': fixed, 'robust_best': best, 'auc': aucs,
    }


def decisions(val, probabilities, choices):
    r = rows(val)
    result = {}
    for policy, choice in choices.items():
        if choice is None:
            p, threshold = np.zeros(len(r)), 1.
        else:
            p, threshold = probabilities[NAMES.index(choice['family'])], choice['threshold']
        result[policy] = counts(r, p, threshold)
    return result


def extract(a):
    require(not a.output.exists(), 'refusing overwrite')
    inputs, records = action_inputs(a.exports)
    cfg = json.loads(a.config.read_text())
    require(len(records) == 1666, 'physical population changed')
    for f in FOLDS:
        m = json.loads((a.exports / f'fold-{f}/manifest.json').read_text())
        require(m['config_sha256'] == sha256_file(a.config), 'config changed')
        for part in ('fit', 'val'):
            for member, source_fold in zip(inputs[f][part + '_recording'], inputs[f][part + '_fold']):
                require(int(source_fold) in FOLDS and cfg['member_folds'][str(member)] == int(source_fold), 'excluded source row')
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open('rb') as stream:
            require(hashlib.file_digest(stream, 'md5').hexdigest() == expected, 'dataset changed')
    by_member = {}
    for rid, (member, start) in records.items():
        require(cfg['member_folds'][member] in FOLDS, 'excluded audio')
        by_member.setdefault(member, []).append((rid, start))
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in by_member}
    require(set(tracks) == set(by_member) and len(tracks) == 122, 'audio population changed')
    extra = {}
    for member in sorted(tracks):
        t = tracks[member]
        au = decode_pcm16_mono_wav(t.audio_zip, t.audio_member)
        samples = np.asarray(au.samples, float) / 32768.
        for rid, start in by_member[member]:
            q = inference_features(samples, start)
            extra[rid] = [q[n] for n in GEOM + ATTACK]
        print(json.dumps({'recording': member, 'rows': len(extra)}), flush=True)
    saved = {}
    for f, arr in inputs.items():
        for part in ('fit', 'val'):
            mask = arr[part + '_b_low'] & (arr[part + '_base_k'] == 3)
            valid = arr[part + '_valid']
            prefix = f'fold_{f}_{part}_'
            for field, source in (('row_id', 'ids'), ('y', 'true_k'), ('fold', 'fold'),
                                  ('recording', 'recording'), ('start', 'start_sample')):
                saved[prefix + field] = arr[part + '_' + source][mask][valid]
            saved[prefix + 'X'] = np.column_stack([arr[part + '_X'][valid],
                                                 np.array([extra[int(i)] for i in saved[prefix + 'row_id']])])
            saved[prefix + 'probability'] = arr[part + '_probability']
        validate_split(split(saved, f, 'fit'), split(saved, f, 'val'), f)
    require(sum(len(split(saved, f, 'val')['y']) for f in FOLDS) == 845, 'VAL population changed')
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'inputs.npz', **saved)
    ids = np.array(sorted(extra))
    np.savez_compressed(a.output / 'extra-features.npz', row_id=ids, features=np.array([extra[i] for i in ids]),
                        recording=np.array([records[i][0] for i in ids]), start=np.array([records[i][1] for i in ids]),
                        names=GEOM + ATTACK)
    write_json(a.output / 'extraction-report.json', {
        'status': 'completed', 'folds': list(FOLDS), 'outer_fold_3_used': False,
        'rows': 1666, 'recordings': 122, 'val_rows': 845, 'bundle_loaded': False,
        'annotation_frequencies_used': False, 'neural_training': False,
        'data_md5': DATA_MD5, 'config_sha256': sha256_file(a.config),
        'manifests_sha256': {str(f): sha256_file(a.exports / f'fold-{f}/manifest.json') for f in FOLDS},
        'sources_sha256': {p: sha256_file(p) for p in (*DEPENDENCIES, str(PROTOCOL))},
        'files_sha256': {n: sha256_file(a.output / n) for n in ('inputs.npz', 'extra-features.npz')},
    })


def total(folds, key):
    keys = ('actions', 'corrections', 'regressions', 'other_k', 'net')
    return {k: sum(r[key][k] for r in folds) for k in keys}


def evaluate(a):
    require(not (a.output / 'report.json').exists(), 'refusing evaluation overwrite')
    source = json.loads(a.reference.read_text())
    data = arrays(a.output / 'inputs.npz')
    results, models, scores = [], {}, {}
    baseline_error = 0.
    for f in FOLDS:
        fit, val = split(data, f, 'fit'), split(data, f, 'val')
        validate_split(fit, val, f)
        inner, states = inner_models(fit)
        choices, diagnostics = choose_on_fit(fit, inner)
        final, probabilities = {}, []
        for name in NAMES:
            Xf, Xv = fit['X'][:, columns(name)], val['X'][:, columns(name)]
            final[name] = fit_model(Xf, fit['y'], name)
            probabilities.append(score(Xv, final[name], name))
        probabilities = np.array(probabilities)
        error = float(np.max(np.abs(probabilities[0] - val['probability'])))
        baseline_error = max(baseline_error, error)
        require(error < 1e-12, 'baseline probability changed')
        fixed = {name: counts(rows(val), probabilities[i], .5) for i, name in enumerate(NAMES)}
        for name in NAMES:
            expected = next(q['p050'] for q in source['variants'][name]['per_fold'] if q['fold'] == f)
            require(fixed[name] == expected, 'source family reproduction changed')
        result = {'fold': f, 'choices': choices, 'fit_diagnostics': diagnostics,
                  'val_fixed050': fixed, 'val_selected': decisions(val, probabilities, choices),
                  'val_enriched059_control': counts(rows(val), probabilities[3], .59)}
        results.append(result); models[str(f)] = {'inner': states, 'final': final}
        scores[f'fold_{f}_inner'] = inner; scores[f'fold_{f}_val'] = probabilities
    policies = {p: total([r['val_selected'] for r in results], p) for p in POLICIES}
    control = total(results, 'val_enriched059_control')
    require(control == source['variants']['base_geom_attack']['robust_best']['total'], 'fixed 0.59 control changed')
    np.savez_compressed(a.output / 'scores.npz', **scores)
    write_json(a.output / 'models.json', models)
    write_json(a.output / 'report.json', {
        'status': 'completed', 'folds': list(FOLDS), 'outer_fold_3_used': False,
        'annotation_frequencies_used': False, 'neural_training': False, 'automatic_promotion': False,
        'fold_results': results, 'total_selected': policies, 'descriptive_control059': control,
        'baseline_max_probability_error': baseline_error,
        'source_report_sha256': sha256_file(a.reference), 'extraction_sha256': sha256_file(a.output / 'extraction-report.json'),
        'limitations': ['Previously inspected internal folds, not independent validation.',
                        'Frozen routing was estimated upstream; only this residual selection is nested.',
                        'Normal audio only; no compressed-path conclusion.'],
    })
    print(json.dumps({'selected': policies, 'control059': control, 'baseline_error': baseline_error}, sort_keys=True))


def replay(root, reference):
    extraction = json.loads((root / 'extraction-report.json').read_text())
    report = json.loads((root / 'report.json').read_text())
    for r in (extraction, report):
        require(r['status'] == 'completed' and tuple(r['folds']) == FOLDS and not r['outer_fold_3_used'], 'scope changed')
    for path, digest in extraction['sources_sha256'].items():
        require(sha256_file(path) == digest, 'source changed: ' + path)
    for name, digest in extraction['files_sha256'].items():
        require(sha256_file(root / name) == digest, 'input file changed')
    require(sha256_file(root / 'extraction-report.json') == report['extraction_sha256'], 'provenance changed')
    require(sha256_file(reference) == report['source_report_sha256'], 'reference report changed')
    data, saved = arrays(root / 'inputs.npz'), arrays(root / 'scores.npz')
    extra = arrays(root / 'extra-features.npz')
    lookup = {int(i): j for j, i in enumerate(extra['row_id'])}
    models = json.loads((root / 'models.json').read_text())
    source = json.loads(reference.read_text())
    results, max_error = [], 0.
    for r in report['fold_results']:
        f = r['fold']; fit, val = split(data, f, 'fit'), split(data, f, 'val')
        validate_split(fit, val, f)
        for d in (fit, val):
            ix = [lookup[int(i)] for i in d['row_id']]
            np.testing.assert_array_equal(d['X'][:, 2:], extra['features'][ix])
            np.testing.assert_array_equal(d['recording'], extra['recording'][ix])
            np.testing.assert_array_equal(d['start'], extra['start'][ix])
        inner, pv = [], []
        for i, name in enumerate(NAMES):
            p = np.empty(len(fit['y'])); Xf, Xv = fit['X'][:, columns(name)], val['X'][:, columns(name)]
            for heldout in sorted(set(fit['fold'])):
                use = fit['fold'] == heldout
                state = models[str(f)]['inner'][name][str(int(heldout))]
                require(state['scale_n_samples_seen'] == int(np.sum(~use & np.isin(fit['y'], (2, 3)))), 'inner train size changed')
                p[use] = score(Xf[use], state, name)
            state = models[str(f)]['final'][name]
            require(state['scale_n_samples_seen'] == int(np.sum(np.isin(fit['y'], (2, 3)))), 'final train size changed')
            v = score(Xv, state, name)
            for actual, expected in ((p, saved[f'fold_{f}_inner'][i]), (v, saved[f'fold_{f}_val'][i])):
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            inner.append(p); pv.append(v)
            require(counts(rows(val), v, .5) == r['val_fixed050'][name], 'fixed decisions changed')
            expected = next(q['p050'] for q in source['variants'][name]['per_fold'] if q['fold'] == f)
            require(counts(rows(val), v, .5) == expected, 'source control changed')
        choices, diagnostics = choose_on_fit(fit, np.array(inner))
        require(choices == r['choices'] and diagnostics == r['fit_diagnostics'], 'FIT selection changed')
        outcomes = decisions(val, np.array(pv), choices)
        require(outcomes == r['val_selected'], 'VAL outcomes changed')
        results.append(outcomes)
    require(max_error < 1e-12, 'score replay drift')
    require({p: total(results, p) for p in POLICIES} == report['total_selected'], 'totals changed')
    return {'status': 'verified', 'folds': list(FOLDS), 'outer_fold_3_used': False,
            'audio_loaded': False, 'model_refitted': False, 'final_models': 16, 'inner_models': 48,
            'max_probability_error': max_error, 'total_selected': report['total_selected'],
            'source_sha256': {n: sha256_file(root / n) for n in ('report.json', 'inputs.npz', 'extra-features.npz', 'models.json', 'scores.npz')}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('extract', 'evaluate', 'replay'))
    for name in ('exports', 'dataset', 'reference'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.phase == 'replay':
        result = replay(a.output, a.reference)
        write_json(a.output / 'replay-check.json', result)
        print(json.dumps(result, sort_keys=True))
    else:
        (extract if a.phase == 'extract' else evaluate)(a)
