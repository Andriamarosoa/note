"""Recompute all two-window spectra features and frozen models without audio/refit."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.audit_v273_harmonic_decay_guard import accounting, extraction_tables
from scripts.audit_v273_internal_residual_acoustics import GROUPS
from scripts.audit_v273_temporal_stability import extract_windows, NAMES as JOINT_NAMES, PROTOCOL as JOINT_PROTOCOL
from scripts.audit_v273_temporal_persistence_guard import (
    FEATURE_NAMES, NAMES as PERSISTENCE_NAMES, PROTOCOL as PERSISTENCE_PROTOCOL,
    persistence_features, score,
)
from scripts.diagnose_v273_temporal_stability import aggregate, diagnose, second_context
from scripts.replay_v273_harmonic_decay_guard import verify as verify_joint_models
from scripts.v273_residual_audit import FOLDS, FEATURES, require, score_from_state, sha256_file, write_json


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def feature_diagnostic(root):
    features = arrays(root / 'features.npz')
    by_id = dict(zip(features['row_id'], features['features'][:, 2]))
    scores = arrays(root / 'scores.npz')
    ids = np.concatenate([scores[f'fold_{f}_val_ids'] for f in FOLDS])
    y = np.concatenate([scores[f'fold_{f}_val_y'] for f in FOLDS])
    x = np.array([by_id[i] for i in ids]); use = np.isin(y, (2, 3))
    result = {'rows': len(y), 'by_true_k': {},
              'K3_vs_K2_persistence_auc': float(roc_auc_score(y[use] == 3, x[use])),
              'annotation_use': 'diagnostic after evaluation; no threshold selection',
              'source_scores_sha256': sha256_file(root / 'scores.npz')}
    for k in sorted(set(y)):
        v = x[y == k]
        result['by_true_k'][str(k)] = {'rows': len(v), 'mean_persistent_components': float(3 * v.mean()),
                                       'counts': dict(Counter(str(round(q * 3)) for q in v))}
    return result


def verify_inner_and_persistence(root, persistence=False):
    names = PERSISTENCE_NAMES if persistence else JOINT_NAMES
    report = json.loads((root / 'report.json').read_text())
    require(not report['outer_fold_3_used'] and report['status'] == 'completed', 'invalid report')
    require(tuple(r['fold'] for r in report['folds']) == FOLDS, 'fold set changed')
    feature = arrays(root / 'features.npz')
    by_id = dict(zip(feature['row_id'], feature['features']))
    scores = arrays(root / 'scores.npz')
    max_error, totals = 0., []
    for fold_report in report['folds']:
        f = fold_report['fold']; prefix = f'fold_{f}_'
        yf, yv = scores[prefix + 'fit_y'], scores[prefix + 'val_y']
        ff, vf = scores[prefix + 'fit_fold'], scores[prefix + 'val_fold']
        fit_ids, val_ids = scores[prefix + 'fit_ids'], scores[prefix + 'val_ids']
        require(set(ff) == set(FOLDS) - {f} and set(vf) == {f}, 'fold isolation failed')
        require(not set(fit_ids) & set(val_ids), 'row overlap')
        Xf, Xv = np.array([by_id[i] for i in fit_ids]), np.array([by_id[i] for i in val_ids])
        models = json.loads((root / f'models-fold-{f}.json').read_text())
        inner_models = json.loads((root / f'inner-models-fold-{f}.json').read_text())
        cv, probabilities = [], []
        for i, name in enumerate(names):
            if persistence:
                feature_names = FEATURE_NAMES[i]
                train, val = Xf[:, :len(feature_names)], Xv[:, :len(feature_names)]
                predict = lambda x, state: score(x, state, feature_names)
            else:
                train, val = Xf[:, i, :2], Xv[:, i, :2]
                predict = lambda x, state: score_from_state(x, state)[1]
            p = predict(val, models[name])
            inner = np.empty(len(yf))
            require(set(inner_models[name]) == {str(v) for v in np.unique(ff)}, 'inner fold models changed')
            for heldout in np.unique(ff):
                valid = ff == heldout
                state = inner_models[name][str(int(heldout))]
                require(state['scale_n_samples_seen'] == int(np.sum(~valid & np.isin(yf, (2, 3)))), 'inner training size changed')
                inner[valid] = predict(train[valid], state)
            for actual, saved in ((p, scores[prefix + 'val_probability'][i]),
                                  (inner, scores[prefix + 'inner_probability'][i])):
                error = float(np.max(np.abs(actual - saved)))
                max_error = max(max_error, error)
                require(error < 1e-12 and np.array_equal(actual >= .5, saved >= .5), 'score/action drift')
            cv.append(accounting(yf, inner)); probabilities.append(p)
            require(cv[-1] == fold_report['fit_cv'][name], 'FIT counts changed')
            require(accounting(yv, p) == fold_report['val_fixed'][name], 'VAL counts changed')
        chosen = min(range(len(names)), key=lambda i: (-cv[i]['global_net'], cv[i]['regressions'], cv[i]['applied'], i))
        chosen = chosen if cv[chosen]['global_net'] > 0 else None
        require(fold_report['selected'] == (None if chosen is None else names[chosen]), 'FIT choice changed')
        p = np.zeros(len(yv)) if chosen is None else probabilities[chosen]
        totals.append(accounting(yv, p))
        require(totals[-1] == fold_report['val_selected'], 'selected count changed')
    for k, v in report['total_selected'].items():
        require(sum(r[k] for r in totals) == v, 'selected total changed')
    for name in names:
        for k, v in report['total_fixed'][name].items():
            require(sum(r['val_fixed'][name][k] for r in report['folds']) == v, 'fixed total changed')
    return {'status': 'verified', 'max_probability_error': max_error,
            'final_models': 8, 'inner_models': 24,
            'selected': [r['selected'] for r in report['folds']]}


def verify(reference, temporal, diagnosis, persistence, cases):
    extraction = json.loads((temporal / 'extraction-report.json').read_text())
    require(extraction['source_sha256']['script'] == sha256_file(Path(__file__).with_name('audit_v273_temporal_stability.py')), 'extraction script changed')
    require(extraction['source_sha256']['protocol'] == sha256_file(JOINT_PROTOCOL), 'original protocol changed')
    pr = json.loads((persistence / 'report.json').read_text())
    require(pr['source_sha256']['script'] == sha256_file(Path(__file__).with_name('audit_v273_temporal_persistence_guard.py')), 'persistence script changed')
    require(pr['source_sha256']['protocol'] == sha256_file(PERSISTENCE_PROTOCOL), 'persistence protocol changed')
    require(extraction['source_sha256']['reference_spectra'] == sha256_file(reference / 'spectra.npz'), 'spectrum source changed')
    require(extraction['source_sha256']['reference_features'] == sha256_file(reference / 'features.npz'), 'feature source changed')
    spectra = arrays(reference / 'spectra.npz'); second = arrays(temporal / 'spectra-second.npz')
    feature = arrays(temporal / 'features.npz'); traces = arrays(temporal / 'temporal.npz')
    control = arrays(reference / 'features.npz')
    require(tuple(feature['variants']) == JOINT_NAMES, 'variant schema changed')
    for d in (second, feature, traces, control):
        np.testing.assert_array_equal(d['row_id'], spectra['row_id'])
    require(len(feature['row_id']) == 1666, 'population changed')
    np.testing.assert_array_equal(second['freq'], spectra['freq'])
    np.testing.assert_array_equal(feature['features'][:, 0], control['features'][:, 0])
    tables = extraction_tables(spectra['freq'])
    max_residual_error = 0.
    for i, row in enumerate(spectra['row_id']):
        xs = np.array([spectra['x'][i], second['x'][i]])
        residuals, pairs, triplets, pool = extract_windows(xs, tables)
        np.testing.assert_array_equal(pairs, traces['pair_f0'][i])
        np.testing.assert_array_equal(triplets, traces['triplet_f0'][i])
        np.testing.assert_array_equal(pool, traces['pool_ids'][i])
        np.testing.assert_array_equal(feature['pair_f0'][i], pairs[[0, 2]])
        np.testing.assert_array_equal(feature['triplet_f0'][i], triplets[[0, 2]])
        max_residual_error = max(max_residual_error, float(np.max(np.abs(residuals[[0, 2]] - feature['features'][i, :, :2]))))
        np.testing.assert_array_equal(residuals[1], traces['second_residuals'][i])
    require(max_residual_error < 1e-10, 'extraction replay drift')
    pv = arrays(persistence / 'features.npz')
    np.testing.assert_array_equal(pv['row_id'], feature['row_id'])
    np.testing.assert_array_equal(pv['features'], persistence_features(feature['features'][:, 0], traces['triplet_f0']))
    joint_models = verify_inner_and_persistence(temporal)
    persistence_models = verify_inner_and_persistence(persistence, True)
    require(feature_diagnostic(persistence) == json.loads((persistence / 'feature-diagnostic.json').read_text()), 'feature diagnostic changed')
    # Independently reuse the historical two-feature verifier for the joint arm.
    verify_joint_models(temporal, JOINT_NAMES)
    original_cases = [json.loads(s) for s in cases.read_text().splitlines()]
    require(len(original_cases) == 488 and all(r['fold'] in FOLDS for r in original_cases), 'case cohort changed')
    source = json.loads((diagnosis / 'report.json').read_text())
    require(source['source_sha256']['cases'] == sha256_file(cases), 'annotation source changed')
    events = json.loads((diagnosis / 'second-events.json').read_text())
    contexts = {r['row_id']: second_context(r['owned_notes'], events[str(r['row_id'])], r['start_sample']) for r in original_cases}
    by_id = {int(row): {key: traces[key][i] for key in ('pair_f0', 'triplet_f0')} for i, row in enumerate(traces['row_id'])}
    rows = diagnose(original_cases, by_id, contexts)
    saved = [json.loads(s) for s in (diagnosis / 'cases.jsonl').read_text().splitlines()]
    require(rows == saved, 'diagnostic traces changed')
    for group in (*GROUPS, 'K3', 'K2', 'all'):
        subset = [r for r in rows if (True if group == 'all' else
                  r['true_K'] == int(group[1]) if len(group) == 2 else r['group'] == group)]
        require(aggregate(subset) == source['summary'][group], 'diagnostic summary changed')
    return {'status': 'verified', 'outer_fold_3_used': False, 'audio_loaded': False, 'model_refitted': False,
            'unique_rows': 1666, 'val_rows': 845, 'diagnostic_cases': 488,
            'max_residual_replay_error': max_residual_error,
            'joint_models': joint_models, 'persistence_models': persistence_models,
            'source_sha256': {'joint_report': sha256_file(temporal / 'report.json'),
                              'persistence_report': sha256_file(persistence / 'report.json'),
                              'diagnostic_report': sha256_file(diagnosis / 'report.json'),
                              'script': sha256_file(__file__)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'temporal', 'diagnosis', 'persistence', 'cases', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    result = verify(a.reference, a.temporal, a.diagnosis, a.persistence, a.cases)
    write_json(a.output, result)
    print(json.dumps(result, sort_keys=True))
