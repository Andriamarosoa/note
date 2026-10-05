"""One additional audio-only selected-triplet persistence feature."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.audit_v273_harmonic_decay_guard import accounting
from scripts.audit_v273_internal_residual_acoustics import match_frequencies
from scripts.v273_residual_audit import FOLDS, FEATURES, fit_classifier, model_state, require, sha256_file, write_json

NAMES = ('first_window', 'with_persistence')
FEATURE_NAMES = (tuple(FEATURES), (*FEATURES, 'selected_triplet_persistence_fraction'))
PROTOCOL = Path('analysis/v273-temporal-persistence-feature-protocol.md')


def train_predict(Xf, yf, Xv, names):
    use = np.isin(yf, (2, 3))
    require(len(np.unique(yf[use])) == 2, 'one-class FIT')
    model = fit_classifier(Xf[use], (yf[use] == 2).astype(int))
    state = model_state(model)
    state['features'] = list(names)
    return model.predict_proba(Xv)[:, 1], state


def score(X, state, names):
    require(state['features'] == list(names) and state['classes'] == [0, 1], 'model schema changed')
    require(state['threshold'] == .5 and X.shape[1] == len(names), 'feature/threshold mismatch')
    value = ((X - state['scale_mean']) / state['scale_scale']) @ np.asarray(state['coef']) + state['intercept']
    return expit(value)


def persistence_features(first, triplets):
    persistence = np.array([match_frequencies(t[0], t[1])[0] / 3 for t in triplets])
    return np.column_stack([first[:, :2], persistence])


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    with np.load(a.temporal / 'features.npz', allow_pickle=False) as z:
        ids, first = z['row_id'], z['features'][:, 0]
    with np.load(a.temporal / 'temporal.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['row_id'])
        X = persistence_features(first, z['triplet_f0'])
    with np.load(a.temporal / 'scores.npz', allow_pickle=False) as z:
        source_scores = dict(z)
    feature_by_id = dict(zip(ids, X))
    reports, saved_scores = [], {}
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'features.npz', row_id=ids, features=X,
                        names=np.array(FEATURE_NAMES[1]))
    for f in FOLDS:
        prefix = f'fold_{f}_'
        yf, yv = source_scores[prefix + 'fit_y'], source_scores[prefix + 'val_y']
        ff, vf = source_scores[prefix + 'fit_fold'], source_scores[prefix + 'val_fold']
        require(set(ff) <= set(FOLDS) - {f} and set(vf) == {f}, 'fold isolation violated')
        fit_ids, val_ids = source_scores[prefix + 'fit_ids'], source_scores[prefix + 'val_ids']
        require(not set(fit_ids) & set(val_ids), 'row overlap')
        Xf, Xv = np.array([feature_by_id[i] for i in fit_ids]), np.array([feature_by_id[i] for i in val_ids])
        inner, final, models, inner_models, cv = [], [], {}, {}, []
        for name, features in zip(NAMES, FEATURE_NAMES):
            dim = len(features); p = np.zeros(len(yf)); inner_models[name] = {}
            for heldout in np.unique(ff):
                valid = ff == heldout
                p[valid], state = train_predict(Xf[~valid, :dim], yf[~valid], Xf[valid, :dim], features)
                inner_models[name][str(int(heldout))] = state
            pv, models[name] = train_predict(Xf[:, :dim], yf, Xv[:, :dim], features)
            inner.append(p); final.append(pv); cv.append(accounting(yf, p))
        inner, final = np.array(inner), np.array(final)
        np.testing.assert_array_equal(inner[0], source_scores[prefix + 'inner_probability'][0])
        np.testing.assert_array_equal(final[0], source_scores[prefix + 'val_probability'][0])
        chosen = min(range(len(NAMES)), key=lambda i: (-cv[i]['global_net'], cv[i]['regressions'], cv[i]['applied'], i))
        chosen = chosen if cv[chosen]['global_net'] > 0 else None
        selected = np.zeros(len(yv)) if chosen is None else final[chosen]
        for split in ('fit', 'val'):
            for key in ('ids', 'y', 'fold'):
                saved_scores[prefix + split + '_' + key] = source_scores[prefix + split + '_' + key]
        saved_scores[prefix + 'inner_probability'] = inner
        saved_scores[prefix + 'val_probability'] = final
        write_json(a.output / f'models-fold-{f}.json', models)
        write_json(a.output / f'inner-models-fold-{f}.json', inner_models)
        reports.append({'fold': f, 'selected': None if chosen is None else NAMES[chosen],
                        'fit_cv': dict(zip(NAMES, cv)),
                        'val_fixed': {name: accounting(yv, p) for name, p in zip(NAMES, final)},
                        'val_selected': accounting(yv, selected)})
    np.savez_compressed(a.output / 'scores.npz', **saved_scores)
    total = lambda rows: {k: sum(r[k] for r in rows) for k in
                          ('rows', 'applied', 'corrections', 'regressions', 'other_k_actions', 'global_net')}
    report = {'status': 'completed', 'variants': list(NAMES), 'feature_names': dict(zip(NAMES, FEATURE_NAMES)),
              'folds': reports, 'outer_fold_3_used': False, 'annotation_frequencies_used': False,
              'neural_training': False, 'automatic_promotion': False,
              'total_selected': total([r['val_selected'] for r in reports]),
              'total_fixed': {name: total([r['val_fixed'][name] for r in reports]) for name in NAMES},
              'control_max_probability_error': 0.,
              'source_sha256': {'protocol': sha256_file(PROTOCOL), 'script': sha256_file(__file__),
                                'temporal_features': sha256_file(a.temporal / 'features.npz'),
                                'temporal': sha256_file(a.temporal / 'temporal.npz'),
                                'temporal_scores': sha256_file(a.temporal / 'scores.npz')},
              'limitations': ['Previously inspected internal folds; not untouched validation.',
                              'Selected-frequency persistence is not proof of a physical source.',
                              'Normal audio only; 46.44 ms extra lookahead.']}
    write_json(a.output / 'report.json', report)
    print(json.dumps({'fixed': report['total_fixed'], 'selected': report['total_selected'],
                      'selections': [r['selected'] for r in reports]}, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('temporal', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
