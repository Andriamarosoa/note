"""Check frozen decay-guard scores and FIT selection without audio or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_v273_harmonic_decay_guard import NAMES, accounting
from scripts.v273_residual_audit import FOLDS, require, score_from_state, write_json


def verify(root, names=NAMES):
    report = json.loads((root / 'report.json').read_text())
    require(report['status'] == 'completed' and not report['outer_fold_3_used'], 'invalid report')
    require([r['fold'] for r in report['folds']] == list(FOLDS), 'fold set changed')
    with np.load(root / 'features.npz', allow_pickle=False) as z:
        require(tuple(z['variants']) == tuple(names), 'variants changed')
        features = dict(zip(z['row_id'], z['features']))
    with np.load(root / 'scores.npz', allow_pickle=False) as z:
        scores = dict(z)
    max_error = 0.0
    all_selected = []
    for row in report['folds']:
        f = row['fold']; prefix = f'fold_{f}_'
        models = json.loads((root / f'models-fold-{f}.json').read_text())
        fit_ids, val_ids = scores[prefix + 'fit_ids'], scores[prefix + 'val_ids']
        require(not set(fit_ids) & set(val_ids), 'FIT/VAL row overlap')
        require(set(scores[prefix + 'fit_fold']) <= set(FOLDS) - {f}, 'FIT fold leak')
        require(set(scores[prefix + 'val_fold']) == {f}, 'VAL fold changed')
        Xv = np.array([features[i] for i in val_ids])
        yf, yv = scores[prefix + 'fit_y'], scores[prefix + 'val_y']
        cv = []
        for i, name in enumerate(names):
            _, p = score_from_state(Xv[:, i, :2], models[name])
            saved = scores[prefix + 'val_probability'][i]
            error = float(np.max(np.abs(p - saved)))
            max_error = max(max_error, error)
            require(error < 1e-12, 'probability replay drift')
            require(np.array_equal(p >= .5, saved >= .5), 'action replay drift')
            require(accounting(yv, p) == row['val_fixed'][name], 'VAL counts changed')
            inner = accounting(yf, scores[prefix + 'inner_probability'][i])
            require(inner == row['fit_cv'][name], 'FIT CV counts changed')
            cv.append(inner)
        chosen = min(range(len(names)), key=lambda i: (-cv[i]['global_net'], cv[i]['regressions'], cv[i]['applied'], i))
        chosen = None if cv[chosen]['global_net'] <= 0 else chosen
        require(row['selected'] == (None if chosen is None else names[chosen]), 'selection changed')
        selected_p = np.zeros(len(yv)) if chosen is None else scores[prefix + 'val_probability'][chosen]
        result = accounting(yv, selected_p)
        require(result == row['val_selected'], 'selected counts changed')
        all_selected.append(result)
    for k, value in report['total_selected'].items():
        require(sum(r[k] for r in all_selected) == value, 'selected totals changed')
    for name in names:
        for k, value in report['total_fixed'][name].items():
            require(sum(r['val_fixed'][name][k] for r in report['folds']) == value, 'fixed totals changed')
    return {'status': 'verified', 'folds': list(FOLDS), 'rows': report['total_selected']['rows'],
            'max_abs_probability_error': max_error, 'model_refitted': False,
            'audio_loaded': False, 'outer_fold_3_used': False,
            'selected': {str(r['fold']): r['selected'] for r in report['folds']}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = verify(a.input)
    if a.output:
        write_json(a.output, result)
    print(json.dumps(result, sort_keys=True))
