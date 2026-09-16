"""One bounded, paired acoustic-decay diagnostic around frozen V27.3.

All corrector training/validation is inside canonical fold 1. Its archived
V27.3 base never fitted/calibrated on that fold. Four composition groups are
left out in turn. This reuses observed development data, not a new outer test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from causal_note import decay_novelty as decay
from scripts import train_v280_balanced_internal as frozen
from scripts.train_boundaries import group_stem

SOURCES = [
    {'run_id': 35061192743, 'run_attempt': 1,
     'head_sha': 'a9630c929b40cd4ba8ce3853ad620576211d18f5',
     'producer_job_id': 104681672224,
     'artifacts': {'v280-i-prepared': {'id': 10432567858,
         'digest': 'sha256:2931345bf7570c2d9501407728e1ce3830f2cfc39f2ef2bf10f24627b4c154a4'}}},
    {'run_id': 34297767492, 'run_attempt': 1,
     'head_sha': 'dd43b4f92b234dc7c84377e18389fd34350cd1b4',
     'artifacts': {'v273-selective-transition-fold-1': {'id': 10086186883,
         'digest': 'sha256:7fe5060ca6b3e901c41855f19209dadd52710cf2ae9d4acc50278a69ad600771'}}},
]
REFERENCE_FILES = {
    'predictions-fold-1.npz': 'e2d524d0c329151cd699a1106a30e7ba2c5d4a56f10f76e7694a3f2f0003ed60',
    'report-fold-1.json': '95d94aaf0eb8ff8c871d977c75366251cb195ea69b066629c8bbaeda705eca7b',
}
GROUPS = ['BN3-154-E', 'Funk2-119-G', 'Funk3-112-C#', 'SS3-98-C']
CONTRACT = {
    'experiment': 'v273_decay_internal_diagnostic_v1', 'schema_version': 1,
    'official_reference': 'V27.3', 'reference_refitted': False,
    'canonical_parent_fold': 1, 'rows': 14001, 'poly_rows': 1776,
    'groups': GROUPS, 'validation': 'leave_one_composition_out_inside_parent_fold',
    'base_training_folds': [0, 2, 3, 4],
    'base_calibration_never_used_parent_fold_labels': True,
    'current_corrector_features_and_labels_from_other_canonical_folds': False,
    'arms': ['control', 'decay'],
    'shared_features': 'V273_count_onehot+V272_frozen_probabilities+6_CQT_summaries',
    'added_features': 'mean_and_max_positive_excess_over_precluster_exponential_prediction',
    'control_added_columns': 'zeros_same_shape',
    'actions': [0, -1, 1],
    'fit_target': 'exact_adjacent_correction_else_keep',
    'model': 'multinomial_logistic_regression', 'C': 1.0, 'solver': 'lbfgs',
    'max_iter': 1000, 'tol': 1e-6, 'seed': 27341,
    'scaling': 'fit_rows_only_per_column_mean_std', 'sample_weighting': 'none',
    'decision': 'largest_action_probability_valid_for_base_count_keep_on_ties',
    'threshold_search': False, 'hyperparameter_search': False,
    'frozen_decay': {'pre_frames': decay.PRE_FRAMES, 'hop_seconds': decay.HOP_SECONDS,
        'log_gain': decay.LOG_GAIN, 'amplitude_floor': decay.AMPLITUDE_FLOOR,
        'relative_floor': decay.RELATIVE_FLOOR, 'min_frames': decay.MIN_VALID_FRAMES,
        'min_decay_per_second': decay.MIN_DECAY_PER_SECOND,
        'max_decay_per_second': decay.MAX_DECAY_PER_SECOND,
        'max_log_rmse': decay.MAX_LOG_RMSE},
    'lookahead': 'existing_cache_at_most_40ms_no_added_future',
    'bootstrap': {'unit': 'composition', 'replicates': 10000, 'seed': 27341,
                  'interpretation': 'descriptive_four_groups_with_overlapping_fit_sets'},
    'gate_vs_both_control_and_reference': {'poly_gain_pp_at_least': 2.0,
        'global_correct_not_down': True, 'false_poly_not_up': True,
        'bootstrap_lower95_strictly_positive': True},
    'gate_positive_compositions_vs_control_at_least': 3,
    'on_pass': 'worth_separate_replication_not_promotion',
    'on_fail': 'no_material_gain_in_this_diagnostic_keep_v273',
    'independent_validation_claim': False, 'automatic_promotion': False,
    'new_outer_evaluation': False, 'automatic_next_training': False,
}
DELTAS = np.array([0, -1, 1], dtype=np.int16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_array(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def emit(event, **details):
    print(json.dumps({'event': event, **details}, sort_keys=True), flush=True)


def reference(directory, arrays):
    paths = {}
    for name, expected in REFERENCE_FILES.items():
        matches = list(Path(directory).rglob(name))
        require(len(matches) == 1 and sha(matches[0]) == expected, 'changed V27.3 source: ' + name)
        paths[name] = matches[0]
    report = json.loads(paths['report-fold-1.json'].read_text())
    require(report['fold'] == 1 and report['protocol']['outer_labels_used_for_threshold_selection'] is False,
            'base reference was calibrated on its held fold')
    with np.load(paths['predictions-fold-1.npz'], allow_pickle=False) as data:
        ref = {key: data[key] for key in data.files}
    rows = np.flatnonzero(arrays['row_fold'] == 1)
    for name, expected in {'global_index': arrays['global_index'][rows],
                           'member': arrays['member'][rows], 'k': arrays['target_cardinality'][rows],
                           'outer_fold': np.ones(len(rows), dtype=np.int16)}.items():
        require(np.array_equal(ref[name], expected), 'reference/crop identity mismatch: ' + name)
    groups = np.array([group_stem(str(m)) for m in ref['member']])
    require(sorted(set(groups)) == GROUPS, 'composition membership changed')
    base = ref['pred273_selective_transition']
    stats = metrics(ref['k'], base)
    require((stats['rows'], stats['poly_rows'], stats['correct'], stats['poly_correct'], stats['false_poly'])
            == (14001, 1776, 11251, 796, 851), 'frozen V27.3 fold-1 counts changed')
    return rows, ref, groups


def feature_table(features, rows):
    ordinary, added, reliability, only = [], [], [], []
    # Only these explicitly selected rows are indexed from the all-fold mmap.
    for offset in range(0, len(rows), 128):
        part = rows[offset:offset+128]
        old, new, audit = decay.summarize_crop(features[part])
        ordinary.append(old); added.append(new)
        reliability.append(audit['reliable_bin_fraction']); only.append(audit['decay_only_max'])
    return {'ordinary': np.concatenate(ordinary), 'decay': np.concatenate(added),
            'reliable_bin_fraction': np.concatenate(reliability), 'decay_only_max': np.concatenate(only)}


def target_actions(truth, base):
    difference = np.asarray(truth) - np.asarray(base)
    return np.where(difference == -1, 1, np.where(difference == 1, 2, 0)).astype(np.int32)


def design(table, base, specialist, arm):
    require(arm in ('control', 'decay'), 'unknown arm')
    extra = table['decay'] if arm == 'decay' else np.zeros_like(table['decay'])
    return np.concatenate([np.eye(7)[base], specialist, table['ordinary'], extra], axis=1)


def fit_predict(x_fit, y_fit, x_held):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    mean = np.asarray(x_fit, dtype=np.float64).mean(axis=0)
    scale = np.asarray(x_fit, dtype=np.float64).std(axis=0)
    scale[scale < 1e-6] = 1.0
    fit = (x_fit - mean) / scale
    held = (x_held - mean) / scale
    classes = np.unique(y_fit)
    probability = np.zeros((len(x_held), 3), dtype=np.float64)
    if len(classes) == 1:
        probability[:, classes[0]] = 1
        state = {'classes': classes, 'coef': np.zeros((1, fit.shape[1])),
                 'intercept': np.zeros(1), 'iterations': np.zeros(1, dtype=np.int32)}
    else:
        model = LogisticRegression(C=CONTRACT['C'], solver=CONTRACT['solver'],
            max_iter=CONTRACT['max_iter'], tol=CONTRACT['tol'], random_state=CONTRACT['seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(fit, y_fit)
        probability[:, model.classes_] = model.predict_proba(held)
        state = {'classes': model.classes_, 'coef': model.coef_,
                 'intercept': model.intercept_, 'iterations': model.n_iter_}
    require(np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1),
            'invalid action probabilities')
    return probability, {**state, 'mean': mean, 'scale': scale}


def decode(probability, base):
    p = np.array(probability, dtype=np.float64, copy=True)
    require(p.shape == (len(base), 3) and np.isfinite(p).all() and (p >= 0).all()
            and np.allclose(p.sum(axis=1), 1), 'invalid action distribution')
    possible = np.asarray(base)[:, None] + DELTAS[None, :]
    p[(possible < 0) | (possible > 6)] = -1
    return (np.asarray(base) + DELTAS[p.argmax(axis=1)]).astype(np.int16)


def predict_saved(state, features):
    """Replay persisted linear coefficients without pickled sklearn objects."""
    x = (features - state['mean']) / state['scale']
    classes = state['classes']
    probability = np.zeros((len(x), 3))
    if len(classes) == 1:
        probability[:, classes[0]] = 1
    else:
        logits = x @ state['coef'].T + state['intercept']
        if len(classes) == 2:
            p = 1 / (1 + np.exp(-np.clip(logits[:, 0], -700, 700)))
            probability[:, classes] = np.column_stack([1-p, p])
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probability[:, classes] = exp / exp.sum(axis=1, keepdims=True)
    return probability


def metrics(truth, prediction):
    k, p = np.asarray(truth), np.asarray(prediction)
    require(k.shape == p.shape and k.ndim == 1 and len(k) > 0 and
            ((k >= 0) & (k <= 6)).all() and ((p >= 0) & (p <= 6)).all(), 'invalid count vectors')
    correct = k == p
    poly = k >= 2
    return {'rows': len(k), 'correct': int(correct.sum()), 'exact_k_pct': float(correct.mean() * 100),
            'poly_rows': int(poly.sum()), 'poly_correct': int((correct & poly).sum()),
            'poly_exact_k_pct': float(correct[poly].mean() * 100) if poly.any() else None,
            'false_poly': int(((k < 2) & (p >= 2)).sum()),
            'under': int((p < k).sum()), 'over': int((p > k).sum()),
            'confusion_true_by_predicted': np.bincount(k * 7 + p, minlength=49).reshape(7, 7).tolist()}


def comparison(truth, baseline, prediction):
    k = np.asarray(truth)
    b, p = baseline == k, prediction == k
    return {'corrected': int((~b & p).sum()), 'regressed': int((b & ~p).sum()),
            'net_correct': int(p.sum() - b.sum()), 'changed': int((baseline != prediction).sum()),
            'poly_corrected': int((~b & p & (k >= 2)).sum()),
            'poly_regressed': int((b & ~p & (k >= 2)).sum()),
            'k2_as_1_before': int(((k == 2) & (baseline == 1)).sum()),
            'k2_as_1_after': int(((k == 2) & (prediction == 1)).sum())}


def bootstrap(groups, truth, baseline, prediction):
    poly = np.asarray(truth) >= 2
    change = (prediction == truth).astype(int) - (baseline == truth).astype(int)
    units = sorted(set(groups))
    stats = np.array([[int(((groups == g) & poly).sum()),
                       int(change[(groups == g) & poly].sum())] for g in units])
    require((stats[:, 0] > 0).all(), 'every bootstrap composition needs polyphonic rows')
    spec = CONTRACT['bootstrap']
    draws = np.random.default_rng(spec['seed']).integers(0, len(units), (spec['replicates'], len(units)))
    sums = stats[draws].sum(axis=1)
    values = 100 * sums[:, 1] / sums[:, 0]
    lower, upper = np.quantile(values, [.025, .975])
    return {'lower_95_pp': float(lower), 'upper_95_pp': float(upper),
            'groups': units, 'replicates': spec['replicates'], 'unit': 'composition',
            'independent_validation_claim': False}


def decide(stats, intervals, by_group):
    candidate = stats['decay']
    gates = {}
    for arm in ('reference', 'control'):
        base = stats[arm]
        require(candidate['rows'] == base['rows'] and candidate['poly_rows'] == base['poly_rows'],
                'unpaired comparison')
        gates[arm] = {
            'poly_gain_at_least_2pp': (candidate['poly_correct'] - base['poly_correct']) * 100
                                    >= 2 * candidate['poly_rows'],
            'global_correct_not_down': candidate['correct'] >= base['correct'],
            'false_poly_not_up': candidate['false_poly'] <= base['false_poly'],
            'bootstrap_lower95_positive': intervals[arm]['lower_95_pp'] > 0,
        }
    positive = sum(m['decay']['poly_correct'] > m['control']['poly_correct'] for m in by_group.values())
    passed = all(all(g.values()) for g in gates.values()) and positive >= 3
    return {'passed': passed, 'criteria_vs': gates, 'positive_compositions_vs_control': positive,
            'decision': CONTRACT['on_pass'] if passed else CONTRACT['on_fail'],
            'official_reference': 'V27.3', 'automatic_promotion': False, 'automatic_next_training': False}


def run(args):
    import scipy
    import sklearn

    start = time.monotonic()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    features, arrays, _ = frozen.load_data(args.prepared_dir)
    rows, ref, groups = reference(args.reference_dir, arrays)
    base = ref['pred273_selective_transition']
    truth = ref['k']
    emit('decay_data_verified', rows=len(rows), compositions=len(set(groups)))
    table = feature_table(features, rows)
    np.savez_compressed(out / 'features.npz', global_index=rows, member=ref['member'], **table)
    audit = {'contract': CONTRACT, 'sources': SOURCES, 'reference_files': REFERENCE_FILES,
        'prepared_manifest_sha256': frozen.SOURCE['manifest_sha256'],
        'source_head_sha': os.environ.get('GITHUB_SHA'), 'source_run_id': os.environ.get('GITHUB_RUN_ID'),
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'indexed_feature_rows_sha256': digest_array(rows),
        'indexed_feature_parent_folds': [1], 'rows_with_reliable_decay': int((table['reliable_bin_fraction'] > 0).sum()),
        'rows_with_decay_only_excess_over_0_01': int((table['decay_only_max'] > .01).sum()),
        'folds': {}}
    predictions = {'reference': base, **{arm: np.full(len(rows), -1, dtype=np.int16)
                                       for arm in CONTRACT['arms']}}
    action_p = {arm: np.full((len(rows), 3), np.nan) for arm in CONTRACT['arms']}
    labels = target_actions(truth, base)
    by_group = {}
    (out / 'models').mkdir()
    for number, held_group in enumerate(GROUPS):
        fit, held = np.flatnonzero(groups != held_group), np.flatnonzero(groups == held_group)
        require(set(groups[fit]).isdisjoint(set(groups[held])) and len(fit) and len(held), 'composition leakage')
        audit['folds'][held_group] = {'fit_groups': sorted(set(groups[fit])), 'held_group': held_group,
            'fit_rows': len(fit), 'held_rows': len(held), 'fit_sha256': digest_array(rows[fit]),
            'held_sha256': digest_array(rows[held]), 'fit_action_counts': np.bincount(labels[fit], minlength=3).tolist(),
            'arms': {}}
        for arm in CONTRACT['arms']:
            x = design(table, base, ref['specialist_probability'], arm)
            emit('decay_training_started', arm=arm, held_group=held_group, fit_rows=len(fit), held_rows=len(held))
            probability, state = fit_predict(x[fit], labels[fit], x[held])
            require(np.allclose(predict_saved(state, x[held]), probability, atol=1e-12, rtol=1e-10),
                    'saved classifier does not replay')
            action_p[arm][held] = probability
            predictions[arm][held] = decode(probability, base[held])
            model_path = out / 'models' / f'fold-{number}-{arm}.npz'
            np.savez_compressed(model_path, **state)
            audit['folds'][held_group]['arms'][arm] = {
                'model_sha256': sha(model_path), 'features': x.shape[1],
                'fit_design_sha256': digest_array(x[fit]), 'iterations': state['iterations'].tolist()}
            emit('decay_training_complete', arm=arm, held_group=held_group, iterations=state['iterations'].tolist())
        by_group[held_group] = {arm: metrics(truth[held], prediction[held]) for arm, prediction in predictions.items()}
    stats = {arm: metrics(truth, pred) for arm, pred in predictions.items()}
    intervals = {arm: bootstrap(groups, truth, predictions[arm], predictions['decay'])
                 for arm in ('reference', 'control')}
    decision = decide(stats, intervals, by_group)
    result = {'status': 'complete', 'contract': CONTRACT, 'metrics': stats, 'by_composition': by_group,
        'decay_changes_vs': {arm: comparison(truth, predictions[arm], predictions['decay'])
                             for arm in ('reference', 'control')},
        'descriptive_bootstrap_vs': intervals, 'decision': decision,
        'runtime_seconds': time.monotonic() - start,
        'limitation': 'four previously observed development compositions; overlapping fit sets; fixed lightweight corrector; no independent confirmation'}
    np.savez_compressed(out / 'predictions.npz', global_index=rows, member=ref['member'], group=groups,
        k=truth, **{'pred_' + arm: p for arm, p in predictions.items()},
        **{'action_probability_' + arm: p for arm, p in action_p.items()})
    audit['files'] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*.npz'))}
    save_json(out / 'audit.json', audit)
    save_json(out / 'comparison.json', result)
    save_json(out / 'decision.json', decision)
    lines = ['# V27.3 — diagnostic interne de décroissance', '',
        'Quatre compositions du fold 1, exclues de l’apprentissage et du calibrage de la référence V27.3.',
        'Chaque correcteur apprend sur trois compositions et prédit la quatrième. Données de développement déjà observées.', '',
        '| Variante | Exact-K poly | Exact-K global | Fausses polyphonies |',
        '|---|---:|---:|---:|']
    for arm, m in stats.items():
        lines.append(f"| {arm} | {m['poly_exact_k_pct']:.4f} % | {m['exact_k_pct']:.4f} % | {m['false_poly']} |")
    lines += ['', f"Décision : **{decision['decision']}**. V27.3 reste la référence.", '',
        'Le contrôle reçoit les mêmes informations ordinaires et le même classifieur ; seuls les indices de décroissance sont masqués.',
        'Aucune recherche de seuil, aucune promotion ni nouvelle évaluation externe automatique.', '',
        'Les intervalles bootstrap par composition sont descriptifs : seulement quatre compositions et des ensembles d’apprentissage qui se recouvrent.']
    for arm in ('reference', 'control'):
        delta = stats['decay']['poly_exact_k_pct'] - stats[arm]['poly_exact_k_pct']
        b = intervals[arm]
        lines.append(f"\nGain polyphonique face à {arm} : {delta:+.4f} points ; intervalle 95 % [{b['lower_95_pp']:+.4f}, {b['upper_95_pp']:+.4f}].")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    emit('decay_diagnostic_complete', decision=decision, metrics=stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
