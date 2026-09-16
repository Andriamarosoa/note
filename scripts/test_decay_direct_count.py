"""Bounded direct-K acoustic ablation; no reference prediction enters a model.

Reuse the frozen fold-1 acoustic summaries so that the new question is the
prediction formulation, not a newly selected feature extractor. This is
previously observed development data, not independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

# These functions only score completed predictions. The old corrector's
# target_actions, design, fit_predict and decode are deliberately not used.
from scripts.test_v273_decay_internal import GROUPS, bootstrap, decide, metrics
from scripts.train_boundaries import group_stem

SOURCE = {
    'run_id': 35065190353, 'run_attempt': 1,
    'head_sha': '6823a421017671d50a5498fd55f9424533e8a336',
    'artifact_name': 'v273-decay-internal-diagnostic',
    'artifact_id': 10434200550,
    'digest': 'sha256:444bfb51d140c69c9dad3937f130a6ca0916dfe0ae0cde2ef478e64b3244e8ee',
    'files': {
        'features.npz': '43812145fb5fe9757bd3917b030604d3a78d9caca291fe97a24b80f853984b6a',
        'predictions.npz': 'e66a762207c57a16a7f5e9f7a06f2eb89b9099cffcca680da6f26b68d242e3d3',
    },
}
CONTRACT = {
    'experiment': 'direct_k_decay_ablation_v1',
    'input': '480_ordinary_acoustic_summaries_plus_160_decay_or_zero_columns',
    'reference_predictions_as_inputs': False,
    'specialist_probabilities_as_inputs': False,
    'target': 'true_K_0_through_6', 'post_prediction_corrector': False,
    'model': 'multinomial_logistic_regression',
    'C': 1.0, 'solver': 'lbfgs', 'max_iter': 10000, 'tol': 1e-6,
    'sample_weighting': 'none', 'seed': 27341,
    'scaling': 'fit_compositions_only_mean_std_floor_1e-6',
    'decoding': 'argmax_K_lowest_K_on_exact_ties',
    'groups': GROUPS, 'parent_fold': 1,
    'validation': 'leave_one_composition_out_four_previously_observed_groups',
    'feature_extractor_changed': False, 'lookahead_ms_at_most': 40,
    'hyperparameter_search': False, 'new_outer_evaluation': False,
    'independent_validation_claim': False,
    'gate': 'same_frozen_2pp_global_false_poly_bootstrap_and_3_of_4_gates',
    'official_reference': 'V27.3', 'automatic_promotion': False,
    'automatic_next_training': False,
}
ARMS = ('control', 'decay')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def emit(event, **details):
    print(json.dumps({'event': event, **details}, sort_keys=True), flush=True)


def load_source(directory):
    paths = {}
    for name, digest in SOURCE['files'].items():
        matches = list(Path(directory).rglob(name))
        require(len(matches) == 1 and sha(matches[0]) == digest, 'changed source: ' + name)
        paths[name] = matches[0]
    with np.load(paths['features.npz'], allow_pickle=False) as f:
        ordinary = f['ordinary'].copy()
        novelty = f['decay'].copy()
        index, members = f['global_index'].copy(), f['member'].copy()
    with np.load(paths['predictions.npz'], allow_pickle=False) as p:
        require(np.array_equal(index, p['global_index']) and np.array_equal(members, p['member']),
                'acoustic/label identity mismatch')
        labels, groups = p['k'].copy(), p['group'].copy()
        # Separate evaluation-only output. Never supplied to fit_pair.
        reference = p['pred_reference'].copy()
    require(ordinary.shape == (14001, 480) and novelty.shape == (14001, 160), 'changed feature shape')
    require(np.isfinite(ordinary).all() and np.isfinite(novelty).all(), 'nonfinite acoustic features')
    require(np.array_equal(groups, [group_stem(str(m)) for m in members]), 'changed composition grouping')
    require(sorted(set(groups)) == GROUPS and len(set(members)) == 40, 'changed source coverage')
    require(len(set(index)) == 14001 and int((labels >= 2).sum()) == 1776, 'changed row coverage')
    stats = metrics(labels, reference)
    require((stats['correct'], stats['poly_correct']) == (11251, 796), 'changed frozen reference')
    return ordinary, novelty, labels, groups, index, reference


def predict_saved(state, x):
    classes = state['classes']
    probability = np.zeros((len(x), 7), dtype=np.float64)
    if len(classes) == 1:
        probability[:, classes[0]] = 1
    else:
        logits = ((x - state['mean']) / state['scale']) @ state['coef'].T + state['intercept']
        if len(classes) == 2:
            positive = 1 / (1 + np.exp(-np.clip(logits[:, 0], -700, 700)))
            probability[:, classes] = np.column_stack([1 - positive, positive])
        else:
            exponential = np.exp(logits - logits.max(axis=1, keepdims=True))
            probability[:, classes] = exponential / exponential.sum(axis=1, keepdims=True)
    return probability


def fit_count(x_fit, k_fit, x_held):
    """Learn K itself. There is no base count, residual action or adjustment."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    x_fit, x_held = np.asarray(x_fit, dtype=np.float64), np.asarray(x_held, dtype=np.float64)
    k_fit = np.asarray(k_fit)
    require(x_fit.ndim == x_held.ndim == 2 and x_fit.shape[1] == x_held.shape[1], 'design shape')
    require(k_fit.shape == (len(x_fit),) and np.issubdtype(k_fit.dtype, np.integer)
            and ((0 <= k_fit) & (k_fit <= 6)).all(), 'invalid true K target')
    require(np.isfinite(x_fit).all() and np.isfinite(x_held).all(), 'nonfinite design')
    mean, scale = x_fit.mean(axis=0), x_fit.std(axis=0)
    scale[scale < 1e-6] = 1
    classes = np.unique(k_fit)
    state = {'mean': mean, 'scale': scale, 'classes': classes}
    if len(classes) == 1:
        state.update(coef=np.zeros((1, x_fit.shape[1])), intercept=np.zeros(1), iterations=np.zeros(1, dtype=int))
        probability = np.zeros((len(x_held), 7))
        probability[:, classes[0]] = 1
    else:
        classifier = LogisticRegression(C=CONTRACT['C'], solver=CONTRACT['solver'],
            max_iter=CONTRACT['max_iter'], tol=CONTRACT['tol'], random_state=CONTRACT['seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            classifier.fit((x_fit - mean) / scale, k_fit)
        probability = np.zeros((len(x_held), 7))
        probability[:, classifier.classes_] = classifier.predict_proba((x_held - mean) / scale)
        state.update(classes=classifier.classes_, coef=classifier.coef_,
                     intercept=classifier.intercept_, iterations=classifier.n_iter_)
    require(np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1), 'invalid count probabilities')
    require(np.allclose(predict_saved(state, x_held), probability, atol=1e-12, rtol=1e-10), 'coefficient replay differs')
    return probability, state


def fit_pair(ordinary, novelty, labels, groups, output):
    """Only acoustic features and true K enter learning; no reference argument."""
    require(sorted(set(groups)) == GROUPS, 'expected four frozen compositions')
    require(len(ordinary) == len(novelty) == len(labels) == len(groups), 'unaligned rows')
    output.mkdir()
    probability = {arm: np.full((len(labels), 7), np.nan) for arm in ARMS}
    audit = []
    coverage = np.zeros(len(labels), dtype=int)
    for number, group in enumerate(GROUPS):
        fit, held = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
        require(len(fit) and len(held) and set(groups[fit]).isdisjoint(set(groups[held])), 'composition leakage')
        coverage[held] += 1
        for arm in ARMS:
            extra = novelty if arm == 'decay' else np.zeros_like(novelty)
            x = np.concatenate([ordinary, extra], axis=1).astype(np.float64)
            emit('direct_count_fit_started', arm=arm, held_composition=group, fit_rows=len(fit))
            p, state = fit_count(x[fit], labels[fit], x[held])
            probability[arm][held] = p
            path = output / f'fold-{number}-{arm}.npz'
            np.savez_compressed(path, **state, fit_rows=fit, held_rows=held)
            with np.load(path, allow_pickle=False) as saved:
                require(np.allclose(predict_saved(saved, x[held]), p, atol=1e-12, rtol=1e-10),
                        'persisted model replay differs')
            audit.append({'held_composition': group, 'arm': arm, 'fit_rows': len(fit),
                'held_rows': len(held), 'classes': state['classes'].tolist(),
                'iterations': state['iterations'].tolist(), 'model_sha256': sha(path)})
            emit('direct_count_fit_complete', arm=arm, held_composition=group, iterations=state['iterations'].tolist())
    require((coverage == 1).all(), 'held rows need exactly one prediction per arm')
    return probability, audit


def run(source_dir, output):
    import scipy
    import sklearn

    if output.exists():
        raise FileExistsError(output)
    ordinary, novelty, labels, groups, index, reference = load_source(source_dir)
    output.mkdir(parents=True)
    emit('direct_count_source_verified', rows=len(labels), poly_rows=int((labels >= 2).sum()),
         reference_used_as_input=False, post_prediction_corrector=False)
    probability, audit = fit_pair(ordinary, novelty, labels, groups, output / 'models')
    predictions = {arm: p.argmax(axis=1).astype(np.int16) for arm, p in probability.items()}
    predictions['reference'] = reference
    stats = {arm: metrics(labels, predictions[arm]) for arm in ('reference', *ARMS)}
    by_group = {g: {arm: metrics(labels[groups == g], p[groups == g])
                   for arm, p in predictions.items()} for g in GROUPS}
    intervals = {arm: bootstrap(groups, labels, predictions[arm], predictions['decay'])
                 for arm in ('reference', 'control')}
    decision = decide(stats, intervals, by_group)
    keys = ('rows', 'poly_rows', 'correct', 'poly_correct', 'exact_k_pct', 'poly_exact_k_pct', 'false_poly')
    public = {'status': 'complete', 'experiment': CONTRACT['experiment'],
        'metrics': {arm: {k: m[k] for k in keys} for arm, m in stats.items()},
        'decay_minus_control_poly_pp': stats['decay']['poly_exact_k_pct'] - stats['control']['poly_exact_k_pct'],
        'positive_compositions_vs_control': decision['positive_compositions_vs_control'],
        'decision': decision['decision'], 'passed': decision['passed'],
        'official_reference': 'V27.3', 'automatic_promotion': False,
        'reference_predictions_as_inputs': False, 'post_prediction_corrector': False,
        'independent_validation_claim': False, 'fitted_models_replayed': len(audit)}
    # Detailed replay files remain local to the job; only aggregate summaries
    # are uploaded by the workflow. No individual rows or coefficients publish.
    np.savez_compressed(output / 'predictions.npz', global_index=index, k=labels, group=groups,
        **{'pred_' + arm: p for arm, p in predictions.items()},
        **{'probability_' + arm: p for arm, p in probability.items()})
    with np.load(output / 'predictions.npz', allow_pickle=False) as saved:
        require(np.array_equal(saved['global_index'], index), 'saved row identity changed')
        for arm in ('reference', *ARMS):
            correct = saved['pred_' + arm] == saved['k']
            require(int(correct.sum()) == stats[arm]['correct'] and
                    int((correct & (saved['k'] >= 2)).sum()) == stats[arm]['poly_correct'],
                    'saved counts do not reproduce summary')
            if arm in ARMS:
                require(np.array_equal(saved['probability_' + arm].argmax(axis=1), saved['pred_' + arm]),
                        'a post-prediction adjustment was applied')
    save_json(output / 'audit.json', {'contract': CONTRACT, 'source': SOURCE, 'folds': audit,
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'source_head_sha': os.environ.get('GITHUB_SHA'), 'by_composition': by_group,
        'descriptive_bootstrap': intervals, 'decision': decision})
    save_json(output / 'summary.json', public)
    lines = ['# Comptage direct avec/sans décroissance — diagnostic interne', '',
        'Les deux modèles prédisent K directement à partir des caractéristiques audio.',
        'Aucune prédiction V27.3 en entrée et aucun correcteur après la sortie.',
        '14 001 exemples, dont 1 776 polyphoniques ; quatre compositions déjà étudiées.', '',
        '| Variante | Exact-K polyphonique | Exact-K global |', '|---|---:|---:|']
    for arm, title in [('reference', 'V27.3 gelée (comparaison seule)'), ('control', 'Comptage direct sans décroissance'),
                       ('decay', 'Comptage direct avec décroissance')]:
        m = stats[arm]
        lines.append(f"| {title} | {m['poly_exact_k_pct']:.4f} % | {m['exact_k_pct']:.4f} % |")
    lines += ['', f"Gain polyphonique de la décroissance : {public['decay_minus_control_poly_pp']:+.4f} points.",
        f"Compositions avec un gain : {public['positive_compositions_vs_control']} / 4.",
        f"Décision : **{public['decision']}**. V27.3 reste la référence officielle.", '',
        'Le résultat concerne ce modèle direct léger et les caractéristiques existantes.',
        'La faiblesse des fenêtres longues dans les graves demeure. Aucune validation indépendante,',
        'promotion, utilisation live ou expérience suivante n’est déclenchée.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    emit('direct_count_complete', **public)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_dir, args.output_dir)
