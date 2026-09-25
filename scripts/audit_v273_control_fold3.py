"""Audit existing control decisions on outer fold 3, without fitting or tuning.

The scalar decoder is independent of the production vectorized implementation.
Counterfactuals are diagnostic ablations, never candidates selected on this fold.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.audit_v273_native_paired_results import (
    load_npz, read_json, require, verify_inventory,
)
from scripts import train_v273_native_paired as paired
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_boundaries import group_stem
from scripts.v273_native_protocol import load_config

COMPONENTS = ('v260_uniform', 'v260_weighted', 'v272_uniform')
PAIRS = ((2, 3), (3, 2), (3, 4), (4, 3))


def scalar_decode(anchor, probabilities, calibration):
    """Labels are absent; compare each original float32 value as a Python float."""
    require(set(probabilities) == set(COMPONENTS), 'component inventory mismatch')
    n = len(anchor)
    for name, size in zip(COMPONENTS, (7, 7, 5)):
        p = np.asarray(probabilities[name])
        require(p.shape == (n, size), 'invalid probability shape')
        require(np.isfinite(p).all() and np.all(p >= 0), 'invalid probability')
        require(np.allclose(p.sum(1), 1, atol=1e-5), 'unnormalized probability')
    values = {name: [] for name in ('uniform', 'weighted', 'specialist', 'low',
                                   'base', 'final', 'rescue_active', 'transition_active')}
    for row, original in enumerate(anchor):
        require(0 <= original <= 6, 'invalid anchor')
        u, w, s = (probabilities[name][row].tolist() for name in COMPONENTS)
        up = max(range(7), key=u.__getitem__)
        wp = max(range(7), key=w.__getitem__)
        sp = max(range(5), key=s.__getitem__) + 2
        low = up if original <= 1 else int(original)
        rescue = bool(original <= 1 and low <= 1 and wp >= 2
                      and w[wp] >= calibration['v271']['threshold'])
        base = wp if rescue else low
        active = False
        final = base
        if (base, sp) in PAIRS:
            threshold = calibration['v273']['thresholds'][f'{base}_to_{sp}']
            active = s[sp - 2] - s[base - 2] >= threshold
            if active:
                final = sp
        for key, value in zip(values, (up, wp, sp, low, base, final, rescue, active)):
            values[key].append(value)
    return {key: np.asarray(value, dtype=bool if key.endswith('_active') else np.int32)
            for key, value in values.items()}


def metrics(k, pred):
    k, pred = np.asarray(k), np.asarray(pred)
    poly = k >= 2
    return {'rows': len(k), 'correct': int(np.sum(pred == k)),
            'over': int(np.sum(pred > k)), 'under': int(np.sum(pred < k)),
            'poly_rows': int(poly.sum()), 'poly_correct': int(np.sum(poly & (pred == k))),
            'poly_over': int(np.sum(poly & (pred > k))),
            'poly_under': int(np.sum(poly & (pred < k)))}


def effect(k, before, after, mask=None):
    if mask is None:
        mask = np.ones(len(k), bool)
    changed = mask & (before != after)
    fixed = changed & (before != k) & (after == k)
    broken = changed & (before == k) & (after != k)
    return {'rows': int(mask.sum()), 'changed': int(changed.sum()),
            'fixed': int(fixed.sum()), 'broken': int(broken.sum()),
            'net_correct': int(fixed.sum() - broken.sum()),
            'new_over': int(np.sum(changed & (before <= k) & (after > k))),
            'removed_over': int(np.sum(changed & (before > k) & (after <= k))),
            'new_under': int(np.sum(changed & (before >= k) & (after < k))),
            'removed_under': int(np.sum(changed & (before < k) & (after >= k)))}


def classify_errors(k, stages):
    """Exhaustive disjoint diagnostic categories; these use labels, not runtime."""
    error = (k >= 2) & (stages['final'] != k)
    base, specialist = stages['base'], stages['specialist']
    good_specialist = error & (specialist == k)
    supported = np.isin(base * 7 + specialist, [a * 7 + b for a, b in PAIRS])
    categories = {
        'correct_specialist_blocked_base_0_or_1': good_specialist & (base <= 1),
        'correct_specialist_blocked_unsupported_transition': good_specialist & (base >= 2) & ~supported,
        'correct_specialist_blocked_margin': good_specialist & supported,
        'only_full_range_head_correct': error & ~good_specialist & (
            (stages['uniform'] == k) | (stages['weighted'] == k)),
        'all_three_count_heads_wrong': error & (stages['uniform'] != k)
            & (stages['weighted'] != k) & (specialist != k),
    }
    require(np.array_equal(sum(x.astype(int) for x in categories.values()), error.astype(int)),
            'diagnostic categories overlap or miss errors')
    return categories


def run(args):
    require(args.fold == 3, 'only outer fold 3 is authorized')
    require(not args.output_dir.exists(), 'output already exists')
    cfg = load_config(args.config)
    files_verified = verify_inventory(args.fold_dir) + verify_inventory(args.bundle_dir)
    source = args.fold_dir / 'paired'
    saved = load_npz(source / 'predictions-fold-3.npz')
    selected = load_npz(args.bundle_dir / 'predictions.npz')
    calibration = read_json(source / 'frozen-calibration.json')['control']
    require(calibration == read_json(args.bundle_dir / 'calibration.json'), 'calibration changed')
    anchor_file = args.fold_dir / 'anchor/v104-nested-eval-3.npz'
    anchor = load_npz(anchor_file)
    for key in ('global_index', 'member', 'k'):
        require(np.array_equal(saved[key], selected[key]), 'selected alignment: ' + key)
        require(np.array_equal(saved[key], anchor[key]), 'anchor alignment: ' + key)
    idx, k, a = saved['global_index'], saved['k'], saved['shared_anchor']
    require(np.all(saved['outer_fold'] == 3), 'wrong fold marker')
    require(len(np.unique(idx)) == len(idx), 'duplicate indices')
    require(all(cfg['member_folds'][str(m)] == 3 for m in saved['member']), 'other-fold rows')
    require(np.array_equal(a, anchor['pred104_deploy']), 'shared anchor mismatch')
    require(np.array_equal(saved['control'], selected['selected']), 'selected model changed')
    probabilities = {}
    for name in COMPONENTS:
        p = load_npz(source / 'control/final' / name / 'probability.npz')
        require(np.array_equal(idx, p['global_index']), 'component alignment')
        probabilities[name] = p['probability']
    stages = scalar_decode(a, probabilities, calibration)
    production = paired.decode(a, probabilities, calibration)
    require(np.array_equal(production, stages['final']), 'scalar/production decoder disagreement')
    require(np.array_equal(production, saved['control']), 'archive replay mismatch')
    exported = scalar_decode(a, {name: selected[name] for name in COMPONENTS}, calibration)
    require(np.array_equal(exported['final'], production), 'exported runtime decisions differ')
    final, base, s = stages['final'], stages['base'], stages['specialist']
    poly = k >= 2
    categories = classify_errors(k, stages)
    stage_predictions = {'anchor': a, 'low_fusion': stages['low'],
                         'poly_rescue': base, 'final': final}
    report = {
        'scope': {'outer_fold': 3, 'training': False, 'threshold_selection': False,
                  'model_modified': False, 'new_corrector': False,
                  'other_outer_folds_evaluated': False, 'development_diagnostic': True},
        'verification': {'inventory_files_verified': files_verified,
                         'independent_scalar_replay_equal': True,
                         'production_replay_equal': True, 'exported_runtime_replay_equal': True,
                         'aligned_rows': len(k), 'config_sha256': digest(args.config),
                         'calibration_sha256': digest(args.bundle_dir / 'calibration.json'),
                         'source_prediction_sha256': digest(source / 'predictions-fold-3.npz')},
        'target_definition': 'min(6, number of annotated note onsets assigned to a 40 ms candidate group within 20 ms of a candidate); not simultaneous sustained pitches',
        'stages': {name: metrics(k, p) for name, p in stage_predictions.items()},
        'confusion_true_by_predicted': np.bincount(k * 7 + final, minlength=49).reshape(7, 7).tolist(),
        'by_true_k': {str(i): metrics(k[k == i], final[k == i]) for i in range(7)},
        'errors_disjoint': {name: metrics(k[mask], final[mask]) for name, mask in categories.items()},
        'stage_effects': {}, 'transition_effects': {},
        'calibration_rescue_inner_effect': calibration['v271']['transition_from_low_k_fusion'],
        'head_diagnostics': {name: metrics(k, stages[name]) for name in ('uniform', 'weighted', 'specialist')},
    }
    chain = [('low_fusion', a, stages['low']), ('poly_rescue', stages['low'], base),
             ('selective_transitions', base, final)]
    for name, before, after in chain:
        report['stage_effects'][name] = {'global': effect(k, before, after),
                                      'poly': effect(k, before, after, poly)}
    for x, y in PAIRS:
        name = f'{x}_to_{y}'
        eligible = (base == x) & (s == y)
        active = eligible & stages['transition_active']
        report['transition_effects'][name] = {
            'inner_selection': calibration['v273']['directions'][name],
            'outer_eligible': int(eligible.sum()), 'outer_activated': int(active.sum()),
            'outer_effect': effect(k, base, final, active),
        }
    # Each policy is defined without labels; none is selected or promoted.
    ablations = {
        'no_selective_transitions': base,
        'no_poly_rescue': paired.v273.apply_transition_corrector(
            stages['low'], probabilities['v272_uniform'], calibration['v273']['thresholds'])[0],
        'conditional_head_on_all_base_ge2': np.where(base >= 2, s, base),
        'conditional_head_on_every_row_INVALID_FOR_K0_K1': s,
    }
    for x, y in PAIRS:
        mask = (base == x) & (s == y) & stages['transition_active']
        ablations[f'disable_{x}_to_{y}'] = np.where(mask, base, final)
    report['diagnostic_ablations_not_model_selection'] = {
        name: {'metrics': metrics(k, pred), 'effect_vs_control': effect(k, final, pred)}
        for name, pred in ablations.items()}
    all_under = poly & (stages['uniform'] < k) & (stages['weighted'] < k) & (s < k)
    report['remaining_network_evidence'] = {
        'final_under_and_all_three_heads_under': int(np.sum(all_under & (final < k))),
        'poly_final_0_or_1': int(np.sum(poly & (final <= 1))),
        'k5_k6': metrics(k[k >= 5], final[k >= 5]),
        'note': 'Wrong head outputs locate a remaining failure; they do not establish an acoustic or annotation cause.',
    }
    report['by_composition'] = {}
    groups = np.asarray([group_stem(str(m)) for m in saved['member']])
    for group in sorted(set(groups)):
        mask = groups == group
        report['by_composition'][group] = metrics(k[mask], final[mask])
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'decision-audit.json', report)
    category = np.full(len(k), '', dtype='U64')
    for name, mask in categories.items():
        category[mask] = name
    np.savez_compressed(args.output_dir / 'decision-audit.npz', global_index=idx, k=k,
                        member=saved['member'], anchor=a, category=category, **stages)
    with (args.output_dir / 'poly-errors.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['global_index', 'member', 'true_k', 'anchor', 'uniform', 'weighted',
                         'specialist', 'low', 'base', 'final', 'audit_category'])
        for i in np.flatnonzero(poly & (final != k)):
            writer.writerow([int(idx[i]), str(saved['member'][i]), int(k[i]), int(a[i]),
                             *(int(stages[name][i]) for name in
                               ('uniform', 'weighted', 'specialist', 'low', 'base', 'final')),
                             str(category[i])])
    print({'control': report['stages']['final'],
           'errors': {name: x['rows'] for name, x in report['errors_disjoint'].items()}}, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', type=int, default=3, choices=[3])
    p.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    for name in ('fold-dir', 'bundle-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
