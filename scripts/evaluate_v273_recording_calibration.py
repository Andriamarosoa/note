"""Leave-one-piece-out calibration within each frozen model's excluded fold."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import numpy as np

from scripts.v273_probability_calibration import (
    BOUNDS, IDENTITY, RIDGE, availability, decode, fit, subset, transform,
)
from scripts.v273_confidence_metrics import summarize
from scripts.verify_v273_selector_repair_artifacts import (
    digest, ids_digest, metrics, named_pairs, require,
)

FOLDS = (0, 1, 2, 4)
ARMS = ('control7', 'corrector8')


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def combine(items):
    d = {k: np.concatenate([x[k] for x in items]) for k in items[0]}
    return subset(d, np.argsort(d['global_index']))


def load_metadata(root):
    lookup = {}; sources = []
    for path in sorted(root.rglob('rows.jsonl')):
        folds = set()
        for line in path.read_text().splitlines():
            r = json.loads(line); gid = int(r['global_index']); f = int(r['fold'])
            require(gid not in lookup and f in FOLDS, 'metadata coverage')
            m = re.fullmatch(r'\d{2}_(.+)_(?:comp|solo)\.jams', r['recording_id'])
            require(m is not None and r['recording_id'][:2] in ('00','01','02','03','04'), 'recording scope')
            lookup[gid] = dict(recording_id=r['recording_id'], piece=m.group(1), fold=f,
                               start_sample=int(r['start_sample']), truth=int(r['true_k']),
                               baseline=int(r['baseline_k']))
            folds.add(f)
        require(len(folds) == 1, 'one feature fold per source')
        sources.append(dict(fold=folds.pop(), sha256=digest(path)))
    require(len(lookup) == 7493 and {s['fold'] for s in sources} == set(FOLDS), 'feature inventory')
    piece_folds = {}; recording_folds = {}
    for r in lookup.values():
        for key, target in [('piece', piece_folds), ('recording_id', recording_folds)]:
            require(r[key] not in target or target[r[key]] == r['fold'], key+' spans folds')
            target[r[key]] = r['fold']
    require(len(piece_folds) == 19 and len(recording_folds) == 169, 'predeclared block count')
    return lookup, sorted(sources, key=lambda s: s['fold'])


def fail_family(y, b, prediction, av):
    family = np.full(len(y), 'keep', dtype='<U48')
    take = prediction != b
    family[take & (prediction == y)] = 'correction'
    family[take & (b == y)] = 'regression'
    reachable = av[np.arange(len(y)), y]
    family[take & (b != y) & (prediction != y) & ~reachable] = 'no_correct_alternative'
    family[take & (b != y) & (prediction != y) & reachable] = 'wrong_choice_despite_correct_alternative'
    return family


def family_counts(family, mask=None):
    if mask is not None:
        family = family[mask]
    return {name: int((family == name).sum()) for name in
            ['keep', 'correction', 'regression', 'no_correct_alternative',
             'wrong_choice_despite_correct_alternative']}


def diagnostics(raw, calibrated):
    y, b = raw['truth'], raw['baseline']
    old, best, old_gain = decode(raw); new, new_best, gain = decode(calibrated)
    av = availability(raw['proposal'], b); has = av.any(1)
    require(np.array_equal(best, new_best), 'calibration changed destination ranking')
    old_take, new_take = old != b, new != b
    old_f = fail_family(y, b, old, av); new_f = fail_family(y, b, new, av)
    count = int(np.log2(raw['proposal'].shape[1]+1))
    single = raw['proposal'][:, [2**i-1 for i in range(count)]]
    masks = np.array([[bool(g & (1 << j)) for j in range(count)]
                      for g in range(1, 2**count)], dtype=np.int16)
    no_member_correct = (single == y[:, None]).astype(np.int16) @ masks.T == 0
    synergy = ((raw['proposal'] == y[:, None]) & no_member_correct & (masks.sum(1)[None] >= 2)).any(1)
    reachable = av[np.arange(len(y)), y]
    joint_only = (b != y) & reachable & ~(single == y[:, None]).any(1)
    same_cp = calibrated['class_probability'][np.arange(len(y)), old]
    result = dict(
        ranking_unchanged=True,
        original_change_population_rescored=dict(rows=int(old_take.sum()),
            expected_corrections=float(same_cp[old_take].sum()),
            expected_regressions=float(calibrated['baseline_correct'][old_take].sum()),
            expected_net=float((same_cp-calibrated['baseline_correct'])[old_take].sum()),
            observed=named_pairs(y, b, old)['global']),
        original_failures=family_counts(old_f), calibrated_failures=family_counts(new_f),
        retained_changes=family_counts(new_f, old_take & new_take),
        cancelled_changes=family_counts(old_f, old_take & ~new_take),
        added_changes=family_counts(new_f, ~old_take & new_take),
        fixed_proposal_limits=dict(initial_errors=int((b != y).sum()),
            errors_with_correct_alternative=int(((b != y) & reachable).sum()),
            errors_without_correct_alternative=int(((b != y) & ~reachable).sum()),
            errors_with_correct_best_destination=int(((b != y) & has & (best == y)).sum()),
            reachable_errors_with_wrong_best_destination=int(((b != y) & reachable & (best != y)).sum())),
        synergy=dict(events_with_correct_group_and_all_its_members_wrong=int(synergy.sum()),
            corrected_before=int((synergy & old_take & (old == y)).sum()),
            corrected_after=int((synergy & new_take & (new == y)).sum()),
            events_correctable_only_by_combination=int(joint_only.sum()),
            combination_only_corrected_before=int((joint_only & (old == y)).sum()),
            combination_only_corrected_after=int((joint_only & (new == y)).sum())),
        confident_errors={str(t): family_counts(new_f, new_take & (new != y) & (gain >= t))
                          for t in (.1, .5)})
    return result, old, new, best, old_gain, gain, old_f, new_f


def evaluate(arm, replay, original, lookup, feature_sources, output):
    repo = Path(__file__).resolve().parents[1]
    proof = json.loads((repo/'analysis/evidence/v273-confidence-audit/verification.json').read_text())
    source = replay/arm
    require(digest(source/'report.json') == proof['arms'][arm]['report_sha256'], 'replay report identity')
    report = json.loads((source/'report.json').read_text())
    pieces = []
    for f in FOLDS:
        path = source/f'fold-{f}-held.npz'
        require(digest(path) == report['files'][path.name]['sha256'], 'held snapshot identity')
        pieces.append(read(path))
    raw = combine(pieces); ids = raw['global_index']
    require(len(ids) == len(set(ids)) == 7493, 'unique eligible IDs')
    for k in ('fold', 'truth', 'baseline'):
        require(np.array_equal(raw[k], [lookup[int(g)][k] for g in ids]), 'metadata '+k)
    metadata = {key: np.array([lookup[int(g)][key] for g in ids])
                for key in ('recording_id', 'piece', 'start_sample')}
    trained_proof = json.loads((repo/'analysis/evidence/v273-learned-selection/verification.json').read_text())
    native_path = original/arm/'predictions.npz'
    require(digest(native_path) == trained_proof['arms'][arm]['predictions_sha256'], 'original predictions identity')
    native = read(native_path); native_lookup = {int(v): i for i, v in enumerate(native['global_index'])}
    pos = np.array([native_lookup[int(v)] for v in ids])
    require(np.array_equal(decode(raw)[0], native['predicted_K'][pos]), 'all archived decisions')
    output.mkdir(parents=True, exist_ok=False)
    calibrated = dict(raw)
    for key in ('baseline_logits', 'pooled_class_logits', 'baseline_correct', 'class_probability', 'other_probability'):
        calibrated[key] = np.full(raw[key].shape, np.nan, dtype=float)
    assignments = np.full(len(ids), -1, np.int32); calibrators = []
    for f in FOLDS:
        for piece in sorted(set(metadata['piece'][raw['fold'] == f])):
            test = (raw['fold'] == f) & (metadata['piece'] == piece)
            learn = (raw['fold'] == f) & (metadata['piece'] != piece)
            require(not set(metadata['recording_id'][test]) & set(metadata['recording_id'][learn]), 'recording leakage')
            require(not set(ids[test]) & set(ids[learn]), 'event leakage')
            fitting = fit(subset(raw, learn))
            transformed = transform(raw['baseline_logits'][test], raw['pooled_class_logits'][test],
                                    raw['proposal'][test], raw['baseline'][test], fitting['parameters'])
            for key, values in transformed.items():
                calibrated[key][test] = values
            assignments[test] = len(calibrators)
            calibrators.append(dict(index=len(calibrators), network_excluded_fold=f,
                network_training_folds=sorted(set(FOLDS)-{f}), test_piece=str(piece),
                calibration_pieces=sorted(set(metadata['piece'][learn])),
                calibration_rows=int(learn.sum()), test_rows=int(test.sum()),
                calibration_recordings=sorted(set(metadata['recording_id'][learn])),
                test_recordings=sorted(set(metadata['recording_id'][test])),
                calibration_ids_sha256=ids_digest(ids[learn]), test_ids_sha256=ids_digest(ids[test]),
                forbidden_overlap=0, **fitting))
    require(len(calibrators) == 19 and (assignments >= 0).all(), 'complete cross-calibration')
    require(all(np.isfinite(calibrated[k]).all() for k in transformed), 'complete calibrated outputs')
    diag, old, new, best, old_gain, gain, old_f, new_f = diagnostics(raw, calibrated)
    prediction = native['frozen_baseline_K'].copy(); prediction[pos] = new
    native_y, native_b = native['true_K'], native['frozen_baseline_K']
    result = dict(status='completed', arm=arm, network_weights_changed=False,
        unseen_data_validation=False, promotion=False, scope='Supervised calibration on other pieces in the same excluded network fold.',
        source_replay_run=37797488453, source_replay_report_sha256=digest(source/'report.json'),
        source_predictions_sha256=digest(native_path), feature_sources=feature_sources,
        configuration=dict(ridge=RIDGE, bounds=BOUNDS, initial_parameters=IDENTITY.tolist(),
            calibration_parameters=4, test_blocks=19, evaluated_events=7493, neural_epochs_added=0,
            fitted_threshold=False, independent_calibration_and_evaluation_pieces=True,
            calibration_uses_labels_in_network_excluded_fold=True),
        original=summarize(raw), calibrated=summarize(calibrated), diagnostics=diag,
        freeze_reference=metrics(native_y, native_b), original_native=metrics(native_y, native['predicted_K']),
        calibrated_native=metrics(native_y, prediction), paired=named_pairs(native_y, native_b, prediction),
        paired_vs_original=named_pairs(native_y, native['predicted_K'], prediction), folds={}, blocks=[])
    for f in FOLDS:
        take = raw['fold'] == f
        result['folds'][str(f)] = dict(original=summarize(subset(raw, take)), calibrated=summarize(subset(calibrated, take)))
    for c in calibrators:
        take = assignments == c['index']
        result['blocks'].append(dict(fold=c['network_excluded_fold'], piece=c['test_piece'], rows=int(take.sum()),
            original=summarize(subset(raw, take)), calibrated=summarize(subset(calibrated, take))))
    np.savez_compressed(output/'calibrated.npz', **calibrated, **metadata, calibrator_index=assignments,
        **{'raw_'+k: raw[k] for k in transformed})
    np.savez_compressed(output/'predictions.npz', global_index=native['global_index'], true_K=native_y,
        frozen_baseline_K=native_b, fold=native['fold'], original_K=native['predicted_K'], predicted_K=prediction,
        eligible_global_index=ids)
    (output/'calibrators.json').write_text(json.dumps(calibrators, indent=2, sort_keys=True)+'\n')
    av = availability(raw['proposal'], raw['baseline'])
    columns = ['global_index','fold','piece','recording_id','start_sample','truth','baseline','original_K','calibrated_K',
               'best_alternative_K','correct_alternative_available','gain_before','gain_after','family_before','family_after']
    cases = []
    for i in np.flatnonzero((new != raw['baseline']) & (new != raw['truth'])):
        cases.append(dict(zip(columns, [int(ids[i]), int(raw['fold'][i]), str(metadata['piece'][i]),
            str(metadata['recording_id'][i]), int(metadata['start_sample'][i]), int(raw['truth'][i]), int(raw['baseline'][i]),
            int(old[i]), int(new[i]), int(best[i]), bool(av[i, raw['truth'][i]]), float(old_gain[i]), float(gain[i]),
            str(old_f[i]), str(new_f[i])])))
    cases.sort(key=lambda r: (-r['gain_after'], r['global_index']))
    with (output/'remaining-failures.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n'); writer.writeheader(); writer.writerows(cases)
    result['highest_confidence_remaining_errors'] = cases[:20]
    result['files'] = {p.name: digest(p) for p in sorted(output.iterdir())}
    (output/'report.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(arm=arm, status='completed', original=result['original']['net'],
        calibrated=result['calibrated']['net'], expected=result['calibrated']['expected_net'],
        changes=result['calibrated']['changes'], event_nll_before=result['original']['nll']['event'],
        event_nll_after=result['calibrated']['nll']['event'], diagnostics=diag)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'refusing overwrite')
    lookup, feature_sources = load_metadata(args.features)
    for arm in ARMS:
        evaluate(arm, args.replay, args.original, lookup, feature_sources, args.output/arm)
        gc.collect()


if __name__ == '__main__':
    main()
