"""Verify calibration provenance, reconstruction and paired counts independently."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit, logsumexp

from scripts.v273_probability_calibration import fit
from scripts.verify_v273_selector_repair_artifacts import (
    digest, ids_digest, metrics, named_pairs, require,
)

FOLDS = (0, 1, 2, 4)
OUTPUT_KEYS = ('baseline_logits', 'pooled_class_logits', 'baseline_correct',
               'class_probability', 'other_probability')


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def assert_near(a, b, name):
    require(np.allclose(a, b, atol=1e-9, rtol=1e-9), name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-id', type=int, default=0)
    p.add_argument('--source-commit', default='local')
    args = p.parse_args()
    evidence = dict(status='verified', source_commit=args.source_commit, run_id=args.run_id,
        source_replay_run=37797488453, network_weights_changed=False, promotion=False,
        independent_unseen_validation=False,
        evaluation_scope='Held-out pieces, calibration labels from other pieces in the same excluded network fold.',
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__), arms={})
    table = []
    metadata = {}; feature_sources = []
    for path in args.features.rglob('rows.jsonl'):
        feature_folds = set()
        for line in path.read_text().splitlines():
            feature = json.loads(line)
            require(feature['global_index'] not in metadata, 'duplicate metadata event')
            metadata[feature['global_index']] = feature
            feature_folds.add(feature['fold'])
        require(len(feature_folds) == 1, 'one fold per feature file')
        feature_sources.append(dict(fold=feature_folds.pop(), sha256=digest(path)))
    feature_sources.sort(key=lambda s: s['fold'])
    for arm in ('control7', 'corrector8'):
        directory = args.results/arm
        r = json.loads((directory/'report.json').read_text())
        require(r['status'] == 'completed' and r['arm'] == arm, 'completed report')
        for name, sha in r['files'].items():
            require(digest(directory/name) == sha, 'output digest '+name)
        require(digest(args.replay/arm/'report.json') == r['source_replay_report_sha256'], 'source replay digest')
        require(digest(args.original/arm/'predictions.npz') == r['source_predictions_sha256'], 'source prediction digest')
        source_report = json.loads((args.replay/arm/'report.json').read_text())
        chunks = []
        for fold in FOLDS:
            path = args.replay/arm/f'fold-{fold}-held.npz'
            require(digest(path) == source_report['files'][path.name]['sha256'], 'snapshot source digest')
            chunks.append(read(path))
        raw = {k: np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
        order = np.argsort(raw['global_index']); raw = {k: v[order] for k, v in raw.items()}
        d = read(directory/'calibrated.npz'); n = len(d['truth']); row = np.arange(n)
        for k in ('truth', 'baseline', 'global_index', 'fold', 'proposal'):
            require(np.array_equal(d[k], raw[k]), 'immutable '+k)
        for k in OUTPUT_KEYS:
            require(np.array_equal(d['raw_'+k], raw[k]), 'immutable raw '+k)
        require(n == 7493 and len(set(d['global_index'])) == n, 'evaluation population')
        require(r['feature_sources'] == feature_sources, 'feature source identities')
        for j, gid in enumerate(d['global_index']):
            feature = metadata[int(gid)]
            name = feature['recording_id']
            piece = re.fullmatch(r'\d{2}_(.+)_(?:comp|solo)\.jams', name).group(1)
            require(d['recording_id'][j] == name and d['piece'][j] == piece, 'source musical block')
            require(d['fold'][j] == feature['fold'] and d['truth'][j] == feature['true_k'], 'source fold and label')
        av = np.stack([(d['proposal'] == k).any(1) & (d['baseline'] != k) for k in range(7)], 1)
        models = json.loads((directory/'calibrators.json').read_text())
        require(len(models) == 19, 'all calibrators')
        used = np.zeros(n, int)
        seen_blocks = set()
        for i, m in enumerate(models):
            f = m['network_excluded_fold']; piece = m['test_piece']
            require((f, piece) not in seen_blocks, 'repeated block'); seen_blocks.add((f, piece))
            test = (d['fold'] == f) & (d['piece'] == piece)
            learn = (d['fold'] == f) & (d['piece'] != piece)
            require(m['index'] == i and np.array_equal(d['calibrator_index'] == i, test), 'calibrator assignment')
            require(m['network_training_folds'] == sorted(set(FOLDS)-{f}), 'network exclusion')
            require(not set(d['recording_id'][test]) & set(d['recording_id'][learn]), 'recording overlap')
            require(not set(d['piece'][test]) & set(d['piece'][learn]), 'piece overlap')
            require(not set(d['global_index'][test]) & set(d['global_index'][learn]), 'event overlap')
            require(m['calibration_pieces'] == sorted(set(d['piece'][learn])), 'calibration pieces')
            require(m['test_recordings'] == sorted(set(d['recording_id'][test])), 'test recordings')
            require(m['calibration_recordings'] == sorted(set(d['recording_id'][learn])), 'calibration recordings')
            require(m['calibration_rows'] == int(learn.sum()) and m['test_rows'] == int(test.sum()), 'split sizes')
            require(m['calibration_ids_sha256'] == ids_digest(d['global_index'][learn]), 'calibration IDs')
            require(m['test_ids_sha256'] == ids_digest(d['global_index'][test]), 'test IDs')
            # Refit with only the predeclared calibration rows; evaluated labels are absent.
            refit = fit({k: v[learn] for k, v in raw.items()})
            assert_near(refit['parameters'], m['parameters'], 'calibration refit')
            ar, br, aq, bq = m['parameters']
            require(.05 <= ar <= 20 and .05 <= aq <= 20 and -10 <= br <= 10 and -10 <= bq <= 10, 'bounds')
            assert_near(d['baseline_logits'][test], ar*raw['baseline_logits'][test].astype(float)+br, 'baseline transform')
            assert_near(d['pooled_class_logits'][test], aq*raw['pooled_class_logits'][test].astype(float)+bq, 'conditional transform')
            used[test] += 1
        require(np.all(used == 1), 'each event evaluated exactly once')
        expected_blocks = {(int(f), str(p)) for f, p in zip(d['fold'], d['piece'])}
        require(seen_blocks == expected_blocks, 'all pieces evaluated')
        assert_near(d['baseline_correct'], expit(d['baseline_logits']), 'sigmoid reconstruction')
        logits = np.column_stack([np.where(av, d['pooled_class_logits'], -np.inf), np.zeros(n)])
        logq = logits-logsumexp(logits, 1, keepdims=True); q = np.exp(logq)
        rr = d['baseline_correct']; b = d['baseline']; y = d['truth']; B = b == y
        cp = (1-rr[:, None])*q[:, :7]+np.eye(7)[b]*rr[:, None]
        assert_near(cp, d['class_probability'], 'joint probabilities')
        assert_near((1-rr)*q[:, 7], d['other_probability'], 'OTHER probability')
        assert_near(cp.sum(1)+d['other_probability'], 1., 'probability simplex')
        best = np.where(av, d['pooled_class_logits'], -np.inf).argmax(1)
        raw_best = np.where(av, raw['pooled_class_logits'], -np.inf).argmax(1)
        require(np.array_equal(best, raw_best), 'within-event ranking invariance')
        take = av.any(1) & (cp[row, best] > rr)
        pred = np.where(take, best, b)
        pairs = named_pairs(y, b, pred)['global']
        for key in ('corrections', 'regressions', 'net'):
            require(pairs[key] == r['calibrated'][key], 'independent '+key)
        require(pairs['changed'] == r['calibrated']['changes'], 'independent change count')
        assert_near((cp[row, pred]-rr)[take].sum(), r['calibrated']['expected_net'], 'announced gain')
        z = d['baseline_logits']; target = np.where(av[row, y], y, 7)
        nll = np.mean(np.logaddexp(0, z)-B*z-(~B)*logq[row, target])
        assert_near(nll, r['calibrated']['nll']['event'], 'joint NLL')
        native = read(directory/'predictions.npz'); old_native = read(args.original/arm/'predictions.npz')
        for key in ('global_index', 'true_K', 'frozen_baseline_K', 'fold'):
            require(np.array_equal(native[key], old_native[key]), 'native alignment '+key)
        lookup = {int(v): i for i, v in enumerate(native['global_index'])}
        pos = np.array([lookup[int(v)] for v in d['global_index']])
        expected = native['frozen_baseline_K'].copy(); expected[pos] = pred
        require(np.array_equal(expected, native['predicted_K']), 'native prediction reconstruction')
        require(np.array_equal(native['original_K'], old_native['predicted_K']), 'archived raw decisions')
        require(metrics(native['true_K'], expected) == r['calibrated_native'], 'native exact metrics')
        require(named_pairs(native['true_K'], native['frozen_baseline_K'], expected) == r['paired'], 'native pairs')
        with (directory/'remaining-failures.csv').open() as handle:
            failures = list(csv.DictReader(handle))
        require({int(x['global_index']) for x in failures} == set(d['global_index'][take & (pred != y)]), 'failure inventory')
        high = [x for x in failures if float(x['gain_after']) >= .1]
        require(len(high) == sum(r['diagnostics']['confident_errors']['0.1'].values()), 'confident errors')
        subset_keys = ['original','calibrated','diagnostics','freeze_reference','original_native','calibrated_native','paired','paired_vs_original']
        evidence['arms'][arm] = dict(report_sha256=digest(directory/'report.json'), files=r['files'],
            calibrators_verified=len(models), events_verified=n, unchanged_destination_ranking=True,
            source_replay_report_sha256=r['source_replay_report_sha256'],
            source_predictions_sha256=r['source_predictions_sha256'], feature_sources=r['feature_sources'],
            **{k: r[k] for k in subset_keys},
            folds={f:{role:{k:s[k] for k in ['changes','corrections','regressions','net','expected_net','nll']}
                      for role,s in roles.items()} for f, roles in r['folds'].items()},
            blocks=[dict(fold=v['fold'], piece=v['piece'], rows=v['rows'],
                         original_net=v['original']['net'], calibrated_net=v['calibrated']['net'],
                         original_nll=v['original']['nll']['event'], calibrated_nll=v['calibrated']['nll']['event'])
                    for v in r['blocks']],
            calibrator_parameters=[dict(fold=m['network_excluded_fold'], piece=m['test_piece'],
                                       parameters=m['parameters'], bounds_hit=m['bounds_hit']) for m in models])
        for name in ('original','calibrated'):
            s = r[name]; nm = r[name+'_native']
            table.append(dict(arm=arm, condition=name, changes=s['changes'], corrections=s['corrections'],
                regressions=s['regressions'], net=s['net'], expected_net=s['expected_net'], optimism=s['optimism'],
                event_nll=s['nll']['event'], conditional_nll=s['nll']['conditional_on_baseline_wrong'],
                exact_global=nm['exact'], exact_poly=nm['poly']['exact']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison = args.output.parent/'calibration-comparison.csv'
    with comparison.open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n'); w.writeheader(); w.writerows(table)
    evidence['comparison_sha256'] = digest(comparison)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified', calibrators=38, evaluations=14986, comparison=table)), flush=True)


if __name__ == '__main__':
    main()
