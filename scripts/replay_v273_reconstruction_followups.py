"""Replay reconstruction diagnostics and frozen guards without audio or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from scripts.audit_v273_hann_shape_guard import NAMES as HANN_NAMES
from scripts.audit_v273_log_frequency_guard import NAMES as LOG_NAMES
from scripts.audit_v273_reconstruction_criterion import run as run_criterion
from scripts.audit_v273_two_variant_frequency_coverage import run as run_coverage
from scripts.replay_v273_harmonic_decay_guard import verify as verify_models
from scripts.v273_residual_audit import require, sha256_file, write_json


def arrays(root):
    with np.load(root / 'features.npz', allow_pickle=False) as z:
        return dict(z)


def rerun_coverage(ranking, guard, cases, saved, output):
    run_coverage(SimpleNamespace(ranking=ranking, guard=guard, cases=cases, output=output))
    require(sha256_file(output / 'report.json') == sha256_file(saved / 'report.json'), 'coverage report changed')
    require(sha256_file(output / 'cases.jsonl') == sha256_file(saved / 'cases.jsonl'), 'coverage cases changed')


def verify(candidate, cases, criterion, log_guard, log_coverage, hann_guard, hann_coverage):
    capacity = arrays(candidate / 'capacity')
    log = arrays(log_guard); hann = arrays(hann_guard)
    require(tuple(log['variants']) == LOG_NAMES and tuple(hann['variants']) == HANN_NAMES, 'variant schema changed')
    np.testing.assert_array_equal(log['row_id'], capacity['row_id'])
    np.testing.assert_array_equal(hann['row_id'], capacity['row_id'])
    np.testing.assert_array_equal(log['features'][:, 0], capacity['features'][:, 3])
    np.testing.assert_array_equal(log['pair_f0'][:, 0], capacity['pair_f0'][:, 3])
    np.testing.assert_array_equal(log['triplet_f0'][:, 0], capacity['triplet_f0'][:, 3])
    np.testing.assert_array_equal(hann['features'][:, 0], capacity['features'][:, 3])
    np.testing.assert_array_equal(hann['pair_f0'][:, 0], capacity['pair_f0'][:, 3])
    np.testing.assert_array_equal(hann['triplet_f0'][:, 0], capacity['triplet_f0'][:, 3])
    with np.load(hann_guard / 'spectra.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['row_id'], capacity['row_id'])
        require(z['x'].shape[0] == len(capacity['row_id']) == 1666, 'spectrum population changed')
        require(z['x'].shape[1] == len(z['freq']) and np.isfinite(z['x']).all(), 'bad spectra')
        require(np.all(z['x'] >= 0), 'negative positive-transition spectrum')
    log_models = verify_models(log_guard, LOG_NAMES)
    hann_models = verify_models(hann_guard, HANN_NAMES)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        criterion_out = temp / 'criterion'
        run_criterion(SimpleNamespace(ranking=candidate / 'ranking', capacity=candidate / 'capacity',
                                      cases=cases, output=criterion_out))
        require(sha256_file(criterion_out / 'report.json') == sha256_file(criterion / 'report.json'), 'criterion report changed')
        require(sha256_file(criterion_out / 'cases.jsonl') == sha256_file(criterion / 'cases.jsonl'), 'criterion cases changed')
        rerun_coverage(candidate / 'ranking', log_guard, cases, log_coverage, temp / 'log-coverage')
        rerun_coverage(candidate / 'ranking', hann_guard, cases, hann_coverage, temp / 'hann-coverage')
    return {'status': 'verified', 'outer_fold_3_used': False, 'audio_loaded': False,
            'model_refitted': False, 'rows': 845, 'unique_spectra': 1666,
            'criterion_cases': 488, 'log_models': log_models, 'hann_models': hann_models,
            'source_sha256': {'cases': sha256_file(cases),
                              'criterion_report': sha256_file(criterion / 'report.json'),
                              'log_report': sha256_file(log_guard / 'report.json'),
                              'hann_report': sha256_file(hann_guard / 'report.json'),
                              'script': sha256_file(__file__)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'cases', 'criterion', 'log_guard', 'log_coverage',
                 'hann_guard', 'hann_coverage', 'output'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    result = verify(a.candidate, a.cases, a.criterion, a.log_guard, a.log_coverage,
                    a.hann_guard, a.hann_coverage)
    write_json(a.output, result)
    print(json.dumps(result, sort_keys=True))
