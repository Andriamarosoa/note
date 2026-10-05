"""Separate harmonic-shape and amplitude-prior effects on frozen internal cases.

All annotated frequencies are diagnostic inputs, never deployment features.
No neural training or fold-3 data. See the committed exploratory protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls

from causal_note.guitarset import SAMPLE_RATE, index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_internal_residual_acoustics import (
    DATA_MD5, GROUPS, match_frequencies,
)
from scripts.v273_residual_audit import FOLDS, require, sha256_file, write_json

SHAPES = ("gaussian", "hann_power")
EXPONENTS = (0.5, 1.0, 2.0)
RIDGE = 1e-10


def hann_transform(delta_hz):
    """Exact DTFT of numpy.hanning(WINDOW) at continuous frequency offsets."""
    delta = np.asarray(delta_hz) / SAMPLE_RATE
    def dirichlet(q):
        return (h.WINDOW * np.sinc(h.WINDOW * q) / np.sinc(q)
                * np.exp(-1j * np.pi * (h.WINDOW - 1) * q))
    return (0.5 * dirichlet(delta)
            - 0.25 * dirichlet(delta - 1 / (h.WINDOW - 1))
            - 0.25 * dirichlet(delta + 1 / (h.WINDOW - 1)))


def harmonic_basis(freq, f0, shape):
    centers = np.arange(1, h.MAX_HARMONICS + 1) * f0
    centers = centers[centers <= freq[-1]]
    delta = freq[:, None] - centers
    if shape == "gaussian":
        return np.exp(-0.5 * (delta / h.KERNEL_HZ) ** 2)
    require(shape == "hann_power", "unknown kernel")
    # Phase-averaged real sinusoid power, normalized to the positive peak.
    return ((np.abs(hann_transform(delta)) ** 2
             + np.abs(hann_transform(freq[:, None] + centers)) ** 2)
            / ((h.WINDOW - 1) / 2) ** 2)


def rigid_dictionary(bases, exponent):
    columns = []
    for B in bases:
        v = B @ np.arange(1, B.shape[1] + 1, dtype=float) ** (-exponent)
        columns.append(v / (np.linalg.norm(v) + 1e-12))
    return np.column_stack(columns)


def gram_nnls(G, b, x2):
    """NNLS for ||x-Da||² from D.T@D and D.T@x, with a tiny numerical ridge."""
    L = cholesky(G + RIDGE * np.eye(len(b)), lower=True, check_finite=False)
    target = solve_triangular(L, b, lower=True, check_finite=False)
    a, _ = nnls(L.T, target, maxiter=20 * len(b))
    cost = max(float(x2 - 2 * a @ b + a @ G @ a), 0.0)
    return cost, a


def fit_groups(bases, x, true_indices):
    """Exhaustively select F0 groups, allowing each harmonic its own weight."""
    D = np.column_stack(bases)
    D /= np.linalg.norm(D, axis=0) + 1e-12
    offsets = np.r_[0, np.cumsum([B.shape[1] for B in bases])]
    columns = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(bases))]
    G, b, x2 = D.T @ D, D.T @ x, float(x @ x)
    best = {}
    true_key = tuple(true_indices)
    forced = None
    for k in (2, 3):
        winner = None
        for combo in itertools.combinations(range(len(bases)), k):
            ids = np.concatenate([columns[i] for i in combo])
            cost, a = gram_nnls(G[np.ix_(ids, ids)], b[ids], x2)
            if winner is None or cost < winner[0]:
                winner = (cost, combo, a)
            if combo == true_key:
                forced = cost
        best[k] = winner
    require(forced is not None, "missing annotation combination")
    return best[2], best[3], forced, x2


def describe(pool, expected, pair, triplet, forced, x2):
    f2, f3 = pool[list(pair[1])], pool[list(triplet[1])]
    m2, _ = match_frequencies(expected, f2)
    m3, _ = match_frequencies(expected, f3)
    return {
        "pair_f0": f2.tolist(), "triplet_f0": f3.tolist(),
        "pair_matches": m2, "triplet_matches": m3,
        "pair_all_owned": m2 == len(expected), "triplet_all_owned": m3 == len(expected),
        "pair_residual": float(pair[0] / (x2 + 1e-12)),
        "triplet_residual": float(triplet[0] / (x2 + 1e-12)),
        "forced_annotation_residual": float(forced / (x2 + 1e-12)),
        "forced_minus_best_same_k": float((forced - (pair if len(expected) == 2 else triplet)[0]) / (x2 + 1e-12)),
    }


def extract(a, rows):
    for name, expected in DATA_MD5.items():
        with (a.dataset / name).open('rb') as stream:
            require(hashlib.file_digest(stream, 'md5').hexdigest() == expected, 'dataset changed')
    wanted = {r['recording_id'] for r in rows}
    tracks = {t.annotation_member: t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks) == wanted, 'missing tracks')
    result = {}
    for member in sorted(wanted):
        t = tracks[member]
        audio = decode_pcm16_mono_wav(t.audio_zip, t.audio_member)
        samples = np.asarray(audio.samples, np.float64) / 32768
        for r in (r for r in rows if r['recording_id'] == member):
            freq, x = h.transition_spectrum(samples, r['start_sample'])
            result[r['row_id']] = x
    return freq, result


def aggregate(rows):
    output = {}
    for group in GROUPS:
        rr = [r for r in rows if r['group'] == group]
        if not rr:
            continue
        output[group] = {'rows': len(rr), 'arms': {}}
        for arm in rr[0]['arms']:
            aa = [r['arms'][arm] for r in rr]
            output[group]['arms'][arm] = {
                'pair_all_owned': sum(r['pair_all_owned'] for r in aa),
                'triplet_all_owned': sum(r['triplet_all_owned'] for r in aa),
                **{k: float(np.mean([r[k] for r in aa])) for k in
                   ('pair_matches', 'triplet_matches', 'pair_residual', 'triplet_residual',
                    'forced_annotation_residual', 'forced_minus_best_same_k')},
            }
    return output


def run(a):
    require(not a.output.exists(), 'refusing overwrite')
    source = json.loads((a.audit / 'report.json').read_text())
    cases_path = a.audit / 'cases.jsonl'
    require(sha256_file(cases_path) == source['cases_sha256'], 'case hash changed')
    config = json.loads(a.config.read_text())
    require(sha256_file(a.config) == source['config_sha256'], 'config changed')
    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    require(all(r['fold'] in FOLDS and config['member_folds'][r['recording_id']] == r['fold'] for r in rows), 'outer fold')
    probe = json.loads((a.audit / 'dictionary-probe.json').read_text())
    require(probe['source_cases_sha256'] == source['cases_sha256'], 'mixed probe source')
    previous = {r['row_id']: r for r in probe['cases']}
    if a.limit:
        rows = rows[:a.limit]
    freq, spectra = extract(a, rows)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'spectra.npz', freq=freq,
                        row_id=np.array([r['row_id'] for r in rows]),
                        x=np.array([spectra[r['row_id']] for r in rows]))
    results = []
    started = time.monotonic()
    with (a.output / 'cases.jsonl').open('w') as stream:
        for row in rows:
            expected = np.array([n['frequency_hz'] for n in row['owned_notes']])
            base = row['decomposition']['pool_f0']
            pool = np.r_[base, expected]
            true_ids = tuple(range(len(base), len(pool)))
            x = spectra[row['row_id']]
            arms = {}
            synthetic = None
            for shape in SHAPES:
                bases = [harmonic_basis(freq, f, shape) for f in pool]
                for exponent in EXPONENTS:
                    D = rigid_dictionary(bases, exponent)
                    pair, G, b, x2 = h.fit_best(D, x, 2)
                    trip, _, _, _ = h.fit_best(D, x, 3)
                    forced, _ = h.nonnegative_cost(G, b, x2, true_ids)
                    name = f'{shape}_p{exponent:g}'
                    arms[name] = describe(pool, expected, pair, trip, forced, x2)
                    if name == 'gaussian_p0.5':
                        if row['row_id'] in previous:
                            require(abs(arms[name]['triplet_residual'] - previous[row['row_id']]['augmented_residual']) < 1e-10, 'old probe replay drift')
                        sx = D[:, list(true_ids)] @ np.linspace(1.0, 0.6, len(true_ids))
                        sb, _, _, sx2 = h.fit_best(D, sx, len(true_ids))
                        recovered = pool[list(sb[1])]
                        synthetic = {'matches': match_frequencies(expected, recovered)[0],
                                     'residual': sb[0] / sx2, 'selected': recovered.tolist()}
                pair, trip, forced, x2 = fit_groups(bases, x, true_ids)
                arms[shape + '_free'] = describe(pool, expected, pair, trip, forced, x2)
            result = {k: row[k] for k in ('row_id', 'fold', 'recording_id', 'start_sample', 'time_seconds', 'group', 'true_K') if k in row}
            result.update(annotated_f0=expected.tolist(), candidate_f0=pool.tolist(),
                          arms=arms, synthetic_control=synthetic)
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            results.append(result)
            if len(results) % 10 == 0:
                print(json.dumps({'cases': len(results), 'seconds': time.monotonic() - started}), flush=True)
    report = {'status': 'completed', 'scope': 'smoke' if a.limit else 'full',
              'cases': len(results), 'outer_fold_3_used': False, 'diagnostic_only': True,
              'neural_training': False, 'automatic_promotion': False,
              'arms': list(results[0]['arms']), 'groups': aggregate(results),
              'by_fold': {str(f): aggregate([r for r in results if r['fold'] == f]) for f in FOLDS},
              'synthetic_control': {'all_owned': sum(r['synthetic_control']['matches'] == r['true_K'] for r in results),
                                    'max_residual': max(r['synthetic_control']['residual'] for r in results)},
              'source_cases_sha256': source['cases_sha256'], 'source_data_md5': DATA_MD5,
              'script_sha256': sha256_file(__file__), 'config_sha256': sha256_file(a.config),
              'cases_sha256': sha256_file(a.output / 'cases.jsonl'),
              'limitations': ['Annotated F0 candidates are oracle diagnostics, not inference inputs.',
                              'Previously inspected internal cases; no independent validation.',
                              'Free harmonic fits have more parameters and can favor subharmonic groups.',
                              'Alternative legacy candidate tie orders are inherited from the source acoustic audit.',
                              'No waveform compression experiment and no Exact-K performance gain claimed.']}
    write_json(a.output / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'dataset', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Smoke check only; never label it full cohort')
    run(p.parse_args())
