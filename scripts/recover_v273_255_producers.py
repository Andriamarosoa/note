"""Recover deployable eight-vote producers and replay their frozen 255 decisions.

Five logistic specialists are refitted with the ORIGINAL fixed training recipe,
using only each manifest's original training IDs. Every cached float32 posterior
must match exactly. The saved corrector weights are not retrained. Its NumPy
forward pass is checked within 2e-6; every group decision must match exactly.
No new-audio annotation or accuracy scoring is accepted by this module.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import joblib
import numpy as np
from scipy.special import erf, softmax
from scripts.v273_selector_contract import FOLDS, EXPERTS, ProducerPool, context_columns, id_digest
from scripts.prepare_v273_group127_producers import cache_key, FrozenProducerPool
from scripts.prepare_v273_learned_corrector import FrozenCorrector, CataloguePool, project_actions
from scripts.v273_group127_contract import seven_candidates
from scripts.v273_catalogue_contract import joint_groups, local_audits
from scripts.v273_regression_loops import group_context

EVIDENCE = Path('analysis/evidence/v273-audit-memory/source-archives')
PREPARED = Path('analysis/evidence/v273-regression-loops/prepared/inputs.npz')
POSTERIOR_ATOL = 2e-6
HARMONIC_ARCHIVE_SHA = {
    0: 'cb11dbf454ab799df82d03d2047ff40cfba7673eaf3ad0ada973c739548dd8b9',
    1: 'aad71a47317ee51d0aaa2c4abf1eced59680d0deab7663478420c21677ef506a',
    2: 'b26d833afad090c9fad171f65f4e36125d1ea0033c07c296dee7331791ab33df',
    4: '83fa031e1d7cc1c262f1aff1dbde42219089b3bf0f9c0b88d92eaf72e6d81b87',
}
GOAT_SHA = {
    'example1_di.wav': 'd05060b71ea19ba2093aef23ba411bafbbeb8f402973b45eb32933c2bd10a458',
    'example2_di.wav': '0e12f7bbb3ca210d3e245f10d24ba19547cfe97d58db087de89bdd4bd375384b',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def prepare_fundamentals(root, archives, output):
    """Preserve original fundamental-only features beyond artifact expiration."""
    need(not output.exists(), 'immutable original feature snapshot')
    d = read(root/PREPARED, ('global_index', 'features', 'feature_names', 'fold'))
    rows, sources = {}, []
    for fold in FOLDS:
        path = archives/f'fold{fold}.zip'
        need(sha(path) == HARMONIC_ARCHIVE_SHA[fold], 'original harmonic artifact SHA')
        with zipfile.ZipFile(path) as z:
            entries = [n for n in z.namelist() if n.endswith('rows.jsonl')]
            need(len(entries) == 1, 'ambiguous original feature payload')
            payload = z.read(entries[0])
            for line in payload.splitlines():
                r = json.loads(line)
                ident = int(r['global_index'])
                need(ident not in rows and int(r['fold']) == fold, 'duplicate/wrong-fold feature row')
                rows[ident] = (r['features'], r['fundamental'])
        sources.append(dict(fold=fold, run_id=37740493178, archive_sha256=sha(path),
                            rows_payload_sha256=hashlib.sha256(payload).hexdigest()))
    need(set(rows) == set(d['global_index']), 'original feature coverage')
    names = list(map(str, d['feature_names']))
    original = np.array([[rows[int(i)][0][n] for n in names] for i in d['global_index']])
    need(np.array_equal(original, d['features']), 'original 58D primary features differ')
    fundamental = np.array([[rows[int(i)][1][n] for n in names] for i in d['global_index']])
    output.mkdir(parents=True)
    np.savez_compressed(output/'historical_fundamental58.npz', global_index=d['global_index'],
                        feature_names=d['feature_names'], fundamental=fundamental)
    write_json(output/'historical_fundamental_provenance.json', dict(sources=sources,
        original_primary_features_bit_exact=True, rows=7493, true_label_fields_copied=False,
        snapshot_sha256=sha(output/'historical_fundamental58.npz')))


def extract_goat(s97, audio, output):
    from scripts.extract_v273_harmonic_trajectory import _frames, features_from_spectrogram, FUNDAMENTAL_BANK
    from scripts.s82_external_pr16_neural_inference import load_wav
    need(not output.exists(), 'immutable new-audio input snapshot')
    a = read(s97, ('row_id', 'member', 'start_samples', 'summary58'))
    primary, fundamental = [], []
    signals = {}
    for name, expected in GOAT_SHA.items():
        need(sha(audio/name) == expected, 'pinned GOAT WAV mismatch')
        signals[name] = load_wav(audio/name)
    for ident, member, start in zip(a['row_id'], a['member'], a['start_samples']):
        name = str(member)
        need(str(ident) == f'{name}:{int(start)}', 'WAV group identity')
        spec, times = _frames(signals[name], int(start))
        d = features_from_spectrogram(spec, times)
        f = features_from_spectrogram(spec, times, bank=FUNDAMENTAL_BANK)
        names = sorted(d)
        need(names == sorted(f) and len(names) == 58, 'acoustic schema')
        primary.append([d[k] for k in names]); fundamental.append([f[k] for k in names])
    primary = np.asarray(primary, np.float64)
    delta = np.abs(primary-a['summary58'])
    # Sparse near-silent activation arithmetic differs across numerical builds.
    # Expose the difference; do not claim bit identity with S97 float32 features.
    scaled = float(np.max(delta/(1+np.abs(a['summary58']))))
    need(scaled < 1e-3, 'new waveform summaries incompatible with pinned S97')
    output.mkdir(parents=True)
    np.savez_compressed(output/'external_audio_inputs.npz', row_id=a['row_id'], features=primary,
                        fundamental=np.asarray(fundamental), feature_names=np.asarray(names))
    write_json(output/'external_audio_provenance.json', dict(WAV_sha256=GOAT_SHA,
        GOAT_commit='55c6e6668ef4546e4635c5e7eaad3062cf7f036b',
        source_S97_sha256=sha(s97), rows=len(primary),
        source_feature_function='scripts.extract_v273_harmonic_trajectory.features_from_spectrogram',
        fundamental_bank='FUNDAMENTAL_BANK', receiver_annotations_read=False,
        primary_float32_identical_to_S97=bool(np.array_equal(primary.astype(np.float32), a['summary58'])),
        primary_max_abs_difference_to_S97=float(delta.max()), primary_max_scaled_difference_to_S97=scaled,
        snapshot_sha256=sha(output/'external_audio_inputs.npz')))


def restore(root, destination):
    manifest = json.loads((root/EVIDENCE/'manifest.json').read_text())
    archive = root/EVIDENCE/'upstream-producers.zip'
    info = manifest['files'][archive.name]
    need(sha(archive) == info['sha256'], 'original producer archive SHA mismatch')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe member')
            data = z.read(name)
            need(hashlib.sha256(data).hexdigest() == info['members'][name]['sha256'], name+' SHA')
            p = destination/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    return info['sha256']


def matrices(features, fundamental, names):
    result = {}
    for name, (field, prefixes) in EXPERTS.items():
        cols = [i for i, n in enumerate(names) if not prefixes or n.startswith(prefixes)]
        result[name] = (fundamental if field == 'fundamental' else features)[:, cols]
    raw = features[:, [names.index(n) for n in context_columns(names)]]
    need(all(np.isfinite(a).all() for a in (*result.values(), raw)), 'nonfinite acoustics')
    return result, raw


def corrector_probabilities(directory, record, raw, baseline):
    path = directory/record['weights']
    need(sha(path) == record['weights_sha256'], 'corrector weights SHA')
    x = np.concatenate([np.clip((raw-record['scaler_mean'])/record['scaler_scale'], -6, 6),
                        np.eye(7)[baseline]], axis=1).astype(np.float32).astype(np.float64)
    with h5py.File(path, 'r') as h:
        layers = [(h[f'layers/{name}/vars/0'][:].astype(np.float64),
                   h[f'layers/{name}/vars/1'][:].astype(np.float64))
                  for name in ('dense', 'dense_1', 'dense_2')]
    blocks = []
    # Preserve original float32 input rounding and weights. Float64 accumulation
    # avoids hardware-dependent float32 noise amplified by joint_groups' sqrt
    # of near-zero variance. The output is rounded back to original float32.
    for start in range(0, len(x), 192):
        a = x[start:start+192]
        for j, (weight, bias) in enumerate(layers):
            a = a@weight+bias
            if j < 2:
                a = a*.5*(1+erf(a/np.sqrt(2)))
        blocks.append(softmax(a, axis=1).astype(np.float32))
    return np.concatenate(blocks) if blocks else np.empty((0, 7), np.float32)


def specialist_probabilities(models, features, baseline):
    p = np.zeros((len(baseline), 6, 5), np.float32)
    p[np.arange(len(baseline)), 0, baseline-2] = 1
    for j, (name, model) in enumerate(zip(EXPERTS, models), 1):
        q = model.predict_proba(features[name])
        p[:, j] = np.maximum(q, 1e-10)/np.maximum(q.sum(1, keepdims=True), 1e-10)
    return p


def votes(models, features, raw, baseline, correctors, record):
    p = specialist_probabilities(models, features, baseline)
    q = corrector_probabilities(correctors, record, raw, baseline)
    return np.concatenate([seven_candidates(p), project_actions(q, baseline)[:, None]], axis=1), p, q


def recover(root, fundamentals, output, saved_specialists=None, saved_report=None):
    need(not output.exists(), 'refuse to overwrite producer evidence')
    source_manifest = json.loads((root/PREPARED.parent/'input-manifest.json').read_text())
    need(sha(root/PREPARED) == source_manifest['inputs_sha256'], 'prepared historical source SHA')
    d = read(root/PREPARED, ('global_index', 'baseline', 'fold', 'features', 'feature_names'))
    f = read(fundamentals, ('global_index', 'feature_names', 'fundamental'))
    need(np.array_equal(d['global_index'], f['global_index']) and
         np.array_equal(d['feature_names'], f['feature_names']), 'fundamental row/column alignment')
    names = list(map(str, d['feature_names']))
    need(d['features'].shape == f['fundamental'].shape == (7493, 58), '7493x58 original inputs')
    x, raw = matrices(d['features'], f['fundamental'], names)
    output.mkdir(parents=True)
    source_sha = restore(root, output/'original')
    original = output/'original'
    # Original development labels serve only to reconstruct original training
    # states and fold-excluded reference audits. No new receiver labels exist.
    with np.load(original/'specialists/producers.npz') as z:
        truth = z['truth'].copy()
    pool = ProducerPool(x, truth, d['baseline'], d['fold'], d['global_index'])
    frozen = FrozenProducerPool({}, truth, d['baseline'], d['fold'], d['global_index'], original/'specialists')
    corrector = FrozenCorrector(truth, d['baseline'], d['fold'], d['global_index'], original/'learned_corrector')
    catalogue = CataloguePool(frozen, corrector)
    records = {tuple(r['train_folds']): r for r in corrector.manifest['producers']}
    saved = None
    if saved_specialists is not None:
        need(saved_report is not None, 'saved model provenance required')
        provenance = json.loads(saved_report.read_text())
        need(sha(saved_specialists) == provenance['reconstructed_models_sha256'] and
             sha(root/PREPARED) == provenance['original_inputs_sha256'] and
             sha(fundamentals) == provenance['fundamental_input_sha256'], 'saved runtime/input SHA')
        need(provenance['specialist_cached_probabilities_bit_exact'] and
             provenance['group_decision_mismatches'] == 0, 'saved reconstruction was not verified')
        saved = joblib.load(saved_specialists)
        need(set(saved) == {cache_key(k) for k in records}, 'saved fourteen configurations')
    report_rows = []
    weights = {}
    for key in sorted(records, key=lambda k: (len(k), k)):
        at = np.flatnonzero(~np.isin(d['fold'], key))
        if saved is None:
            p = pool.predict(key, at, set(FOLDS)-set(key))
            need(pool.manifests[key]['train_id_sha256'] == frozen.manifests[key]['train_id_sha256'], 'original training split changed')
            models = pool.models[key]
        else:
            models = saved[cache_key(key)]
            p = specialist_probabilities(models, {k: a[at] for k, a in x.items()}, d['baseline'][at])
        expected = frozen.predict(key, at, set(FOLDS)-set(key))
        need(np.array_equal(p, expected), f'{key}: specialist cached posteriors differ; max abs {float(np.max(np.abs(p-expected)))}')
        q = corrector_probabilities(original/'learned_corrector', records[key], raw[at], d['baseline'][at])
        q0 = corrector.probabilities(key, at, set(FOLDS)-set(key))
        error = float(np.max(np.abs(q-q0)))
        need(error <= POSTERIOR_ATOL, f'{key}: original corrector forward error {error}')
        v = np.concatenate([seven_candidates(p), project_actions(q, d['baseline'][at])[:, None]], 1)
        pred = joint_groups(v, d['baseline'][at])[0]
        reference = joint_groups(catalogue.predict(key, at, set(FOLDS)-set(key)), d['baseline'][at])[0]
        need(np.array_equal(pred, reference), f'{key}: reconstructed group decisions differ')
        weights[cache_key(key)] = models
        r = dict(train_folds=list(key), receiver_rows=len(at),
                 original_training_ids_sha256=frozen.manifests[key]['train_id_sha256'],
                 specialist_probability_max_error=0., corrector_probability_max_error=error,
                 group_decisions_checked=int(pred.size), group_decision_mismatches=0)
        report_rows.append(r)
        print(json.dumps(r), flush=True)
    if saved is None:
        joblib.dump(weights, output/'specialists.joblib', compress=3)
    else:
        (output/'specialists.joblib').write_bytes(saved_specialists.read_bytes())
    report = dict(status='RECOVERED_EIGHT_VOTE_RUNTIME_HISTORICAL_GROUP_DECISIONS_VERIFIED',
        historical_events=7493, original_fold_configurations=14, logistic_models_reconstructed=70,
        original_corrector_checkpoints=14, specialist_cached_probabilities_bit_exact=True,
        maximum_corrector_posterior_error=max(r['corrector_probability_max_error'] for r in report_rows),
        group_decisions_checked=sum(r['group_decisions_checked'] for r in report_rows),
        group_decision_mismatches=0, configurations=report_rows,
        reconstruction_method='Original ProducerPool fixed hyperparameters refit on original manifest training IDs; frozen corrector weights, NumPy GELU forward.',
        corrector_arithmetic='Original float32 inputs and weights, float64 dense/GELU accumulation, float32 posteriors; unchanged 2e-6 archive tolerance and exact group-decision gate.',
        specialists_refitted_this_run=saved is None,
        saved_specialists_sha256=sha(saved_specialists) if saved_specialists is not None else None,
        original_development_training_labels_used=True, receiver_labels_used_for_fit=False,
        corrector_retrained=False, new_audio_labels_read=False,
        source_archive_sha256=source_sha, original_inputs_sha256=sha(root/PREPARED),
        fundamental_input_sha256=sha(fundamentals), reconstructed_models_sha256=sha(output/'specialists.joblib'),
        full_S45_16_families_complete=False, S80_promoted=False)
    write_json(output/'report.json', report)
    return report


def infer(root, runtime, external, s5, out):
    need(not out.exists(), 'refuse to overwrite new-audio group evidence')
    proof = json.loads((runtime/'report.json').read_text())
    need(sha(runtime/'specialists.joblib') == proof['reconstructed_models_sha256'], 'runtime model SHA')
    need(sha(root/PREPARED) == proof['original_inputs_sha256'], 'runtime reference input SHA')
    d = read(root/PREPARED, ('global_index', 'baseline', 'fold', 'features', 'feature_names'))
    with np.load(external, allow_pickle=False) as z:
        need(set(z.files) == {'row_id', 'features', 'fundamental', 'feature_names'}, 'external allowlist excludes labels')
        e = {k: z[k].copy() for k in z.files}
    p = read(s5, ('row_id', 'freeze_K'))
    need(np.array_equal(p['row_id'], e['row_id']), 'new audio row alignment')
    need(np.array_equal(d['feature_names'], e['feature_names']), 'feature column order')
    names = list(map(str, d['feature_names']))
    # These historical families are anchored to freeze_local_combo, NOT S18.
    eligible = np.flatnonzero(np.isin(p['freeze_K'], (2, 3, 4)))
    need(len(eligible) > 0 and len(set(map(str, e['row_id']))) == len(e['row_id']), 'eligible unique external events')
    base = p['freeze_K'][eligible].astype(int)
    x, raw = matrices(e['features'][eligible], e['fundamental'][eligible], names)
    _, ref_raw = matrices(d['features'], d['features'], names)
    original = runtime/'original'
    with np.load(original/'specialists/producers.npz') as z:
        truth = z['truth'].copy()
    frozen = FrozenProducerPool({}, truth, d['baseline'], d['fold'], d['global_index'], original/'specialists')
    corrector = FrozenCorrector(truth, d['baseline'], d['fold'], d['global_index'], original/'learned_corrector')
    catalogue = CataloguePool(frozen, corrector)
    records = {tuple(r['train_folds']): r for r in corrector.manifest['producers']}
    weights = joblib.load(runtime/'specialists.joblib')
    blocks, direct, audits, contexts, source_votes, reports = [], [], [], [], [], []
    receiver_ids = -np.arange(1, len(eligible)+1, dtype=np.int64)
    for outer in FOLDS:
        key = tuple(f for f in FOLDS if f != outer)
        v, _, _ = votes(weights[cache_key(key)], x, raw, base, original/'learned_corrector', records[key])
        proposals, desc, _ = joint_groups(v, base)
        train = np.flatnonzero(d['fold'] != outer)
        ref_votes, _ = catalogue.crossfit(train, {outer})
        ref_proposals = joint_groups(ref_votes, d['baseline'][train])[0]
        local = local_audits(ref_raw[train], raw, ref_proposals, truth[train], d['baseline'][train],
                             proposals, base, d['global_index'][train], receiver_ids)
        ctx, _ = group_context(proposals, local, base)
        blocks.append(proposals); direct.append(desc); audits.append(local); contexts.append(ctx); source_votes.append(v)
        reports.append(dict(outer_branch=outer, original_training_folds=list(key),
            audit_reference_rows=len(train), audit_reference_ids_sha256=id_digest(d['global_index'][train]),
            selected_receiver_labels_used=False,
            mean_distinct_group_destinations=float(np.mean([len(set(a)) for a in proposals]))))
    out.mkdir(parents=True)
    np.savez_compressed(out/'external_255x9.npz', row_id=e['row_id'], eligible_positions=eligible,
        eligible_row_id=e['row_id'][eligible], freeze_K=p['freeze_K'], fold_ids=np.asarray(FOLDS),
        votes=np.asarray(source_votes), proposal=np.asarray(blocks, dtype=np.int8),
        direct_features=np.asarray(direct), group_audits=np.asarray(audits),
        group_context=np.asarray(contexts), acoustic58=e['features'][eligible])
    report = dict(status='RECOVERED_EIGHT_PRODUCERS_EXECUTED_ON_REAL_GOAT_255X9',
        external_events=len(e['row_id']), eligible_events=len(eligible),
        eligible_row_ids=e['row_id'][eligible].tolist(), anchor='freeze_local_combo K2/K3/K4',
        fold_branches=list(FOLDS), group_audits_shape=list(np.asarray(audits).shape),
        original_reference_audits=True, original_reference_training_labels_used=True,
        new_audio_annotations_read=False, branches=reports,
        external_input_sha256=sha(external), external_prediction_sha256=sha(out/'external_255x9.npz'),
        cross_fold_aggregation_selected=False, GOAT_is_exposed_smoke_data=True,
        full_S1_S4_risk_logits_completed=False, full_S45_16_families_complete=False,
        independent_ExactK_measured=False, S80_promoted=False)
    write_json(out/'report.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'extract', 'recover', 'infer'), required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--fundamentals', type=Path)
    p.add_argument('--saved-specialists', type=Path)
    p.add_argument('--saved-recovery-report', type=Path)
    p.add_argument('--archives', type=Path)
    p.add_argument('--s97', type=Path)
    p.add_argument('--audio', type=Path)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--external', type=Path)
    p.add_argument('--s5', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.stage == 'prepare':
        need(a.archives is not None, 'original harmonic archives required')
        prepare_fundamentals(a.root, a.archives, a.out)
    elif a.stage == 'extract':
        need(a.s97 is not None and a.audio is not None, 'pinned WAVs and S97 groups required')
        extract_goat(a.s97, a.audio, a.out)
    elif a.stage == 'recover':
        need(a.fundamentals is not None, 'original fundamental features required')
        recover(a.root, a.fundamentals, a.out, a.saved_specialists, a.saved_recovery_report)
    else:
        need(all(v is not None for v in (a.runtime, a.external, a.s5)), 'sealed runtime/acoustic/S5 inputs required')
        infer(a.root, a.runtime, a.external, a.s5, a.out)
