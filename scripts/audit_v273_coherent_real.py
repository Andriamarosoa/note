"""Frozen phase-preserving prediction diagnostic on one internal partition.

Consumes saved native-model predictions only for error stratification. Does not
train a network, change a count, or read outer-fold audio.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from causal_note.guitarset import index_guitarset, load_boundary_slots
from scripts.audit_v273_acoustic_residual import ownership
from scripts.probe_v273_coherent_decay import CONFIGS, LONG_HISTORY, RCOND, forecast
from scripts.rebuild_v273_sources import digest, verify_dataset, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.train_boundaries import group_stem
from scripts.train_v100_spectral_string_slots import (
    PRE_SAMPLES, POST_SAMPLES, _pcm_window, _spectral_map_from_segment,
    decode_pcm16_mono_wav)
from scripts.v273_decay_native_inputs import native_decay_features
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import partitions, require

METHODS = ['legacy_log_power', 'spectral_flux', 'zero_wave', *CONFIGS]
PROTOCOL = Path('analysis/v273-coherent-real-protocol.md')
SOURCE = '8f77db192b6274587965de54cac9bf1a1a84ac25'


def npy_hash(array):
    # Bundle field digests include the .npy header; they are not array_hash.
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def measure(wave):
    x = np.asarray(wave, np.float32)
    require(x.shape == (LONG_HISTORY + POST_SAMPLES,), 'wrong waveform support')
    require(np.isfinite(x).all(), 'invalid PCM')
    past = x[LONG_HISTORY-PRE_SAMPLES:LONG_HISTORY].astype(float)
    future = x[LONG_HISTORY:].astype(float)
    scale = max(float(np.mean(past**2)), 1e-12)
    spectral = _spectral_map_from_segment(x[LONG_HISTORY-PRE_SAMPLES:]).astype(np.float16)
    features, audit = native_decay_features(spectral[None])
    result = dict(legacy_log_power=float(features.max()),
        spectral_flux=float(spectral[9:, :, 2].max()),
        zero_wave=float(np.sqrt(np.mean(future**2)/scale)),
        pre_rms=float(np.sqrt(np.mean(past**2))), post_rms=float(np.sqrt(np.mean(future**2))),
        legacy_reliable_bands=int(audit['reliable'].sum()))
    for name, cfg in CONFIGS.items():
        try:
            prediction, info = forecast(x[LONG_HISTORY-cfg['history']:LONG_HISTORY],
                                       POST_SAMPLES, order=cfg['order'], lag=cfg['lag'])
            with np.errstate(over='raise', invalid='raise'):
                result[name] = float(np.sqrt(np.mean((future-prediction)**2)/scale))
                result[name+'_forecast_rms_ratio'] = float(np.sqrt(np.mean(prediction**2)/scale))
            result[name+'_rank'] = info['rank']
            result[name+'_coefficient_l2'] = info['coefficient_l2']
            result[name+'_failed'] = 0
        except (FloatingPointError, np.linalg.LinAlgError):
            result[name] = np.nan
            result[name+'_forecast_rms_ratio'] = np.nan
            result[name+'_rank'] = -1
            result[name+'_coefficient_l2'] = np.nan
            result[name+'_failed'] = 1
    return result


def event_context(groups, events):
    assigned, exact, eligible, _ = ownership(groups, events[:, 0])
    result = []
    for i, group in enumerate(groups):
        start = int(group[0]); end = start + POST_SAMPLES
        own = assigned == i
        future = (events[:, 0] >= start) & (events[:, 0] < end)
        result.append(dict(k=int(exact[i]), future_births=int(future.sum()),
            owned_before=int(np.sum(own & (events[:, 0] < start))),
            owned_future=int(np.sum(own & future)),
            owned_after=int(np.sum(own & (events[:, 0] >= end))),
            foreign_future=int(np.sum(future & ~own)),
            local_eligible_births=int(eligible[i].sum()),
            contested_births=int(np.sum(eligible[i] & ~own)),
            overlapping_notes=int(np.sum((events[:, 0] < end) & (events[:, 1] > start)))))
    return result


def process_track(task):
    track, path, global_ids, starts, labels, predictions, output = task
    began = time.monotonic()
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['global_index'], global_ids)
        require(str(z['member'][0]) == track.annotation_member, 'timing track mismatch')
        offsets = z['full_candidate_offsets']; full = z['full_candidate_samples']
        require(offsets[0] == 0 and offsets[-1] == len(full) and np.all(np.diff(offsets) > 0),
                'invalid full timestamps')
        groups = [full[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    np.testing.assert_array_equal(starts, [g[0] for g in groups])
    require(all(np.all(np.diff(g) >= 0) for g in groups), 'unsorted candidates')
    events = np.asarray([(b.onset_sample, b.offset_sample)
        for slot in load_boundary_slots(track.annotation_zip, track.annotation_member)
        for b in slot], np.int64)
    context = event_context(groups, events)
    np.testing.assert_array_equal(labels, [r['k'] for r in context])
    audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
    samples = np.asarray(audio.samples, np.float32)/32768.
    records = []
    for i, (idx, start) in enumerate(zip(global_ids, starts)):
        wave = _pcm_window(samples, int(start)-LONG_HISTORY, LONG_HISTORY+POST_SAMPLES)
        row = dict(global_index=int(idx), member=track.annotation_member,
            composition=group_stem(track.annotation_member), start=int(start),
            padded=int(start < LONG_HISTORY or start+POST_SAMPLES > len(samples)),
            predicted_local_only=int(predictions[0][i]),
            predicted_with_neighbors=int(predictions[1][i]), **context[i], **measure(wave))
        records.append(row)
    target = Path(output)/path.name.replace('.npz', '.json')
    write_json(target, records)
    return dict(member=track.annotation_member, rows=len(records),
        annotations=len(events), assigned_onsets=int(labels.sum()),
        seconds=time.monotonic()-began, records_file=target.name,
        timing_sha256=digest(path))


def auc(score, label, weight=None):
    """Weighted rank AUC with half credit for ties; no score sign selection."""
    score = np.asarray(score, float); label = np.asarray(label, bool)
    weight = np.ones(len(score)) if weight is None else np.asarray(weight, float)
    finite = np.isfinite(score)
    score, label, weight = score[finite], label[finite], weight[finite]
    positive = weight[label].sum(); negative = weight[~label].sum()
    if not positive or not negative:
        return None
    order = np.argsort(score, kind='stable')
    score, label, weight = score[order], label[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score) != 0)+1]
    p = np.add.reduceat(weight*label, starts)
    n = np.add.reduceat(weight*(~label), starts)
    return float(np.sum(p*(np.cumsum(n)-n/2))/(positive*negative))


def describe(values):
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    return dict(rows=len(values), failed=int(len(values)-len(valid)),
        median=float(np.median(valid)) if len(valid) else None,
        q10=float(np.quantile(valid, .1)) if len(valid) else None,
        q90=float(np.quantile(valid, .9)) if len(valid) else None,
        maximum=float(valid.max()) if len(valid) else None)


def compare(data, mask, label):
    return dict(rows=int(mask.sum()), positives=int(np.sum(mask & label)),
        negatives=int(np.sum(mask & ~label)),
        auc={m:auc(data[m][mask], label[mask]) for m in METHODS})


def error_analysis(data, arm):
    k = data['k']; pred = data['predicted_'+arm]; members = data['member']; start = data['start']
    over = (k < 4) & (pred > k); correct = pred == k
    result = dict(overcounts=int(over.sum()), exact=int(correct.sum()), by_true_k={},
        by_predicted_k={}, matched={})
    for value in range(4):
        same = k == value
        result['by_true_k'][str(value)] = dict(rows=int(same.sum()),
            overcounts=int(np.sum(same & over)), correct=int(np.sum(same & correct)),
            over_without_future_birth=int(np.sum(same & over & (data['future_births'] == 0))),
            over_with_contested_birth=int(np.sum(same & over & (data['contested_births'] > 0))),
            scores={m:dict(over=describe(data[m][same & over]),
                           correct=describe(data[m][same & correct])) for m in METHODS})
        errors, controls = [], []
        for member in np.unique(members):
            er = np.flatnonzero((members == member) & same & over)
            co = np.flatnonzero((members == member) & same & correct)
            if len(co):
                for row in er:
                    errors.append(row)
                    controls.append(co[np.argmin(np.abs(start[co]-start[row]))])
        errors = np.asarray(errors, int); controls = np.asarray(controls, int)
        result['matched'][str(value)] = dict(pairs=len(errors), unique_controls=len(set(controls)),
            scores={m:dict(median_error_minus_correct=describe(data[m][errors]-data[m][controls])['median'],
                fraction_error_higher=float(np.mean(data[m][errors] > data[m][controls])) if len(errors) else None,
                finite_pairs=int(np.sum(np.isfinite(data[m][errors]) & np.isfinite(data[m][controls])))) for m in METHODS})
    for value in range(1, 7):
        mask = (pred == value) & (correct | over)
        result['by_predicted_k'][str(value)] = compare(data, mask, correct)
    return result


def summarize(records):
    data = {key:np.asarray([r[key] for r in records]) for key in records[0]}
    k = data['k']; all_rows = np.ones(len(k), bool); novelty = data['future_births'] > 0
    result = dict(rows=len(k), tracks=len(set(data['member'])), compositions=len(set(data['composition'])),
        by_true_k={str(v):int(np.sum(k == v)) for v in range(7)},
        novelty=compare(data, all_rows, novelty),
        novelty_unpadded=compare(data, data['padded'] == 0, novelty),
        any_owned=compare(data, all_rows, k > 0), poly=compare(data, all_rows, k >= 2),
        adjacent={str(v)+'_vs_'+str(v-1):compare(data, (k == v) | (k == v-1), k == v)
                  for v in range(1, 5)},
        errors={arm:error_analysis(data, arm) for arm in ('local_only', 'with_neighbors')},
        per_track={m:compare(data, data['member'] == m, novelty) for m in sorted(set(data['member']))},
        temporal_coverage=dict(padded_rows=int(data['padded'].sum()),
            owned_onsets=int(k.sum()), owned_before=int(data['owned_before'].sum()),
            owned_future=int(data['owned_future'].sum()), owned_after=int(data['owned_after'].sum()),
            rows_with_owned_before=int(np.sum(data['owned_before'] > 0)),
            rows_with_owned_after=int(np.sum(data['owned_after'] > 0)),
            k_positive_no_future_birth=int(np.sum((k > 0) & ~novelty))),
        diagnostics={name:dict(failed_rows=int(data[name+'_failed'].sum()),
            forecast_rms_ratio=describe(data[name+'_forecast_rms_ratio']),
            forecast_above_ten_times_past=int(np.sum(data[name+'_forecast_rms_ratio'] > 10)),
            no_future_birth_score=describe(data[name][~novelty])) for name in CONFIGS})
    groups, index = np.unique(data['composition'], return_inverse=True)
    rng = np.random.default_rng(9302026); differences = []
    wide = 'wave_long_wider_recurrence'
    for _ in range(2000):
        weights = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))[index]
        a = auc(data[wide], novelty, weights); b = auc(data['zero_wave'], novelty, weights)
        if a is not None and b is not None:
            differences.append(a-b)
    interval = np.quantile(differences, [.025, .975]).tolist()
    adjacent_gain = {key:row['auc'][wide]-row['auc']['zero_wave']
                     for key,row in result['adjacent'].items()}
    result['primary_composition_bootstrap'] = dict(seed=9302026, draws=2000,
        valid_draws=len(differences), composition_groups=groups.tolist(),
        comparison=wide+' minus zero_wave',
        observed_difference=result['novelty']['auc'][wide]-result['novelty']['auc']['zero_wave'],
        percentile_95=interval)
    result['training_gate'] = dict(passed=bool(interval[0] > 0 and adjacent_gain['2_vs_1'] > 0
        and adjacent_gain['3_vs_2'] > 0 and all(not result['diagnostics'][n]['failed_rows'] for n in CONFIGS)),
        adjacent_auc_difference=adjacent_gain, native_training_launched=False)
    return result


def run(args):
    require(not args.output.exists(), 'refusing to overwrite audit')
    verify_dataset(args.dataset)
    for root in (args.geometry, args.local_only, args.with_neighbors):
        validate_original_inventory(root)
    cfg = load_config(args.config)
    require(digest(args.config) == '7c27989eb3fe330fdff203e388e76ae4c64d8b8e4838cab4d7c4622b4104451a',
            'partition configuration changed')
    with np.load(args.geometry/'rows.npz', allow_pickle=False) as z:
        members = z['member']; starts = z['start']
    geometry_report = json.loads((args.geometry/'report.json').read_text())
    for key, array in (('members', members), ('cluster_start_samples', starts)):
        require(npy_hash(array) == geometry_report['bundle_fields'][key]['sha256'], 'row identity changed')
    ids = partitions(members, cfg)['inner_val']
    require(len(ids) == 15952, 'wrong internal validation row count')
    saved = []; sources = {}
    for arm in ('local_only', 'with_neighbors'):
        path = getattr(args, arm)/'epoch-12-validation.npz'
        with np.load(path, allow_pickle=False) as z:
            saved.append({key:np.asarray(z[key]) for key in z.files})
        item = saved[-1]
        np.testing.assert_array_equal(item['global_index'], ids)
        np.testing.assert_array_equal(item['member'], members[ids])
        np.testing.assert_array_equal(item['probability'].argmax(1), item['predicted'])
        require(np.isfinite(item['probability']).all(), 'nonfinite source probability')
        np.testing.assert_allclose(item['probability'].sum(1), 1, atol=1e-6)
        sources[arm] = dict(prediction_sha256=digest(path), report_sha256=digest(getattr(args, arm)/'report.json'))
    np.testing.assert_array_equal(saved[0]['k'], saved[1]['k'])
    require(int(np.sum(saved[0]['k'] >= 2)) == 2111, 'wrong poly population')
    allowed = {m for m, fold in cfg['member_folds'].items() if fold == 0}
    require(set(members[ids]) == allowed and len(allowed) == 50, 'wrong internal tracks')
    paths = {}
    for path in sorted((args.geometry/'full-timing').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            name = str(z['member'][0])
        if name in allowed:
            require(name not in paths, 'duplicate timing track')
            paths[name] = path
    require(set(paths) == allowed, 'missing timing')
    tracks = {t.annotation_member:t for t in index_guitarset(args.dataset) if t.annotation_member in allowed}
    args.output.mkdir(parents=True); temporary = args.output/'tracks'; temporary.mkdir()
    started = time.monotonic()
    frozen = dict(protocol_sha256=digest(PROTOCOL), script_sha256=digest(__file__),
        predictor_sha256=digest('scripts/probe_v273_coherent_decay.py'),
        predictor_source_commit=SOURCE, configs=CONFIGS, rcond=RCOND,
        config_sha256=digest(args.config), geometry_report_sha256=digest(args.geometry/'report.json'),
        geometry_inventory_sha256=digest(args.geometry/'file-sha256.json'), sources=sources,
        dataset_md5={p.name:digest(p,'md5') for p in (args.dataset/'annotation.zip', args.dataset/'audio_mono-pickup_mix.zip')},
        numpy=np.__version__, workers=args.workers, outer_rows_evaluated=0,
        scope='all internal validation rows for outer fold 3; composition fold 0',
        no_output_correction=True, no_new_model_inference=True, training_launched=False)
    write_json(args.output/'frozen-inputs.json', frozen)
    tasks = []
    for member in sorted(allowed):
        selected = np.flatnonzero(members[ids] == member)
        global_ids = ids[selected]
        tasks.append((tracks[member], paths[member], global_ids, starts[global_ids], saved[0]['k'][selected],
                      [s['predicted'][selected] for s in saved], str(temporary)))
    records = []; track_reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_track, task) for task in tasks]
        for future in as_completed(futures):
            info = future.result(); track_reports.append(info)
            records += json.loads((temporary/info['records_file']).read_text())
            print(json.dumps(dict(tracks=len(track_reports), rows=len(records), member=info['member'],
                                  elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
    records.sort(key=lambda r:r['global_index'])
    np.testing.assert_array_equal([r['global_index'] for r in records], ids)
    output = args.output/'rows.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    report = dict(status='completed', **frozen, **summarize(records),
        track_replay=sorted(track_reports, key=lambda r:r['member']),
        rows_sha256=digest(output), elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'report.json', report)
    print(json.dumps(dict(completed=True, novelty=report['novelty'],
                         training_gate=report['training_gate'], bootstrap=report['primary_composition_bootstrap'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'geometry', 'local-only', 'with-neighbors', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    run(p.parse_args())
