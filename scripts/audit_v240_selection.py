"""Replay canonical V24 outer predictions; audit counting versus selection, no fit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import wave
import zipfile

import numpy as np

from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v177_candidate_centric as v177
from scripts import train_v240_categorical_k_candidate_subset as v240
from scripts import train_v92_string_factorized_cardinality as v92
from scripts.evaluate_v8_boundaries import _reference_positions
from scripts.evaluate_boundaries import match_boundaries, _count_metrics
from scripts.train_v91_ordinal_cardinality import _dataset_split

SOURCE_RUN = 34168832668
SOURCE_SHA = '980a052e6cd9eee7896962ceb3a458e0ae258334'
OUTER_ROWS = 76768
COUNT_KEYS = ('true_positive', 'false_positive', 'false_negative', 'reference_count', 'prediction_count')


def score(refs, predictions, tolerance):
    nr = npred = tp = 0
    for member, reference in refs.items():
        values = predictions.get(member, ())
        nr += len(reference)
        npred += len(values)
        tp += len(match_boundaries(reference, values, tolerance))
    return asdict(_count_metrics(nr, npred, tp))


def aggregate(rows):
    return asdict(_count_metrics(sum(r['reference_count'] for r in rows),
                                sum(r['prediction_count'] for r in rows),
                                sum(r['true_positive'] for r in rows)))


def ranked_ids(rank, mask, counts):
    ids = np.full((len(rank), 6), -1, dtype=np.int32)
    for row in range(len(rank)):
        valid = np.flatnonzero(mask[row] > .5)
        order = valid[np.argsort(-rank[row, valid], kind='stable')]
        take = min(int(counts[row]), len(order), 6)
        ids[row, :take] = order[:take]
    return ids


def compare_maps(a, b):
    members = set(a) | set(b)
    return {'tracks': len(members), 'tracks_with_different_timestamps': sum(a.get(m, ()) != b.get(m, ()) for m in members),
            'tracks_with_different_counts': sum(len(a.get(m, ())) != len(b.get(m, ())) for m in members)}


def audit(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cache = v100._load_spectral_caches(args.cache_dir)
    _, tracks, locked = _dataset_split(args.dataset_dir)
    allowed = {t.annotation_member for t in tracks}
    if set(cache['members']) != allowed or allowed & {t.annotation_member for t in locked}:
        raise RuntimeError('cache/split mismatch')
    samples, reconstruction = v102._reconstruct_candidates(cache)
    # Detect whether the inherited sort/compaction preserves candidate IDs.
    id_order_bad = 0
    width_ms = []
    for row, (seq, mask) in enumerate(zip(cache['sequence'], cache['mask'])):
        ids, rel = v92._candidate_relative_samples(seq, mask)
        if not np.array_equal(ids, np.arange(len(ids))) or np.any(np.diff(rel) < 0):
            id_order_bad += 1
        width_ms.append(float(np.ptp(rel)) * 1000 / v102.SAMPLE_RATE if len(rel) else 0.)
    refs = {}
    # Same frame count and reference helper as production; read WAV headers only.
    for track in tracks:
        with zipfile.ZipFile(track.audio_zip) as z, z.open(track.audio_member) as stream, wave.open(stream) as wav:
            frame_count = wav.getnframes()
        refs[track.annotation_member] = _reference_positions(track, frame_count)[0]
    by_fold, source_files, all_idx, seen_members = [], [], [], set()
    for fold in range(5):
        hits = list(args.fold_dir.glob(f'**/report-fold-{fold}.json'))
        if len(hits) != 1:
            raise RuntimeError(f'fold {fold}: expected one report')
        rp = hits[0]; pp = rp.with_name(f'predictions-fold-{fold}.npz')
        report = json.loads(rp.read_text())
        p = report['protocol']
        if not p.get('candidate_subset_supervision_requires_complete_match') or p['candidate_target_tolerance_ms'] != 50.0:
            raise RuntimeError('not corrected V24 artifacts')
        if p['historical_validation_or_locked12_indexed_or_evaluated'] is not False:
            raise RuntimeError('invalid fold protocol')
        with np.load(pp, allow_pickle=False) as z:
            data = {k: np.asarray(z[k]) for k in ('global_index', 'outer_fold', 'member', 'k', 'categorical_k', 'candidate_selected_ids', 'candidate_rank_distribution', 'candidate_valid_count', 'candidate_subset_target')}
        idx = data['global_index'].astype(np.int64)
        if np.any(data['outer_fold'] != fold) or not np.array_equal(data['member'].astype(str), cache['members'][idx].astype(str)):
            raise RuntimeError('fold/member alignment mismatch')
        if not np.array_equal(data['k'], np.minimum(cache['exact'][idx], 6)):
            raise RuntimeError('annotation cardinality mismatch')
        members = set(data['member'].astype(str))
        if members & seen_members or not members <= allowed:
            raise RuntimeError('outer member overlap or unauthorized member')
        seen_members |= members; all_idx.extend(idx.tolist())
        local_refs = {m: refs[m] for m in members}
        selected = data['candidate_selected_ids'].astype(np.int32)
        mask = cache['mask'][idx]
        if not np.array_equal(data['candidate_valid_count'], np.sum(mask > .5, axis=1)):
            raise RuntimeError('candidate mask mismatch')
        for row, ids in enumerate(selected):
            valid_ids = ids[ids >= 0]
            if len(set(valid_ids)) != len(valid_ids) or np.any(valid_ids >= mask.shape[1]) or np.any(mask[row, valid_ids] <= .5):
                raise RuntimeError('invalid selected candidate identity')
        count = np.sum(selected >= 0, axis=1)
        rank = data['candidate_rank_distribution']
        if not np.all(np.isfinite(rank)):
            raise RuntimeError('nonfinite saved candidate ranking')
        actual_map = v177._direct_prediction_map(cache, samples, idx, selected, np.ones(len(idx), bool))
        frozen_map = v100._cached_prediction_map(cache, idx, count)
        replay50 = score(local_refs, actual_map, 2205)
        recorded = report['strata']['aggregate'][v240.MODEL_KEY]['metrics']['global']
        if any(replay50[k] != recorded[k] for k in COUNT_KEYS):
            raise RuntimeError(f'fold {fold}: selected IDs do not reproduce published counts')
        frozen50 = score(local_refs, frozen_map, 2205)
        frozen_recorded = report['v240']['same_realized_count_frozen_ranking_strata']['aggregate']['global']
        if any(frozen50[k] != frozen_recorded[k] for k in COUNT_KEYS):
            raise RuntimeError('frozen replay mismatch')
        true_k = data['k'].astype(np.int32)
        learned_true_ids = ranked_ids(rank, mask, true_k)
        learned_pred_ids = ranked_ids(rank, mask, count)
        rank_replay_map = v177._direct_prediction_map(cache, samples, idx, learned_pred_ids, np.ones(len(idx), bool))
        oracle_map = v177._direct_prediction_map(cache, samples, idx, learned_true_ids, np.ones(len(idx), bool))
        frozen_oracle = v100._cached_prediction_map(cache, idx, true_k)
        reverse = ranked_ids(-rank, mask, count)
        reverse_map = v177._direct_prediction_map(cache, samples, idx, reverse, np.ones(len(idx), bool))
        tolerance_metrics = {str(ms): {'direct': score(local_refs, actual_map, round(ms * v102.SAMPLE_RATE / 1000)),
                                     'frozen_same_count': score(local_refs, frozen_map, round(ms * v102.SAMPLE_RATE / 1000)),
                                     'reversed_same_count': score(local_refs, reverse_map, round(ms * v102.SAMPLE_RATE / 1000))}
                             for ms in (5, 10, 20, 50)}
        poly = true_k >= 2
        by_fold.append({'fold': fold, 'rows': len(idx), 'selected_epochs': report['data']['selected_epochs'],
                        'v104': report['strata']['aggregate']['v104']['metrics']['global'],
                        'direct': replay50, 'frozen_same_count': frozen50,
                        'oracle_true_k_learned_ranking': score(local_refs, oracle_map, 2205),
                        'oracle_true_k_frozen_ranking': score(local_refs, frozen_oracle, 2205),
                        'maps_direct_vs_frozen': compare_maps(actual_map, frozen_map),
                        'saved_rank_replay_vs_actual': compare_maps(rank_replay_map, actual_map),
                        'ranking_replay_different_selected_sets': sum(set(a[a>=0]) != set(b[b>=0]) for a,b in zip(selected,learned_pred_ids)),
                        'tolerance_diagnostics_ms': tolerance_metrics,
                        'poly_rows': int(poly.sum()), 'poly_k_correct': int(np.sum(data['categorical_k'][poly] == true_k[poly])),
                        'k_confusion': np.bincount(true_k * 7 + data['categorical_k'], minlength=49).reshape(7,7).tolist(),
                        'source_supervision': report['data']['anonymous_event_targets']})
        source_files.extend({'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in (rp,pp))
        print(json.dumps({'fold_completed': fold, 'replayed_f1': replay50['f1'], 'oracle_learned_f1': by_fold[-1]['oracle_true_k_learned_ranking']['f1']}), flush=True)
    if sorted(all_idx) != list(range(OUTER_ROWS)) or seen_members != allowed:
        raise RuntimeError('five-fold coverage is not exact')
    metric_names = ('v104', 'direct', 'frozen_same_count', 'oracle_true_k_learned_ranking', 'oracle_true_k_frozen_ranking')
    totals = {key: aggregate([f[key] for f in by_fold]) for key in metric_names}
    tolerance_totals = {str(ms): {name: aggregate([f['tolerance_diagnostics_ms'][str(ms)][name] for f in by_fold])
                                for name in ('direct','frozen_same_count','reversed_same_count')} for ms in (5,10,20,50)}
    output = {'protocol': {'analysis_only': True, 'training': False, 'threshold_tuning': False,
                           'source_run': SOURCE_RUN, 'source_sha': SOURCE_SHA, 'outer_rows': len(all_idx),
                           'historical_validation_or_locked12_evaluated': False,
                           'oracle_is_diagnostic_only': True, 'official_tolerance_ms': 50,
                           'saved_ranking_ties': 'stable candidate index; replay differences explicitly reported'},
              'aggregate': totals, 'tolerance_diagnostics_ms': tolerance_totals,
              'identity_order_bad_rows': id_order_bad, 'reconstruction': reconstruction,
              'cluster_width_ms': {'max': max(width_ms), 'median': float(np.median(width_ms))},
              'poly_k_exact': sum(f['poly_k_correct'] for f in by_fold) / sum(f['poly_rows'] for f in by_fold),
              'maps_tracks_changed': sum(f['maps_direct_vs_frozen']['tracks_with_different_timestamps'] for f in by_fold),
              'k_confusion': np.sum([f['k_confusion'] for f in by_fold], axis=0).tolist(),
              'folds': by_fold, 'source_files': source_files}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'report.json').write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
    lines = ['# V24 corrected selection audit', '', f'Source run: {SOURCE_RUN}; source commit: `{SOURCE_SHA}`.', '',
             'Five outer folds, exact coverage; no training or locked-set evaluation.', '',
             '| Diagnostic | Global F1 | TP | FP | FN |', '|---|---:|---:|---:|---:|']
    for name, r in totals.items():
        lines.append(f"| {name} | {r['f1']:.8f} | {r['true_positive']} | {r['false_positive']} | {r['false_negative']} |")
    lines += ['', 'Oracle true-K results are diagnostic ceilings, not deployable models.',
              f'Candidate ID ordering violations: {id_order_bad}.',
              f"Tracks with changed direct/frozen timestamps: {output['maps_tracks_changed']}.",
              f"Polyphonic categorical K exact: {output['poly_k_exact']:.6f}."]
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('folds','source_files')}, indent=2, sort_keys=True), flush=True)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--fold-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    audit(p.parse_args())
