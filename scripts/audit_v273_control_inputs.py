"""Check fold-3 labels, candidate reconstruction and acoustic-window coverage.

No training, inference on another outer fold, or annotation-dependent decoder.
Timing associations are observational and are not reported as causal proof.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

from causal_note.guitarset import SAMPLE_RATE, load_boundary_slots
from scripts.audit_v273_control_fold3 import metrics
from scripts.audit_v273_native_paired_results import load_npz, require
from scripts import train_v92_string_factorized_cardinality as slots
from scripts import train_v100_spectral_string_slots as spectral
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_native_protocol import load_config, verify_native_cache


def nearest_assignment(onset, samples, row_ids, radius=882):
    """Use the original nearest-candidate, lowest-row-index tie rule."""
    left = np.searchsorted(samples, onset - radius, side='left')
    right = np.searchsorted(samples, onset + radius, side='right')
    if left == right:
        return None
    distance = np.abs(samples[left:right].astype(np.int64) - onset)
    minimum = int(distance.min())
    return int(row_ids[left:right][distance == minimum].min()), minimum


def load_outer_metadata(directory, cfg):
    keys = ('sequence', 'mask', 'stats', 'exact', 'members', 'top_samples', 'slot_targets')
    parts = {key: [] for key in keys}
    indices, offset = [], 0
    hist = {'inner_fit': np.zeros(7, np.int64), 'inner_validation': np.zeros(7, np.int64),
            'final_fit': np.zeros(7, np.int64)}
    checks = {'finite_spectral': True, 'finite_model_inputs': True,
              'nonempty_candidate_masks': True, 'nonzero_spectral_rows': 0,
              'spectral_shapes': []}
    for path in sorted(Path(directory).rglob('v100-spectral-shard-*.npz')):
        with np.load(path, allow_pickle=False) as z:
            members = z['members'].astype(str)
            fold = np.asarray([cfg['member_folds'][m] for m in members])
            k = np.minimum(z['exact'].astype(int), 6)
            for name, mask in [('inner_fit', (fold != 3) & (fold != 0)),
                               ('inner_validation', fold == 0), ('final_fit', fold != 3)]:
                hist[name] += np.bincount(k[mask], minlength=7)
            take = np.flatnonzero(fold == 3)
            indices.append(offset + take)
            offset += len(members)
            for key in keys:
                value = z[key][take]
                parts[key].append(value)
                if key in ('sequence', 'stats'):
                    checks['finite_model_inputs'] &= bool(np.isfinite(value).all())
            mask = z['mask'][take]
            checks['nonempty_candidate_masks'] &= bool(np.all(mask.sum(1) > 0))
            maps = z['spectral'][take]
            checks['finite_spectral'] &= bool(np.isfinite(maps).all())
            checks['nonzero_spectral_rows'] += int(np.any(maps != 0, axis=(1, 2, 3)).sum())
            checks['spectral_shapes'].append(list(maps.shape))
    require(offset == cfg['expected_rows'], 'cache row count changed')
    require(all(checks[name] for name in ('finite_spectral', 'finite_model_inputs',
                                        'nonempty_candidate_masks')), 'invalid model input')
    return ({key: np.concatenate(value) for key, value in parts.items()},
            np.concatenate(indices), {name: value.tolist() for name, value in hist.items()}, checks)


def audit_annotations(cache, annotation_zip):
    """Audit all annotations of the outer-fold tracks against retained candidates."""
    candidate_samples, reconstruction = slots._reconstruct_candidates(cache)
    n = len(candidate_samples)
    assigned = np.zeros(n, np.int32)
    occupied = np.zeros((n, 6), np.int16)
    outside_map = np.zeros(n, np.int32)
    near_assignment_boundary = np.zeros(n, np.int32)
    max_relative = np.full(n, -1, np.int32)
    starts = np.asarray([min(x) for x in candidate_samples], np.int64)
    by_member = defaultdict(list)
    for i, member in enumerate(cache['members']):
        by_member[str(member)].append(i)
    total = missed = compared = 0
    assignments = []
    for member, rows in sorted(by_member.items()):
        samples = np.concatenate([candidate_samples[i] for i in rows])
        row_ids = np.concatenate([np.full(len(candidate_samples[i]), i, np.int64) for i in rows])
        order = np.argsort(samples, kind='stable')
        samples, row_ids = samples[order], row_ids[order]
        for slot, boundaries in enumerate(load_boundary_slots(annotation_zip, member)):
            for note in boundaries:
                total += 1
                onset = int(note.onset_sample)
                actual = nearest_assignment(onset, samples, row_ids)
                # Independent full-array search audits the accelerated search and tie-break.
                distances = np.abs(samples.astype(np.int64) - onset)
                minimum = int(distances.min())
                expected = None if minimum > slots.LOCAL_RADIUS_SAMPLES else (
                    int(row_ids[distances == minimum].min()), minimum)
                require(actual == expected, 'annotation assignment implementation disagreement')
                compared += 1
                if actual is None:
                    missed += 1
                    continue
                row, distance = actual
                relative = onset - starts[row]
                outside = relative < -spectral.PRE_SAMPLES or relative >= spectral.POST_SAMPLES
                assigned[row] += 1
                occupied[row, slot] += 1
                outside_map[row] += int(outside)
                near_assignment_boundary[row] += int(abs(distance - slots.LOCAL_RADIUS_SAMPLES) <= 3)
                max_relative[row] = max(max_relative[row], relative)
                assignments.append((row, slot, onset, distance, relative, int(outside)))
    return {
        'reconstructed_exact': assigned, 'reconstructed_occupancy': (occupied > 0).sum(1),
        'same_string_extra_onsets': np.maximum(occupied - 1, 0).sum(1),
        'outside_spectral_window_events': outside_map,
        'near_assignment_boundary_events': near_assignment_boundary,
        'cluster_start_sample': starts, 'max_relative_onset_sample': max_relative,
    }, {
        'reconstruction': reconstruction, 'reference_onsets': total,
        'unassigned_reference_onsets': missed, 'assigned_reference_onsets': total - missed,
        'independent_nearest_assignment_comparisons': compared,
        'independent_assignment_equal': True,
        'annotation_sha256': digest(annotation_zip),
        'limitation': 'Candidates are reconstructed from retained float16 features; discrepancies with pre-truncation exact labels do not alone prove incorrect annotations. Audio duration filtering is not replayed.',
    }, np.asarray(assignments, dtype=np.int64).reshape(-1, 6)


def run(args):
    require(args.fold == 3, 'only outer fold 3 is authorized')
    require(not args.output_dir.exists(), 'output already exists')
    expected_md5 = 'b39b78e63d3446f2e54ddb7a54df9b10'
    require(hashlib.md5(args.annotation_zip.read_bytes()).hexdigest() == expected_md5,
            'annotation archive differs from the established GuitarSet source')
    cfg = load_config(args.config)
    verified = verify_native_cache(args.cache_dir, args.config)
    cache, index, hist, input_checks = load_outer_metadata(args.cache_dir, cfg)
    predictions = load_npz(args.decision_dir / 'decision-audit.npz')
    require(np.array_equal(index, predictions['global_index']), 'input/prediction indices differ')
    require(np.array_equal(cache['members'], predictions['member']), 'input members differ')
    k, final = predictions['k'], predictions['final']
    require(np.array_equal(np.minimum(cache['exact'], 6), k), 'input count labels differ')
    arrays, annotations, events = audit_annotations(cache, args.annotation_zip)
    slot_k = cache['slot_targets'].sum(1)
    candidate_count = cache['mask'].sum(1)
    conditions = {
        'cached_slot_count_differs_from_count_target': slot_k != k,
        'reconstructed_count_differs_from_count_target': np.minimum(arrays['reconstructed_exact'], 6) != k,
        'reconstructed_slots_differ_from_cached_slot_count': arrays['reconstructed_occupancy'] != slot_k,
        'same_string_multiple_onsets': arrays['same_string_extra_onsets'] > 0,
        'annotated_onset_outside_spectral_window': arrays['outside_spectral_window_events'] > 0,
        'candidate_count_smaller_than_k': candidate_count < k,
        'candidate_mask_at_48_limit': candidate_count == 48,
        'annotation_near_20ms_boundary': arrays['near_assignment_boundary_events'] > 0,
    }
    report = {
        'scope': {'outer_fold': 3, 'evaluated_rows': len(k), 'training': False,
                  'outer_labels_used_for_model_selection': False, 'model_modified': False},
        'inputs': verified, 'input_checks': input_checks, 'training_label_histograms': hist,
        'annotations': annotations,
        'spectral_window_relative_to_first_retained_candidate_ms': [
            -spectral.PRE_SAMPLES * 1000 / SAMPLE_RATE, spectral.POST_SAMPLES * 1000 / SAMPLE_RATE],
        'source_definitions': {'candidate_group_ms': 40, 'onset_assignment_radius_ms': 20,
                               'k': 'annotated onset count clipped at 6',
                               'spectral_input_is_not_the_only_model_input': True},
        'associations_not_causal_proof': {
            name: {'present': metrics(k[mask], final[mask]),
                   'absent': metrics(k[~mask], final[~mask]),
                   'global_indices': index[mask].tolist()}
            for name, mask in conditions.items()},
        'raw_count_above_6_rows': int(np.sum(cache['exact'] > 6)),
        'remaining_limit': 'No waveform listening, pitch-source separation or retraining ablation establishes the acoustic reason for incorrect network probabilities.',
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'input-audit.json', report)
    np.savez_compressed(args.output_dir / 'input-audit.npz', global_index=index, k=k,
                        raw_exact=cache['exact'], cached_slot_count=slot_k,
                        candidate_count=candidate_count, **arrays)
    np.savez_compressed(args.output_dir / 'onset-assignments.npz',
                        columns=np.asarray(['outer_row', 'string', 'onset_sample', 'distance_sample',
                                            'relative_sample', 'outside_spectral_window']), values=events)
    print({name: entry['present'] for name, entry in report['associations_not_causal_proof'].items()}, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', type=int, default=3, choices=[3])
    p.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    for name in ('cache-dir', 'annotation-zip', 'decision-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
