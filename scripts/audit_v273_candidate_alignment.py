"""Audit ambiguity in legacy cache timestamp recovery; never change a model.

The highest fused score provides an annotation-free additional constraint.
When it does not identify one origin, this diagnostic keeps the legacy origin
and reports the ambiguity. This is not a production reconstruction patch.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from causal_note.guitarset import load_boundary_slots
from scripts.audit_v273_control_fold3 import metrics
from scripts.audit_v273_control_inputs import nearest_assignment
from scripts.audit_v273_native_paired_results import load_npz, require
from scripts.rebuild_v273_sources import digest, write_json
from scripts import train_v92_string_factorized_cardinality as slots


def possible_origins(sequence, mask, top):
    """Enumerate origins consistent with exact top times and highest-score ties."""
    rows, relative = slots._candidate_relative_samples(sequence, mask)
    top = np.asarray(top, dtype=np.int64)
    top = top[top >= 0]
    require(len(top) > 0 and len(rows) > 0, 'empty candidate metadata')
    starts = np.unique((top[:, None] - relative[None, :]).ravel())
    exact = []
    for start in starts:
        positions = start + relative
        if all(np.any(positions == t) for t in top):
            exact.append(int(start))
    score = np.asarray(sequence[rows, -3], dtype=np.float64)
    # True float32 maximum must be one of the maximum float16 scores retained.
    rank_origins = set(int(top[0] - rel) for rel in relative[score == score.max()])
    constrained = sorted(set(exact) & rank_origins)
    return exact, constrained


def assign_counts(members, candidates, annotation_zip):
    groups = defaultdict(list)
    for i, member in enumerate(members):
        groups[str(member)].append(i)
    counts = np.zeros(len(members), np.int32)
    for member, rows in sorted(groups.items()):
        sample = np.concatenate([candidates[i] for i in rows])
        row = np.concatenate([np.full(len(candidates[i]), i) for i in rows])
        order = np.argsort(sample, kind='stable')
        sample, row = sample[order], row[order]
        for boundaries in load_boundary_slots(annotation_zip, member):
            for note in boundaries:
                got = nearest_assignment(note.onset_sample, sample, row)
                if got is not None:
                    counts[got[0]] += 1
    return counts


def run(args):
    require(not args.output_dir.exists(), 'output already exists')
    c = load_npz(args.input_dir / 'outer-candidate-metadata.npz')
    inp = load_npz(args.input_dir / 'input-audit.npz')
    d = load_npz(args.decision_dir / 'decision-audit.npz')
    require(np.array_equal(c['global_index'], d['global_index']), 'row alignment changed')
    n = len(c['global_index'])
    old, proposed = np.empty(n, np.int64), np.empty(n, np.int64)
    possibilities = np.empty(n, np.int16)
    rank_possibilities = np.empty(n, np.int16)
    legacy_disallowed = np.zeros(n, bool)
    minimum_relative = np.empty(n, np.int32)
    candidates = []
    for i, (seq, mask, top) in enumerate(zip(c['sequence'], c['mask'], c['top_samples'])):
        old[i], _ = slots._recover_cluster_start(seq, mask, top)
        exact, rank = possible_origins(seq, mask, top)
        require(len(exact) > 0 and len(rank) > 0, 'metadata cannot recover an exact rank-consistent origin')
        possibilities[i], rank_possibilities[i] = len(exact), len(rank)
        legacy_disallowed[i] = old[i] not in rank
        proposed[i] = rank[0] if len(rank) == 1 else old[i]
        _, relative = slots._candidate_relative_samples(seq, mask)
        minimum_relative[i] = min(relative)
        candidates.append((proposed[i] + relative).astype(np.int64))
    recomputed = assign_counts(c['members'], candidates, args.annotation_zip)
    mismatch_before = inp['reconstructed_exact'] != c['exact']
    mismatch_after = recomputed != c['exact']
    changed = old != proposed
    same = c['members'][1:] == c['members'][:-1]
    full = (c['truncated'][1:] == 0) & (c['truncated'][:-1] == 0)
    bad_before = same & full & (np.diff(old) <= 1764)
    bad_after = same & full & (np.diff(proposed) <= 1764)
    k, pred = d['k'], d['final']
    grid = np.arange(1765, dtype=np.int32)
    recovered_grid = np.rint((grid.astype(np.float32) / 1764).astype(np.float16).astype(np.float64) * 1764)
    require(np.array_equal(grid, recovered_grid), 'float16 relative timestamp round-trip not exact')
    report = {
        'scope': {'outer_fold': 3, 'rows': n, 'model_changed': False,
                  'spectra_recomputed': False, 'thresholds_changed': False,
                  'labels_used_to_choose_origins': False,
                  'rank_recovery_is_a_diagnostic_not_a_model_fix': True},
        'metadata_sha256': digest(args.input_dir / 'outer-candidate-metadata.npz'),
        'legacy_exact_ambiguity_rows': int(np.sum(possibilities > 1)),
        'legacy_origin_inconsistent_with_max_score': int(legacy_disallowed.sum()),
        'unique_origin_with_score_constraint_rows': int(np.sum(rank_possibilities == 1)),
        'still_ambiguous_with_score_constraint_rows': int(np.sum(rank_possibilities > 1)),
        'diagnostic_shifted_rows': int(changed.sum()),
        'shifted_global_indices': c['global_index'][changed].tolist(),
        'shift_sample_histogram': {str(v): int(np.sum((proposed - old) == v))
                                 for v in np.unique(proposed - old) if v != 0},
        'before_nontruncated_group_spacing_violations': int(bad_before.sum()),
        'after_nontruncated_group_spacing_violations': int(bad_after.sum()),
        'cached_count_disagreements_before': int(mismatch_before.sum()),
        'cached_count_disagreements_after': int(mismatch_after.sum()),
        'mismatched_global_indices_after': c['global_index'][mismatch_after].tolist(),
        'raw_onset_total_original': int(c['exact'].sum()),
        'raw_onset_total_reconstructed_before': int(inp['reconstructed_exact'].sum()),
        'raw_onset_total_reconstructed_after': int(recomputed.sum()),
        'relative_timestamp_round_trip_values_verified': 1765,
        'relative_timestamp_round_trip_errors': 0,
        'baseline_metrics_on_shifted_rows_not_a_correction_score': metrics(k[changed], pred[changed]),
        'limitations': [
            'Original absolute timestamps were not stored; maximum scores were quantized to float16.',
            'Rank-ambiguous origins are deliberately left unchanged.',
            'No inference on a regenerated spectrum or retraining establishes an exact-K improvement.',
            'Onsets outside the group, alternative score constraints and the old supervision explain only a bounded subset of errors.',
        ],
        'safe_source_fix': 'Store original integer cluster starts and absolute candidate samples when mining; use and verify them when generating slots and spectra. Keep the existing benchmark immutable.',
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'alignment-audit.json', report)
    np.savez_compressed(args.output_dir / 'alignment-audit.npz', global_index=c['global_index'],
                        legacy_origin=old, diagnostic_origin=proposed,
                        exact_origin_options=possibilities, score_origin_options=rank_possibilities,
                        original_exact=c['exact'], before_exact=inp['reconstructed_exact'],
                        after_exact=recomputed, original_truncated=c['truncated'])
    print({key:value for key,value in report.items() if key not in
           ('shifted_global_indices', 'shift_sample_histogram')}, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input-dir', 'decision-dir', 'annotation-zip', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
