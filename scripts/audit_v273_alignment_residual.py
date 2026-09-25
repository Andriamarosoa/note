"""Audit the two residual label discrepancies using saved event assignments.

Scope is the 8,812 onsets already assigned by the annotation audit, not its
454 unassigned onsets. No labels are used to locate candidate-group endpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.audit_v273_control_inputs import nearest_assignment
from scripts.audit_v273_native_paired_results import load_npz, require, verify_inventory
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v92_string_factorized_cardinality import _candidate_relative_samples


def replay_events(members, candidates, events):
    groups = defaultdict(list)
    event_groups = defaultdict(list)
    for i, member in enumerate(members):
        groups[str(member)].append(i)
    for event in events:
        event_groups[str(members[int(event[0])])].append(event)
    counts = np.zeros(len(members), np.int32)
    assignments = []
    for member, rows in sorted(groups.items()):
        samples = np.concatenate([candidates[i] for i in rows])
        ids = np.concatenate([np.full(len(candidates[i]), i) for i in rows])
        order = np.argsort(samples, kind='stable')
        samples, ids = samples[order], ids[order]
        for event in event_groups[member]:
            got = nearest_assignment(int(event[2]), samples, ids)
            require(got is not None, 'an already assigned event became unassigned')
            counts[got[0]] += 1
            assignments.append((int(event[0]), int(event[1]), int(event[2]), got[0], got[1]))
    return counts, np.asarray(assignments, dtype=np.int64)


def run(args):
    require(not args.output_dir.exists(), 'output already exists')
    verified = verify_inventory(args.audit_dir)
    c = load_npz(args.audit_dir / 'inputs/outer-candidate-metadata.npz')
    a = load_npz(args.audit_dir / 'alignment/alignment-audit.npz')
    e = load_npz(args.audit_dir / 'inputs/onset-assignments.npz')['values']
    require(np.array_equal(c['global_index'], a['global_index']), 'index mismatch')
    legacy, ranked, endpoints = [], [], []
    full_end = np.empty(len(c['global_index']), dtype=np.int64)
    for i, (seq, mask, stats) in enumerate(zip(c['sequence'], c['mask'], c['stats'])):
        _, relative = _candidate_relative_samples(seq, mask)
        old, start = a['legacy_origin'][i], a['diagnostic_origin'][i]
        width = int(round(float(stats[1]) * 1764))
        full_end[i] = start + width
        legacy.append(old + relative)
        ranked.append(start + relative)
        if c['truncated'][i] == 0:
            require(min(relative) == 0, 'untruncated cluster lost its first position')
            require(max(relative) == width or (max(relative) == 0 and width == 1),
                    'untruncated cluster extent differs from original statistics')
            endpoints.append(start + relative)
        else:
            endpoints.append(np.unique(np.concatenate([start + relative, [start, start + width]])))
    before, _ = replay_events(c['members'], legacy, e)
    rank_counts, _ = replay_events(c['members'], ranked, e)
    augmented_counts, assignments = replay_events(c['members'], endpoints, e)
    require(np.array_equal(before, a['before_exact']), 'saved original event assignments do not replay')
    require(np.array_equal(rank_counts, a['after_exact']), 'saved rank-based assignments do not replay')
    mismatch = rank_counts != c['exact']
    details = []
    for i in np.flatnonzero(mismatch):
        event_rows = assignments[(assignments[:, 0] == i) | (assignments[:, 3] == i)]
        details.append({'global_index': int(c['global_index'][i]), 'member': str(c['members'][i]),
                        'original_count': int(c['exact'][i]), 'rank_reconstructed_count': int(rank_counts[i]),
                        'with_original_endpoints_count': int(augmented_counts[i]),
                        'truncated_candidates': int(c['truncated'][i]),
                        'retained_last_sample': int(max(ranked[i])),
                        'original_end_sample_from_stats': int(full_end[i]),
                        'events_after_endpoint_check': event_rows.tolist()})
    report = {
        'scope': {'outer_fold': 3, 'rows': len(c['global_index']), 'assigned_onsets_replayed': len(e),
                  'unassigned_onsets_not_rechecked': 454, 'model_or_spectra_modified': False,
                  'annotations_used_to_choose_origins_or_endpoints': False},
        'verified_source_files': verified,
        'source_inventory_sha256': digest(args.audit_dir / 'file-sha256.json'),
        'legacy_event_replay_equal': True, 'rank_event_replay_equal': True,
        'count_disagreements_before': int(np.sum(before != c['exact'])),
        'count_disagreements_after_rank_constraint': int(mismatch.sum()),
        'count_disagreements_after_restoring_known_endpoints': int(np.sum(augmented_counts != c['exact'])),
        'new_count_disagreements_from_endpoint_check': int(np.sum(~mismatch & (augmented_counts != c['exact']))),
        'onset_total_preserved': int(augmented_counts.sum()),
        'residual_rows': details,
        'interpretation': 'The remaining pair loses a candidate near the last onset when 57 candidates are reduced to 48; slot-label reconstruction assigns that onset to the next group. Original group extent supplies an annotation-free endpoint check.',
        'safe_source_fix': 'Preserve original integer group starts and complete untruncated candidate times for supervision; keep retained candidate times separately for model inputs. Use the original group start for the spectral window.',
        'not_claimed': 'This is an input/supervision audit, not a retrained model or an exact-K score improvement.',
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'alignment-residual-audit.json', report)
    np.savez_compressed(args.output_dir / 'alignment-residual-audit.npz', global_index=c['global_index'],
                        original_count=c['exact'], legacy_count=before, rank_count=rank_counts,
                        endpoint_count=augmented_counts)
    print(report, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    run(p.parse_args())
