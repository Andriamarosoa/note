"""Reopen saved maps and independently check ownership, coverage and identities."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from causal_note.guitarset import index_guitarset, load_boundary_slots
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_ownership_context import owners_at_samples


def exhaustive_sample_owners(groups):
    """Enumerate every candidate's radius; strict updates preserve lower-ID ties."""
    length = max(int(g.max()) for g in groups)+883
    best = np.full(length, 883, np.int16)
    owner = np.full(length, -1, np.int32)
    distance = np.abs(np.arange(-882, 883, dtype=np.int16))
    for group_id, group in enumerate(groups):
        for position in group:
            left, right = max(0, int(position)-882), int(position)+883
            d = distance[left-(int(position)-882):]
            update = d < best[left:right]
            best[left:right][update] = d[update]
            owner[left:right][update] = group_id
    return owner


def verify(root, dataset, geometry):
    report = json.loads((root/'report.json').read_text())
    assert digest(root/'rows.csv') == report['rows_sha256']
    with (root/'rows.csv').open() as stream:
        rows = {int(r['global_index']):r for r in csv.DictReader(stream)}
    assert len(rows) == report['rows'] == 15952
    allowed = {r['member'] for r in report['track_replay']}
    tracks = {t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in allowed}
    source_report = json.loads((geometry/'report.json').read_text())
    timing = {r['member']:geometry/r['timing_file'] for r in source_report['tracks']}
    historical = np.load(geometry/'geometry.npy', mmap_mode='r')
    checked = assigned_total = positive_support = exact_owned = warmup = 0
    fractions_checked = mask_queries = 0
    by_position = {'before':0, 'future':0, 'after':0}
    taper = np.hanning(256)
    for info in report['track_replay']:
        member = info['member']; track = tracks[member]
        assert digest(root/info['map_file']) == info['map_sha256']
        assert digest(root/info['records_file']) == info['records_sha256']
        with np.load(timing[member], allow_pickle=False) as z:
            offsets, flat = z['full_candidate_offsets'], z['full_candidate_samples']
            groups = [flat[a:b] for a,b in zip(offsets[:-1], offsets[1:])]
            indices = z['global_index']
        onsets = np.array([b.onset_sample
            for slot in load_boundary_slots(track.annotation_zip, track.annotation_member)
            for b in slot], np.int64)
        owner = owners_at_samples(groups, onsets)
        dense_owner = exhaustive_sample_owners(groups)
        with np.load(root/info['map_file'], allow_pickle=False) as z:
            assert z['maps'].shape == (len(groups), 31, 64, 4)
            assert z['maps'].dtype == np.float16 and np.isfinite(z['maps']).all()
            assert z['maps'].min() >= 0
            assert z['ownership_packed'].shape == (len(groups), 512)
            np.testing.assert_array_equal(z['global_index'], indices)
            np.testing.assert_array_equal(z['frame_ends'], -1052+128*np.arange(31))
            np.testing.assert_array_equal(z['ownership_fraction'][:, :, 0], historical[indices, :, 1])
            for i, idx in enumerate(indices):
                source = rows[int(idx)]; origin = int(groups[i][0])
                assert source['member'] == member and int(source['start']) == origin == z['start'][i]
                assert z['decision_sample'][i] >= max(origin+2788, int(groups[i][-1])+1764+5)
                bits = np.unpackbits(z['ownership_packed'][i], bitorder='little').astype(bool)
                queries = origin-1308+np.arange(4096)
                # Enumerating every candidate's entire radius checks every bit,
                # independently of the builder's sorted-neighbor lookup.
                valid = (queries >= 0) & (queries < len(dense_owner))
                expected = np.zeros(len(queries), bool)
                expected[valid] = dense_owner[queries[valid]] == i
                np.testing.assert_array_equal(bits, expected)
                mask_queries += len(bits)
                padded = np.r_[np.zeros(1792, bool), bits]
                for frame in range(31):
                    np.testing.assert_array_equal(z['ownership_fraction'][i, frame],
                        [bits[frame*128:frame*128+256].mean(),
                         padded[frame*128:frame*128+2048].mean()])
                    fractions_checked += 2
                relative = onsets[owner == i]-origin
                assert len(relative) == int(source['k']) == int(source['assigned_onsets'])
                within = (relative >= -1308) & (relative < 2788)
                assert within.all() and bits[relative+1308].all()
                window_positions = relative[:, None]-(z['frame_ends'][None, :]-256)
                in_window = (window_positions >= 0) & (window_positions < 256)
                hann_values = taper[np.clip(window_positions, 0, 255)]*in_window
                assert np.all(np.any(hann_values > 0, axis=1))
                by_position['before'] += int(np.sum(relative < 0))
                by_position['future'] += int(np.sum((relative >= 0) & (relative < 1764)))
                by_position['after'] += int(np.sum(relative >= 1764))
                # Include the absolute block alignment in warm-up accounting.
                is_warmup = ((origin-3100)//128)*128 < 8820
                assert int(is_warmup) == int(source['predictor_initialization_padded'])
                warmup += int(is_warmup)
                assigned_total += len(relative); positive_support += len(relative); exact_owned += len(relative)
                checked += 1
    assert checked == len(rows) and assigned_total == positive_support == exact_owned == 9127
    coverage = report['coverage']
    assert warmup == coverage['predictor_initialization_padded']
    for name in by_position:
        assert by_position[name] == coverage['owned_'+name]
    result = dict(passed=True, rows_checked=checked, tracks_checked=len(tracks),
        assigned_onsets=assigned_total, positive_hann_support_onsets=positive_support,
        exact_owned_onsets=exact_owned, position_counts=by_position,
        ownership_bits_independently_checked=mask_queries,
        ownership_frame_fractions_checked=fractions_checked,
        exact_block_aligned_warmup_rows=warmup, outer_rows_evaluated=0,
        verifier_sha256=digest(__file__), report_sha256=digest(root/'report.json'),
        cached_map_hashes_verified=True, counts_from_annotations_not_feature_inputs=True)
    write_json(root/'verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'dataset', 'geometry'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    verify(args.root, args.dataset, args.geometry)
