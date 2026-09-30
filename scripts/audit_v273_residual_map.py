"""Frozen, label-free residual-map extraction on the existing inner partition.

Annotations are used only after feature construction to audit temporal coverage.
This program neither trains a model nor changes an existing predicted count.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np

from causal_note.guitarset import index_guitarset, load_boundary_slots
from scripts.audit_v273_acoustic_residual import ownership
from scripts.audit_v273_coherent_real import describe, npy_hash
from scripts.rebuild_v273_sources import digest, verify_dataset, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.train_v100_spectral_string_slots import decode_pcm16_mono_wav
from scripts.v273_ownership_context import owners_at_samples
from scripts.v273_ownership_experiment import sorted_proposals
from scripts.v273_residual_map import (
    CONFIG, FRAME_ENDS, FREQUENCIES, ResidualStream, innovation,
    onset_coverage, ownership_input, time_frequency_map)

PROTOCOL = Path('analysis/v273-residual-map-protocol.md')
BASELINE_SHA = '42110bc6e5a04fc7f5d253876bb5c543cd96a23d0e8bd9a86617b8fa7096738e'
CONFIG_SHA = '7c27989eb3fe330fdff203e388e76ae4c64d8b8e4838cab4d7c4622b4104451a'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def process_track(task):
    track, timing_path, sources, watermarks, decisions, historical, output = task
    began = time.monotonic()
    indices = np.array([int(r['global_index']) for r in sources], np.int64)
    starts = np.array([int(r['start']) for r in sources], np.int64)
    with np.load(timing_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['global_index'], indices)
        require(str(z['member'][0]) == track.annotation_member, 'timing member differs')
        offsets, flat = z['full_candidate_offsets'], z['full_candidate_samples']
        require(offsets[0] == 0 and offsets[-1] == len(flat) and np.all(np.diff(offsets) > 0),
                'invalid candidate offsets')
        groups = [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    np.testing.assert_array_equal(starts, [g[0] for g in groups])
    samples = np.asarray(decode_pcm16_mono_wav(track.audio_zip, track.audio_member).samples,
                         np.float32)/32768.
    residual, blocks = innovation(samples)
    count = len(groups)
    maps = np.empty((count, 31, 64, 4), np.float16)
    masks = np.empty((count, 512), np.uint8)
    fractions = np.empty((count, 31, 2), np.float32)
    proposals = sorted_proposals(groups)
    records = []
    for i, (idx, start) in enumerate(zip(indices, starts)):
        features = time_frequency_map(samples, residual, int(start))
        maps[i] = features
        require(np.isfinite(maps[i]).all(), 'nonfinite float16 map')
        masks[i], fractions[i] = ownership_input(groups, i,
            complete_through=int(watermarks[i]), decision_sample=int(decisions[i]),
            proposals=proposals)
        np.testing.assert_array_equal(fractions[i, :, 0], historical[i, :, 1])
        quantization = np.abs(features-maps[i].astype(np.float32))
        # Sum sampled band powers, not an integral over all audio frequencies.
        power = np.expm1(features.astype(np.float64))
        row = dict(global_index=int(idx), member=track.annotation_member,
            composition=sources[i]['composition'], start=int(start),
            proposal_complete_through=int(watermarks[i]), decision_sample=int(decisions[i]),
            decision_delay_ms=float((decisions[i]-start)*1000/44100),
            extra_decision_samples=0,
            map_audio_padded=int(start < 3100 or start+2788 > len(samples)),
            predictor_initialization_padded=int(start < 3100+8820),
            map_min=float(features.min()), map_max=float(features.max()),
            float16_max_abs_error=float(quantization.max()),
            float16_mean_abs_error=float(quantization.mean()),
            residual_spatial_std=float(features[:, :, [1, 3]].std()),
            residual_time_variance=float(features[:, :, [1, 3]].var(axis=0).mean()),
            residual_frequency_variance=float(features[:, :, [1, 3]].var(axis=1).mean()),
            residual_to_observed_short_bandpower=float(power[:, :, 1].sum()/max(power[:, :, 0].sum(), 1e-20)),
            residual_to_observed_long_bandpower=float(power[:, :, 3].sum()/max(power[:, :, 2].sum(), 1e-20)))
        records.append(row)

    # This boundary is deliberate: no annotation has been loaded above it.
    onsets = np.asarray([b.onset_sample
        for slot in load_boundary_slots(track.annotation_zip, track.annotation_member)
        for b in slot], np.int64)
    assigned, exact, _, _ = ownership(groups, onsets)
    np.testing.assert_array_equal(assigned, owners_at_samples(groups, onsets))
    np.testing.assert_array_equal(exact, [int(r['k']) for r in sources])
    for i, row in enumerate(records):
        relative = onsets[assigned == i]-starts[i]
        bits = np.unpackbits(masks[i], bitorder='little').astype(bool)
        within = (relative >= -1308) & (relative < 2788)
        covered = onset_coverage(relative).any(axis=1)
        owned = np.zeros(len(relative), bool)
        owned[within] = bits[relative[within]+1308]
        before = relative < 0
        after = relative >= 1764
        counts = dict(owned_before=int(before.sum()), owned_future=int((~before & ~after).sum()),
                      owned_after=int(after.sum()))
        require(all(counts[key] == int(sources[i][key]) for key in counts),
                'source onset coverage changed')
        row.update(k=int(exact[i]), **counts, assigned_onsets=len(relative),
            covered_onsets=int(covered.sum()), owned_mask_onsets=int(owned.sum()),
            newly_covered_onsets=int(np.sum(covered & (before | after))))
        require(covered.all() and owned.all(), 'assigned onset lacks positive owned support')

    # Independently stop decoding the model input before any later sample.
    prefix_check = None
    if track.annotation_member.startswith('00_') and track.annotation_member.endswith('_comp.jams'):
        middle = count//2; stop = int(starts[middle])+2788
        stream = ResidualStream()
        prefix = samples[:stop]
        chunked = np.concatenate([stream.process(prefix[a:a+65537])
                                  for a in range(0, len(prefix), 65537)])
        full_map = time_frequency_map(samples, residual, int(starts[middle]))
        prefix_map = time_frequency_map(prefix, chunked, int(starts[middle]))
        np.testing.assert_array_equal(prefix_map, full_map)
        prefix_check = dict(global_index=int(indices[middle]), input_stop_exclusive=stop,
            prefix_samples=len(prefix), chunk_samples=65537, max_abs_difference=0., passed=True)

    output = Path(output)
    name = timing_path.stem
    cache = output/'maps'/f'{name}.npz'
    np.savez_compressed(cache, maps=maps, ownership_packed=masks,
        ownership_fraction=fractions, global_index=indices, start=starts,
        proposal_complete_through=watermarks, decision_sample=decisions,
        frame_ends=FRAME_ENDS, frequencies_hz=FREQUENCIES,
        member=np.array([track.annotation_member]))
    with np.load(cache, allow_pickle=False) as z:
        for key, expected in (('maps', maps), ('ownership_packed', masks),
                              ('ownership_fraction', fractions), ('global_index', indices)):
            np.testing.assert_array_equal(z[key], expected)
    write_json(output/'tracks'/f'{name}.json', records)
    radius = np.asarray([b['max_pole_radius_per_lag'] for b in blocks])
    require(np.isfinite(radius).all() and np.all(radius <= 1+1e-7), 'unstable stream')
    return dict(member=track.annotation_member, rows=count, audio_samples=len(samples),
        stream_blocks=len(blocks), unstable_blocks=int(np.sum(radius > 1+1e-7)),
        max_pole_radius=float(radius.max()),
        max_abs_reflection=max(b['max_abs_reflection'] for b in blocks),
        nonfinite_maps=0, feature_failures=0, prefix_check=prefix_check,
        timing_sha256=digest(timing_path), map_file=str(cache.relative_to(output)),
        map_sha256=digest(cache), map_compressed_bytes=cache.stat().st_size,
        map_array_bytes=maps.nbytes, ownership_bytes=masks.nbytes+fractions.nbytes,
        residual_sha256=npy_hash(residual), records_file=f'tracks/{name}.json',
        records_sha256=digest(output/'tracks'/f'{name}.json'),
        seconds=time.monotonic()-began)


def row_summary(rows):
    keys = ['map_min', 'map_max', 'float16_max_abs_error', 'float16_mean_abs_error',
        'residual_spatial_std', 'residual_time_variance', 'residual_frequency_variance',
        'residual_to_observed_short_bandpower', 'residual_to_observed_long_bandpower',
        'decision_delay_ms']
    return dict(rows=len(rows), **{key:describe([r[key] for r in rows]) for key in keys})


def run(args):
    require(not args.output.exists(), 'refusing to overwrite an existing audit')
    verify_dataset(args.dataset)
    validate_original_inventory(args.geometry)
    require(digest(args.config) == CONFIG_SHA, 'partition configuration changed')
    require(digest(args.baseline/'rows.csv') == BASELINE_SHA, 'baseline rows changed')
    baseline = json.loads((args.baseline/'report.json').read_text())
    require(baseline['rows_sha256'] == BASELINE_SHA and baseline['outer_rows_evaluated'] == 0
            and baseline['status'] == 'completed', 'invalid baseline')
    config = json.loads(args.config.read_text())
    allowed = {m for m, fold in config['member_folds'].items() if fold == 0}
    with (args.baseline/'rows.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 15952 and len(allowed) == 50 and
            {r['member'] for r in rows} == allowed, 'wrong internal population')
    ids = np.array([int(r['global_index']) for r in rows], np.int64)
    require(np.all(np.diff(ids) > 0), 'duplicate or reordered row identifiers')
    with np.load(args.geometry/'rows.npz', allow_pickle=False) as z:
        members, starts = z['member'], z['start']
        watermarks, decisions = z['proposal_complete_through'], z['decision_sample_support']
    geometry_report = json.loads((args.geometry/'report.json').read_text())
    for key, array in (('members', members), ('cluster_start_samples', starts)):
        require(npy_hash(array) == geometry_report['bundle_fields'][key]['sha256'],
                'source row identity changed')
    np.testing.assert_array_equal(ids, np.flatnonzero(np.isin(members, list(allowed))))
    np.testing.assert_array_equal(members[ids], [r['member'] for r in rows])
    np.testing.assert_array_equal(starts[ids], [int(r['start']) for r in rows])
    geometry = np.load(args.geometry/'geometry.npy', mmap_mode='r', allow_pickle=False)
    timing = {r['member']:args.geometry/r['timing_file']
              for r in geometry_report['tracks'] if r['member'] in allowed}
    require(set(timing) == allowed, 'missing timing source')
    tracks = {t.annotation_member:t for t in index_guitarset(args.dataset)
              if t.annotation_member in allowed}
    require(set(tracks) == allowed, 'missing audio source')
    args.output.mkdir(parents=True)
    (args.output/'maps').mkdir(); (args.output/'tracks').mkdir()
    frozen = dict(protocol_sha256=digest(PROTOCOL), script_sha256=digest(__file__),
        feature_builder_sha256=digest('scripts/v273_residual_map.py'),
        estimator_sha256=digest('scripts/v273_stable_prediction.py'),
        config_sha256=digest(args.config), baseline_rows_sha256=BASELINE_SHA,
        geometry_inventory_sha256=digest(args.geometry/'file-sha256.json'),
        geometry_report_sha256=digest(args.geometry/'report.json'),
        source_archives=baseline['sources'], dataset_md5=baseline['dataset_md5'],
        parameters=CONFIG, workers=args.workers, numpy=np.__version__, python=platform.python_version(),
        scope='all 15952 inner-validation rows; composition fold 0 of outer fold 3',
        outer_rows_evaluated=0, training_launched=False, output_corrector=False,
        exact_k_improvement_measured=False)
    write_json(args.output/'frozen-inputs.json', frozen)
    tasks = []
    for member in sorted(allowed):
        positions = np.flatnonzero(members[ids] == member)
        selected = ids[positions]
        tasks.append((tracks[member], timing[member], [rows[p] for p in positions],
            watermarks[selected], decisions[selected], np.asarray(geometry[selected]), str(args.output)))
    began = time.monotonic(); reports = []; records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_track, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result(); reports.append(result)
            records += json.loads((args.output/result['records_file']).read_text())
            print(json.dumps(dict(completed_tracks=len(reports), rows=len(records),
                member=result['member'], seconds=round(time.monotonic()-began, 1))), flush=True)
    records.sort(key=lambda r:r['global_index'])
    np.testing.assert_array_equal([r['global_index'] for r in records], ids)
    path = args.output/'rows.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    counts = {key:sum(r[key] for r in records) for key in ('assigned_onsets', 'covered_onsets',
        'owned_mask_onsets', 'newly_covered_onsets', 'owned_before', 'owned_future', 'owned_after',
        'map_audio_padded', 'predictor_initialization_padded', 'extra_decision_samples')}
    require(counts['assigned_onsets'] == counts['covered_onsets'] == counts['owned_mask_onsets'] == 9127,
            'incomplete coverage')
    prefix_checks = [r['prefix_check'] for r in reports if r['prefix_check'] is not None]
    require(len(prefix_checks) == 5 and all(r['passed'] for r in prefix_checks), 'missing prefix checks')
    summary = dict(status='completed', **frozen, rows=len(records), tracks=len(reports),
        coverage=counts, descriptive_statistics=row_summary(records),
        by_true_k={str(k):row_summary([r for r in records if r['k'] == k])
                   for k in sorted({r['k'] for r in records})},
        per_track_statistics={m:row_summary([r for r in records if r['member'] == m])
                              for m in sorted(allowed)},
        stream_blocks=sum(r['stream_blocks'] for r in reports),
        max_pole_radius=max(r['max_pole_radius'] for r in reports),
        max_abs_reflection=max(r['max_abs_reflection'] for r in reports),
        unstable_blocks=sum(r['unstable_blocks'] for r in reports), feature_failures=0, nonfinite_maps=0,
        cached_bytes=sum(r['map_compressed_bytes'] for r in reports),
        map_array_bytes=sum(r['map_array_bytes'] for r in reports),
        ownership_array_bytes=sum(r['ownership_bytes'] for r in reports),
        prefix_checks=prefix_checks, track_replay=sorted(reports, key=lambda r:r['member']),
        representation_gate=dict(passed=True, prepares_native_ablation_only=True,
                                 implies_exact_k_gain=False),
        rows_sha256=digest(path), elapsed_seconds=time.monotonic()-began)
    write_json(args.output/'report.json', summary)
    print(json.dumps({k:summary[k] for k in ('rows', 'tracks', 'coverage', 'stream_blocks',
        'max_pole_radius', 'cached_bytes', 'representation_gate', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'geometry', 'baseline', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    run(parser.parse_args())
