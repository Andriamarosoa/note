"""Extend native spectra on saved exact candidates, auditing outer fold 3 only.

No candidate inference, label fitting, count prediction, or training. The
window is determined from the existing grouping/verification protocol.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

from causal_note.guitarset import index_guitarset, load_boundary_slots, SAMPLE_RATE
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts.train_v86_state_transition_proposals import MAX_HORIZON
from scripts.train_v90_structured_cluster_cardinality import CLUSTER_WINDOW_SAMPLES
from scripts.train_v92_string_factorized_cardinality import LOCAL_RADIUS_SAMPLES
from scripts.candidate_timing import full_samples, timing_fields
from scripts.spectral_window import LEGACY, COVERED, cache_window, window_metadata
from scripts.audit_v273_control_inputs import nearest_assignment
from scripts.rebuild_v273_sources import digest, write_json, verify_dataset


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def same_bytes(a, b, name):
    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
            'changed frozen input: ' + name)


def audit(root, dataset_dir, config, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    verify_dataset(dataset_dir)
    cfg = json.loads(config.read_text())
    allowed = {m for m, fold in cfg['member_folds'].items() if fold == 3}
    require(len(allowed) == 50, 'unexpected fold 3 track count')
    require(COVERED.post_samples == CLUSTER_WINDOW_SAMPLES + MAX_HORIZON,
            'window no longer equals the declared verification horizon')
    latest = CLUSTER_WINDOW_SAMPLES + LOCAL_RADIUS_SAMPLES
    require(COVERED.centers[0] <= -LOCAL_RADIUS_SAMPLES and COVERED.centers[-1] >= latest,
            'window does not cover the protocol assignment bound')
    indexed = {t.annotation_member: t for t in index_guitarset(dataset_dir)}
    paths = sorted(root.rglob('v100-spectral-shard-*.npz'))
    require(len(paths) == 50, 'expected 50 saved exact spectral shards')
    output_dir.mkdir(parents=True)
    seen, reports, recovered = set(), [], []
    totals = dict(rows=0, poly_rows=0, annotations=0, assigned=0, unassigned=0,
                  legacy_outside_events=0, covered_outside_events=0,
                  legacy_outside_center_events=0, covered_outside_center_events=0,
                  left_padded_rows=0, right_padded_rows=0, assigned_outside_audio=0)
    for number, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            old = {key: np.asarray(data[key]) for key in data.files}
        require(int(old['schema_version'][0]) == 2 and cache_window(old, version=2) == LEGACY,
                'expected an exact-timing legacy spectral cache')
        timing = timing_fields(old, required=True)
        names = set(old['members'].astype(str))
        require(len(names) == 1 and not names & seen and names <= allowed, 'wrong/duplicate fold member')
        seen |= names
        member = next(iter(names))
        track = indexed[member]
        full = full_samples(old)
        require(all(int(x[-1] - x[0]) <= CLUSTER_WINDOW_SAMPLES for x in full),
                'candidate group exceeds the protocol bound')
        spectral, _ = v100._spectral_maps_for_cache(old, dataset_dir, window=COVERED)
        same_bytes(spectral[:, :LEGACY.time_frames], old['spectral'], 'historical spectral prefix')
        require(np.isfinite(spectral).all(), 'nonfinite spectral input')

        # Independent runtime API on the identical saved full groups.
        records, clusters = [], []
        for samples in full:
            ids = list(range(len(records), len(records) + len(samples)))
            records.extend({'sample': int(s)} for s in samples)
            clusters.append({'member': member, 'indices': ids})
        runtime = v100._spectral_maps_for_runtime([track], clusters, records, window=COVERED)
        same_bytes(spectral, runtime.astype(np.float16), 'cache/runtime spectra')
        destination = output_dir / 'cache' / f'track-{number:02d}' / 'v100-spectral-shard-00.npz'
        destination.parent.mkdir(parents=True)
        v100._save_spectral_cache(destination, old, spectral, old['slot_targets'], window=COVERED)
        with np.load(destination, allow_pickle=False) as data:
            new = {key: np.asarray(data[key]) for key in data.files}
        preserved = set(old) - {'schema_version', 'spectral'}
        require(set(new) == set(old) | set(window_metadata(COVERED)), 'cache field inventory changed')
        for key in preserved:
            same_bytes(new[key], old[key], key)
        loaded = v100._load_spectral_caches(destination.parent)
        require(cache_window(loaded) == COVERED, 'covered window did not survive loading')
        np.testing.assert_array_equal(loaded['spectral'], spectral)

        old_sup = v102._derive_supervision(old['members'], [], dataset_dir,
                    expected_slot_targets=old['slot_targets'], assignment_cache=old)
        new_sup = v102._derive_supervision(new['members'], [], dataset_dir,
                    expected_slot_targets=new['slot_targets'], assignment_cache=new)
        for key in (0, 1, 3):  # pitch, mask, physical onset; Gaussian support intentionally extends
            same_bytes(old_sup[key], new_sup[key], f'supervision component {key}')
        require(new_sup[2].shape == (len(full), 6, 31), 'time supervision has wrong shape')
        np.testing.assert_allclose(new_sup[2].sum(2), new_sup[1], atol=2e-7)
        require(new_sup[4]['same_slot_collisions'] == 0, 'new same-string ambiguity needs audit')
        require(new_sup[4]['outside_frame_center_range'] == 0, 'new time target outside centers')

        audio = v100.decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
        audio_length = len(audio.samples)
        starts = timing['cluster_start_samples']
        local = {key: 0 for key in totals}
        local['rows'], local['poly_rows'] = len(full), int((old['exact'] >= 2).sum())
        local['left_padded_rows'] = int((starts < COVERED.pre_samples).sum())
        local['right_padded_rows'] = int((starts + COVERED.post_samples > audio_length).sum())
        flat = np.concatenate(full)
        row_ids = np.repeat(np.arange(len(full)), [len(x) for x in full])
        order = np.argsort(flat, kind='stable')
        flat, row_ids = flat[order], row_ids[order]
        counts = np.zeros(len(full), np.int32)
        occupied = np.zeros((len(full), 6), np.int32)
        for slot, boundaries in enumerate(load_boundary_slots(track.annotation_zip, member)):
            for note in boundaries:
                local['annotations'] += 1
                onset = int(note.onset_sample)
                assignment = nearest_assignment(onset, flat, row_ids)
                if assignment is None:
                    local['unassigned'] += 1
                    continue
                row, distance = assignment
                counts[row] += 1
                occupied[row, slot] += 1
                local['assigned'] += 1
                relative = onset - int(starts[row])
                local['assigned_outside_audio'] += int(not 0 <= onset < audio_length)
                require(-LOCAL_RADIUS_SAMPLES <= relative <= latest, 'assigned onset violates protocol bound')
                for name, window in (('legacy', LEGACY), ('covered', COVERED)):
                    outside = not -window.pre_samples <= relative < window.post_samples
                    local[name + '_outside_events'] += int(outside)
                    local[name + '_outside_center_events'] += int(not window.centers[0] <= relative <= window.centers[-1])
                    if name == 'legacy' and outside:
                        recovered.append(dict(member=member, track_row=int(row), slot=slot,
                            onset_sample=onset, relative_sample=relative, k=int(old['exact'][row])))
        np.testing.assert_array_equal(counts, old['exact'])
        np.testing.assert_array_equal(occupied, old['slot_targets'])
        require(local['assigned'] == new_sup[4]['assigned_events'] and
                local['unassigned'] == new_sup[4]['unassigned_events'], 'assignment implementations disagree')
        require(local['covered_outside_events'] == local['covered_outside_center_events'] ==
                local['assigned_outside_audio'] == 0, 'coverage or audio-boundary regression')
        for key, value in local.items():
            totals[key] += value
        reports.append(dict(member=member, totals=local, source_path=str(path.relative_to(root)),
                            source_sha256=digest(path), covered_sha256=digest(destination),
                            preserved_byte_identical_fields=sorted(preserved),
                            spectral_prefix_byte_identical=True, runtime_cache_identical=True,
                            physical_supervision_byte_identical=True))
        write_json(output_dir / 'progress.json', dict(completed_tracks=len(reports), totals=totals))
        print(f'{number+1}/50 {member} rows={len(full)} recovered={local["legacy_outside_events"]}', flush=True)
    require(seen == allowed, 'incomplete fold 3 coverage')
    for key, expected in dict(rows=15279, poly_rows=1969, annotations=9266, assigned=8812,
                              unassigned=454, legacy_outside_events=96).items():
        require(totals[key] == expected, 'pinned population changed: ' + key)
    result = dict(status='passed', outer_fold=3, track_count=len(reports), totals=totals,
        recovered_row_count=len({(r['member'], r['track_row']) for r in recovered}),
        recovered_poly_row_count=len({(r['member'], r['track_row']) for r in recovered if r['k'] >= 2}),
        recovered_events=recovered, tracks=reports, config_sha256=digest(config),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        window=dict(name=COVERED.name, pre_samples=COVERED.pre_samples, post_samples=COVERED.post_samples,
                    frames=COVERED.time_frames, earliest_assignable_relative_sample=-LOCAL_RADIUS_SAMPLES,
                    latest_assignable_relative_sample=latest, frame_centers=COVERED.centers.tolist(),
                    declared_finalization_delay_ms=COVERED.post_samples * 1000 / SAMPLE_RATE),
        candidate_remining_performed=False, model_training_performed=False,
        exact_k_improvement_measured=False, other_outer_folds_evaluated=False,
        limitations=['This audits acoustic coverage, not trained counting accuracy.',
                     'Finalization waits for the existing declared group-plus-verifier horizon; live latency is not measured.',
                     'Zero padding at file boundaries is reported and never treated as observed audio.',
                     'The 454 unassigned annotations remain outside the candidate assignment protocol.'])
    write_json(output_dir / 'report.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'dataset-dir', 'config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.dataset_dir, a.config, a.output_dir)
