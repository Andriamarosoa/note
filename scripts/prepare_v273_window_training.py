"""Mine each non-outer track once and preserve exact native 31-frame inputs.

Both training arms will read these same files; the 23-frame control is a slice.
The already audited outer fold 3 is restored separately and never re-mined here.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np

from scripts import candidate_timing as timing
from scripts import train_v91_ordinal_cardinality as v91
from scripts import train_v92_string_factorized_cardinality as v92
from scripts import train_v100_spectral_string_slots as v100
from scripts.spectral_window import LEGACY, COVERED
from scripts.rebuild_v273_sources import validate_sources, verify_dataset, digest, write_json
from scripts.v273_native_protocol import load_config
from scripts.audit_v273_control_inputs import nearest_assignment
from causal_note.guitarset import load_boundary_slots


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cfg = load_config(args.config)
    allowed = {m for m, f in cfg['member_folds'].items() if f != 3}
    if len(allowed) != 190 or args.parts != 10 or not 0 <= args.part < args.parts:
        raise ValueError('expected ten disjoint parts of the 190 training/validation tracks')
    source = validate_sources(args.source_dir)
    verify_dataset(args.dataset_dir)
    _, train, locked = v91._dataset_split(args.dataset_dir)
    tracks = sorted((t for t in train if t.annotation_member in allowed), key=lambda t: t.annotation_member)
    if {t.annotation_member for t in tracks} != allowed or allowed & {t.annotation_member for t in locked}:
        raise RuntimeError('training source scope mismatch')
    model_args = SimpleNamespace(base_model=args.source_dir / 'v84/control.epoch-01.keras')
    for version, stem in (('v86', 'v86-state-transition-refiner'),
                          ('v87', 'v87-causal-candidate-memory'), ('v88', 'v88-regime-moe')):
        setattr(model_args, version + '_weights', args.source_dir / version / (stem + '.weights.h5'))
        setattr(model_args, version + '_report', args.source_dir / version / 'report.json')
    floor, _, enc86, enc87, model88 = v91._load_frozen_stack(model_args)
    args.output_dir.mkdir(parents=True)
    reports = []
    for i, track in enumerate(tracks):
        if i % args.parts != args.part:
            continue
        member = track.annotation_member
        print(f'PREPARE {i + 1}/190 {member}', flush=True)
        streams, records, x88, out88 = v91._represent_full((track,), model_args.base_model, floor, enc86, enc87, model88)
        clusters, fused, assignment, sequence, mask, stats, target, exact, truncated = v91._cluster_data((track,), records, x88, out88)
        directory = args.output_dir / f'track-{i:03d}'
        directory.mkdir()
        fields = timing.capture_timing(clusters, records, fused, v91.MAX_CANDIDATES)
        v91._save_cache(directory / 'v91-cache-shard-00.npz', sequence=sequence, mask=mask, stats=stats,
            target=target, exact=exact, truncated=truncated, members=[member] * len(clusters),
            top_samples=v91._top_samples(clusters, records, fused), track_members=[member], timing=fields)
        cache = v91._load_caches(directory)
        full = timing.full_samples(cache)
        if any(int(a[-1] - a[0]) > v100.CLUSTER_WINDOW_SAMPLES for a in full):
            raise RuntimeError('group exceeds the assignment coverage bound')
        slots, _, slot_diag = v92._derive_cache_slot_targets(cache, args.dataset_dir)
        spectra, _ = v100._spectral_maps_for_cache(cache, args.dataset_dir, window=COVERED)
        old, _ = v100._spectral_maps_for_cache(cache, args.dataset_dir, window=LEGACY)
        np.testing.assert_array_equal(spectra[:, :23], old)
        live = v100._spectral_maps_for_runtime((track,), clusters, records, window=COVERED)
        np.testing.assert_array_equal(spectra, live.astype(np.float16))
        if not np.isfinite(spectra).all():
            raise RuntimeError('nonfinite acoustic data')
        flat = np.concatenate(full)
        rows = np.repeat(np.arange(len(full)), [len(a) for a in full])
        order = np.argsort(flat, kind='stable')
        flat, rows = flat[order], rows[order]
        counts = np.zeros(len(full), np.int32)
        occupied = np.zeros((len(full), 6), np.int32)
        assigned = unassigned = outside = 0
        for slot, boundaries in enumerate(load_boundary_slots(track.annotation_zip, member)):
            for note in boundaries:
                nearest = nearest_assignment(int(note.onset_sample), flat, rows)
                if nearest is None:
                    unassigned += 1
                    continue
                row, _ = nearest
                counts[row] += 1
                occupied[row, slot] += 1
                assigned += 1
                relative = int(note.onset_sample) - int(fields['cluster_start_samples'][row])
                outside += int(not COVERED.centers[0] <= relative <= COVERED.centers[-1])
        np.testing.assert_array_equal(counts, exact)
        np.testing.assert_array_equal(occupied > 0, slots > .5)
        if outside:
            raise RuntimeError('assigned training event outside covered time axis')
        path = directory / 'v100-spectral-shard-00.npz'
        v100._save_spectral_cache(path, cache, spectra, slots, window=COVERED)
        reports.append(dict(member=member, rows=len(exact), poly_rows=int((exact >= 2).sum()),
            fold=cfg['member_folds'][member], spectral_sha256=digest(path),
            assigned_annotations=assigned, unassigned_annotations=unassigned,
            same_string_collisions=int(slot_diag['same_slot_collisions']),
            occupancy_count_mismatches=int((slots.sum(1) != np.minimum(exact, 6)).sum()),
            independent_k_targets_equal=True, old_spectral_prefix_equal=True, runtime_cache_equal=True,
            outside_covered_centers=outside))
        write_json(args.output_dir / 'progress.json', dict(completed=len(reports), expected=19, tracks=reports))
        del streams, records, x88, out88, clusters, sequence, spectra, old, live, cache
        gc.collect()
    if len(reports) != 19:
        raise RuntimeError('incomplete training shard')
    write_json(args.output_dir / 'report.json', dict(status='passed', part=args.part, parts=10,
        tracks=reports, config_sha256=digest(args.config), source_state_sha256=digest(args.source_dir / 'rebuild-state.json'),
        proposal_source_kind=source['source_kind'], source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        external_fold=3, outer_tracks_processed=0, candidate_mining_shared_between_arms=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'dataset-dir', 'config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--part', type=int, required=True)
    p.add_argument('--parts', type=int, default=10)
    run(p.parse_args())
