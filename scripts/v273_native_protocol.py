"""Frozen composition partitions and verified rebuilt inputs for native A/B."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import numpy as np

from scripts.train_boundaries import group_stem
from scripts.rebuild_v273_sources import digest


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    mapping, groups = cfg['member_folds'], cfg['composition_folds']
    if len(mapping) != cfg['expected_tracks'] or len(groups) != cfg['expected_compositions']:
        raise RuntimeError('frozen partition inventory mismatch')
    if set(mapping.values()) != set(range(5)):
        raise RuntimeError('expected five frozen folds')
    for member, fold in mapping.items():
        if groups.get(group_stem(member)) != fold:
            raise RuntimeError('composition split across folds')
    if [f['fold'] for f in cfg['folds']] != list(range(5)):
        raise RuntimeError('invalid fold budget inventory')
    for f in cfg['folds']:
        if f['meta_fold'] != min(set(range(5)) - {f['fold']}):
            raise RuntimeError('meta-validation fold changed')
        for name, limit in (('v260_uniform', 20), ('v260_weighted', 20), ('v272_uniform', 30)):
            if not 1 <= f['epochs'][name] <= limit:
                raise RuntimeError('invalid archived epoch budget')
    return cfg


def frozen_group_folds(cache, train_split, manifest):
    cfg = load_config(manifest)
    mapping, assignment = cfg['member_folds'], cfg['composition_folds']
    if {t.annotation_member for t in train_split} != set(mapping):
        raise RuntimeError('train tracks differ from archived V27.3 partitions')
    if set(map(str, cache['track_members'])) != set(mapping):
        raise RuntimeError('native cache track inventory mismatch')
    counts = Counter()
    for member in cache['members']:
        member = str(member)
        if member not in mapping:
            raise RuntimeError('cache row outside frozen train tracks')
        counts[group_stem(member)] += 1
    groups = [sorted(g for g, fold in assignment.items() if fold == f) for f in range(5)]
    loads = [sum(counts[g] for g in group) for group in groups]
    if not all(loads):
        raise RuntimeError('empty frozen fold')
    return dict(assignment), groups, loads, counts


def verify_native_cache(cache_dir, config):
    """Verify all eight source files before training, without loading spectra."""
    cfg = load_config(config)
    paths = sorted(Path(cache_dir).rglob('v100-spectral-shard-*.npz'))
    if len(paths) != 8 or {p.name for p in paths} != {f'v100-spectral-shard-{i:02d}.npz' for i in range(8)}:
        raise RuntimeError('expected eight unique native cache shards')
    rows = poly = 0
    tracks = []
    hashes = {}
    for path in paths:
        provenance = json.loads((path.parent.parent/'provenance.json').read_text())
        if provenance['source_kind'] != cfg['source_kind'] or provenance['source_code_sha'] != cfg['source_commit']:
            raise RuntimeError('rebuilt cache producer mismatch')
        actual = digest(path)
        if actual != provenance['sha256']:
            raise RuntimeError('rebuilt native cache checksum mismatch')
        hashes[path.name] = actual
        with np.load(path, allow_pickle=False) as z:
            if not {'spectral', 'stats', 'sequence', 'mask', 'target', 'exact', 'members',
                    'top_samples', 'slot_targets', 'track_members'}.issubset(z.files):
                raise RuntimeError('incomplete native cache')
            rows += len(z['exact'])
            poly += int(np.sum(z['exact'] >= 2))
            tracks.extend(z['track_members'].astype(str).tolist())
    if rows != cfg['expected_rows'] or poly != cfg['expected_poly_rows']:
        raise RuntimeError('rebuilt cache row counts changed')
    if len(tracks) != len(set(tracks)) or set(tracks) != set(cfg['member_folds']):
        raise RuntimeError('rebuilt cache track coverage mismatch')
    return {'rows': rows, 'poly_rows': poly, 'tracks': len(tracks), 'sha256': hashes}
