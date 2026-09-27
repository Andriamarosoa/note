"""Fixed composition partitions and memory-mapped native count inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.rebuild_v273_sources import digest, write_json
from scripts.spectral_window import COVERED, cache_window
from scripts.candidate_timing import timing_fields
from scripts.v273_native_protocol import load_config

ARRAY_KEYS = ('sequence', 'mask', 'stats', 'spectral', 'exact', 'members', 'cluster_start_samples')
FRAMES = (23, 31)
WEIGHTINGS = ('uniform', 'weighted')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def array_hash(array):
    a = np.asarray(array)
    h = hashlib.sha256(str((a.shape, str(a.dtype))).encode())
    for start in range(0, len(a), 256):
        h.update(np.ascontiguousarray(a[start:start+256]).tobytes())
    return h.hexdigest()


def partitions(members, cfg):
    names = np.asarray(members).astype(str)
    require(set(names) == set(cfg['member_folds']), 'unexpected track inventory')
    folds = np.asarray([cfg['member_folds'][m] for m in names])
    result = dict(inner_fit=np.flatnonzero((folds != 3) & (folds != 0)),
                  inner_val=np.flatnonzero(folds == 0), final_fit=np.flatnonzero(folds != 3),
                  outer=np.flatnonzero(folds == 3))
    require(all(len(v) for v in result.values()), 'empty partition')
    require(np.intersect1d(result['outer'], result['final_fit']).size == 0, 'outer leakage')
    require(np.intersect1d(result['inner_fit'], result['inner_val']).size == 0, 'inner leakage')
    require(np.array_equal(np.sort(np.r_[result['inner_fit'], result['inner_val']]), result['final_fit']), 'incomplete final fit')
    return result


def epoch_order(indices, seed, epoch):
    indices = np.asarray(indices, np.int64)
    return indices[np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(indices))]


def batch_inputs(cache, indices, frames):
    require(frames in FRAMES, 'unsupported treatment')
    return {'candidate_set': np.asarray(cache['sequence'][indices], np.float32),
            'candidate_mask': np.asarray(cache['mask'][indices], np.float32),
            'cluster_stats': np.asarray(cache['stats'][indices], np.float32),
            'spectral_map': np.asarray(cache['spectral'][indices, :frames], np.float32)}


def pack(training_root, outer_root, config, output):
    require(not output.exists(), 'bundle already exists')
    cfg = load_config(config)
    expected_source = None
    training_reports = []
    for i in range(10):
        path = training_root / f'window-training-part-{i}' / 'report.json'
        r = json.loads(path.read_text())
        require(r['part'] == i and r['parts'] == 10 and r['status'] == 'passed' and
                r['config_sha256'] == digest(config) and r['outer_tracks_processed'] == 0,
                'invalid preparation part')
        identity = (r['source_sha'], r['source_state_sha256'])
        if expected_source is None:
            expected_source = identity
        require(identity == expected_source, 'preparation producer differs')
        training_reports += r['tracks']
    require(len(training_reports) == 190, 'missing training reports')
    outer_report = json.loads((outer_root / 'report.json').read_text())
    require(outer_report['status'] == 'passed' and outer_report['outer_fold'] == 3 and
            outer_report['totals']['rows'] == 15279, 'invalid pinned outer audit')
    hashes = {r['member']: r['spectral_sha256'] for r in training_reports}
    hashes.update({r['member']: r['covered_sha256'] for r in outer_report['tracks']})
    paths = list(training_root.rglob('v100-spectral-shard-*.npz')) + list(outer_root.rglob('v100-spectral-shard-*.npz'))
    records, shapes, dtypes, seen = [], {}, {}, set()
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            require(int(z['schema_version'][0]) == 3 and cache_window(z, version=3) == COVERED, 'wrong input schema')
            timing_fields(z, required=True)
            names = set(z['members'].astype(str))
            require(len(names) == 1 and not names & seen, 'duplicate or mixed track shard')
            member = next(iter(names))
            seen |= names
            require(member in hashes and digest(path) == hashes[member], 'unverified cache bytes')
            expected_outer = path.is_relative_to(outer_root)
            require((cfg['member_folds'][member] == 3) == expected_outer, 'outer/training source mix')
            for key in ARRAY_KEYS:
                a = z[key]
                if key not in shapes:
                    shapes[key], dtypes[key] = a.shape[1:], a.dtype
                require(a.shape[1:] == shapes[key] and a.dtype == dtypes[key], 'inconsistent native array format')
                require(len(a) == len(z['exact']), 'row alignment mismatch')
                if a.dtype.kind in 'fc':
                    require(np.isfinite(a).all(), 'nonfinite input')
            records.append((member, path, len(z['exact'])))
    require(len(records) == 240 and seen == set(cfg['member_folds']), 'incomplete source tracks')
    records.sort(key=lambda x: x[0])
    size = sum(n for _, _, n in records)
    output.mkdir(parents=True)
    arrays = {key: np.lib.format.open_memmap(output / (key + '.npy'), mode='w+', dtype=dtypes[key],
                                            shape=(size, *shapes[key])) for key in ARRAY_KEYS}
    offset = 0
    for member, path, count in records:
        with np.load(path, allow_pickle=False) as z:
            for key in ARRAY_KEYS:
                arrays[key][offset:offset+count] = z[key]
        offset += count
    for a in arrays.values():
        a.flush()
    idx = partitions(arrays['members'], cfg)
    k = np.minimum(arrays['exact'].astype(np.int32), 6)
    require(len(idx['outer']) == 15279 and int((k[idx['outer']] >= 2).sum()) == 1969, 'outer population changed')
    manifest = dict(status='verified', config_sha256=digest(config), rows=size, tracks=240,
        fields={key: dict(sha256=digest(output/(key+'.npy')), shape=list(a.shape), dtype=str(a.dtype)) for key,a in arrays.items()},
        partitions={key: dict(rows=len(value), sha256=array_hash(value),
                     classes=np.bincount(k[value], minlength=7).tolist()) for key,value in idx.items()},
        source_cache_sha256=hashes, candidate_inputs_shared=True, control_spectra='literal first 23 frames of common 31-frame array',
        preparation_reports=training_reports, source_proposal_state_sha256=expected_source[1],
        full_v273_reproduction=False)
    write_json(output/'bundle.json', manifest)
    print(json.dumps({key:value for key,value in manifest.items() if key not in ('fields','source_cache_sha256','preparation_reports')}, indent=2))


def load_bundle(root, config):
    cfg = load_config(config)
    report = json.loads((root/'bundle.json').read_text())
    require(report['status'] == 'verified' and report['config_sha256'] == digest(config), 'invalid bundle identity')
    arrays = {}
    for key in ARRAY_KEYS:
        path = root/(key+'.npy')
        require(digest(path) == report['fields'][key]['sha256'], 'bundle checksum changed: '+key)
        arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    idx = partitions(arrays['members'], cfg)
    for key, value in idx.items():
        require(array_hash(value) == report['partitions'][key]['sha256'], 'partition changed')
    return arrays, idx, report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-root', 'outer-root', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    pack(a.training_root, a.outer_root, a.config, a.output)
