"""Extract fresh, unsupervised native NOTE DSP at every MIDI256 step.

Only the signed FIT80 track names, durations and annotations are reused from
S800. Its learned posteriors and historical MIDI are NOT prediction inputs.
No CAL/HELD/TEST audio or annotations are extracted. Cache is research-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.io import wavfile
from causal_note.strict_live import SR, HOP, feature_track
from causal_note.s706_sharp_spectra import sharp_track

SCHEMA = 'S813_FIT80_FRESH_UNSUPERVISED_DSP_MIDI256_V1'


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True,
                                   allow_nan=False) + '\n', encoding='utf-8')


def targets_at_ends(truth, ends):
    pitch = np.zeros((len(ends), 49), bool)
    cardinality = np.zeros(len(ends), np.int16)
    times = ends / SR
    for note in truth:
        p = int(note['pitch']) - 40
        if not 0 <= p < 49:
            raise ValueError('FIT truth outside original MIDI40..88 scope')
        mask = (times >= note['time']) & (times < note['time'] + note['duration'])
        pitch[mask, p] = True
        cardinality += mask
    return pitch, np.minimum(cardinality, 6).astype(np.uint8)


def extract_track(audio, *, block_size=256):
    ends, native = feature_track(audio, block_size=block_size, hop=HOP)
    sharp = sharp_track(audio, ends)
    if native.shape != (len(ends), 49, 126) or sharp.shape != (len(ends), 49, 96):
        raise AssertionError('Native/sharp DSP ABI changed')
    x = np.concatenate((native, sharp), axis=-1).astype(np.float16)
    if not np.isfinite(x).all() or not np.array_equal(ends, np.arange(256, len(audio)+1, 256)):
        raise AssertionError('Fresh uniform MIDI256 features required')
    return ends, x


def prepare_one(job):
    row, traces, audio_zip, outdir = job
    outdir, traces = Path(outdir), Path(traces)
    stem = f"fit_{row['index']:03d}"
    meta_path = traces / (stem + '.json')
    if digest_file(meta_path) != row['json_sha256']:
        raise AssertionError('S800 annotation/source metadata SHA mismatch')
    meta = json.loads(meta_path.read_text())
    if meta['name'] != row['name'] or meta['composition'] != row['composition']:
        raise AssertionError('FIT track identity changed')
    member = meta['name'] + '_mic.wav'
    with zipfile.ZipFile(audio_zip) as archive:
        blob = archive.read(member)
    rate, pcm = wavfile.read(io.BytesIO(blob))
    if rate != SR or pcm.ndim != 1 or pcm.dtype not in (np.dtype('int16'), np.dtype('int32')):
        raise ValueError('Require original mono 44100 Hz integer PCM; no offline resampling')
    samples = int(meta['samples'])
    if samples > len(pcm):
        raise ValueError('Source duration exceeds real PCM')
    audio = pcm[:samples].astype(np.float32) / (32768. if pcm.dtype == np.int16 else 2147483648.)
    started = time.perf_counter()
    ends, x = extract_track(audio)
    extraction_seconds = time.perf_counter() - started
    # Labels are read for scoring/training only, after extraction has finished.
    y, k = targets_at_ends(meta['fit_truth'], ends)
    with np.load(traces / (stem + '.npz'), allow_pickle=False) as old:
        if not np.array_equal(ends, old['metric_ends']):
            raise AssertionError('New acoustic256 and source metric256 differ')
    arrays = {'features': x, 'ends': ends, 'pitch_target': y, 'count_target': k}
    files = {}
    for name, value in arrays.items():
        path = outdir / (stem + '_' + name + '.npy')
        np.save(path, value, allow_pickle=False)
        files[name] = {'file': path.name, 'sha256': digest_file(path)}
    result = dict(index=row['index'], name=meta['name'], composition=meta['composition'],
        texture=meta['texture'], samples=samples, frames=len(ends), shape=list(x.shape),
        audio_member=member, audio_member_sha256=hashlib.sha256(blob).hexdigest(),
        original_json_sha256=row['json_sha256'], files=files,
        extraction_seconds=extraction_seconds,
        extraction_mean_ms_per_frame=1000*extraction_seconds/len(ends),
        extraction_timing_is_not_live_latency=True)
    save_json(outdir / (stem + '_manifest.json'), result)
    print('S813_FRESH256', row['index'], meta['name'], len(ends),
          round(extraction_seconds, 2), flush=True)
    return result


def prepare(args):
    if args.out.exists():
        raise FileExistsError('Immutable S813 extraction output already exists')
    source = json.loads(args.manifest.read_text())
    if (source.get('schema') != 'S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1'
            or not source.get('complete') or not source.get('FIT_source_parity_S795')
            or any(source.get(k) for k in ('CAL_opened','HELD_opened','TEST_opened'))):
        raise ValueError('Authentic completed FIT80 manifest required')
    rows = sorted(source['fit_files'], key=lambda r: r['index'])
    if len(rows) != 80 or [r['index'] for r in rows] != list(range(80)):
        raise ValueError('Exactly all 80 FIT tracks required')
    if set(r['composition'] for r in rows) != {'BN1','Funk2','Jazz1','Rock3','SS1'}:
        raise ValueError('FIT composition scope changed')
    for row in rows:
        for suffix, field in (('.json','json_sha256'),('.npz','npz_sha256')):
            if digest_file(args.traces / (f"fit_{row['index']:03d}"+suffix)) != row[field]:
                raise ValueError('S800 source trace SHA mismatch')
    registry = json.loads(args.protocol.read_text())['guitar_archives']['GuitarSet_audio']
    if args.audio_zip.stat().st_size != registry['size'] or digest_file(args.audio_zip) != registry['sha256']:
        raise ValueError('Pinned full GuitarSet audio archive SHA mismatch')
    args.out.mkdir(parents=True)
    report = dict(schema=SCHEMA, complete=False, source_manifest_sha256=digest_file(args.manifest),
        audio_archive_sha256=registry['sha256'], source_run=39352880035,
        sample_rate_hz=SR, feature_hop_samples=256, decision_hop_samples=256,
        metric_hop_samples=256, resolution_ms=256/SR*1000, feature_dimension=222,
        feature_dtype='float16_to_float32', learned_frontend=False,
        H0_events_or_posteriors_used_as_features=False, added_lookahead_samples=0,
        fit_historically_explored=True, generalization_verified=False,
        CAL_opened=False, HELD_opened=False, TEST_opened=False, promotion=False)
    save_json(args.out/'report.json', report)
    jobs = [(r, str(args.traces), str(args.audio_zip), str(args.out)) for r in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(prepare_one, jobs))
    report.update(complete=True, tracks=records, total_frames=sum(r['frames'] for r in records))
    if report['total_frames'] != 306536:
        raise AssertionError('Authentic FIT80 frame count changed')
    save_json(args.out/'report.json', report)
    print('S813_EXTRACTION_COMPLETE', report['total_frames'], flush=True)


def main():
    p = argparse.ArgumentParser()
    for arg in ('manifest', 'traces', 'audio-zip', 'out'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--protocol', type=Path,
                   default=Path('analysis/evidence/v273-s613-strict-live/protocol.json'))
    p.add_argument('--workers', type=int, choices=(1,2,3), default=2)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
