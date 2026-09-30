"""Build only the fixed inner fit/validation maps; no annotations are consumed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.rebuild_v273_sources import digest, verify_dataset, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.train_v100_spectral_string_slots import decode_pcm16_mono_wav
from scripts.v273_ownership_experiment import load_geometry, sorted_proposals
from scripts.v273_residual_map import CONFIG, innovation, time_frequency_map, ownership_input
from scripts.v273_residual_native import experiment_identity
from scripts.v273_window_experiment import load_bundle, require, array_hash


def construct(task):
    track, path, ids, starts, watermarks, decisions, historical, output = task
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['global_index'],ids)
        require(str(z['member'][0]) == track.annotation_member,'timing member changed')
        flat,offsets = z['full_candidate_samples'],z['full_candidate_offsets']
        groups = [flat[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    np.testing.assert_array_equal(starts,[g[0] for g in groups])
    samples = np.asarray(decode_pcm16_mono_wav(track.audio_zip,track.audio_member).samples,np.float32)/32768.
    residual,blocks = innovation(samples)
    maps = np.empty((len(ids),31,64,4),np.float16)
    masks = np.empty((len(ids),512),np.uint8)
    fractions = np.empty((len(ids),31,2),np.float32)
    proposals = sorted_proposals(groups)
    error = 0.
    for i,origin in enumerate(starts):
        value = time_frequency_map(samples,residual,int(origin))
        maps[i] = value
        require(np.isfinite(maps[i]).all(),'nonfinite map')
        error = max(error,float(np.max(np.abs(value-maps[i].astype(np.float32)))))
        masks[i],fractions[i] = ownership_input(groups,i,complete_through=int(watermarks[i]),
            decision_sample=int(decisions[i]),proposals=proposals)
        np.testing.assert_array_equal(fractions[i,:,0],historical[i,:,1])
    target = Path(output)/'tracks'/path.name
    np.savez_compressed(target,maps=maps,ownership_packed=masks,ownership_fraction=fractions,
        global_index=ids,start=starts,member=np.array([track.annotation_member]))
    return dict(member=track.annotation_member,rows=len(ids),file=str(target),sha256=digest(target),
        stream_blocks=len(blocks),max_pole_radius=max(b['max_pole_radius_per_lag'] for b in blocks),
        max_float16_error=error,reused_validation_cache=False,timing_sha256=digest(path))


def prepare(args):
    require(not args.output.exists(),'output already exists')
    identity = experiment_identity()
    verify_dataset(args.dataset)
    validate_original_inventory(args.bundle)
    cache,parts,manifest = load_bundle(args.bundle,args.config)
    geometry,geometry_rows,geometry_report = load_geometry(args.geometry,cache,manifest,args.config)
    cfg = json.loads(args.config.read_text())
    allowed = {m for m,f in cfg['member_folds'].items() if f != 3}
    tracks = {t.annotation_member:t for t in index_guitarset(args.dataset) if t.annotation_member in allowed}
    require(len(tracks) == len(allowed) == 190,'wrong training track inventory')
    ids = np.sort(np.r_[parts['inner_fit'],parts['inner_val']])
    require(len(ids) == 59309 and len(parts['inner_fit']) == 43357 and len(parts['inner_val']) == 15952,
            'wrong training population')
    reuse = {}
    if args.validation_cache:
        source = json.loads((args.validation_cache/'report.json').read_text())
        require(source['feature_builder_sha256'] == identity['feature_builder_sha256'] and
                source['estimator_sha256'] == identity['stable_estimator_sha256'] and
                source['parameters'] == CONFIG and source['representation_gate']['passed'],
                'validation cache differs from audited representation')
        reuse = {r['member']:r for r in source['track_replay']}
        require(set(reuse) == {m for m,f in cfg['member_folds'].items() if f == 0},'wrong reused tracks')
    args.output.mkdir(parents=True);(args.output/'tracks').mkdir()
    maps = np.lib.format.open_memmap(args.output/'maps.npy',mode='w+',dtype=np.float16,
                                    shape=(len(cache['members']),31,64,4))
    maps[:] = np.nan
    packed = np.lib.format.open_memmap(args.output/'ownership-packed.npy',mode='w+',dtype=np.uint8,
                                      shape=(len(cache['members']),512))
    packed[:] = 0
    tasks = []; reports = []; started = time.monotonic()
    for track_record in geometry_report['tracks']:
        member = track_record['member']; selected = np.flatnonzero(cache['members'] == member)
        require(member in allowed,'outer track in geometry')
        path = args.geometry/track_record['timing_file']
        if member in reuse:
            old = reuse[member]; target = args.validation_cache/old['map_file']
            require(digest(target) == old['map_sha256'] and digest(path) == old['timing_sha256'],
                    'reused cache or timing changed')
            with np.load(target,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['global_index'],selected)
                np.testing.assert_array_equal(z['start'],cache['cluster_start_samples'][selected])
                np.testing.assert_array_equal(z['ownership_fraction'][:,:,0],geometry[selected,:,1])
                maps[selected] = z['maps']; packed[selected] = z['ownership_packed']
            reports.append(dict(member=member,rows=len(selected),file=str(target),sha256=digest(target),
                stream_blocks=old['stream_blocks'],max_pole_radius=old['max_pole_radius'],
                reused_validation_cache=True,timing_sha256=digest(path)))
            continue
        tasks.append((tracks[member],path,selected,np.asarray(cache['cluster_start_samples'][selected]),
            geometry_rows['proposal_complete_through'][selected],geometry_rows['decision_sample_support'][selected],
            np.asarray(geometry[selected]),str(args.output)))
    write_json(args.output/'progress.json',dict(status='preparing',completed_tracks=len(reports),expected_tracks=190))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(construct,t) for t in tasks]
        for future in as_completed(futures):
            r = future.result();reports.append(r)
            with np.load(r['file'],allow_pickle=False) as z:
                maps[z['global_index']] = z['maps'];packed[z['global_index']] = z['ownership_packed']
            progress = dict(status='preparing',completed_tracks=len(reports),expected_tracks=190,
                            member=r['member'],seconds=round(time.monotonic()-started,1))
            write_json(args.output/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    require(len(reports) == 190 and {r['member'] for r in reports} == allowed,'missing maps')
    for start in range(0,len(ids),256):
        value = maps[ids[start:start+256]]
        require(np.isfinite(value).all() and (value>=0).all(),'invalid inner maps')
    for start in range(0,len(parts['outer']),256):
        require(np.isnan(maps[parts['outer'][start:start+256]]).all(),'outer maps generated')
    maps.flush();packed.flush()
    # Publish only portable per-track identities, not scratch paths.
    for r in reports:
        r['file'] = Path(r['file']).name
    report = dict(status='prepared',**identity,numpy=np.__version__,parameters=CONFIG,
        rows=59309,fit_rows=43357,validation_rows=15952,tracks=190,outer_rows_evaluated=0,
        annotations_consumed=False,output_corrector=False,training_performed=False,
        bundle_sha256=digest(args.bundle/'bundle.json'),geometry_report_sha256=digest(args.geometry/'report.json'),
        fit_indices_sha256=array_hash(parts['inner_fit']),val_indices_sha256=array_hash(parts['inner_val']),
        maps_sha256=digest(args.output/'maps.npy'),ownership_sha256=digest(args.output/'ownership-packed.npy'),
        maps_shape=list(maps.shape),maps_dtype=str(maps.dtype),
        max_pole_radius=max(r['max_pole_radius'] for r in reports),
        reused_validation_tracks=sum(r['reused_validation_cache'] for r in reports),
        track_replay=sorted(reports,key=lambda r:r['member']),script_sha256=digest(__file__),
        elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'report.json',report)
    write_json(args.output/'progress.json',dict(status='prepared',completed_tracks=190,expected_tracks=190))
    print(json.dumps({k:v for k,v in report.items() if k not in ('track_replay','parameters')}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','bundle','geometry','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--validation-cache',type=Path)
    p.add_argument('--workers',type=int,default=6)
    prepare(p.parse_args())
