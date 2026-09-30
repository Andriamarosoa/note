"""Prepare matched original/octave-up maps on the existing inner tracks only."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.rebuild_v273_sources import digest,verify_dataset,write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.train_v100_spectral_string_slots import decode_pcm16_mono_wav
from scripts.v273_ownership_experiment import load_geometry
from scripts.v273_high_pitch_map import CONFIG,time_frequency_map
from scripts.v273_high_pitch_native import experiment_identity
from scripts.v273_high_pitch_registers import prepare as prepare_registers
from scripts.v273_window_experiment import load_bundle,require,array_hash


def construct(task):
    track,ids,starts,output=task
    samples=np.asarray(decode_pcm16_mono_wav(track.audio_zip,track.audio_member).samples,np.float32)/32768.
    maps=np.empty((len(ids),31,64,4),np.float16)
    error=0.
    for i,start in enumerate(starts):
        value=time_frequency_map(samples,int(start));maps[i]=value
        require(np.isfinite(maps[i]).all(),'float16 overflow')
        error=max(error,float(np.max(np.abs(value-maps[i].astype(np.float32)))))
    target=Path(output)/'tracks'/('track-'+str(int(ids[0]))+'.npz')
    np.savez_compressed(target,maps=maps,global_index=ids,start=starts,member=np.array([track.annotation_member]))
    return dict(member=track.annotation_member,rows=len(ids),file=str(target),sha256=digest(target),max_float16_error=error)


def prepare(args):
    require(not args.output.exists(),'output already exists')
    identity=experiment_identity();verify_dataset(args.dataset)
    validate_original_inventory(args.bundle)
    cache,parts,manifest=load_bundle(args.bundle,args.config)
    _,_,geometry_report=load_geometry(args.geometry,cache,manifest,args.config)
    cfg=json.loads(args.config.read_text())
    allowed={m for m,f in cfg['member_folds'].items() if f!=3}
    tracks={t.annotation_member:t for t in index_guitarset(args.dataset) if t.annotation_member in allowed}
    require(set(tracks)==allowed and len(tracks)==190,'wrong inner track inventory')
    ids=np.sort(np.r_[parts['inner_fit'],parts['inner_val']])
    require(len(ids)==59309 and len(parts['inner_fit'])==43357 and len(parts['inner_val'])==15952,'wrong partitions')
    args.output.mkdir(parents=True);(args.output/'tracks').mkdir()
    maps=np.lib.format.open_memmap(args.output/'maps.npy',mode='w+',dtype=np.float16,shape=(len(cache['members']),31,64,4))
    maps[:]=np.nan
    tasks=[];reports=[];started=time.monotonic()
    for member in sorted(allowed):
        selected=np.flatnonzero(cache['members']==member)
        tasks.append((tracks[member],selected,np.asarray(cache['cluster_start_samples'][selected]),str(args.output)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(construct,t) for t in tasks]
        for future in as_completed(futures):
            record=future.result();reports.append(record)
            with np.load(record['file'],allow_pickle=False) as z:
                np.testing.assert_array_equal(cache['members'][z['global_index']],np.repeat(record['member'],len(z['global_index'])))
                np.testing.assert_array_equal(cache['cluster_start_samples'][z['global_index']],z['start'])
                maps[z['global_index']]=z['maps']
            progress=dict(status='preparing',completed_tracks=len(reports),expected_tracks=190,member=record['member'],seconds=round(time.monotonic()-started,1))
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    require(len(reports)==190 and {r['member'] for r in reports}==allowed,'missing maps')
    for start in range(0,len(ids),256):
        value=maps[ids[start:start+256]]
        require(np.isfinite(value).all() and (value>=0).all(),'invalid inner maps')
    for start in range(0,len(parts['outer']),256):
        require(np.isnan(maps[parts['outer'][start:start+256]]).all(),'outer maps generated')
    maps.flush()
    # Evaluation-only annotations are created separately, after label-free maps.
    registers=prepare_registers(args.dataset/'annotation.zip',args.geometry,cache,parts['inner_val'],args.output/'validation-registers.npz')
    for r in reports:r['file']=Path(r['file']).name
    report=dict(status='prepared',**identity,numpy=np.__version__,parameters=CONFIG,
        rows=59309,fit_rows=43357,validation_rows=15952,tracks=190,outer_rows_evaluated=0,
        feature_builder_consumes_annotations=False,output_corrector=False,training_performed=False,
        bundle_sha256=digest(args.bundle/'bundle.json'),geometry_report_sha256=digest(args.geometry/'report.json'),
        fit_indices_sha256=array_hash(parts['inner_fit']),val_indices_sha256=array_hash(parts['inner_val']),
        maps_sha256=digest(args.output/'maps.npy'),maps_shape=list(maps.shape),maps_dtype=str(maps.dtype),
        validation_registers=registers,track_replay=sorted(reports,key=lambda r:r['member']),
        script_sha256=digest(__file__),elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('track_replay','parameters')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','bundle','geometry','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    prepare(p.parse_args())
