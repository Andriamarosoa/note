"""Restore full pinned proposals and construct annotation-free paired geometry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import numpy as np

from scripts.candidate_timing import full_samples
from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import extract_verified, validate_original_inventory
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, array_hash, require, ARRAY_KEYS
from scripts.v273_ownership_experiment import geometry_for_track


def prepare(args):
    require(not args.output.exists(),'geometry output exists')
    launch=json.loads(args.launch.read_text())
    cfg=load_config(args.config)
    require(digest(args.config)==launch['config_sha256'],'partition protocol differs')
    validate_original_inventory(args.bundle)
    cache,parts,manifest=load_bundle(args.bundle,args.config)
    args.download.mkdir(parents=True,exist_ok=True)
    def restore(item):
        archive=args.download/item['name']
        subprocess.run(['gh','release','download',launch['source_release'],'--repo','Andriamarosoa/note',
                        '--pattern',item['name'],'--dir',str(args.download)],check=True)
        target=args.download/archive.stem
        extract_verified(archive,target,item['sha256'])
        return target
    require(len(launch['training_archives'])==10,'missing training shards')
    with ThreadPoolExecutor(max_workers=3) as pool:
        roots=list(pool.map(restore,launch['training_archives']))
    allowed={m for m,f in cfg['member_folds'].items() if f!=3}
    require(len(allowed)==190,'training inventory changed')
    args.output.mkdir(parents=True)
    (args.output/'full-timing').mkdir()
    geometry=np.lib.format.open_memmap(args.output/'geometry.npy',mode='w+',dtype=np.float32,
                                     shape=(len(cache['members']),31,2))
    geometry[:]=np.nan
    n=len(cache['members']); losses=np.full(n,-1,np.int32)
    watermarks=np.full(n,-1,np.int64); decision=np.full(n,-1,np.int64)
    seen=set(); records=[]; source_hashes={}
    for root in roots:
        prep=json.loads((root/'report.json').read_text())
        require(prep['status']=='passed' and prep['outer_tracks_processed']==0 and
                prep['config_sha256']==digest(args.config),'wrong source preparation')
        for path in sorted(root.rglob('v100-spectral-shard-*.npz')):
            with np.load(path,allow_pickle=False) as z:
                names=set(z['members'].astype(str));require(len(names)==1,'mixed track')
                member=next(iter(names))
                require(member in allowed and member not in seen,'outer/duplicate track')
                require(digest(path)==manifest['source_cache_sha256'][member],'source bytes differ')
                ids=np.flatnonzero(cache['members']==member)
                require(len(ids)==len(z['exact']),'source rows differ')
                for key in ARRAY_KEYS:
                    np.testing.assert_array_equal(z[key],cache[key][ids])
                groups=full_samples(z)
                g,l,w,d=geometry_for_track(groups)
                geometry[ids]=g;losses[ids]=l;watermarks[ids]=w;decision[ids]=d
                target=args.output/'full-timing'/f'track-{len(records):03d}.npz'
                np.savez_compressed(target,global_index=ids,member=np.asarray([member]),
                    full_candidate_samples=z['full_candidate_samples'],
                    full_candidate_offsets=z['full_candidate_offsets'])
                source_hashes[member]=digest(path)
                records.append(dict(member=member,rows=len(ids),fold=cfg['member_folds'][member],
                    competition_rows=int((l>0).sum()),
                    model_geometry_changed_rows=int(np.any(g[:,:,0]!=g[:,:,1],axis=1).sum()),
                    max_decision_support_ms=float(np.max(d-cache['cluster_start_samples'][ids])*1000/44100),
                    timing_file=target.relative_to(args.output).as_posix(),timing_sha256=digest(target)))
                seen.add(member)
            print(json.dumps(dict(completed_tracks=len(seen),member=member)),flush=True)
    require(seen==allowed,'incomplete proposal inventory')
    inner=np.r_[parts['inner_fit'],parts['inner_val']]
    require(len(inner)==59309 and len(parts['inner_val'])==15952,'wrong population')
    require(np.isfinite(geometry[inner]).all() and (geometry[inner]>=0).all() and
            (geometry[inner]<=1).all(),'invalid ownership features')
    require(np.isnan(geometry[parts['outer']]).all() and (losses[parts['outer']]==-1).all(),
            'outer context was generated')
    np.testing.assert_array_equal(losses[inner]>0,np.any(geometry[inner,:,0]!=geometry[inner,:,1],axis=1))
    geometry.flush()
    np.savez_compressed(args.output/'rows.npz',member=cache['members'],start=cache['cluster_start_samples'],
        lost_eligible_samples=losses,proposal_complete_through=watermarks,decision_sample_support=decision)
    write_json(args.output/'report.json',dict(status='verified',experiment='native_ownership_context',
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        launch_sha256=digest(args.launch),config_sha256=digest(args.config),
        bundle_fields=manifest['fields'],source_cache_sha256=source_hashes,
        geometry_sha256=array_hash(geometry),tracks=records,rows=59309,
        input_columns=['local_eligibility_fraction','neighbor_ownership_fraction'],
        label_free_builder=True,full_untruncated_candidates=True,all_sample_prefix_checks_passed=True,
        frame_support_samples=256,hop_samples=128,frames=31,
        candidate_merge_lookahead_samples=5,
        scope='190 non-outer tracks only; internal fit and validation of outer fold 3',
        outer_tracks_processed=0,outer_rows_evaluated=0,
        timing_limit='Sample-support bound on the frozen causal proposal stream; no measured end-to-end latency.',
        competition_rows=int((losses[inner]>0).sum())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','config','launch','download','output'):
        p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
