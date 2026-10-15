"""S600: raw frozen BP outputs and explicit decoder loops on TRAIN only.

Never reads reserved test audio or annotations. Cached source bytes, clips,
posteriors and provenance are immutable. Development labels are explicit.
"""
from __future__ import annotations
import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics
from scripts.s592_s597_urmp_prospective import (
    read, fetch, decode_prepared, parse_notes, truth_groups, digest, OFFSET, SECONDS, SR)
from scripts.s277_original_freeze_dataset3_native_prelabel import pcm

MANIFEST=Path('analysis/v273_s599_urmp_train_test_metadata_only.json')
MANIFEST_SHA='83e8a211e7e93e75809bc84a6b5469173956f86c7b66a02766c911547edab746'


def train_specs(old_only=False):
    need(sha(MANIFEST)==MANIFEST_SHA,'S599 committed manifest changed')
    m=read(MANIFEST)
    need(not ({s['piece'] for s in m['train']} & {s['piece'] for s in m['test']}), 'Composition leakage')
    return [s for s in m['train'] if not old_only or s['piece'] in m['old_exposed_pieces']
            and s['source_id'] in {v['source_id'] for v in read('analysis/v273_s592_urmp_metadata_only.json')['selected']}]


def extract(out, old_only=False):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, run_inference
    import importlib.metadata
    model=Model(ICASSP_2022_MODEL_PATH)
    out.mkdir(parents=True,exist_ok=True)
    versions={p:importlib.metadata.version(p) for p in ('basic-pitch','tensorflow','numpy','scipy','librosa')}
    weight_hashes={str(p.relative_to(ICASSP_2022_MODEL_PATH)):sha(p)
        for p in sorted(Path(ICASSP_2022_MODEL_PATH).rglob('*')) if p.is_file()}
    for spec in train_specs(old_only):
        sid=spec['source_id']; folder=out/sid; folder.mkdir(exist_ok=True)
        if (folder/'audio_report.json').exists():
            r=read(folder/'audio_report.json')
            need(r['manifest_sha256']==MANIFEST_SHA and r['raw_sha256']==sha(folder/'raw.npz'), 'Immutable posterior cache')
            print('S600 CACHE',spec['folder'],flush=True);continue
        data=fetch(spec['audio']); signal=decode_prepared(data)
        clip=signal[int(OFFSET*SR):int((OFFSET+SECONDS)*SR)]
        (folder/'clip.wav').write_bytes(pcm(clip))
        raw=run_inference(str(folder/'clip.wav'),model)
        need(set(raw)=={'onset','note','contour'} and all(np.isfinite(v).all() for v in raw.values()),'Invalid BP posterior')
        np.savez_compressed(folder/'raw.npz',**raw)
        save(folder/'audio_report.json',dict(source_id=sid,folder=spec['folder'],split='train',
            manifest_sha256=MANIFEST_SHA,source_audio_sha256=digest(data),clip_sha256=sha(folder/'clip.wav'),
            raw_sha256=sha(folder/'raw.npz'),shapes={k:list(v.shape) for k,v in raw.items()},
            versions=versions,weight_hashes=weight_hashes,labels_used_for_audio_inference=False,
            original_rate=16000,inference_wav_rate=44100,offset=OFFSET,seconds=SECONDS))
        print('S600 AUDIO',spec['folder'],{k:v.shape for k,v in raw.items()},flush=True)


def labels(out, old_only=False):
    """Only manifest TRAIN annotations, never test. Exposed data for development."""
    out.mkdir(parents=True,exist_ok=True)
    all_truth={};proof={}
    def one(spec):
        sid=spec['source_id'];folder=out/sid;folder.mkdir(exist_ok=True)
        if not (folder/'training_annotations.json').exists():
            events=[]; hashes=[]
            for index,item in enumerate(spec['annotations']):
                data=fetch(item,annotation=True)
                ins=Path(item['path']).name.split('_')[2]
                events.extend(parse_notes(data,ins,index+1))
                hashes.append(dict(path=item['path'],sha256=digest(data)))
            save(folder/'training_annotations.json',dict(split='train',source_id=sid,events=events,source_hashes=hashes))
        r=read(folder/'training_annotations.json')
        print('S600 TRAIN ANNOTATION',spec['folder'],flush=True)
        return sid,truth_groups(r['events']),sha(folder/'training_annotations.json')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sid,truth,h in pool.map(one,train_specs(old_only)):
            all_truth[sid]=truth;proof[sid]=h
    name='truth_old5.json' if old_only else 'truth_train37.json'
    record=dict(development_only=True,manifest_sha256=MANIFEST_SHA,truth=all_truth,proof=proof)
    if (out/name).exists():
        need(read(out/name)['truth']==all_truth,'Immutable training truth changed')
        save(out/(name.replace('.json','_byte_verification.json')),record)
    else:
        save(out/name,record)
    print('S600 TRAIN LABELS',len(all_truth),sum(len(x['notes']) for g in all_truth.values() for x in g),flush=True)


def decode(raw, onset=.5, frame=.3, length=11, melodia=True, energy=11, infer=True):
    from basic_pitch.note_creation import output_to_notes_polyphonic, model_frames_to_time
    times=model_frames_to_time(len(raw['note']))
    notes=output_to_notes_polyphonic(raw['note'],raw['onset'],onset_thresh=onset,
        frame_thresh=frame,min_note_len=length,melodia_trick=melodia,energy_tol=energy,infer_onsets=infer,
        min_freq=None,max_freq=None)
    return sorted([dict(time=float(times[st]),end=min(SECONDS,float(times[en])),notes=[int(p)],score=float(a))
        for st,en,p,a in notes if 40<=p<=88 and 0<=times[st]<SECONDS and en>st],key=lambda x:(x['time'],x['notes']))


def sweep(out, report, old_only=False):
    need(not report.exists(),'Every decoder loop has an immutable report')
    specs=train_specs(old_only); name='truth_old5.json' if old_only else 'truth_train37.json'
    truth=read(out/name)['truth']; need(set(truth)=={s['source_id'] for s in specs},'All declared development sources')
    meta={s['source_id']:dict(composer=s['folder']) for s in specs}
    posterior={s['source_id']:dict(np.load(out/s['source_id']/'raw.npz',allow_pickle=False)) for s in specs}
    scores={};configs={};predictions={}
    for onset,frame,length in itertools.product((.4,.5,.6),(.2,.3,.4),(3,5,8,11)):
        key=f'o{onset}_f{frame}_l{length}'; config=dict(onset=onset,frame=frame,length=length)
        pred={sid:decode(raw,**config) for sid,raw in posterior.items()}
        scores[key]=metrics({key:pred},truth,meta)[key]; configs[key]=config
        predictions[key]=pred
        print('S600 DECODER DEVELOPMENT',key,scores[key]['fullstream']['80'],flush=True)
    best=max(scores,key=lambda k:scores[k]['fullstream']['80']['F1'])
    save(report,dict(development_only=True,configs=configs,scores=scores,best=best,
        manifest_sha256=MANIFEST_SHA,truth_sha256=sha(out/name),
        raw_hashes={s:sha(out/s/'raw.npz') for s in posterior}))
    save(report.with_suffix('.predictions.json'),predictions)
    print('S600 BEST DEVELOPMENT',best,scores[best]['fullstream']['80'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('extract','labels','sweep'))
    p.add_argument('--out',type=Path,default=Path('model/S600'));p.add_argument('--old-only',action='store_true')
    p.add_argument('--report',type=Path,default=Path('model/S600/decoder_sweep_old5.json'))
    a=p.parse_args()
    if a.stage=='extract':extract(a.out,a.old_only)
    elif a.stage=='labels':labels(a.out,a.old_only)
    else:sweep(a.out,a.report,a.old_only)
