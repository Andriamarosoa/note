"""Train an independent direct acoustic onset head; preserve all experiments."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from dataclasses import asdict
from types import SimpleNamespace

import joblib
import numpy as np
from scipy.io import wavfile
from threadpoolctl import threadpool_limits

from onset256.acoustic_attack import AcousticEvidenceStream, AttackPolicy, DIM, SCHEMA
from onset256.acoustic_calibration import decode_cache, count_cache, truth_arrays
from onset256.model import load
from scripts.onset_event_integrity256 import COMPS, digest, dump, one, verify_inputs, metric, counts, sum_scores
from scripts.replay_native256 import match_onsets

SEED = 2026102304
PREREGISTRATION = dict(
    schema=SCHEMA, seed=SEED, feature_dim=DIM, hop_samples=256,
    future_audio_samples=0, added_hold_samples=0,
    reference=dict(tp=8050, fp=4231, fn=3217),
    target_width_hops=[6, 12],
    training="three compositions; next cyclic composition CAL; remaining OUTER",
    model=dict(max_leaf_nodes=31, max_iter=240, learning_rate=.06,
               l2_regularization=20., min_samples_leaf=40, max_bins=127),
    negative_sampling=dict(easy=.025, native_attack_above_012=.20),
    calibration="maximize pooled current-hop event F1 on CAL only; baseline allowed",
    cuts=[.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95],
    reset_ratios=[.25,.5], gap_hops=[7,14],
    modes=["direct_only","union_baseline","veto_baseline_005","veto_baseline_015"],
    no_outer_threshold_tuning=True, performance_optimization_deferred=True,
    historical_upstream_overlap=True, external_generalization_proven=False,
)


def collect_fold(task):
    root, out, comp = task
    root, out = Path(root), Path(out)
    with threadpool_limits(limits=1):
        manifest = json.loads((root/'manifest/report.json').read_text())
        native, s701 = load('model/live/strict_note/NOTE_strict_live.npz',
                            'model/onset256_research_39915038301/s701.joblib')
        records = []
        with zipfile.ZipFile(root/'GuitarSet_audio.zip') as z:
            for item in manifest['fit_files']:
                if item['composition'] != comp: continue
                annotation = root/'traces'/f"fit_{item['index']:03d}.json"
                if digest(annotation) != item['json_sha256']:
                    raise ValueError('signed annotation changed')
                meta = json.loads(annotation.read_text())
                sr, pcm = wavfile.read(io.BytesIO(z.read(meta['name']+'_mic.wav')))
                if sr != 44100 or pcm.dtype != np.int16 or pcm.ndim != 1:
                    raise ValueError('original WAV ABI mismatch')
                audio = pcm[:meta['samples']].astype(np.float32)/32768.
                n = len(audio)//256
                path = out/f"features_{item['index']:03d}.npy"
                if path.exists(): raise FileExistsError('immutable feature cache')
                x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                              shape=(n,49,DIM))
                stream = AcousticEvidenceStream(native, s701)
                t0 = time.monotonic()
                for frame in range(n):
                    start = frame*256
                    x[frame] = stream.push(audio[start:start+256], start_sample=start)
                x.flush(); del x
                # Annotation labels only enter training metadata after PCM ingestion.
                row = dict(index=item['index'],track=meta['name'],composition=comp,
                           frames=n,truth=meta['fit_truth'],
                           annotation_sha256=item['json_sha256'],
                           feature_sha256=digest(path))
                dump(out/f"track_{item['index']:03d}.json",row)
                records.append(row)
                print('ACOUSTIC_FEATURES',comp,len(records),n,
                      round(time.monotonic()-t0,2),flush=True)
        return records


def collect(args):
    verify_inputs(args.root)
    args.out.mkdir(parents=True,exist_ok=True)
    dest=args.out/'features'
    if dest.exists(): raise FileExistsError('immutable collection')
    dest.mkdir()
    dump(args.out/'preregistration.json',PREREGISTRATION)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups=list(pool.map(collect_fold,[(str(args.root),str(dest),c) for c in COMPS]))
    rows=[r for group in groups for r in group]
    if len(rows)!=80 or sum(r['frames'] for r in rows)!=306536:
        raise AssertionError('incomplete original audio')
    dump(args.out/'collection_report.json',dict(complete=True,schema=SCHEMA,
        tracks=80,frames=306536,feature_dim=DIM,hop_samples=256,
        future_audio_samples=0,added_hold_samples=0,per_track=rows))


def frame_targets(n_frames, truth, width):
    """Supervision only: onset in the past width hops, uncertain edges ignored.

    Labels do not manufacture candidates and never enter the inference API.
    Negative labels exclude the +/-80ms boundary neighborhood outside the
    positive interval, so an acceptable slightly early/late detection is not
    deliberately taught as a false attack.
    """
    y=np.zeros((n_frames,49),np.int8)
    ends=np.arange(1,n_frames+1)*256/44100.
    for note in truth:
        pitch=int(note['pitch'])-40
        age=ends-float(note['time'])
        edge=(age>=-4*256/44100.)&(age<=14*256/44100.)
        y[edge,pitch]=-1
    for note in truth:
        pitch=int(note['pitch'])-40
        age=ends-float(note['time'])
        y[(age>=0)&(age<=width*256/44100.),pitch]=1
    return y


def make_samples(args):
    rows=json.loads((args.out/'collection_report.json').read_text())['per_track']
    dest=args.out/'samples'
    if dest.exists():raise FileExistsError('immutable training samples')
    dest.mkdir()
    report=[]
    for comp in COMPS:
        for width in PREREGISTRATION['target_width_hops']:
            xx=[];yy=[]
            for r in rows:
                if r['composition']!=comp:continue
                x=np.load(args.out/'features'/f"features_{r['index']:03d}.npy",mmap_mode='r')
                y=frame_targets(r['frames'],r['truth'],width)
                rng=np.random.default_rng(SEED+r['index'])
                # Index 126+12 is native P(onset) in the signed 69D ABI.
                fraction=np.where(x[:,:,138]>.12,.20,.025)
                keep=(y==1)|((y==0)&(rng.random(y.shape)<fraction))
                xx.append(np.asarray(x[keep],np.float32));yy.append(y[keep])
            x=np.concatenate(xx);y=np.concatenate(yy)
            np.savez(dest/f'{comp}_w{width}.npz',X=x,y=y)
            report.append(dict(composition=comp,width=width,rows=len(y),positive=int(y.sum())))
            print('ACOUSTIC_SAMPLES',comp,width,len(y),int(y.sum()),flush=True)
    dump(dest/'report.json',report)


def baseline_mask(row, predictions):
    mask=np.zeros((row['frames'],49),bool)
    for sample,pitch in predictions[row['track']]:
        if sample%256 or not 256<=sample<=row['frames']*256:
            raise AssertionError('reference outside causal PCM256 grid')
        mask[sample//256-1,pitch-40]=True
    if int(mask.sum())!=len(predictions[row['track']]):
        raise AssertionError('reference duplicate at same sample/pitch')
    return mask


def mask_events(mask):
    frame,pitch=np.nonzero(mask)
    return [dict(sample=int((f+1)*256),pitch=int(p+40)) for f,p in zip(frame,pitch)]


def probabilities(model, row, directory):
    x=np.load(directory/'features'/f"features_{row['index']:03d}.npy",mmap_mode='r')
    flat=x.reshape(-1,DIM)
    q=np.empty(len(flat),np.float32)
    for start in range(0,len(flat),8192):
        q[start:start+8192]=model.predict_proba(flat[start:start+8192])[:,1]
    return q.reshape(row['frames'],49)


def policies():
    for cut in PREREGISTRATION['cuts']:
        for ratio in PREREGISTRATION['reset_ratios']:
            for gap in PREREGISTRATION['gap_hops']:
                for mode in PREREGISTRATION['modes']:
                    yield AttackPolicy(cut=cut,reset_ratio=ratio,gap_hops=gap,
                        use_baseline=mode!='direct_only',
                        baseline_cut=.05 if mode=='veto_baseline_005' else
                                     .15 if mode=='veto_baseline_015' else 0.)


def fit_fold(task):
    from sklearn.ensemble import HistGradientBoostingClassifier
    directory,outer=Path(task[0]),task[1]
    rows=json.loads((directory/'collection_report.json').read_text())['per_track']
    reference=json.loads(Path('series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json').read_text())
    cal=COMPS[(COMPS.index(outer)+1)%len(COMPS)]
    train=[c for c in COMPS if c not in (outer,cal)]
    cal_rows=[r for r in rows if r['composition']==cal]
    output=directory/'trained'/outer
    output.mkdir(parents=True)
    cache={r['track']:dict(baseline=baseline_mask(r,reference),truth=truth_arrays(r['truth']))
           for r in cal_rows}
    baseline=sum_scores([metric(*count_cache(v['baseline'],*v['truth'])) for v in cache.values()])
    best=dict(name='reference',score=baseline,policy=asdict(AttackPolicy(enabled=False)),width=None)
    history=[]
    with threadpool_limits(limits=1):
        for width in PREREGISTRATION['target_width_hops']:
            groups=[np.load(directory/'samples'/f'{c}_w{width}.npz') for c in train]
            x=np.concatenate([g['X'] for g in groups]);y=np.concatenate([g['y'] for g in groups])
            for g in groups:g.close()
            model=HistGradientBoostingClassifier(**PREREGISTRATION['model'],
                early_stopping=False,random_state=SEED+COMPS.index(outer))
            print('ACOUSTIC_TRAIN_BEGIN',outer,cal,width,len(y),int(y.sum()),flush=True)
            started=time.monotonic();model.fit(x,y);del x,y
            cp=output/f'acoustic_w{width}.joblib'
            joblib.dump(model,cp,compress=3)
            print('ACOUSTIC_TRAIN_DONE',outer,width,round(time.monotonic()-started,2),flush=True)
            q={r['track']:probabilities(model,r,directory) for r in cal_rows}
            candidates=[]
            for policy in policies():
                scores=[]
                for r in cal_rows:
                    v=cache[r['track']]
                    mask=decode_cache(q[r['track']],v['baseline'],policy)
                    scores.append(metric(*count_cache(mask,*v['truth'])))
                scores=sum_scores(scores)
                candidates.append(dict(policy=asdict(policy),score=scores))
            choice=max(candidates,key=lambda r:(r['score']['f1'],r['score']['tp'],-r['score']['fp']))
            # The optimized count routine must agree with the original scorer
            # for the selected calibration policy, on every CAL recording.
            policy=AttackPolicy(**choice['policy'])
            for r in cal_rows:
                v=cache[r['track']];mask=decode_cache(q[r['track']],v['baseline'],policy)
                expected=counts(mask_events(mask),r['truth'])
                observed=metric(*count_cache(mask,*v['truth']))
                if any(expected[k]!=observed[k] for k in ('tp','fp','fn')):
                    raise AssertionError('fast CAL scorer changed Hungarian TP')
            history.append(dict(width=width,best=choice,all_policies=candidates))
            print('ACOUSTIC_CAL',outer,width,json.dumps(choice),flush=True)
            if choice['score']['f1']>best['score']['f1']+1e-12:
                best=dict(name='direct_acoustic',width=width,**choice)
        meta=dict(schema=SCHEMA,trained_hop_samples=256,dim=DIM,
                  outer_composition=outer,cal_composition=cal,train_compositions=train,
                  selected=best,cal_reference=baseline,history=history,
                  frozen_before_outer=True,historical_upstream_overlap=True,
                  external_generalization_proven=False,added_hold_samples=0,
                  future_audio_samples=0)
        if best['name']!='reference':
            cp=output/f"acoustic_w{best['width']}.joblib"
            meta['checkpoint_sha256']=digest(cp)
        dump(output/'frozen_model_and_policy.json',meta)
        # Freeze the entire new model and decoder BEFORE any OUTER prediction.
        policy=AttackPolicy(**best['policy'])
        model=joblib.load(cp) if best['name']!='reference' else None
        selected={};q_hashes={}
        outer_rows=[r for r in rows if r['composition']==outer]
        for r in outer_rows:
            base=baseline_mask(r,reference)
            prob=probabilities(model,r,directory) if model else np.zeros(base.shape,np.float32)
            p=output/f"probability_{r['index']:03d}.npy";np.save(p,prob)
            q_hashes[p.name]=digest(p)
            selected[r['track']]=decode_cache(prob,base,policy)
        dump(output/'outer_prediction_freeze.json',dict(probabilities=q_hashes,
             model_and_policy_sha256=digest(output/'frozen_model_and_policy.json')))
        sealed={};records=[]
        for r in outer_rows:
            events=mask_events(selected[r['track']])
            sealed[r['track']]=sorted((e['sample'],e['pitch']) for e in events)
            records.append(dict(track=r['track'],composition=outer,
                                score=counts(events,r['truth'])))
        result=dict(outer=outer,cal=cal,train=train,selected=best,
                    outer_score=sum_scores([r['score'] for r in records]))
        dump(output/'result.json',result);dump(output/'predictions.json',sealed)
        print('ACOUSTIC_OUTER',outer,json.dumps(result['outer_score']),flush=True)
        return result,records,sealed


def train(args):
    if (args.out/'trained').exists():raise FileExistsError('immutable training experiment')
    (args.out/'trained').mkdir()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(fit_fold,[(str(args.out),c) for c in COMPS]))
    folds=[];rows=[];sealed={}
    for fold,records,predicted in results:
        folds.append(fold);rows.extend(records);sealed.update(predicted)
    report=dict(complete=True,folds=folds,per_track=rows,
                totals=sum_scores([r['score'] for r in rows]),
                models_trained=10,hop_samples=256,added_hold_samples=0,
                future_audio_samples=0,fresh_audio_replay_required=True,
                historical_upstream_overlap=True,external_generalization_proven=False,
                promoted=False)
    dump(args.out/'trained/report.json',report)
    dump(args.out/'trained/sealed_predictions.json',sealed)
    print('ACOUSTIC_TOTAL',json.dumps(report['totals']),flush=True)


def replay_fold(task):
    from onset256.acoustic_live import load_acoustic_stream
    root,directory,comp=Path(task[0]),Path(task[1]),task[2]
    manifest=json.loads((root/'manifest/report.json').read_text())
    expected=json.loads((directory/'trained'/comp/'predictions.json').read_text())
    reference=json.loads(Path('series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json').read_text())
    result=[]
    with threadpool_limits(limits=1),zipfile.ZipFile(root/'GuitarSet_audio.zip') as audio:
        for item in manifest['fit_files']:
            if item['composition']!=comp:continue
            annotation=root/'traces'/f"fit_{item['index']:03d}.json"
            if digest(annotation)!=item['json_sha256']:raise ValueError('annotation changed')
            meta=json.loads(annotation.read_text())
            sr,pcm=wavfile.read(io.BytesIO(audio.read(meta['name']+'_mic.wav')))
            if sr!=44100 or pcm.dtype!=np.int16 or pcm.ndim!=1:raise ValueError('audio ABI')
            wav=pcm[:meta['samples']].astype(np.float32)/32768.
            stream=load_acoustic_stream(directory/'trained'/comp,fold=comp,
                base_bundle='model/onset256_research_39915038301',
                recovery_bundle='model/onset256_recovery_20261023',
                native_path='model/live/strict_note/NOTE_strict_live.npz')
            events=[];old=[]
            cached=np.load(directory/'features'/f"features_{item['index']:03d}.npy",mmap_mode='r')
            for frame,start in enumerate(range(0,len(wav)-255,256)):
                decision=stream.push(wav[start:start+256],start_sample=start)
                if decision.sample!=start+256:raise AssertionError('delayed output')
                if not np.array_equal(cached[frame],stream.reference.risk.acoustic_evidence):
                    raise AssertionError('fresh acoustic features differ from training cache')
                events.extend(dict(sample=e.sample,pitch=e.pitch) for e in decision.intents)
                old.extend((e.sample,e.pitch) for e in stream.last_reference.intents)
            if stream.finish():raise AssertionError('pending events at EOF')
            keys=sorted((e['sample'],e['pitch']) for e in events)
            if keys!=[tuple(e) for e in expected[meta['name']]]:
                raise AssertionError('fresh acoustic output differs: '+meta['name'])
            if sorted(old)!=[tuple(e) for e in reference[meta['name']]]:
                raise AssertionError('68.370987 reference changed: '+meta['name'])
            if stream.calls!=len(wav)//256:raise AssertionError('missing PCM256 decision')
            result.append(dict(track=meta['name'],composition=comp,frames=stream.calls,
                score=counts(events,meta['fit_truth']),exact_event_parity=True,
                exact_feature_parity=True,exact_reference_parity=True))
            print('ACOUSTIC_FRESH_TRACK',comp,len(result),flush=True)
    dump(directory/'replayed'/f'{comp}.json',result)
    return result


def replay(args):
    verify_inputs(args.root)
    destination=args.out/'replayed'
    if destination.exists():raise FileExistsError('immutable fresh replay')
    destination.mkdir()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups=list(pool.map(replay_fold,[(str(args.root),str(args.out),c) for c in COMPS]))
    finish_replay(args,[r for g in groups for r in g])


def finish_replay(args,rows):
    destination=args.out/'replayed'
    score=sum_scores([r['score'] for r in rows])
    previous=json.loads((args.out/'trained/report.json').read_text())['totals']
    if len(rows)!=80 or sum(r['frames'] for r in rows)!=306536:
        raise AssertionError('incomplete fresh audio')
    if any(score[k]!=previous[k] for k in ('tp','fp','fn')):
        raise AssertionError('fresh score mismatch')
    dump(destination/'report.json',dict(complete=True,totals=score,per_track=rows,
        tracks=80,frames=306536,hop_samples=256,added_hold_samples=0,
        future_audio_samples=0,exact_event_parity=True,exact_feature_parity=True,
        exact_reference_parity=True,historical_upstream_overlap=True,
        external_generalization_proven=False,promoted=False))
    print('ACOUSTIC_FRESH_TOTAL',json.dumps(score),flush=True)


def audit(args):
    fresh=json.loads((args.out/'replayed/report.json').read_text())
    rows=json.loads((args.out/'collection_report.json').read_text())['per_track']
    old=json.loads(Path('series/ONSET_SCORE_RECOVERY_HGB256/evidence/sealed_predictions.json').read_text())
    new=json.loads((args.out/'trained/sealed_predictions.json').read_text())
    records=[]
    for row in rows:
        name=row['track'];truth=row['truth']
        a=[SimpleNamespace(sample=s,pitch=p) for s,p in old[name]]
        b=[SimpleNamespace(sample=s,pitch=p) for s,p in new[name]]
        ma=match_onsets(a,truth);mb=match_onsets(b,truth)
        ta={j for _,j in ma};tb={j for _,j in mb}
        sa=metric(len(ma),len(a)-len(ma),len(truth)-len(ma))
        sb=metric(len(mb),len(b)-len(mb),len(truth)-len(mb))
        aset={tuple(e) for e in old[name]};bset={tuple(e) for e in new[name]}
        gained=sorted(tb-ta);lost=sorted(ta-tb)
        records.append(dict(track=name,composition=row['composition'],
            texture='solo' if name.endswith('_solo') else 'comp',before=sa,after=sb,
            delta_f1_points=100*(sb['f1']-sa['f1']),
            new_truth_matches=len(gained),lost_truth_matches=len(lost),
            recovered_truth=[dict(index=i,**truth[i]) for i in gained],
            lost_truth=[dict(index=i,**truth[i]) for i in lost],
            added_events=sorted(bset-aset),removed_events=sorted(aset-bset)))
    groups={}
    for key in ('composition','texture'):
        groups[key]={}
        for value in sorted({r[key] for r in records}):
            selected=[r for r in records if r[key]==value]
            before=sum_scores([r['before'] for r in selected])
            after=sum_scores([r['after'] for r in selected])
            groups[key][value]=dict(before=before,after=after,
                                   delta_f1_points=100*(after['f1']-before['f1']))
    before=sum_scores([r['before'] for r in records])
    after=sum_scores([r['after'] for r in records])
    if (before['tp'],before['fp'],before['fn'])!=(8050,4231,3217):
        raise AssertionError('audit reference mismatch')
    if any(after[k]!=fresh['totals'][k] for k in ('tp','fp','fn')):
        raise AssertionError('audit replay mismatch')
    report=dict(complete=True,before=before,after=after,
        delta_f1_points=100*(after['f1']-before['f1']),
        delta={k:after[k]-before[k] for k in ('tp','fp','fn')},
        recovered_truth=sum(r['new_truth_matches'] for r in records),
        lost_truth=sum(r['lost_truth_matches'] for r in records),
        tracks_improved=sum(r['delta_f1_points']>1e-10 for r in records),
        tracks_regressed=sum(r['delta_f1_points']< -1e-10 for r in records),
        tracks_unchanged=sum(abs(r['delta_f1_points'])<=1e-10 for r in records),
        groups=groups,per_track=records)
    dump(args.out/'comparison.json',report)
    print('ACOUSTIC_PAIRED_AUDIT',json.dumps({k:v for k,v in report.items()
        if k not in ('groups','per_track')}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--phase',choices=['collect','samples','train','replay','audit'],default='collect')
    args=p.parse_args()
    {'collect':collect,'samples':make_samples,'train':train,'replay':replay,'audit':audit}[args.phase](args)


if __name__=='__main__': main()
