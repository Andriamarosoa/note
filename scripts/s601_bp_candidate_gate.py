"""S601: composition-blocked candidate learning on frozen BP posteriors.

Offline note-on research, not a causal/Live model. Input features contain
only audio-derived probabilities and candidate timing. File/instrument names,
annotations, piece identities and folds never enter the feature matrix.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import itertools
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scripts.s600_bp_native_development import train_specs, decode, MANIFEST_SHA
from scripts.s592_s597_urmp_prospective import read
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics
from scripts.audit_s597_paired_transfer import maximum_count


MODEL_CONFIG=dict(n_estimators=160,max_depth=18,min_samples_leaf=4,
                  max_features=.8,random_state=60101,n_jobs=2)
DECODE_CONFIGS=(dict(onset=.5,frame=.3,length=11),
                dict(onset=.6,frame=.3,length=5),
                dict(onset=.35,frame=.15,length=2))


def candidates(raw):
    from basic_pitch.note_creation import model_frames_to_time, get_infered_onsets
    times=model_frames_to_time(len(raw['note'])); bank={}
    def add(frame,pitch,end,origin):
        if not (0<=times[frame]<24 and 40<=pitch<=88):return
        k=(int(frame),int(pitch))
        if k not in bank:bank[k]=dict(frame=int(frame),pitch=int(pitch),time=float(times[frame]),
            ends=[0.]*4,origin=[0.]*4)
        bank[k]['ends'][origin]=max(bank[k]['ends'][origin],float(end-times[frame]))
        bank[k]['origin'][origin]=1.
    for origin,config in enumerate(DECODE_CONFIGS):
        for event in decode(raw,**config):
            frame=int(np.argmin(np.abs(times-event['time'])))
            add(frame,event['notes'][0],event['end'],origin)
    inferred=get_infered_onsets(raw['onset'],raw['note'])
    for pitch in range(40,89):
        p=pitch-21
        for frame in find_peaks(inferred[:,p],height=.20,distance=3)[0]:
            # Candidate recall gate only; no truth, no hard instrument rule.
            if raw['note'][frame:min(len(times),frame+9),p].max()<.15:continue
            end=frame+1
            while end<len(times)-1 and raw['note'][end,p]>=.15:end+=1
            add(int(frame),pitch,min(24.,float(times[end])),3)
    result=[bank[k] for k in sorted(bank)]
    need(result and len({(r['frame'],r['pitch']) for r in result})==len(result),'Unique nonempty candidate bank')
    return result


def feature_matrix(raw, events):
    from basic_pitch.note_creation import get_infered_onsets
    frame=raw['note']; onset=raw['onset']; contour=raw['contour']
    inferred=get_infered_onsets(onset,frame);n=len(frame)
    names=[];X=[]
    f_offsets=(-16,-12,-8,-4,-2,-1,0,1,2,3,4,6,8,12,16,24)
    o_offsets=(-8,-4,-2,-1,0,1,2,4,8)
    neighbors=(-24,-19,-12,-7,-2,-1,1,2,7,12,19,24)
    for r in events:
        v=[];columns=[];t=r['frame'];p=r['pitch']-21
        def put(name,value):columns.append(name);v.append(float(value))
        put('MIDI_pitch_scaled',r['pitch']/127.)
        for j in range(4):
            put(f'decoder_origin_{j}',r['origin'][j]);put(f'decoder_duration_{j}',min(2.,r['ends'][j]))
        for offset in f_offsets:put(f'frame_{offset}',frame[np.clip(t+offset,0,n-1),p])
        for label,a in (('onset',onset),('inferred',inferred)):
            for offset in o_offsets:put(f'{label}_{offset}',a[np.clip(t+offset,0,n-1),p])
        for label,a in (('frame',frame),('onset',onset),('inferred',inferred)):
            for start,end in ((-16,-8),(-8,-2),(-2,3),(1,6),(6,13),(13,25)):
                chunk=a[max(0,t+start):min(n,t+end),p]
                put(f'{label}_mean_{start}_{end}',chunk.mean() if len(chunk) else 0.)
                put(f'{label}_max_{start}_{end}',chunk.max() if len(chunk) else 0.)
        for delta in neighbors:
            q=p+delta
            for label,a in (('frame',frame),('onset',onset)):
                for lo,hi in ((-4,0),(0,5),(5,13)):
                    z=a[max(0,t+lo):min(n,t+hi),q] if 0<=q<88 else []
                    put(f'{label}_neighbor_{delta}_{lo}_{hi}',np.max(z) if len(z) else 0.)
        for lo,hi in ((-4,0),(0,5),(5,13)):
            z=frame[max(0,t+lo):min(n,t+hi)]
            put(f'polyphony_sum_{lo}_{hi}',z.sum(axis=1).mean() if len(z) else 0.)
            put(f'polyphony_gt03_{lo}_{hi}',(z>.3).sum(axis=1).mean() if len(z) else 0.)
            for delta in (-1,0,1):
                q=3*(p+delta)
                zc=contour[max(0,t+lo):min(n,t+hi),max(0,q):min(contour.shape[1],q+3)]
                put(f'contour_{delta}_{lo}_{hi}',zc.max() if zc.size else 0.)
        same=sorted(x['time'] for x in events if x['pitch']==r['pitch'])
        before=[u for u in same if u<r['time']];after=[u for u in same if u>r['time']]
        put('previous_candidate_distance',min(2.,r['time']-before[-1]) if before else 2.)
        put('next_candidate_distance',min(2.,after[0]-r['time']) if after else 2.)
        if not names:names=columns
        need(names==columns,'Stable candidate feature schema')
        X.append(v)
    matrix=np.asarray(X,np.float32)
    need(np.isfinite(matrix).all(),'Finite audio features')
    return matrix,names


def build(root,out):
    out.mkdir(parents=True,exist_ok=True);names=None;proof={}
    for s in train_specs():
        sid=s['source_id'];folder=out/sid;folder.mkdir(exist_ok=True)
        if (folder/'report.json').exists():
            r=read(folder/'report.json')
            need(r['raw_sha256']==sha(root/sid/'raw.npz') and r['features_sha256']==sha(folder/'features.npz'), 'Immutable candidate cache')
        else:
            raw=dict(np.load(root/sid/'raw.npz',allow_pickle=False))
            events=candidates(raw);X,names=feature_matrix(raw,events)
            save(folder/'candidates.json',events);np.savez_compressed(folder/'features.npz',X=X)
            r=dict(source_id=sid,raw_sha256=sha(root/sid/'raw.npz'),
                candidates_sha256=sha(folder/'candidates.json'),features_sha256=sha(folder/'features.npz'),
                feature_names=names,candidate_count=len(events),labels_used=False)
            save(folder/'report.json',r)
        names=r['feature_names'];proof[sid]=sha(folder/'report.json')
        print('S601 CANDIDATES',s['folder'],r['candidate_count'],flush=True)
    save(out/'dataset.json',dict(manifest_sha256=MANIFEST_SHA,feature_names=names,
        sources=proof,source_count=37,composition_count=22,labels_in_features=False))


def emit(events,probability,threshold,gap,shift=0.):
    chosen=[];by_pitch=collections.defaultdict(list)
    for i in sorted(np.flatnonzero(probability>=threshold),key=lambda j:(-probability[j],events[j]['time'],events[j]['pitch'])):
        r=events[i];p=r['pitch'];t=r['time']+shift
        if 0<=t<24 and all(abs(t-v)>=gap for v in by_pitch[p]):
            chosen.append(dict(time=float(t),notes=[p],probability=float(probability[i])))
            by_pitch[p].append(t)
    return sorted(chosen,key=lambda x:(x['time'],x['notes']))


def counts(guess,truth,tol=.08):
    tp=sum(maximum_count([r['time'] for r in guess if p in r['notes']],
                        [r['time'] for r in truth if p in r['notes']],tol) for p in range(40,89))
    npred=sum(len(r['notes']) for r in guess);ngt=sum(len(r['notes']) for r in truth)
    return dict(TP=tp,FP=npred-tp,FN=ngt-tp,F1=2*tp/max(1,npred+ngt))


def load_data(root,out):
    truth=read(root/'truth_train37.json')['truth'];specs=train_specs()
    X=[];Y=[];mask=[];source=[];events={};coverage={};offsets={}
    for s in specs:
        sid=s['source_id'];ev=read(out/sid/'candidates.json');events[sid]=ev
        x=np.load(out/sid/'features.npz',allow_pickle=False)['X']
        dist=np.array([min((abs(r['time']-g['time']) for g in truth[sid] if r['pitch'] in g['notes']),default=100.) for r in ev])
        # Several candidate clocks may describe one attack. Their nearby
        # alternatives stay positive; NMS is evaluated after prediction.
        Y.extend(dist<=.05);mask.extend((dist<=.05)|(dist>.10));source.extend([sid]*len(ev));X.append(x)
        offsets[sid]=len(source)-len(ev)
        coverage[sid]=counts([dict(time=r['time'],notes=[r['pitch']]) for r in ev],truth[sid])
    return np.concatenate(X),np.asarray(Y,np.int8),np.asarray(mask),np.array(source),events,truth,specs,coverage


def cv(root,out,report):
    from sklearn.ensemble import ExtraTreesClassifier
    need(not report.exists(),'Immutable grouped CV experiment')
    X,Y,valid,source,events,truth,specs,coverage=load_data(root,out)
    pieces=sorted({s['piece'] for s in specs},key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    folds=[pieces[i::5] for i in range(5)]
    prob=np.full(len(Y),np.nan,np.float32);foldlog=[]
    for fold,held in enumerate(folds):
        held_sources=[s['source_id'] for s in specs if s['piece'] in held]
        test=np.isin(source,held_sources);train=~test & valid
        need(not ({s['piece'] for s in specs if s['source_id'] in set(source[train])}&set(held)),'Composition leakage in CV')
        model=ExtraTreesClassifier(**MODEL_CONFIG);model.fit(X[train],Y[train])
        prob[test]=model.predict_proba(X[test])[:,1]
        foldlog.append(dict(fold=fold,held_pieces=held,held_sources=held_sources,
            train_candidates=int(train.sum()),held_candidates=int(test.sum())))
        print('S601 FOLD',fold,held,int(train.sum()),int(test.sum()),flush=True)
    need(np.isfinite(prob).all(),'Every candidate must have a held-composition prediction')
    np.savez_compressed(report.with_suffix('.probabilities.npz'),probability=prob,source=source,target=Y,valid=valid)
    scores={};best=None;best_config=None;best_value=-1.
    for threshold,gap,shift in itertools.product((.30,.35,.40,.45,.50,.55,.60,.65,.70,.75),(.05,.08,.10),(-.02,0.,.02)):
        key=f'p{threshold:.2f}_gap{int(gap*1000)}_shift{int(shift*1000)}'
        config=dict(threshold=threshold,gap=gap,shift=shift)
        pred={sid:emit(ev,prob[source==sid],**config) for sid,ev in events.items()}
        per={sid:counts(pred[sid],truth[sid]) for sid in truth}
        c={k:sum(v[k] for v in per.values()) for k in ('TP','FP','FN')}
        c['F1']=2*c['TP']/max(1,2*c['TP']+c['FP']+c['FN'])
        scores[key]=dict(config=config,fullstream80=c,per_source80=per)
        if c['F1']>best_value:best=key;best_config=config;best_value=c['F1']
    pred={sid:emit(ev,prob[source==sid],**best_config) for sid,ev in events.items()}
    meta={s['source_id']:dict(composer=s['folder']) for s in specs}
    full=metrics({'S601_OOF':pred},truth,meta)['S601_OOF']
    need(full['fullstream']['80']['TP']==scores[best]['fullstream80']['TP'],'Independent/Hungarian match parity')
    save(report,dict(status='S601_COMPOSITION_BLOCKED_DEVELOPMENT_NOT_FINAL_TEST',
        model_config=MODEL_CONFIG,feature_dimensions=X.shape[1],candidate_count=len(Y),
        positive_candidates=int(Y.sum()),ambiguous_excluded=int((~valid).sum()),
        manifest_sha256=MANIFEST_SHA,dataset_sha256=sha(out/'dataset.json'),truth_sha256=sha(root/'truth_train37.json'),
        folds=foldlog,all_decoder_trials=scores,best=best,best_config=best_config,best_metrics=full,
        candidate_coverage=coverage,OOF_probability_sha256=sha(report.with_suffix('.probabilities.npz')),
        selection_uses_OOF_labels_so_not_an_independent_test=True))
    save(report.with_suffix('.predictions.json'),pred)
    print('S601 BEST OOF DEVELOPMENT',best,full['fullstream']['80'],flush=True)


def fit(root,out,report,destination):
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    need(not destination.exists(),'Fit final checkpoint once, only after selection')
    r=read(report);need(r['dataset_sha256']==sha(out/'dataset.json'),'Candidate dataset changed after CV')
    X,Y,valid,source,events,truth,specs,coverage=load_data(root,out)
    model=ExtraTreesClassifier(**r['model_config']);model.fit(X[valid],Y[valid])
    joblib.dump(model,destination,compress=3)
    save(destination.with_suffix('.json'),dict(model_sha256=sha(destination),selection_report_sha256=sha(report),
        config=r['best_config'],model_config=r['model_config'],manifest_sha256=MANIFEST_SHA,
        train_sources=sorted(events),train_pieces=sorted({s['piece'] for s in specs}),
        test_audio_read=False,test_annotations_read=False))
    print('S601 FINAL FIT',destination,sha(destination),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('build','cv','fit'))
    p.add_argument('--root',type=Path,default=Path('model/S600'))
    p.add_argument('--out',type=Path,default=Path('model/S601'))
    p.add_argument('--report',type=Path,default=Path('model/S601/grouped_cv.json'))
    p.add_argument('--destination',type=Path,default=Path('model/S601/candidate_gate.joblib'))
    a=p.parse_args()
    if a.stage=='build':build(a.root,a.out)
    elif a.stage=='cv':cv(a.root,a.out,a.report)
    else:fit(a.root,a.out,a.report,a.destination)
