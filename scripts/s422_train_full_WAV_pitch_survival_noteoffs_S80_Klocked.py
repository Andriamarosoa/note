"""S422 — continuous 10-ms pitch-survival NoteOff model, S80 K immutable.

Train a real discriminative note-active(time) model on complete GuitarSet MIC
WAV trajectories, with predictions for each already-created S420 MIDI note.
This is NOT another static median-duration corrector: the classifier examines
pitch presence and harmonic changes across 25ms..5s, including all K0 times.
At infer time it predicts the NoteOff where survival P(active) crosses a
TRAIN-ONLY selected threshold, with an optional S420-safe abstention policy.

All inner thresholds/policies are selected on compositions distinct from the
fold they evaluate. No YourMT teacher, no true K, no GT pitch or GT duration
goes into held-song inference. Original S80 59309 per-row K output stays fixed.

S80 train overlap & S419 nonnested gate mean this is development evidence,
NOT independent end-to-end generalization.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s411_extract_continuous_GuitarSet_WAV_pitch_survival_features import (
    decode_pcm,continuous_pitch_strength,SR)
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import (
    get_gt,match_predicted_to_reference)
from scripts.s414_compare_official_yourmt3_full_notes_vs_S408_S412_guitarset import score
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold)

N_EVENTS=59309
N_NOTES=33475
GT_NOTES=34954
HORIZONS=np.asarray(list(range(25,1026,25))+list(range(1100,2101,100))
       +[2300,2500,2800,3200,3600,4000,4500,5000],dtype=np.float32)
CANDIDATES=((.35,'replace'),(.50,'replace'),(.65,'replace'),
            (.35,'extend_only'),(.50,'extend_only'),(.65,'extend_only'),
            (.50,'extend_confident'))
SEED=422
DIM=15

def must(x,msg):
    if not x:raise ValueError(msg)
def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def build_continuous_survival(audio_zip,p,verbose=True):
    """Full source audio is processed before any JAMS is opened."""
    tracks=p['track_member'].astype(str)
    midi=p['MIDI'].astype(np.int16)
    onset=p['onset_samples'].astype(np.int64)
    old=p['native_offset_samples'].astype(np.int64)
    must(len(midi)==N_NOTES and len(set(tracks))==190,
         'S422 needs original complete S420 frozen MIDI and track cohort')
    out=np.zeros((len(midi),len(HORIZONS),DIM),np.float32)
    hop_samples=HORIZONS[None,:]*SR/1000.
    with zipfile.ZipFile(audio_zip) as arc:
        names=collections.defaultdict(list)
        for name in arc.namelist():
            if PurePosixPath(name).name.endswith('_mic.wav'):
                names[PurePosixPath(name).name[:-8]+'.jams'].append(name)
        for z,track in enumerate(sorted(set(tracks))):
            mic=names[PurePosixPath(track).name]
            must(len(mic)==1,'S422 original complete WAV ambiguous '+track)
            audio=decode_pcm(arc.read(mic[0]))
            frame,s,rise,fall,energy=continuous_pitch_strength(audio)
            inds=np.flatnonzero(tracks==track)
            on=onset[inds]
            pit=midi[inds]-40
            must(np.all((pit>=0)&(pit<49)),'S422 MIDI range unexpected')
            target=on[:,None]+hop_samples
            positions=np.minimum(np.searchsorted(frame,target,side='left'),
                                 len(frame)-1)
            before=np.maximum(positions-5,0)
            after=np.minimum(positions+5,len(frame)-1)
            at=np.minimum(np.searchsorted(frame,on),len(frame)-1)
            pidx=pit[:,None]
            onset_strength=s[at,pit][:,None]
            now=s[positions,pidx]
            prev=s[before,pidx]
            future=s[after,pidx]
            competition=np.maximum(s[positions,np.clip(pidx+12,0,48)],
                        s[positions,np.clip(pidx-12,0,48)])
            relative=np.exp(np.clip(now-onset_strength,-12,8))
            gt_duration_was_not_read=True
            valid=(target<=frame[-1])
            source_old_ms=(old[inds]-on)/SR*1000
            # New note attacks and old K are the ONLY discrete event logic;
            # K0 intervals remain in spectral trajectories.
            sorted_note_on=np.sort(on)
            nxt_i=np.searchsorted(sorted_note_on,on,side='right')
            next_any=np.where(nxt_i<len(sorted_note_on),
                sorted_note_on[np.minimum(nxt_i,len(sorted_note_on)-1)]-on,
                int(6*SR))/SR*1000
            next_same=np.full(len(inds),6000.,np.float32)
            for pitch in np.unique(pit):
                loc=np.flatnonzero(pit==pitch)
                arr=np.sort(on[loc])
                ix=np.searchsorted(arr,on[loc],side='right')
                okay=ix<len(arr)
                next_same[loc[okay]]=(arr[ix[okay]]-on[loc[okay]])/SR*1000
            X=np.stack([
                np.broadcast_to(HORIZONS[None,:]/1000.,now.shape),
                now,prev,future,
                rise[positions,pidx],fall[positions,pidx],
                relative,
                now-competition,
                energy[positions],
                energy[positions]-energy[at][:,None],
                np.broadcast_to(midi[inds,None]/88.,now.shape),
                np.broadcast_to(source_old_ms[:,None]/1000.,now.shape),
                np.broadcast_to(next_any[:,None]/1000.,now.shape),
                np.broadcast_to(next_same[:,None]/1000.,now.shape),
                valid.astype(np.float32)
            ],axis=-1)
            must(X.shape==(len(inds),len(HORIZONS),DIM)
              and np.isfinite(X).all() and gt_duration_was_not_read,
              'S422 continuous pitch-survival inputs invalid')
            out[inds]=X
            if verbose and (z+1)%25==0:
                print('S422 RAW_AUDIO',z+1,'of',len(set(tracks)),flush=True)
    must(np.isfinite(out).all() and out.shape==(N_NOTES,len(HORIZONS),DIM),
         'S422 real continuous source incomplete')
    return out

def read_training_targets(labels,p):
    """Only after predictions are frozen. Return GT Off absolute; unmatched=-1."""
    tracks=p['track_member'].astype(str)
    onset=p['onset_samples'].astype(np.int64)
    midi=p['MIDI'].astype(int)
    original={}
    with zipfile.ZipFile(labels) as z:
        matched=collections.defaultdict(list)
        for member in z.namelist():
            if member.endswith('.jams'):
                matched[PurePosixPath(member).name].append(member)
        for t in np.unique(tracks):
            files=matched[PurePosixPath(t).name]
            must(len(files)==1,'S422 original GuitarSet JAMS missing/duplicate')
            original[t]=get_gt(z.read(files[0]))
    must(len(original)==190 and sum(map(len,original.values()))==GT_NOTES,
         'S422 GT JAMS original cardinality wrong')
    gt_end=np.full(N_NOTES,-1,np.int64)
    for track,truth in original.items():
        ii=np.flatnonzero(tracks==track)
        candidates=[(int(onset[i]),int(midi[i]),int(i)) for i in ii]
        for pi,gi in match_predicted_to_reference(candidates,truth,2205):
            j=candidates[pi][2]
            gt_end[j]=int(truth[gi][1])
    must(int((gt_end>0).sum())==25395,
         'S422 frozen S420 correctly matched NoteOn/MIDI count changed')
    return gt_end,original

def survival_estimate(model,features,threshold):
    q=model.predict_proba(features.reshape(-1,DIM))[:,1]
    q=q.reshape((len(features),len(HORIZONS)))
    # Enforce valid survival P(T>t) that cannot increase with note age.
    p_alive=np.minimum.accumulate(q,axis=1)
    after=p_alive<threshold
    idx=np.argmax(after,axis=1)
    missing=~after.any(axis=1)
    idx[missing]=len(HORIZONS)-1
    prev=np.maximum(idx-1,0)
    low=HORIZONS[prev]
    high=HORIZONS[idx]
    p0=p_alive[np.arange(len(idx)),prev]
    p1=p_alive[np.arange(len(idx)),idx]
    frac=np.clip((p0-threshold)/(p0-p1+1e-6),0.,1.)
    crossing=np.where(idx==0,HORIZONS[0],low+(high-low)*frac)
    crossing=np.where(missing,HORIZONS[-1],crossing)
    return crossing.astype(np.float32),p_alive

def corrected_offsets(old,on,estimated,alive,threshold,policy):
    duration=(old-on)/SR*1000
    proposed=np.maximum(30,estimated)
    if policy=='replace':
        picked=proposed
    elif policy=='extend_only':
        picked=np.maximum(duration,proposed)
    elif policy=='extend_confident':
        # Acoustically persuasive long hold only: note appears still alive
        # at 500ms, S420 already short. Entire rule fixed pre-evaluation.
        at500=int(np.flatnonzero(HORIZONS==500)[0])
        can=(alive[:,at500]>threshold)&(estimated>=500)&(duration<500)
        picked=np.where(can,np.maximum(duration,proposed),duration)
    else:raise ValueError('Unknown survival policy')
    return on+np.rint(np.clip(picked,30,5000)*SR/1000).astype(np.int64)

def fit(X,gt_end,on,mask,fold):
    indices=np.flatnonzero(mask)
    must(len(indices)>8000,'S422 source label train too small')
    durations=(gt_end[indices]-on[indices])/SR*1000
    y=(durations[:,None]>HORIZONS[None,:]).astype(np.uint8)
    feats=X[indices].reshape(-1,DIM)
    labels=y.ravel()
    must(int(labels.sum())>10000 and int((labels==0).sum())>200000,
        'S422 survival training lacks positives/negatives')
    weights=np.ones_like(y,dtype=np.float32)
    # Physical events lasting >1s are rare. Train-only class weighting
    # counteracts age imbalance; no held label/threshold used to decide this.
    weights[:,HORIZONS>=750]*=1.4
    weights[(y==1)&(np.broadcast_to(HORIZONS[None,:],y.shape)>=750)]*=2.
    clf=HistGradientBoostingClassifier(
        max_iter=75,max_leaf_nodes=18,min_samples_leaf=110,
        learning_rate=.08,l2_regularization=5.,
        early_stopping=False,random_state=SEED+fold)
    clf.fit(feats,labels,sample_weight=weights.ravel())
    return clf

def get_score(tracks,midi,on,offs,gt):
    predicted=collections.defaultdict(list)
    for i,m in enumerate(tracks):
        predicted[str(m)].append((int(on[i]),int(offs[i]),int(midi[i]),True))
    result,_=score(predicted,gt)
    return result

def per_duration(p,offset_new,offset_old,gt):
    by=collections.defaultdict(lambda:collections.Counter())
    member=p['track_member'].astype(str)
    for t,ref in gt.items():
        ii=np.flatnonzero(member==t)
        candidates=[(int(p['onset_samples'][k]),int(p['MIDI'][k]),int(k)) for k in ii]
        for pi,gi in match_predicted_to_reference(candidates,ref,2205):
            j=candidates[pi][2];start,finish,_,_=ref[gi]
            length=(finish-start)/SR*1000
            name='<100ms' if length<100 else (
                 '100-499ms' if length<500 else
                 '500-999ms' if length<1000 else '>=1000ms')
            a=abs(int(offset_old[j])-finish)<=2205
            b=abs(int(offset_new[j])-finish)<=2205
            c=by[name];c['n']+=1;c['S420']+=int(a);c['S422']+=int(b)
            c['corrections']+=int(b and not a)
            c['regressions']+=int(a and not b)
    return {k:dict(v,net=int(v['S422']-v['S420'])) for k,v in by.items()}

def run(original_s420,s80,audio,annotations,out):
    must(not out.exists(),'S422 output immutable')
    p=read_npz(original_s420)
    r=read_npz(s80)
    require_keys={'source_row_id','MIDI','track_member','onset_samples',
                  'native_offset_samples','original_S80_K','original_S80_row_id'}
    must(require_keys.issubset(p),'S422 S420 exact source predictions missing')
    ids=p['original_S80_row_id'].astype(str)
    midi=p['MIDI'].astype(int)
    tracks=p['track_member'].astype(str)
    on=p['onset_samples'].astype(np.int64)
    baseline=p['native_offset_samples'].astype(np.int64)
    counts=p['original_S80_K'].astype(np.int8)
    must(len(ids)==N_EVENTS and len(set(ids))==N_EVENTS
      and np.array_equal(ids,r['row_id'].astype(str))
      and np.array_equal(counts,r['s80_K'].astype(np.int8))
      and len(midi)==len(tracks)==len(on)==len(baseline)==N_NOTES
      and int(counts.sum())==N_NOTES
      and np.all(baseline>on)
      and np.isin(counts,range(7)).all(),
      'S80 output K was changed or original MIDI cohort drifted')
    original_counts=collections.Counter(p['source_row_id'].astype(str))
    must(all(original_counts[s]==int(k) for s,k in zip(ids,counts)),
      'S420 per row count not the genuine S80 baseline')
    X=build_continuous_survival(audio,p)
    # The first call that opens JAMS is BELOW this source-only extraction.
    gt_end,GT=read_training_targets(annotations,p)
    comps=np.array([composition_key(t) for t in tracks])
    folds=np.array([composition_fold(c) for c in comps],np.int8)
    chosen=np.zeros(N_NOTES,np.int64)
    policy_reports=[]
    out.mkdir(parents=True)
    for fold in range(4):
        held=folds==fold
        train=(folds!=fold)&(gt_end>0)
        # Composition-disjoint INTERNAL model and threshold selection.
        inner_val=train & np.array([
          int(hashlib.sha256(('S422-inner-'+c).encode()).hexdigest()[:8],16)%5==0
          for c in comps],bool)
        inner_train=train & ~inner_val
        must(inner_train.sum()>5000 and inner_val.sum()>500
          and not (set(comps[inner_train])&set(comps[inner_val]))
          and not (set(comps[train])&set(comps[held])),
          'S422 inner/outer song composition overlap')
        cal=fit(X,gt_end,on,inner_train,fold+20)
        observed=gt_end[inner_val]
        inner_base=int(np.sum(np.abs(baseline[inner_val]-observed)<=2205))
        curves={}
        for threshold,policy in CANDIDATES:
            if threshold not in curves:
                curves[threshold]=survival_estimate(cal,X[inner_val],threshold)
        tested=[]
        for threshold,policy in CANDIDATES:
            estimate,alive=curves[threshold]
            potential=corrected_offsets(baseline[inner_val],on[inner_val],
                      estimate,alive,threshold,policy)
            hits=int(np.sum(np.abs(potential-observed)<=2205))
            tested.append(dict(threshold=threshold,policy=policy,
              hits=hits,delta_vs_S420=hits-inner_base))
        # Only an OUTER-TRAIN policy can be activated. Default no change.
        best=max(tested,key=lambda a:(a['hits'],-a['threshold'],
            a['policy']=='extend_confident'))
        active=best['delta_vs_S420']>=7
        if active:
            cls=fit(X,gt_end,on,train,fold+40)
            estimate,alive=survival_estimate(cls,X[held],best['threshold'])
            chosen[held]=corrected_offsets(baseline[held],on[held],
                  estimate,alive,best['threshold'],best['policy'])
            dest=out/f'S422_full_WAV_survival_songheld_fold{fold}.joblib'
            joblib.dump(dict(network=cls,threshold=best['threshold'],
              policy=best['policy'],K_is_frozen=True,
              none_of_YOURMT_teacher_used=True,
              held_composition_fold=fold),dest,compress=3)
            model_hash=sha(dest)
        else:
            chosen[held]=baseline[held]
            model_hash=None
        policy_reports.append(dict(fold=fold,train=int(train.sum()),
           held=int(held.sum()),inner_train=int(inner_train.sum()),
           inner_validation=int(inner_val.sum()),inner_original_correct=inner_base,
           candidate_inner_results=tested,selected_threshold=best['threshold'],
           selected_policy=best['policy'],active=bool(active),
           train_compositions=len(set(comps[train])),
           held_compositions=len(set(comps[held])),
           composition_overlap=0,model_SHA256=model_hash))
        print('S422 FOLD',json.dumps(policy_reports[-1]),flush=True)
    must(np.all(chosen>on) and len(chosen)==N_NOTES,
         'S422 invalid NoteOff outputs')
    # Freeze MIDI output and independent exact-K replay BEFORE measuring GT.
    artifact=out/'S422_NATIVE_K_PROTECTED_full_MIDI_events.jsonl'
    bytrack=collections.defaultdict(list)
    for i,t in enumerate(tracks):
        bytrack[t].append(dict(source_row_id=str(p['source_row_id'][i]),
             pitch=int(midi[i]),onset_sample=int(on[i]),
             offset_sample=int(chosen[i]),source='S422_survival_or_S420_abstain'))
    with artifact.open('w') as f:
        for t,events in sorted(bytrack.items()):
            f.write(json.dumps(dict(member=t,notes=events))+'\n')
    replay=collections.Counter()
    for line in artifact.read_text().splitlines():
        for e in json.loads(line)['notes']:
            must(e['offset_sample']>e['onset_sample'],'Invalid exported NoteOff')
            replay[e['source_row_id']]+=1
    must(int(sum(replay.values()))==N_NOTES
        and all(replay[s]==int(k) for s,k in zip(ids,counts)),
        'S422 CRITICAL per-row K invariant failed after independent JSONL replay')
    before=get_score(tracks,midi,on,baseline,GT)
    after=get_score(tracks,midi,on,chosen,GT)
    must(before['matched_onset_pitch_50ms']==25395
        and before['correct_offsets_conditional_on_matched_pitch_onset']==16140,
        'S420 frozen F1 baseline not independently reproduced')
    by_duration=per_duration(p,chosen,baseline,GT)
    npz=out/'S422_NATIVE_pitch_survival_full_MIDI_note_offsets.npz'
    np.savez_compressed(npz,source_row_id=p['source_row_id'],
       MIDI=midi,track_member=tracks,onset_samples=on,
       S420_offset_samples=baseline,S422_offset_samples=chosen,
       original_S80_row_id=ids,original_S80_K=counts,composition_fold=folds)
    report=dict(status='S422_S420_NATIVE_FULL_WAV_PITCH_SURVIVAL_OFFSET_AUDIT',
       S80_predicted_K_full_lock=True,
       genuine_59309_S80_rows_preserved=True,
       independent_JSONL_replay_all_S80_K_pass=True,
       original_exactK_correct_global=53529,new_exactK_correct_global=53529,
       original_exactK_correct_poly=6054,new_exactK_correct_poly=6054,
       exactK_regressions=0,
       original_MIDI_pitch_onset_unchanged=True,
       note_count=N_NOTES,full_GuitarSet_native_events=N_EVENTS,
       survival_acoustic_every_10ms_including_K0=True,
       max_note_survival_context_ms=int(HORIZONS[-1]),
       no_YourMT3_weights_predictions_training_or_inference=True,
       inner_composition_only_threshold_policy_selection=True,
       source_MIC_different_than_S80_pickup=True,
       S80_historical_GuitarSet_overlap=True,
       not_prospective_whole_pipeline_independent=True,
       no_promotions=True,
       metrics=dict(S420={k:v for k,v in before.items() if k!='by_track'},
                    S422={k:v for k,v in after.items() if k!='by_track'}),
       by_true_duration=by_duration,
       policy_fold_proofs=policy_reports,
       file_hashes=dict(S420=sha(original_s420),S80=sha(s80),
          raw_MIC=sha(audio),GT=sha(annotations),notes=sha(artifact),NPZ=sha(npz)))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S422 SCORES',json.dumps({a:{k:v for k,v in b.items()
        if k!='by_reference_note_duration'} for a,b in report['metrics'].items()}),flush=True)
    print('S422 BY_DURATION',json.dumps(by_duration),flush=True)
    print('S422 PASS every 59309 S80 original K unchanged from emitted MIDI JSONL',flush=True)

def test():
    must(len(HORIZONS)>=50 and 500 in HORIZONS and HORIZONS[-1]==5000,
         'S422 full continuous horizons incomplete')
    class Fake:
        def predict_proba(self,z):
            q=np.where(z[:,0]<.6,.83,np.where(z[:,0]<1.2,.52,.08))
            return np.column_stack([1-q,q])
    x=np.zeros((3,len(HORIZONS),DIM),np.float32)
    x[:,:,0]=HORIZONS[None,:]/1000.
    d,alive=survival_estimate(Fake(),x,.5)
    must(np.all(d>=1200) and np.all(np.diff(alive,axis=1)<=1e-6),
         'S422 physical survival monotonic continuation test')
    old=np.array([10000,20000,30000],np.int64)
    on=np.array([1000,10000,20000],np.int64)
    result=corrected_offsets(old,on,d,alive,.5,'extend_only')
    must(np.all(result>=old) and np.all(result>on),
         'S422 survival-only corrector may not shorten baseline')
    print('S422 SELFTEST PASS real survival probability crossing, no GT model features, long-note windows')
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    for name in ('s420','s80','audio','annotations','out'):
        parser.add_argument('--'+name,type=Path)
    a=parser.parse_args()
    if a.self_test:test()
    else:
        for name in ('s420','s80','audio','annotations','out'):
            must(getattr(a,name)is not None,'Missing '+name)
        run(a.s420,a.s80,a.audio,a.annotations,a.out)
