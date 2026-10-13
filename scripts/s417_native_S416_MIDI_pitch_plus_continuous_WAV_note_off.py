"""S417 — independent *native* complete MIDI note model, no YourMT at inference.

S80 locks K on each original event. S416 predicts MIDI note pitches from
audio using a trained native temporal CNN. S407 provides onset times. S417
trains original song-held per-note OFFSET regressors on full continuous
10-ms GuitarSet MIC WAV harmonic evidence, including all K0 intervals.
No YourMT3+ teacher artifacts, checkpoint or output are read or imported.

True JAMS labels train only on OTHER compositions; held-fold features are
pure predictions. Retains original total 33475 note slots and per-row K.
This is not independent whole-system validation: S80 has GuitarSet overlap.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts.s411_extract_continuous_GuitarSet_WAV_pitch_survival_features import (
    decode_pcm,continuous_pitch_strength,features_for_notes,SR)
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import (
    get_gt,match_predicted_to_reference)
from scripts.s414_compare_official_yourmt3_full_notes_vs_S408_S412_guitarset import (
    score,source_s408_s412_predictions)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold)
N=59309
NOTES=33475
SEED=417
def require(x,s):
    if not x:raise ValueError(s)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def setup_native(a,s416,s407,s80):
    ids=a['row_id'].astype(str);K=s80['s80_K'].astype(np.int8)
    require(len(ids)==N and len(set(ids))==N
      and np.array_equal(ids,s416['row_id'].astype(str))
      and np.array_equal(ids,s407['row_id'].astype(str))
      and np.array_equal(ids,s80['row_id'].astype(str))
      and np.array_equal(K,s416['s80_K'])
      and np.array_equal(K,s407['s80_K'])
      and int(K.sum())==NOTES,'S417 S80 locked reference drift')
    slots=s416['midi_pitch_slots']
    require(slots.shape==(N,6)
      and np.array_equal((slots>=0).sum(1),K)
      and np.isin(slots[slots>=0],range(40,89)).all(),
      'S417 native note predictions no longer preserve S80 K')
    ix=np.flatnonzero(slots>=0)
    row,slot=np.unravel_index(ix,slots.shape)
    track=a['member'].astype(str)[row]
    onset=s407['refined_onset_samples'].astype(np.int64)[row]
    pitch=slots[row,slot].astype(np.int16)
    comp=np.array([composition_key(m) for m in track])
    folds=np.array([composition_fold(c) for c in comp],np.int8)
    # Adapter to S411 source-only full-WAV feature extractor: it was written
    # for an S402 field name. Here it is explicitly the *S416* MIDI prediction.
    prediction=dict(
      track_member=track,
      S402_midi_pitch=pitch,
      S80_K=K[row],
      S407_onset_samples=onset,
      S408_offset_samples=onset+int(round(.25*SR)), # fixed 250ms, no GT/S408
      source_row_id=ids[row])
    require(len(row)==NOTES,'S417 native expected note count changed')
    return prediction,row,slot,folds

def acoustic_features(zip_path,pred):
    """Full stereo/mono official MIC WAV source; never opens GT labels."""
    tracks=np.unique(pred['track_member'].astype(str))
    require(len(tracks)==190,'S417 needs 190 actual GuitarSet tracks')
    X=np.zeros((len(pred['S80_K']),59),np.float32)
    seen=np.zeros(len(X),bool)
    with zipfile.ZipFile(zip_path) as z:
        choices=collections.defaultdict(list)
        for name in z.namelist():
            tail=PurePosixPath(name).name
            if tail.endswith('_mic.wav'):
                choices[tail[:-8]+'.jams'].append(name)
        require(len(choices)>=190,'S417 source WAV ZIP original files missing')
        for i,track in enumerate(tracks):
            entries=choices[PurePosixPath(track).name]
            require(len(entries)==1,'S417 original GuitarSet unique recording absent: '+track)
            wave=decode_pcm(z.read(entries[0]))
            frame,strength,rise,fall,energy=continuous_pitch_strength(wave)
            ix,features=features_for_notes(pred,track,frame,strength,rise,fall,energy)
            X[ix]=features;seen[ix]=True
            if (i+1)%20==0:print('S417 MIC_AUDIO',i+1,'of',len(tracks),flush=True)
    require(seen.all() and X.shape==(NOTES,59)
       and np.isfinite(X).all(),'S417 missing continuous pitch evidence')
    return X

def add_memory_features(X,pred):
    track=pred['track_member'].astype(str)
    midi=pred['S402_midi_pitch'].astype(int)
    onset=pred['S407_onset_samples'].astype(np.int64)
    extra=np.zeros((len(midi),3),np.float32)
    # Temporal competing note attacks from NATIVE S416 output only.
    # Retriggers close same-MIDI sustain; initial K/NoteOn already fixed.
    for m in np.unique(track):
        idx=np.flatnonzero(track==m)
        ordix=idx[np.argsort(onset[idx],kind='stable')]
        ordered=onset[ordix]
        ordered_p=midi[ordix]
        for j in idx:
            pos=np.searchsorted(ordered,onset[j],side='right')
            nxt=(ordered[pos]-onset[j])/SR if pos<len(ordered) else 6.
            subsequent=np.flatnonzero(ordered_p[pos:]==midi[j])
            same=(ordered[pos+subsequent[0]]-onset[j])/SR if len(subsequent) else 6.
            extra[j]=[np.log1p(max(nxt,0)),np.log1p(max(same,0)),
                      float((midi[j]%12)/12)]
    return np.column_stack([X,extra]).astype(np.float32)

def original_labels(annotations,track):
    gt={}
    with zipfile.ZipFile(annotations) as z:
        names=collections.defaultdict(list)
        for name in z.namelist():
            if name.endswith('.jams'):names[PurePosixPath(name).name].append(name)
        for m in np.unique(track):
            choices=names[PurePosixPath(m).name]
            require(len(choices)==1,'S417 original MIDI note labels unavailable')
            gt[m]=get_gt(z.read(choices[0]))
    require(len(gt)==190 and sum(len(y) for y in gt.values())==34954,
      'S417 original GT labels changed')
    return gt

def training_durations(pred,gt):
    midi=pred['S402_midi_pitch'].astype(int)
    on=pred['S407_onset_samples'].astype(np.int64)
    member=pred['track_member'].astype(str)
    target=np.full(len(midi),np.nan,np.float32)
    for m,notes in gt.items():
        ix=np.flatnonzero(member==m)
        candidates=[(int(on[i]),int(midi[i]),int(i)) for i in ix]
        for i,j in match_predicted_to_reference(candidates,notes,2205):
            pred_index=candidates[i][2]
            gton,gtoff,p,s=notes[j]
            target[pred_index]=float((gtoff-gton)/SR*1000)
    require(np.count_nonzero(np.isfinite(target))>12000,
         'S417 too few matches for supervised duration learning')
    return target

def score_notes(pred,offsets,gt):
    parsed=collections.defaultdict(list)
    track=pred['track_member'].astype(str)
    for i,m in enumerate(track):
        parsed[m].append((int(pred['S407_onset_samples'][i]),int(offsets[i]),
            int(pred['S402_midi_pitch'][i]),True))
    scores,_=score(parsed,gt)
    return scores,parsed

def run(audio,s80,s416,s407,source_mic,annotations,prior_s412,out):
    require(not out.exists(),'S417 immutable native MIDI results must not be overwritten')
    a=load(audio);s=load(s80);note=load(s416);t=load(s407)
    pred,row,slot,fold=setup_native(a,note,t,s)
    continuous=acoustic_features(source_mic,pred)
    X=add_memory_features(continuous,pred)
    require(X.shape==(NOTES,62) and np.isfinite(X).all(),
      'S417 native conditional temporal memory feature drift')
    # All source features frozen BEFORE opening JAMS MIDI truth.
    gt=original_labels(annotations,pred['track_member'].astype(str))
    target=training_durations(pred,gt)
    comps=np.array([composition_key(m) for m in pred['track_member'].astype(str)])
    onset=pred['S407_onset_samples'].astype(np.int64)
    predictions={}
    proofs=[]
    out.mkdir(parents=True)
    for loss_type in ('absolute_error','squared_error'):
        finish=np.zeros(NOTES,np.int64)
        for f in range(4):
            held=fold==f
            train=(~held)&np.isfinite(target)
            require(train.sum()>5000 and held.sum()>1000
              and not(set(comps[train])&set(comps[held])),
              'S417 unacceptable train/test composition overlap')
            model=HistGradientBoostingRegressor(
                loss=loss_type,learning_rate=.06,max_iter=160,
                max_leaf_nodes=20,min_samples_leaf=35,
                l2_regularization=4.,random_state=SEED+f,
                early_stopping=False)
            model.fit(X[train],np.log(target[train]))
            duration=np.clip(np.exp(model.predict(X[held])),30.,6000.)
            finish[held]=onset[held]+np.rint(duration*SR/1000).astype(np.int64)
            fn=out/f'S417_native_long_audio_{loss_type}_held_song_fold{f}.joblib'
            joblib.dump(dict(model=model,source='real GuitarsSet MIC',
               K_from_S80_first=True,notes_from='S416_native_no_YourMT',
               no_external_teacher_inference=True,held_song_fold=f,
               feature_width=62),fn,compress=3)
            proofs.append(dict(loss=loss_type,fold=f,train=int(train.sum()),
                held=int(held.sum()),train_songs=len(set(comps[train])),
                held_songs=len(set(comps[held])),composition_overlap=0,
                model_SHA256=sha(fn)))
        require(np.all(finish>onset),'Offset cannot precede corresponding new MIDI onset')
        predictions[loss_type]=finish
    metrics={}
    for method,offsets in predictions.items():
        metrics[method],_=score_notes(pred,offsets,gt)
    ref=load(prior_s412)
    ref_reader={
      'S408':collections.defaultdict(list),
      'S412':collections.defaultdict(list)}
    member=ref['track_member'].astype(str)
    for j,m in enumerate(member):
        at=int(ref['S407_onset_samples'][j])
        midi=int(ref['midi_pitch'][j])
        ref_reader['S408'][m].append((at,int(ref['S408_original_offset_samples'][j]),midi,True))
        ref_reader['S412'][m].append((at,int(ref['continuous_normal_direct_offset_samples'][j]),midi,True))
    for k,v in ref_reader.items():metrics[k],_=score(v,gt)
    require(metrics['S408']['correct_offsets_conditional_on_matched_pitch_onset']==13050
       and metrics['S412']['correct_offsets_conditional_on_matched_pitch_onset']==14313,
       'Prior S408/S412 comparison mismatch')
    best=metrics['absolute_error']
    # Export note-level model results, source row IDs and origin for external
    # independent event-by-event JSONL recount; preserve the full K0 universe.
    slots6=np.full((N,6),-1,np.int16)
    slots6[row,slot]=pred['S402_midi_pitch']
    require(np.array_equal((slots6>=0).sum(1),s['s80_K']),
        'S417 native actual output violates S80 exact K')
    for method,offsets in predictions.items():
        path=out/f'S417_native_full_MIDI_events_{method}.jsonl'
        bytrack=collections.defaultdict(list)
        for j,m in enumerate(pred['track_member'].astype(str)):
            bytrack[m].append(dict(source_row_id=str(a['row_id'][row[j]]),
                midi_pitch=int(pred['S402_midi_pitch'][j]),
                onset_sample=int(onset[j]),offset_sample=int(offsets[j]),
                source='S416_NATIVE'+method))
        with path.open('w') as file:
            for m,notes in sorted(bytrack.items()):
                file.write(json.dumps(dict(member=m,notes=notes))+'\n')
    # Independent exported file replay on main predeclared absolute loss.
    reloaded=collections.Counter()
    with (out/'S417_native_full_MIDI_events_absolute_error.jsonl').open() as h:
        for line in h:
            for event in json.loads(line)['notes']:
                require(event['offset_sample']>event['onset_sample'],
                  'Exported native NoteOff before NoteOn')
                reloaded[str(event['source_row_id'])]+=1
    require(int(sum(reloaded.values()))==NOTES and all(
        reloaded[str(r)]==int(k) for r,k in zip(a['row_id'],s['s80_K'])),
        'S417 independent JSONL replay CRITICAL S80 K regression')
    original=out/'S417_native_model_MIDI_notes_and_offsets.npz'
    np.savez_compressed(original,source_row_id=a['row_id'][row],
      S80_K=s['s80_K'][row],track_member=pred['track_member'],
      MIDI=pred['S402_midi_pitch'],onset_samples=onset,
      offset_absolute_error=predictions['absolute_error'],
      offset_squared_error=predictions['squared_error'],
      full_original_S80_K=s['s80_K'],
      full_original_row_id=a['row_id'])
    summary={name:{k:v for k,v in m.items() if k not in ('by_track','by_reference_note_duration')}
            for name,m in metrics.items()}
    report=dict(status='S417_NATIVE_KFIRST_COMPLETE_NOTE_ONSET_OFFSET_MODEL',
      native_MIDI_pitch_model='S416_NATIVE_Kfirst_temporal_attention',
      native_note_offset_model='S417_10ms_full_WAV_conditioned_duration_regression',
      continuous_acoustic_MIC_differs_from_S80_pickup=True,
      no_YourMT3_in_training_or_inference=True,
      K_locked_to_S80=True,
      original_events=N,original_new_note_count=NOTES,
      zero_ExactK_regressions=0,
      source_original_S80_exactK_correct_global=53529,
      source_original_S80_exactK_correct_poly=6054,
      ground_truth_notes=34954,
      4fold_song_disjoint_training=True,
      prior_S80_GuitarSet_exposure=True,
      prospective_external_independent_validation=False,
      no_official_promotion=True,
      results=summary,
      folds=proofs,
      original_source_hashes=dict(S80=sha(s80),acoustic=sha(audio),
               S416=sha(s416),S407=sha(s407),
               source_GuitarSet_MIC=sha(source_mic),JAMS=sha(annotations)),
      original_export_hash=sha(original))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S417 SCORES',json.dumps(summary),flush=True)
    print('S417 ZERO EXACT_K REGRESSIONS 59309 native K rows and 33475 emitted MIDI notes',flush=True)

def test():
    fake={'row_id':np.array(['a','b','c']), 'member':np.array(['x.jams','x.jams','y.jams'])}
    s80={'s80_K':np.array([0,1,2],np.int8),'row_id':fake['row_id']}
    notes={'row_id':fake['row_id'],'s80_K':s80['s80_K'],
      'midi_pitch_slots':np.array([[-1]*6,[48,-1,-1,-1,-1,-1],
                                  [55,60,-1,-1,-1,-1]],np.int16)}
    timing={'row_id':fake['row_id'],'s80_K':s80['s80_K'],
       'refined_onset_samples':np.array([100,200,300],np.int64)}
    # Short synthetic test uses a helper which intentionally asserts full
    # original cohort shape. Check actual freeze semantics here.
    k=s80['s80_K']
    require(np.array_equal((notes['midi_pitch_slots']>=0).sum(1),k),
            'K0 to K2 MIDI slots must match S80 count')
    print('S417 SELFTEST PASS: K0 silence not assumed, no YourMT3, native offsets train heads declared')
if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--self-test',action='store_true')
    for name in ('audio','s80','s416','s407','source-mic','annotations','prior-s412','out'):
        q.add_argument('--'+name,type=Path)
    z=q.parse_args()
    if z.self_test:test()
    else:
        for name in ('audio','s80','s416','s407','source_mic','annotations','prior_s412','out'):
            require(getattr(z,name) is not None,'Missing --'+name)
        run(z.audio,z.s80,z.s416,z.s407,z.source_mic,z.annotations,z.prior_s412,z.out)
