"""S420 — native full MIDI note hybrid: S419 note-pitch selection + S417
native WAV temporal NoteOff, with strict original S80 per-event K.

No YourMT3+ inference, weights or predictions used anywhere. S419 chooses
native S402/S416 pitch events using only predicted S80 K. The four already
trained S417 held-song duration models are reused WITHOUT fitting/tuning
against the held songs, on fresh 10ms full-MIC acoustic features for the
new S419-selected pitch events. These duration models were trained on S416
candidates, so distribution shift to some S402 pitches is a limitation.

Native S80 reference completely untouched, and S408 reference NOT promoted.
"""
from __future__ import annotations
import argparse,collections,csv,json
from pathlib import Path
import joblib,numpy as np
from scripts.s417_native_S416_MIDI_pitch_plus_continuous_WAV_note_off import (
    setup_native,acoustic_features,add_memory_features,
    original_labels,score_notes,load)
from scripts.s414_compare_official_yourmt3_full_notes_vs_S408_S412_guitarset import score
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import match_predicted_to_reference
from scripts.s411_extract_continuous_GuitarSet_WAV_pitch_survival_features import SR

N=59309
NOTES=33475
def need(x,s):
    if not x:raise ValueError(s)
def output_row_events(path,pred,row,offsets):
    by_track=collections.defaultdict(list)
    all_ids=np.asarray(pred['source_row_id']).astype(str)
    for i,member in enumerate(pred['track_member'].astype(str)):
        by_track[member].append(dict(
            source_row_id=str(all_ids[i]),pitch=int(pred['S402_midi_pitch'][i]),
            onset_sample=int(pred['S407_onset_samples'][i]),
            offset_sample=int(offsets[i]),source='S419_native_pitch_S417_native_offset'))
    with path.open('w',encoding='utf8') as out:
        for member,events in sorted(by_track.items()):
            out.write(json.dumps(dict(member=member,notes=events))+'\n')
    # Independent whole-file recount, never derive count only from an
    # auxiliary array copied from S80 or model internals.
    reread=collections.Counter()
    with path.open(encoding='utf8') as fp:
        for line in fp:
            for ev in json.loads(line)['notes']:
                need(ev['offset_sample']>ev['onset_sample'],
                     'S420 output NoteOff not strictly after NoteOn')
                need(40<=ev['pitch']<=88,'S420 output MIDI note out of range')
                reread[ev['source_row_id']]+=1
    return reread

def run(audio,s80,s419,s407,source_mic,model_dir,prior_s417,prior_s412,annotations,out):
    need(not out.exists(),'S420 immutable audit folder exists')
    a=load(audio);ref=load(s80);chosen=load(s419);timing=load(s407)
    p,row,slot,fold=setup_native(a,chosen,timing,ref)
    need(len(row)==NOTES and len(a['row_id'])==N,
         'S420 frozen source K cardinality drift')
    continuous=acoustic_features(source_mic,p)
    X=add_memory_features(continuous,p)
    need(X.shape==(NOTES,62),'S420 feature width does not equal S417 trained models')
    onset=p['S407_onset_samples'].astype(np.int64)
    offset=np.zeros(NOTES,np.int64)
    modelhash={}
    for f in range(4):
        source=model_dir/f'S417_native_long_audio_absolute_error_held_song_fold{f}.joblib'
        need(source.is_file(),'Missing S417 truly trained native note-off model '+str(source))
        data=joblib.load(source)
        need(data['K_from_S80_first'] is True
          and data['no_external_teacher_inference'] is True
          and data['feature_width']==62
          and data['held_song_fold']==f,
          'S417 native offset source provenance mismatch')
        mask=fold==f
        duration=np.clip(np.exp(data['model'].predict(X[mask])),30,6000)
        offset[mask]=onset[mask]+np.rint(duration*SR/1000).astype(np.int64)
        modelhash[str(f)]=sha(source)
    need(np.all(offset>onset),'S420 predicted NoteOff not after NoteOn')
    out.mkdir(parents=True)
    exported=out/'S420_native_full_MIDI_note_events.jsonl'
    real_rows=output_row_events(exported,p,row,offset)
    ids=a['row_id'].astype(str);K=ref['s80_K'].astype(np.int8)
    need(len(ids)==N and int(K.sum())==int(sum(real_rows.values()))==NOTES
      and all(real_rows[str(i)]==int(k) for i,k in zip(ids,K)),
      'CRITICAL Exact K REGRESSION: actual S420 exported NoteOn count differs from S80')
    # Output frozen before the annotations are used for SCORING only.
    gt=original_labels(annotations,p['track_member'].astype(str))
    main,_=score_notes(p,offset,gt)
    older=load(prior_s417)
    # Baselines evaluated with their ORIGINAL pitch and onset predictions,
    # all against the exact same 190 songs and 34954 truth MIDI notes.
    predictions=collections.defaultdict(list)
    predictions412=collections.defaultdict(list)
    for i,track in enumerate(older['track_member'].astype(str)):
        predictions[track].append((int(older['onset_samples'][i]),
              int(older['offset_absolute_error'][i]),int(older['MIDI'][i]),True))
    challenger=load(prior_s412)
    for i,track in enumerate(challenger['track_member'].astype(str)):
        predictions412[track].append((int(challenger['S407_onset_samples'][i]),
          int(challenger['continuous_normal_direct_offset_samples'][i]),
          int(challenger['midi_pitch'][i]),True))
    S417,_=score(predictions,gt)
    S412,_=score(predictions412,gt)
    need(S417['correct_offsets_conditional_on_matched_pitch_onset']==16017
      and S412['correct_offsets_conditional_on_matched_pitch_onset']==14313,
      'Native S417 and original S412 full-note reference drift')
    # F1 complete notes may increase or decrease; no silent promotion.
    scores={}
    for name,s in [('S412',S412),('S417_native',S417),('S420_native',main)]:
        scores[name]={k:v for k,v in s.items() if k!='by_track'}
    outnpz=out/'S420_original_K_NATIVE_full_notes_33475.npz'
    np.savez_compressed(outnpz,source_row_id=p['source_row_id'],MIDI=p['S402_midi_pitch'],
       track_member=p['track_member'],onset_samples=onset,
       native_offset_samples=offset,original_S80_K=K,
       original_S80_row_id=ids,
       native_note_slots=chosen['midi_pitch_slots'])
    report=dict(
      status='S420_S419_NATIVE_NOTE_HEAD_WITH_S417_NATIVE_CONTINUOUS_NOTE_OFF',
      same_original_events=N,new_note_instances=NOTES,
      S80_REFERENCE_K0_K6_UNTOUCHED=True,
      original_ExactK_correct_global=53529,
      new_ExactK_correct_global=53529,
      original_ExactK_correct_poly=6054,
      new_ExactK_correct_poly=6054,
      independently_replayed_JSONL_row_ExactK=True,
      exactK_regressions=0,
      no_YourMT3_in_training_or_inference=True,
      no_GT_for_any_model_input=True,
      no_retraining_or_test_set_tuning=True,
      S417_note_duration_models_trained_on_S416_pitch_candidates=True,
      S419_selected_pitch_domain_shift_to_some_S402_pitches=True,
      prior_S80_GuitarSet_training_overlap_known=True,
      source_MIC_differs_from_S80_pickup=True,
      no_official_promotion=True,
      metrics=scores,
      selected_head='S419_native_mixed_S402_S416',
      native_note_offsets='S417_frozen_long_audio_absolute_error',
      S417_model_sha256=modelhash,
      source_hashes=dict(S80=sha(s80),S419=sha(s419),S407=sha(s407),
          source_MIC=sha(source_mic),JAMS=sha(annotations)),
      predictions_sha256=sha(outnpz),actual_MIDI_JSONL_SHA256=sha(exported))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S420 RESULTS',json.dumps({name:{k:v for k,v in s.items()
       if k not in ('by_reference_note_duration',)} for name,s in scores.items()}),flush=True)
    print('S420 PASS zero regressions of ALL original 59309 S80 K, via INDEPENDENT exported MIDI JSONL recount',flush=True)

def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        fn=Path(folder)/'events.jsonl'
        pred={'source_row_id':np.array(['0','1','1','2','2','2']),
          'track_member':np.array(['a.jams']*6),
          'S402_midi_pitch':np.array([48,48,60,48,52,55]),
          'S407_onset_samples':np.array([100,200,200,300,300,300])}
        offset=np.array([800,900,1000,1100,800,700])
        counts=output_row_events(fn,pred,np.arange(6),offset)
        need([counts[str(i)] for i in range(3)]==[1,2,3],
             'S420 independent JSONL K recount 1,2,3')
    print('S420 SELFTEST PASS K1 K2 K3 counted from actual exported NoteOn rows')
if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--self-test',action='store_true')
    for name in ('audio','s80','s419','s407','source-mic','model-dir',
                 'prior-s417','prior-s412','annotations','out'):
        q.add_argument('--'+name,type=Path)
    a=q.parse_args()
    if a.self_test:self_test()
    else:
        for name in ('audio','s80','s419','s407','source_mic','model_dir','prior_s417',
                     'prior_s412','annotations','out'):
            need(getattr(a,name)is not None,'Missing --'+name)
        run(a.audio,a.s80,a.s419,a.s407,a.source_mic,a.model_dir,
            a.prior_s417,a.prior_s412,a.annotations,a.out)
