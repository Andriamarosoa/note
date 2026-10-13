"""S414 exact same GuitarSet MIDI full-note benchmark YourMT3+ vs S408/S412.

YourMT3+ notes are AUTHENTIC previously generated 2026-10-07 by official
YPTF.MoE+Multi(noPS) checkpoint run 37605163312, not rerun/synthesized.
Input fold0,1,2,4 artifact notes.jsonl (onset, offset, pitch, instrument).
No labels used to change model predictions. Evaluation reads original
190 JAMS GuitarSet annotations and matches ONE TO ONE same MIDI pitch,
onset difference <=50ms. Correct complete note also needs offset <=50ms.

The baseline S408/S412 note heads are development-exposed to GuitarSet
upstream S80. The pretrained YourMT3+ includes GuitarSet in its training
corpus, so this is an IN-SAMPLE DESCRIPTIVE comparison for BOTH systems,
NOT independent generalization and NOT latency matched.
"""
from __future__ import annotations
import argparse,collections,json,math,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import (
    get_gt,match_predicted_to_reference)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
SR=44100
FOLDS=(0,1,2,4)
N_GT=34954
TRACKS=190
def need(b,msg):
    if not b:raise ValueError(msg)
def load_ymt_folds(fold_paths):
    predicted={}
    run_file_hash={}
    anomalies=collections.Counter()
    for fold,path in zip(FOLDS,fold_paths):
        fn=path/'notes.jsonl'
        need(fn.is_file(),'Missing real YourMT3+ fold notes.jsonl: '+str(fn))
        run_file_hash[str(fold)]=sha(fn)
        count=0
        for ln in fn.read_text(encoding='utf-8').splitlines():
            if not ln.strip():continue
            o=json.loads(ln)
            member=str(o['member'])
            need(member.endswith('.jams') and member not in predicted,
                 'Duplicate or bad YourMT3+ note track '+member)
            need(member[:2] in ('00','01','02','03','04'),
                 'Unexpected GuitarSet player in YourMT3+ fold')
            notes=[]
            for n in o['notes']:
                start=n.get('onset_sample')
                if start is None:
                    need(isinstance(n.get('onset'),(float,int)),'YourMT3+ Note onset missing')
                    start=round(n['onset']*SR)
                end=n.get('offset_sample')
                if end is None:
                    off=n.get('offset')
                    if isinstance(off,(float,int)) and math.isfinite(off):
                        end=round(off*SR)
                pp=n.get('pitch')
                need(isinstance(pp,(int,float)) and
                     math.isfinite(float(pp)) and float(pp).is_integer(),
                     'YourMT3+ MIDI pitch invalid')
                if end is None or int(end)<=int(start):
                    anomalies['missing_or_bad_offset']+=1
                    # Keep note with invalid offset as predicted FP on full-note
                    # and as matchable onset candidate; invalid offset sentinel.
                    end=-1
                notes.append((int(start),int(end),int(pp),
                              bool(n.get('global_index') is not None)))
            predicted[member]=notes
            count+=1
        anomalies['fold_'+str(fold)+'_tracks']=count
    need(len(predicted)==TRACKS,
         'Expected EXACTLY 190 original YourMT3+ source tracks; got '+str(len(predicted)))
    return predicted,dict(anomalies),run_file_hash

def read_original_jams(path,tracks):
    labels={}
    with zipfile.ZipFile(path) as z:
        filenames=collections.defaultdict(list)
        for name in z.namelist():
            if name.endswith('.jams'):filenames[PurePosixPath(name).name].append(name)
        for track in sorted(tracks):
            items=filenames[PurePosixPath(track).name]
            need(len(items)==1,'Original GuitarSet JAMS track ambiguous '+track)
            labels[track]=get_gt(z.read(items[0]))
    need(len(labels)==TRACKS and sum(len(x) for x in labels.values())==N_GT,
         'Original GuitarSet annotation count changed')
    return labels

def source_s408_s412_predictions(path):
    with np.load(path,allow_pickle=False) as z:
        required=('track_member','midi_pitch','S407_onset_samples',
           'S408_original_offset_samples','continuous_normal_direct_offset_samples')
        need(set(required).issubset(z.files),'Missing genuine frozen S408/S412 note outputs')
        a={x:z[x].copy() for x in required}
    n=len(a['midi_pitch'])
    need(n==33475,'Source S408/S412 33475 predicted notes altered')
    out={'S408':collections.defaultdict(list),
         'S412_continuous':collections.defaultdict(list)}
    for i in range(n):
        track=str(a['track_member'][i]);start=int(a['S407_onset_samples'][i])
        pitch=int(a['midi_pitch'][i])
        out['S408'][track].append((start,int(a['S408_original_offset_samples'][i]),pitch,True))
        out['S412_continuous'][track].append((start,int(a['continuous_normal_direct_offset_samples'][i]),pitch,True))
    return out

def duration_bin(ms):
    return ('<100ms' if ms<100 else '100-499ms' if ms<500 else
            '500-999ms' if ms<1000 else '>=1000ms')

def score(predicted,truth,only_native_events=False,keep_examples=False):
    summary=collections.Counter()
    by_duration=collections.defaultdict(collections.Counter)
    by_track=collections.defaultdict(collections.Counter)
    examples=[]
    for track,reference in truth.items():
        notes=predicted.get(track,[])
        if only_native_events:
            notes=[n for n in notes if n[3]]
        summary['pred']+=len(notes)
        summary['ref']+=len(reference)
        pairs=match_predicted_to_reference(
             [(n[0],n[2],i) for i,n in enumerate(notes)],
             reference,2205)
        summary['onset_pitch_tp']+=len(pairs)
        by_track[track]['matched']+=len(pairs)
        for pi,gi in pairs:
            p=notes[pi];r=reference[gi]
            off=abs(p[1]-r[1])<=2205 if p[1]>p[0] else False
            dur=(r[1]-r[0])/SR*1000
            off_rel=abs(p[1]-r[1])<=max(2205,int(.2*(r[1]-r[0]))) if p[1]>p[0] else False
            summary['complete_50ms_tp']+=int(off)
            summary['complete_50ms_or20pct_tp']+=int(off_rel)
            if p[1]>p[0]:
                summary['valid_offset_matched']+=1
                summary['abs_offset_error_samples']+=abs(p[1]-r[1])
            binname=duration_bin(dur)
            by_duration[binname]['match']+=1
            by_duration[binname]['complete_50ms_tp']+=int(off)
            by_track[track]['complete_50ms_tp']+=int(off)
            if keep_examples and len(examples)<8 and not off:
                examples.append(dict(track=track,pitch=p[2],
                     predicted_onset_sec=round(p[0]/SR,3),
                     gt_duration_ms=round(dur,1),
                     predicted_offset_sec=round(p[1]/SR,3),
                     gt_offset_sec=round(r[1]/SR,3)))
    n=summary['pred'];g=summary['ref']
    tp=summary['onset_pitch_tp'];full=summary['complete_50ms_tp']
    metric=dict(predicted_notes=n,ground_truth_notes=g,
       matched_onset_pitch_50ms=tp,
       onset_pitch_precision=tp/n if n else None,
       onset_pitch_recall=tp/g if g else None,
       onset_pitch_F1=2*tp/(n+g) if n+g else None,
       correct_offsets_conditional_on_matched_pitch_onset=full,
       conditional_offset_50ms_accuracy=full/tp if tp else None,
       full_note_F1_50ms=2*full/(n+g) if n+g else None,
       full_note_F1_50ms_or20pct=
          2*summary['complete_50ms_or20pct_tp']/(n+g) if n+g else None,
       matched_valid_offsets=int(summary['valid_offset_matched']),
       mean_absolute_offset_error_ms=
          summary['abs_offset_error_samples']/summary['valid_offset_matched']/
          SR*1000 if summary['valid_offset_matched'] else None,
       by_reference_note_duration={name:dict(
          matched_pitch_onset=x['match'],
          correct_offset=x['complete_50ms_tp'],
          conditional_accuracy=x['complete_50ms_tp']/x['match'] if x['match'] else None)
          for name,x in sorted(by_duration.items())},
       by_track={name:dict(n_onset_pitch=x['matched'],n_complete=x['complete_50ms_tp'])
                  for name,x in sorted(by_track.items())}
    )
    return metric,examples

def compare(official,ours,truth):
    scores={}
    for name,data in [(name,ours[name]) for name in ('S408','S412_continuous')]+[
            ('YourMT3plus_all_decoded',official),
            ('YourMT3plus_only_native_assigned',official)]:
        restrict=name.endswith('_native_assigned')
        result,_=score(data,truth,only_native_events=restrict)
        scores[name]=result
    need(scores['S408']['predicted_notes']==33475
      and scores['S408']['matched_onset_pitch_50ms']==25120
      and scores['S408']['correct_offsets_conditional_on_matched_pitch_onset']==13050,
      'S408 pinned baseline label/timing/pitch matching drift')
    need(scores['S412_continuous']['correct_offsets_conditional_on_matched_pitch_onset']==14313,
      'S412 challenger pinned note score drift')
    return scores

def run(folds,s412,annotations,out):
    need(not out.exists(),'S414 immutable original benchmark cannot overwrite')
    upstream,anomalies,hashes=load_ymt_folds(folds)
    candidate=source_s408_s412_predictions(s412)
    truth=read_original_jams(annotations,upstream)
    for name in ('S408','S412_continuous'):
        need(set(candidate[name])==set(upstream),
             'Different acoustic cohort tracks! No valid comparison: '+name)
    scores=compare(upstream,candidate,truth)
    report=dict(
       status='S414_REAL_OFFICIAL_YOURMT3_PLUS_VS_S408_S412_SAME_GUITARSET_NOTES',
       original_GuitarSet_tracks=TRACKS,
       ground_truth_note_events=N_GT,
       official_YourMT3_source_run_id=37605163312,
       official_checkpoint='YPTF.MoE+Multi (noPS)',
       source_audio_of_YourMT3='GuitarSet mono pickup mix',
       new_continuous_S412_offset_audio_source='GuitarSet mono microphone',
       both_same_original_190_tracks_and_JAMS=True,
       note_matching='same MIDI pitch, one-to-one onset <=50ms, offset absolute <=50ms',
       extra_offsets='offset<=max(50ms,20% true note duration)',
       historical_YourMT3_model_may_have_seen_GuitarSet=True,
       S80_model_also_historically_overlaps_GuitarSet=True,
       no_latency_matched_comparison=True,
       descriptive_development_only=True,
       no_model_weights_or_hyperparams_modified=True,
       YourMT3_note_source_hashes=hashes,
       malformed_or_missing_YourMT3_offsets=anomalies,
       S412_predicted_note_source_sha256=sha(s412),
       ground_truth_JAMS_sha256=sha(annotations),
       metrics=scores,
       no_automatic_promotion=True)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Small report avoids github job logs truncation.
    summary={n:{k:v for k,v in row.items() if k not in ('by_track','by_reference_note_duration')}
        for n,row in scores.items()}
    print('S414 COMPARISON',json.dumps(summary),flush=True)
    print('S414 DURATION',json.dumps({k:v['by_reference_note_duration'] for k,v in scores.items()}),flush=True)
    print('S414 PASS real checkpoints notes onset+pitch+offset against shared GuitarSet annotations',flush=True)

def selftest():
    ref={'00_a.jams':[(100,550,48,0),(200,750,52,1)]}
    p={'00_a.jams':[(101,550,48,True),(200,800,52,False)]}
    a,_=score(p,ref)
    need(a['matched_onset_pitch_50ms']==2
       and a['correct_offsets_conditional_on_matched_pitch_onset']==2
       and a['full_note_F1_50ms']==1.,
       'S414 synthetic note comparison bad')
    restricted,_=score(p,ref,only_native_events=True)
    need(restricted['predicted_notes']==1 and
         restricted['matched_onset_pitch_50ms']==1,
         'S414 native cohort filtering must be explicit')
    print('S414 SELFTEST PASS genuine same-pitch onset and offset matching')
if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--self-test',action='store_true')
    for f in FOLDS:q.add_argument('--fold'+str(f),type=Path)
    q.add_argument('--s412',type=Path)
    q.add_argument('--annotations',type=Path)
    q.add_argument('--out',type=Path)
    z=q.parse_args()
    if z.self_test:selftest()
    else:
        for n in ('fold0','fold1','fold2','fold4','s412','annotations','out'):
            need(getattr(z,n) is not None,'Missing --'+n)
        run([getattr(z,'fold'+str(f)) for f in FOLDS],
            z.s412,z.annotations,z.out)
