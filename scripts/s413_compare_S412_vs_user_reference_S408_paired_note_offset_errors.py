"""S413 — independent paired correction/regression audit on FROZEN S412 vs
user-selected S408 Offset DEVELOPMENT reference. Does not modify predictions.

Cross-check all 25120 pitch+onset match pairs against original JAMS.
Counts corrections, regressions and long-note failures by composition fold,
S80 K, real duration and octave. S412 can only be promoted later if approved.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import get_gt,match_predicted_to_reference
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
SR=44100
def need(x,s):
    if not x:raise ValueError(s)
def group_name(ms):
    if ms<100:return '<100ms'
    if ms<500:return '100-499ms'
    if ms<1000:return '500-999ms'
    return '>=1000ms'
def compare(records):
    n=len(records)
    old=sum(int(r[0]) for r in records)
    new=sum(int(r[1]) for r in records)
    corr=sum(int(r[1] and not r[0]) for r in records)
    regress=sum(int(r[0] and not r[1]) for r in records)
    return dict(n=n,S408_correct=old,S412_correct=new,
       S408_accuracy=old/n if n else None,S412_accuracy=new/n if n else None,
       corrections=corr,regressions=regress,net=new-old,
       S408_early_over50ms=sum(int(r[2]<-50) for r in records),
       S408_late_over50ms=sum(int(r[2]>50) for r in records),
       S412_early_over50ms=sum(int(r[3]<-50) for r in records),
       S412_late_over50ms=sum(int(r[3]>50) for r in records),
       S408_MAE_ms=float(np.mean([abs(r[2]) for r in records])) if n else None,
       S412_MAE_ms=float(np.mean([abs(r[3]) for r in records])) if n else None)
def run(predictions,annotation,out):
    need(not out.exists(),'immutable S413 report destination required')
    with np.load(predictions,allow_pickle=False) as z:
        required=('track_member','midi_pitch','S80_K','S407_onset_samples',
          'S408_original_offset_samples','continuous_normal_direct_offset_samples',
          'continuous_long_weighted_direct_offset_samples','composition_fold')
        need(set(required).issubset(z.files),'S413 source policy outputs missing')
        p={key:z[key].copy() for key in required}
    n=len(p['midi_pitch'])
    need(n==33475,'Original S408 frozen 33475 note instances drift')
    tracks=p['track_member'].astype(str)
    pitches=p['midi_pitch'].astype(int)
    onset=p['S407_onset_samples'].astype(np.int64)
    old=p['S408_original_offset_samples'].astype(np.int64)
    direct=p['continuous_normal_direct_offset_samples'].astype(np.int64)
    weighted=p['continuous_long_weighted_direct_offset_samples'].astype(np.int64)
    need(np.all(old>onset) and np.all(direct>onset) and np.all(weighted>onset),
         'S413 predicted note offset not after onset')
    gt={}
    with zipfile.ZipFile(annotation,'r') as arch:
        names=collections.defaultdict(list)
        for s in arch.namelist():
            if s.endswith('.jams'):names[PurePosixPath(s).name].append(s)
        for track in sorted(set(tracks)):
            a=names[PurePosixPath(track).name]
            need(len(a)==1,'S413 ground truth track ambiguous')
            gt[track]=get_gt(arch.read(a[0]))
    need(len(gt)==190 and sum(map(len,gt.values()))==34954,
         'GuitarSet reference GT count missing')
    matched={}
    comparable=collections.defaultdict(list)
    by_k=collections.defaultdict(list)
    by_duration=collections.defaultdict(list)
    by_fold=collections.defaultdict(list)
    by_octave=collections.defaultdict(list)
    control_weighted=collections.defaultdict(list)
    for track,reference in gt.items():
        ii=np.flatnonzero(tracks==track)
        guesses=[(int(onset[j]),int(pitches[j]),int(j)) for j in ii]
        pairs=match_predicted_to_reference(guesses,reference,2205)
        for a,b in pairs:
            idx=guesses[a][2]
            start,stop,pp,string=reference[b]
            delta_o=(int(old[idx])-stop)/SR*1000
            delta_d=(int(direct[idx])-stop)/SR*1000
            delta_w=(int(weighted[idx])-stop)/SR*1000
            r=(abs(delta_o)<=50,abs(delta_d)<=50,delta_o,delta_d)
            w=(abs(delta_o)<=50,abs(delta_w)<=50,delta_o,delta_w)
            comparable['ALL'].append(r)
            by_k[f"K{int(p['S80_K'][idx])}"].append(r)
            by_duration[group_name((stop-start)/SR*1000)].append(r)
            by_fold[str(int(p['composition_fold'][idx]))].append(r)
            by_octave[str(pitches[idx]//12-1)].append(r)
            control_weighted[group_name((stop-start)/SR*1000)].append(w)
    need(len(comparable['ALL'])==25120,'S413 paired MIDI/onset 25120 drift')
    o=dict(
       all=compare(comparable['ALL']),
       by_predicted_K={k:compare(x) for k,x in by_k.items()},
       by_duration={k:compare(x) for k,x in by_duration.items()},
       by_song_fold={k:compare(x) for k,x in by_fold.items()},
       by_octave={k:compare(x) for k,x in by_octave.items()},
       long_weighted_control_by_duration={k:compare(x) for k,x in control_weighted.items()})
    need(o['all']['S408_correct']==13050 and o['all']['S412_correct']==14313,
         'S413 must reproduce S408 and S412 held-song frozen reference metrics')
    report=dict(status='S413_FROZEN_S412_VS_ACTIVE_S408_PAIRED_ERRORS',
        active_user_selected_offset_reference='S408_PITCH_CONDITIONED_SPARSE_FUTURE_AUDIO_NOTE_DURATION_REGRESSOR',
        challenger='S412_CONTINUOUS_WAV_MIC_NORMAL_DIRECT',
        note_K_pitch_onset_unchanged=True,
        original_predictor_data_GuitarSet_exposed_prior_to_this_audit=True,
        no_new_training_or_predictor_mutation=True,
        S408_baseline_still_active=True,
        primary_metric='full note MIDI+onset+offset ±50ms and conditioned Offset ±50ms',
        report=o,
        prediction_source_SHA256=sha(predictions),annotations_SHA256=sha(annotation),
        promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S413 PAIRED',json.dumps(o['all']),flush=True)
    print('S413 BY_DURATION',json.dumps(o['by_duration']),flush=True)
    print('S413 BY_FOLD',json.dumps(o['by_song_fold']),flush=True)
    print('S413 LONG_WEIGHTS',json.dumps(o['long_weighted_control_by_duration']),flush=True)
    print('S413 PASS paired analysis, S408 remains user-selected baseline',flush=True)
def selftest():
    x=[(True,False,-60.,-90.),(False,True,55.,1.),(True,True,0.,0.)]
    d=compare(x)
    need(d['S408_correct']==2 and d['S412_correct']==2 and
         d['corrections']==1 and d['regressions']==1,'S413 paired correction/regression accounting')
    print('S413 SELFTEST PASS')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--self-test',action='store_true')
    for n in ('predictions','annotations','out'):a.add_argument('--'+n,type=Path)
    z=a.parse_args()
    if z.self_test:selftest()
    else:
        for name in ('predictions','annotations','out'):
            need(getattr(z,name)is not None,'Missing '+name)
        run(z.predictions,z.annotations,z.out)
