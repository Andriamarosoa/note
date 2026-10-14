"""S409 — Independent immutable OUTPUT audit of S408 note-level offsets.

No model, hyperparameter, checkpoint or correction changed. Interpret where
S408 genuinely improves or regresses versus the exact same notes with
the no-future acoustic model and per-pitch duration-median baselines.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import (
    get_gt,match_predicted_to_reference)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
SR=44100
def require(x,s):
    if not x:raise ValueError(s)
def duration_class(ms):
    if ms<100:return '<100 ms'
    if ms<250:return '100–249 ms'
    if ms<500:return '250–499 ms'
    if ms<1000:return '500–999 ms'
    if ms<2000:return '1000–1999 ms'
    return '>=2000 ms'
def result_set(items):
    n=len(items)
    if not n:return dict(n=0)
    cp=sum(int(i['pitch_median']) for i in items)
    cn=sum(int(i['no_future']) for i in items)
    cs=sum(int(i['S408']) for i in items)
    return dict(
        n=n,
        S408_correct=cs, S408_acc=cs/n,
        pitch_median_correct=cp, pitch_median_acc=cp/n,
        no_future_correct=cn,no_future_acc=cn/n,
        corrections_vs_no_future=sum(1 for i in items if i['S408'] and not i['no_future']),
        regressions_vs_no_future=sum(1 for i in items if i['no_future'] and not i['S408']),
        net_vs_no_future=cs-cn,
        S408_median_absolute_offset_error_ms=float(np.median([i['abs_ms'] for i in items])),
        S408_mean_absolute_offset_error_ms=float(np.mean([i['abs_ms'] for i in items])),
        S408_early_error_count=sum(1 for i in items if i['signed_ms']< -50),
        S408_late_error_count=sum(1 for i in items if i['signed_ms']>50))
def run(source,annotations,out):
    require(not out.exists(),'S409 immutable analysis cannot overwrite')
    with np.load(source,allow_pickle=False) as z:
        need={'source_row_id','track_member','S80_K','S402_midi_pitch','S407_onset_samples',
              'S408_offset_samples','no_future_offset_samples','median_pitch_offset_samples',
              'median_global_offset_samples'}
        require(need.issubset(z.files),'S408 offset NPZ source contract unavailable')
        vals={k:z[k].copy() for k in need}
    n=len(vals['track_member'])
    require(n==33475 and len(np.unique(vals['source_row_id']))>15000,
            'S408 source 33475 predictions or row ID drift')
    members=vals['track_member'].astype(str)
    onsets=vals['S407_onset_samples'].astype(np.int64)
    pitches=vals['S402_midi_pitch'].astype(int)
    offsets=vals['S408_offset_samples'].astype(np.int64)
    blind=vals['no_future_offset_samples'].astype(np.int64)
    median=vals['median_pitch_offset_samples'].astype(np.int64)
    require(np.all(offsets>onsets),'S408 offsets must follow predicted note onsets')
    gt={}
    with zipfile.ZipFile(annotations,'r') as f:
        names=collections.defaultdict(list)
        for name in f.namelist():
            if name.endswith('.jams'):names[PurePosixPath(name).name].append(name)
        for name in np.unique(members):
            item=names[PurePosixPath(name).name]
            require(len(item)==1,'original JAMS membership ambiguous '+name)
            gt[name]=get_gt(f.read(item[0]))
    require(len(gt)==190 and sum(map(len,gt.values()))==34954,
            'original GT track or note count mismatch')
    checks=[]
    samples=[]
    for track,reference in gt.items():
        candidates=np.flatnonzero(members==track)
        pred=[(int(onsets[i]),int(pitches[i]),int(i)) for i in candidates]
        pairs=match_predicted_to_reference(pred,reference,2205)
        for position,gtindex in pairs:
            global_index=pred[position][2]
            gtstart,gtend,gtpitch,string_slot=reference[gtindex]
            e=int(offsets[global_index])-gtend
            b=int(blind[global_index])-gtend
            p=int(median[global_index])-gtend
            checks.append(dict(K=int(vals['S80_K'][global_index]),
                 midi=int(pitches[global_index]),
                 octave=int(pitches[global_index]//12-1),
                 physical_string=int(string_slot),
                 duration_category=duration_class((gtend-gtstart)/SR*1000),
                 true_duration_ms=(gtend-gtstart)/SR*1000,
                 signed_ms=e/SR*1000,abs_ms=abs(e)/SR*1000,
                 S408=abs(e)<=2205,no_future=abs(b)<=2205,
                 pitch_median=abs(p)<=2205,
                 S408_rel=abs(e)<=max(2205,int(.2*(gtend-gtstart))),
                 no_future_rel=abs(b)<=max(2205,int(.2*(gtend-gtstart)))))
            if abs(e)>0.5*SR and len(samples)<35:
                samples.append(dict(track=track,MIDI=int(pitches[global_index]),
                    K=int(vals['S80_K'][global_index]),
                    gt_duration_ms=round((gtend-gtstart)/SR*1000,1),
                    predicted_duration_ms=round((int(offsets[global_index])-int(onsets[global_index]))/SR*1000,1),
                    error_ms=round(e/SR*1000,1)))
    require(len(checks)==25120,'S408 matched pitch+onset cardinality drift')
    scoped={
      'all_matched_note_onset_and_pitch':result_set(checks),
      'by_predicted_K':{f'K{k}':result_set([v for v in checks if v['K']==k])
                      for k in range(1,7)},
      'by_original_duration':{name:result_set([v for v in checks if v['duration_category']==name])
                      for name in ('<100 ms','100–249 ms','250–499 ms','500–999 ms','1000–1999 ms','>=2000 ms')},
      'by_MIDI_octave':{str(x):result_set([v for v in checks if v['octave']==x])
                       for x in range(1,8)},
      'by_physical_string_truth_only':{str(x):result_set([v for v in checks if v['physical_string']==x])
                       for x in range(6)}
    }
    overall=scoped['all_matched_note_onset_and_pitch']
    require(overall['S408_correct']==13050 and overall['no_future_correct']==11124
            and overall['pitch_median_correct']==7806,
            'Original S408 results not reproduced')
    report=dict(status='S409_S408_OFFSET_REGRESSION_CAUSE_AUDIT_FROZEN',
      original_predictions=n,original_GT_notes=34954,matched_pitch_onset=25120,
      no_model_fit_no_threshold_change=True,original_S408_results_reproduced=True,
      onset_tolerance_ms=50,offset_tolerance_ms=50,
      groups=scoped,examples_large_signed_errors=samples,
      inputs_sha256=dict(source=sha(source),labels=sha(annotations)),
      prior_song_disjoint_head_validation_not_fully_independent_system=True,
      promoted=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S409 ALL',json.dumps(overall),flush=True)
    print('S409 BY_K',json.dumps(scoped['by_predicted_K']),flush=True)
    print('S409 BY_DURATION',json.dumps(scoped['by_original_duration']),flush=True)
    print('S409 BY_OCTAVE',json.dumps(scoped['by_MIDI_octave']),flush=True)
    print('S409 PASS no new inference and metrics exactly reproduce S408',flush=True)
def selftest():
    fake=[dict(K=1,midi=48,octave=3,physical_string=0,duration_category='<100 ms',
        signed_ms=-75,abs_ms=75,S408=False,no_future=True,pitch_median=False),
        dict(K=2,midi=60,octave=4,physical_string=2,duration_category='250–499 ms',
        signed_ms=0.,abs_ms=0.,S408=True,no_future=False,pitch_median=True)]
    s=result_set(fake)
    require(s['S408_correct']==1 and s['corrections_vs_no_future']==1
        and s['regressions_vs_no_future']==1 and s['net_vs_no_future']==0,
        'frozen offset paired correction/regression audit wrong')
    print('S409 SELFTEST PASS offset regression account and duration strata')
if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--self-test',action='store_true')
    for n in ('source','annotations','out'):q.add_argument('--'+n,type=Path)
    a=q.parse_args()
    if a.self_test:selftest()
    else:
        for n in ('source','annotations','out'):
            require(getattr(a,n) is not None,'Missing '+n)
        run(a.source,a.annotations,a.out)
