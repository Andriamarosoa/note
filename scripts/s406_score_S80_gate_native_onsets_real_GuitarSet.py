"""S406 — first REAL audit: use frozen S80 cardinality to suppress spurious
raw onset candidates, without touching onset/offset model weights.

GuitarSet 59,309 original V88 acoustic event groups and 190 JAMS tracks.
PREDICTIONS and times are completely determined BEFORE reading labels.
For each candidate interval [start,end] (width <=40ms), propose an event
at its original START sample. Compare (i) all native events K=1 and (ii)
S80 predicted K copies at the SAME event start. ALSO report midpoint and
end as transparent timing diagnostics (not preselected using labels).
One-to-one chronological matching, tolerances ±20ms and ±50ms, no pitch
identity required in this audit.

All GuitarSet results are DEVELOPMENT ONLY, because S80 S397 was selected
with GuitarSet test exposure and partially trained on these songs.
These metrics are NOT directly comparable to earlier V7 60-song split.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path,PurePosixPath
import zipfile
import numpy as np
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import read_note_events
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

N=59309
GT_TOTAL_EXPECTED=34954
S80_CORRECT_EXPECTED=53529
K2_TRUE=7385

def req(ok,msg):
    if not ok:raise ValueError(msg)
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def onset_match(predicted,ground_truth,radius):
    """Order-preserving one-to-one match of separate onset instances.

    For one-dimensional nearest-within-radius intervals, optimal maximum
    cardinality matching admits monotonic order. We use two pointers.
    No GT signal is passed back to a predictor.
    """
    p=np.sort(np.asarray(predicted,np.int64))
    t=np.sort(np.asarray(ground_truth,np.int64))
    a=b=matches=0
    signed=[]
    while a<len(p) and b<len(t):
        diff=int(p[a]-t[b])
        if abs(diff)<=radius:
            matches+=1;signed.append(diff)
            a+=1;b+=1
        elif diff < -radius:
            a+=1
        else:b+=1
    denom=len(p)+len(t)
    return dict(pred=int(len(p)),ref=int(len(t)),tp=int(matches),
       fp=int(len(p)-matches),fn=int(len(t)-matches),
       precision=float(matches/len(p)) if len(p) else (1. if not len(t) else 0.),
       recall=float(matches/len(t)) if len(t) else 1.,
       f1=float(2*matches/denom) if denom else 1.,
       median_signed_ms=float(np.median(signed)*1000/44100) if signed else None,
       median_abs_ms=float(np.median(np.abs(signed))*1000/44100) if signed else None,
       p90_abs_ms=float(np.percentile(np.abs(signed),90)*1000/44100) if signed else None)
def aggregate(per_track):
    out={}
    for key,scores in per_track.items():
        ps=int(sum(z['pred'] for z in scores))
        gt=int(sum(z['ref'] for z in scores))
        tp=int(sum(z['tp'] for z in scores))
        out[key]=dict(pred=ps,ref=gt,tp=tp,fp=ps-tp,fn=gt-tp,
           precision=tp/ps if ps else None,
           recall=tp/gt if gt else None,
           F1=2*tp/(ps+gt) if ps+gt else None)
    return out

def calc_pred(starts,ends,K,method):
    if method=='start':ts=starts
    elif method=='mid':ts=starts+(ends-starts)//2
    elif method=='end':ts=ends
    else:raise ValueError('unknown event timestamp')
    return np.repeat(ts.astype(np.int64),K.astype(np.int64))

def run(audio,ref,jams,truth,out):
    req(not out.exists(),'immutable S406 evidence only')
    a=npz(audio);r=npz(ref);t=npz(truth)
    row=a['row_id'].astype(str)
    req(len(row)==N and np.array_equal(row,r['row_id'].astype(str))
       and np.array_equal(row,t['row_id'].astype(str)),
       'original 59309 audio/S80/labels row identity mismatch')
    K=r['s80_K'].astype(np.int8)
    trueK=t['true_K'].astype(np.int8)
    req(int(np.count_nonzero(K==trueK))==S80_CORRECT_EXPECTED
      and int(np.count_nonzero(trueK>=2))==K2_TRUE
      and np.isin(K,range(7)).all(),'S397 guarded S80 reference drift')
    member=a['member'].astype(str)
    start=a['start_samples'].astype(np.int64)
    end=a['end_samples'].astype(np.int64)
    req(np.all(end>=start) and np.all(end-start<=1764),
        'native event width not within original 40ms')
    # PRECOMMIT prediction timestamps, all policies, before opening JAMS.
    prediction={}
    for m in ('start','mid','end'):
        prediction[m]={}
        for track in np.unique(member):
            ix=np.flatnonzero(member==track)
            order=ix[np.argsort(start[ix],kind='stable')]
            prediction[m][track]=dict(
                raw=calc_pred(start[order],end[order],
                     np.ones(len(order),np.int8),m),
                S80=calc_pred(start[order],end[order],K[order],m),
                truthK_as_candidate_upper_bound=calc_pred(
                     start[order],end[order],trueK[order],m))
    req(set(prediction)=={'start','mid','end'} and
       sum(len(v['S80']) for v in prediction['start'].values())==int(K.sum()),
       'prediction was not completely frozen before labels opened')

    with zipfile.ZipFile(jams) as archive:
        names=collections.defaultdict(list)
        for n in archive.namelist():
            if n.endswith('.jams'):
                names[PurePosixPath(n).name].append(n)
        labels={}
        for track in sorted(set(member)):
            candidates=names.get(PurePosixPath(track).name,[])
            req(len(candidates)==1,'missing/ambiguous unique original note JAMS '+track)
            labels[track]=np.asarray([x[0] for x in
                           read_note_events(archive.read(candidates[0]))],np.int64)
    req(len(labels)==len(set(member)) and
         sum(len(a) for a in labels.values())==GT_TOTAL_EXPECTED,
         'JAMS total note count or track count not original S400')

    metrics={}
    per_track={}
    for time_mode in prediction:
        for radius_ms,radius_sample in ((20,882),(50,2205)):
            scores=collections.defaultdict(list)
            for track,policies in prediction[time_mode].items():
                for name,ts in policies.items():
                    scores[name].append(onset_match(ts,labels[track],radius_sample))
            name=f'{time_mode}_tolerance_{radius_ms}ms'
            per_track[name]=scores
            metrics[name]=aggregate(scores)
    # Never combine a GT condition with new S80 action until AFTER all preds
    # have been frozen. Audits K0 false candidate groups and catchable notes.
    group_counts=dict(
        all_native_event_groups=N,
        reference_S80_predicted_K0=int(np.count_nonzero(K==0)),
        reference_S80_predicted_Kpositive=int(np.count_nonzero(K>0)),
        actual_K0=int(np.count_nonzero(trueK==0)),
        actual_Kpositive=int(np.count_nonzero(trueK>0)),
        K0_true_suppressed=int(np.count_nonzero((K==0)&(trueK==0))),
        K0_missed_real_onsets=int(np.count_nonzero((K==0)&(trueK>0))),
        nonK0_false_trigger_groups=int(np.count_nonzero((K>0)&(trueK==0))),
        sum_true_K_assigned_to_native_intervals=int(trueK.sum()),
        sum_S80_predicted_K_attacks=int(K.sum()),
        original_annotation_onsets_all_tracks=GT_TOTAL_EXPECTED,
        original_JAMS_onsets_unassigned_or_outside_groups=int(GT_TOTAL_EXPECTED-trueK.sum())
    )
    # This report is strictly diagnostic. No GT information returns to K.
    obj=dict(status='S406_FROZEN_S80_COUNT_GATE_ON_REAL_ONSET_CANDIDATES',
      S80_reference='S80_S397_K5_GUARDED',
      original_GuitarSet_native_groups=N,
      unique_tracks=len(labels),
      predicted_before_annotation_opened=True,
      GT_note_pitches_NOT_used_in_prediction=True,
      offset_prediction_evaluated=False,
      historical_V7_onset_score_not_directly_comparable_different_tracks=True,
      timings_natively_precommitted=['start','mid','end'],
      headline_timing='start',
      tolerance_ms=[20,50],
      no_model_retrained=True,
      metrics=metrics,group_counts=group_counts,
      source_S80_SHA256=sha(ref),source_S258_audio_SHA256=sha(audio),
      source_JAMS_SHA256=sha(jams),
      development_with_prior_GuitarSet_exposure=True,
      independent_generalization_NOT_proven=True,
      S80_exactK_reference_modified=False,
      protocol_warning='No per-pitch note identity or acoustic NoteOff here. Compare only same 190 tracks and matching tolerance.')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(obj,indent=2)+'\n')
    print('S406 GROUP_COUNTS',json.dumps(group_counts),flush=True)
    print('S406 METRICS',json.dumps(metrics),flush=True)
    print('S406 PASS label-blind native event prediction one-to-one matched after freeze',flush=True)

def selftest():
    truth=np.asarray([100,100,200,310,400],np.int64)
    p=np.asarray([100,105,200,304,1000],np.int64)
    s=onset_match(p,truth,10)
    req(s['tp']==4 and s['fp']==1 and s['fn']==1,
        'S406 duplicates / unmatched samples miscounted')
    req(np.array_equal(calc_pred(np.array([100,230]),np.array([140,250]),
        np.array([0,2]),'mid'),[240,240]),
        'S80 K0 must suppress onset and K2 emit 2 attacks')
    req(onset_match([],[],882)['f1']==1.,
        'empty tracks must not error')
    print('S406 SELFTEST PASS sorted one-to-one no duplicate reuse; K0=>0 K2=>2')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for name in ('audio','s80','annotations','truth','out'):
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        for name in ('audio','s80','annotations','truth','out'):
            req(getattr(args,name) is not None,'missing '+name)
        run(args.audio,args.s80,args.annotations,args.truth,args.out)
