"""Post-decision S748 six-axis + per-K diagnostics, never inference inputs.

Audits only already emitted MIDI and independently read annotations. This
module must NEVER be imported from a live source selector or fitted model.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from causal_note.strict_live import SR
from scripts.s613_train_strict_live import event_score
from scripts.s704_joint_metrics import metric

REGISTERS = (("low_E2_E3",40,52),("middle_F3_A4",53,69),
             ("high_Asharp4_E6",70,88))


def _f1(tp,fp,fn):
    return float(2*tp/max(1,2*tp+fp+fn))


def axis_audit(events,truth,ends):
    """Track K0..K6 confusion, onset strata, offsets, pitch and near harmonics.

    Only honest time-available events; no external offset oracle, teacher
    K, future label input or postprocessing edits to the candidate stream.
    """
    ends=np.asarray(ends,np.int64)
    if ends.ndim!=1 or len(ends)==0 or np.any(np.diff(ends)<=0):
        raise ValueError("S748 invalid decision timestamps")
    source=metric(events,truth,ends)  # validates emitted event causality
    pred=np.zeros((len(ends),49),bool)
    gt=np.zeros_like(pred)
    active={}
    periods=[]
    for e in events:
        pitch=int(e["pitch"])-40
        if not 0<=pitch<49:
            raise ValueError("S748 pitch outside MIDI40..88")
        frame=int(np.searchsorted(ends,int(e["available_sample"]),side="left"))
        if e["type"]=="note_on":
            if pitch in active:
                raise ValueError("S748 doubled same pitch without NoteOff")
            active[pitch]=(frame,e)
        elif e["type"]=="note_off":
            if pitch not in active:
                raise ValueError("S748 missing corresponding NoteOn")
            start,on=active.pop(pitch)
            pred[start:frame,pitch]=True
            periods.append((pitch+40,on,e))
        else:
            raise ValueError("S748 unknown MIDI event type")
    if active:
        raise ValueError("S748 stream ended without matched NoteOff")
    times=ends/SR
    note_duration=Counter()
    true_count=np.zeros(len(ends),dtype=np.int32)
    for n in truth:
        pitch=int(n["pitch"])-40
        if not 0<=pitch<49:
            raise ValueError("S748 ground truth pitch outside MIDI40..88")
        valid=(times>=float(n["time"]))&(
            times<float(n["time"])+float(n["duration"]))
        gt[valid,pitch]=True
        true_count+=valid.astype(np.int32)
        if float(n["duration"])>=1:
            note_duration["long"]+=1
        else:
            note_duration["short"]+=1
    # Preserve the signed S704 K definition, including coincident
    # same-pitch truth instances even though frame-pitch is binary.
    pred_count=pred.sum(1).astype(np.int32)
    confused=np.zeros((7,7),dtype=np.int64)
    for actual,proposed in zip(np.minimum(true_count,6),
                               np.minimum(pred_count,6)):
        confused[actual,proposed]+=1
    per_k={}
    for k in range(7):
        ix=true_count==k if k<6 else true_count>=6
        if not np.any(ix):
            continue
        # Match signed S704's K0..6 clipping, even for ground-truth
        # K>6. The unsaturated true K remains exposed separately via
        # k_gt6_frames; otherwise per-K percentages contradict Exact-K.
        diff=np.minimum(pred_count[ix],6)-np.minimum(true_count[ix],6)
        per_k[str(k)]=dict(
            true_frames=int(ix.sum()),
            exact=int(np.count_nonzero(diff==0)),
            under=int(np.count_nonzero(diff<0)),
            over=int(np.count_nonzero(diff>0)),
            exact_rate=float(np.mean(diff==0)))
    _, matched=event_score(events,truth)
    match_idx={i for i,_ in matched}
    matched_gt={j for _,j in matched}
    onsets=[e for e in events if e["type"]=="note_on"]
    offsets_by_reason=Counter(
        e["reason"] for e in events if e["type"]=="note_off")
    strata={}
    for key,lo,hi in REGISTERS:
        mask=np.zeros(49,bool);mask[lo-40:hi-39]=True
        tps=int(np.count_nonzero(pred[:,mask]&gt[:,mask]))
        fps=int(np.count_nonzero(pred[:,mask]&~gt[:,mask]))
        fns=int(np.count_nonzero(~pred[:,mask]&gt[:,mask]))
        predicted_ids=[i for i,e in enumerate(onsets)
                       if lo<=int(e["pitch"])<=hi]
        actual_ids=[j for j,n in enumerate(truth)
                    if lo<=int(n["pitch"])<=hi]
        ons_tp=sum(i in match_idx for i in predicted_ids)
        ons_fp=len(predicted_ids)-ons_tp
        ons_fn=len(actual_ids)-sum(j in matched_gt for j in actual_ids)
        strata[key]=dict(onset_f1=_f1(ons_tp,ons_fp,ons_fn),
                         frame_pitch_f1=_f1(tps,fps,fns),
                         onset_tp=ons_tp,onset_fp=ons_fp,onset_fn=ons_fn,
                         frame_pitch_tp=tps,frame_pitch_fp=fps,frame_pitch_fn=fns)
    near_harmonics={}
    near_harmonics_directional={}
    miss=gt&~pred
    false_positive=pred&~gt
    for offset in (12,19,24):
        # Both relative directions: true E2 falsely detected as E3
        # (+12), AND true E3 falsely detected as E2 (-12).
        # The S748 prototype only counted the second case.
        false_above=np.zeros_like(pred)
        false_below=np.zeros_like(pred)
        false_above[:,:49-offset]=false_positive[:,offset:]
        false_below[:,offset:]=false_positive[:,:49-offset]
        above=int(np.count_nonzero(miss&false_above))
        below=int(np.count_nonzero(miss&false_below))
        near_harmonics_directional[str(offset)]=dict(
            false_above=above,false_below=below)
        # OR per missed truth pitch/frame to avoid double counting if
        # BOTH lower and higher false alarms are present.
        near_harmonics[str(offset)]=int(np.count_nonzero(
            miss&(false_above|false_below)))
    return dict(
        source_metrics={k:float(source[k]) for k in
          ("onset_f1","exact_k","active_exact_k","offset_accuracy",
           "full_note_f1","frame_pitch_f1")},
        confusion_k0_k6=confused.tolist(),
        per_true_k=per_k,
        k_gt6_frames=int(np.count_nonzero(true_count>6)),
        register=strata,
        harmonic_false_alarms_near_miss=near_harmonics,
        harmonic_false_alarms_by_direction=near_harmonics_directional,
        note_durations=dict(note_duration),
        offset_reasons=dict(offsets_by_reason),
        max_predicted_voices=int(pred_count.max()),
        max_truth_voices=int(true_count.max()),
        overcount_active_frames=int(np.count_nonzero(
            (true_count>0)&(pred_count>true_count))),
        undercount_active_frames=int(np.count_nonzero(
            (true_count>0)&(pred_count<true_count))),
        missing_first_onsets=int(len(truth)-len(matched_gt)))


def paired_audit(reference,candidate,truth,ends):
    """Side-by-side six metrics without changing any event input."""
    original=axis_audit(reference,truth,ends)
    proposed=axis_audit(candidate,truth,ends)
    names=original["source_metrics"]
    return dict(baseline=original,trial=proposed,
                delta={key:proposed["source_metrics"][key]-names[key]
                       for key in names},
                strict_six_nonregression=all(
                    proposed["source_metrics"][key]>=names[key]-1e-12
                    for key in names))
