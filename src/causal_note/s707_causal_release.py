"""Causal S675 release replay for new onsets, with a six-note emitted budget.

Research evaluator: first crossing and known forward timers only. No decision
uses the future absence of a crossing to backdate an old native release.
"""
import numpy as np
from causal_note.strict_live import HOP,SR
from scripts.s672_causal_offset_timing_grid import (
    note_instances,native_rhythm_clock,frame_matrix,MIN_AGE_FRAMES,MAX_AGE_FRAMES,
)


def received_history(sampled, n):
    """A block value is available starting at hop 3,7,...; never earlier."""
    out=np.zeros((n,*sampled.shape[1:]),np.float32)
    out[3:]=np.repeat(sampled,4,axis=0)[:max(0,n-3)]
    return out


def prepare_hazards(item, events, model):
    ends=item['ends'];n=len(ends)
    feat=received_history(item['native_features'],n)
    post=received_history(item['native_post'],n)
    k=received_history(item['native_k'],n)
    notes=note_instances(events,ends)
    clock=native_rhythm_clock(events,ends)
    rows=[];indices=[];frames_all=[]
    for i,note in enumerate(notes):
        lo=note['start']+MIN_AGE_FRAMES
        lo+=((3-lo)%4)
        hi=min(note['next_start'],len(ends),note['start']+MAX_AGE_FRAMES)
        frames=np.arange(lo,hi,4,dtype=np.int64)
        frames_all.append(frames)
        if len(frames):
            rows.append(frame_matrix(feat,post,k,ends,note['pitch'],note['start'],frames,clock))
            indices.append((i,len(frames)))
    probabilities=[np.empty(0,np.float32) for _ in notes]
    if rows:
        x=np.concatenate(rows)
        p=model.predict_proba(x)[:,1];at=0
        for i,length in indices:
            probabilities[i]=p[at:at+length].copy();at+=length
    return dict(notes=notes,frames=frames_all,probabilities=probabilities,
                onsets=[e for e in events if e['type']=='note_on'])


def render_release(plan, ends, samples, *, gate=.60, persistence=1, grace=24):
    planned=[]
    for i,n in enumerate(plan['notes']):
        on=dict(plan['onsets'][i]);planned.append((on,i))
        choices=[(len(ends),'stream_end'),(n['start']+MAX_AGE_FRAMES,'s707_max_age')]
        if n['next_start']<len(ends):choices.append((n['next_start'],'s707_restrike'))
        if n['native_end'] is not None:
            choices.append((n['native_end']+grace,'s707_forward_fallback'))
        hit=plan['probabilities'][i]>=gate
        if len(hit):
            count=np.convolve(hit.astype(int),np.ones(persistence,int),'full')[:len(hit)]
            first=np.flatnonzero(count>=persistence)
            if len(first):choices.append((int(plan['frames'][i][first[0]]),'s707_causal_hazard'))
        frame,reason=min(choices,key=lambda z:z[0])
        decision=int(ends[frame]) if frame<len(ends) else samples
        available=min(samples,((decision+4*HOP-1)//(4*HOP))*4*HOP)
        off=dict(type='note_off',pitch=n['pitch'],decision_sample=decision,
                 available_sample=available,time=available/SR,reason=reason,velocity=0)
        planned.append((off,i))
    planned.sort(key=lambda row:(row[0]['available_sample'],row[0]['decision_sample'],
        0 if row[0]['type']=='note_off' else 1,row[1]))
    active={};events=[];skipped=0
    for e,i in planned:
        pitch=e['pitch']
        if e['type']=='note_on':
            if pitch in active:
                raise AssertionError('Planned restrike must close the previous note first')
            if len(active)>=6:
                skipped+=1;continue
            active[pitch]=i;events.append(e)
        elif active.get(pitch)==i:
            del active[pitch];events.append(e)
    if active:raise AssertionError('Release plan left notes open')
    return events,skipped
