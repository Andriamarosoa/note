"""Independent recount of S704/S705/S706 JSONL events; no model is unpickled."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.optimize import linear_sum_assignment

SR,HOP=44100,256
METRICS=('onset_f1','exact_k','active_exact_k','offset_accuracy','full_note_f1','frame_pitch_f1')


def recount(events, truth, samples):
    ends=np.arange(HOP,samples+1,HOP,dtype=np.int64)
    notes=[];active={};last_sample=-1;maximum=0
    for e in events:
        available=int(e['available_sample'])
        if (available<int(e['decision_sample']) or available<last_sample
                or abs(e['time']-available/SR)>1e-10):
            raise AssertionError('Emission order/time availability violation')
        last_sample=available
        p=int(e['pitch'])
        if e['type']=='note_on':
            if p in active:raise AssertionError('Duplicate open pitch')
            active[p]=len(notes);notes.append([p,float(e['time']),None,None])
            maximum=max(maximum,len(active))
        elif p in active:
            n=notes[active.pop(p)];n[2]=float(e['time']);n[3]=e['reason']
    if active:raise AssertionError('Incomplete serialized stream')
    pairs=[]
    for pitch in range(40,89):
        pi=[i for i,n in enumerate(notes) if n[0]==pitch]
        ti=[i for i,n in enumerate(truth) if n['pitch']==pitch]
        if not pi or not ti:continue
        dist=np.abs(np.array([notes[i][1] for i in pi])[:,None]-np.array([truth[j]['time'] for j in ti])[None,:])
        a,b=linear_sum_assignment(np.where(dist<=.080,dist,1e6))
        pairs.extend((pi[i],ti[j]) for i,j in zip(a,b) if dist[i,j]<=.080)
    valid=correct=long_correct=0;delays=[]
    for i,j in pairs:
        p,start,stop,reason=notes[i];target=truth[j]
        expected=target['time']+target['duration']
        delays.append((start-target['time'])*1000)
        if expected>=ends[-1]/SR:continue
        valid+=1
        good=reason not in ('stream_end','shutdown','stream_error') and abs(stop-expected)<=.050
        correct+=good;long_correct+=good and target['duration']>=1.
    diff_pred=np.zeros((len(ends)+1,49),np.int32)
    diff_gt=np.zeros_like(diff_pred)
    for pitch,start,stop,_ in notes:
        a,b=np.searchsorted(ends/SR,[start,stop],side='left')
        diff_pred[a,pitch-40]+=1;diff_pred[b,pitch-40]-=1
    for n in truth:
        a,b=np.searchsorted(ends/SR,[n['time'],n['time']+n['duration']],side='left')
        diff_gt[a,n['pitch']-40]+=1;diff_gt[b,n['pitch']-40]-=1
    pred=np.cumsum(diff_pred[:-1],axis=0);gt=np.cumsum(diff_gt[:-1],axis=0)
    pc=np.minimum(pred.sum(1),6);gc=np.minimum(gt.sum(1),6)
    pb,gb=pred>0,gt>0
    return dict(tp=len(pairs),fp=len(notes)-len(pairs),fn=len(truth)-len(pairs),
        k_correct=int((pc==gc).sum()),frames=len(ends),
        active_k_correct=int(((pc==gc)&(gc>0)).sum()),active_frames=int((gc>0).sum()),
        offset_correct=int(correct),offset_eligible=valid,long_offset_correct=int(long_correct),
        frame_pitch_tp=int((pb&gb).sum()),frame_pitch_fp=int((pb&~gb).sum()),
        frame_pitch_fn=int((~pb&gb).sum()),max_active=maximum,latencies=delays)


def total(rows):
    sums={k:sum(r[k] for r in rows) for k in rows[0] if k not in ('latencies','max_active')}
    den=2*sums['tp']+sums['fp']+sums['fn']
    sums.update(onset_f1=2*sums['tp']/max(1,den),exact_k=sums['k_correct']/max(1,sums['frames']),
        active_exact_k=sums['active_k_correct']/max(1,sums['active_frames']),
        offset_accuracy=sums['offset_correct']/max(1,sums['offset_eligible']),
        full_note_f1=2*sums['offset_correct']/max(1,den),
        frame_pitch_f1=2*sums['frame_pitch_tp']/max(1,2*sums['frame_pitch_tp']+sums['frame_pitch_fp']+sums['frame_pitch_fn']))
    sums['max_active']=max(r['max_active'] for r in rows)
    sums['all_targets_above_80']=all(sums[m]>.8 for m in METRICS)
    return sums


def verify(path):
    with zipfile.ZipFile(path) as z:
        def member(name):
            choices=[n for n in z.namelist() if n==name or n.endswith('/'+name)]
            if len(choices)!=1:raise ValueError('Missing or ambiguous artifact member '+name)
            return choices[0]
        report=json.loads(z.read(member('report.json')))
        verified=dict(artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),schema=report['schema'],
            checkpoint_sha256=report['checkpoint_sha256'],frozen_candidate=report['frozen_candidate'],
            loops_completed=len(report['rounds']),policy_evaluations=sum(len(r['scores']) for r in report['rounds']),
            TEST_opened=report['TEST_opened'],promoted=report['promoted'])
        for phase in ('HELD','DEV'):
            rows=[json.loads(line) for line in z.read(member(phase.lower()+'_events.jsonl')).splitlines()]
            if len({r['name'] for r in rows})!=len(rows):raise AssertionError('Repeated source name')
            by_variant=defaultdict(list)
            by_texture=defaultdict(lambda:defaultdict(list))
            for row in rows:
                for name,events in row['events'].items():
                    counts=recount(events,row['truth'],row['samples'])
                    by_variant[name].append(counts)
                    by_texture[row['name'].rsplit('_',1)[1]][name].append(counts)
            verified[phase]={name:total(v) for name,v in by_variant.items()}
            for name,stats in verified[phase].items():
                expected=report[phase][name]
                for k,v in stats.items():
                    if k not in expected:continue
                    if isinstance(v,float):
                        assert abs(v-expected[k])<1e-12,(phase,name,k,v,expected[k])
                    else:
                        assert v==expected[k],(phase,name,k,v,expected[k])
            verified[phase]['sources']=len(rows)
            verified[phase]['by_texture']={t:{n:total(v) for n,v in items.items()} for t,items in by_texture.items()}
        verified['all_targets_above_80']=all(verified[p]['candidate']['all_targets_above_80'] for p in ('HELD','DEV'))
        assert verified['all_targets_above_80']==report['all_targets_above_80']
        verified['all_raw_event_counts_match_report']=True
        return verified


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=verify(args.archive)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({p:{n:{m:r[m] for m in METRICS} for n,r in result[p].items()
        if n in ('native','native_s675_causal','candidate')} for p in ('HELD','DEV')},indent=2))
