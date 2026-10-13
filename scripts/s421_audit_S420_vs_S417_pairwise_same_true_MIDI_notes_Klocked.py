"""S421 — independently compare FROZEN S417 vs S420 exported full MIDI
notes and audit exact S80 K0-K6 per 59,309 native event row IDs.

No training, no corrections, no teacher. Every ground-truth GuitarSet note
is independently marked complete only if a unique predicted note has
correct pitch, onset <=50ms and offset <=50ms. Corrections/regressions
refer to SAME ground-truth note IDs, unlike conditional accuracy where
model-to-truth matches might be different.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import (
    get_gt,match_predicted_to_reference)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
SR=44100
N=59309
PRED=33475
GT=34954
def req(b,msg):
    if not b:raise ValueError(msg)
def read_prediction(path):
    per_track={};counter=collections.Counter()
    for ln in path.read_text(encoding='utf8').splitlines():
        if not ln.strip():continue
        obj=json.loads(ln);m=str(obj['member'])
        req(m.endswith('.jams') and m not in per_track,'MIDI export duplicated track')
        notes=[]
        for item in obj['notes']:
            pitch=int(item.get('pitch',item.get('midi_pitch',-1)))
            start=int(item['onset_sample']);stop=int(item['offset_sample'])
            row=str(item['source_row_id'])
            req(40<=pitch<=88 and start>=0 and stop>start,
                'Exported full MIDI note malformed')
            notes.append((start,stop,pitch,row))
            counter[row]+=1
        per_track[m]=notes
    req(len(per_track)==190 and sum(counter.values())==PRED,
        'S421 source MIDI note count or original 190 tracks wrong')
    return per_track,counter
def load_gt(path,tracks):
    with zipfile.ZipFile(path) as z:
        options=collections.defaultdict(list)
        for fn in z.namelist():
            if fn.endswith('.jams'):options[PurePosixPath(fn).name].append(fn)
        ref={}
        for m in sorted(tracks):
            files=options[PurePosixPath(m).name]
            req(len(files)==1,'S421 original GuitarSet JAMS ambiguous '+m)
            ref[m]=get_gt(z.read(files[0]))
    req(len(ref)==190 and sum(map(len,ref.values()))==GT,
        'S421 ground truth note identity/size drift')
    return ref
def success_set(pred,ref):
    items=[(x[0],x[2],i) for i,x in enumerate(pred)]
    hits=set(); matched=0
    for pi,gi in match_predicted_to_reference(items,ref,2205):
        matched+=1
        off=int(pred[pi][1]);target=int(ref[gi][1])
        if abs(off-target)<=2205:hits.add(gi)
    return hits,matched
def duration_group(ms):
    return '<100ms' if ms<100 else '100-499ms' if ms<500 else (
        '500-999ms' if ms<1000 else '1000ms+')
def count_pair(rows):
    a=sum(bool(r[0]) for r in rows);b=sum(bool(r[1]) for r in rows)
    return dict(n=len(rows),S417_correct=a,S420_correct=b,
      correction=sum(not r[0] and r[1] for r in rows),
      regression=sum(r[0] and not r[1] for r in rows),
      net=b-a,
      S417_acc=a/len(rows) if rows else None,
      S420_acc=b/len(rows) if rows else None)
def run(old_events,new_events,s80,annotations,out):
    req(not out.exists(),'S421 report cannot be overwritten')
    old,countold=read_prediction(old_events)
    new,countnew=read_prediction(new_events)
    req(set(old)==set(new),'S421 compared different GuitarSet tracks')
    with np.load(s80,allow_pickle=False) as z:
        ids=z['row_id'].astype(str);K=z['s80_K'].astype(np.int8)
    req(len(ids)==N and len(set(ids))==N and K.shape==(N,)
        and np.isin(K,range(7)).all() and int(K.sum())==PRED,
        'Original S80 K reference changed')
    req(all(countold[r]==int(k) and countnew[r]==int(k)
          for r,k in zip(ids,K)),
        'CRITICAL S421 EXPORT MIDI event count differs from original S80 K')
    req(all(r in set(ids) for r in countold) and
        all(r in set(ids) for r in countnew),
        'S421 foreign note event row source')
    truth=load_gt(annotations,old)
    allpairs=[];bydur=collections.defaultdict(list)
    byfold=collections.defaultdict(list);byoctave=collections.defaultdict(list)
    bycomp=collections.defaultdict(list)
    matches={'S417':0,'S420':0}
    for m,gt in truth.items():
        a,ma=success_set(old[m],gt)
        b,mb=success_set(new[m],gt)
        matches['S417']+=ma;matches['S420']+=mb
        fold=composition_fold(composition_key(m))
        for i,(start,stop,pitch,slot) in enumerate(gt):
            pair=(i in a,i in b)
            allpairs.append(pair)
            ms=(stop-start)/SR*1000
            bydur[duration_group(ms)].append(pair)
            byfold[str(fold)].append(pair)
            byoctave[str(int(pitch)//12-1)].append(pair)
            bycomp[composition_key(m)].append(pair)
    counts=count_pair(allpairs)
    req(counts['n']==GT and counts['S417_correct']==16017
        and counts['S420_correct']==16140
        and matches['S417']==25125 and matches['S420']==25395,
        'Original S417 or S420 score does not reproduce exactly')
    grouped={
      'by_true_duration':{k:count_pair(v) for k,v in bydur.items()},
      'by_composition_fold':{k:count_pair(v) for k,v in byfold.items()},
      'by_midi_octave':{k:count_pair(v) for k,v in byoctave.items()},
      'by_composition':{k:count_pair(v) for k,v in bycomp.items()}
    }
    req(sum(v['n'] for v in grouped['by_true_duration'].values())==GT,
        'S421 group duration loss')
    report=dict(status='S421_FROZEN_NATIVE_COMPLETE_NOTE_PAIRED_REGRESSION_AUDIT',
       original_event_groups=N,original_S80_total_new_notes=PRED,
       all_row_K0_to_K6_preserved_in_BOTH_exported_JSONL=True,
       exactK_regressions=0,no_model_fit_or_change=True,
       GT_note_population=GT,
       actual_s417_and_s420_one_to_one_pitch_plus_onset_matches=matches,
       complete_note_f1=dict(S417=2*16017/(PRED+GT),
                             S420=2*16140/(PRED+GT)),
       comparison_on_exact_same_GuitarSet_reference_note_IDs=counts,
       groups=grouped,
       source_hashes=dict(S417=sha(old_events),S420=sha(new_events),
                          S80=sha(s80),JAMS=sha(annotations)),
       previously_GuitarSet_exposed_development_only=True,
       official_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S421 PAIRED',json.dumps(counts),flush=True)
    print('S421 BY DURATION',json.dumps(grouped['by_true_duration']),flush=True)
    print('S421 BY FOLD',json.dumps(grouped['by_composition_fold']),flush=True)
    print('S421 PASS zero K regression from BOTH real exported MIDI note JSONLs',flush=True)
def test():
    a=[(1,0),(0,1),(1,1),(0,0)]
    x=count_pair(a)
    req(x['correction']==1 and x['regression']==1
        and x['net']==0,'S421 paired correction/regression independent note check')
    print('S421 SELFTEST PASS pairwise corrections and regressions from same actual notes')
if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--self-test',action='store_true')
    for name in ('s417','s420','s80','annotations','out'):q.add_argument('--'+name,type=Path)
    p=q.parse_args()
    if p.self_test:test()
    else:
        for name in ('s417','s420','s80','annotations','out'):
            req(getattr(p,name)is not None,'Missing '+name)
        run(p.s417,p.s420,p.s80,p.annotations,p.out)
