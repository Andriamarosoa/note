"""S137: sealed audio predictions then external GOAT MIDI smoke. New S135, not historical S80.
No tuning or retraining. True K is assigned from note-on within 20ms to a
single nearest native interval (tie: earliest interval). This is only
two independent GOAT excerpts; reference promotion is prohibited.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
SR=44100
RAD=882
MIDI_HASH={'example1.mid':'9a7551c0aa24bf512d98f505c0c59832617904d3b6117ec4cd1d73729e7cb7e5',
           'example2.mid':'a795c561041ec8093c75cd4ca501336c0aefbc93aba6c20593f85edf5da01ab8'}
BAD={'truth','true_K','target','label','labels','midi','y','k'}
def require(c,m):
    if not c:raise ValueError(m)
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,keys):
    with np.load(p,allow_pickle=False) as z:
        require(set(z.files)==set(keys) and not BAD.intersection(z.files),'strict label-free NPZ schema')
        return {k:z[k].copy() for k in z.files}
def seal(s97,s85,s135,out):
    require(not out.exists(),'immutable predictions')
    a=read(s97,('row_id','summary58','flow245','geometry3','sequence42x49',
                'morphology490','physics294','member','start_samples','end_samples'))
    b=read(s85,('member','start_samples','end_samples','incumbent_freeze_K',
                'pr16_policy_K','pr16_vote_count','pr16_policy22_K',
                'pr16_network_probabilities'))
    c=read(s135,('row_id','predicted_K','probabilities'))
    require(len(a['row_id'])==len(b['member'])==len(c['row_id'])==57,'57 aligned events')
    require(np.array_equal(a['row_id'],c['row_id']) and
            all(np.array_equal(a[k],b[k]) for k in ('member','start_samples','end_samples')),
            'S97/S85/S135 native timing alignment')
    member=a['member'].astype(str)
    starts=a['start_samples'].astype(np.int64);ends=a['end_samples'].astype(np.int64)
    require(set(member)=={'example1_di.wav','example2_di.wav'},'GOAT source allowlist')
    require(np.all(starts>0) and np.all(ends>=starts) and
            np.all(ends-starts<=1764),'original 40ms group contract')
    require(np.array_equal(a['row_id'],np.asarray([f'{m}:{int(s)}' for m,s in zip(member,starts)])),
            'S97 row ID construction changed')
    parent=b['incumbent_freeze_K'].astype(np.int8)
    proposed=c['predicted_K'].astype(np.int8)
    require(np.isin(parent,range(7)).all() and np.isin(proposed,range(7)).all(),
            'valid K0..K6')
    require(c['probabilities'].shape==(57,7) and
            np.array_equal(c['probabilities'].argmax(1),proposed),'S135 probability/K integrity')
    out.mkdir(parents=True)
    dest=out/'predictions_before_labels.npz'
    np.savez_compressed(dest,row_id=a['row_id'],member=member,starts=starts,
                        ends=ends,incumbent_K=parent,candidate_K=proposed)
    report={'status':'SEALED_S135_NEW_STUDENT_VS_OFFICIAL_FREEZE',
            'events':57,'candidate_is_historical_S80':False,
            'candidate_is_new_student':True,'original_S80_Exact_K_measured':False,
            'source_s97_sha256':digest(s97),'source_s85_sha256':digest(s85),
            'source_s135_sha256':digest(s135),'prediction_sha256':digest(dest),
            'candidate_changes':int((parent!=proposed).sum()),
            'reference_K_hist':np.bincount(parent,minlength=7).tolist(),
            'candidate_K_hist':np.bincount(proposed,minlength=7).tolist(),
            'no_midi_read':True,'true_K_scored':False,'promotion':False}
    (out/'seal.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
def onsets(p):
    import mido
    midi=mido.MidiFile(str(p))
    seconds=0.; notes=[]
    # mido.MidiFile iteration merges tracks and supplies delta seconds.
    for msg in midi:
        seconds+=float(msg.time)
        if msg.type=='note_on' and msg.velocity>0:
            notes.append(int(np.rint(seconds*SR)))
    require(len(notes)>0 and min(notes)>=0,'expected absolute MIDI note-on stream')
    return np.asarray(notes,np.int64)
def nearest(note,ids,starts,ends):
    dist=np.maximum.reduce((starts[ids]-note,note-ends[ids],
                            np.zeros(len(ids),np.int64)))
    valid=np.flatnonzero(dist<=RAD)
    if not len(valid):return None
    return int(ids[min(valid,key=lambda j:(int(dist[j]),int(starts[ids[j]]),int(ids[j])))])
def score(predictions,manifest,midi_root,out):
    require(not out.exists(),'immutable scored output')
    info=json.loads(manifest.read_text())
    require(info['status']=='SEALED_S135_NEW_STUDENT_VS_OFFICIAL_FREEZE' and
            info['no_midi_read'] and not info['true_K_scored'],'label-free seal')
    require(digest(predictions)==info['prediction_sha256'],'SEALED SHA256 changed')
    with np.load(predictions,allow_pickle=False) as z:
        require(set(z.files)=={'row_id','member','starts','ends','incumbent_K','candidate_K'},
                'postlabel schema mismatch')
        rows={k:z[k].copy() for k in z.files}
    members=rows['member'].astype(str)
    starts=rows['starts'].astype(np.int64);ends=rows['ends'].astype(np.int64)
    parent=rows['incumbent_K'].astype(np.int8)
    proposed=rows['candidate_K'].astype(np.int8)
    require(len(members)==57,'smoke cohort changed')
    counts=np.zeros(57,np.int64); sources={}
    for i in (1,2):
        file=f'example{i}.mid';name=f'example{i}_di.wav'
        midi=midi_root/file
        require(digest(midi)==MIDI_HASH[file],'MIDI locked SHA changed')
        events=onsets(midi);idx=np.flatnonzero(members==name)
        assigned=0
        for note in events:
            loc=nearest(note,idx,starts,ends)
            if loc is not None:counts[loc]+=1;assigned+=1
        sources[name]={'total_note_on_in_MIDI':len(events),'assigned_note_on':assigned,
                       'unassigned_note_on':len(events)-assigned,'midi_sha256':digest(midi)}
    truth=np.minimum(counts,6).astype(np.int8)
    correct_b=parent==truth;correct_p=proposed==truth
    poly=truth>=2
    overall={'events':57,'incumbent_correct':int(correct_b.sum()),
             'candidate_correct':int(correct_p.sum()),
             'incumbent_accuracy':float(correct_b.mean()),
             'candidate_accuracy':float(correct_p.mean()),
             'corrections':int(np.sum(~correct_b&correct_p)),
             'regressions':int(np.sum(correct_b&~correct_p)),
             'net_gain':int(correct_p.sum()-correct_b.sum()),
             'changed_decisions':int(np.sum(parent!=proposed))}
    require(overall['net_gain']==overall['corrections']-overall['regressions'],
            'paired conservation error')
    byk={str(k):{'events':int((truth==k).sum()),
                 'incumbent_correct':int((correct_b&(truth==k)).sum()),
                 'candidate_correct':int((correct_p&(truth==k)).sum())} for k in range(7)}
    bytrack={name:{'events':int((members==name).sum()),
                   'incumbent_correct':int(np.sum(correct_b&(members==name))),
                   'candidate_correct':int(np.sum(correct_p&(members==name)))} for name in sorted(set(members))}
    result={'status':'S137_REAL_GOAT_2_FILE_MINI_SMOKE_NEW_STUDENT_NOT_S80',
            'proof_predicted_before_opening_MIDI_SHA256':digest(predictions),
            'true_K_assigned_one_note_on_to_nearest_interval_within_20ms':True,
            'two_recordings_not_promotion_grade':True,'source_midi':sources,
            'overall':overall,
            'poly_true_K2plus':{'events':int(poly.sum()),
                'incumbent_correct':int(np.sum(correct_b&poly)),
                'candidate_correct':int(np.sum(correct_p&poly))},
            'true_K_by_class':byk,'by_track':bytrack,
            'original_S80_tested':False,'official_reference_promoted':False}
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'revealed_true_K.npz',row_id=rows['row_id'],
                        true_K=truth,incumbent_K=parent,candidate_K=proposed)
    print(json.dumps(result,indent=2))
def selftest():
    starts=np.array([10000,12000,16000])
    ends=starts+200
    ids=np.arange(3)
    assert nearest(10100,ids,starts,ends)==0
    assert nearest(12400,ids,starts,ends)==1
    assert nearest(17082,ids,starts,ends)==2
    assert nearest(17083,ids,starts,ends) is None
    print('S137 PASS 20ms matching and no double assignment')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--self-test',action='store_true')
    sub=a.add_subparsers(dest='mode')
    b=sub.add_parser('seal')
    for key in ('s97','s85','s135','out'):b.add_argument('--'+key,type=Path,required=True)
    b=sub.add_parser('score')
    for key in ('predictions','manifest','midi-root','out'):
        b.add_argument('--'+key,type=Path,required=True)
    q=a.parse_args()
    if q.self_test:selftest()
    elif q.mode=='seal':seal(q.s97,q.s85,q.s135,q.out)
    elif q.mode=='score':score(q.predictions,q.manifest,q.midi_root,q.out)
    else:a.error('choose seal or score')
