"""S400 — Ground-truth MIDI note pitch audit for S399 guarded-S80 decoder.

Only this evaluation program reads GuitarSet JAMS note_midi labels. Ground
truth note onsets are assigned once to 40ms V88 native event intervals using
the existing 20ms nearest endpoint original scoring contract. Never use
labels to tune S399 after opening JAMS. Full-note Exact Chord is a MIDI
MULTISET metric, distinct from exact cardinality K. Reports oracle true-K
pitch ranking separately and always identifies label/coverage failures.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path,PurePosixPath
import zipfile
import numpy as np
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    MIDI_MIN,MIDI_MAX,choose_pitches,note_name,sha)
def need(x,m):
    if not x: raise ValueError(m)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def midi_value(v):
    if isinstance(v,dict):
        for key in ('midi','pitch','note','value'):
            if key in v:
                v=v[key];break
    need(not isinstance(v,bool) and isinstance(v,(int,float))
         and np.isfinite(float(v)) and 0<=float(v)<=127,
         'JAMS MIDI pitch must be finite numeric 0..127 (fractional tuning is valid)')
    # GuitarSet JAMS note_midi values MAY BE FRACTIONAL from tuning/bends.
    # Use nearest MIDI semitone, without inspecting test metrics.
    return int(np.floor(float(v)+0.5))
def read_note_events(raw):
    obj=json.loads(raw.decode('utf-8'))
    annotations=obj.get('annotations')
    need(isinstance(annotations,list),'JAMS annotations list absent')
    events=[]
    for ann in annotations:
        if ann.get('namespace')!='note_midi':continue
        data=ann.get('data')
        need(isinstance(data,list),'JAMS note_midi data list invalid')
        metadata=ann.get('annotation_metadata') or {}
        string_source=metadata.get('data_source',None)
        need(str(string_source) in ('0','1','2','3','4','5'),
             'JAMS six physical guitar string labels absent')
        for obs in data:
            val=float(obs['time'])
            need(np.isfinite(val) and val>=0,'bad JAMS note onsets')
            sample=int(round(44100*val))
            pitch=midi_value(obs['value'])
            events.append((sample,pitch,int(string_source)))
    return sorted(events)
def scoped(y,reference,pred,oracle,valid,proposed):
    idx=np.flatnonzero(valid)
    n=len(idx)
    if not n:return dict(n=0)
    tp=refn=detn=tp_oracle=exact=oracle_exact=correct_k=dup_truth=0
    for i in idx:
        gt=collections.Counter(int(p) for p in y[i] if p>=0)
        pp=collections.Counter(int(p) for p in pred[i] if p>=0)
        oo=collections.Counter(int(p) for p in oracle[i] if p>=0)
        tp+=sum((gt&pp).values());tp_oracle+=sum((gt&oo).values())
        refn+=sum(gt.values());detn+=sum(pp.values())
        exact+=int(gt==pp);oracle_exact+=int(gt==oo)
        correct_k+=int(reference[i]==proposed[i])
        dup_truth+=int(len(gt)!=sum(gt.values()))
    precision=tp/detn if detn else (1. if refn==0 else 0.)
    recall=tp/refn if refn else (1. if detn==0 else 0.)
    return dict(n=n,complete_pitch_multiset_exact=exact,
       complete_pitch_multiset_exact_rate=exact/n,
       trueK_oracle_rank_multiset_exact=oracle_exact,
       trueK_oracle_rank_multiset_exact_rate=oracle_exact/n,
       reference_ExactK_correct=correct_k,
       reference_ExactK=correct_k/n,
       midi_note_tp=tp,true_note_total=refn,predicted_note_total=detn,
       micro_pitch_precision=precision,micro_pitch_recall=recall,
       micro_pitch_F1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.,
       trueK_oracle_note_tp=tp_oracle,
       trueK_oracle_pitch_recall=(tp_oracle/refn if refn else 1.),
       duplicate_pitch_chords_unrepresentable=dup_truth)
def run(audio,source,source_report,original_truth,annotations,out):
    need(not out.exists(),'S400 labeled audit may not overwrite immutable outputs')
    m=json.loads(Path(source_report).read_text())
    need(m['status']=='S399_LABEL_BLIND_REAL_PITCH_SOURCE_S80_K5_GUARD_REFERENCE'
      and m['labels_read'] is False and m['predictor_uses_ground_truth'] is False
      and sha(source)==m['output_sha256'],
      'S399 note predictions must be label-blind and SHA-frozen')
    z=load(source)
    a=load(audio);t=load(original_truth)
    row=z['row_id'].astype(str)
    members=a['member'].astype(str)
    starts=a['start_samples'].astype(np.int64)
    ends=a['end_samples'].astype(np.int64)
    k=z['s80_K'].astype(int)
    yK=t['true_K'].astype(int)
    n=len(row)
    need(n==59309 and np.array_equal(row,a['row_id'].astype(str))
      and np.array_equal(row,t['row_id'].astype(str))
      and int((yK==5).sum())==355 and int((yK>=2).sum())==7385
      and z['midi_pitch_slots'].shape==(n,6)
      and z['note_candidate_scores'].shape==(n,49)
      and np.array_equal((z['midi_pitch_slots']>=0).sum(1),k)
      and np.array_equal(t['original_fold'].astype(int),np.sort(np.unique(t['original_fold'].astype(int)))[np.searchsorted(np.unique(t['original_fold']),t['original_fold'])]),
      'S399 event source or true-K contract mismatch')
    # Index only JAMS member names; strict basename uniqueness, never fuzzy
    # matching or guessing by song prefixes.
    with zipfile.ZipFile(annotations,'r') as arch:
        candidates=[item for item in arch.namelist()
                    if item.endswith('.jams') and not item.endswith('/')]
        by_name=collections.defaultdict(list)
        for item in candidates:by_name[PurePosixPath(item).name].append(item)
        absent=[]
        trackmap={}
        for member in sorted(set(members)):
            if member in candidates:real=member
            else:
                found=by_name.get(PurePosixPath(member).name,[])
                need(len(found)<=1, 'ambiguous annotation JAMS basename: '+member)
                if not found: absent.append(member);continue
                real=found[0]
            trackmap[member]=real
        need(not absent,'missing exact GuitarSet JAMS for '+repr(absent[:8]))
        true_pitch=np.full((n,6),-1,np.int16)
        note_counts=np.zeros(n,np.int16)
        out_of_vocab=np.zeros(n,bool)
        unassigned_total=0;events_total=0
        members_valid=0
        for member,record in trackmap.items():
            ix=np.flatnonzero(members==member)
            order=np.argsort(starts[ix],kind='stable')
            rows=ix[order];st=starts[rows];en=ends[rows]
            need(np.all(st[1:]>en[:-1]),'V88 native onset groups overlap '+member)
            ev=read_note_events(arch.read(record))
            events_total+=len(ev)
            note_onsets=np.asarray([e[0] for e in ev],np.int64)
            _,assignment=assign_onsets(note_onsets,st,en)
            by_row=collections.defaultdict(list)
            for (_,pitch,slot),j in zip(ev,assignment):
                if j<0:
                    unassigned_total+=1;continue
                by_row[int(rows[int(j)])].append((slot,pitch))
            for ix2,labeled in by_row.items():
                # Preserve duplicate pitch values across different strings.
                labeled.sort()
                note_counts[ix2]=len(labeled)
                if len(labeled)>6:continue
                for q,(_,pitch) in enumerate(labeled):
                    true_pitch[ix2,q]=pitch
                    if not MIDI_MIN<=pitch<=MIDI_MAX:out_of_vocab[ix2]=True
            members_valid+=1
    consistent=note_counts==yK
    invalid_more_than_six = note_counts > 6
    # Any count above six cannot be represented in 6 slots.
    valid=consistent & ~out_of_vocab & ~invalid_more_than_six
    target_counter=collections.Counter()
    for i in np.flatnonzero(~valid):
        reason=('K mismatch' if not consistent[i] else
                'pitch out-of-range' if out_of_vocab[i] else 'too many notes')
        target_counter[reason]+=1
    oracle=np.full((n,6),-1,np.int16)
    for i in range(0,n,1024):
        j=min(n,i+1024)
        oracle[i:j]=choose_pitches(z['note_candidate_scores'][i:j],yK[i:j])[0]
    pred=z['midi_pitch_slots']
    measurements={
       'all_eligible_native_events':scoped(true_pitch,yK,pred,oracle,valid,k),
       'positive_K1_to_K6':scoped(true_pitch,yK,pred,oracle,valid & (yK>=1),k),
       'poly_K2_to_K6':scoped(true_pitch,yK,pred,oracle,valid & (yK>=2),k),
       'by_true_K':{f'K{x}':scoped(true_pitch,yK,pred,oracle,valid&(yK==x),k)
                    for x in range(7)}
    }
    # Scoring samples expose labels ONLY here; no tuning allowed from the result
    errors=[]
    for i in np.flatnonzero(valid & (yK>=2)):
        if collections.Counter(int(p) for p in pred[i] if p>=0) == collections.Counter(int(p) for p in true_pitch[i] if p>=0):
            continue
        if len(errors)>=25:
            break
        errors.append(dict(row_id=str(row[i]),real_K=int(yK[i]),S80_K=int(k[i]),
         true_midi=[int(x) for x in true_pitch[i] if x>=0],
         candidate_midi=[int(x) for x in pred[i] if x>=0],
         true_notes=[note_name(x) for x in true_pitch[i] if x>=0],
         candidate_notes=[note_name(x) for x in pred[i] if x>=0]))
    report=dict(status='S400_FIRST_PITCH_IDENTITY_EVALUATION_S80_S397_DEVELOPMENT',
       reference_id='S80_S397_K5_GUARDED',
       read_JAMS_labels_ONLY_AFTER_S399_SHA_sealed=True,
       N=n,annotated_members=members_valid,
       annotation_note_events=events_total,unassigned_note_events=unassigned_total,
       label_cardinality_match=int(consistent.sum()),
       label_cardinality_mismatch=int((~consistent).sum()),
       annotation_events_usable=int(valid.sum()),
       unusable_reason_counts=dict(target_counter),
       source_note_SHA256=sha(source),annotation_zip_SHA256=sha(annotations),
       metrics=measurements,
       errors_sample=errors,
       output_notes_are_MIDI_multi_set=True,
       duplicate_pitch_same_octave_limit=True,
       active_reference_ExactK_source_training_overlap=True,
       decoder_is_zero_shot_fixed_formula=True,
       any_scorer_trained_on_test_labels=False,
       external_unseen_musical_evaluation=False,
       reference_exactK_unchanged=True,
       promoted_to_production=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('errors_sample','metrics')},indent=2),flush=True)
    print('S400 METRICS',json.dumps(measurements),flush=True)
    print('S400 PASS: scored real MIDI note identities, preserved S80 K and separated oracle',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('audio','source','source-report','original-truth','annotations','out'):
        p.add_argument('--'+k,required=True,type=Path)
    v=p.parse_args()
    run(v.audio,v.source,v.source_report,v.original_truth,v.annotations,v.out)
