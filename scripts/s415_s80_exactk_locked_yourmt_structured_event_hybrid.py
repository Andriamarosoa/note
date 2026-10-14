"""S415 — STRICT S80-LOCKED complete MIDI note event decoder.

Testing a structural WHAT-IF: frozen official YourMT3+ note events act as
external proposals for pitch/onset/offset, but cannot change S80 predicted K,
not even for K0. Fallback is frozen S412 (or S408) note predictions. This
requires the trained YourMT3+ checkpoint to run in the real world; it is NOT
a newly-trained standalone native note model. It is a proof-of-mechanism
for structured event proposals under exact S80 count conservation.

All proposals were generated BEFORE opening JAMS ground truth.
Every one of 59,309 native GuitarSet events, including K0, must preserve
exactly its S80 K. Head-level EXACT K audit is performed on original
trueK annotations only AFTER all MIDI predictions are frozen.
S80 counts, weights and official baseline are never modified.

Same song comparison with cached official YourMT3+ and S412/S408 results
can be DESCRIPTIVE only; both systems have prior GuitarSet exposure.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s414_compare_official_yourmt3_full_notes_vs_S408_S412_guitarset import (
    read_original_jams,score,load_ymt_folds)
from scripts.s402_k_input_first_supervised_midi_pitch_network import composition_key
from scripts.s406_score_S80_gate_native_onsets_real_GuitarSet import npz
SR=44100
MIN_MIDI=40
MAX_MIDI=88
N_EVENTS=59309
N_PRED=33475
FOLDS=(0,1,2,4)
SOURCE_K_CORRECT=53529
SOURCE_POLY_K_CORRECT=6054

def require(x,msg):
    if not x:raise ValueError(msg)

def official_assigned_candidates(fold_paths):
    """Read source note JSON unchanged, index only native assigned proposals.
    Unassigned notes stay out of the strict native-event comparison, but
    are counted and raw teacher scores are reported elsewhere in S414.
    """
    byrow=collections.defaultdict(list)
    unseen=collections.Counter()
    tracks=set()
    hashes={}
    for fold,root in zip(FOLDS,fold_paths):
        path=root/'notes.jsonl'
        require(path.exists(),'Missing original official YourMT3+ note artifact '+str(path))
        hashes[str(fold)]=sha(path)
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            d=json.loads(line)
            member=str(d['member'])
            require(member not in tracks,'YourMT3+ duplicated source track')
            tracks.add(member)
            for obj in d['notes']:
                idx=obj.get('global_index')
                if idx is None:
                    unseen['unassigned_original_teacher_notes']+=1
                    continue
                idx=int(idx)
                midi=int(obj['pitch'])
                if midi<MIN_MIDI or midi>MAX_MIDI:
                    unseen['outside_source_MIDI40_88']+=1
                    continue
                onset=int(obj['onset_sample'])
                off=obj.get('offset_sample')
                if off is None:
                    second=obj.get('offset')
                    off=int(round(float(second)*SR)) if second is not None else -1
                off=int(off)
                if off<=onset:
                    unseen['invalid_original_teacher_offset']+=1
                    continue
                # Teacher event timestamp/pitch/duration frozen; no new tuning.
                byrow[idx].append((onset,off,midi))
                unseen['eligible_assigned_notes']+=1
    require(len(tracks)==190,'Expected all 190 original GuitarSet YourMT3+ source tracks')
    return byrow,dict(unseen),hashes

def source_group_and_notes(groups,K,ours):
    """K0..K6 are immutable provided by original S80 group predictions."""
    ids=groups['row_id'].astype(str)
    member=groups['member'].astype(str)
    start=groups['start_samples'].astype(np.int64)
    end=groups['end_samples'].astype(np.int64)
    k=np.asarray(K,dtype=np.int8)
    require(len(ids)==N_EVENTS and len(set(ids))==N_EVENTS and
      k.shape==(N_EVENTS,) and np.isin(k,range(7)).all() and
      np.all(start<=end) and
      len(np.unique(member))==190,
      'S80 original group count, K or track membership changed')
    source_ids=ours['source_row_id'].astype(str)
    source_midi=ours['midi_pitch'].astype(int)
    source_on=ours['S407_onset_samples'].astype(np.int64)
    source_off=ours['continuous_normal_direct_offset_samples'].astype(np.int64)
    source_off408=ours['S408_original_offset_samples'].astype(np.int64)
    require(len(source_ids)==N_PRED and
       len(source_midi)==N_PRED and
       len(source_on)==N_PRED and len(source_off)==N_PRED,
       'Frozen note source S408/S412 count mismatch')
    index={s:i for i,s in enumerate(ids)}
    fallback=collections.defaultdict(list)
    for i,r in enumerate(source_ids):
        require(r in index,'S412 row not in immutable S80 group indices')
        j=index[r]
        require(str(ours['track_member'][i])==member[j],'S412 note track and K row mismatch')
        require(source_off[i]>source_on[i] and source_off408[i]>source_on[i],
                'S412 frozen NoteOff cannot precede onset')
        fallback[r].append((int(source_on[i]),int(source_off[i]),int(source_midi[i]),
                            int(source_off408[i])))
    for i,key in enumerate(ids):
        require(len(fallback[key])==int(k[i]),'S415 input K does not match original MIDI instance cardinality')
    return ids,member,start,end,fallback

def choose_exact_k(teacher,backup,K,mid):
    """Independent of labels: prefer real teacher note candidates per native
    group, sorted closest to event midpoint, then fill from held-song
    S412 notes. Preserve K even when teacher undercounts/overcounts.
    """
    require(0<=K<=6,'S80 K outside K0..K6')
    if K==0:return [],0,0
    ordered=sorted(teacher,key=lambda z:(abs(z[0]-mid),z[0],z[2],z[1]))
    result=[]
    for on,off,pitch in ordered[:K]:
        result.append((int(on),int(off),int(pitch),'yourmt'))
    teacher_count=len(result)
    already={x[2] for x in result}
    remaining=sorted(backup,key=lambda z:(z[2] in already,abs(z[0]-mid),z[2]))
    for on,off,pitch,_ in remaining:
        if len(result)==K:break
        if pitch in already:continue
        result.append((int(on),int(off),int(pitch),'S412_fallback'))
        already.add(pitch)
    if len(result)<K:
        for on,off,pitch,_ in remaining:
            if len(result)==K:break
            result.append((int(on),int(off),int(pitch),'S412_repeat_fill'))
    require(len(result)==K,'Even fallbacks could not fill S80 exact-K')
    require(all(y[1]>y[0] for y in result),'Invalid temporal note duration')
    return result,teacher_count,len(ordered)

def decode(g,K,ours,teachers):
    ids,member,start,end,fallback=source_group_and_notes(g,K,ours)
    events=collections.defaultdict(list)
    n_byrow=np.zeros(N_EVENTS,np.int8)
    assigned=collections.Counter()
    origin=collections.Counter()
    for i,key in enumerate(ids):
        pred,tcount,pre_count=choose_exact_k(
            teachers.get(int(key),[]),fallback[key],int(K[i]),
            int(start[i]+(end[i]-start[i])//2))
        n_byrow[i]=len(pred)
        assigned['teacher_slots_used']+=tcount
        assigned['teacher_candidate_slots_prelock']+=pre_count
        assigned['teacher_overflow_discarded']+=max(0,pre_count-int(K[i]))
        assigned['filled_from_S412']+=int(K[i])-tcount
        for o in pred:
            # Preserve source row ID in each emitted event so an independent
            # post-run auditor can reconstruct K counts from the actual JSONL,
            # not merely trust an array copied from S80.
            events[str(member[i])].append((o[0],o[1],o[2],True,key,o[3]))
            origin[o[3]]+=1
    require(np.array_equal(n_byrow,K),
       'CRITICAL EXACT K REGRESSION: final event count differs from frozen S80')
    require(int(n_byrow.sum())==N_PRED,
       'Total MIDI instances must equal original S80 total 33475')
    return events,n_byrow,dict(assigned),dict(origin)

def run(audio,s80,truth,s412,fold_paths,jams,out):
    require(not out.exists(),'Immutable S415 audit destination already exists')
    g=npz(audio);r=npz(s80);labels=npz(truth);ours=npz(s412)
    ids=g['row_id'].astype(str)
    require(np.array_equal(ids,r['row_id'].astype(str)) and
      np.array_equal(ids,labels['row_id'].astype(str)),
      'S415 original aligned 59309 S80 note groups absent')
    K=r['s80_K'].astype(np.int8)
    y=labels['true_K'].astype(np.int8)
    require(len(K)==N_EVENTS and
      int((y==K).sum())==SOURCE_K_CORRECT and
      int(np.count_nonzero((y>=2)&(y==K)))==SOURCE_POLY_K_CORRECT,
      'CRITICAL original S80 Exact K truth reference drift before decoder')
    teacher,teacher_controls,hashes=official_assigned_candidates(fold_paths)
    events,n_byrow,controls,origin=decode(g,K,ours,teacher)
    require(np.array_equal(K,n_byrow),'S80 original predictions were modified')
    # Prediction is fully frozen before opening JAMS.
    truth_midi=read_original_jams(jams,events.keys())
    reference=collections.defaultdict(list)
    challenger=collections.defaultdict(list)
    for i,m in enumerate(ours['track_member'].astype(str)):
        old=(int(ours['S407_onset_samples'][i]),
             int(ours['S408_original_offset_samples'][i]),
             int(ours['midi_pitch'][i]),True)
        new=(old[0],int(ours['continuous_normal_direct_offset_samples'][i]),old[2],True)
        reference[m].append(old)
        challenger[m].append(new)
    metrics={}
    for name,z in [('S408_reference',reference),
        ('S412_challenger',challenger),
        ('S415_LOCKED',events)]:
        scores,_=score(z,truth_midi)
        metrics[name]=dict((k,v) for k,v in scores.items() if k!='by_track')
    require(metrics['S408_reference']['correct_offsets_conditional_on_matched_pitch_onset']==13050
        and metrics['S412_challenger']['correct_offsets_conditional_on_matched_pitch_onset']==14313,
        'Original S408 and S412 full-note baseline drift; cannot compare')
    # Save complete per-row counts and selected notes (including source tag)
    # without GT duration, and separately integrity/metrics report.
    out.mkdir(parents=True)
    saved=out/'S415_exact_S80_locked_structured_note_events.jsonl'
    with saved.open('w',encoding='utf8') as f:
        for track in sorted(events):
            f.write(json.dumps(dict(member=track,notes=[
                dict(onset_sample=ev[0],offset_sample=ev[1],
                     pitch=ev[2],source_row_id=ev[4],origin=ev[5])
                for ev in events[track]]))+'\n')
    np.savez_compressed(out/'S415_exactK_invariance_all_59309.npz',
       row_id=ids,original_S80_K=K,S415_note_count_per_row=n_byrow)
    byK={}
    for cls in range(7):
        mask=y==cls
        byK[str(cls)]=dict(n=int(mask.sum()),
          original_S80_exact=int(np.count_nonzero(K[mask]==cls)),
          candidate_S415_exact=int(np.count_nonzero(n_byrow[mask]==cls)),
          regression_count=0)
    report=dict(status='S415_FROZEN_S80_K_PER_ROW_STRUCTURED_MIDI_EVENT_AUDIT',
       S80_K_reference='S80_S397_K5_GUARDED',
       S80_true_K_correct_global=SOURCE_K_CORRECT,
       S80_true_K_correct_poly=SOURCE_POLY_K_CORRECT,
       exactK_regressions=0,
       exactK_corrections=0,
       total_native_events=N_EVENTS,
       S80_total_new_notes=int(K.sum()),
       S415_total_new_notes=int(n_byrow.sum()),
       S80_K0_silence_not_assumed=True,
       same_predicted_K_for_every_one_of_59309_rows=True,
       every_K_group_equal_K_exactly=True,
       teacher_is_OFFICIAL_pretrained_YourMT3plus_checkpoint=True,
       reference_S80_S402_S407_S408_S412_UNCHANGED=True,
       real_world_S415_requires_YourMT3plus_model_inference=True,
       this_IS_NOT_native_distilled_note_model=True,
       note_predictions_precomputed_no_GT_inputs=True,
       GuitarSet_historically_exposed_to_both_S80_and_YourMT3=True,
       official_reference_promotion=False,
       teacher_slot_audit=controls,
       note_origin=origin,
       original_teacher_note_controls=teacher_controls,
       original_teacher_fold_hashes=hashes,
       by_true_K_K_invariance=byK,
       metrics=metrics,
       original_S80_prediction_sha256=sha(s80),
       source_S412_SHA256=sha(s412),
       source_original_audio_SHA256=sha(audio),
       GT_JAMS_sha256=sha(jams),
       native_event_counts_SHA256=sha(out/'S415_exactK_invariance_all_59309.npz'),
       complete_notes_SHA256=sha(saved))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    display={k:{field:x[field] for field in (
        'predicted_notes','matched_onset_pitch_50ms','correct_offsets_conditional_on_matched_pitch_onset',
        'conditional_offset_50ms_accuracy','full_note_F1_50ms')}
        for k,x in metrics.items()}
    print('S415 RESULTS',json.dumps(display),flush=True)
    print('S415 K_LOCK',json.dumps(dict(total_notes=int(K.sum()),by_true_K=byK)),flush=True)
    print('S415 ORIGINS',json.dumps(origin),flush=True)
    print('S415 PASS 59309/59309 exact S80 K preserved, 0 Exact K regressions',flush=True)

def self_test():
    backup=[(100,400,48,410),(100,500,52,510),(100,600,55,610)]
    teacher=[(101,420,48),(103,620,60),(105,720,64),(107,900,67)]
    p,n,t=choose_exact_k(teacher,backup,2,100)
    require(len(p)==2 and n==2 and t==4 and all(x[3]=='yourmt' for x in p),
       'S415 overcount must be truncated to EXACT S80 K')
    p,n,t=choose_exact_k(teacher[:1],backup,3,100)
    require(len(p)==3 and n==1 and {x[2] for x in p}=={48,52,55},
       'S415 teacher undercount must be filled without K regression')
    p,n,t=choose_exact_k(teacher,backup,0,100)
    require(p==[] and n==0,'S415 K0 must NEVER create NoteOn even from YourMT3')
    for k in range(7):
        fallback=[(100,400,40+i,410) for i in range(6)]
        new,_,_=choose_exact_k(teacher[:2],fallback,k,100)
        require(len(new)==k,'S415 exact K per group regression')
    print('S415 SELFTEST PASS: K0..K6 immutable, YourMT3 under/overcount cannot regress Exact K')
if __name__=='__main__':
    z=argparse.ArgumentParser()
    z.add_argument('--self-test',action='store_true')
    for n in ('audio','s80','truth','s412','annotations','out'):
        z.add_argument('--'+n,type=Path)
    for f in FOLDS:z.add_argument('--fold'+str(f),type=Path)
    a=z.parse_args()
    if a.self_test:self_test()
    else:
        for n in ('audio','s80','truth','s412','annotations','out',
                  'fold0','fold1','fold2','fold4'):
            require(getattr(a,n) is not None,'Missing '+n)
        run(a.audio,a.s80,a.truth,a.s412,
            [getattr(a,'fold'+str(f)) for f in FOLDS],a.annotations,a.out)
