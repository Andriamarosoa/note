"""S423 — independent frozen S422 vs S420 identical-GT-note Offset audit.

No refit, calibration or threshold changes. Count predictions from both
actually exported JSONL files and recheck all 59309 S80 per-row K. Compare
exact same true GuitarSet notes for positive/negative full-note corrections.
A model that only lowers MAE or improves long-notes while regressing
the main metric is not silently promoted.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s421_audit_S420_vs_S417_pairwise_same_true_MIDI_notes_Klocked import (
    read_prediction,load_gt,success_set,count_pair,duration_group)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

N=59309
NOTES=33475
GT=34954
def need(x,s):
    if not x:raise ValueError(s)
def run(official_s80,base_notes,new_notes,annotations,out):
    need(not out.exists(),'S423 immutable report cannot overwrite')
    base,oldcounts=read_prediction(base_notes)
    new,newcounts=read_prediction(new_notes)
    with np.load(official_s80,allow_pickle=False) as z:
        ids=z['row_id'].astype(str);k=z['s80_K'].astype(np.int8)
    need(len(ids)==N and int(k.sum())==NOTES
       and all(oldcounts[r]==int(y) and newcounts[r]==int(y)
           for r,y in zip(ids,k)),
       'S423 independent ACTUAL emitted JSONL Exact-K mismatch')
    need(set(base)==set(new) and len(base)==190,
         'S423 comparison must use original 190 songs')
    originals=load_gt(annotations,base)
    total=[];groups=collections.defaultdict(list)
    folddict=collections.defaultdict(list)
    byK=collections.defaultdict(list)
    counts={'S420_onset_pitch':0,'S422_onset_pitch':0}
    for song,ref in originals.items():
        a=base[song];b=new[song]
        # Guarantee SAME predicted note identity and onset in both models.
        # Only NoteOff may change. K0 = no new NoteOn, not silence.
        make=lambda items:collections.Counter((q[0],q[2],q[3]) for q in items)
        need(make(a)==make(b),
             'S423 S422 changed pitch or onset, not merely Offset')
        ab,ma=success_set(a,ref)
        bb,mb=success_set(b,ref)
        need(ma==mb,'S423 identical pitch/onset should give same match cardinality')
        counts['S420_onset_pitch']+=ma
        counts['S422_onset_pitch']+=mb
        fold=str(composition_fold(composition_key(song)))
        for i,(start,stop,pitch,string) in enumerate(ref):
            pair=(i in ab,i in bb)
            total.append(pair)
            byK[fold].append(pair)
            folddict[fold].append(pair)
            ms=(stop-start)/44100*1000
            groups[duration_group(ms)].append(pair)
    paired=count_pair(total)
    need(paired['n']==GT and paired['S417_correct']==16140
       and counts['S420_onset_pitch']==25395
       and counts['S422_onset_pitch']==25395,
       'S423 prior full-note S420 denominator or score changed')
    # Aliases S417 -> S420, S420 -> S422 because shared pure
    # count_pair originally designed for S417-to-S420 audit.
    def relabel(x):
        return dict(n=x['n'],S420_correct=x['S417_correct'],
            S422_correct=x['S420_correct'],
            corrections=x['correction'],regressions=x['regression'],
            net=x['net'])
    table=relabel(paired)
    strata={name:{key:relabel(count_pair(rows)) for key,rows in bucket.items()}
            for name,bucket in [('by_duration',groups),('by_fold',folddict)]}
    report=dict(status='S423_INDEPENDENT_REAL_JSONL_SURVIVAL_OFFSET_PAIRWISE_AUDIT',
       same_real_190_GuitarSet_tracks=True,
       same_total_GT_notes=GT,
       same_source_59309_S80_K_per_row=True,
       same_33475_predicted_MIDI_instances=True,
       exact_K_regressions=0,
       unchanged_MIDI_pitch_and_NoteOn=True,
       matched_original_MIDI_NoteOn_pitch=counts,
       trained_models_not_refit=True,
       decisions_and_survival_thresholds_UNCHANGED=True,
       F1_formula='2*full_note_TP/(33475+34954)',
       S420_note_complete_F1_50ms=2*16140/(NOTES+GT),
       S422_note_complete_F1_50ms=2*table['S422_correct']/(NOTES+GT),
       score=table,
       groups=strata,
       long_notes_1000ms_plus=strata['by_duration'].get('1000ms+'),
       S420_user_reference_and_S80_unchanged=True,
       no_promotion=True,
       historical_GuitarSet_overlap_with_upstream_S80=True,
       source_sha256=dict(S80=sha(official_s80),S420=sha(base_notes),
                          S422=sha(new_notes),JAMS=sha(annotations)))
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S423 PAIRED',json.dumps(table),flush=True)
    print('S423 BY_DURATION',json.dumps(strata['by_duration']),flush=True)
    print('S423 BY_FOLD',json.dumps(strata['by_fold']),flush=True)
    print('S423 PASS independent exact S80 K0-K6 preservation from MIDI files',flush=True)

def test():
    a=[(True,False),(False,True),(True,True)]
    x=count_pair(a)
    need(x['correction']==1 and x['regression']==1,
         'S423 independent paired loss/gain accounting')
    print('S423 SELFTEST PASS: frozen corrections vs regressions are distinct')
if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--self-test',action='store_true')
    for name in ('s80','s420','s422','annotations','out'):
        q.add_argument('--'+name,type=Path)
    a=q.parse_args()
    if a.self_test:test()
    else:
        for name in ('s80','s420','s422','annotations','out'):
            need(getattr(a,name)is not None,'Missing --'+name)
        run(a.s80,a.s420,a.s422,a.annotations,a.out)
