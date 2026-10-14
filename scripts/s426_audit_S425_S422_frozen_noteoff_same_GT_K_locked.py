"""S426 — independently audit real S425 vs S422 exported MIDI NoteOff results.

Reparse FROZEN full-event MIDI JSONL (not the training report), verify each
S80 source event exports exactly original K0..K6, and evaluate corrected /
regressed SAME true GuitarSet note IDs. No model fit, threshold selection or
reference changes. This does NOT remove historical S80/GuitarSet exposure.
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
ORIGINAL=16340
IMPROVED=16543
def need(x,s):
    if not x:raise ValueError(s)
def parse(p):
    tracks,counts=read_prediction(p)
    return tracks,counts
def recode(rows):
    r=count_pair(rows)
    return dict(n=r["n"],S422_correct=r["S417_correct"],
                S425_correct=r["S420_correct"],
                corrections=r["correction"],regressions=r["regression"],
                net=r["net"])
def run(source_s80,S422,S425,annotation,out):
    need(not out.exists(),"S426 immutable independent audit exists")
    old,counts_old=parse(S422)
    new,counts_new=parse(S425)
    need(set(old)==set(new) and len(old)==190,
         "S426 not comparing same original 190 GuitarSet tracks")
    with np.load(source_s80,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        K=z["s80_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N and int(K.sum())==NOTES
         and np.all(np.isin(K,range(7))),"S426 source S80 predictions changed")
    need(all(counts_old[row]==int(k) and counts_new[row]==int(k)
             for row,k in zip(ids,K)),
         "S426 CRITICAL S80 source-note count regression in actual JSONL")
    valid=set(ids)
    need(all(x in valid for x in counts_new) and all(x in valid for x in counts_old),
         "S426 foreign output MIDI row")
    gt=load_gt(annotation,old)
    matched={"S422":0,"S425":0}
    grouped=collections.defaultdict(list)
    byfold=collections.defaultdict(list)
    bycomp=collections.defaultdict(list)
    same_notes=[]
    for track,ref in gt.items():
        a=old[track];b=new[track]
        # Source-row ID + pitch + onset MUST match at the exact count level.
        key=lambda items:collections.Counter((x[0],x[2],x[3]) for x in items)
        need(key(a)==key(b),
             "S426 S425 accidentally modified original NoteOn, MIDI or source K")
        true_old,m0=success_set(a,ref)
        true_new,m1=success_set(b,ref)
        need(m0==m1,"S426 source MIDI/Onset matches unexpectedly changed")
        matched["S422"]+=m0
        matched["S425"]+=m1
        fold=str(composition_fold(composition_key(track)))
        for idx,(start,finish,pitch,string) in enumerate(ref):
            record=(idx in true_old,idx in true_new)
            same_notes.append(record)
            age_ms=(finish-start)/44100*1000
            grouped[duration_group(age_ms)].append(record)
            byfold[fold].append(record)
            bycomp[composition_key(track)].append(record)
    total=recode(same_notes)
    need(total["n"]==GT and total["S422_correct"]==ORIGINAL
         and total["S425_correct"]==IMPROVED
         and matched=={"S422":25395,"S425":25395},
         "S426 independent true note reproduction fails pinned S425 baseline")
    report=dict(
        status="S426_INDEPENDENT_S425_S422_FROZEN_MIDI_NOTE_PAIRED_AUDIT",
        original_S80_exactK_reference_unchanged=True,
        original_K_rows=N,
        actual_exported_MIDI_note_on_events=NOTES,
        actual_per_row_K_identical_for_all_59309_events=True,
        K0_no_new_NoteOn_confirmed=True,
        exactK_regressions=0,
        true_GuitarSet_note_events=GT,
        MIDI_pitch_and_onset_instances_identical=True,
        matched_pitch_onset=matched,
        same_true_note_corrections_vs_regressions=total,
        by_GuitarSet_true_duration={s:recode(g) for s,g in grouped.items()},
        by_held_composition_fold={s:recode(g) for s,g in byfold.items()},
        by_composition={s:recode(g) for s,g in bycomp.items()},
        S422_F1=2*ORIGINAL/(NOTES+GT),
        S425_F1=2*IMPROVED/(NOTES+GT),
        no_model_training_or_threshold_change=True,
        upstream_meta_features_not_fully_nested=True,
        S80_historical_GuitarSet_train_overlap=True,
        independent_unseen_music_validation=False,
        old_user_references_S80_S402_S408_unmodified=True,
        live_mode_untouched=True,
        official_promotion=False,
        SHA256=dict(original_K=sha(source_s80),
           original_S422_notes=sha(S422),
           new_S425_notes=sha(S425),
           original_JAMS=sha(annotation)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S426 PAIRED",json.dumps(total),flush=True)
    print("S426 BY_DURATION",json.dumps(report["by_GuitarSet_true_duration"]),flush=True)
    print("S426 BY_FOLD",json.dumps(report["by_held_composition_fold"]),flush=True)
    print("S426 PASS: 0 S80 K changes in actual exported MIDI, independent correction/regr audit",flush=True)
def selftest():
    x=recode([(True,False),(False,True),(True,True),(False,False)])
    need(x["S422_correct"]==2 and x["S425_correct"]==2
         and x["corrections"]==1 and x["regressions"]==1,"incorrect paired corrections")
    print("S426 SELFTEST PASS")
if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for n in ("s80","s422","s425","annotations","out"):
        q.add_argument("--"+n,type=Path)
    a=q.parse_args()
    if a.self_test:selftest()
    else:
        for n in ("s80","s422","s425","annotations","out"):
            need(getattr(a,n)is not None,"missing --"+n)
        run(a.s80,a.s422,a.s425,a.annotations,a.out)
