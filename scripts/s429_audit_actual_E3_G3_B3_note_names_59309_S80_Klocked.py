"""S429 — independent NOTE NAME (E3/G3/B3) accuracy & confusion audit.

Reparse actual 59,309 exported note-name JSONL from S428 and compare
the original frozen S419 predictions with real JAMS annotations.
S428 ranker selection was trained separately; this file reads ONLY
frozen predictions for independent paired replay. No training, no music
theory oracle inserted into predictions. Never alter S80 K0..K6.

Distinguishes exact musical pitch identities from just good note counts.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    load,labels,composition_key,composition_fold,need)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha,note_name

N=59309
POLY=7385
SUM_K=33475

def signature(x):
    return collections.Counter(int(z) for z in x if z>=0)
def note_pair(target,predicted):
    aa=signature(target);bb=signature(predicted)
    return aa==bb, sum((aa&bb).values())

def categorize_confusion(ref,pred):
    """Diagnose missed / extra pitches after exact matches removed."""
    truth=signature(ref)
    found=signature(pred)
    common=truth&found
    left=list((truth-common).elements())
    right=list((found-common).elements())
    cats=collections.Counter()
    used=set()
    for one in sorted(left):
        if not right:
            cats["missing_fundamental"]+=1
            continue
        gaps=[abs(int(p)-int(one)) for p in right]
        j=int(np.argmin(gaps));delta=int(right.pop(j))-int(one)
        if abs(delta)==12:kind="octave_confusion"
        elif abs(delta) in (1,2):kind="neighbor_semitones"
        elif abs(delta) in (7,19):kind="fifth_or_harmonic_19"
        elif abs(delta)==24:kind="two_octaves"
        else:kind="other_pitch_interval"
        cats[kind]+=1
        used.add(one)
    cats["extra_predicted_notes"]+=len(right)
    return cats

def run(audio,s80,source419,source428,trueK,annotations,names,out):
    need(not out.exists(),"S429 report immutable")
    a=load(audio);r=load(s80);base=load(source419)
    c=load(source428);t=load(trueK)
    ids=a["row_id"].astype(str);K=r["s80_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
       and all(np.array_equal(ids,x["row_id"].astype(str))
               for x in (r,base,c,t))
       and np.array_equal(K,base["s80_K"])
       and np.array_equal(K,c["s80_K"])
       and int((K==t["true_K"]).sum())==53529
       and int(((K==t["true_K"])&(t["true_K"]>=2)).sum())==6054,
       "S429 S80 original K or 59309 event IDs drift")
    old=base["midi_pitch_slots"].astype(np.int16)
    new=c["midi_pitch_slots"].astype(np.int16)
    need(old.shape==new.shape==(N,6)
         and np.array_equal((old>=0).sum(1),K)
         and np.array_equal((new>=0).sum(1),K),
         "S429 source note pitch inventory changed")
    # REPARSE truly exported musical Note NAMES rather than trusting NPZ.
    line_count=0
    with names.open(encoding="utf8") as f:
        for i,line in enumerate(f):
            row=json.loads(line)
            pitches=[int(p) for p in new[i] if p>=0]
            need(row["source_row_id"]==ids[i]
               and row["K"]==int(K[i])
               and row["pitches"]==pitches
               and row["notes"]==[note_name(p) for p in pitches],
               "S429 actual Note-name JSONL event differs from original S80 K")
            line_count+=1
    need(line_count==N,"S429 omitted original GuitarSet note event")
    multi,truth,meta=labels(a,annotations,t)
    members=a["member"].astype(str)
    comps=np.array([composition_key(m) for m in members])
    folds=np.array([composition_fold(z) for z in comps],np.int8)
    byK={};byfold={}
    def audit(mask):
        count=collections.Counter()
        conf=collections.Counter()
        for i in np.flatnonzero(mask):
            a_hit,a_tp=note_pair(truth[i],old[i])
            b_hit,b_tp=note_pair(truth[i],new[i])
            count["n"]+=1
            count["old_chords"]+=int(a_hit);count["new_chords"]+=int(b_hit)
            count["old_true_notes"]+=a_tp;count["new_true_notes"]+=b_tp
            count["total_true_notes"]+=len(truth[i][truth[i]>=0])
            count["corrections"]+=int(b_hit and not a_hit)
            count["regressions"]+=int(a_hit and not b_hit)
            if not b_hit:
                conf.update(categorize_confusion(truth[i],new[i]))
        return dict(count,net_exact=int(count["new_chords"]-count["old_chords"]),
            error_types=dict(conf))
    for k in range(7):byK[str(k)]=audit(t["true_K"]==k)
    for f in range(4):byfold[str(f)]=audit(folds==f)
    poly=audit(t["true_K"]>=2)
    isolated=audit(t["true_K"]==1)
    need(poly["n"]==POLY and poly["old_chords"]==2775,
         "S429 original S419 polyphonic exact chords incorrect")
    scored_notes=dict(
      S419_exact=poly["old_chords"],S428_exact=poly["new_chords"],
      poly_total=POLY,
      S419_note_recall=poly["old_true_notes"]/poly["total_true_notes"],
      S428_note_recall=poly["new_true_notes"]/poly["total_true_notes"])
    report=dict(status="S429_INDEPENDENT_MUSICAL_NOTE_NAME_IDENTITY_AUDIT",
      same_original_S80_K_per_all_59309_events=True,
      exactK_regressions=0,real_name_rows_checked=line_count,
      total_emitted_new_notes=SUM_K,
      actual_labels_from_original_GuitarSet_JAMS=True,
      no_model_training_or_predictions_changed=True,
      no_YourMT_teacher_used=True,
      no_Live_or_NoteOff_change=True,
      development_not_independent_of_S80_GuitarSet_overlap=True,
      benchmark_promoted=False,
      notes_polyphonic=scored_notes,polyphonic_detail=poly,
      single_note_detail=isolated,
      by_true_K=byK,by_held_composition_fold=byfold,
      source_sha256=dict(S80=sha(s80),S419=sha(source419),
                          S428=sha(source428),JAMS=sha(annotations),
                          note_names=sha(names)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S429 INDEPENDENT PITCH RESULT",json.dumps(scored_notes),flush=True)
    print("S429 BY K",json.dumps(byK),flush=True)
    print("S429 PASS actual note E3/G3/B3 names K equal original S80 on all 59309 rows",flush=True)

def selftest():
    x=np.array([52,55,59,-1,-1,-1],np.int16)
    y=np.array([64,55,59,-1,-1,-1],np.int16)
    assert [note_name(v) for v in x if v>=0]==["E3","G3","B3"]
    assert categorize_confusion(x,y)["octave_confusion"]==1
    assert note_pair(x,x)==(True,3)
    print("S429 SELFTEST PASS E3/G3/B3 note names, octave correction audit")
if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for v in ("audio","s80","s419","s428","truth","annotations","notes","out"):
        q.add_argument("--"+v,type=Path)
    v=q.parse_args()
    if v.self_test:selftest()
    else:
        for key in ("audio","s80","s419","s428","truth","annotations","notes","out"):
            need(getattr(v,key) is not None,"missing --"+key)
        run(v.audio,v.s80,v.s419,v.s428,v.truth,v.annotations,v.notes,v.out)
