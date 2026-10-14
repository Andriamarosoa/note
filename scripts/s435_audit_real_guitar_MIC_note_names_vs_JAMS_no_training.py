"""S435 — audit ACTUAL real-guitar S434 predicted E3/note names vs JAMS.

This audit runs strictly AFTER the S434 original S80+native note names
were frozen. It does not refit, select thresholds, substitute a teacher,
modify K, or change any predictions. Uses ALL native groups of the chosen
piece to assign true JAMS attacks without boundary-truncation leakage.

The GuitarSet piece was historically exposed to S80; S434 uses MIC
recording rather than S258 pickup-mix training audio. This is a
DEVELOPMENT diagnostic, NOT independent cross-dataset accuracy.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import need
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import read_note_events
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    note_name,sha)
from scripts.s429_audit_actual_E3_G3_B3_note_names_59309_S80_Klocked import (
    categorize_confusion)

N=59309
def calculate(notes_file,source_report,original_audio,original_s80,truek,annotations,out):
    need(not out.exists(),"S435 audit cannot overwrite existing results")
    r=json.loads(source_report.read_text())
    need(r["status"]=="S434_AUTHENTIC_GUITARSET_MIC_WAV_NO_LABELS_WITH_FROZEN_S80_NATIVE_GROUPS"
         and r["no_truth_JAMS_or_labels_opened"],
         "S435 input predictions must have been sealed before annotations")
    predicted=[json.loads(x) for x in notes_file.read_text().splitlines()]
    member=r["original_member"]
    need(len(predicted)==r["original_source_event_count"]
         and len(predicted)>3,
         "S435 real frozen note name sample was replaced or truncated")
    with np.load(original_audio,allow_pickle=False) as z:
        need({"row_id","member","start_samples","end_samples"}<=set(z.files),
             "S435 original source member/timing archive missing")
        ids=z["row_id"].astype(str)
        members=z["member"].astype(str)
        start=z["start_samples"].astype(np.int64)
        end=z["end_samples"].astype(np.int64)
    with np.load(original_s80,allow_pickle=False) as z:
        need({"row_id","s80_K"}<=set(z.files),"S435 original S80 missing")
        need(np.array_equal(z["row_id"].astype(str),ids),
             "S435 S80 source row join not identical")
        original_K=z["s80_K"].astype(np.int8)
    with np.load(truek,allow_pickle=False) as z:
        need({"row_id","true_K"}<=set(z.files),
             "S435 annotation-only K archive missing")
        need(np.array_equal(z["row_id"].astype(str),ids),
             "S435 original truth event labels misaligned")
        yK=z["true_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and int(np.sum(original_K))==33475,
         "S435 original 59309 reference source changed")
    positions=np.flatnonzero(members==member)
    positions=positions[np.argsort(start[positions],kind="stable")]
    need(len(positions)>0,"S435 no full authentic source track events")
    with zipfile.ZipFile(annotations) as z:
        candidates=[x for x in z.namelist()
            if PurePosixPath(x).name==PurePosixPath(member).name]
        need(len(candidates)==1,
             "S435 true GuitarSet annotation must match original track ID")
        events=read_note_events(z.read(candidates[0]))
    source_ons=np.array([q[0] for q in events],np.int64)
    _,assigned=assign_onsets(source_ons,start[positions],end[positions])
    targets=collections.defaultdict(list)
    for (sample,pitch,string),slot in zip(events,assigned):
        if slot<0:continue
        targets[int(positions[int(slot)])].append(int(pitch))
    # Verify original JAMS onboarding is EXACTLY identical to historical
    # S258 true K on the entire piece, including excluded source rows.
    need(all(len(targets[i])==int(yK[i]) for i in positions),
         "S435 ground truth assignment does not reproduce full original GuitarSet true K")
    original_ref={rid:i for i,rid in enumerate(ids)}
    analysis=[]
    matrix=collections.defaultdict(lambda:collections.Counter())
    confusion=collections.Counter()
    note_to_number={note_name(i):i for i in range(40,89)}
    totals=collections.Counter()
    for line in predicted:
        key=str(line["source_row_id"])
        need(key in original_ref,"S435 foreign predicted row ID")
        i=original_ref[key]
        need(member==members[i],"S435 prediction leaks into different guitar piece")
        k=int(original_K[i])
        names=line["notes"]
        need(line["K"]==k and len(names)==k and
             line["start_sample"]==int(start[i]),
             "S435 original S80 K/time altered in frozen name predictions")
        need(all(x in note_to_number for x in names),
             "S435 nonmusical name outside native model 40..88")
        y=list(targets[i]);p=[note_to_number[x] for x in names]
        y_count=collections.Counter(y);p_count=collections.Counter(p)
        exact=bool(y_count==p_count)
        tp=int(sum((y_count&p_count).values()))
        totals["n"]+=1
        totals["source_predicted_notes"]+=k
        totals["true_notes"]+=len(y)
        totals["notes_TP"]+=tp
        totals["correct_chords"]+=int(exact)
        totals["correct_K"]+=int(k==int(yK[i]))
        matrix[f"true_K{int(yK[i])}"]["events"]+=1
        matrix[f"true_K{int(yK[i])}"]["chords_correct"]+=int(exact)
        if not exact:confusion.update(categorize_confusion(y,p))
        if len(analysis)<25:
            analysis.append(dict(original_row_id=key,source_K=k,true_K=int(yK[i]),
                predicted_notes=names,reference_notes=[note_name(z) for z in y],
                exact=exact))
    need(totals["n"]==len(predicted) and
         totals["source_predicted_notes"]==sum(len(x["notes"]) for x in predicted),
         "S435 frozen true note accounting inconsistent")
    f1=2*totals["notes_TP"]/(totals["true_notes"]+totals["source_predicted_notes"]) if (
        totals["true_notes"]+totals["source_predicted_notes"]) else 1.
    report=dict(status="S435_FROZEN_REAL_GUITARSET_MIC_WAV_NOTE_IDENTITY_DIAGNOSTIC",
        prior_S434_WAV_source_member=member,
        original_GuitarSet_native_full_piece_true_K_reconstructed=True,
        original_S80_row_K_untouched=True,
        exact_K_regressions=0,
        true_annotations_opened_only_after_predictions_sealed=True,
        real_recorded_guitar_MIC_audio_not_synthetic=True,
        compared_true_note_names_actual_JAMS=True,
        sample_source_events=totals["n"],
        sample_correct_K=totals["correct_K"],
        sample_exactK_accuracy=totals["correct_K"]/totals["n"],
        sample_correct_chords=totals["correct_chords"],
        sample_full_chord_accuracy=totals["correct_chords"]/totals["n"],
        sample_note_TP=totals["notes_TP"],
        sample_true_note_instances=totals["true_notes"],
        sample_predicted_note_instances=totals["source_predicted_notes"],
        sample_note_F1=f1,
        accuracy_is_on_historically_exposed_GuitarSet_not_blind=True,
        original_MIC_source_differs_from_S258_pickup_mix=True,
        sample_size_small_and_not_generalizable=True,
        no_independent_benchmark_claim=True,
        no_training_threshold_selection_or_model_changes=True,
        original_live_and_offsets_unchanged=True,
        by_true_K={k:dict(v) for k,v in matrix.items()},
        unmatched_pitch_confusion_types=dict(confusion),
        first_25_examples=analysis,
        previous_S428_full_GuitarSet_crossfit_40_96pct_NOT_DIRECTLY_COMPARABLE=True,
        official_promotion=False,
        sources_SHA256=dict(output_predictions=sha(notes_file),
            original_annotations=sha(annotations),S80=sha(original_s80),
            original_events=sha(original_audio),previous_truth_K=sha(truek)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S435 DIAGNOSTIC",json.dumps(dict(n=totals["n"],
        exactK=totals["correct_K"],exact_chords=totals["correct_chords"],
        note_F1=f1,confusions=dict(confusion))),flush=True)
    print("S435 NOTE EXAMPLES",json.dumps(analysis[:12]),flush=True)
    print("S435 PASS true JAMS notes compared AFTER real S434 WAV predictions frozen",flush=True)

def selftest():
    from collections import Counter
    a=Counter([52,55]);b=Counter([52,59])
    need(sum((a&b).values())==1
         and note_name(52)=="E3" and note_name(55)=="G3",
         "S435 acoustic note exact multiset accounting failed")
    print("S435 SELFTEST PASS")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    for field in ("notes","s434-source-report","original-audio",
                  "original-s80","original-true-k","annotations","out"):
        parser.add_argument("--"+field,type=Path)
    x=parser.parse_args()
    if x.self_test:selftest()
    else:
        for name in ("notes","s434_source_report","original_audio",
                     "original_s80","original_true_k","annotations","out"):
            need(getattr(x,name)is not None,"missing --"+name)
        calculate(x.notes,x.s434_source_report,x.original_audio,
                  x.original_s80,x.original_true_k,x.annotations,x.out)
