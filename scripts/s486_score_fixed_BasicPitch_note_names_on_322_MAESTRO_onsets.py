"""S486 independent paired note-names evaluation of S485 frozen Basic Pitch.

Use EXACT SAME S476 322 true MAESTRO MIDI onsets and the 73 true poly
new-note groups on which original V88 K baseline scored 42/322, 7/73.
Evaluate pretrained model note-on predictions including whole-note set,
K, recall/precision and attack alignment, using locked ±80ms onset
comparison (±150ms descriptive alternative, NOT tuned). Do not retrain
Spotify or source V88, never promote S80 based on V88 comparison.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
import numpy as np
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import match_times
from scripts.s485_freeze_real_MAESTRO_basic_pitch_pretrained_note_names import name
def need(c,msg):
    if not c:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def group_notes(items):
    """Group pre-frozen model NEW NOTE events into onset groups within 20ms."""
    groups=[]
    for e in sorted(items,key=lambda x:(float(x["predicted_onset_s"]),int(x["pitch_MIDI"]))):
        t=float(e["predicted_onset_s"])
        if not groups or t-float(groups[-1]["onset"])>.020:
            groups.append(dict(onset=t,pitches=set()))
        groups[-1]["pitches"].add(int(e["pitch_MIDI"]))
    return groups
def evaluate(truth,actual,tolerance):
    ref_on=np.asarray([float(q["onset_at_excerpt_s"]) for q in truth],float)
    candidate_on=np.asarray([float(q["onset"]) for q in actual],float)
    linked=match_times(ref_on,candidate_on,tolerance)
    assigned={i:j for i,j,_ in linked}
    alltruth=0;allpred=0;allintersect=0;sets_right=0;Kright=0
    poly_n=polyK=polyexact=0
    truthK0=0;onsets_matched=len(linked)
    outcomes=[]
    for i,row in enumerate(truth):
        target=set(map(int,row["true_new_onset_MIDI_pitches"]))
        pred=set(actual[assigned[i]]["pitches"]) if i in assigned else set()
        correct=len(pred)==len(target)
        exact=pred==target
        ispoly=len(target)>=2
        alltruth+=len(target);allpred+=len(pred);allintersect+=len(target&pred)
        Kright+=int(correct);sets_right+=int(exact)
        if ispoly:
            poly_n+=1;polyK+=int(correct);polyexact+=int(exact)
        truthK0+=int(len(pred)==0)
        outcomes.append(dict(
            row_id=row["row_id"],source_sha=row["row_id"].split(":")[0],
            truth_MIDI=sorted(target),proposed_MIDI=sorted(pred),
            truth_note_names=[name(v) for v in sorted(target)],
            predicted_note_names=[name(v) for v in sorted(pred)],
            exactK=correct,exact_note_set=exact,
            true_poly=ispoly,matched=bool(i in assigned)))
    return dict(
        tolerance_milliseconds=int(round(tolerance*1000)),
        original_reference_true_attacks=len(truth),
        predicted_attack_groups_total=len(actual),
        predicted_attack_groups_matched_to_reference=onsets_matched,
        annotated_onsets_without_matched_model_attack=len(truth)-onsets_matched,
        exactK_correct=Kright,
        exactK_accuracy=Kright/len(truth),
        exact_note_sets_correct=sets_right,
        exact_note_set_accuracy=sets_right/len(truth),
        true_poly_events=poly_n,
        true_poly_exactK=polyK,
        true_poly_exactK_accuracy=(polyK/poly_n if poly_n else None),
        true_poly_exact_notes=polyexact,
        true_poly_exact_note_set_accuracy=(
            polyexact/poly_n if poly_n else None),
        true_notes_total=alltruth,
        predicted_notes_on_matched_truth=allpred,
        correct_pitch_note_count=allintersect,
        note_precision=allintersect/allpred if allpred else None,
        note_recall=allintersect/alltruth if alltruth else None,
        reference_attacks_model_predicted_no_note=truthK0),outcomes
def run(note_file,note_report,truth_file,truth_report,out):
    need(not out.exists(),"S486 postfreeze comparison cannot overwrite previous result")
    s=json.loads(note_report.read_text())
    t=json.loads(truth_report.read_text())
    need(s["status"]=="S485_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI"
         and s["frozen_notes_SHA256"]==sha(note_file)
         and s["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False
         and s["model_fitted_or_labels_tuned"] is False
         and s["prediction_windows_piano"]==3
         and t["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and t["SHA256"]["paired_label_comparison"]==sha(truth_file)
         and t["matched_onsets"]==322,
         "S486 independent third source original model and MIDI note truth contract changed")
    notes=[json.loads(l) for l in note_file.read_text().splitlines()]
    truth=[json.loads(l) for l in truth_file.read_text().splitlines()]
    need(len(truth)==322 and len(notes)==s["total_notes_in_MIDI40to88"],
         "S486 fixed pretrained note predictions or original true attacked groups missing")
    group=collections.defaultdict(list)
    anchors=collections.defaultdict(list)
    for note in notes:group[note["WAV_filename_SHA256"]].append(note)
    for row in truth:anchors[row["row_id"].split(":")[0]].append(row)
    need(len(group)==3 and len(anchors)==3 and set(group)==set(anchors),
         "S486 all 3 original piano source WAVs must have valid positive estimated notes")
    reports={};detailed={}
    for lag in (.080,.150):
        full=[];per=[]
        for ident in sorted(anchors):
            true=sorted(anchors[ident],key=lambda r:r["onset_at_excerpt_s"])
            prediction=group_notes(group[ident])
            measures,events=evaluate(true,prediction,lag)
            reports[ident]=measures if lag==.080 else reports[ident]
            per.append(dict(source_sha=ident,**measures))
            full.extend(events)
        counts={
            "original_reference_true_attacks":sum(q["original_reference_true_attacks"] for q in per),
            "predicted_attack_groups_total":sum(q["predicted_attack_groups_total"] for q in per),
            "predicted_attack_groups_matched_to_reference":sum(q["predicted_attack_groups_matched_to_reference"] for q in per),
            "exactK_correct":sum(q["exactK_correct"] for q in per),
            "exact_note_sets_correct":sum(q["exact_note_sets_correct"] for q in per),
            "true_poly_events":sum(q["true_poly_events"] for q in per),
            "true_poly_exactK":sum(q["true_poly_exactK"] for q in per),
            "true_poly_exact_notes":sum(q["true_poly_exact_notes"] for q in per),
            "correct_pitch_note_count":sum(q["correct_pitch_note_count"] for q in per),
            "true_notes_total":sum(q["true_notes_total"] for q in per),
            "predicted_notes_on_matched_truth":sum(q["predicted_notes_on_matched_truth"] for q in per)}
        need(counts["original_reference_true_attacks"]==322
             and counts["true_poly_events"]==73,
             "S486 original 322 paired attack events 73 poly label comparability drift")
        counts.update(
            note_set_accuracy=counts["exact_note_sets_correct"]/322,
            exactK_accuracy=counts["exactK_correct"]/322,
            poly_exactK_accuracy=counts["true_poly_exactK"]/73,
            poly_note_set_accuracy=counts["true_poly_exact_notes"]/73,
            note_precision=(counts["correct_pitch_note_count"]/
                            max(1,counts["predicted_notes_on_matched_truth"])),
            note_recall=(counts["correct_pitch_note_count"]/
                         max(1,counts["true_notes_total"])))
        detailed[str(int(lag*1000))]=dict(aggregate=counts,per_piece=per)
        if lag==.080:paired=full
    need(int(sum(int(row["original_V88_K"]==row["true_new_onset_K"])
                 for row in truth))==42
         and int(sum(int(row["original_V88_K"]==row["true_new_onset_K"])
                 for row in truth if row["true_new_onset_K"]>=2))==7,
         "S486 original GuitarSet-trained V88 same-data old baseline not 42/322 new-note")
    out.mkdir(parents=True)
    pairfile=out/"S486_THIRD_PIANO_TRUE_VS_BASIC_PITCH_NOTE_NAME_PAIRS_80ms.jsonl"
    pairfile.write_text("".join(json.dumps(x)+"\n" for x in paired))
    report=dict(status="S486_PAIRED_PRETRAINED_BASIC_PITCH_PIANO_322_MIDI_NEW_ONSETS",
        model="Spotify Basic Pitch pretrained 0.4.0",
        musical_note_identification_not_MIDI_file_output=True,
        original_MAESTRO_three_piano_true_attack_rows=322,
        original_V88_same322_ExactK_correct=42,
        original_V88_same73_true_poly_ExactK_correct=7,
        BasicPitch_no_training_on_this_trial=True,
        no_inference_threshold_optimized_on_MAESTRO=True,
        MIDI_was_opened_historically_S471_S476=True,
        this_does_not_explain_S80_full_K=True,
        comparison_multiple_fixed_tolerances_not_model_selection=True,
        original_S80_Live_and_notes_untouched=True,
        no_promotion=True,
        by_precommitted_tolerance=detailed,
        SHA256=dict(S485_pretrained_frozen_notes=sha(note_file),
             S476_original_MAESTRO_true_pitch_events=sha(truth_file),
             S486_paired_80ms=sha(pairfile)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S486 REAL NOTES",json.dumps(dict(
        true322=322,
        V88_original_correct=42,
        V88_true_poly_correct=7,
        estimated_by_80ms=detailed["80"]["aggregate"],
        estimated_by_150ms=detailed["150"]["aggregate"])),flush=True)
def selftest():
    a=group_notes([
        dict(predicted_onset_s=.1,pitch_MIDI=52),
        dict(predicted_onset_s=.11,pitch_MIDI=55),
        dict(predicted_onset_s=.3,pitch_MIDI=59)])
    need(len(a)==2 and a[0]["pitches"]=={52,55}
         and name(52)=="E3",
         "S486 note name and grouped polyphonic onset not stable")
    print("S486 SELFTEST PASS E3 new pitch group and two notes simultaneous")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for key in ("notes","notes-report","truth","truth-report","out"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("notes","notes_report","truth","truth_report","out"):
            need(getattr(a,key)is not None,"S486 missing "+key)
        run(a.notes,a.notes_report,a.truth,a.truth_report,a.out)
