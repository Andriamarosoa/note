"""S529: independent no-refit risk-protected note-birth rescue at original V88
piano event centers. Read the six true trained S528 model predictions.

Predeclared four combinations: raw-only or raw+BP-trained 49-pitch
probabilities, each at previously fixed threshold .50/.70. Strict KEEP
policy never touches a frozen Basic Pitch NOTE-BIRTH set when it already
contains a note; only empty Basic Pitch birth events may receive added
pitches. This policy uses no ground truth in inference and cannot regress
a previously exactly correct *nonempty* original birth set.

Report genuine source-held note pitch TP, FP and exact chords, per true K
and separately original 102 S498 lost BP events. Report proposals on the
other 253 V88 events as UNKNOWN, not true or false note onsets. Retro-
spective development on 3 historically exposed piano recordings;
not independent prospective validation, not full-stream event precision,
not source S80 score or Live change. NO PROMOTION.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def need(x, message):
    if not x:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_lines(p):
    return [json.loads(z) for z in Path(p).read_text().splitlines() if z.strip()]


def frozen_sources(a):
    old = json.loads(a.audio_report.read_text())
    pre = json.loads(a.trained_report.read_text())
    truth = json.loads(a.truth_report.read_text())
    miss = json.loads(a.missing_report.read_text())
    need(old["status"] == "S527_PRELABEL_ALL_575_V88_PIANO_ONSET_CANDIDATES_49_RAW_PITCH_FEATURES"
         and old["frozen_feature_ledger_SHA256"] == sha(a.audio)
         and old["no_original_true_MIDI_annotation_read"]
         and pre["status"] == "S528_TRAINED_THREE_RECORDING_HELD_PIANO_NOTE_HEADS_ALL_V88_K0_SITES"
         and pre["predictions_sha256"] == sha(a.trained)
         and pre["truly_trained_6_models"] == 6
         and pre["three_recording_LOO_training_no_held_truth_in_fit"]
         and truth["status"] == "S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and truth["SHA256"]["paired_label_comparison"] == sha(a.truth)
         and miss["status"] == "S526_BP_CANONICAL_LOST_TRUE_V88_K0_AND_KPOS_PIANO_ATTACKS"
         and miss["every_event_ledger_SHA256"] == sha(a.missing),
         "S529 original S527, six S528 fitted models, postfreeze MIDI truths or S526 audit changed")
    frozen = read_lines(a.audio)
    learned = read_lines(a.trained)
    labels = read_lines(a.truth)
    lost = read_lines(a.missing)
    ids = [v["V88_original_event_id"] for v in frozen]
    need(len(ids) == len(set(ids)) == 575
         and [v["V88_original_event_id"] for v in learned] == ids,
         "S529 no V88 acoustic event or held probability may be dropped or re-ordered")
    idx = {row:i for i,row in enumerate(ids)}
    Y = np.zeros((575,49),bool)
    observed = np.zeros(575,bool)
    missing = np.zeros(575,bool)
    for r in labels:
        i = idx[r["row_id"]]
        need(not observed[i], "S529 repeated original true music attack")
        pitch = list(map(int, r["true_new_onset_MIDI_pitches"]))
        need(len(pitch)>0 and all(40<=n<=88 for n in pitch),
             "S529 real new-note piano targets must be complete MIDI 40..88")
        Y[i,np.asarray(pitch,dtype=int)-40] = True
        observed[i] = True
    for r in lost:
        i = idx[r["row_id"]]
        missing[i] = r["new_note_event_missing_from_training_rows_S505"]
    need(observed.sum() == 322 and missing.sum() == 102
         and not np.any(missing&~observed),
         "S529 cannot silently drop 102 previously missing true piano attacks")
    B = np.zeros_like(Y)
    for i, r in enumerate(frozen):
        for p in r["old_basicpitch_birth_pitches"]:
            if 40<=int(p)<=88:B[i,int(p)-40]=True
    probs = {}
    for key, field in (("raw_only","raw_only_49_pitch_prob"),
                       ("with_BP","with_BP_49_pitch_prob")):
        arr=np.asarray([r[field] for r in learned],np.float32)
        need(arr.shape==(575,49) and np.isfinite(arr).all()
             and (arr>=0).all() and (arr<=1).all(),
             "S529 original 49 MIDI named-pitch learned held probabilities invalid")
        probs[key]=arr
    originalK=np.array([r["original_V88_K_soft_prior_ONLY"] for r in frozen],int)
    rec=[r["source_recording_sha"] for r in frozen]
    need(len(set(rec)) == 3, "S529 only original three independent MAESTRO recording groups")
    return frozen, Y, observed, missing, B, probs, originalK, rec


def protect(B, probability, threshold):
    output=B.copy()
    blank=~B.any(axis=1)
    output[blank]=probability[blank]>=threshold
    need(np.array_equal(output[~blank],B[~blank]),
         "S529 must NEVER delete or replace original nonempty BP-note proposals")
    return output


def counts(Y,B,Q,mask):
    truth=Y[mask];prev=B[mask];nxt=Q[mask]
    old=np.all(truth==prev,axis=1);new=np.all(truth==nxt,axis=1)
    tp=int(np.sum(truth&nxt))
    fp=int(np.sum((~truth)&nxt))
    fn=int(np.sum(truth&(~nxt)))
    changes=np.any(prev!=nxt,axis=1)
    return dict(n=int(mask.sum()),BP_birth_exact_chords=int(old.sum()),
        protected_exact_chords=int(new.sum()),
        complete_chord_fixes=int(np.sum(~old&new&changes)),
        complete_chord_regressions=int(np.sum(old&~new&changes)),
        incorrect_to_incorrect_edits=int(np.sum(~old&~new&changes)),
        modified_attack_events=int(changes.sum()),
        original_BP_exactK=int(np.sum(prev.sum(axis=1)==truth.sum(axis=1))),
        protected_exactK=int(np.sum(nxt.sum(axis=1)==truth.sum(axis=1))),
        true_note_pitch_TP=tp,
        false_note_pitch_FP_on_KNOWN_positive_attacks=fp,
        missed_true_pitch_FN=fn,
        precision_CONDITIONAL_on_true_attack=tp/max(1,tp+fp),
        recall_CONDITIONAL_on_true_attack=tp/max(1,tp+fn),
        F1_CONDITIONAL_on_true_attack=2*tp/max(1,2*tp+fp+fn),
        new_correct_pitches_added=int(np.sum((~prev)&nxt&truth)),
        false_extra_pitches_added_on_KNOWN_true_attacks=int(np.sum((~prev)&nxt&(~truth))),
        true_pitches_removed=int(np.sum(prev&(~nxt)&truth)),
        wrong_pitches_removed=int(np.sum(prev&(~nxt)&(~truth))))


def run(a):
    need(not a.out.exists(), "S529 immutable no-training audit already exists")
    original,Y,observed,missing,B,probs,sourceK,rec=frozen_sources(a)
    output = {}
    every_change=[]
    for kind in ("raw_only","with_BP"):
        for threshold in (.50,.70):
            key=kind+"_"+str(threshold)
            Q=protect(B,probs[kind],threshold)
            noteK=Y.sum(axis=1)
            metrics=dict(overall_known_true_attacks=counts(Y,B,Q,observed),
                  original_102_canonical_BP_missing_true_attacks=counts(Y,B,Q,observed&missing),
                  original_220_canonical_BP_matched_true_attacks=counts(Y,B,Q,observed&~missing),
                  by_true_newnote_K={str(k):counts(Y,B,Q,observed&(noteK==k))
                                    for k in sorted(set(noteK[observed]))},
                  per_recording={z:counts(Y,B,Q,observed&(np.asarray(rec)==z))
                                 for z in sorted(set(rec))})
            known=metrics["overall_known_true_attacks"]
            need(known["complete_chord_regressions"]==0
                 and known["true_pitches_removed"]==0
                 and known["wrong_pitches_removed"]==0,
                 "S529 protected birth note selection must preserve all previously correct chords")
            unknown=~observed
            new_unlabeled=(~B.any(axis=1))&Q.any(axis=1)&unknown
            metrics["UNLABELED_candidates_count"] = int(unknown.sum())
            metrics["UNLABELED_new_note_births_potential_false_or_true"] = int(new_unlabeled.sum())
            metrics["UNLABELED_new_pitch_ids_unscored"]=int(np.sum(Q[unknown]&~B[unknown]))
            metrics["UNLABELED_original_K0_proposal_count"]=int(np.sum(new_unlabeled&(sourceK==0)))
            metrics["negative_audio_onsets_are_UNKNOWN_not_groundtruth"]=True
            output[key]=metrics
            print("S529 PROTECT OLD CHORDS AND ADD ONLY ON BLANK BP BIRTHS",
                  json.dumps(dict(policy=key,summary=metrics)),flush=True)
            for i in np.flatnonzero(np.any(B!=Q,axis=1)):
                every_change.append(dict(
                    original_V88_id=original[i]["V88_original_event_id"],
                    original_recording_SHA=rec[i],
                    policy=key,
                    model_onset_is_verified_true=bool(observed[i]) if observed[i] else None,
                    true_NOTE_PITCH_FOR_AUDIT=(np.flatnonzero(Y[i])+40).tolist() if observed[i] else None,
                    original_BP_canonical_missing_FOR_AUDIT=bool(missing[i]) if observed[i] else None,
                    old_BP_birth_notes=(np.flatnonzero(B[i])+40).tolist(),
                    new_named_note_MIDI=(np.flatnonzero(Q[i])+40).tolist(),
                    original_V88_K=int(sourceK[i]),
                    never_use_MIDI_labels_to_choose_this_action=True))
    a.out.mkdir(parents=True)
    changes=a.out/"S529_ALL_PROTECTED_ADDED_PITCHES_AND_UNLABELED_EVENT_PROPOSALS.jsonl"
    changes.write_text("".join(json.dumps(r)+"\n" for r in every_change))
    report=dict(
        status="S529_ORIGINAL_V88_ONSET_K0_PROTECTED_NOTE_PITCH_RESCUE_NO_RETRAIN",
        four_predeclared_BirthOnly_KeepBP_0p5_0p7_policies=True,
        no_new_training_or_test_tuned_thresholds=True,
        all_three_recording_held_S528_pretrained_models_sha_consumed=True,
        preserved_all_nonempty_BP_birth_sets=True,
        original_322_TRUE_annotated_piano_onsets_retained=True,
        original_102_missing_BP_canonical_true_onsets_retained=True,
        original_253_UNLABELED_V88_groups_not_false_event_labels=True,
        original_8_TRUE_onsets_outside_V88_unchanged_missed=True,
        historical_exploratory_cohort_revisited_NOT_prospective_blind=True,
        F1_conditional_positive_attacks_NOT_stream_wide=True,
        source_S80_ExactK_and_Live_unchanged=True,
        NO_PROMOTION=True,
        policies=output,
        source_sha256=dict(S527=sha(a.audio),S528=sha(a.trained),
                           S476=sha(a.truth),S526=sha(a.missing)),
        changed_note_events_ledger_SHA256=sha(changes))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


def selftest():
    truth=np.zeros((2,49),bool);truth[0,12]=True;truth[1,15]=True
    base=np.zeros_like(truth);base[0,12]=True
    prob=np.zeros((2,49),float);prob[0,13]=1.;prob[1,15]=.91
    Q=protect(base,prob,.7)
    r=counts(truth,base,Q,np.array([True,True]))
    need(r["complete_chord_regressions"]==0 and
         r["complete_chord_fixes"]==1 and
         r["protected_exact_chords"]==2 and
         not Q[0,13] and Q[1,15],
         "S529 only infer brand-new notes where BP has ZERO note births")
    print("S529 SELFTEST PASS no regression of preexisting chords even if model invents harmonic")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for name in ("audio","audio-report","trained","trained-report",
                 "truth","truth-report","missing","missing-report","out"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for name in ("audio","audio_report","trained","trained_report",
                     "truth","truth_report","missing","missing_report","out"):
            need(getattr(a,name) is not None,"S529 missing "+name)
        run(a)
