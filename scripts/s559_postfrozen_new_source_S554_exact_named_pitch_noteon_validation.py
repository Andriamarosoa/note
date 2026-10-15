"""S559 post-frozen NoteOn pitch+attack F1 over 5 source-disjoint S554 songs.

GUARD: S558 predictions and all 11 old models must be SHA-sealed on S554
real WAV before this script can retrieve a byte of the corresponding
unseen MAESTRO TRAIN-split MIDI file. Six fixed policies; one-to-one same
pitch onset events +/-30, 50, 80 ms; full unmatched predictions FP,
full uncovered true notes FN. Compare S528 vs new S546 within same
source. No adaptation, threshold change, retraining or promotion.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import fetch_midi,midi_events,onset_groups
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need,sha,score_event_birth
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics,repeat_truth_intervals

POLICIES=("BP_birth","S528_raw","S528_dedup80","S546_raw","S546_dedup80","S546_dedup100_harm080")
def readrows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def run(a):
    need(not a.out.exists(),"S559 postlabel audit cannot overwrite")
    meta=json.loads(a.manifest.read_text())
    audio=json.loads(a.audio_report.read_text())
    feat=json.loads(a.feature_report.read_text())
    freeze=json.loads(a.frozen_report.read_text())
    need(meta["status"]=="S554_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and meta["five_next_distinct_composers"]==5
         and meta["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"]
         and meta["source_MIDI_files_not_read"]
         and meta["source_WAV_files_not_read"]
         and audio["status"]=="S555_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and audio["preregistered_S554_SHA"]==sha(a.manifest)
         and not audio["paired_MIDI_annotation_files_OPENED"]
         and feat["status"]=="S557_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
         and feat["no_original_true_MIDI_annotation_read"]
         and freeze["status"]=="S558_FIVE_COMPOSER_S554_PITCH_ONSET_SOURCE_FROZEN_BEFORE_ANY_MIDI"
         and freeze["original_MIDI_NOT_opened_for_predictions"]
         and freeze["frozen_named_note_prediction_sha"]==sha(a.predictions)
         and freeze["all_six_policies_declared_before_labels"]==list(POLICIES)
         and freeze["source_sha"]["S554"]==sha(a.manifest)
         and freeze["source_sha"]["S555"]==sha(a.audio_report)
         and freeze["source_sha"]["S557"]==sha(a.features)
         and feat["frozen_feature_ledger_SHA256"]==sha(a.features),
         "S559 original fully frozen S554 new song note-on predictions must be SHA proven BEFORE opening target MIDI")
    chosen={x["filename_SHA256"]:x for x in meta["chosen_metadata_only"]}
    original={x["WAV_filename_SHA256"]:x
              for x in audio["exact_original_audio_source_fingerprints"]}
    need(set(chosen)==set(original) and len(chosen)==5,
         "S559 original five source audio fingerprints mismatch preregistration")
    predicted=readrows(a.predictions)
    by=collections.defaultdict(list)
    for p in predicted:
        need(p["policy"] in POLICIES and p["MIDI_ANNOTATIONS_NEVER_OPENED"]
             and p["original_recording_filename_sha"] in chosen,
             "S559 original policies/notes must come exclusively from frozen five S554 WAVs")
        by[(p["policy"],p["original_recording_filename_sha"])].append(
              dict(time=float(p["onset_s"]),notes=sorted(set(map(int,p["pitch_MIDI"])))))
    for policy in POLICIES:
        record_map={x["source_sha"]:x for x in freeze["predictions_by_policy"][policy]["per_source"]}
        for ident in chosen:
            need(len(by[(policy,ident)])==record_map[ident]["emitted_attack_groups"],
                 "S559 no rewritten/deleted pre-MIDI policy attack events may be scored")
    # SOURCE MIDI EMBARGO ENDS BELOW: validated all upstream SHA proofs.
    truths={}
    proof=[]
    for ident,item in sorted(chosen.items()):
        raw=fetch_midi(meta["source_audio_mirror"],item["MIDI_filename_path_ONLY_not_opened"])
        midi=midi_events(raw)
        start=float(item["audio_excerpt_start_seconds"])
        groups=onset_groups(midi,start,float(item["audio_excerpt_duration_seconds"]))
        targets=[dict(time=float(g["onset"]),
             notes=[int(p) for p in new_notes(midi,start,float(g["onset"]))
                    if 40<=int(p)<=88]) for g in groups]
        missed_outside=sum(sum(int(p<40 or p>88) for p in new_notes(midi,start,float(g["onset"])))
                            for g in groups)
        truth=[x for x in targets if x["notes"]]
        need(truth and len(groups)>0,"S559 valid true MIDI40..88 notes required")
        truths[ident]=truth
        record=dict(original_audio_filename_sha=ident,
            composer=item["composer"],official_MAESTRO_split="train",
            original_MIDI_sha256=hashlib.sha256(raw).hexdigest(),
            original_MIDI_onset_groups=len(groups),
            scorable_onset_groups=len(truth),
            scorable_true_MIDI_note_pitches=sum(len(x["notes"]) for x in truth),
            true_pitch_events_below_above_MIDI40_88=missed_outside,
            original_MIDI_same_pitch_restrikes_under80ms=len(repeat_truth_intervals(truth,.080)),
            original_MIDI_same_pitch_restrikes_under100ms=len(repeat_truth_intervals(truth,.100)))
        proof.append(record)
        print("S559 NEW COMPOSER REAL MAESTRO TRAIN MIDI OPENED ONLY AFTER S558 SHA",
              json.dumps(record),flush=True)
    results={}
    for policy in POLICIES:
        agg=collections.defaultdict(collections.Counter)
        composer={}
        for ident,truth in sorted(truths.items()):
            events=sorted(by[(policy,ident)],key=lambda r:r["time"])
            metrics={}
            for tol in (.03,.05,.08):
                key=str(int(tol*1000))
                outcome=note_metrics(events,truth,tol)
                agg[key].update(TP=outcome["TP"],FP=outcome["FP"],FN=outcome["FN"])
                metrics[key]=outcome
            chord=score_event_birth(events,truth)
            composer[ident]=dict(pitch_F1_by_tolerance_ms=metrics,
                               event_chord_audit_80ms=chord,
                               total_predicted_pitch_events=sum(len(e["notes"]) for e in events))
        total={}
        for tol,c in agg.items():
            t=dict(c)
            t["precision"]=t["TP"]/max(1,t["TP"]+t["FP"])
            t["recall"]=t["TP"]/max(1,t["TP"]+t["FN"])
            t["F1"]=2*t["TP"]/max(1,2*t["TP"]+t["FP"]+t["FN"])
            total[tol]=t
        results[policy]=dict(complete_pitch_onset_F1=total,per_new_composer=composer)
        print("S559 TRUE EXTERNAL NOTEON FULL STREAM F1 AFTER FROZEN SONG PRELABEL",
             json.dumps(dict(policy=policy,metrics=total)),flush=True)
    comparisons={}
    for variant in ("raw","dedup80"):
        a0=results["S528_"+variant]["complete_pitch_onset_F1"]["80"]
        b0=results["S546_"+variant]["complete_pitch_onset_F1"]["80"]
        comparisons[variant]=dict(new_S546_minus_original_S528_F1_percentage_points=100*(b0["F1"]-a0["F1"]),
           delta_TP=b0["TP"]-a0["TP"],delta_FP=b0["FP"]-a0["FP"],delta_FN=b0["FN"]-a0["FN"],
           raw_F1_S528=a0["F1"],new_F1_S546=b0["F1"],
           individual_composer_deltas={
                chosen[ident]["composer"]:100*(results["S546_"+variant]["per_new_composer"][ident]
                                  ["pitch_F1_by_tolerance_ms"]["80"]["F1"] -
                                  results["S528_"+variant]["per_new_composer"][ident]
                                  ["pitch_F1_by_tolerance_ms"]["80"]["F1"])
                for ident in sorted(chosen)})
    a.out.mkdir(parents=True)
    report=dict(status="S559_EXTERNAL_POSTFROZEN_FIVE_NEW_COMPOSER_FULL_MIDI_NOTEON_F1",
      S554_meta_selected_before_ANY_test_WAV_or_MIDI=True,
      S555_S548_S557_audio_frozen_before_MIDI=True,
      S558_all_six_named_note_policies_SHA_VERIFIED_before_true_MIDI=True,
      all_five_original_MIDI_used_ONLY_in_S559_scoring=True,
      no_model_training_or_threshold_selection_using_S554_MIDI=True,
      MAESTRO_original_split="train",NOT_official_MAESTRO_test=True,
      composers_disjoint_from_eight_previous_training_composer_components_by_METADATA=True,
      whole_repo_historical_exposure_not_fully_audited=True,
      matching_pitch_exact_and_time_tolerance_ms=[30,50,80],
      full_false_positive_note_birth_events_counted=True,
      supported_note_pitches_MIDI_40_to_88_only=True,
      genuine_source_original_note_onsets=sum(x["scorable_onset_groups"] for x in proof),
      genuine_true_pitch_notes=sum(x["scorable_true_MIDI_note_pitches"] for x in proof),
      no_reference_S80_ExactK_freeze_local_combo_live_changes=True,no_promotion=True,
      SHA_frozen_S558_prediction=sha(a.predictions),
      true_MIDI_provenance=proof,policies=results,paired_comparisons=comparisons)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S559 FINAL NO PROMOTION INDEPENDENT NEW COMPOSER SCORE",json.dumps(
        dict(source_count=5,midi_true_note_pitches=report["genuine_true_pitch_notes"],
             paired=comparisons)),flush=True)
def selftest():
    truth=[dict(time=.1,notes=[52,55]),dict(time=.2,notes=[52])]
    guess=[dict(time=.1,notes=[52,55]),dict(time=.2,notes=[52])]
    need(note_metrics(guess,truth,.03)["TP"]==3 and
         len(POLICIES)==6,"S559 full note and time one-to-one real retransmission scoring")
    print("S559 SELFTEST PASS pitch + time ±30ms and six immutable policies")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","audio-report","feature-report","features","frozen-report","predictions","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","audio_report","feature_report","features","frozen_report","predictions","out"):
            need(getattr(a,z) is not None,"S559 missing "+z)
        run(a)
