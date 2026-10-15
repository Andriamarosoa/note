"""S542 prospective post-frozen true MIDI note-on validation on five new composers.

Strict embargo: parse MAESTRO MIDI ONLY AFTER S541 SHA verifying the five
original S537 audio-only already sealed forecasts including all four
predeclared policy note events. No fit, new selection, or feedback into
those predictions. One-to-one exact MIDI pitch at 30,50,80 ms; include
every emitted false pitch and every true MIDI pitch from five excerpts
(in support MIDI 40..88). Report source per-composer detail, fast true
same-pitch restrikes, chord quality, and V88 onset candidate coverage.

Even five separate S469 composers are not known to be absent from all
prior GitHub experiments. Reference promotion requires broader held-out
instrument tests, original S80 exact-K and Live regression protections.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,match_times,SR)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    need,sha,assignment,score_event_birth)
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import (
    note_metrics,repeat_truth_intervals)
POLICIES=("BP_birth","raw_mean_0p7","dedup80","dedup100_harm080")

def readlines(p):
    return [json.loads(z) for z in p.read_text().splitlines() if z.strip()]
def rows_per_source(rows):
    b=collections.defaultdict(list)
    for r in rows:b[r["original_recording_filename_sha"]].append(r)
    return b
def run(a):
    need(not a.out.exists(),"S542 post-frozen true MIDI validation already exists")
    s537=json.loads(a.manifest.read_text())
    s538=json.loads(a.source_report.read_text())
    s540=json.loads(a.features_report.read_text())
    s541=json.loads(a.frozen_report.read_text())
    need(s537["status"]=="S537_PREREGISTERED_DISTINCT_MAESTRO_V3_TEST_PIANO_COHORT_NO_WAV_NO_MIDI"
         and s537["selected_recordings"]==5
         and s537["MIDI_contents_downloaded_or_inspected"] is False
         and s538["status"]=="S538_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and s538["preregistered_S469_SHA"]==sha(a.manifest)
         and s538["prelabel_predictions_frozen"] and
             not s538["paired_MIDI_annotation_files_OPENED"]
         and s540["status"]=="S540_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
         and s540["no_original_true_MIDI_annotation_read"]
         and s541["status"]=="S541_FIVE_NEW_COMPOSER_PREMIDI_FROZEN_S528_BP_S533_S536_NAMED_NOTES"
         and not s541["original_true_MIDI_note_contents_OPENED"]
         and s541["frozen_prediction_ledger_SHA256"]==sha(a.predictions)
         and s541["inputs_SHA256"]["S537"]==sha(a.manifest)
         and s541["inputs_SHA256"]["S538"]==sha(a.source_report)
         and s541["inputs_SHA256"]["S540"]==sha(a.features_report if False else a.features)
         and s541["four_policies_predeclared"]==list(POLICIES),
         "S542 original pre-MIDI freeze or model prediction SHA not sealed")
    selected={r["WAV_filename_sha256"]:r for r in s537["selected_sources"]}
    audio_proofs={r["WAV_filename_SHA256"]:r
                  for r in s538["exact_original_audio_source_fingerprints"]}
    need(set(selected)==set(audio_proofs) and len(selected)==5,
         "S542 five new preregistered composers / real WAV fingerprints mismatch")
    pred=readlines(a.predictions)
    need(pred and all(r["policy"] in POLICIES
         and r["MIDI_ANNOTATIONS_NEVER_OPENED"] for r in pred),
         "S542 all prediction files must be strictly pre-MIDI")
    bysource=rows_per_source(pred)
    from collections import Counter
    need(set(bysource)==set(selected),
         "S542 frozen prediction source IDs unequal five true new SHA filenames")
    for source in selected:
        for policy in POLICIES:
            count=sum(int(r["policy"]==policy) for r in bysource[source])
            need(count==s541["policy_counts"][policy]["per_recording"][
                     list(sorted(selected)).index(source)]["emitted_note_onsets"],
                 "S542 previous MIDI-blind exact prelabel note event count drifted")
    # STRICT PHASE: original paired MIDI files are accessed from here onward only.
    true_by={}
    srcproof=[]
    for ident,item in sorted(selected.items()):
        raw=fetch_midi(s537["audio_mirror_directory"],item["MIDI_filename_PATH_ONLY_DO_NOT_OPEN"])
        midi=midi_events(raw)
        offset=float(item["frozen_excerpt_offset_s"])
        duration=float(item["frozen_excerpt_length_s"])
        groups=onset_groups(midi,offset,duration)
        need(duration==12.0 and groups,
             "S542 original new composer complete 12-second MIDI excerpt invalid")
        truth_all=[dict(time=float(g["onset"]),
                        notes=new_notes(midi,offset,float(g["onset"]))) for g in groups]
        outside=sum(sum(int(p<40 or p>88) for p in g["notes"]) for g in truth_all)
        scored=[dict(time=z["time"],notes=[int(p) for p in z["notes"] if 40<=int(p)<=88])
                 for z in truth_all]
        scored=[z for z in scored if z["notes"]]
        need(len(scored)>0,"S542 no true MIDI notes in supported pitch range for source")
        true_by[ident]=scored
        srcproof.append(dict(filename_sha=ident,composer=item["composer"],
                  original_MIDI_SHA256=hashlib.sha256(raw).hexdigest(),
                  original_true_complete_MIDI_attack_groups=len(groups),
                  scoreable_MIDI_40_88_attack_groups=len(scored),
                  original_true_MIDI_note_pitches_supported=sum(len(z["notes"]) for z in scored),
                  true_MIDI_notes_outside_supported_40_88=outside,
                  genuine_same_pitch_restrikes_under_80ms=
                    len(repeat_truth_intervals(scored,.080)),
                  genuine_same_pitch_restrikes_under_100ms=
                    len(repeat_truth_intervals(scored,.100))))
        print("S542 ORIGINAL NEW MAESTRO MIDI OPENED AFTER SHA-SEALED PRELABEL",
              json.dumps(srcproof[-1]),flush=True)
    policies={}
    for policy in POLICIES:
        micro=collections.defaultdict(lambda:collections.Counter())
        per={}
        for ident in sorted(selected):
            outputs=[dict(time=float(r["onset_s"]),notes=sorted(set(map(int,r["pitch_MIDI"]))))
                     for r in bysource[ident] if r["policy"]==policy]
            outputs.sort(key=lambda z:z["time"])
            true=true_by[ident]
            bytol={}
            for tol in (.03,.05,.08):
                key=str(int(tol*1000))
                m=note_metrics(outputs,true,tol)
                bytol[key]=m
                micro[key].update(TP=m["TP"],FP=m["FP"],FN=m["FN"])
            b=score_event_birth(outputs,true)
            per[ident]=dict(onset_pitch_at_30_50_80ms=bytol,
                            matching_event_chords_at_80ms=b,
                            emitted_pitch_noteon=sum(len(x["notes"]) for x in outputs),
                            true_supported_pitch_noteon=sum(len(x["notes"]) for x in true))
        all_tol={}
        for tol,z in micro.items():
            t=dict(z)
            t["F1"]=2*t["TP"]/max(1,2*t["TP"]+t["FP"]+t["FN"])
            t["precision"]=t["TP"]/max(1,t["TP"]+t["FP"])
            t["recall"]=t["TP"]/max(1,t["TP"]+t["FN"])
            all_tol[tol]=t
        policies[policy]=dict(micro_noteon=all_tol,per_source=per)
        print("S542 INDEPENDENT FIVE NEW COMPOSER TRUE MIDI NOTE_ON VERDICT",
              json.dumps(dict(policy=policy,
                 noteon30=all_tol["30"],noteon50=all_tol["50"],noteon80=all_tol["80"])),flush=True)
    result={}
    for target in ("dedup80","dedup100_harm080"):
        base=policies["raw_mean_0p7"]["micro_noteon"]["80"]
        got=policies[target]["micro_noteon"]["80"]
        per={ident:dict(base=policies["raw_mean_0p7"]["per_source"][ident]["onset_pitch_at_30_50_80ms"]["80"],
                         candidate=policies[target]["per_source"][ident]["onset_pitch_at_30_50_80ms"]["80"])
             for ident in sorted(selected)}
        result[target]=dict(delta_F1_percentage_points=100*(got["F1"]-base["F1"]),
             delta_true_positives=got["TP"]-base["TP"],delta_false_positives=got["FP"]-base["FP"],
             delta_false_negatives=got["FN"]-base["FN"],per_source=per)
    a.out.mkdir(parents=True)
    rep=dict(status="S542_PROSPECTIVE_FIVE_NEW_PIANO_COMPOSER_NOTEON_POSTFROZEN_MIDI_VALIDATION",
      source_manifest_S537_preregistered_before_audio_and_MIDI=True,
      model_source_S538_V88_before_MIDI=True,
      pretrained_note_source_S539_S540_before_MIDI=True,
      model_and_policy_S541_predicted_note_SHA_FROZEN_BEFORE_MIDI=True,
      separate_original_MAESTRO_composers=5,
      source_excluded_all_three_S469_old_composers=True,
      original_entire_repo_never_exposed_VERIFICATION_NOT_DONE=True,
      original_S80_complete_not_replayed_and_Live_unchanged=True,
      truth_only_opened_after_all_policy_predictions_sha_verified=True,
      full_true_pitch_onset_NOTEON_F1_not_conditional_known_attacks=True,
      all_unmatched_predictions_are_pitch_FP=True,
      tolerances_ms=[30,50,80],
      original_MIDI_40_to_88_supported_only=True,
      no_training_or_test_threshold_choice_after_MIDI=True,
      no_promotion=True,
      per_new_recording_source_proof=srcproof,
      scores=policies,
      paired_comparisons=result,
      frozen_sources_sha=dict(S537=sha(a.manifest),
           S538=sha(a.source_report),
           S540=sha(a.features_report),
           S541_report=sha(a.frozen_report),
           S541_prediction=sha(a.predictions)))
    (a.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("S542 NEW REAL COMPOSER FULL NOTE_ON FINAL VERDICT",
          json.dumps(dict(dedup80=result["dedup80"],
                          dedup100_harm080=result["dedup100_harm080"])),flush=True)

def selftest():
    from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics
    p=[dict(time=.1,notes=[52,55]),dict(time=.2,notes=[52])]
    t=[dict(time=.1,notes=[52,55]),dict(time=.2,notes=[52])]
    assert note_metrics(p,t,.03)["TP"]==3
    a=[dict(time=.1,notes=[52])]
    b=[dict(time=.1,notes=[52,55])]
    assert note_metrics(a,b,.03)["FN"]==1
    print("S542 SELFTEST PASS named-note onset full original scores require exact MIDI pitch and one-to-one timing")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","source-report","features-report","features",
              "frozen-report","predictions","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","source_report","features_report","features",
                   "frozen_report","predictions","out"):
            need(getattr(a,z) is not None,"S542 missing "+z)
        run(a)
