"""S583 score SHA-frozen fifth new composer NoteOn only AFTER MIDI embargo lifted.

Four S582 predeclared source-trained policies: S569 old23, S578 general28,
S578 specialist replacing 12 medium pitch probabilities, and 50:50 blend.
All prediction note onsets on S577 real WAV are frozen in earlier successful
S582 GitHub artifact and SHA verified before THIS SCRIPT fetches any paired
MIDI. Test identical exact pitch names / onset ±30,50,80ms, one-to-one
across all V88 K0 candidates and every missed true ground truth onset.
Do not adjust threshold, architecture, retrain, or promote S80 in this job.
"""
from __future__ import annotations
import argparse,json,collections,hashlib
from pathlib import Path
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import fetch_midi,midi_events,onset_groups
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need,sha,score_event_birth
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics,repeat_truth_intervals

POLICIES=("S569_old23_control","S578_general28","S578_mid_replace","S578_mid_blend50")
RANGES={"low_40_59":(40,59),"mid_60_71":(60,71),"high_72_88":(72,88)}
def readrows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def part(seq,lo,hi):
    return [dict(time=float(g["time"]),notes=[int(n) for n in g["notes"] if lo<=int(n)<=hi])
            for g in seq if any(lo<=int(n)<=hi for n in g["notes"])]
def run(a):
    need(not a.out.exists(),"S583 immutable fifth real ground truth audit cannot overwrite")
    m=json.loads(a.manifest.read_text())
    wav=json.loads(a.wav_report.read_text())
    feat=json.loads(a.feature_report.read_text())
    freeze=json.loads(a.freeze_report.read_text())
    checks={
       "S577 original prereg metadata only":m["status"]=="S577_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI",
       "28 previous composers excluded":m["original_previous_S469_3_S537_5_S545_5_S554_5_S561_5_S570_5_components_excluded"],
       "all 5 distinct co-composers":m["five_next_distinct_composers"]==5 and m["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"],
       "original source pre-exposure":m["source_MIDI_files_not_read"] and m["source_WAV_files_not_read"],
       "S579 original genuine audio":wav["status"]=="S579_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
           and wav["preregistered_S577_SHA"]==sha(a.manifest)
           and not wav["paired_MIDI_annotation_files_OPENED"],
       "S581 original WAV notes":feat["status"]=="S581_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
           and feat["frozen_feature_ledger_SHA256"]==sha(a.features)
           and feat["no_original_true_MIDI_annotation_read"],
       "S582 four policy SHA seal":freeze["status"]=="S582_FIFTH_COMPOSER_NEW_MIDDLE_HEAD_FOUR_POLICIES_FROZEN_BEFORE_MIDI"
           and freeze["source_MIDI_NOT_read_before_frozen_named_note_event_sha"]
           and freeze["frozen_named_noteon_SHA"]==sha(a.predictions)
           and freeze["four_predeclared_old_vs_new_same_song_policies"]==list(POLICIES)
           and freeze["source_shas"]["S577"]==sha(a.manifest)
           and freeze["source_shas"]["S579"]==sha(a.wav_report)
           and freeze["source_shas"]["S581"]==sha(a.features),
    }
    if not all(checks.values()):
        print("S583 MIDI EMBARGO GUARDS FAILED",json.dumps({k:v for k,v in checks.items() if not v}),flush=True)
    need(all(checks.values()),"S583 only already frozen SHA sealed original audio pitch forecasts eligible for MIDI")
    chosen={x["filename_SHA256"]:x for x in m["chosen_metadata_only"]}
    audio_hash={x["WAV_filename_SHA256"]:x for x in wav["exact_original_audio_source_fingerprints"]}
    need(len(chosen)==5 and set(chosen)==set(audio_hash),
         "S583 original S579 WAV/file source SHA cannot differ from preregistration")
    note_ledger=readrows(a.predictions)
    by=collections.defaultdict(list)
    for row in note_ledger:
        need(row["policy"] in POLICIES and row["MIDI_ANNOTATIONS_NEVER_OPENED"]
             and row["original_recording_filename_sha"] in chosen,
             "S583 no predictions from future knowledge or non-preregistered sources")
        by[(row["policy"],row["original_recording_filename_sha"])].append(
             dict(time=float(row["onset_s"]),notes=sorted(set(map(int,row["pitch_MIDI"])))))
    for policy in POLICIES:
        frozen_count={x["source_sha"]:x for x in freeze["policy_counts"][policy]["per_source"]}
        for k in chosen:
            need(len(by[(policy,k)])==frozen_count[k]["emitted_attack_groups"],
                 "S583 cannot drop/alter one original predicted false pitch event before scoring")
    # NO TARGET MIDI FETCH PRIOR TO ALL GUARDS ABOVE.
    original_truth={};proof=[]
    for ident,item in sorted(chosen.items()):
        blob=fetch_midi(m["source_audio_mirror"],item["MIDI_filename_path_ONLY_not_opened"])
        midi=midi_events(blob)
        excerpt=float(item["audio_excerpt_start_seconds"])
        groups=onset_groups(midi,excerpt,float(item["audio_excerpt_duration_seconds"]))
        note_groups=[dict(time=float(g["onset"]),
            notes=[int(p) for p in new_notes(midi,excerpt,float(g["onset"])) if 40<=int(p)<=88])
            for g in groups]
        gt=[x for x in note_groups if x["notes"]]
        need(gt,"S583 one recorded piano performance needs genuine MIDI ground truth")
        original_truth[ident]=gt
        proof.append(dict(original_audio_filename_sha=ident,composer=item["composer"],
           official_MAESTRO_split="train",original_MIDI_sha256=hashlib.sha256(blob).hexdigest(),
           original_MIDI_onset_groups=len(groups),
           scorable_onset_groups=len(gt),
           scorable_true_MIDI_note_pitches=sum(len(z["notes"]) for z in gt),
           true_fast_restrikes_under80ms=len(repeat_truth_intervals(gt,.080)),
           true_fast_restrikes_under100ms=len(repeat_truth_intervals(gt,.100))))
        print("S583 NEW MUSIC ORIGINAL MIDI OPENED AFTER FOUR FROZEN POLICIES",
              json.dumps(proof[-1]),flush=True)
    totals={}
    for policy in POLICIES:
        aggregate={str(x):collections.Counter() for x in (30,50,80)}
        registers={r:collections.Counter() for r in RANGES}
        per_song={}
        for ident,gt in sorted(original_truth.items()):
            guess=sorted(by[(policy,ident)],key=lambda z:z["time"])
            times={str(t):note_metrics(guess,gt,t/1000.) for t in (30,50,80)}
            for t,stats in times.items():aggregate[t].update({k:stats[k] for k in ("TP","FP","FN")})
            prs={}
            for name,(lo,hi) in RANGES.items():
                s=note_metrics(part(guess,lo,hi),part(gt,lo,hi),.08)
                registers[name].update({k:s[k] for k in ("TP","FP","FN")})
                prs[name]=s
            per_song[ident]=dict(composer=chosen[ident]["composer"],scores=times,
                                 registers=prs,chord_accuracy=score_event_birth(guess,gt))
        total={}
        for time,c in aggregate.items():
            z=dict(c);z["F1"]=2*z["TP"]/max(1,2*z["TP"]+z["FP"]+z["FN"])
            z["precision"]=z["TP"]/max(1,z["TP"]+z["FP"])
            z["recall"]=z["TP"]/max(1,z["TP"]+z["FN"])
            total[time]=z
        rr={}
        for name,c in registers.items():
            z=dict(c);z["F1"]=2*z["TP"]/max(1,2*z["TP"]+z["FP"]+z["FN"])
            rr[name]=z
        need(all(sum(z[k] for z in rr.values())==total["80"][k] for k in ("TP","FP","FN")),
             "S583 all 49 source MIDI pitch counts must exactly sum across 3 music registers")
        totals[policy]=dict(fullstream=total,by_register=rr,per_song=per_song)
        print("S583 TRUE NEXT MUSIC INDEPENDENT FULLSTREAM MIDI NOTEON RESULTS",
              json.dumps(dict(policy=policy,full80=total["80"],mid=rr["mid_60_71"])),flush=True)
    comparisons={}
    for policy in POLICIES[1:]:
        original=totals["S578_general28"]["fullstream"]["80"]
        curr=totals[policy]["fullstream"]["80"]
        comparisons[policy]=dict(deltaF1_points=100*(curr["F1"]-original["F1"]),
            deltaTP=curr["TP"]-original["TP"],deltaFP=curr["FP"]-original["FP"],
            deltaFN=curr["FN"]-original["FN"],
            mid_deltaF1_points=100*(totals[policy]["by_register"]["mid_60_71"]["F1"]-
                                    totals["S578_general28"]["by_register"]["mid_60_71"]["F1"]))
    a.out.mkdir(parents=True)
    report=dict(status="S583_REAL_FIFTH_MAESTRO_SOURCE_HELD_MIDDLE_SPECIALIST_NOTEON_F1",
       source_MIDI_opened_ONLY_after_S582_audio_model_notes_frozen_SHA=True,
       original_music_source_count=5,
       new_28_source_general_and_mid_specialist_weights_untrained_on_this_audio=True,
       true_genuine_MIDI_note_count=sum(x["scorable_true_MIDI_note_pitches"] for x in proof),
       metric_complete_exact_MIDI_pitch_and_onset_tolerances_ms=[30,50,80],
       three_pitch_registers=list(RANGES),
       original_BasicPitch_births_always_preserved=True,
       official_MAESTRO_original_split="train",
       NOT_official_MAESTRO_test=True,
       no_reference_S80_Live_GuitarSet_regression_was_performed=True if False else False,
       no_promotion=True,
       frozen_predictions_SHA=sha(a.predictions),original_S582_sha=sha(a.freeze_report),
       true_MIDI_provenance=proof,policies=totals,
       paired_comparisons_against_same_songs_global28=comparisons)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    gt=[dict(time=.1,notes=[60,64]),dict(time=.16,notes=[60])]
    out=[dict(time=.1,notes=[60,64]),dict(time=.16,notes=[60])]
    need(note_metrics(out,gt,.03)["TP"]==3 and len(POLICIES)==4,
         "S583 must count genuine MIDI fast same-pitch reattack as separate true NoteOn")
    need(sum(len(part(gt,*r)) for r in RANGES.values())==len(gt),
         "S583 pure-mid example has one unique register per note")
    print("S583 SELFTEST PASS genuine repeated MIDI note and exact pitch + onset matcher")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("manifest","wav-report","feature-report","features",
              "freeze-report","predictions","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("manifest","wav_report","feature_report","features",
                  "freeze_report","predictions","out"):
            need(getattr(a,k)is not None,"S583 missing "+k)
        run(a)
