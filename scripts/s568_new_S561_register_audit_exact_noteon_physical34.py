"""S568 independent post-validation register audit; diagnostic ONLY after S567.

Reopen the same SHA-pinned authentic five S561 MAESTRO official train split
MIDI onsets now that original S566 predictions were irreversibly sealed
and S567 scored. Evaluate separately register 40-59,60-71,72-88 under
+/- 30/50/80ms exact MIDI pitch one-to-one, across the same six frozen
policies. Each range's TP/FP/FN sums exactly to original S567 full score.
Measure old-vs-new true pitch recovery and risky physical repeated strikes.

No new threshold, retrained model, test adjustment, or promotion in S568.
Scores in S567 were on never-before-trained source names but cannot
guarantee none were exposed in any historical unrelated repository branch.
"""
from __future__ import annotations
import argparse,json,hashlib,collections
from pathlib import Path
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import fetch_midi,midi_events,onset_groups
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import sha,need
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics
RANGES={"low_40_59":(40,59),"mid_60_71":(60,71),"high_72_88":(72,88)}
POLICIES=("BP_birth","S546_dedup80","S546_dedup100_harm080","S562_raw7_dedup80","S562_raw7_dedup100_harm080","S562_fundamental34_dedup80","S562_fundamental34_dedup100_harm080")
def rows(f):return [json.loads(r) for r in f.read_text().splitlines() if r.strip()]
def part(events,lo,hi):
    return [dict(time=float(z["time"]),notes=[int(p) for p in z["notes"] if lo<=int(p)<=hi])
            for z in events if any(lo<=int(p)<=hi for p in z["notes"])]
def run(a):
    need(not a.out.exists(),"S568 immutable root cause audit report")
    m=json.loads(a.manifest.read_text())
    frozen=json.loads(a.freeze_report.read_text())
    result=json.loads(a.score_report.read_text())
    need(m["status"]=="S561_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and m["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"]
         and frozen["status"]=="S566_FIVE_NEW_S561_SOUNDFROZEN_S546_VS_S562_NOTEON_BEFORE_MIDI"
         and frozen["output_pitch_name_ledger_SHA"]==sha(a.frozen_predictions)
         and result["status"]=="S567_EXTERNAL_POSTFROZEN_FIVE_NEW_COMPOSER_FULL_MIDI_NOTEON_F1"
         and result["S566_all_seven_named_note_policies_SHA_VERIFIED_before_true_MIDI"]
         and result["SHA_frozen_S566_prediction"]==sha(a.frozen_predictions)
         and result["no_model_training_or_threshold_selection_using_S561_MIDI"]
         and set(result["policies"])==set(POLICIES),
         "S568 original independent S567 SHA-frozen test required")
    selected={r["filename_SHA256"]:r for r in m["chosen_metadata_only"]}
    proof={p["original_audio_filename_sha"]:p for p in result["true_MIDI_provenance"]}
    need(set(selected)==set(proof) and len(selected)==5,
         "S568 official MIDI ground truth provenance 5 original musicians")
    forecast=collections.defaultdict(list)
    for r in rows(a.frozen_predictions):
        need(r["MIDI_ANNOTATIONS_NEVER_OPENED"] and r["policy"] in POLICIES,
             "S568 frozen prediction policy cannot use MIDI truth")
        forecast[(r["original_recording_filename_sha"],r["policy"])].append(dict(
            time=float(r["onset_s"]),notes=r["pitch_MIDI"]))
    totals={p:{range_name:{tol:collections.Counter()
             for tol in ("30","50","80")} for range_name in RANGES} for p in POLICIES}
    per_song={}
    for ident,item in sorted(selected.items()):
        midi_blob=fetch_midi(m["source_audio_mirror"],item["MIDI_filename_path_ONLY_not_opened"])
        need(hashlib.sha256(midi_blob).hexdigest()==proof[ident]["original_MIDI_sha256"],
             "S568 original S567 full MIDI source SHA changed")
        events=midi_events(midi_blob)
        start=float(item["audio_excerpt_start_seconds"])
        onset=onset_groups(events,start,float(item["audio_excerpt_duration_seconds"]))
        truth=[dict(time=float(g["onset"]),
             notes=[int(z) for z in new_notes(events,start,float(g["onset"]))
                    if 40<=int(z)<=88]) for g in onset]
        truth=[x for x in truth if x["notes"]]
        need(len(truth)==proof[ident]["scorable_onset_groups"],
             "S568 complete old S567 true attack group count changed")
        song={}
        for name,(lo,hi) in RANGES.items():
            ref=part(truth,lo,hi)
            song[name]=dict(true_MIDI_pitch_onsets=sum(len(x["notes"]) for x in ref),
                            true_chord_groups_with_one_or_more_pitches=len(ref))
            for policy in POLICIES:
                predicted=part(forecast[(ident,policy)],lo,hi)
                for tol in ("30","50","80"):
                    mm=note_metrics(predicted,ref,int(tol)/1000.)
                    totals[policy][name][tol].update(TP=mm["TP"],FP=mm["FP"],FN=mm["FN"])
        per_song[ident]=dict(composer=item["composer"],by_register=song)
    results={}
    for pol in POLICIES:
        results[pol]={}
        for name in RANGES:
            results[pol][name]={}
            for tol in ("30","50","80"):
                r=dict(totals[pol][name][tol])
                r["F1"]=2*r["TP"]/max(1,2*r["TP"]+r["FP"]+r["FN"])
                results[pol][name][tol]=r
        for tol in ("30","50","80"):
            aggregated=collections.Counter()
            for name in RANGES:
                q=results[pol][name][tol]
                aggregated.update(TP=q["TP"],FP=q["FP"],FN=q["FN"])
            source=result["policies"][pol]["complete_pitch_onset_F1"][tol]
            need((aggregated["TP"],aggregated["FP"],aggregated["FN"])
                 ==(source["TP"],source["FP"],source["FN"]),
                 "S568 per MIDI register TP/FP/FN must exactly reproduce all 5 frozen S567 scores")
        print("S568 EXACT PITCH NEW COMPOSER REGISTER CAUSAL AUDIT",
              json.dumps(dict(policy=pol,registers={q:results[pol][q]["80"] for q in RANGES})),flush=True)
    gains={}
    for name in RANGES:
        old=results["S546_dedup100_harm080"][name]["80"]
        new=results["S562_fundamental34_dedup100_harm080"][name]["80"]
        final=results["S562_raw7_dedup100_harm080"][name]["80"]
        gains[name]=dict(true_MIDI_notes=old["TP"]+old["FN"],
            improvement_S562_physical34_vs_S546_100ms_harmonic=dict(
                deltaTP=new["TP"]-old["TP"],deltaFP=new["FP"]-old["FP"],
                deltaFN=new["FN"]-old["FN"],
                delta_F1_points=100*(new["F1"]-old["F1"])),
            S562_raw7_vs_physical34_same100ms=dict(
                deltaTP=final["TP"]-new["TP"],deltaFP=final["FP"]-new["FP"],
                deltaFN=final["FN"]-new["FN"],
                delta_F1_points=100*(final["F1"]-new["F1"])))
    a.out.mkdir(parents=True)
    report=dict(status="S568_TRUE_INDEPENDENT_REGISTER_PITCH_NOTEON_CAUSES_POST_S567",
       original_real_MAESTRO_MIDI_SHA_verification_for_all_five=True,
       all_register_counts_S567_independently_replayed_for_every_policy_tolerance=True,
       three_registers_midi40to88=list(RANGES),
       measurements_are_POSTHOC_NOT_new_prospective_validation=True,
       MIDI_truth_did_not_change_any_frozen_prediction=True,
       cannot_promote_original_S80_or_ExactK_or_Live=True,
       no_promotion=True,
       original_S567_F1_report_SHA=sha(a.score_report),
       original_frozen_S566_pitch_event_SHA=sha(a.frozen_predictions),
       full_register_scores=results,per_original_piano_song=per_song,
       register_progress_and_remaining_errors=gains)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S568 FINAL VALIDATED REGISTER GAPS FOR NEXT SOURCE DESIGNED RESEARCH",
          json.dumps(gains),flush=True)
def selftest():
    e=[dict(time=.1,notes=[52,65,84]),dict(time=.2,notes=[65])]
    need(sum(sum(len(z["notes"]) for z in part(e,*r)) for r in RANGES.values())==4,
         "S568 register attribution of each true pitch note event exactly once")
    print("S568 SELFTEST PASS exact one-register-per-pitch and repeated MIDI notes")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("manifest","freeze-report","score-report","frozen-predictions","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("manifest","freeze_report","score_report","frozen_predictions","out"):
            need(getattr(a,k) is not None,"S568 missing "+k)
        run(a)
