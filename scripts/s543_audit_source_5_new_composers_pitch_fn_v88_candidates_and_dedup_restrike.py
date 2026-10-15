"""S543 forensic independent five-new-song MIDI recall bottleneck audit.

Inputs are SHA-locked S537 manifest, S538 original V88 pre-label NPZ, S541
policy ledger, S542 independent postlabel results. No model training or
threshold search. Reopen MIDI only AFTER S541 validation and measure:
 - how many MIDI attack groups V88 could ever reach;
 - missing true MIDI pitches WITH/WITHOUT V88 candidate nearby;
 - S533 dedup note deletes that turn originally correct MIDI pitches into FN;
 - new composer missing onsets by original V88 predicted K (including K0);
 - genuine under100ms same pitch MIDI re-strikes.

Deductions are diagnostic, NEVER post-select a decision on scored data.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,match_times,SR)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    need,sha,assignment)
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import (
    repeat_truth_intervals)
def lines(f):
    return [json.loads(z) for z in f.read_text().splitlines() if z.strip()]
def truth_pitch_keys(events):
    out=collections.defaultdict(list)
    for e in events:
        for pitch in e["notes"]:out[int(pitch)].append(float(e["time"]))
    return out
def matched_truth_pitch_keys(pred,truth):
    a=truth_pitch_keys(pred);b=truth_pitch_keys(truth)
    matched=set()
    for pitch in sorted(set(a)|set(b)):
        pairs=assignment(a[pitch],b[pitch],.08)
        for i,j in pairs:matched.add((pitch,j))
    # j is indexed within b[pitch] separate by pitch
    return matched
def run(a):
    need(not a.out.exists(),"S543 postlabel forensics immutable")
    man=json.loads(a.manifest.read_text())
    v88=json.loads(a.v88_report.read_text())
    frozen=json.loads(a.freeze_report.read_text())
    scored=json.loads(a.truth_report.read_text())
    need(man["status"]=="S537_PREREGISTERED_DISTINCT_MAESTRO_V3_TEST_PIANO_COHORT_NO_WAV_NO_MIDI"
         and v88["status"]=="S538_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and v88["output_sha256"]==sha(a.v88)
         and frozen["status"]=="S541_FIVE_NEW_COMPOSER_PREMIDI_FROZEN_S528_BP_S533_S536_NAMED_NOTES"
         and frozen["frozen_prediction_ledger_SHA256"]==sha(a.predictions)
         and scored["status"]=="S542_PROSPECTIVE_FIVE_NEW_PIANO_COMPOSER_NOTEON_POSTFROZEN_MIDI_VALIDATION"
         and scored["frozen_sources_sha"]["S541_prediction"]==sha(a.predictions)
         and scored["truth_only_opened_after_all_policy_predictions_sha_verified"],
         "S543 cannot diagnose unless independent S542 AFTER original SHA-frozen MIDI proof")
    with np.load(a.v88,allow_pickle=False) as z:
        member=z["member"].astype(str)
        starts=z["start_samples"].astype(int)/SR
        kval=z["original_V88_K"].astype(int)
    pred=lines(a.predictions)
    perpred=collections.defaultdict(list)
    for z in pred:perpred[(z["original_recording_filename_sha"],z["policy"])].append(
        dict(time=float(z["onset_s"]),notes=z["pitch_MIDI"]))
    out={}
    allcounts=collections.Counter()
    for item in man["selected_sources"]:
        ident=item["WAV_filename_sha256"]
        raw=fetch_midi(man["audio_mirror_directory"],item["MIDI_filename_PATH_ONLY_DO_NOT_OPEN"])
        sha_m=hashlib.sha256(raw).hexdigest()
        expected=next(z for z in scored["per_new_recording_source_proof"] if z["filename_sha"]==ident)
        need(expected["original_MIDI_SHA256"]==sha_m,
             "S543 original independently evaluated song MIDI SHA changed")
        original=midi_events(raw)
        offset=float(item["frozen_excerpt_offset_s"])
        tr=onset_groups(original,offset,float(item["frozen_excerpt_length_s"]))
        truth=[dict(time=float(x["onset"]),notes=[int(p) for p in
                     new_notes(original,offset,float(x["onset"])) if 40<=int(p)<=88])
                     for x in tr]
        truth=[x for x in truth if x["notes"]]
        ids=np.flatnonzero(member==ident)
        native=starts[ids]
        pairs=match_times(native,np.asarray([z["time"] for z in truth]),.08)
        matched_gt={j for i,j,d in pairs}
        k_matched=collections.Counter(int(kval[ids[i]]) for i,j,d in pairs)
        time_cov=np.abs(np.asarray([t["time"] for t in truth])[:,None]-native[None,:])<=.08
        all_near_events=np.any(time_cov,axis=1) if len(native) else np.zeros(len(truth),bool)
        missed_groups=sum(j not in matched_gt for j in range(len(truth)))
        missed_pitches=sum(len(g["notes"]) for j,g in enumerate(truth) if j not in matched_gt)
        no_pitch_native=sum(len(g["notes"]) for j,g in enumerate(truth) if not all_near_events[j])
        full_truth=sum(len(x["notes"]) for x in truth)
        oldpred=perpred[(ident,"raw_mean_0p7")]
        dupe=perpred[(ident,"dedup80")]
        strict=perpred[(ident,"dedup100_harm080")]
        matchedraw=matched_truth_pitch_keys(oldpred,truth)
        matchedd=matched_truth_pitch_keys(dupe,truth)
        matchedstrong=matched_truth_pitch_keys(strict,truth)
        lost_raw_to_d=matchedraw-matchedd
        rescued_d=matchedd-matchedraw
        to_coverage=[]
        truthpitch=truth_pitch_keys(truth)
        native_near_pitch=collections.Counter()
        for pitch,t in truthpitch.items():
            for j,tt in enumerate(t):
                if np.any(np.abs(native-tt)<=.08):
                    native_near_pitch["covered"]+=1
                else:
                    native_near_pitch["not_covered"]+=1
        sample=dict(composer=item["composer"],
            true_piano_onsets=len(truth),true_named_pitch_notes=full_truth,
            native_V88_attack_candidates=len(ids),
            native_candidate_times_one_to_one_matched=len(pairs),
            true_MIDI_onset_groups_without_unique_candidate=missed_groups,
            true_MIDI_note_pitches_in_without_unique_candidate_groups=missed_pitches,
            true_pitch_names_NOT_even_near_any_V88_time=NativeNone if False else int(native_near_pitch["not_covered"]),
            true_pitch_names_WITH_V88_time_nearby=int(native_near_pitch["covered"]),
            observed_native_V88_original_K_on_matched_truth=dict(k_matched),
            noteon_raw_true_detected=len(matchedraw),
            noteon_dedup80_true_detected=len(matchedd),
            noteon_dedup100_harm080_true_detected=len(matchedstrong),
            true_pitch_note_events_lost_BY_dedup80_for_audit=[
                dict(pitch=int(p),true_occurrence_for_pitch=int(j),
                     true_attack_s=truthpitch[p][j]) for p,j in sorted(lost_raw_to_d)],
            true_pitch_note_events_gained_BY_dedup80_for_audit=[
                dict(pitch=int(p),true_occurrence_for_pitch=int(j),
                     true_attack_s=truthpitch[p][j]) for p,j in sorted(rescued_d)],
            original_real_same_pitch_attacks_under_80ms=
                len(repeat_truth_intervals(truth,.08)),
            original_real_same_pitch_attacks_under_100ms=
                len(repeat_truth_intervals(truth,.100)))
        need(sample["noteon_raw_true_detected"]==
                 scored["scores"]["raw_mean_0p7"]["per_source"][ident]
                   ["onset_pitch_at_30_50_80ms"]["80"]["TP"]
             and sample["noteon_dedup80_true_detected"]==
                 scored["scores"]["dedup80"]["per_source"][ident]
                   ["onset_pitch_at_30_50_80ms"]["80"]["TP"],
            "S543 independent exact one-to-one per pitch scores failed replay")
        out[ident]=sample
        allcounts.update(true_onsets=len(truth),true_note_pitches=full_truth,
             original_v88_candidates=len(ids),one_to_one_native_matched=len(pairs),
             unmatched_original_attack_groups=missed_groups,
             uncovered_note_pitches=int(native_near_pitch["not_covered"]),
             true_pitch_lost_on_dedup=len(lost_raw_to_d),
             true_pitch_gained_on_dedup=len(rescued_d))
        print("S543 FORENSIC VERIFIED NEW COMPOSER NOTE RECALL BOTTLENECK",
              json.dumps(dict(rec=ident,summary=sample)),flush=True)
    total=dict(allcounts)
    need(len(out)==5 and total["true_note_pitches"]==754
         and total["true_onsets"]==518 and
         total["true_pitch_lost_on_dedup"]==1
         and total["true_pitch_gained_on_dedup"]==0,
         "S543 original S542 full truth and one true-pitch regression must remain reproducible")
    a.out.mkdir(parents=True)
    report=dict(status="S543_NEW_FIVE_COMPOSER_ORIGINAL_V88_MISSING_NOTE_AND_DEDUP_FAILURE_FORENSICS",
         underlying_S542_all_5_new_sources_independently_evaluated=True,
         native_V88_onset_group_candidate_ceiling_for_NOTE_ON_measured=True,
         exact_midi_pitch_name_per_true_note_matched_separately=True,
         original_S536_deletion_true_one_note_regression_independently_explained=True,
         original_true_754_MIDI_pitches_all_replayed=True,
         note_model_suppressed_repeated_strikes_diagnosed_READ_ONLY=True,
         analysis_AFTER_held_test_MIDI_is_POSTHOC_and_CAN_NOT_TUNE_TEST=True,
         original_source_S80_K_Live_and_weights_untouched=True,no_promotion=True,
         totals=total,per_recording=out,
         SHA_sources=dict(S537=sha(a.manifest),S538=sha(a.v88),
                          S541=sha(a.predictions),S542=sha(a.truth_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    t=[dict(time=.1,notes=[52,55]),dict(time=.2,notes=[52])]
    x=[dict(time=.1,notes=[52,55]),dict(time=.2,notes=[52])]
    assert len(matched_truth_pitch_keys(x,t))==3
    y=[dict(time=.1,notes=[52,55])]
    assert len(matched_truth_pitch_keys(y,t))==2
    print("S543 SELFTEST PASS exact MIDI note identity and repeat must be distinct")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","v88","v88-report","freeze-report","predictions","truth-report","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","v88","v88_report","freeze_report","predictions","truth_report","out"):
            need(getattr(a,z) is not None,"S543 missing "+z)
        run(a)
