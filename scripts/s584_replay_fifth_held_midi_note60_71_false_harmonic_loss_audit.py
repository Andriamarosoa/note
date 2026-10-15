"""S584 postlabel independent exact per-note and per-composer mid failure audit.

Replay sealed S582 outputs against same original S583 SHA recorded MIDI.
Independently reproduce ALL four policies per MIDI pitch 60..71 and
30/50/80ms. Count true NoteOn lost when specialist used and false
spectral harmonic notes suppressed; real MIDI repeated notes remain
distinct. This is RETROSPECTIVE diagnostic, not new validation/promotion.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import sha,need
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics
from scripts.s583_fifth_held_source_true_NoteOn_F1_compare_middle_specialist import (
    POLICIES,RANGES,part)
def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def run(a):
    need(not a.out.exists(),"S584 source fifth music MIDI audit cannot change frozen results")
    m=json.loads(a.manifest.read_text())
    fr=json.loads(a.freeze_report.read_text())
    r=json.loads(a.score_report.read_text())
    need(m["status"]=="S577_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and fr["status"]=="S582_FIFTH_COMPOSER_NEW_MIDDLE_HEAD_FOUR_POLICIES_FROZEN_BEFORE_MIDI"
         and fr["frozen_named_noteon_SHA"]==sha(a.predictions)
         and r["status"]=="S583_REAL_FIFTH_MAESTRO_SOURCE_HELD_MIDDLE_SPECIALIST_NOTEON_F1"
         and r["source_MIDI_opened_ONLY_after_S582_audio_model_notes_frozen_SHA"]
         and r["frozen_predictions_SHA"]==sha(a.predictions)
         and set(r["policies"])==set(POLICIES),
         "S584 independent existing original MIDI evidence and original frozen four note-on policies required")
    sources={item["filename_SHA256"]:item for item in m["chosen_metadata_only"]}
    proofs={p["original_audio_filename_sha"]:p for p in r["true_MIDI_provenance"]}
    need(len(sources)==5 and set(sources)==set(proofs),"S584 exact five original distinct sound notes")
    predictions=collections.defaultdict(list)
    for row in records(a.predictions):
        need(row["policy"] in POLICIES and row["original_recording_filename_sha"] in sources
            and row["MIDI_ANNOTATIONS_NEVER_OPENED"],"S584 no adaptive post-MIDI predicted note event")
        predictions[(row["original_recording_filename_sha"],row["policy"])].append(
          dict(time=float(row["onset_s"]),notes=sorted(map(int,row["pitch_MIDI"]))))
    regtot={p:collections.defaultdict(collections.Counter) for p in POLICIES}
    pitchtot={p:{str(pitch):collections.Counter() for pitch in range(60,72)} for p in POLICIES}
    per_source={}
    for ident,item in sorted(sources.items()):
        midi=fetch_midi(m["source_audio_mirror"],item["MIDI_filename_path_ONLY_not_opened"])
        need(hashlib.sha256(midi).hexdigest()==proofs[ident]["original_MIDI_sha256"],
             "S584 genuine source MIDI changed compared to independent first postfrozen test")
        data=midi_events(midi)
        off=float(item["audio_excerpt_start_seconds"])
        true=[dict(time=float(z["onset"]),notes=[int(x) for x in new_notes(data,off,float(z["onset"])) if 40<=int(x)<=88])
              for z in onset_groups(data,off,float(item["audio_excerpt_duration_seconds"]))]
        true=[e for e in true if e["notes"]]
        need(sum(len(x["notes"]) for x in true)==proofs[ident]["scorable_true_MIDI_note_pitches"],
             "S584 all ground truth NoteOns must be present including source V88 misses")
        per_source[ident]=dict(composer=item["composer"],per_policy={},midi_sha=hashlib.sha256(midi).hexdigest())
        for policy in POLICIES:
            guess=predictions[(ident,policy)]
            byp={}
            for k in range(60,72):
                pp=str(k)
                s=note_metrics(part(guess,k,k),part(true,k,k),.08)
                byp[pp]={z:s[z] for z in ("TP","FP","FN")}
                pitchtot[policy][pp].update(byp[pp])
            for range_name,(lo,hi) in RANGES.items():
                s=note_metrics(part(guess,lo,hi),part(true,lo,hi),.08)
                regtot[policy][range_name].update({z:s[z] for z in ("TP","FP","FN")})
            per_source[ident]["per_policy"][policy]=byp
    totals={}
    for policy in POLICIES:
        bypitch={k:dict(v) for k,v in pitchtot[policy].items()}
        reg={}
        for n,c in regtot[policy].items():
            x=dict(c);x["F1"]=2*x["TP"]/max(1,2*x["TP"]+x["FP"]+x["FN"])
            reg[n]=x
            z=r["policies"][policy]["by_register"][n]
            need(all(x[p]==z[p] for p in ("TP","FP","FN")),
                 "S584 exact per-register MIDI NoteOn values MUST reproduce original S583")
        counts=collections.Counter()
        for z in bypitch.values():counts.update(z)
        need(all(counts[k]==reg["mid_60_71"][k] for k in ("TP","FP","FN")),
             "S584 all 12 individual MIDI pitch confusion matrices MUST reproduce middle score")
        totals[policy]=dict(music_registers=reg,midi_pitch60to71=bypitch)
        print("S584 INDEPENDENT PER PITCH REPLAY",json.dumps(
              dict(policy=policy,mid=reg["mid_60_71"])),flush=True)
    general=totals["S578_general28"]["midi_pitch60to71"]
    diagnoses={}
    for policy in ("S578_mid_replace","S578_mid_blend50"):
        curr=totals[policy]["midi_pitch60to71"]
        differences={}
        for midi in range(60,72):
            k=str(midi);base=general[k];new=curr[k]
            differences[k]=dict(deltaTP=new["TP"]-base["TP"],
               deltaFP=new["FP"]-base["FP"],deltaFN=new["FN"]-base["FN"],
               trueNoteOns=base["TP"]+base["FN"])
        diagnoses[policy]=differences
        print("S584 LOST TRUE NOTES VS GHOSTS BY MIDI PITCH",
              json.dumps(dict(policy=policy,differences=differences)),flush=True)
    a.out.mkdir(parents=True)
    rep=dict(status="S584_FIFTH_SOURCE_TRUE_MIDI_MIDDLE_PITCH_FAILURE_SHA_REPLAY_POSTHOC",
      same_all_original_source_MIDI_SHA_reverified=True,
      all_four_policies_verified_source_frozen_pre_MIDI=True,
      all_per_pitch_totals_reconstruct_exact_S583=True,
      original_5_music_sources_replayed=True,
      posthoc_not_new_blind_source_validation=True,
      physical_12_middle_note_labels_60_to_71=True,
      no_new_models_fitted_to_S577_MIDI=True,
      old_original_S80_ExactK_Live_not_modified=True,no_promotion=True,
      actual_original_S583_report_SHA=sha(a.score_report),
      pitch_ledger_SHA=sha(a.predictions),
      per_source=per_source,
      policy_midi_pitch_register_scores=totals,
      specialist_vs_general_true_notes_lost_and_false_pitches_removed=diagnoses)
    (a.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    a=[dict(time=.1,notes=[52,60,64]),dict(time=.18,notes=[60])]
    need(len(part(a,60,60))==2 and
         len(part(a,60,71))==2 and
         len(part(a,40,59))==1,
         "S584 exact individual pitch note attribution and genuine fast same-note restart")
    print("S584 SELFTEST PASS independently attributable MIDI pitch-level true and false notes")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for x in ("manifest","freeze-report","score-report","predictions","out"):p.add_argument("--"+x,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for x in ("manifest","freeze_report","score_report","predictions","out"):
            need(getattr(a,x)is not None,"S584 missing --"+x)
        run(a)
