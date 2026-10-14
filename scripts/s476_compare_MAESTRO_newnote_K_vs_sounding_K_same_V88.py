"""S476 audit comparability: new notes struck versus all sounding notes in piano.

The GuitarSet original K training target counts newly assigned note-ons per
candidate cluster. S471 MIDI currently uses sounding notes at +30ms, including
earlier held notes. A piano often has K(sound)>K(new), so interpreting V88
accuracy on K(sound) as a source generalization score is risky.
This audit uses exactly the same S470 frozen V88 event rows and S471 matched
MIDI onsets, but reports BOTH frozen label definitions side-by-side.
No K model training, retuning, or source reference changes.
"""
from __future__ import annotations
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,need,sha)
GROUP=.020
def new_notes(events,offset,anchor):
    selected=sorted({int(p) for st,en,p in events if
        (offset+anchor-1e-6)<=st <=(offset+anchor+GROUP+1e-6)})
    return selected
def run(frozen,frozen_report,manifest,s471,s471_report,out):
    need(not out.exists(),"S476 label-definition audit report immutable")
    spec=json.loads(manifest.read_text())
    old=json.loads(frozen_report.read_text())
    scored=json.loads(s471_report.read_text())
    need(spec["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
         and old["status"]=="S470_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and old["output_sha256"]==sha(frozen)
         and scored["status"]=="S471_EXTERNAL_THIRD_REAL_MAESTRO_PIANO_V88_MIDI_POSTFREEZE"
         and scored["SHA256"]["S471_MIDI_ledger"]==sha(s471)
         and scored["SHA256"]["S470_frozen_native_inference"]==sha(frozen)
         and scored["original_S80_complete_chain_NOT_RUN"],
         "S476 source original V88 and already scored true piano active notes changed")
    with np.load(frozen,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        K=z["original_V88_K"].astype(np.int8)
        members=z["member"].astype(str)
    mapping={id:i for i,id in enumerate(ids)}
    originals=[json.loads(s) for s in s471.read_text().splitlines()]
    need(len(originals)==scored["cohort_sums"]["matched"]==322
         and len(ids)==575
         and len({r["row_id"] for r in originals})==322,
         "S476 exact original third source MIDI matched-onset IDs missing")
    bymember=collections.defaultdict(list)
    for r in originals:bymember[r["row_id"].split(":")[0]].append(r)
    itembyhash={x["filename_SHA256"]:x for x in spec["selected_sources"]}
    need(len(itembyhash)==3 and set(bymember)==set(itembyhash),
         "S476 source piano composer/recording shift")
    stats=collections.Counter()
    bypiece=[]
    details=[]
    for ident,item in sorted(itembyhash.items()):
        raw=fetch_midi(spec["audio_mirror_directory"],item["MIDI_filename_path_ONLY_not_opened"])
        need(hashlib.sha256(raw).hexdigest()==next(
            x["MIDI_bytes_SHA256"] for x in scored["per_recording"]
            if x["piano_filename_SHA"]==ident),
             "S476 same previously audited exact piano MIDI bytes required")
        events=midi_events(raw)
        offset=float(item["audio_excerpt_start_seconds"])
        anchors=onset_groups(events,offset,float(item["audio_excerpt_duration_seconds"]))
        changes=[];news=[];sound=[];correct_new=correct_sound=0
        eligible_new=eligible_sound=0
        poly_new=poly_sound=0
        poly_hit_new=poly_hit_sound=0
        for entry in bymember[ident]:
            i=mapping[entry["row_id"]]
            anchor=float(entry["annotated_attack_at_excerpt_seconds"])
            possible=[g for g in anchors if abs(g["onset"]-anchor)<1e-6]
            need(len(possible)==1 and
                 entry["true_K"]==possible[0]["trueK"],
                 "S476 true score row anchor or previously opened MIDI active count differs")
            onset_pitches=new_notes(events,offset,anchor)
            sounding=possible[0]["pitches"]
            nk=len(onset_pitches);ak=len(sounding)
            needs_new=(1<=nk<=6 and all(40<=p<=88 for p in onset_pitches))
            needs_sound=(1<=ak<=6 and all(40<=p<=88 for p in sounding))
            if needs_new:
                eligible_new+=1;correct_new+=int(int(K[i])==nk)
                if nk>=2:
                    poly_new+=1;poly_hit_new+=int(int(K[i])==nk)
            if needs_sound:
                eligible_sound+=1;correct_sound+=int(int(K[i])==ak)
                if ak>=2:
                    poly_sound+=1;poly_hit_sound+=int(int(K[i])==ak)
            if needs_new and needs_sound:
                changes.append(int(nk!=ak))
            stats["scored_matched"]+=1
            stats["same_definitions"]+=int(nk==ak)
            stats["piano_notes_new"]+=nk;stats["piano_notes_active"]+=ak
            details.append(dict(row_id=entry["row_id"],original_V88_K=int(K[i]),
                onset_at_excerpt_s=anchor,
                true_new_onset_MIDI_pitches=onset_pitches,
                true_new_onset_K=nk,
                true_sounding_at_plus30ms_K=ak,
                both_MIDI40_88=bool(needs_new and needs_sound)))
        for key,v in (("new_eligible",eligible_new),("new_correct",correct_new),
                      ("sound_eligible",eligible_sound),("sound_correct",correct_sound),
                      ("new_poly",poly_new),("new_poly_correct",poly_hit_new),
                      ("sound_poly",poly_sound),("sound_poly_correct",poly_hit_sound)):
            stats[key]+=v
        piece=dict(source_MAESTRO_hash=ident,composer=item["composer"],
          original_matched_annotated_attack_events=len(bymember[ident]),
          new_notes_MIDI40_88_exactK_n=eligible_new,
          new_notes_original_V88_K_exact=correct_new,
          new_notes_true_poly_n=poly_new,
          new_notes_true_poly_exact=poly_hit_new,
          active_sounding_MIDI40_88_exactK_n=eligible_sound,
          active_sounding_original_V88_K_exact=correct_sound,
          active_sounding_true_poly_n=poly_sound,
          active_sounding_true_poly_exact=poly_hit_sound,
          distinct_K_definitions_among_both_scoreable=int(sum(changes)))
        bypiece.append(piece)
        print("S476 NEWvsACTIVE per piano source",json.dumps(piece),flush=True)
    need(stats["scored_matched"]==322 and
         stats["sound_eligible"]==322 and
         stats["sound_correct"]==24 and
         stats["sound_poly"]==232 and stats["sound_poly_correct"]==6,
         "S476 fails exact independent S471 old active-pitch baseline; comparisons invalid")
    out.mkdir(parents=True)
    ledger=out/"S476_ORIGINAL_S470_PIANO_NEW_NOTE_VS_SOUNDING_322.jsonl"
    ledger.write_text("".join(json.dumps(r)+"\n" for r in details))
    report=dict(status="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY",
         source="MAESTRO_v3_test_3_acoustic_pianos",
         matched_onsets=322,
         original_V88_predicted_K_unchanged=True,
         zero_S80_replay_asserted=True,
         threshold_or_model_tuning_on_piano=False,
         original_MAESTRO_MIDI_previously_opened_at_S471=True,
         new_note_truth_counts_pitch_set_onset_GROUP20ms=True,
         sounding_truth_counts_all_active_at_plus30ms=True,
         original_GuitarSet_target_K_is_new_onset_assignment=True,
         simultaneous_new_and_active_note_label_comparison=True,
         summary=dict(stats),per_piano= bypiece,
         original_soundK_exact=24,
         original_soundK_poly_exact=6,
         new_noteK_accuracy=(
             stats["new_correct"]/stats["new_eligible"] if stats["new_eligible"] else None),
         new_noteK_poly_accuracy=(
             stats["new_poly_correct"]/stats["new_poly"] if stats["new_poly"] else None),
         this_is_V88_NOT_complete_S80=True,
         no_promotion=True,
         SHA256=dict(S470_frozen=sha(frozen),
                     S471_matched_active_labels=sha(s471),
                     paired_label_comparison=sha(ledger)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S476 BOTH LABELS",json.dumps(dict(
        total=322,sums=dict(stats),
        new_K_accuracy=report["new_noteK_accuracy"],
        new_poly_accuracy=report["new_noteK_poly_accuracy"],
        active_K_accuracy=24/322,
        active_poly_accuracy=6/232)),flush=True)
def selftest():
    events=[(.1,.7,52),(.11,.3,55),(.4,.9,59)]
    n=new_notes(events,0,.4)
    a=onset_groups(events,0,1)
    need(n==[59] and len(a)==2 and a[1]["trueK"]==2,
         "S476 distinguish one new piano strike from two sustained MIDI voices")
    print("S476 SELFTEST PASS onset count is NOT sustained cardinality at poly piano")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for arg in ("frozen","frozen-report","manifest","s471","s471-report","out"):
        p.add_argument("--"+arg,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("frozen","frozen_report","manifest","s471","s471_report","out"):
            need(getattr(a,key)is not None,"S476 missing "+key)
        run(a.frozen,a.frozen_report,a.manifest,a.s471,a.s471_report,a.out)
