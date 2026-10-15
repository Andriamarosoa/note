"""S541 freeze new original 5-composer NoteOn predictions before opening MIDI.

Load the three original S528 source-held trained raw_only pitch experts,
ensemble their probabilities with a preregistered MEAN. Reuse the EXACT
S533 and S536 same-pitch duplicate/harmonic algorithms without fitting.
All 5 S537 preselected new composer WAVs were already processed by
S538 V88 and S539 Basic Pitch, and S540 acoustic 49x11, no MIDI.

PREDECLARE 4 policies, no truth-directed parameter fitting:
- BP_birth
- raw_mean_0p7: BP-protected pitch >=0.7
- dedup80: raw_mean_0p7 plus 80ms same-pitch dedup
- dedup100_harm080: raw_mean_0p7 100ms dedup and weak higher partial veto
File output is immutable: no MIDI labels read or downloaded by this code.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    proposed_notes,need,sha)
from scripts.s533_harmonic_competition_temporal_duplicate_loops import emit as duplicate_emit
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as restrike_emit

POLICIES=("BP_birth","raw_mean_0p7","dedup80","dedup100_harm080")
def readrows(f):
    return [json.loads(s) for s in f.read_text().splitlines() if s.strip()]

def read(a):
    m=json.loads(a.manifest.read_text())
    src=json.loads(a.source_report.read_text())
    bp=json.loads(a.bp_report.read_text())
    feat=json.loads(a.features_report.read_text())
    trained=json.loads(a.old_training_report.read_text())
    need(m["status"]=="S537_PREREGISTERED_DISTINCT_MAESTRO_V3_TEST_PIANO_COHORT_NO_WAV_NO_MIDI"
         and m["selected_recordings"]==5
         and not m["MIDI_contents_downloaded_or_inspected"]
         and src["status"]=="S538_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and src["prelabel_predictions_frozen"]
         and src["preregistered_S469_SHA"]==sha(a.manifest)
         and not src["paired_MIDI_annotation_files_OPENED"]
         and bp["status"]=="S539_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI"
         and bp["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False
         and feat["status"]=="S540_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
         and feat["frozen_feature_ledger_SHA256"]==sha(a.features)
         and feat["no_original_true_MIDI_annotation_read"]
         and trained["status"]=="S528_TRAINED_THREE_RECORDING_HELD_PIANO_NOTE_HEADS_ALL_V88_K0_SITES"
         and trained["truly_trained_6_models"]==6,
         "S541 original source/audio/features or model training provenance invalid")
    rows=readrows(a.features)
    counts=collections.Counter(r["source_recording_sha"] for r in rows)
    original={r["WAV_filename_sha256"] for r in m["selected_sources"]}
    need(len(rows)==feat["original_V88_onset_candidates"] and
         set(counts)==original and len(counts)==5
         and len({r["V88_original_event_id"] for r in rows})==len(rows),
         "S541 prelabel every original held composer V88 K0/1/2 event")
    return m,src,bp,feat,trained,rows

def probabilities(rows,a,training_report):
    import joblib
    x=np.asarray([r["per_midi_features_49x11"] for r in rows],np.float32)
    need(x.shape==(len(rows),49,11) and np.isfinite(x).all(),
         "S541 strict original S528 physical note feature ABI must remain unchanged")
    allprob=[]
    actual=[]
    source_models=[v for v in training_report["models"] if v["feature_policy"]=="raw_only"]
    need(len(source_models)==3,"S541 needs exactly original S528 three real raw-only piano experts")
    for k,rec in enumerate(sorted(source_models,key=lambda r:r["held_original_recording_SHA"])):
        # filenames are guaranteed by the S528 immutable training artifact.
        name=f"S528_raw_only_held{k}_ACTUAL_PIANO_AUDIO_NOTE_HEAD.joblib"
        checkpoint=a.weights/name
        need(checkpoint.is_file() and
             sha(checkpoint)==rec["trained_model_SHA256"],
             "S541 S528 actual frozen trained checkpoint or original SHA altered")
        pack=joblib.load(checkpoint)
        cols=pack["feature_columns"]
        need(cols==[0,1,2,3,4,5,10] and
             pack["held_sha"]==rec["held_original_recording_SHA"],
             "S541 original 7 raw acoustic pitch features and model-held SHA changed")
        model=pack["model"]
        need(1 in model.classes_,"S541 frozen raw-only classifier lacks true-note class")
        p=model.predict_proba(x[:,:,cols].reshape(-1,len(cols)))[:,list(model.classes_).index(True)]
        prob=p.reshape(len(rows),49).astype(np.float32)
        allprob.append(prob)
        actual.append(dict(filename=name,sha256=sha(checkpoint),
                           old_held_composer_SHA=rec["held_original_recording_SHA"],
                           new_five_composers_ALL_EXCLUDED_from_weights=True))
    result=np.mean(np.stack(allprob,axis=0),axis=0)
    need(result.shape==(len(rows),49) and
         np.isfinite(result).all() and np.all((0<=result)&(result<=1)),
         "S541 ensemble 3 source-trained MIDI note-pitch models invalid")
    return result,actual

def run(a):
    need(not a.out.exists(),"S541 immutable five new composer note freeze must never overwrite")
    m,src,bp,feat,trained,rows=read(a)
    probability,provenance=probabilities(rows,a,trained)
    allheads=[dict(V88_original_event_id=r["V88_original_event_id"],
                  raw_only_49_pitch_prob=probability[i].round(6).tolist())
              for i,r in enumerate(rows)]
    recordings=collections.defaultdict(list)
    for i,r in enumerate(rows):
        recordings[r["source_recording_sha"]].append((r,allheads[i]))
    ledgers=[]
    reportpol={}
    for policy in POLICIES:
        byrec=[]
        for rec in sorted(recordings):
            items=recordings[rec]
            if policy=="BP_birth" or policy=="raw_mean_0p7":
                groups=[]
                for r,head in items:
                    notes=proposed_notes(r,head,"BP_birth" if policy=="BP_birth" else "raw_only:0.7")
                    if notes:
                        groups.append(dict(time=float(r["original_V88_candidate_start_s"]),
                                           notes=sorted(notes)))
            elif policy=="dedup80":
                groups,stat=duplicate_emit(items,dict(dedup_ms=80,harmonic_ratio=0))
            elif policy=="dedup100_harm080":
                groups,stat=restrike_emit(items,dict(dedup_ms=100,
                           harmonic_ratio=.8,rearticulation_ratio=0))
            else:raise RuntimeError("S541 unknown policy")
            groups.sort(key=lambda x:(x["time"],tuple(x["notes"])))
            for e in groups:
                need(0<=e["time"]<12 and len(e["notes"])>0
                     and all(40<=int(q)<=88 for q in e["notes"]),
                     "S541 invalid frozen NoteOn MIDI pitch/time prelabel")
                ledgers.append(dict(policy=policy,original_recording_filename_sha=rec,
                    onset_s=round(float(e["time"]),6),
                    pitch_MIDI=sorted(int(q) for q in e["notes"]),
                    note_NAMES=[str(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][int(q)%12])
                                +str(int(q)//12-1) for q in e["notes"]],
                    MIDI_ANNOTATIONS_NEVER_OPENED=True))
            byrec.append(dict(recording_sha=rec,emitted_note_onsets=len(groups),
                         emitted_named_pitch_notes=sum(len(x["notes"]) for x in groups)))
        reportpol[policy]=dict(per_recording=byrec,
                  event_count=sum(z["emitted_note_onsets"] for z in byrec),
                  named_note_pitch_count=sum(z["emitted_named_pitch_notes"] for z in byrec))
        print("S541 FROZEN FIVE NEW SONGS POLICY NO MIDI",json.dumps(
            dict(policy=policy,events=reportpol[policy]["event_count"],
                 notes=reportpol[policy]["named_note_pitch_count"])),flush=True)
    a.out.mkdir(parents=True)
    ledger=a.out/"S541_FIVE_NEW_MAESTRO_ORIGINAL_REAL_WAV_NOTE_NAMES_ALL_POLICIES_PREMIDI.jsonl"
    ledger.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in ledgers))
    weights=a.out/"S541_FIVE_NEW_SONGS_3X_S528_ORIGINAL_HELD_PITCH_PROBABILITIES_PREMIDI.jsonl"
    weights.write_text("".join(json.dumps(dict(V88_original_event_id=r["V88_original_event_id"],
           source_original_recording_sha=r["source_recording_sha"],
           mean_of_3_real_S528_original_trained_acoustic_note_head_probs=head["raw_only_49_pitch_prob"]))
           +"\n" for r,head in zip(rows,allheads)))
    report=dict(
       status="S541_FIVE_NEW_COMPOSER_PREMIDI_FROZEN_S528_BP_S533_S536_NAMED_NOTES",
       original_S537_metadata_before_ANY_WAV_MIDI=True,
       original_source_GUITARSET_S80_complete_not_replayed=True,
       all_FIVE_new_S537_composers_scored_without_labels=True,
       original_five_source_V88_events=len(rows),
       original_s528_three_SOURCE_HELD_real_audio_models_ENSEMBLE_MEAN=True,
       raw7_acoustic_feature_ABI_unchanged=True,
       actual_original_S528_weight_checkpoints_SHA_verified=provenance,
       four_policies_predeclared=list(POLICIES),
       fixed_pitch_prob_threshold=0.7,
       fixed_same_note_dedup_ms=[80,100],
       last_100ms_harmonic_veto_threshold=.8,
       original_BP_birth_chords_never_suppressed=True,
       frozen_prediction_ledger_SHA256=sha(ledger),
       frozen_model_3_ensemble_probabilities_SHA256=sha(weights),
       original_true_MIDI_note_contents_OPENED=False,
       no_test_MIDI_used_to_tune_or_choose_policy=True,
       original_S80_ExactK_freeze_local_combo_Live_unchanged=True,
       no_promotion=True,
       policy_counts=reportpol,
       inputs_SHA256=dict(S537=sha(a.manifest),S538=sha(a.source_report),
                          S539=sha(a.bp_report),S540=sha(a.features),
                          S528=sha(a.old_training_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S541 FIVE NEW COMPOSER MIDI-BLIND FULL NAMED NOTE FORECAST SEALED",
          json.dumps(dict(models=3,source=5,events=len(rows),
                          ledger_SHA=sha(ledger),NO_MIDI_READ=True)),flush=True)

def selftest():
    import numpy as np
    r=dict(old_basicpitch_birth_pitches=[52,55],
           original_V88_candidate_start_s=.1,
           per_midi_features_49x11=np.zeros((49,11)).tolist())
    p=dict(raw_only_49_pitch_prob=[0]*49)
    assert proposed_notes(r,p,"raw_only:0.7")=={52,55}
    need(POLICIES==("BP_birth","raw_mean_0p7","dedup80","dedup100_harm080"),
         "S541 prospective policy list not pinned before true MIDI evaluation")
    print("S541 SELFTEST PASS policies and protected BP chords exact before any MIDI")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for s in ("manifest","source-report","bp-report","features",
              "features-report","old-training-report","weights","out"):
        p.add_argument("--"+s,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for s in ("manifest","source_report","bp_report","features",
                   "features_report","old_training_report","weights","out"):
            need(getattr(a,s) is not None,"S541 missing "+s)
        run(a)
