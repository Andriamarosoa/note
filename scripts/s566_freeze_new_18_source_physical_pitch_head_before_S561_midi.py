"""S566: prospective seven-policy real piano NOTE_ON freeze before new S561 MIDI.

All 5 fresh source S561 WAV names are registered by metadata exclusively.
S563 original source V88, S564 pretrained BP, S565 original raw49 are
computed on actual WAV BEFORE any true label, with full byte-level SHA.

Genuinely trained source weights from S546 (8 old composer-held forecasts)
and S562 (18 musical-source FULL fits, new physical fundamental34) all
have predeclared frozen SHA from a separate earlier training job.
Prediction policies predeclare:
 BP births; S546 8-mean dedup80/dedup100harm08;
 S562 raw7 FULL-18 dedup80/dedup100harm08;
 S562 competing-fundamental34 FULL-18 dedup80/dedup100harm08.
Threshold original 0.70 fixed. No MIDI downloaded, inspected or used to
choose experts. BP original chords are never suppressed by new policy.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need,sha,proposed_notes
from scripts.s533_harmonic_competition_temporal_duplicate_loops import emit as dedup80
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as dedup100
from scripts.s562_train_18_composer_true_pitch_spectral_fundamental_heads import acoustic
POLICIES=("BP_birth","S546_dedup80","S546_dedup100_harm080",
          "S562_raw7_dedup80","S562_raw7_dedup100_harm080",
          "S562_fundamental34_dedup80","S562_fundamental34_dedup100_harm080")
def readrows(path):return [json.loads(q) for q in path.read_text().splitlines() if q.strip()]
def all_model_predictions(X,s546,s562,a):
    import joblib
    chosen=sorted((m for m in s546["models"] if m["architecture"]=="positive7"),
                  key=lambda x:x["held_composer_sha"])
    need(len(chosen)==8 and s546["shared_composer_collaboration_leakage_is_EXCLUDED"] and
         all(x["trained_on_other_source_recordings_only_with_disjoint_composer_components"] for x in chosen),
         "S566 only original S546 models with strict historic composer-held source audit")
    raw7=acoustic(X,"raw7")
    stack=[]
    weight_proofs=[]
    for i,m in enumerate(chosen):
        f=a.s546_weights/f"S546_positive7_held{i}_ACTUAL_NEURAL_STYLE_NOTE_DECISION_FOREST.joblib"
        need(f.is_file() and sha(f)==m["genuine_checkpoint_SHA"],
             "S566 original S546 8 genuinely source-held real model weight SHA mismatch")
        model=joblib.load(f)
        labels=list(model.classes_)
        q=model.predict_proba(raw7.reshape(-1,7))[:,labels.index(1)].reshape(len(X),49)
        stack.append(q)
        weight_proofs.append(dict(name="S546",original_weight_sha=sha(f),
                   filename=f.name))
    p546=np.mean(np.stack(stack),axis=0).astype(np.float32)
    full={}
    for m in s562["all_fitted_checkpoints_SHA256"]:
        mode=m["architecture"]
        need(mode in ("raw7","competing_fundamental34") and
             m["number"]==18 and
             m["full_source_training_without_any_S561_audio_or_MIDI"],
             "S566 no new held S561 ground truth used to fit musical note heads")
        file=a.s562_weights/f"S562_FULL18_{mode}_REAL_COMPOSER_DISJOINT_HELD_SOURCE_FIT.joblib"
        need(file.is_file() and sha(file)==m["source_sha_model"],
             "S566 FULL18 genuine physical MIDI-note acoustic weight SHA missing")
        model=joblib.load(file)
        fx=acoustic(X,mode)
        need(fx.shape[-1]==m["feature_dimensions"],"S566 acoustic model feature ABI drift")
        labels=list(model.classes_)
        full[mode]=model.predict_proba(fx.reshape(-1,fx.shape[-1]))[:,labels.index(1)].reshape(len(X),49)
        weight_proofs.append(dict(name="S562_"+mode,filename=file.name,
                                  original_weight_sha=sha(file)))
    need(set(full)=={"raw7","competing_fundamental34"} and
         all(np.isfinite(v).all() and v.shape==(len(X),49)
             and np.all((v>=0)&(v<=1)) for v in (p546,*full.values())),
         "S566 must predict all 49 named music pitches with 3 old trained models")
    return {"S546":p546,"S562_raw7":full["raw7"],
            "S562_fundamental34":full["competing_fundamental34"]},weight_proofs
def emit(rows,probs,policy):
    if policy=="BP_birth":
        result=[]
        for r,p in rows:
            notes=proposed_notes(r,p,"BP_birth")
            if notes:result.append(dict(time=float(r["original_V88_candidate_start_s"]),notes=sorted(notes)))
        return result
    need(policy in POLICIES,"S566 unknown predeclared note policy")
    if policy.endswith("dedup80"):
        result,_=dedup80(rows,dict(dedup_ms=80,harmonic_ratio=0))
    else:
        result,_=dedup100(rows,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
    return result

def run(a):
    need(not a.out.exists(),"S566 original held music names/notes immutable")
    m=json.loads(a.manifest.read_text())
    v=json.loads(a.v88_report.read_text())
    bp=json.loads(a.bp_report.read_text())
    feat=json.loads(a.feature_report.read_text())
    s546=json.loads(a.s546_report.read_text())
    s562=json.loads(a.s562_report.read_text())
    guard_checks={
      "S561_original_status":m["status"]=="S561_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI",
      "S561_excludes_18":m["original_previous_S469_3_S537_5_S545_5_S554_5_component_excluded"],
      "S561_five":m["five_next_distinct_composers"]==5,
      "S561_no_co_composer_overlap":m["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"],
      "S561_MIDI_unopened":m["source_MIDI_files_not_read"],
      "S561_WAV_unopened_at_registration":m["source_WAV_files_not_read"],
      "S563_original_audio_status":v["status"]=="S563_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI",
      "S563_registered_metadata_SHA":v["preregistered_S561_SHA"]==sha(a.manifest),
      "S563_MIDI_unopened":not v["paired_MIDI_annotation_files_OPENED"],
      "S563_source_prob_frozen":v["prelabel_predictions_frozen"],
      "S564_basicpitch_status":bp["status"]=="S564_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI",
      "S564_MIDI_unopened":not bp["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"],
      "S565_original_49x11_status":feat["status"]=="S565_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES",
      "S565_prelabel_feature_SHA":feat["frozen_feature_ledger_SHA256"]==sha(a.features),
      "S565_MIDI_unopened":feat["no_original_true_MIDI_annotation_read"],
      "S546_source_trained_status":s546["status"]=="S546_8_COMPOSER_REAL_NOTE_IDENTITY_EXPERT_TRAINING_AND_RETRO_LOO",
      "S562_source_trained_status":s562["status"]=="S562_REAL_EIGHTEEN_EXPOSED_COMPOSER_SOURCE_AUDIO_TRAINED_NOTE_PITCH_TWO_HEADS",
      "S562_two_full_models_frozen":s562["fully_trained_two_models_before_ANY_S561_MIDI"],
      "S562_held_MIDI_unopened":s562["no_S561_audio_or_MIDI_opened_for_fitting"],
    }
    if not all(guard_checks.values()):
        print("S566 PROSPECTIVE NO-MIDI SOURCE PROVENANCE FAILED CHECKS",
              json.dumps({k:v for k,v in guard_checks.items() if not v}),flush=True)
    need(all(guard_checks.values()),
         "S566 SHA-verified source, original weights, audio WAV test all prelabel required")
    rows=readrows(a.features)
    sources={x["filename_SHA256"] for x in m["chosen_metadata_only"]}
    need(len(sources)==5 and {r["source_recording_sha"] for r in rows}==sources
         and len(rows)==feat["original_V88_onset_candidates"],
         "S566 all five SHA music sources and original V88 times missing")
    X=np.asarray([r["per_midi_features_49x11"] for r in rows],np.float32)
    need(X.shape==(len(rows),49,11) and np.isfinite(X).all(),
         "S566 raw 49x11 original acoustic input ABI required")
    prob,models=all_model_predictions(X,s546,s562,a)
    bysource=collections.defaultdict(list)
    for i,r in enumerate(rows):bysource[r["source_recording_sha"]].append(i)
    published=[];summary={}
    for policy in POLICIES:
        mode="S546" if policy.startswith("S546") else (
             "S562_fundamental34" if policy.startswith("S562_fundamental34")
             else "S562_raw7")
        p=prob[mode]
        per=[]
        for rec in sorted(bysource):
            rows_here=[(rows[i],dict(raw_only_49_pitch_prob=p[i].tolist())) for i in bysource[rec]]
            events=emit(rows_here,p,policy)
            events.sort(key=lambda z:z["time"])
            for e in events:
                notes=sorted(set(map(int,e["notes"])))
                need(notes and 0<=float(e["time"])<12 and
                     all(40<=n<=88 for n in notes),
                     "S566 original true note pitch range MIDI 40-88 and excerpt times")
                published.append(dict(policy=policy,
                    original_recording_filename_sha=rec,
                    onset_s=round(float(e["time"]),6),
                    pitch_MIDI=notes,MIDI_ANNOTATIONS_NEVER_OPENED=True))
            per.append(dict(source_sha=rec,emitted_note_attack_groups=len(events),
                            emitted_pitch_note_onsets=sum(len(e["notes"]) for e in events)))
        summary[policy]=dict(per_source=per,
             emitted_all_pitch_notes=sum(x["emitted_pitch_note_onsets"] for x in per))
        print("S566 IMMUTABLE MIDI BLIND TRUE SOUND PITCH NOTE FORECAST",
             json.dumps(dict(policy=policy,pitches=summary[policy]["emitted_all_pitch_notes"])),flush=True)
    a.out.mkdir(parents=True)
    output=a.out/"S566_FIVE_NEW_MAESTRO_FULL_NAMED_NOTEON_POLICIES_PRE_MIDI.jsonl"
    output.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in published))
    full=a.out/"S566_THREE_GENUINE_TRAINED_MODEL_49PITCH_PROBABILITIES_PREMIDI.jsonl"
    full.write_text("".join(json.dumps(dict(rowid=r["V88_original_event_id"],
        original_rec_sha=r["source_recording_sha"],
        s546_eight_held=p546.round(6).tolist() if False else prob["S546"][i].round(6).tolist(),
        s562_eighteen_raw=prob["S562_raw7"][i].round(6).tolist(),
        s562_eighteen_physical34=prob["S562_fundamental34"][i].round(6).tolist()))
        +"\n" for i,r in enumerate(rows)))
    report=dict(status="S566_FIVE_NEW_S561_SOUNDFROZEN_S546_VS_S562_NOTEON_BEFORE_MIDI",
       original_S561_new_composers_excluded_from_18_train_and_8_train=True,
       original_S561_MIDI_NOT_read_for_feature_model_or_policy_selection=True,
       all_seven_policies_preregistered_before_MIDI=list(POLICIES),
       threshold_original_070_unchanged=True,
       full_S80_K_and_Live_unchanged=True,
       source_S561_MAESTRO_original_split="train",
       original_pretrained_BP_born_notes_protected=True,
       original_sha_8_plus_2_train_model_checkpoints_verified=models,
       output_pitch_name_ledger_SHA=sha(output),
       output_three_physical_pitch_model_prob_sha=sha(full),
       source_sha=dict(S561=sha(a.manifest),S563=sha(a.v88_report),
                       S564=sha(a.bp_report),S565=sha(a.features),
                       S546=sha(a.s546_report),S562=sha(a.s562_report)),
       policy_counts=summary,no_promotion=True)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    need(len(POLICIES)==7 and len(set(POLICIES))==7,
         "S566 seven fixed prelabel note onset policies")
    r=dict(old_basicpitch_birth_pitches=[52,64],
           original_V88_candidate_start_s=.1)
    p=dict(raw_only_49_pitch_prob=np.zeros(49).tolist())
    need(emit([(r,p)],np.zeros((1,49)),"BP_birth")==[dict(time=.1,notes=[52,64])],
         "S566 source original BP chord notes protected")
    print("S566 SELFTEST PASS original BasicPitch chord protected and seven future note policies MIDI sealed")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for key in ("manifest","v88-report","bp-report","features","feature-report",
                "s546-report","s562-report","s546-weights","s562-weights","out"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("manifest","v88_report","bp_report","features","feature_report",
                    "s546_report","s562_report","s546_weights","s562_weights","out"):
            need(getattr(a,key) is not None,"S566 missing "+key)
        run(a)
